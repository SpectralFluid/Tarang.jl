# ---------------------------------------------------------------------------
# Per-equation F gather (equation space, matches L_min rows)
#
# The subproblem stepper needs F in equation space — the same row ordering
# as `L_min`. Earlier `gather_outputs!` packed F in *variable* space, which
# (a) misaligned PDE rows and (b) silently dropped BC F values because BC
# equations have no time derivative.
#
# `gather_eqn_F!` walks the equations in the original order and packs:
#   - PDE rows: from `pde_F_fields[tidx]` (one entry per state field target)
#   - BC rows:  from the equation's own F expression (constants projected
#               onto the current Fourier mode)
# then applies `pre_left` to match L_min's filtered row space.
# ---------------------------------------------------------------------------

"""
    _is_bulk_eqn_size(sz, Nz) -> Bool

Does an equation whose block occupies `sz` rows sit in the BULK of a subproblem
with a coupled (Chebyshev) axis of length `Nz`?

This is the SAME rule `subproblem_matrix_build.jl` uses to split `sp.bulk_rows`
from `sp.bc_rows` — it is factored out here so the two cannot drift. A bulk block
spans the whole coupled axis (one row per Chebyshev mode, times any component
count); a BC/constraint block is a handful of rows that do not.
"""
@inline _is_bulk_eqn_size(sz::Int, Nz::Int) = Nz > 1 ? (sz >= Nz && sz % Nz == 0) : sz > 1

_is_zero_F_expr(::Nothing) = true
_is_zero_F_expr(::ZeroOperator) = true
_is_zero_F_expr(x::Number) = x == 0
_is_zero_F_expr(c::ConstantOperator) = c.value == 0
_is_zero_F_expr(::Any) = false

"""
    _evaluate_alg_F(F_expr, sp) -> ComplexF64

Dispatch-based evaluation of an algebraic-equation F expression for a given
subproblem. Returns the complex value to write into the BC row of the raw
equation-space vector.

Currently supports:
- `ZeroOperator` / `nothing` → 0
- `ConstantOperator` / `Number` → `v * Nx` at DC, 0 elsewhere
- `ArrayOperator` → unnormalized RFFT of the grid-space array, picked at the
  subproblem's Fourier mode (for space-dependent BCs)
"""
_evaluate_alg_F(::Nothing, ::Subproblem; warn_unsupported::Bool=true) = ComplexF64(0)
_evaluate_alg_F(::ZeroOperator, ::Subproblem; warn_unsupported::Bool=true) = ComplexF64(0)
_evaluate_alg_F(c::ConstantOperator, sp::Subproblem; warn_unsupported::Bool=true) =
    _bc_constant_projection(Float64(c.value), sp)
_evaluate_alg_F(x::Number, sp::Subproblem; warn_unsupported::Bool=true) =
    _bc_constant_projection(Float64(x), sp)
_evaluate_alg_F(a::ArrayOperator, sp::Subproblem; warn_unsupported::Bool=true) =
    _bc_array_projection(a.value, sp)
function _evaluate_alg_F(expr, sp::Subproblem; warn_unsupported::Bool=true)
    # A COMPOUND CONSTANT — `T(z=0) = 10*25`, `= h*T_amb`, `= 1/Re` — arrives here as a
    # Multiply/Add/Divide operator tree, not as a ConstantOperator. It used to fall through to
    # the silent zero below, so the boundary condition was enforced as 0 with no warning and the
    # solve reported success. `_is_const_or_param` / `_extract_scalar` already fold exactly these
    # node types for the L/M matrices; use them here too.
    if _is_const_or_param(expr)
        return _bc_constant_projection(Float64(_extract_scalar(expr)), sp)
    end
    # Anything else genuinely is not supported as a BC right-hand side. Enforcing it as zero is a
    # silently wrong answer, which is worse than a slow or absent one — say so.
    #
    # `warn_unsupported=false` means the caller established this expression belongs to a BULK
    # equation, whose value `apply_bc_override!` never reads (it writes `sp.bc_rows` only). See
    # `gather_alg_F!`. Warning there is a FALSE ALARM, and a costly one: it tells the user their
    # solve is silently wrong when it is not. A BVP/NonlinearBoundaryValueProblem has no `∂t` in ANY equation, so the
    # `is_alg` test below classifies the main PDE as an algebraic row — that is how
    # `Δ(u) + l1 + l2 = u*u + g` came to be reported as an unsupported boundary condition.
    if warn_unsupported
        @warn "Boundary condition right-hand side of type $(typeof(expr)) is not supported and is " *
              "being enforced as ZERO. Supported: a constant, a compound constant (`10*25`, `h*T_amb`), " *
              "or a grid array (space-dependent BC). Rewrite the BC, or the solve will silently " *
              "satisfy the wrong condition." maxlog=5
    end
    return ComplexF64(0)
end

"""
Project a constant value onto the current subproblem's Fourier mode.

Tarang uses unnormalized `FFTW.rfft` / `FFTW.fft` along each separable
axis. For a constant `v` in grid space, the only nonzero Fourier
coefficient is at the full-DC mode `(kx=0, ky=0, ...)`, with value
`v * Nx * Ny * ...` (product of all Fourier-axis grid sizes). All
non-DC Fourier modes project to zero.

For problems without any Fourier axis (e.g. a pure-coupled BVP), the
DC-mode value is `v` itself — no scaling applied.
"""
function _bc_constant_projection(v::Float64, sp::Subproblem)
    v == 0 && return ComplexF64(0)
    return _bc_constant_projection(v, _bc_fourier_axis_sizes(sp),
                                    _subproblem_fourier_group_indices(sp))
end

function _bc_constant_projection(v::Float64, sizes::Tuple, fourier_idx::Tuple)
    # Every separable (Fourier) axis of this subproblem must be at its DC
    # mode (global index 1) for a constant to have any contribution.
    for k in fourier_idx
        if k != 1
            return ComplexF64(0)
        end
    end
    # DC on every Fourier axis → `v * ∏ N_k` via unnormalized FFTs.
    scale = 1.0
    for N in sizes
        scale *= Float64(N)
    end
    return ComplexF64(v * scale)
end

"""
    _bc_fourier_axis_sizes(sp) -> Tuple{Vararg{Int}}

Ordered list of grid sizes for the problem's separable (Fourier) axes. Used
to determine the expected output shape of a BC array so that lower-rank
user inputs (e.g. `sin(x)` in a 3D problem) can be broadcast to the full
output before being transformed.
"""
function _bc_fourier_axis_sizes(sp::Subproblem)
    cached = sp.runtime.bc_fourier_sizes
    cached !== nothing && return cached
    sizes = Int[]
    seen = Set{String}()
    for var in sp.problem.variables
        for comp in scalar_components(var)
            for basis in comp.bases
                basis === nothing && continue
                isa(basis, FourierBasis) || continue
                label = String(basis.meta.element_label)
                label in seen && continue
                push!(seen, label)
                push!(sizes, basis.meta.size)
            end
        end
    end
    # A new compiled subproblem is required when the problem's bases change,
    # just as for its matrices and existing field-size/index caches.
    return sp.runtime.bc_fourier_sizes = Tuple(sizes)
end

"""
    _subproblem_fourier_group_indices(sp) -> Tuple{Vararg{Int}}

Return the 1-based Fourier mode indices for every separable axis of this
subproblem. For a 2D problem with one Fourier axis this is `(kx_global,)`;
for a 3D problem it's `(kx_global, ky_global)`.
"""
function _subproblem_fourier_group_indices(sp::Subproblem)
    cached = sp.runtime.bc_fourier_indices
    cached !== nothing && sp.runtime.bc_fourier_group === sp.group && return cached
    idx = Int[]
    for g in sp.group
        g isa Integer || continue
        push!(idx, g + 1)
    end
    sp.runtime.bc_fourier_group = sp.group
    return sp.runtime.bc_fourier_indices = Tuple(idx)
end

"""
    _bc_array_projection(arr, sp)

Project a grid-space array `arr` (from a space-dependent BC) onto the current
subproblem's Fourier mode. Returns a `ComplexF64` value suitable for writing
into the BC row of the raw equation-space vector.

The FFT result is cached by original array identity and boundary-plane shape in
the problem's compiled runtime. Expansion onto the full boundary plane happens
only on a cache miss, so all modes sharing an `ArrayOperator` reuse one FFT per
refresh, including profiles that vary along only one tangential coordinate.
"""
function _bc_array_projection(arr::AbstractArray, sp::Subproblem)::ComplexF64
    (arr === nothing || length(arr) == 0) && return ComplexF64(0)
    return _bc_array_projection(arr, sp, _bc_fourier_axis_sizes(sp),
                                _subproblem_fourier_group_indices(sp))
end

# Recover the concrete geometry tuple types at the cache boundary, so tuple
# iteration and shape-key lookup do not box integers once per Fourier mode.
function _bc_array_projection(arr::AbstractArray, sp::Subproblem,
                              fourier_sizes::S, fourier_idx::I)::ComplexF64 where {S<:Tuple,I<:Tuple}
    if isempty(fourier_sizes)
        # No Fourier axes at all (pure-coupled / BVP-like). Use arr[1] as
        # the DC-mode value.
        return ComplexF64(first(arr))
    end

    coeffs = _get_or_compute_bc_array_coeffs!(arr, sp, fourier_sizes)
    coeffs === nothing && return ComplexF64(0)
    # Refine the cache's rank-erased array type before reading a scalar. This
    # keeps both indexing and the ComplexF64 return unboxed in the mode loop.
    if coeffs isa Vector{ComplexF64} || coeffs isa Matrix{ComplexF64} ||
       coeffs isa Array{ComplexF64,3}
        return _sample_bc_coefficients(coeffs, fourier_idx)
    end
    @warn "BC array projection: unsupported coefficient rank $(ndims(coeffs))" maxlog=3
    return ComplexF64(0)
end

function _sample_bc_coefficients(coeffs::Array{ComplexF64,N}, fourier_idx::Tuple)::ComplexF64 where N
    if N == 1
        kx = isempty(fourier_idx) ? 1 : first(fourier_idx)
        return (kx >= 1 && kx <= length(coeffs)) ?
               ComplexF64(coeffs[kx]) : ComplexF64(0)
    elseif N == 2
        length(fourier_idx) >= 2 || return ComplexF64(0)
        kx, ky = fourier_idx[1], fourier_idx[2]
        return (1 <= kx <= size(coeffs, 1) && 1 <= ky <= size(coeffs, 2)) ?
               ComplexF64(coeffs[kx, ky]) : ComplexF64(0)
    elseif N == 3
        length(fourier_idx) >= 3 || return ComplexF64(0)
        kx, ky, kz = fourier_idx[1], fourier_idx[2], fourier_idx[3]
        return (1 <= kx <= size(coeffs, 1) &&
                1 <= ky <= size(coeffs, 2) &&
                1 <= kz <= size(coeffs, 3)) ?
               ComplexF64(coeffs[kx, ky, kz]) : ComplexF64(0)
    else
        @warn "BC array projection: unsupported coefficient rank $(ndims(coeffs))" maxlog=3
        return ComplexF64(0)
    end
end

# Only FFT working storage survives boundary refresh. Coefficients returned by
# the cache remain independently owned snapshots: another boundary, or the next
# refresh, must never overwrite an array retained by a caller. The coordinating
# task consumes the scratch synchronously, just as it owns the BC value cache.
const _BC_FFT_SCRATCH_CAPACITY = 8

struct BCProjectionScratch{T,N,P}
    plane::Array{T,N}
    plan::P
end

function _bc_projection_scratch!(context, ::Type{T}, shape::NTuple{N,Int}) where {T,N}
    cache = get!(context.workspaces, :bc_fft_scratch) do
        Dict{Tuple, Any}()
    end::Dict{Tuple, Any}
    key = (T, shape)
    scratch = get(cache, key, nothing)
    scratch !== nothing && return scratch
    length(cache) >= _BC_FFT_SCRATCH_CAPACITY && empty!(cache)
    plane = Array{T}(undef, shape)
    flags = FFTW.ESTIMATE | FFTW.UNALIGNED
    plan = T <: Complex ? FFTW.plan_fft(plane; flags) : FFTW.plan_rfft(plane; flags)
    return cache[key] = BCProjectionScratch(plane, plan)
end

function _compute_bc_coefficients!(scratch::BCProjectionScratch, arr)
    _copy_bc_plane!(scratch.plane, arr)
    # Each refreshed boundary owns its output, while input and plan are reused.
    shape = size(scratch.plane)
    coeff_shape = eltype(scratch.plane) <: Complex ? shape :
                  Base.setindex(shape, first(shape) ÷ 2 + 1, 1)
    coefficients = Array{ComplexF64}(undef, coeff_shape)
    mul!(coefficients, scratch.plan, scratch.plane)
    return coefficients
end

"""
    _get_or_compute_bc_array_coeffs!(arr, sp, fourier_sizes) -> coefficients

Cache-backed helper that:
1. Reshapes/broadcasts `arr` to the full Fourier-output shape (derived from
   `fourier_sizes`) so lower-rank inputs work in higher-dim problems.
2. Takes an unnormalized forward FFT along the first axis (via `rfft` for
   real input) and complex FFT along remaining axes.
3. Returns the complex coefficient array for downstream indexing.

The result is cached by original array identity (`IdDict`) and Fourier plane
shape. Looking up the original array before expansion avoids retaining and
transforming a fresh full plane for every mode. Boundary refresh invalidates
this cache; gather boundary data on the coordinating task before local solves.
"""
function _get_or_compute_bc_array_coeffs!(arr::AbstractArray,
                                          sp::Subproblem,
                                          fourier_sizes::Union{Tuple, Vector{Int}})
    context = compiled_problem(sp.problem).caches
    cache = context.bc_rfft
    shape = Tuple(fourier_sizes)
    by_shape = get(cache, arr, nothing)
    cached = by_shape === nothing ? nothing : get(by_shape, shape, nothing)
    cached !== nothing && return cached

    coeffs = try
        T = eltype(arr) <: Complex ? ComplexF64 : Float64
        scratch = _bc_projection_scratch!(context, T, shape)
        _compute_bc_coefficients!(scratch, arr)
    catch err
        # Invalid/ambiguous geometry is a user error, not a zero boundary.
        err isa ArgumentError && rethrow()
        @warn "BC array FFT failed: $err" maxlog=1
        return nothing
    end

    if by_shape === nothing
        by_shape = Dict{Tuple, Array{ComplexF64}}()
        cache[arr] = by_shape
    end
    by_shape[shape] = coeffs
    return coeffs
end

"""Copy/broadcast a boundary into FFT input without intermediate plane copies."""
function _copy_bc_plane!(plane::Array{T,N}, arr::AbstractArray) where {T,N}
    shape = size(plane)
    if length(arr) == 1
        fill!(plane, first(arr))
    elseif size(arr) == shape
        copyto!(plane, arr)
    elseif ndims(arr) == N && all(d -> size(arr, d) in (1, shape[d]), 1:N)
        plane .= arr
    elseif ndims(arr) == 1 && N >= 2 && count(==(length(arr)), shape) > 1
        throw(ArgumentError("BC array of length $(length(arr)) is ambiguous on a boundary " *
                            "plane of size $shape; provide its axis shape (e.g. (1, N))."))
    elseif length(arr) == length(plane)
        copyto!(plane, reshape(arr, shape))
    elseif ndims(arr) == 1 && count(==(length(arr)), shape) == 1
        axis = findfirst(==(length(arr)), shape)
        reshaped = reshape(arr, ntuple(d -> d == axis ? length(arr) : 1, N))
        plane .= reshaped
    elseif ndims(arr) < N && all(d -> size(arr, d) in (1, shape[d]), 1:N)
        plane .= reshape(arr, ntuple(d -> size(arr, d), N))
    else
        throw(ArgumentError("BC array shape $(size(arr)) incompatible with Fourier output " *
                            "shape $shape; provide a broadcastable boundary plane."))
    end
    return plane
end

"""
    invalidate_bc_array_cache!(problem)

Clear the per-problem BC-array FFT cache. Call when BC arrays change (e.g.
at the start of each step, or after evaluating new time-dependent array
values via `_apply_bc_values_to_equations!`).
"""
function invalidate_bc_array_cache!(problem)
    empty!(compiled_problem(problem).caches.bc_rfft)
    return
end

"""
    gather_eqn_F!(dest, sp, solver, pde_F_fields, state_fields)

Pack PDE-equation F values into equation-space. For each equation with a time
derivative (`M` term), pull F from `pde_F_fields` at the equation's target
state-field indices. Algebraic/BC equations (no `M` term) contribute ZERO to
this vector — they are handled separately via `gather_alg_F!` + a direct RHS
override in the stepper, because the IMEX-RK accumulated formula gives the
wrong `1/γ` scaling for inhomogeneous algebraic constraints.
"""
function gather_eqn_F!(dest::AbstractVector{ComplexF64}, sp::Subproblem, solver,
                       pde_F_fields::Vector, state_fields::Vector)
    eqn_sizes = _subproblem_eqn_sizes(sp)
    eqn_targets = _subproblem_eqn_targets(sp, state_fields)
    I_raw = _subproblem_raw_eqn_size(sp)

    raw = _subproblem_cached_vector!(sp, :gather_eqn_F_raw, I_raw; like=dest)
    _gather_eqn_F_raw!(raw, sp, pde_F_fields, state_fields, eqn_sizes, eqn_targets)
    compress_equation_space!(dest, sp, raw)
    return dest
end

# Recover the concrete cached-buffer type before indexing it or walking field
# targets. The equation geometry is already compiled; no EquationIR lookup is
# needed for PDE rows, and all field values are still gathered at this call.
function _gather_eqn_F_raw!(raw::AbstractVector{ComplexF64}, sp::Subproblem,
                            pde_F_fields::Vector, state_fields::Vector,
                            eqn_sizes::Vector{Int}, eqn_targets::Vector{Vector{Int}})
    fill!(raw, zero(eltype(raw)))

    kx_global = Int(_kx_index_global(sp))

    i0 = 0
    for eq_idx in eachindex(eqn_sizes)
        eq_size = eqn_sizes[eq_idx]
        if eq_size == 0
            continue
        end

        target_indices = eqn_targets[eq_idx]
        if !isempty(target_indices)
            offset = i0
            for tidx in target_indices
                if tidx >= 1 && tidx <= length(pde_F_fields)
                    fld = pde_F_fields[tidx]
                    if fld !== nothing
                        offset = _gather_field_raw!(raw, offset, fld, kx_global, sp)::Int
                        continue
                    end
                end
                if tidx >= 1 && tidx <= length(state_fields)
                    offset += subproblem_field_size(sp, state_fields[tidx])
                end
            end
        end
        # Algebraic rows intentionally left zero — see gather_alg_F!.

        i0 += eq_size
    end

    return raw
end

function _subproblem_algebraic_blocks(sp::Subproblem, eqns::AbstractVector)
    cached = sp.runtime.algebraic_blocks
    cached !== nothing && return cached
    eqn_sizes = _subproblem_eqn_sizes(sp)
    basis = _subproblem_cheb_basis_from_sp(sp)
    Nz = basis === nothing ? 1 : basis.meta.size
    blocks = _SubproblemAlgebraicBlock[]
    offset = 0
    for (eq_idx, eq_data) in enumerate(eqns)
        n = eqn_sizes[eq_idx]
        if n > 0 && _is_zero_m_term(get(eq_data, "M", nothing))
            push!(blocks, _SubproblemAlgebraicBlock(eq_idx, offset, n,
                                                   _is_bulk_eqn_size(n, Nz)))
        end
        offset += n
    end
    sp.runtime.algebraic_blocks = blocks
    return blocks
end

# Is this BC F expression provably IMMUTABLE — a value fixed at problem build?
# `ConstantOperator.value::Float64` is an immutable struct field, so constants
# and compound-constant trees over them can never change. A `ScalarField`
# parameter or an `ArrayOperator` leaf is read LIVE at each gather
# (`_extract_scalar` / `_bc_array_projection` read the current data), and users
# ramp those mid-run without registering the BC as time-dependent — so any such
# leaf makes the expression mutable and the gather-once skip unsafe.
_alg_F_expr_immutable(::Nothing) = true
_alg_F_expr_immutable(::ZeroOperator) = true
_alg_F_expr_immutable(::Number) = true
_alg_F_expr_immutable(::ConstantOperator) = true
_alg_F_expr_immutable(n::NegateOperator) = _alg_F_expr_immutable(n.operand)
_alg_F_expr_immutable(e::AddOperator)      = _alg_F_expr_immutable(e.left) && _alg_F_expr_immutable(e.right)
_alg_F_expr_immutable(e::SubtractOperator) = _alg_F_expr_immutable(e.left) && _alg_F_expr_immutable(e.right)
_alg_F_expr_immutable(e::MultiplyOperator) = _alg_F_expr_immutable(e.left) && _alg_F_expr_immutable(e.right)
_alg_F_expr_immutable(e::DivideOperator)   = _alg_F_expr_immutable(e.left) && _alg_F_expr_immutable(e.right)
_alg_F_expr_immutable(e::PowerOperator)    = _alg_F_expr_immutable(e.left) && _alg_F_expr_immutable(e.right)
_alg_F_expr_immutable(::Any) = false

"""
    alg_F_is_static(sp) -> Bool

`true` iff every algebraic (BC) F expression this subproblem gathers is
provably immutable, making the steppers' gather-once ALG_F skip safe. The
expression TREES are fixed after problem build, so the classification is
computed once and cached in `sp.runtime.alg_F_static`.
"""
function alg_F_is_static(sp::Subproblem)
    cached = sp.runtime.alg_F_static
    cached !== nothing && return cached::Bool
    static = true
    eqns = sp.problem.equation_data
    for block in _subproblem_algebraic_blocks(sp, eqns)
        eq_data = eqns[block.equation]
        F_expr = get(eq_data, "F_expr", nothing)
        if F_expr === nothing
            F_expr = get(eq_data, "F", nothing)
        end
        if !_alg_F_expr_immutable(F_expr)
            static = false
            break
        end
    end
    sp.runtime.alg_F_static = static
    return static
end

"""
    gather_alg_F!(dest, sp)

Pack algebraic-constraint F values (from BC / constraint equations that have
no time derivative) into equation-space, with zeros at PDE rows.

For each equation without an `M` term, evaluate its stored `F` expression
(typically a `ConstantOperator`) and project onto the current Fourier mode.
The result is used by the stepper to OVERRIDE the BC rows of the RHS with
`dt * a_ii * F_alg`, yielding `L_row * X = F_alg` at each stage — the correct
enforcement of the algebraic constraint.

This override is necessary because the standard IMEX-RK accumulated RHS
formula gives `L_row * X = (A^E[i,j]/a_ii) * F_BC` which is wrong by a factor
of `1/γ` for inhomogeneous algebraic constraints.
"""
function gather_alg_F!(dest::AbstractVector{ComplexF64}, sp::Subproblem)
    problem = sp.problem
    eqns = problem.equation_data

    blocks = _subproblem_algebraic_blocks(sp, eqns)
    I_raw = _subproblem_raw_eqn_size(sp)

    # Build the sparse BC F vector on the HOST via scalar writes (a few nonzero
    # entries at BC row offsets, the rest zero). Then upload once into the
    # device-resident `raw` buffer via `_assign_to_buffer!` — this keeps
    # scalar indexing off of GPU arrays so the helper is safe under
    # `CUDA.allowscalar(false)`.
    raw_cpu = sp.runtime.gather_alg_F_raw_cpu
    if raw_cpu === nothing || length(raw_cpu) != I_raw
        raw_cpu = zeros(ComplexF64, I_raw)
        sp.runtime.gather_alg_F_raw_cpu = raw_cpu
    end
    _gather_alg_F_raw!(raw_cpu, sp, eqns, blocks)

    # Upload the CPU-built raw vector into the device-resident raw buffer.
    # This is an INTENTIONAL one-shot H2D upload of a freshly host-built staging
    # vector (the scalar writes above must stay off device arrays), NOT field
    # staging — so use a direct copyto! rather than `_assign_to_buffer!`, whose
    # same-architecture guard exists to catch *accidental* CPU/GPU mixing and
    # would (correctly, for its purpose) refuse this transfer. Routing through
    # it killed every GPU coupled subproblem step at the pre-stage ALG_F gather.
    raw = _subproblem_cached_vector!(sp, :gather_alg_F_raw, I_raw; like=dest)
    copyto!(raw, raw_cpu)

    compress_equation_space!(dest, sp, raw)
    # Record WHICH vector now holds the gathered ALG_F, so static-BC steppers
    # can skip this whole host-rebuild + upload on later steps (see the field's
    # comment in SubproblemRuntimeCache).
    sp.runtime.alg_F_gathered_into = dest
    return dest
end

function _gather_alg_F_raw!(raw::Vector{ComplexF64}, sp::Subproblem,
                            eqns::AbstractVector,
                            blocks::Vector{_SubproblemAlgebraicBlock})
    fill!(raw, zero(ComplexF64))
    for block in blocks
        # Boundary refresh can replace either the expression or the entire IR
        # entry. Cache only its row geometry; always read the current value.
        eq_data = eqns[block.equation]
        F_expr = get(eq_data, "F_expr", nothing)
        F_expr === nothing && (F_expr = get(eq_data, "F", nothing))
        _write_alg_F_block!(raw, F_expr, sp, block.offset, block.size, block.bulk)
    end
    return raw
end

function _write_alg_F_block!(raw::Vector{ComplexF64}, F_expr, sp::Subproblem,
                             offset::Int, size::Int, bulk::Bool)
    _is_zero_F_expr(F_expr) && return nothing
    # In a BVP the bulk PDE is algebraic too. Its value is computed as before,
    # but unsupported-expression warnings apply only to actual boundary rows.
    coeff = _evaluate_alg_F(F_expr, sp; warn_unsupported=!bulk)::ComplexF64
    if coeff != 0
        # A vector boundary equation repeats the same prescribed scalar over
        # its component rows, matching kron(I_ncomp, row) in its LHS matrix.
        @inbounds for r in 1:size
            raw[offset + r] = coeff
        end
    end
    return nothing
end
