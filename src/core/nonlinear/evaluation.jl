# Set true to collect wall-clock timing stats; false = zero overhead (dead-code eliminated).
const _TRACK_NL_TIMING = false

_ivp_expression_children(op::AdvectionOperator) = (op.velocity, op.scalar)
_ivp_expression_children(op::NonlinearAdvectionOperator) = (op.velocity,)
_ivp_expression_children(op::ConvectiveOperator) = (op.field1, op.field2)

function evaluate(op::Union{AdvectionOperator, NonlinearAdvectionOperator}, layout::Symbol=:g)
    result = evaluate_nonlinear_term(op, layout)
    ensure_layout!(result, layout)
    return result
end

function evaluate(op::ConvectiveOperator, layout::Symbol=:g)
    result = evaluate_convective_operator(op)
    ensure_layout!(result, layout)
    return result
end

# Main nonlinear evaluation functions
"""Evaluate u·∇φ nonlinear term using transform method"""
function evaluate_nonlinear_term(op::AdvectionOperator, layout::Symbol=:g)

    velocity = op.velocity
    scalar = op.scalar

    # Get distributor for transform operations
    dist = velocity.dist

    # Create nonlinear evaluator if not exists
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    evaluator = dist.nonlinear_evaluator

    # Compute gradient of scalar field: ∇φ
    grad_scalar = evaluate_gradient(Gradient(scalar, dist.coordsys), :g)

    # Evaluate u·∇φ = u_x ∂φ/∂x + u_y ∂φ/∂y (+ u_z ∂φ/∂z in 3D)
    # Use in-place accumulation to avoid lazy Add trees and per-iteration allocation
    result = ScalarField(dist, "$(op.name)_$(scalar.name)", scalar.bases, promote_type(velocity.dtype, scalar.dtype))
    preset_scales!(result, scalar.scales)
    result_data = grid_data!(result)
    fill!(result_data, zero(scalar.dtype))

    # Sum velocity components times gradient components (in-place)
    for i in 1:length(velocity.components)
        product = evaluate_transform_multiply(velocity.components[i], grad_scalar.components[i], evaluator; own=false)
        product_data = grid_data!(product)
        if isa(result_data, PencilArrays.PencilArray) && isa(product_data, PencilArrays.PencilArray)
            parent(result_data) .+= parent(product_data)
        else
            result_data .+= product_data
        end
    end

    return result
end

"""Evaluate (u·∇)u nonlinear momentum term"""
function evaluate_nonlinear_term(op::NonlinearAdvectionOperator, layout::Symbol=:g)

    velocity = op.velocity
    dist = velocity.dist

    # Create nonlinear evaluator if needed
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    evaluator = dist.nonlinear_evaluator

    # Result is a vector field
    result = VectorField(dist, dist.coordsys, "$(op.name)_$(velocity.name)", velocity.bases, velocity.dtype)
    preset_scales!(result, first(velocity.components).scales)

    # For each component: (u·∇)u_i = u_j ∂u_i/∂x_j (summed over j)
    # Use in-place accumulation to avoid lazy Add trees and allocation per iteration
    for i in 1:length(velocity.components)
        comp_data = grid_data!(result.components[i])
        fill!(comp_data, zero(velocity.dtype))

        # Sum over all spatial directions (in-place)
        for j in 1:length(velocity.components)
            coord = dist.coordsys[j]
            du_i_dx_j = evaluate_differentiate(Differentiate(velocity.components[i], coord, 1), :g)

            product = evaluate_transform_multiply(velocity.components[j], du_i_dx_j, evaluator; own=false)
            product_data = grid_data!(product)
            if isa(comp_data, PencilArrays.PencilArray) && isa(product_data, PencilArrays.PencilArray)
                parent(comp_data) .+= parent(product_data)
            else
                comp_data .+= product_data
            end
        end
    end

    return result
end

# The direct evaluator historically computes same-kind products at the first
# operand's precision. Ordinary field arithmetic explicitly requests promotion.
_nonlinear_product_dtype(a, b) = (a.dtype <: Real) == (b.dtype <: Real) ?
    a.dtype : promote_type(a.dtype, b.dtype)

"""Efficiently multiply two fields using transform method and proper dealiasing.

    Fourier and ChebyshevT axes with padding enabled evaluate the product on
    an oversampled grid and project back to the retained modes. Serial CPU
    and GPU use FFTs. MPI resizes each axis while its pencil is local.
    Other polynomial axes keep their original collocation nodes.

    # Buffer ownership

    The dealiased paths write into `_NL_RESULT_POOL`, a rotating pool of
    `_NL_RESULT_POOL_SIZE` buffers whose index is shared across every caller of
    this evaluator — including the solver RHS. `own=true` (the default) copies the
    result out, so the returned field is safe to store, return across an API
    boundary, or hand to user code.

    Pass `own=false` ONLY when the result is fully consumed before control returns
    to the caller (an immediate `copyto!`, `.+=`, or arithmetic that allocates its
    own output). Borrowing and then retaining is a silent wrong-answer bug: the
    buffer is reissued after the index wraps and the retained field changes value
    with no error. `Base.:*(::ScalarField, ::ScalarField)` did exactly that, and
    because the index is shared with the RHS, one held product was lost after
    eight internal ones. See `_own_borrowed_field` in `src/core/field_pool.jl`.

    Defaulting to `own=true` makes a forgotten annotation slow, not wrong.
    """
function evaluate_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator;
                                     result_layout::Symbol=:g, own::Bool=true)

    _TRACK_NL_TIMING && (start_time = time())

    _check_field_arithmetic_compatibility(field1, field2, "multiply")
    if _has_nonunit_arithmetic_scale(field1)
        # Scaled fields use the caller's working grid, as in field arithmetic.
        result = field1 * field2
        ensure_layout!(result, result_layout)
        return result
    end

    # Try proper 3/2-rule padded dealiasing. Gate on the per-axis basis `dealias`
    # settings (factor > 1 on a Fourier or ChebyshevT axis) rather than the evaluator's global
    # default, so bases set to dealias ≤ 1 are computed without dealiasing.
    result_dtype = _nonlinear_product_dtype(field1, field2)
    if _any_axis_dealias(field1.bases, evaluator.dealiasing_factor)
        # Match precision without changing either operand's Fourier interpolant.
        T = real(result_dtype)
        field1 = _product_operand_at_precision(field1, T)
        field2 = _product_operand_at_precision(field2, T)
        # Detect the MPI pencil path WITHOUT forcing a layout first, so the
        # distributed routine can truncate inputs in coefficient space (avoids a
        # c→g→c round trip when operands arrive in coeff space — the usual case).
        if evaluator.dist.size > 1
            # MPI path. Primary: FULL 3/2-rule padded dealiasing via the transpose-pad
            # distributed FFT (evaluate_padded_multiply_distributed) — the cross-rank
            # redistribution needed to embed the N-mode spectrum into the 3N/2 padded
            # spectrum is done with PencilArrays transposes, so the result matches the
            # serial padded multiply to roundoff (round-7 audit 2026-06). Covers 2D + 3D
            # (incl. 2D process mesh) pure-Fourier and mixed Fourier×(Cheb/Jacobi).
            # Fallback: it returns `nothing` for the cases it does not cover (>3D, a
            # decomposed non-Fourier axis), routing to local 2/3-rule
            # truncation-after-multiply (evaluate_truncated_multiply_distributed).
            padded = if field1.dist.use_pencil_arrays
                evaluate_padded_multiply_distributed(field1, field2, evaluator; result_layout)
            else
                _transpose_padded_product(field1, field2, evaluator, result_layout)
            end
            result = padded !== nothing ? padded :
                     evaluate_truncated_multiply_distributed(field1, field2, evaluator;
                                                             result_layout=result_layout)
            evaluator.performance_stats.total_evaluations += 1
            if _TRACK_NL_TIMING
                elapsed = time() - start_time
                evaluator.performance_stats.total_time += elapsed
                evaluator.performance_stats.dealiasing_time += elapsed
            end
            return own ? _own_borrowed_field(result) : result
        else
            # Serial path can consume compatible Fourier coefficients directly.
            ws = _get_padded_workspace!(evaluator, field1.bases, T;
                                         real_input=field1.dtype <: Real && field2.dtype === field1.dtype)
            if ws !== nothing
                result = evaluate_padded_multiply(field1, field2, evaluator, ws; result_layout)
                ensure_layout!(result, result_layout)  # serial: cheap FFT, no transpose
                evaluator.performance_stats.total_evaluations += 1
                if _TRACK_NL_TIMING
                    elapsed = time() - start_time
                    evaluator.performance_stats.total_time += elapsed
                    evaluator.performance_stats.dealiasing_time += elapsed
                end
                return own ? _own_borrowed_field(result) : result
            end
        end
    end

    # Direct product for bases without an active padded workspace.
    ensure_layout!(field1, :g)
    ensure_layout!(field2, :g)
    result = ScalarField(field1.dist, "_nl_product", field1.bases, result_dtype)

    result_data = grid_data!(result)
    field1_data = get_grid_data(field1)
    field2_data = get_grid_data(field2)

    if isa(result_data, PencilArrays.PencilArray)
        parent(result_data) .= parent(field1_data) .* parent(field2_data)
    else
        gpu_multiply_fields!(result["g"], field1["g"], field2["g"])
    end

    if _any_axis_dealias(field1.bases, evaluator.dealiasing_factor)
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end

    ensure_layout!(result, result_layout)
    evaluator.performance_stats.total_evaluations += 1
    _TRACK_NL_TIMING && (evaluator.performance_stats.total_time += time() - start_time)

    return result
end

"""
    gpu_multiply_fields!(result_data, data1, data2)

GPU-accelerated pointwise multiplication.
This function is overridden by the CUDA extension to use GPU kernels.
"""
function gpu_multiply_fields!(result_data::AbstractArray, data1::AbstractArray, data2::AbstractArray)
    # Default CPU implementation using broadcasting
    result_data .= data1 .* data2
    return result_data
end

# Number of physical nodes to use in a padded product. Lobatto endpoints
# correspond to the endpoints of half an even periodic grid.
function _padded_product_nodes(basis, n, factor)
    strength = _axis_dealias_factor(basis, factor)
    if basis isa FourierBasis
        return _padded_fourier_size(n, strength)
    elseif basis isa ChebyshevT && n > 1 && strength > 1
        return _padded_fourier_size(2 * (n - 1), strength) ÷ 2 + 1
    end
    return n
end

# The sweep owns the input buffer and may overwrite it. The local DCT uses
# normalized Chebyshev coefficients, so this resize needs no FFT length factor.
function _resize_chebyshev_product_axis!(out, input, axis, field)
    isempty(input) && return out
    arch = field.dist.architecture
    if arch isa CPU && field.dist.use_pencil_arrays
        transforms = _field_transform_bundle(field).transforms
        transform = only(t for t in transforms if t isa ChebyshevTransform && t.axis == axis)
        _apply_forward!(input, input, transform)
        return _apply_backward!(out, input, transform, size(out, axis))
    end
    dct_in_dim!(input, axis, :forward, arch)
    keep = 1:min(size(input, axis), size(out, axis))
    size(out, axis) > length(keep) && fill!(out, zero(eltype(out)))
    selectdim(out, axis, keep) .= selectdim(input, axis, keep)
    return dct_in_dim!(out, axis, :backward, arch)
end

# Resize one axis while a rank owns the entire axis. Both pencils have the
# same transverse ranges in this layout.
function _resize_transpose_axis!(dst, src, axis, field)
    topo = src.topology
    layout = if axis == 1 && topo.Rx > 1
        length(src.global_shape) == 2 && topo.Ry == 1 ? YLocal : XLocal
    elseif axis == 2 && topo.Ry > 1
        YLocal
    else
        ZLocal
    end
    layout != ZLocal && transpose_z_to_y!(src)
    layout == XLocal && transpose_y_to_x!(src)
    dst.buffers.active_layout[] = layout
    input, output = current_data(src), current_data(dst)
    if field.bases[axis] isa ChebyshevT
        _resize_chebyshev_product_axis!(output, input, axis, field)
    elseif !isempty(input)
        arch = src.buffers.architecture
        fft_in_dim!(input, axis, :forward, arch)
        cosine_dims = field.dtype <: Real ? (axis,) : ()
        if dst.global_shape[axis] > src.global_shape[axis]
            _pad_spectral!(output, input, size(input), size(output), [axis], cosine_dims)
        else
            _truncate_spectral!(output, input, size(output), size(input), [axis], cosine_dims)
        end
        fft_in_dim!(output, axis, :backward, arch)
    end
    layout == XLocal && transpose_x_to_y!(dst)
    layout != ZLocal && transpose_y_to_z!(dst)
    return dst
end

"""Padded product on the explicit transpose backend, using rank-local buffers."""
function _transpose_padded_product(a, b, evaluator, result_layout)
    dist = a.dist
    dist.closed && throw(ArgumentError("cannot multiply fields on a closed Distributor"))
    shape = map(basis -> basis.meta.size, a.bases)
    padded_shape = ntuple(d -> _padded_product_nodes(a.bases[d], shape[d], evaluator.dealiasing_factor), length(shape))
    axes_to_pad = findall(d -> padded_shape[d] != shape[d], eachindex(shape))
    template = _product_template(a, b)
    padded_workspace(sz) = get!(dist.transpose_workspace_cache, (:nonlinear, sz, template.dtype)) do
        _build_transpose_workspace(template, sz, nothing)
    end
    stages = [padded_workspace(shape)]
    for axis in axes_to_pad
        shape = Base.setindex(shape, padded_shape[axis], axis)
        push!(stages, padded_workspace(shape))
    end
    function interpolate!(field)
        first(stages).buffers.active_layout[] = ZLocal
        first(stages).buffers.z_local_data .= grid_data!(field)
        for (i, axis) in enumerate(axes_to_pad)
            _resize_transpose_axis!(stages[i+1], stages[i], axis, field)
        end
        return last(stages).buffers.z_local_data
    end
    padded = interpolate!(a)
    if a === b
        padded .*= padded
    else
        # Retain the first interpolant while the second uses the same transpose buffers.
        i = findfirst(x -> typeof(x) === typeof(padded) && size(x) == size(padded), evaluator.scratch_arrays)
        saved = if isnothing(i)
            push!(evaluator.scratch_arrays, similar(padded))
            last(evaluator.scratch_arrays)
        else
            evaluator.scratch_arrays[i]
        end
        copyto!(saved, padded)
        interpolate!(b)
        padded .*= saved
    end
    for i in reverse(eachindex(axes_to_pad))
        _resize_transpose_axis!(stages[i], stages[i+1], axes_to_pad[i], template)
    end
    result = _checkout_nl_result!(evaluator, template)
    data = first(stages).buffers.z_local_data
    T = real(result.dtype)
    scale = prod((T(shape[d]) / T(first(stages).global_shape[d]) for d in axes_to_pad if a.bases[d] isa FourierBasis); init=one(T))
    if result.dtype <: Real
        get_grid_data(result) .= scale .* real.(data)
    else
        get_grid_data(result) .= scale .* data
    end
    result.current_layout = :g
    ensure_layout!(result, result_layout)
    return result
end

# ── Cached workspace for the distributed padded multiply ─────────────────────
# evaluate_padded_multiply_distributed runs an IDENTICAL transpose / pad / FFT
# SEQUENCE on every call for a fixed problem (topology, global size and the
# decomposition are stable). This workspace lets the routine REUSE across calls:
#   (a) the intermediate Pencils — immutable metadata that PencilArrays.transpose!
#       only reads, so sharing is safe (kills the ~14 fresh Pencils/call);
#   (b) the complex N- and Mp-sized scratch PencilArray buffers (kills the ~14
#       full-grid PencilArray allocations/call);
#   (c) plan-backed in-place local-axis FFTs replacing out-of-place fft()/ifft()
#       (kills the ~12 full-grid FFT-output allocations/call).
# The result field comes from the shared rotating pool via _checkout_nl_result!
# (kills the fresh ScalarField/call). Mirrors the cached-plan+scratch pattern of
# _get_batched_backward_plan!/_BATCHED_PENCIL_PLAN_CACHE and the in-place-plan +
# CPU/GPU split of _get_padded_workspace!/PaddedDealiasingWorkspace (serial sibling).
#
# Scratch is leased until its last read, then reused by a later operation with
# the same pencil geometry. Both padded operands remain leased until multiplied;
# spent transpose and FFT inputs can be reused immediately after completion.
# Every buffer is FULLY overwritten
# (transpose! destination, a `.=` copy, or a full-coverage pad/truncate) before it
# is read, so stale data from a previous call is never observed — exactly the
# property the original relied on when it allocated `undef` buffers each call.
mutable struct PaddedDistDealiasWorkspace{CT}
    topo::Any
    pencils::Dict{Tuple{Tuple, Tuple}, Any}     # (global size, decomp dims) → memoized NoPermutation Pencil
    buffers::Vector{Any}
    in_use::BitVector
    transpose_count::Int                         # decomposition changes in the last product
    transpose_elements::Int                      # global elements carried by those transposes
    fwd_plans::Dict{Tuple{Tuple, Int}, Any}      # (local array size, axis) → in-place forward FFT plan
    inv_plans::Dict{Tuple{Tuple, Int}, Any}      # (local array size, axis) → in-place inverse FFT plan
end

const _PADDED_DIST_WS_CACHE = Dict{Tuple, Any}()

# Keyed (like _get_batched_backward_plan!) on the problem identity — topology
# object, basis kinds/sizes, complex eltype, and the global/padded sizes +
# decomposition that fully determine the transpose sequence. `topo` is the exact
# object the cached pencils are built from, so reused pencils stay compatible with
# the field's grid pencil in transpose! (the same assumption the original makes by
# taking `topo = topology(pencil(gd1))`).
function _get_padded_dist_workspace!(dist, topo, bases, ::Type{CT}, N, Mp, decomp0, fourier_axes) where {CT}
    key = (objectid(dist), objectid(topo), CT,
           map(b -> (nameof(typeof(b)), b.meta.size), bases), N, Mp, decomp0, fourier_axes)
    cached = get(_PADDED_DIST_WS_CACHE, key, nothing)
    cached === nothing || return cached::PaddedDistDealiasWorkspace{CT}
    ws = PaddedDistDealiasWorkspace{CT}(topo,
        Dict{Tuple{Tuple, Tuple}, Any}(), Any[], BitVector(), 0, 0,
        Dict{Tuple{Tuple, Int}, Any}(), Dict{Tuple{Tuple, Int}, Any}())
    _PADDED_DIST_WS_CACHE[key] = ws
    return ws
end

# Memoized NoPermutation Pencil for (size, decomp). Identity-stable: the same
# (size, decomp) returns the SAME object, so _padded_dist_buf! hits the `===` path.
@inline function _padded_dist_pencil!(ws::PaddedDistDealiasWorkspace, sz, dd)
    key = (Tuple(sz), Tuple(dd))
    p = get(ws.pencils, key, nothing)
    p === nothing || return p
    p = PencilArrays.Pencil(ws.topo, Tuple(sz), Tuple(dd); permute=PencilArrays.NoPermutation())
    ws.pencils[key] = p
    return p
end

# Two pencils may share a scratch buffer iff their geometry matches — same global
# size + decomposition + permutation fully determines the local memory layout and
# transpose! validity. `===` (memoized mkpen pencils) is the fast path; the
# structural branch runs only for grid pencils pencil(gd)/pencil(rg), whose
# geometry is fixed per problem (so it is always true on reuse). Permutation lives
# in the Pencil type parameters, so typeof(...) === typeof(...) compares it exactly.
@inline function _pen_compatible(a, b)
    a === b && return true
    return PencilArrays.size_global(a) == PencilArrays.size_global(b) &&
           Tuple(PencilArrays.decomposition(a)) == Tuple(PencilArrays.decomposition(b)) &&
           typeof(PencilArrays.permutation(a)) === typeof(PencilArrays.permutation(b))
end

# Reuse only an unleased buffer with the exact geometry; active operands never alias.
function _padded_dist_buf!(ws::PaddedDistDealiasWorkspace{CT}, pen) where {CT}
    for i in eachindex(ws.buffers)
        ws.in_use[i] && continue
        b = ws.buffers[i]::PencilArrays.PencilArray
        if _pen_compatible(PencilArrays.pencil(b), pen)
            ws.in_use[i] = true
            return b
        end
    end
    nb = PencilArrays.PencilArray{CT}(undef, pen)
    push!(ws.buffers, nb)
    push!(ws.in_use, true)
    return nb
end

function _release_padded_dist_buf!(ws::PaddedDistDealiasWorkspace, buffer)
    for i in eachindex(ws.buffers)
        if ws.buffers[i] === buffer
            ws.in_use[i] = false
            return nothing
        end
    end
    throw(ArgumentError("Cannot release a buffer outside the padded workspace"))
end

# PencilArrays.transpose! completes the destination before returning. Only count
# decomposition changes: a local permutation/copy does not communicate with ranks.
function _padded_dist_transpose!(ws::PaddedDistDealiasWorkspace, dest, src)
    dp, sp = PencilArrays.pencil(dest), PencilArrays.pencil(src)
    if PencilArrays.decomposition(dp) != PencilArrays.decomposition(sp)
        ws.transpose_count += 1
        ws.transpose_elements += prod(PencilArrays.size_global(sp))
    end
    PencilArrays.transpose!(dest, src)
    _release_padded_dist_buf!(ws, src)
    return dest
end

# In-place single-local-axis FFT plans, cached by (local size, axis). CPU: FFTW
# MEASURE on a throwaway (MEASURE scribbles its planning input) + UNALIGNED so one
# plan serves every same-size buffer (the shape recurs across up1/up2/res). GPU /
# generic: AbstractFFTs.plan_fft! dispatches to CUFFT. This is the same concrete-
# CPU-branch / generic-fallback split used by _get_padded_workspace!.
function _padded_dist_fwd_plan!(ws::PaddedDistDealiasWorkspace, arr, a::Int)
    key = (size(arr), a)
    p = get(ws.fwd_plans, key, nothing)
    p === nothing || return p
    p = arr isa Array ? FFTW.plan_fft!(similar(arr), a; flags=FFTW.MEASURE | FFTW.UNALIGNED) :
                        plan_fft!(similar(arr), a)
    ws.fwd_plans[key] = p
    return p
end

function _padded_dist_inv_plan!(ws::PaddedDistDealiasWorkspace, arr, a::Int)
    key = (size(arr), a)
    p = get(ws.inv_plans, key, nothing)
    p === nothing || return p
    p = arr isa Array ? FFTW.plan_ifft!(similar(arr), a; flags=FFTW.MEASURE | FFTW.UNALIGNED) :
                        plan_ifft!(similar(arr), a)
    ws.inv_plans[key] = p
    return p
end

# Resize one local pencil axis at a time. Fourier FFT normalization is applied
# once after truncation. The Chebyshev DCTs carry their own normalization.
function evaluate_padded_multiply_distributed(field1::ScalarField, field2::ScalarField,
                                              evaluator::NonlinearEvaluator; result_layout::Symbol=:g)
    bases = field1.bases
    D = length(bases)
    (D == 2 || D == 3) || return nothing
    isfourier(b) = isa(b, Union{RealFourier, ComplexFourier})
    fourier_axes = Tuple(d for d in 1:D if isfourier(bases[d]))
    isempty(fourier_axes) && return nothing
    dist = field1.dist
    factor = evaluator.dealiasing_factor
    template = _product_template(field1, field2)
    T = real(template.dtype)
    CT = Complex{T}
    N = ntuple(d -> bases[d].meta.size, D)
    Mp = ntuple(d -> _padded_product_nodes(bases[d], N[d], factor), D)
    padded_axes = Tuple(d for d in 1:D if Mp[d] != N[d])
    isempty(padded_axes) && return nothing
    ensure_layout!(field1, :g); ensure_layout!(field2, :g)
    gd1 = get_grid_data(field1); gd2 = get_grid_data(field2)
    (gd1 isa PencilArrays.PencilArray && gd2 isa PencilArrays.PencilArray) || return nothing
    pen = PencilArrays.pencil(gd1)
    topo = PencilArrays.topology(pen)
    decomp0 = Tuple(PencilArrays.decomposition(pen))
    # At most D-1 decomposed axes, i.e. at least one axis is local. A D-1
    # decomposition (2D on a 1-D mesh, 3D on a 2-D mesh) leaves exactly one; a
    # COARSER process mesh — a 3D field on a 1-D slab mesh, the common
    # `mesh=(nprocs,)` case — leaves two, which the sweep below handles as a set.
    # Requiring equality here silently routed every 3D slab-decomposed run to the
    # 2/3-rule fallback, so it dropped resolved modes serial and a 2-D mesh keep.
    length(decomp0) <= D - 1 || return nothing
    all(d -> isfourier(bases[d]), decomp0) || return nothing   # only pad/transpose decomposed FOURIER axes
    ws = _get_padded_dist_workspace!(dist, topo, bases, CT, N, Mp, decomp0, fourier_axes)
    fill!(ws.in_use, false)
    ws.transpose_count = 0
    ws.transpose_elements = 0
    mkpen(sz, dd) = _padded_dist_pencil!(ws, sz, dd)

    function tolog(gd, imag_grid=nothing, exponents=(0, 0))
        gc = _padded_dist_buf!(ws, PencilArrays.pencil(gd))
        if imag_grid === nothing
            parent(gc) .= parent(gd)
        else
            e1, e2 = exponents
            parent(gc) .= complex.(ldexp.(parent(gd), -e1), ldexp.(parent(imag_grid), -e2))
        end
        logical_pen = mkpen(N, decomp0)
        _pen_compatible(PencilArrays.pencil(gc), logical_pen) && return gc
        a = _padded_dist_buf!(ws, logical_pen)
        _padded_dist_transpose!(ws, a, gc)
    end
    # Bring axis `a` into a LOCAL slot. "Local" is a SET, not a single axis: a
    # D-1 decomposition leaves one local axis, a coarser mesh leaves several, and
    # `a` may already be local — then no transpose is needed. Otherwise swap `a`
    # with one currently-local axis: a one-decomposed-dim change, which
    # PencilArrays.transpose! allows. `prefer` chooses WHICH local axis takes its
    # place (0 = any); the final alignment loop needs that to land a specific axis
    # in a specific decomposition slot.
    makelocal(cur, csz, dec, a, prefer::Int) = !(a in dec) ? (cur, dec) : begin
        locs = setdiff(1:D, dec)
        l = (prefer != 0 && prefer in locs) ? prefer : first(locs)
        ndec = [d == a ? l : d for d in dec]
        nxt = _padded_dist_buf!(ws, mkpen(csz, ndec))
        _padded_dist_transpose!(ws, nxt, cur)
        (nxt, ndec)
    end
    function resize_local!(pout, pin, axis, field)
        input, output = parent(pin), parent(pout)
        if bases[axis] isa ChebyshevT
            return _resize_chebyshev_product_axis!(output, input, axis, field)
        end
        _padded_dist_fwd_plan!(ws, input, axis) * input
        cosine_dims = field.dtype <: Real ? (axis,) : ()
        if size(output, axis) > size(input, axis)
            _pad_spectral!(output, input, size(input), size(output), [axis], cosine_dims)
        else
            _truncate_spectral!(output, input, size(output), size(input), [axis], cosine_dims)
        end
        _padded_dist_inv_plan!(ws, output, axis) * output
    end
    sweep(start, sdec, ssz, tgt, axes_order, field) = begin
        cur = start; dec = collect(sdec); csz = collect(ssz)
        for a in axes_order
            cur, dec = makelocal(cur, csz, dec, a, 0)
            csz[a] = tgt[a]
            nxt = _padded_dist_buf!(ws, mkpen(csz, dec))
            resize_local!(nxt, cur, a, field)
            _release_padded_dist_buf!(ws, cur)
            cur = nxt
        end
        cur
    end

    # Interpolation with cosine Nyquist bins maps real grids to real grids.
    # Linearity lets one complex pencil carry both real operands through every
    # FFT, DCT and transpose. Their product then occupies the same buffer.
    product_exponent = 0
    if field1.dtype <: Real && field2.dtype <: Real && _pen_compatible(pen, PencilArrays.pencil(gd2))
        # Normalize each operand independently so a weak field survives the
        # packed FFT. Powers of two rescale exactly, including subnormal inputs.
        peaks = [maximum(abs, parent(g); init=zero(T)) for g in (gd1, gd2)]
        MPI.Allreduce!(peaks, MPI.MAX, dist.comm)
        if any(iszero, peaks) && all(isfinite, peaks)
            result = _checkout_nl_result!(evaluator, template)
            fill!(parent(grid_data!(result)), zero(T))
            ensure_layout!(result, result_layout)
            return result
        end
        exponents = (last(frexp(peaks[1])), last(frexp(peaks[2])))
        product_exponent = sum(exponents)
        up1 = sweep(tolog(gd1, gd2, exponents), decomp0, N, Mp, padded_axes, field1)
        packed = parent(up1)
        @. packed = real(packed) * imag(packed)
    else
        up1 = sweep(tolog(gd1), decomp0, N, Mp, padded_axes, field1)
        up2 = sweep(tolog(gd2), decomp0, N, Mp, padded_axes, field2)
        parent(up1) .*= parent(up2)
        _release_padded_dist_buf!(ws, up2)
    end
    pdec = Tuple(PencilArrays.decomposition(PencilArrays.pencil(up1)))
    # Separable truncations commute. Undo the padding order so the first axis is
    # already local, and shrink each axis BEFORE communicating the next pencil.
    res = sweep(up1, pdec, Mp, N, reverse(padded_axes), template)
    parent(res) .*= prod(T(Mp[d]) / T(N[d]) for d in fourier_axes)

    result = _checkout_nl_result!(evaluator, template)
    rg = grid_data!(result)
    # Align `res` (NoPermutation) to the result grid pencil's EXACT ordered
    # decomposition so the final transpose is perm-only. Matching just the local
    # axis is NOT enough: res and rg can share the decomposed-axis SET but in a
    # different ORDER (e.g. res (2,3) vs rg (3,2) for a 3D Cheb-Fourier-Fourier
    # field), and PencilArrays.transpose! forbids a >1-slot decomp difference. Walk
    # the decomp tuple slot by slot, each fix being one valid single-swap.
    rgdec = collect(PencilArrays.decomposition(PencilArrays.pencil(rg)))
    rdec = collect(PencilArrays.decomposition(PencilArrays.pencil(res)))
    for i in eachindex(rgdec)
        rdec[i] == rgdec[i] && continue
        res, rdec = makelocal(res, collect(N), rdec, rgdec[i], 0)         # free the target axis
        res, rdec = makelocal(res, collect(N), rdec, rdec[i], rgdec[i])   # swap it into slot i
    end
    rc = if _pen_compatible(PencilArrays.pencil(res), PencilArrays.pencil(rg))
        res
    else
        _padded_dist_transpose!(ws, _padded_dist_buf!(ws, PencilArrays.pencil(rg)), res)
    end
    if result.dtype <: Complex
        parent(rg) .= parent(rc)
    else
        parent(rg) .= ldexp.(real.(parent(rc)), product_exponent)
    end
    _release_padded_dist_buf!(ws, rc)
    ensure_layout!(result, result_layout)
    return result
end

"""2D transform-based multiplication using PencilFFTs"""
function evaluate_2d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)

    # Create result field (static name avoids string allocation per call)
    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # Try to find matching transform configuration
    shape_key = shape  # Use tuple directly as key (no string allocation)

    if haskey(evaluator.pencil_transforms, shape_key)
        # Use precomputed transform config. PencilFFT configs keep the old
        # optimized path; FFTW configs fall back to direct multiply plus the
        # standard spectral truncation pass.
        transform_info = evaluator.pencil_transforms[shape_key]
        if transform_info isa PencilTransformConfig
            data1 = get_pencil_compatible_data(field1, transform_info.config)
            data2 = get_pencil_compatible_data(field2, transform_info.config)
            result_data = data1 .* data2

            if evaluator.dealiasing_factor > 1.0
                result_data = apply_2d_dealiasing(result_data, transform_info, evaluator.dealiasing_factor)
            end

            set_pencil_compatible_data!(result, result_data, transform_info.config)
        else
            result["g"] .= field1["g"] .* field2["g"]

            if evaluator.dealiasing_factor > 1.0
                apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
            end
        end

    else
        # Fallback to direct multiplication without PencilFFT optimization
        @debug "No precomputed transforms for shape $shape, using fallback"
        result["g"] .= field1["g"] .* field2["g"]

        # Apply basic dealiasing if possible
        if evaluator.dealiasing_factor > 1.0
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end

    return result
end

"""3D transform-based multiplication using 3D PencilFFTs"""
function evaluate_3d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)

    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # For 3D, we need more sophisticated pencil management
    # This would involve 3D pencil decomposition across all three spatial dimensions

    if length(evaluator.dist.mesh) >= 3
        # Use 3D PencilFFT approach
        @warn "3D dealiasing via padding is not yet implemented; using undealiased multiplication" maxlog=1

        # Direct multiplication for now - would implement full 3D PencilFFT logic
        result["g"] .= field1["g"] .* field2["g"]

        # Apply 3D dealiasing
        if evaluator.dealiasing_factor > 1.0
            apply_3d_dealiasing!(result, evaluator.dealiasing_factor)
        end

    else
        # Fallback for insufficient parallelization
        @warn "3D nonlinear multiplication falling back to undealiased pointwise multiply" maxlog=1
        result["g"] .= field1["g"] .* field2["g"]

        if evaluator.dealiasing_factor > 1.0
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end

    return result
end

"""Fallback multiplication for unsupported dimensions"""
function evaluate_fallback_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)

    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # Simple pointwise multiplication
    result["g"] .= field1["g"] .* field2["g"]

    # Apply dealiasing if requested
    if evaluator.dealiasing_factor > 1.0
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end

    return result
end

# Integration with existing operator evaluation
evaluate_operator(op::AdvectionOperator) = evaluate_nonlinear_term(op)
evaluate_operator(op::NonlinearAdvectionOperator) = evaluate_nonlinear_term(op)
evaluate_operator(op::ConvectiveOperator) = evaluate_convective_operator(op)
function evaluate_operator(op::NonlinearOperator)
    throw(ArgumentError("Nonlinear operator evaluation not implemented for $(typeof(op))"))
end

"""Get the cached NonlinearEvaluator for a distributor, creating one if needed."""
function _get_evaluator(dist::Distributor)
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    return dist.nonlinear_evaluator
end

"""Evaluate general convective operator"""
function evaluate_convective_operator(op::ConvectiveOperator)

    field1, field2 = op.field1, op.field2

    if op.operation == :multiply
        # Simple multiplication
        if isa(field1, ScalarField) && isa(field2, ScalarField)
            evaluator = _get_evaluator(field1.dist)
            return evaluate_transform_multiply(field1, field2, evaluator)
        else
            throw(ArgumentError("Multiplication not implemented for field types $(typeof(field1)), $(typeof(field2))"))
        end

    elseif op.operation == :dot_product
        # Dot product between vectors
        if isa(field1, VectorField) && isa(field2, VectorField)
            return evaluate_vector_dot_product(field1, field2)
        else
            throw(ArgumentError("Dot product requires two vector fields"))
        end

    elseif op.operation == :cross_product
        # Cross product between vectors
        if isa(field1, VectorField) && isa(field2, VectorField)
            return evaluate_vector_cross_product(field1, field2)
        else
            throw(ArgumentError("Cross product requires two vector fields"))
        end

    else
        throw(ArgumentError("Unknown convective operation: $(op.operation)"))
    end
end

"""Evaluate v1·v2 dot product"""
function evaluate_vector_dot_product(v1::VectorField, v2::VectorField)

    if length(v1.components) != length(v2.components)
        throw(ArgumentError("Vector fields must have same number of components"))
    end

    evaluator = _get_evaluator(v1.dist)
    n = length(v1.components)
    c1 = v1.components[1]

    # Distributed pure-Fourier: sum the component products in COEFFICIENT space so
    # the `n` per-product backward transposes collapse into ONE (performed by the
    # caller's `ensure_layout!(result, layout)`). Valid because the inverse
    # transform is linear: Σ backward(pᵢ) == backward(Σ pᵢ). Each product is
    # returned truncated-in-coeff via `result_layout=:c` (skips its own backward).
    if n >= 2 && c1.dist.use_pencil_arrays && c1.dist.size > 1 &&
       all(b -> isa(b, Union{RealFourier, ComplexFourier}), c1.bases)
        # The first product becomes the accumulator and IS returned, so it must be
        # owned; the rest are added into it and borrow.
        result = evaluate_transform_multiply(v1.components[1], v2.components[1], evaluator; result_layout=:c)
        rc = coeff_data!(result)
        for i in 2:n
            product = evaluate_transform_multiply(v1.components[i], v2.components[i], evaluator; result_layout=:c, own=false)
            pc = coeff_data!(product)
            if isa(rc, PencilArrays.PencilArray) && isa(pc, PencilArrays.PencilArray)
                parent(rc) .+= parent(pc)
            else
                rc .+= pc
            end
        end
        return result  # coeff; caller's ensure_layout! does the single backward
    end

    # Serial / non-Fourier: sum products in grid (original path).
    # Owned: with n == 1 the loop never runs and this value is returned directly.
    result = evaluate_transform_multiply(v1.components[1], v2.components[1], evaluator)
    for i in 2:n
        product = evaluate_transform_multiply(v1.components[i], v2.components[i], evaluator; own=false)
        result = result + product
    end

    return result
end

"""Evaluate v1×v2 cross product (3D only)"""
function evaluate_vector_cross_product(v1::VectorField, v2::VectorField)

    if length(v1.components) != 3 || length(v2.components) != 3
        throw(ArgumentError("Cross product requires 3D vector fields"))
    end
    if v1.coordsys !== v2.coordsys
        throw(ArgumentError("Cross product requires identical coordinate systems"))
    end
    if v1.bases != v2.bases
        throw(ArgumentError("Cannot compute cross product of VectorFields with different bases"))
    end

    evaluator = _get_evaluator(v1.dist)

    handedness = (hasproperty(v1.coordsys, :right_handed) && v1.coordsys.right_handed === false) ? -1 : 1

    # Cross product: (a×b)_x = a_y*b_z - a_z*b_y
    #                (a×b)_y = a_z*b_x - a_x*b_z
    #                (a×b)_z = a_x*b_y - a_y*b_x

    result = VectorField(v1.dist, v1.coordsys, "cross_$(v1.name)_$(v2.name)", v1.bases, v1.dtype)

    # x-component
    term1 = evaluate_transform_multiply(v1.components[2], v2.components[3], evaluator; own=false)
    term2 = evaluate_transform_multiply(v1.components[3], v2.components[2], evaluator; own=false)
    result.components[1] = (term1 - term2) * handedness

    # y-component
    term1 = evaluate_transform_multiply(v1.components[3], v2.components[1], evaluator; own=false)
    term2 = evaluate_transform_multiply(v1.components[1], v2.components[3], evaluator; own=false)
    result.components[2] = (term1 - term2) * handedness

    # z-component
    term1 = evaluate_transform_multiply(v1.components[1], v2.components[2], evaluator; own=false)
    term2 = evaluate_transform_multiply(v1.components[2], v2.components[1], evaluator; own=false)
    result.components[3] = (term1 - term2) * handedness

    return result
end

# Helper to evaluate any operand (Future, Operator, or Field)
function _evaluate_any_operand(arg, layout::Symbol)
    if isa(arg, Future)
        return evaluate(arg; force=true)
    elseif isa(arg, Operator)
        return evaluate(arg, layout)
    else
        return arg
    end
end

# Evaluate methods for DotProduct and CrossProduct from arithmetic.jl
"""Evaluate DotProduct of two VectorFields"""
function evaluate(op::DotProduct, layout::Symbol=:g)
    args = future_args(op)
    if length(args) != 2
        throw(ArgumentError("DotProduct expects exactly two operands"))
    end

    # Evaluate operands first - they might be operators that return VectorFields
    v1 = _evaluate_any_operand(args[1], layout)
    v2 = _evaluate_any_operand(args[2], layout)

    if !isa(v1, VectorField) || !isa(v2, VectorField)
        throw(ArgumentError("DotProduct requires two VectorField operands, got $(typeof(v1)) and $(typeof(v2))"))
    end

    result = evaluate_vector_dot_product(v1, v2)
    ensure_layout!(result, layout)
    return result
end

"""Evaluate CrossProduct of two VectorFields"""
function evaluate(op::CrossProduct, layout::Symbol=:g)
    args = future_args(op)
    if length(args) != 2
        throw(ArgumentError("CrossProduct expects exactly two operands"))
    end

    # Evaluate operands first - they might be operators that return VectorFields
    v1 = _evaluate_any_operand(args[1], layout)
    v2 = _evaluate_any_operand(args[2], layout)

    if !isa(v1, VectorField) || !isa(v2, VectorField)
        throw(ArgumentError("CrossProduct requires two VectorField operands, got $(typeof(v1)) and $(typeof(v2))"))
    end

    result = evaluate_vector_cross_product(v1, v2)
    for comp in result.components
        ensure_layout!(comp, layout)
    end
    return result
end

# Convenience constructors
"""Create advection operator u·∇φ"""
function advection(u::VectorField, φ::ScalarField)
    return AdvectionOperator(u, φ)
end

"""Create nonlinear momentum operator (u·∇)u"""
function nonlinear_momentum(u::VectorField)
    return NonlinearAdvectionOperator(u)
end

"""Create convective operator"""
function convection(f1, f2, op::Symbol)
    return ConvectiveOperator(f1, f2, op)
end

"""Log nonlinear evaluation performance statistics"""
function log_nonlinear_performance(stats::NonlinearPerformanceStats)

    if MPI.Initialized()
        mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if mpi_rank == 0
            @info "Nonlinear evaluation performance:"
            @info "  Total evaluations: $(stats.total_evaluations)"
            @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
            @info "  Average time per evaluation: $(round(stats.total_time/max(stats.total_evaluations, 1), digits=6)) seconds"
            @info "  Dealiasing time: $(round(stats.dealiasing_time, digits=3)) seconds ($(round(100*stats.dealiasing_time/max(stats.total_time, 1e-10), digits=1))%)"
            @info "  Transform time: $(round(stats.transform_time, digits=3)) seconds ($(round(100*stats.transform_time/max(stats.total_time, 1e-10), digits=1))%)"
        end
    else
        @info "Nonlinear evaluation performance:"
        @info "  Total evaluations: $(stats.total_evaluations)"
        @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
        @info "  Average time per evaluation: $(round(stats.total_time/max(stats.total_evaluations, 1), digits=6)) seconds"
    end
end
