# ============================================================================
# GPU Transform Implementations
# ============================================================================

# Field data accessors: imported from Tarang (get_grid_data, get_coeff_data,
# set_grid_data!, set_coeff_data!) — use field.buffers.grid/coeff internally.

# ============================================================================
# Distributed GPU Detection
# ============================================================================

"""
    is_distributed_gpu(arch, nprocs::Int)

Whether a GPU field spans several ranks, so a direct transform must take the
distributed multi-GPU path (a rank-local transform of the slab would be wrong).

Deliberately independent of NCCL. It used to require `nccl_available()`, i.e.
that `init_nccl!` had been called — which nothing does — so the distributed path
was unreachable. NCCL.jl is now resolved on demand when the plan is built, and a
missing NCCL raises a clear error on every rank (`_distributed_dct_preflight`)
instead of silently selecting another path.
"""
function is_distributed_gpu(arch::GPU, nprocs::Int)
    return nprocs > 1
end

is_distributed_gpu(arch::CPU, nprocs::Int) = false
is_distributed_gpu(arch, nprocs::Int) = false

"""
    needs_distributed_dct(field)

Check if the field requires distributed DCT (has Chebyshev basis in any dimension).
Returns true only for 3D fields with at least one Chebyshev basis.
"""
function needs_distributed_dct(field)
    bases = field.bases
    if length(bases) != 3
        return false  # Only support 3D distributed DCT for now
    end
    for b in bases
        if b isa ChebyshevT
            return true
        end
    end
    return false
end

# ============================================================================
# Distributed DCT Plan Cache
# ============================================================================

# Global cache for distributed DCT plans (thread-safe).
# Value type is Any: DistributedDCTPlan is defined in dct_distributed.jl, which
# is included AFTER this file — naming it here is an UndefVarError at load time.
const DISTRIBUTED_DCT_PLAN_CACHE = Dict{Tuple, Any}()
const DISTRIBUTED_DCT_PLAN_LOCK = ReentrantLock()
# Cache keys in the order their plans were built; teardown follows it (see
# `_distributed_dct_teardown_order`). Guarded by DISTRIBUTED_DCT_PLAN_LOCK.
const DISTRIBUTED_DCT_PLAN_BUILD_ORDER = Tuple[]

"""
    get_or_create_distributed_dct_plan(field)

Get or create a cached distributed DCT plan for the given field.
Thread-safe caching by field and execution context (global shape, process mesh,
element type, dtype-resolved axis kinds, communicator identity, and CUDA device).
Building a plan is collective over `field.dist.comm` and is preceded by
`_distributed_dct_preflight`, so every rank must request it in the same order —
as every rank does when it transforms the same field.
"""
function get_or_create_distributed_dct_plan(field)
    dist = field.dist

    # Create cache key from field properties. axis_kind is part of the key because
    # the plan's per-axis dispatch and coefficient shape depend on it. It is
    # resolved with the FIELD dtype: a RealFourier dim 1 is rfft-halved only for
    # real data, so a real and a complex field on the same bases need different
    # plans (the dtype-blind `axis_kinds(bases)` gave both the half-spectrum one).
    global_shape_tuple = Tuple(Tarang.global_shape(field.domain))
    proc_grid = _distributed_dct_proc_grid(dist)
    T = real(eltype(get_grid_data(field)))
    axis_kind = Tarang._distributed_gpu_dct_axis_kinds(field.bases, field.dtype)
    comm_token = _distributed_dct_comm_token(dist.comm)
    device_id = _dct_cache_device_id(dist.architecture)

    key = _distributed_dct_plan_cache_key(
        global_shape_tuple, proc_grid, T, axis_kind, comm_token, device_id)

    lock(DISTRIBUTED_DCT_PLAN_LOCK) do
        if haskey(DISTRIBUTED_DCT_PLAN_CACHE, key)
            return DISTRIBUTED_DCT_PLAN_CACHE[key]
        end

        # Same guard as transpose_workspace!: building a plan splits communicators,
        # which must never happen after close(dist) released them.
        dist.closed && throw(ArgumentError(
            "cannot build a distributed DCT plan from a closed Distributor"))

        # Every rank agrees the plan can run (NCCL, buffer shapes) before any
        # rank starts the collective construction below.
        _distributed_dct_preflight(field, global_shape_tuple, proc_grid, axis_kind)

        # Create pencil decomposition from field's distributor
        pencil = get_or_create_pencil(dist, global_shape_tuple)

        # Create basis-aware plan from the same kinds the key holds.
        plan = _build_distributed_dct_plan(pencil, axis_kind, T)

        DISTRIBUTED_DCT_PLAN_CACHE[key] = plan
        push!(DISTRIBUTED_DCT_PLAN_BUILD_ORDER, key)
        return plan
    end
end

"""
    _distributed_dct_preflight(field, global_shape, proc_grid, axis_kind)

Collective check run by EVERY rank of `field.dist.comm` before a distributed DCT
plan is built (which is itself collective: the pencil splits communicators and
the transpose buffer creates NCCL communicators). It confirms that

- NCCL.jl is usable. It is resolved on demand from the user's session, so
  `init_nccl!` need not have run; the pencil transposes create their own NCCL
  sub-communicators and have no MPI fallback.
- the field's buffers are exactly what the plan reads and writes on this rank:
  the column-major pencil blocks of the global grid and coefficient shapes —
  Tarang's `get_local_array_size` convention — with the field dtype on the grid
  and `Complex{T}` coefficients. The wrappers below refuse to substitute buffers.

The per-rank verdicts are combined with one `Allreduce`, so a problem on any rank
makes every rank raise this error before a single transpose is posted, instead
of one rank failing while its peers block inside NCCL.
"""
function _distributed_dct_preflight(field, gshape::NTuple{3,Int}, proc_grid::NTuple{2,Int},
                                    axis_kind::NTuple{3,Symbol})
    dist = field.dist
    gc = column_major_grid_coords(dist.rank, proc_grid)
    coeff_Nx = axis_kind[1] === :real_fourier ? Tarang.rfft_len(gshape[1]) : gshape[1]
    grid_z = compute_pencil_shapes(gshape, proc_grid, gc)[3]
    coeff_z = compute_pencil_shapes((coeff_Nx, gshape[2], gshape[3]), proc_grid, gc)[3]
    grid, coeff = get_grid_data(field), get_coeff_data(field)
    CT = Complex{real(field.dtype)}
    problem = if !(Tarang.nccl_available() || Tarang._try_load_nccl())
        "NCCL.jl is not loaded on rank $(dist.rank). The distributed DCT-I transposes run " *
        "over NCCL and have no MPI fallback: run `using NCCL` on every rank before the " *
        "first transform (init_nccl! is not required)"
    elseif !(grid isa CuArray{field.dtype,3}) || size(grid) != grid_z
        "rank $(dist.rank) stores a $(typeof(grid)) grid block of size $(size(grid)), but " *
        "the pencil decomposition (process grid $proc_grid) reads a " *
        "CuArray{$(field.dtype), 3} of size $grid_z (scaled grids are not supported here)"
    elseif !(coeff isa CuArray{CT,3}) || size(coeff) != coeff_z
        "rank $(dist.rank) stores a $(typeof(coeff)) coefficient block of size " *
        "$(size(coeff)), but the plan writes a CuArray{$CT, 3} of size $coeff_z"
    else
        nothing
    end
    failed = MPI.Allreduce(problem === nothing ? 0 : 1, MPI.SUM, dist.comm)
    failed == 0 && return nothing
    error("Distributed multi-GPU DCT-I transform of field $(repr(field.name)) cannot run: " *
          something(problem, "the pre-flight check failed on $failed other rank(s)") *
          ". Every rank raised this error before starting a collective transpose.")
end

"""
    _distributed_dct_comm_token(comm)

Return a process-local identity token for an MPI communicator object. MPI
communicators with equal sizes (or even congruent groups) still have independent
collective contexts and must never share a cached distributed plan.
"""
_distributed_dct_comm_token(comm::MPI.Comm) = objectid(comm)

"""
    _distributed_dct_plan_cache_key(global_shape, proc_grid, T, axis_kind,
                                    comm_token, device_id)

Build the complete cache key for a distributed DCT plan. Kept as a pure helper
so context separation can be regression-tested without requiring a GPU/NCCL run.
"""
_distributed_dct_plan_cache_key(global_shape, proc_grid, T, axis_kind,
                                comm_token, device_id) =
    (global_shape, proc_grid, T, axis_kind, comm_token, device_id)

"""
    _compute_proc_grid(nprocs::Int)

Compute a 2D process grid (P1, P2) for the given number of processes.
Prefers a square-ish distribution.
"""
function _compute_proc_grid(nprocs::Int)
    P1 = isqrt(nprocs)
    while nprocs % P1 != 0
        P1 -= 1
    end
    P2 = nprocs ÷ P1
    return (P1, P2)
end

"""
    _distributed_dct_proc_grid(dist)

Return the two-dimensional process grid `(P1, P2)` matching how `dist` decomposes
a 3D field (`decomposed_axes`, `use_pencil_arrays=false`): a 2-D mesh splits
dims 1 and 2, and a 1-D mesh `(P,)` splits dim 1 only — i.e. `(P, 1)`, not the
square-ish heuristic, whose blocks would disagree with every field buffer. The
heuristic remains only for a distributor without a mesh.
"""
function _distributed_dct_proc_grid(dist)
    mesh = dist.mesh
    if mesh isa Tuple && length(mesh) == 2 && prod(mesh) == dist.size
        return (mesh[1], mesh[2])
    elseif mesh isa Tuple && length(mesh) == 1 && mesh[1] == dist.size
        return (mesh[1], 1)
    end
    return _compute_proc_grid(dist.size)
end

"""
    get_or_create_pencil(dist, global_shape::NTuple{3, Int})

Get or create PencilDecomposition from a Distributor.
"""
function get_or_create_pencil(dist, global_shape::NTuple{3, Int})
    nprocs = dist.size
    rank = dist.rank
    comm = dist.comm

    # Honor a user-supplied process mesh: Distributor stores it as `dist.mesh`
    # (a 2D tuple for 3D domains, e.g. (4, 2)). Deriving the grid from
    # _compute_proc_grid alone ignored it — e.g. mesh (4, 2) silently became
    # pencil grid (2, 4), so the pencil's local shapes disagreed with every
    # mesh-derived buffer and the shape asserts rejected the run. Fall back to
    # the square-ish heuristic only when no usable 2D mesh is present.
    proc_grid = _distributed_dct_proc_grid(dist)

    # Align the pencil's block ownership with the distributor's column-major
    # field-buffer convention (design decision #5). Only matters when P1>1 AND
    # P2>1 (np≥4); coincides with row-major for 1×P / P×1 grids. Needs np≥4 GPU
    # validation.
    gc = column_major_grid_coords(rank, proc_grid)
    return PencilDecomposition(global_shape, proc_grid, rank, comm; grid_coords=gc)
end

"""
    _distributed_dct_teardown_order(build_order, cache; comm_token=nothing) -> Vector

Keys of the cached plans to finalize, in the order the plans were BUILT,
restricted to the communicator `comm_token` when given (key slot 5).

Finalizing a plan destroys NCCL communicators and frees MPI communicators, both
collective, so every rank must walk its plans in the same order. Build order is
rank-invariant (each build is collective over its communicator); iterating the
cache `Dict` is not, because its keys hold process-local communicator tokens
(`objectid`) and device ids, so their hash order differs from rank to rank.
"""
function _distributed_dct_teardown_order(build_order, cache; comm_token=nothing)
    return Tuple[key for key in build_order
                 if haskey(cache, key) && (comm_token === nothing || key[5] == comm_token)]
end

"""
    _finalize_distributed_dct_plans!(keys)

Finalize the cached plans under `keys` in that order, dropping each from the
cache. Every plan is finalized even if an earlier one throws — a rank that
stopped early would strand its peers in the remaining collective teardowns —
and the first error is rethrown afterwards. Caller holds the plan lock.
"""
function _finalize_distributed_dct_plans!(keys)
    first_error = nothing
    for key in keys
        plan = pop!(DISTRIBUTED_DCT_PLAN_CACHE, key)
        filter!(!=(key), DISTRIBUTED_DCT_PLAN_BUILD_ORDER)
        try
            finalize_distributed_dct_plan!(plan)
        catch err
            first_error === nothing && (first_error = err)
        end
    end
    first_error === nothing || throw(first_error)
    return nothing
end

"""
    clear_distributed_dct_plan_cache!()

Finalize and clear all cached distributed DCT plans (thread-safe). Collective:
every rank must call it at the same point; plans are torn down in build order.
"""
function clear_distributed_dct_plan_cache!()
    lock(DISTRIBUTED_DCT_PLAN_LOCK) do
        _finalize_distributed_dct_plans!(_distributed_dct_teardown_order(
            DISTRIBUTED_DCT_PLAN_BUILD_ORDER, DISTRIBUTED_DCT_PLAN_CACHE))
        empty!(DISTRIBUTED_DCT_PLAN_CACHE)
        empty!(DISTRIBUTED_DCT_PLAN_BUILD_ORDER)
    end
end

"""Finalize and drop every cached distributed DCT plan built on `dist.comm`, in build order."""
function Tarang._close_backend_plan_caches!(dist::Tarang.Distributor, ::GPU{CuDevice})
    token = _distributed_dct_comm_token(dist.comm)
    lock(DISTRIBUTED_DCT_PLAN_LOCK) do
        _finalize_distributed_dct_plans!(_distributed_dct_teardown_order(
            DISTRIBUTED_DCT_PLAN_BUILD_ORDER, DISTRIBUTED_DCT_PLAN_CACHE; comm_token=token))
    end
    return nothing
end

# ============================================================================
# Distributed GPU Transform Functions
# ============================================================================

"""
    distributed_gpu_forward_transform!(field::ScalarField)

Forward transform using distributed GPU DCT for Chebyshev fields.
Transforms from grid space (Z-pencil layout) into the field's own coefficient
buffer, which holds this rank's block of Tarang's coefficient array (Z-local).
"""
function distributed_gpu_forward_transform!(field::ScalarField)
    # Get or create distributed DCT plan
    plan = get_or_create_distributed_dct_plan(field)

    # Get local grid data
    data = get_grid_data(field)

    # Ensure we're in Z-pencil orientation (grid space)
    set_orientation!(plan.pencil, :z_pencil)

    # The driver lands coeffs **Z-local on the coeff pencil** (complex,
    # half-spectrum on dim 1 for an rfft-halved dim-1 axis) — NOT X-pencil; see
    # distributed_forward_dct!. Write into the field's buffers only. Swapping in a
    # freshly allocated array when they do not fit (as this used to) silently
    # changed the field's coefficient layout under every other consumer; the
    # plan's pre-flight verified these shapes collectively on every rank.
    T = real(eltype(data))
    shape = plan.coeff_pencil.z_pencil_shape
    coeffs = get_coeff_data(field)
    if size(data) != plan.pencil.z_pencil_shape || !(coeffs isa CuArray{Complex{T},3}) ||
       size(coeffs) != shape
        error("distributed multi-GPU forward transform of field $(repr(field.name)): rank " *
              "$(field.dist.rank) holds a $(size(data)) grid block and a $(typeof(coeffs)) " *
              "coefficient block of size $(size(coeffs)), but the plan reads a " *
              "$(plan.pencil.z_pencil_shape) grid block and writes Complex{$T} coefficients " *
              "of size $shape. Field buffers must keep the shapes they were allocated with.")
    end

    # Perform distributed transform (resets pencil orientations internally).
    distributed_forward_dct!(coeffs, data, plan)

    return true
end

"""
    distributed_gpu_backward_transform!(field::ScalarField)

Backward transform using distributed GPU DCT for Chebyshev fields.
Transforms the field's Z-local coefficient block into its grid buffer
(Z-pencil layout), preserving the field's real or complex dtype.
"""
function distributed_gpu_backward_transform!(field::ScalarField)
    plan = get_or_create_distributed_dct_plan(field)

    # Get local coefficient data (Z-local on the coeff pencil, complex).
    coeffs = get_coeff_data(field)

    # Write into the field's own grid buffer (see the forward wrapper for why a
    # mismatch is an error rather than a substitution). The driver starts from
    # Z-local coeffs and resets pencil orientations.
    T = field.dtype
    shape = plan.pencil.z_pencil_shape
    data = get_grid_data(field)
    if !(data isa CuArray{T,3}) || size(data) != shape ||
       size(coeffs) != plan.coeff_pencil.z_pencil_shape
        error("distributed multi-GPU backward transform of field $(repr(field.name)): rank " *
              "$(field.dist.rank) holds a $(size(coeffs)) coefficient block and a " *
              "$(typeof(data)) grid block of size $(size(data)), but the plan reads " *
              "$(plan.coeff_pencil.z_pencil_shape) coefficients and writes a $T grid block " *
              "of size $shape. Field buffers must keep the shapes they were allocated with.")
    end

    # Perform distributed inverse transform.
    distributed_backward_dct!(data, coeffs, plan)

    return true
end

"""
    Tarang._distributed_gpu_transform!(arch::GPU, field, direction) -> Bool

Route a multi-GPU field to the dedicated distributed DCT-I backend
(`DistributedDCTPlan`) when `Tarang._uses_distributed_gpu_dct` selects it.
`forward_transform!`/`backward_transform!` call this BEFORE their
TransposableField early return (every multi-rank GPU field has transposable
storage, and that path cannot hold a RealFourier half spectrum). Returns `false`
for every other field, which then continues down the generic path.
"""
function Tarang._distributed_gpu_transform!(arch::GPU{CuDevice}, field::ScalarField, direction::Symbol)
    Tarang._uses_distributed_gpu_dct(arch, field.dist.size, field.bases) || return false
    ensure_device!(arch)
    if direction === :forward
        Tarang._with_gpu_workspace(distributed_gpu_forward_transform!, field.dist, field)
    elseif direction === :backward
        Tarang._with_gpu_workspace(distributed_gpu_backward_transform!, field.dist, field)
    else
        throw(ArgumentError("direction must be :forward or :backward, got $(repr(direction))"))
    end
    return true
end

"""
    _gpu_forward_transform_impl!(field::ScalarField)

GPU-specific forward transform using CUFFT (extension-local implementation).
Reached through the dispatched backend method
`Tarang._gpu_forward_transform_backend!(::GPU, field)` defined below — `::GPU` is
more specific than Tarang's `::AbstractArchitecture` fallback, so this EXTENDS
rather than overwrites (which is why a same-signature `gpu_forward_transform!`
method here would be illegal, and why this used to go through a `Ref` hook).
Returns `true` if the GPU transform was applied. Unsupported configurations call
`Tarang.gpu_transform_unsupported` with a reason (it throws); a bare `false` is
still accepted by the core and raises a reason-less error. No CPU fallback is
performed either way.
Supports:
- Pure Fourier (RealFourier, ComplexFourier) - uses cuFFT
- Mixed Fourier×Chebyshev fields use the GPU DCT-I path.
- Single-GPU pure-Chebyshev fields (scaled or not) use the GPU DCT-I path.
  Eligible distributed 3D fields use the multi-GPU DCT-I path above.
"""
# Backend entry points — the more-specific `::GPU` methods Tarang dispatches to.
Tarang._gpu_forward_transform_backend!(::GPU{CuDevice}, field::ScalarField) =
    _gpu_forward_transform_impl!(field)
Tarang._gpu_backward_transform_backend!(::GPU{CuDevice}, field) =
    _gpu_backward_transform_impl!(field)

# Shorthand for a refusal that carries its reason to the user.
_refuse_fwd(field, reason) = Tarang.gpu_transform_unsupported(field, :forward, reason)
_refuse_bwd(field, reason) = Tarang.gpu_transform_unsupported(field, :backward, reason)

function _gpu_forward_transform_impl!(field::ScalarField)
    arch = field.dist.architecture
    if !Tarang.is_gpu(arch)
        return false
    end

    data_g = get_grid_data(field)
    if !isa(data_g, CuArray)
        return false
    end

    gpu_arch = arch::GPU

    # Ensure correct device is active for multi-GPU support
    ensure_device!(gpu_arch)

    # Determine transform type based on bases
    bases = field.bases
    if isempty(bases)
        _refuse_fwd(field, "the field has no spectral bases")
    end

    nprocs = field.dist.size
    # Distributed multi-GPU spectral transform (basis-aware DCT-I + Fourier), gated
    # behind the support predicate. Fires for nprocs>1 (is_distributed_gpu; NCCL is
    # resolved when the plan is built) AND a Chebyshev-containing 3D field
    # (needs_distributed_dct). forward_transform! itself reaches the same driver
    # earlier, via Tarang._distributed_gpu_transform!, before its TransposableField
    # early return; this branch serves direct gpu_forward_transform! calls.
    if is_distributed_gpu(arch, nprocs) && needs_distributed_dct(field)
        if Tarang.distributed_gpu_supported(field)
            return distributed_gpu_forward_transform!(field)
        else
            # Unsupported layout; core raises without CPU staging.
            _refuse_fwd(field, "the distributed multi-GPU DCT-I path declined this field: " *
                               Tarang.distributed_gpu_unsupported_reason(field))
        end
    end

    # Never interpret a rank-local slab as a complete field — for ANY basis
    # combination, not just Fourier. The old guard fired only `if has_fourier`,
    # which let a multi-rank pure-Chebyshev field (e.g. 2D Cheb×Cheb, which the
    # 3D-only distributed-DCT gate above does not catch) fall through to the
    # LOCAL all_chebyshev DCT-I branch below and silently transform each rank's
    # slab as if it were the whole field. Mirrors the unconditional guard in
    # _gpu_backward_transform_impl!.
    if nprocs > 1
        error("Direct forward_transform! is unavailable for GPU fields with " *
              "$(nprocs) MPI processes outside the supported distributed DCT-I path. " *
              "A rank-local transform would be incorrect; CPU staging fallback is disabled. " *
              "Use TransposableField or a supported distributed transform.")
    end

    # Use LOCAL array size from actual data (not global domain size)
    # This is critical for distributed computing where each rank owns a portion
    local_grid_shape = size(data_g)

    if !Tarang.should_use_gpu_fft(field, local_grid_shape)
        _refuse_fwd(field, "the transform-selection heuristic declined the device path " *
                           "for a $(local_grid_shape) array")
    end

    # Classify bases
    all_fourier = all(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    all_chebyshev = all(b -> isa(b, ChebyshevT), bases)
    has_fourier = any(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    has_chebyshev = any(b -> isa(b, ChebyshevT), bases)
    input_T = eltype(data_g)
    coeff_T = Tarang.coefficient_eltype(field.dtype)

    # Reuse the mixed-plan driver for pure Chebyshev fields: with no Fourier
    # axes it is a sequence of the verified on-device DCT-I kernels. A scaled
    # grid keeps the basis coefficient count — the driver truncates or zero-pads
    # each axis, as the CPU chain does (`plan.coeff_shape`).
    if all_chebyshev
        length(bases) <= 3 ||
            _refuse_fwd(field, "the pure-Chebyshev device path handles 1D-3D fields, got " *
                               "$(length(bases))D")
        plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, input_T)
        existing_coeff = get_coeff_data(field)
        if !(existing_coeff isa CuArray) || eltype(existing_coeff) != input_T ||
           size(existing_coeff) != plan.coeff_shape
            set_coeff_data!(field, CUDA.zeros(input_T, plan.coeff_shape...))
        end
        gpu_mixed_forward_transform!(get_coeff_data(field), data_g, plan)
        return true
    end

    if all_fourier
        # Pure Fourier case - use optimized multi-dimensional FFT.
        #
        # CANONICAL LAYOUT (src/core/domain.jl `_fourier_output_size`): only the
        # FIRST Fourier axis is halved, and only if it is RealFourier (rfft of real
        # data). Every other axis — including a RealFourier axis that is NOT first —
        # stays full size (the CPU chain sees complex input there and runs a full
        # C2C fft, fourier.jl `_fourier_forward`). So:
        #   - R2C (multi-dim rfft: R2C on dim 1, C2C on rest) ONLY when bases[1] is
        #     RealFourier AND the grid data is real;
        #   - otherwise full multi-dim C2C, coeff shape == grid shape.
        # A rfft along a non-first RealFourier axis would produce a coeff buffer of
        # a NON-canonical shape (halved along the wrong axis), silently misaligning
        # every consumer sized from `coefficient_shape`.
        # Ask the shared layout rules instead of re-deriving "rfft only when the
        # first axis is RealFourier AND the data is still real" (the multi-dim
        # cuFFT rfft plan is R2C on dim 1 and C2C on the rest, which is exactly
        # the chain `forward_layout` describes).
        use_r2c = Tarang.forward_axis_op(bases[1], local_grid_shape[1],
                                         input_T <: Complex).op === :rfft

        if use_r2c
            # dim 1 is RealFourier with real data: multi-dim rfft (R2C on dim 1,
            # C2C on the rest) — matches the canonical halved-first-axis layout.
            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, input_T; real_input=true)
            _, local_coeff_shape, _ = Tarang.forward_layout(bases, local_grid_shape, input_T)

            existing_coeff = get_coeff_data(field)
            needs_alloc = !(existing_coeff isa CuArray) ||
                          eltype(existing_coeff) != coeff_T ||
                          size(existing_coeff) != local_coeff_shape
            if needs_alloc
                set_coeff_data!(field, CUDA.zeros(coeff_T, local_coeff_shape...))
            end

            gpu_forward_fft!(get_coeff_data(field), data_g, plan)
        else
            # Full C2C on all axes: covers (a) all-ComplexFourier, (b) RealFourier
            # NOT on dim 1 (canonical layout keeps it full size), and (c) complex
            # input with RealFourier on dim 1 (CPU falls back to fft too).
            # C2C requires complex input; promote real data if needed — into
            # cached scratch, not a fresh per-call device array.
            if input_T <: Real
                plan_T = Complex{input_T}
                fft_input = get_gpu_dct_scratch(gpu_arch, local_grid_shape, plan_T, 1)[1]
                fft_input .= data_g
            else
                fft_input = data_g
                plan_T = input_T
            end
            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, plan_T; real_input=false)

            existing_coeff = get_coeff_data(field)
            needs_alloc = !(existing_coeff isa CuArray) ||
                          eltype(existing_coeff) != plan_T ||
                          size(existing_coeff) != local_grid_shape
            if needs_alloc
                set_coeff_data!(field, CUDA.zeros(plan_T, local_grid_shape...))
            end

            gpu_forward_fft!(get_coeff_data(field), fft_input, plan)
        end

        return true

    elseif has_fourier && has_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # LIVE mixed Fourier×Chebyshev path (2D/3D). The Chebyshev stages use
        # `gpu_dct1_along_dim!` — DCT-I on the Gauss-Lobatto grid, Tarang's convention
        # (see mixed_transforms.jl) — NOT DCT-II. Verified: forward→backward round-trip is
        # identity (4.4e-16) and the coefficients are bit-identical to the CPU mixed
        # transform. (A prior comment here mislabeled this a dead DCT-II branch — it is not.)
        # Supports: Fourier-Chebyshev, Fourier-Fourier-Chebyshev, Fourier-Chebyshev-Chebyshev, etc.

        # Get or create mixed transform plan (determines correct coeff_shape)
        plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, input_T)
        local_coeff_shape = plan.coeff_shape

        existing_coeff = get_coeff_data(field)
        needs_alloc = !(existing_coeff isa CuArray) ||
                      eltype(existing_coeff) != coeff_T ||
                      size(existing_coeff) != local_coeff_shape
        if needs_alloc
            set_coeff_data!(field, CUDA.zeros(coeff_T, local_coeff_shape...))
        end

        # Execute mixed transform
        gpu_mixed_forward_transform!(get_coeff_data(field), data_g, plan)

        return true
    end

    # Unsupported combinations (e.g. Legendre).
    _refuse_fwd(field, "no device transform covers this basis combination " *
                       "(supported: pure Fourier, pure Chebyshev, and mixed " *
                       "Fourier×Chebyshev in 2D/3D)")
end

"""
    _gpu_backward_c2c_fft!(field, gpu_arch, data_c, local_grid_shape)

Shared full multi-dimensional C2C inverse FFT for the pure-Fourier backward
path: promotes real coefficient data if needed, sizes/allocates the grid buffer
to `local_grid_shape` (== coeff shape; C2C preserves shape), and stores the
complex inverse-FFT result — mirroring the CPU chain, which keeps the complex
ifft output for C2C axes.
"""
function _gpu_backward_c2c_fft!(field::ScalarField, gpu_arch::GPU, data_c::CuArray,
                                local_grid_shape::Tuple)
    coeff_T = eltype(data_c)
    plan_T = coeff_T <: Real ? Complex{coeff_T} : coeff_T
    # Ordinary complex-to-complex inverses need no staging. Only request
    # scratch for real input promotion or a real field's complex inverse output.
    # When both are needed, retain distinct slots for the out-of-place FFT.
    needs_real_output = field.dtype <: Real
    scratch = if coeff_T <: Real || needs_real_output
        count = coeff_T <: Real && needs_real_output ? 2 : 1
        get_gpu_dct_scratch(Tarang.architecture(data_c), local_grid_shape, plan_T, count)
    else
        nothing
    end
    if coeff_T <: Real
        fft_input = scratch[1]
        fft_input .= data_c
    else
        fft_input = data_c
    end
    plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, plan_T; real_input=false)

    # A REAL-dtype field (a ComplexFourier axis, or a full-length dim-1 spectrum)
    # has an `Array{dtype}`-typed grid buffer, so storing the complex inverse in it
    # throws `TypeError: in setfield!, expected CuArray{Float64}, got CuArray{ComplexF64}`.
    # The forward promoted real grid data, so the imaginary part is roundoff: keep
    # the real part. Mirrors the CPU `_backward_final_real!` (fourier.jl)
    # and `gpu_mixed_backward_transform!`'s `T <: Real` branch.
    real_T = field.dtype
    if real_T <: Real
        inverse_output = scratch[end]
        gpu_backward_fft!(inverse_output, fft_input, plan)
        existing_grid = get_grid_data(field)
        if existing_grid === nothing || !(existing_grid isa CuArray) ||
           eltype(existing_grid) != real_T || size(existing_grid) != local_grid_shape
            set_grid_data!(field, CUDA.zeros(real_T, local_grid_shape...))
        end
        get_grid_data(field) .= real.(inverse_output)
        return nothing
    end

    existing_grid = get_grid_data(field)
    needs_alloc = existing_grid === nothing ||
                  !(existing_grid isa CuArray) ||
                  eltype(existing_grid) != plan_T ||
                  size(existing_grid) != local_grid_shape
    if needs_alloc
        set_grid_data!(field, CUDA.zeros(plan_T, local_grid_shape...))
    end

    gpu_backward_fft!(get_grid_data(field), fft_input, plan)
    return nothing
end

"""
    _gpu_backward_transform_impl!(field::ScalarField)

GPU-specific backward transform using CUFFT (extension-local implementation).
Reached through `Tarang._gpu_backward_transform_backend!(::GPU, field)` — see the
forward implementation for why this is a dispatched backend rather than a hook.
Returns `true` if the GPU transform was applied. Unsupported configurations call
`Tarang.gpu_transform_unsupported` with a reason (it throws); no CPU fallback is
performed.
Supports:
- Pure Fourier (RealFourier, ComplexFourier) - uses cuFFT; a spectrum that does
  not fit the scaled grid (base coefficients after a scale change) is resampled
  on every axis and inverted by the staged driver
- Mixed Fourier×Chebyshev fields use the GPU DCT-I path.
- Single-GPU pure-Chebyshev fields (scaled or not) use the GPU DCT-I path.
  Eligible distributed 3D fields use the multi-GPU DCT-I path above.
"""
function _gpu_backward_transform_impl!(field::ScalarField)
    arch = field.dist.architecture
    if !Tarang.is_gpu(arch)
        return false
    end

    data_c = get_coeff_data(field)
    if !isa(data_c, CuArray)
        return false
    end

    gpu_arch = arch::GPU

    # Ensure correct device is active for multi-GPU support
    ensure_device!(gpu_arch)

    bases = field.bases
    if isempty(bases)
        _refuse_bwd(field, "the field has no spectral bases")
    end

    nprocs = field.dist.size
    # Distributed multi-GPU inverse transform — mirror of gpu_forward_transform!
    # (backward_transform! reaches the driver earlier, via
    # Tarang._distributed_gpu_transform!). Gated behind the support predicate.
    if is_distributed_gpu(arch, nprocs) && needs_distributed_dct(field)
        if Tarang.distributed_gpu_supported(field)
            return distributed_gpu_backward_transform!(field)
        else
            # Unsupported layout; core raises without CPU staging.
            _refuse_bwd(field, "the distributed multi-GPU DCT-I path declined this field: " *
                               Tarang.distributed_gpu_unsupported_reason(field))
        end
    end

    # Never interpret a rank-local slab as a complete field. The distributed
    # DCT-I branch above is the only direct ScalarField multi-rank path handled
    # here; pure Fourier GPU+MPI must use TransposableField/distributed APIs.
    if nprocs > 1
        error("Direct backward_transform! is unavailable for GPU fields with " *
              "$(nprocs) MPI processes outside the supported distributed DCT-I path. " *
              "A rank-local inverse would be incorrect; CPU staging fallback is disabled. " *
              "Use TransposableField or a supported distributed transform.")
    end

    local_coeff_shape = size(data_c)
    # The inverse lands on the field's scaled grid (one process: local == global).
    # It used to be inferred — from the coefficients on every pure-Fourier axis but
    # the first, which therefore kept base resolution, and from the existing grid
    # buffer for mixed fields, whose plan then expected coefficients of the scaled
    # grid and rejected the base coefficients a scale change leaves behind.
    local_grid_shape = Tarang.get_scaled_shape(field)

    # Classify bases
    all_fourier = all(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    all_chebyshev = all(b -> isa(b, ChebyshevT), bases)
    has_fourier = any(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    has_chebyshev = any(b -> isa(b, ChebyshevT), bases)

    # Pure Chebyshev is the no-Fourier case of the mixed-plan DCT-I driver, which
    # zero-pads or truncates each coefficient axis to its (scaled) grid length.
    if all_chebyshev
        length(bases) <= 3 ||
            _refuse_bwd(field, "the pure-Chebyshev device path handles 1D-3D fields, got " *
                               "$(length(bases))D")
        return _gpu_backward_staged!(field, gpu_arch, data_c, local_grid_shape)
    end

    if all_fourier
        # Pure Fourier case — exact mirror of _gpu_forward_transform_impl!:
        # the forward uses R2C (multi-dim rfft) ONLY when bases[1] is RealFourier
        # and the data is real; every other combination — all-ComplexFourier,
        # RealFourier NOT on dim 1 (canonical layout keeps such axes FULL size,
        # src/core/domain.jl `_fourier_output_size`), complex input — is a full
        # multi-dim C2C with coeff shape == grid shape.
        #
        # Classification comes from the SHARED rule (layout.jl), so the device
        # path cannot disagree with the CPU chain about which axes are inverted
        # directly and which must first be resampled onto the scaled grid — on
        # EVERY axis; the rule's ordering also handles N=1/N=2, where
        # rfft_len(N) == N, and refuses a half-spectrum of an unknown grid.
        rules, _, _ = Tarang.backward_resample_rules(bases, local_coeff_shape,
                                                     local_grid_shape, field.dtype)
        if !Tarang.should_use_gpu_fft(field, local_grid_shape)
            _refuse_bwd(field, "the transform-selection heuristic declined the device " *
                               "path for a $(local_grid_shape) array")
        end
        if any(rule -> rule[3] !== :copy, rules)
            # Base coefficients on a scaled grid: resample every such axis, then
            # invert axis by axis (the multi-dim plan has no place for the Nyquist
            # fold a half-spectrum downsample needs between its stages).
            return _gpu_backward_staged!(field, gpu_arch, data_c, local_grid_shape)
        end

        first_is_real = isa(bases[1], RealFourier)

        if first_is_real && !(field.dtype <: Complex)
            # dim 1 stored as an rfft half-spectrum (forward used R2C); every
            # other axis already matches the grid (all rules are `:copy`).
            coeff_T_bk = eltype(data_c)
            coeff_T_bk <: Complex ||
                _refuse_bwd(field, "expected a complex $(length(local_coeff_shape))D " *
                                   "spectrum, got $(coeff_T_bk) coefficients of shape " *
                                   "$(local_coeff_shape)")
            if local_coeff_shape[1] != Tarang.rfft_len(local_grid_shape[1])
                # Full-length dim 1: a C2C spectrum despite the real dtype (e.g.
                # coefficients written directly). CPU stores the complex ifft.
                _gpu_backward_c2c_fft!(field, gpu_arch, data_c, local_coeff_shape)
                return true
            end

            real_T = field.dtype
            existing_grid = get_grid_data(field)
            needs_alloc = existing_grid === nothing ||
                          !(existing_grid isa CuArray) ||
                          eltype(existing_grid) != real_T ||
                          size(existing_grid) != local_grid_shape
            if needs_alloc
                set_grid_data!(field, CUDA.zeros(real_T, local_grid_shape...))
            end

            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, real_T; real_input=true)
            gpu_backward_fft!(get_grid_data(field), data_c, plan)
        else
            # Full multi-dim C2C inverse (grid shape == coeff shape): covers
            # all-ComplexFourier, RealFourier on a non-first dim (those axes were
            # transformed C2C at full size — canonical layout), and complex-dtype
            # fields with RealFourier on dim 1 (forward used C2C). Mirrors the CPU
            # chain, which stores the complex ifft result.
            _gpu_backward_c2c_fft!(field, gpu_arch, data_c, local_grid_shape)
        end

        return true

    elseif has_fourier && has_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # LIVE mixed Fourier×Chebyshev inverse path (2D/3D) — mirror of the forward
        # branch. The Chebyshev stages use `gpu_dct1_along_dim!`
        # (inverse DCT-I), NOT DCT-II; forward→backward round-trips to identity (4.4e-16).
        # (A prior comment here mislabeled this a dead DCT-II branch — it is not.)
        # Supports: Fourier-Chebyshev, Fourier-Fourier-Chebyshev, Fourier-Chebyshev-Chebyshev, etc.
        if !Tarang.should_use_gpu_fft(field, local_grid_shape)
            _refuse_bwd(field, "the transform-selection heuristic declined the device " *
                               "path for a $(local_grid_shape) array")
        end
        return _gpu_backward_staged!(field, gpu_arch, data_c, local_grid_shape)
    end

    # Unsupported combinations (e.g. Legendre).
    _refuse_bwd(field, "no device inverse transform covers this basis combination " *
                       "(supported: pure Fourier, pure Chebyshev, and mixed " *
                       "Fourier×Chebyshev in 2D/3D)")
end

# ============================================================================
# GPU FFT Transforms using CUFFT
# ============================================================================

"""
    GPUFFTPlan

Wrapper for CUFFT plans that work with CuArrays.
"""
struct GPUFFTPlan{P, IP}
    plan::P
    iplan::IP
    size::Tuple{Vararg{Int}}
    is_real::Bool
end

"""
    plan_gpu_fft(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)

Create a GPU FFT plan using CUFFT for element type `T`.

**Important:** `local_size` should be the LOCAL array shape (what this process owns),
not the global domain size. In distributed computing, each rank creates plans
for its local data portion.

For multi-GPU: ensures the plan is created on the correct device.
"""
function plan_gpu_fft(arch::GPU{CuDevice}, local_size::Tuple, T::Type; real_input::Bool=false)
    # Ensure we're on the correct device for plan creation
    ensure_device!(arch)

    complex_T = T <: Complex ? T : Complex{T}

    if real_input
        # Real-to-complex FFT (like rfft)
        # Plan is created based on local array dimensions
        dummy_in = CUDA.zeros(T, local_size...)
        plan = CUFFT.plan_rfft(dummy_in)
        return GPUFFTPlan(plan, inv(plan), local_size, true)
    else
        # Complex-to-complex FFT
        dummy = CUDA.zeros(complex_T, local_size...)
        plan = CUFFT.plan_fft(dummy)
        iplan = CUFFT.plan_ifft(dummy)

        return GPUFFTPlan(plan, iplan, local_size, false)
    end
end

# Fallback for generic GPU: delegates to device-specific version using current device,
# ensuring proper device context via ensure_device!
plan_gpu_fft(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false) =
    plan_gpu_fft(GPU{CuDevice}(CUDA.device()), local_size, T; real_input=real_input)

"""
    gpu_forward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)

Execute forward FFT on GPU.
"""
function gpu_forward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)
    mul!(output, plan.plan, input)
    return output
end

"""
    gpu_backward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)

Execute backward (inverse) FFT on GPU.

The cuFFT plan preserves the input coefficients during a real inverse.
"""
function gpu_backward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)
    return mul!(output, plan.iplan, input)
end

# ============================================================================
# GPU Transform Plan Cache (Thread-Safe)
# ============================================================================

"""
    GPUTransformCache

Thread-safe cache for GPU FFT plans to avoid recreation overhead.
Keys include device ID for multi-GPU safety.
Uses a ReentrantLock to protect concurrent access from multiple Julia threads.
"""
struct GPUTransformCache
    plans::Dict{Tuple, GPUFFTPlan}
    lock::ReentrantLock
end

const GPU_TRANSFORM_CACHE = GPUTransformCache(Dict{Tuple, GPUFFTPlan}(), ReentrantLock())

"""
    get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)

Get or create a cached GPU FFT plan (thread-safe).
Plans are cached by (device_id, size, element_type, real_input).
"""
function get_gpu_fft_plan(arch::GPU{CuDevice}, local_size::Tuple, T::Type; real_input::Bool=false)
    device_id = CUDA.deviceid(arch.device)
    key = (device_id, local_size, T, real_input)

    lock(GPU_TRANSFORM_CACHE.lock) do
        return get!(_transform_cache(GPU_TRANSFORM_CACHE.plans), key) do
            plan_gpu_fft(arch, local_size, T; real_input)
        end
    end
end

# Generic GPU uses the current device, matching plan_gpu_fft's fallback.
get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false) =
    get_gpu_fft_plan(GPU{CuDevice}(CUDA.device()), local_size, T; real_input=real_input)

"""
    clear_gpu_transform_cache!()

Clear cached CUDA FFT plans and the calling task's asynchronous real inverses.
"""
function clear_gpu_transform_cache!()
    lock(GPU_TRANSFORM_CACHE.lock) do
        empty!(Tarang._gpu_workspace_cache(_ASYNC_REAL_INVERSES))
        empty!(_transform_cache(GPU_TRANSFORM_CACHE.plans))
    end
end
