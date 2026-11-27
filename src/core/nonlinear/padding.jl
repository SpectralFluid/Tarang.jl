"""Padding, truncation, and padded multiplication helpers for nonlinear evaluation."""

# ============================================================================
# Padded Dealiasing (3/2-rule) — GPU and MPI compatible
# ============================================================================

@inline function _padded_fourier_size(n::Int, factor::Real)
    factor <= 1 && return n
    m = ceil(Int, factor * n)
    # Keeping both Nyquist images requires a strict bound: at m = 3n/2,
    # quadratics at ±n wrap onto the retained ∓n/2 modes.
    if factor >= 1.5
        m = max(m, 3n ÷ 2 + 1)
    end
    return 2cld(m, 2)
end

# A Lobatto grid with n samples is the first half (including both endpoints)
# of an even periodic grid with 2(n-1) samples. Fourier padding on this grid
# is cosine-series interpolation. Folding the two endpoint modes on return
# gives the truncated Chebyshev series, including its highest retained degree.
function _product_fft_layout(bases, factor)
    nodes = map(b -> b.meta.size, bases)
    cosine_dims = findall(eachindex(bases)) do d
        bases[d] isa ChebyshevT && nodes[d] > 1 && _axis_dealias_factor(bases[d], factor) > 1
    end
    dims = findall(d -> bases[d] isa FourierBasis || d in cosine_dims, eachindex(bases))
    shape = ntuple(d -> d in cosine_dims ? 2 * (nodes[d] - 1) : nodes[d], length(nodes))
    padded = ntuple(length(nodes)) do d
        d in dims ? _padded_fourier_size(shape[d], _axis_dealias_factor(bases[d], factor)) : shape[d]
    end
    return shape, padded, dims, cosine_dims
end

@kernel function _extend_lobatto_grid_kernel!(out, @Const(grid))
    I = @index(Global, NTuple)
    J = ntuple(d -> I[d] <= size(grid, d) ? I[d] : 2 * size(grid, d) - I[d], Val(length(I)))
    @inbounds out[I...] = grid[J...]
end

function _extend_lobatto_grid!(arch, out, grid)
    for d in 1:ndims(out)
        n, m = size(grid, d), size(out, d)
        m == n || m == 2 * (n - 1) || throw(DimensionMismatch(
            "Cannot extend $(size(grid)) to the Lobatto FFT grid $(size(out))"))
    end
    launch!(arch, _extend_lobatto_grid_kernel!, out, grid; ndrange=size(out))
    return out
end

@inline function _lobatto_grid_view(data, grid)
    size(data) == size(grid) && return data
    return view(data, axes(grid)...)
end

"""
    PaddedDealiasingWorkspace{T, A}

Pre-allocated workspace for proper 3/2-rule dealiasing (Orszag 1971).
Works on CPU, GPU, and MPI-distributed (PencilArray) data.

Type parameters:
- `T`: Float type (Float32 or Float64)
- `A`: Array type (Array{Complex{T}} for CPU, CuArray{Complex{T}} for GPU)

For MPI-distributed data, the workspace operates on each rank's LOCAL array
(parent of PencilArray), padding only non-decomposed Fourier dimensions.
"""
mutable struct PaddedDealiasingWorkspace{T<:AbstractFloat, A<:AbstractArray{Complex{T}}} <: AbstractNonlinearTransformConfig
    original_shape::Tuple{Vararg{Int}}
    padded_shape::Tuple{Vararg{Int}}
    fourier_dims::Vector{Int}
    cosine_dims::Vector{Int}

    # Pre-allocated padded arrays (on same architecture as input data)
    padded1::A
    padded2::A
    # Operands are transformed/padded sequentially, then this same original-size
    # buffer receives the truncated product. padded1 also holds the product.
    spectrum::A

    # FFT plans: FFTW.MEASURE for CPU, plain plan_fft for GPU
    plan_forward::AbstractFFTPlan
    plan_backward::AbstractFFTPlan

    # In-place plans for the shared ORIGINAL-size spectral buffer,
    # so the per-call forward/inverse FFTs reuse buffers instead of allocating.
    plan_spec_forward::AbstractFFTPlan
    plan_spec_backward::AbstractFFTPlan

    # Architecture for dispatch
    arch::AbstractArchitecture
    cached_operand::Any
end

"""
    _get_padded_workspace!(evaluator, bases, dtype; local_shape=nothing,
                           local_fourier_dims=nothing, arch=nothing)

Get or create a cached padded dealiasing workspace.

For serial fields, the FFT grid includes every Fourier axis and the even
extensions of ChebyshevT axes that request padding.
For MPI-distributed fields, pass `local_shape` (the per-rank array shape) and
`local_fourier_dims` (only the non-decomposed Fourier dimensions to pad).
"""
function _get_padded_workspace!(evaluator::NonlinearEvaluator, bases::Tuple, dtype::Type{T};
                                real_input::Bool=false,
                                local_shape::Union{Nothing, Tuple}=nothing,
                                local_fourier_dims::Union{Nothing, Vector{Int}}=nothing,
                                arch::Union{Nothing, AbstractArchitecture}=nothing) where T
    if real_input && local_shape === nothing && arch === nothing
        return _get_real_padded_workspace!(evaluator, bases, T)
    end
    _arch = arch !== nothing ? arch : evaluator.dist.architecture
    # Use tuple key to avoid string allocation on every call
    key = (hash(bases), dtype, hash(local_shape), is_gpu(_arch))
    if haskey(evaluator.pencil_transforms, key)
        return evaluator.pencil_transforms[key]::PaddedDealiasingWorkspace{T}
    end

    factor = evaluator.dealiasing_factor

    if local_shape === nothing
        orig_t, pad_t, fourier_dims, cosine_dims = _product_fft_layout(bases, factor)
    else
        orig_t = local_shape
        fourier_dims = something(local_fourier_dims, Int[])
        cosine_dims = Int[]
        pad_t = ntuple(length(orig_t)) do d
            d in fourier_dims || return orig_t[d]
            strength = d <= length(bases) ? _axis_dealias_factor(bases[d], factor) : factor
            _padded_fourier_size(orig_t[d], strength)
        end
    end
    isempty(fourier_dims) && return nothing

    # Allocate padded arrays on the correct architecture
    padded1 = zeros(_arch, Complex{T}, pad_t...)
    padded2 = zeros(_arch, Complex{T}, pad_t...)
    spectrum = zeros(_arch, Complex{T}, orig_t...)

    # Create FFT plans — CPU gets FFTW.MEASURE, GPU gets plain plan_fft.
    # `plan_*!` are in-place (mutate their argument); the spec plans operate on the
    # original-size shared buffer.
    if is_gpu(_arch)
        # For GPU: AbstractFFTs.plan_fft! dispatches to CUFFT (no flags arg).
        # In-place plans, like the CPU branch — the out-of-place forms allocated
        # (and then copied) a fresh padded-size device array per FFT, three times
        # per product evaluation.
        plan_forward, plan_backward = _device_complex_fft_plans!(padded1, fourier_dims)
        plan_spec_forward, plan_spec_backward = _device_complex_fft_plans!(spectrum, fourier_dims)
    else
        # In-place padded plans (UNALIGNED — applied to padded1/padded2):
        # the per-call `ws.padded .= plan * ws.padded` then transforms in place instead
        # of allocating a fresh padded array each FFT/IFFT (~one padded array per call).
        plan_forward = FFTW.plan_fft!(padded1, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
        plan_backward = FFTW.plan_ifft!(padded1, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
        plan_spec_forward = FFTW.plan_fft!(spectrum, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
        plan_spec_backward = FFTW.plan_ifft!(spectrum, fourier_dims; flags=FFTW.MEASURE | FFTW.UNALIGNED)
    end

    ws = PaddedDealiasingWorkspace{T, typeof(padded1)}(
        orig_t, pad_t, fourier_dims, cosine_dims,
        padded1, padded2, spectrum,
        plan_forward, plan_backward,
        plan_spec_forward, plan_spec_backward, _arch, nothing
    )
    evaluator.pencil_transforms[key] = ws
    return ws
end

# ============================================================================
# Spectral Padding / Truncation — slice-based for GPU compatibility
# ============================================================================

"""Compute positive and negative frequency index ranges for padding/truncation."""
function _freq_ranges(N::Int, M::Int, is_fourier::Bool)
    if !is_fourier
        return (1:N, 1:N, 1:0, 1:0)  # (orig_pos, pad_pos, orig_neg, pad_neg) — full copy, no neg
    end
    Nh = N ÷ 2
    pos_orig = 1:Nh+1
    pos_pad = 1:Nh+1
    n_neg = N - Nh - 1  # number of negative frequencies
    if n_neg > 0
        neg_orig = N-n_neg+1:N
        neg_pad = M-n_neg+1:M
    else
        neg_orig = 1:0  # empty
        neg_pad = 1:0
    end
    return (pos_orig, pos_pad, neg_orig, neg_pad)
end

# The block copies place the coarse Nyquist bin at +N/2. A real
# trigonometric interpolant splits it between both signs. A complex Fourier
# field assigns it entirely to -N/2. Chebyshev even extensions always split.
function _place_padded_nyquist!(padded, original_shape, padded_shape, fourier_dims, cosine_dims)
    for d in fourier_dims
        n = original_shape[d]
        (iseven(n) && padded_shape[d] > n) || continue
        positive = selectdim(padded, d, n ÷ 2 + 1)
        negative = selectdim(padded, d, padded_shape[d] - n ÷ 2 + 1)
        if d in cosine_dims
            positive .*= real(eltype(padded))(0.5)
            negative .= positive
        else
            negative .= positive
            fill!(positive, zero(eltype(padded)))
        end
    end
    return padded
end

function _collect_truncated_nyquist!(padded, original_shape, padded_shape, fourier_dims, cosine_dims)
    for d in fourier_dims
        n = original_shape[d]
        (iseven(n) && padded_shape[d] > n) || continue
        positive = selectdim(padded, d, n ÷ 2 + 1)
        negative = selectdim(padded, d, padded_shape[d] - n ÷ 2 + 1)
        if d in cosine_dims
            positive .+= negative
        else
            positive .= negative
        end
    end
    return padded
end

"""
    _pad_spectral!(padded, spec_data, original_shape, padded_shape, fourier_dims)

Zero-pad spectral coefficients. Uses slice-based operations for GPU compatibility.
"""
function _pad_spectral!(padded::AbstractArray{Complex{T}}, spec_data::AbstractArray{Complex{T}},
                        original_shape::Tuple, padded_shape::Tuple,
                        fourier_dims::Vector{Int}, cosine_dims=fourier_dims) where T
    fill!(padded, zero(Complex{T}))
    ndim = length(original_shape)

    if ndim == 1
        r = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        # @views: the RHS range slice is a view, not a materialized copy. Without
        # it `spec_data[r[1]]` allocates a fresh Vector every call — the dominant
        # per-step allocation for 1D nonlinear products (the 2D/3D paths below
        # already use views).
        @views padded[r[2]] .= spec_data[r[1]]
        if length(r[3]) > 0
            @views padded[r[4]] .= spec_data[r[3]]
        end
    elseif ndim == 2
        _pad_spectral_sliced_2d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    elseif ndim == 3
        _pad_spectral_sliced_3d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    end
    _place_padded_nyquist!(padded, original_shape, padded_shape, fourier_dims, cosine_dims)
end

function _pad_spectral_sliced_2d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
    r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)

    # @views: range-indexed RHS slices are views, not materialized copies (no per-call alloc).
    @views padded[r1[2], r2[2]] .= spec_data[r1[1], r2[1]]
    if length(r2[3]) > 0
        @views padded[r1[2], r2[4]] .= spec_data[r1[1], r2[3]]
    end
    if length(r1[3]) > 0
        @views padded[r1[4], r2[2]] .= spec_data[r1[3], r2[1]]
    end
    if length(r1[3]) > 0 && length(r2[3]) > 0
        @views padded[r1[4], r2[4]] .= spec_data[r1[3], r2[3]]
    end
end

function _pad_spectral_sliced_3d!(padded, spec_data, original_shape, padded_shape, fourier_dims)
    r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
    r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)
    r3 = _freq_ranges(original_shape[3], padded_shape[3], 3 in fourier_dims)

    # Copy all 8 quadrant combinations (pos/neg for each of 3 dims)
    for (s1, d1) in ((r1[1], r1[2]), (r1[3], r1[4]))
        length(s1) == 0 && continue
        for (s2, d2) in ((r2[1], r2[2]), (r2[3], r2[4]))
            length(s2) == 0 && continue
            for (s3, d3) in ((r3[1], r3[2]), (r3[3], r3[4]))
                length(s3) == 0 && continue
                @views padded[d1, d2, d3] .= spec_data[s1, s2, s3]
            end
        end
    end
end

"""
    _truncate_spectral!(result, padded_spec, original_shape, padded_shape, fourier_dims)

Truncate padded spectral coefficients back to original size.
Uses slice-based operations for GPU compatibility.
"""
function _truncate_spectral!(result::AbstractArray{Complex{T}}, padded_spec::AbstractArray{Complex{T}},
                             original_shape::Tuple, padded_shape::Tuple,
                             fourier_dims::Vector{Int}, cosine_dims=fourier_dims) where T
    ndim = length(original_shape)

    _collect_truncated_nyquist!(padded_spec, original_shape, padded_shape, fourier_dims, cosine_dims)

    if ndim == 1
        r = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        # @views: avoid materializing the RHS range slice (see `_pad_spectral!`).
        @views result[r[1]] .= padded_spec[r[2]]
        if length(r[3]) > 0
            @views result[r[3]] .= padded_spec[r[4]]
        end
    elseif ndim == 2
        r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)
        @views result[r1[1], r2[1]] .= padded_spec[r1[2], r2[2]]
        if length(r2[3]) > 0
            @views result[r1[1], r2[3]] .= padded_spec[r1[2], r2[4]]
        end
        if length(r1[3]) > 0
            @views result[r1[3], r2[1]] .= padded_spec[r1[4], r2[2]]
        end
        if length(r1[3]) > 0 && length(r2[3]) > 0
            @views result[r1[3], r2[3]] .= padded_spec[r1[4], r2[4]]
        end
    elseif ndim == 3
        r1 = _freq_ranges(original_shape[1], padded_shape[1], 1 in fourier_dims)
        r2 = _freq_ranges(original_shape[2], padded_shape[2], 2 in fourier_dims)
        r3 = _freq_ranges(original_shape[3], padded_shape[3], 3 in fourier_dims)
        for (d1, s1) in ((r1[1], r1[2]), (r1[3], r1[4]))
            length(d1) == 0 && continue
            for (d2, s2) in ((r2[1], r2[2]), (r2[3], r2[4]))
                length(d2) == 0 && continue
                for (d3, s3) in ((r3[1], r3[2]), (r3[3], r3[4]))
                    length(d3) == 0 && continue
                    @views result[d1, d2, d3] .= padded_spec[s1, s2, s3]
                end
            end
        end
    end
end

# ============================================================================
# Padded Multiply — architecture-aware
# ============================================================================

# Function barrier: `src` arrives `Any`-typed (from `on_architecture` on an
# abstract-typed arch field). Dispatching here recovers its concrete type so the
# convert-copy into `dst` is a specialized, allocation-free broadcast.
@inline function _copy_convert_into!(dst::AbstractArray{Complex{T}}, src::AbstractArray) where {T}
    dst .= Complex{T}.(src)
    return dst
end

_padded_coefficient_compatible(f, shape) =
    all(b -> b isa Union{RealFourier,ComplexFourier}, f.bases) &&
    (f.scales === nothing || all(isone, f.scales)) &&
    size(get_local_data(get_coeff_data(f))) == shape

function _complex_operand_spectrum!(ws, field)
    if field.dtype <: Complex && field.current_layout === :c &&
       _padded_coefficient_compatible(field, ws.original_shape)
        copyto!(ws.spectrum, get_local_data(get_coeff_data(field)))
    else
        ensure_layout!(field, :g)
        src = on_architecture(ws.arch, get_local_data(get_grid_data(field)))
        if isempty(ws.cosine_dims)
            _copy_convert_into!(ws.spectrum, src)
        else
            _extend_lobatto_grid!(ws.arch, ws.spectrum, src)
        end
        ws.plan_spec_forward * ws.spectrum
    end
    return ws.spectrum
end

"""
    evaluate_padded_multiply(field1, field2, evaluator, ws)

Multiply two fields with proper 3/2-rule padded dealiasing.
Works on CPU and GPU. For MPI data, operates on the local array.
"""
function evaluate_padded_multiply(field1::ScalarField, field2::ScalarField,
                                  evaluator::NonlinearEvaluator,
                                  ws::PaddedDealiasingWorkspace{T};
                                  destination::Union{Nothing,ScalarField}=nothing,
                                  result_layout::Symbol=:g, operand_key=nothing) where T
    # Step 1: FFT to spectral along Fourier dimensions, in place via pre-built
    # plans (no per-call allocation). Plans dispatch to CUFFT for GPU arrays.
    cosine_dims = field1.dtype <: Real ? ws.fourier_dims : ws.cosine_dims
    _complex_operand_spectrum!(ws, field1)
    _pad_spectral!(ws.padded1, ws.spectrum, ws.original_shape, ws.padded_shape, ws.fourier_dims, cosine_dims)
    ws.plan_backward * ws.padded1
    if operand_key === nothing || ws.cached_operand !== operand_key
        ws.cached_operand = nothing
        _complex_operand_spectrum!(ws, field2)
        _pad_spectral!(ws.padded2, ws.spectrum, ws.original_shape, ws.padded_shape, ws.fourier_dims, cosine_dims)
        ws.plan_backward * ws.padded2
        ws.cached_operand = operand_key
    end

    # Step 4: Multiply on padded grid
    ws.padded1 .*= ws.padded2

    # Step 5: FFT product back (in place)
    ws.plan_forward * ws.padded1

    # Step 6: Truncate to original coefficients
    # Truncation overwrites every original-size element; no clear is needed.
    _truncate_spectral!(ws.spectrum, ws.padded1, ws.original_shape, ws.padded_shape, ws.fourier_dims, cosine_dims)

    # Step 7: IFFT to grid and normalize
    # Normalization: padded IFFT divides by M, but we want result on N-grid.
    # Scale by M/N per padded Fourier dimension.
    scale = one(T)
    for d in ws.fourier_dims
        scale *= T(ws.padded_shape[d]) / T(ws.original_shape[d])
    end
    # Write result to a pooled output field (rotating buffers — distinct for
    # consecutively-held products like cross product; no per-call allocation).
    result = destination === nothing ? _checkout_nl_result!(evaluator, field1) : destination
    if result_layout === :c && result.dtype <: Complex &&
       _padded_coefficient_compatible(result, ws.original_shape)
        dst = get_local_data(get_coeff_data(result))
        @. dst = ws.spectrum * scale
        result.current_layout = :c
        return result
    end
    ws.plan_spec_backward * ws.spectrum
    result_data = grid_data!(result)

    # Write the scaled result directly into the (preallocated) output grid via a
    # fused broadcast — no intermediate `scaled_result` allocation. The workspace
    # and the output field share an architecture, so no device transfer is needed.
    dst = get_local_data(result_data)
    values = _lobatto_grid_view(ws.spectrum, dst)
    if field1.dtype <: Real
        @. dst = real(values) * scale
    else
        @. dst = values * scale
    end
    ensure_layout!(result, result_layout)
    return result
end

# ============================================================================
# Distributed (MPI) dealiasing via 2/3-rule truncation
# ============================================================================

"""
    _dealias_truncate_field!(field, dealiasing_factor)

Zero all Fourier modes with |k| > N/(2·dealiasing_factor) on `field`, in place.
Forward-transforms, applies the per-rank global-wavenumber cutoff (distributed)
or the standard cutoff (serial), then transforms back to grid space.
"""
function _dealias_truncate_field!(field::ScalarField, dealiasing_factor::Float64; final_layout::Symbol=:g)
    forward_transform!(field)
    coeff_data = get_coeff_data(field)

    if isa(coeff_data, PencilArrays.PencilArray)
        _apply_spectral_cutoff_distributed!(
            coeff_data, field.bases, dealiasing_factor, field.dtype)
    else
        nb = length(field.bases)
        cutoffs = ntuple(nb) do i
            c = _axis_dealias_cutoff(field.bases[i], dealiasing_factor)
            c === nothing ? size(coeff_data, i) : c
        end
        bundle = _field_transform_bundle(field)
        rfft_dims = ntuple(i -> isa(field.bases[i], RealFourier) &&
                                _axis_uses_rfft(bundle, i), nb)
        apply_spectral_cutoff!(coeff_data, cutoffs, rfft_dims)
    end

    # Leave in coeff when the caller will consume/sum in coeff (saves a backward
    # transpose — e.g. dot-product terms summed in coeff with one final backward).
    final_layout === :g && backward_transform!(field)
    return field
end

# ============================================================================
# Batched distributed backward transform (MPI message coalescing)
#
# A per-field backward issues its own all-to-all transpose. Stacking `k`
# same-shaped fields and transforming once issues ONE transpose for all of them.
# We use PencilFFTs' `extra_dims`: the batch is a trailing LOCAL dimension that
# rides through the SAME transpose/permutation sequence as the field's own plan,
# so each slice's layout is identical to the field's coeff array (verified) and
# stacking is a plain copy. Latency win that grows with rank count (transpose
# count is otherwise independent of nprocs under slab decomposition).
# ============================================================================

"""Get or build (cached) a batched PencilFFTPlan for `B` stacked fields. Uses the
same global shape / transforms / process mesh as the per-field plan plus
`extra_dims=(B,)`, so the per-slice layout matches each field's coeff array."""
function _get_batched_backward_plan!(field::ScalarField, B::Int)
    bundle = _field_transform_bundle(field)
    input_pencil = bundle.pencil_fft_input
    input_pencil === nothing && error(
        "batched PencilFFT requested for a field without an input pencil")
    cache = bundle.batched_plan_cache

    # The cache belongs to this exact domain/dtype plan bundle. Evicting the
    # bundle therefore releases its batched plan and both full-size scratch
    # arrays as one lifecycle unit; an object-id-only global cache leaked those
    # arrays and could eventually return a stale plan after identity reuse.
    cached = get(cache, B, nothing)
    cached !== nothing && return cached

    transforms = _pencil_transform_tuple(bundle.forward_ops)
    real_dtype = typeof(real(zero(field.dtype)))
    plan = PencilFFTs.PencilFFTPlan(
        input_pencil, transforms, real_dtype; extra_dims=(B,))
    # Reused scratch buffers (coeff-side input, grid-side output) — consumed
    # within each call, so caching removes the two per-call allocations.
    cstack = PencilFFTs.allocate_output(plan)
    gstack = PencilFFTs.allocate_input(plan)
    entry = (plan, cstack, gstack)
    cache[B] = entry
    return entry
end

"""
    _pencil_batched_backward!(fields::Vector{<:ScalarField})

Backward-transform `k` same-shaped distributed fields (currently coefficient
layout) using ONE batched PencilFFTs transpose instead of `k` separate ones.
Stacks each field's coeff PencilArray along the trailing batch (extra) axis,
applies the single batched `\\`, and unstacks the grid result into each field.

Fast path requires CPU-backed PencilArray coeff data sharing `dist`/`bases`;
otherwise falls back to per-field `backward_transform!`.
"""
function _pencil_batched_backward!(fields::Vector{<:ScalarField})
    k = length(fields)
    k == 0 && return
    f0 = fields[1]
    cd0 = coeff_data!(f0)
    if k == 1 || !isa(cd0, PencilArrays.PencilArray) || is_gpu_array(cd0)
        for f in fields
            backward_transform!(f)
        end
        return
    end

    if any(f -> f.domain !== f0.domain || f.dtype !== f0.dtype, fields)
        foreach(backward_transform!, fields)
        return
    end

    # A mixed distributed domain has a second, coupled inverse stage: transpose
    # to `pencil_solve`, apply the local Chebyshev DCT, and transpose back.
    # The extra-dimension PencilFFT plan below batches only the Fourier stage, so
    # using it here would mark Fourier-inverted/Chebyshev-coefficient data as a
    # valid grid. Keep the optimization to pure-Fourier bundles until that
    # coupled stage has an explicitly batched implementation.
    if any(basis -> !(basis isa Union{RealFourier, ComplexFourier}), f0.bases)
        foreach(backward_transform!, fields)
        return
    end

    plan, cstack, gstack = _get_batched_backward_plan!(f0, k)
    cp = parent(cstack)
    nd = ndims(cp)
    for (i, f) in enumerate(fields)
        cf = parent(coeff_data!(f))
        dst = selectdim(cp, nd, i)
        size(dst) == size(cf) ||
            error("_pencil_batched_backward!: coeff slice $(size(dst)) != field coeff $(size(cf)) — batched-plan layout mismatch")
        copyto!(dst, cf)
    end

    ldiv!(gstack, plan, cstack)                 # ONE transpose for all k slices, in place
    gp = parent(gstack)
    for (i, f) in enumerate(fields)
        gdp = parent(get_grid_data(f))
        slice = selectdim(gp, nd, i)
        if eltype(gdp) <: Real && eltype(slice) <: Complex
            gdp .= real.(slice)
        else
            gdp .= slice
        end
        f.current_layout = :g
    end
    return
end

"""
    _truncate_coeff_into_grid!(dst, src, dealiasing_factor)

Copy `src` into scratch `dst` and band-limit it to |k| ≤ N/(2·factor), leaving
`dst` in GRID layout ready to multiply. The cutoff is applied in COEFFICIENT
space, so when `src` already lives in coefficient space (the usual case —
operands come from spectral derivatives) only a single backward transform is
needed. The old path forced `src` to grid, copied, then forward-transformed the
copy just to truncate it and backward-transformed again — a redundant c→g→c
round trip this avoids.
"""
function _truncate_coeff_into_grid!(dst::ScalarField, src::ScalarField, dealiasing_factor::Float64)
    _truncate_coeff_only!(dst, src, dealiasing_factor)
    backward_transform!(dst)  # coeff → grid: the single required transform
    return dst
end

"""
    _truncate_coeff_only!(dst, src, dealiasing_factor)

The coefficient-space half of `_truncate_coeff_into_grid!`: copy `src`'s coeff
into `dst` and apply the spectral cutoff, leaving `dst` in COEFFICIENT layout
(no backward transform). Split out so several inputs can be truncated, then
backward-transformed together in one batched transpose (see
`_pencil_batched_backward!`).
"""
function _truncate_coeff_only!(dst::ScalarField, src::ScalarField, dealiasing_factor::Float64)
    ensure_layout!(src, :c)   # forward only if src was in grid; no-op if already coeff
    sc = get_coeff_data(src)
    dc = get_coeff_data(dst)  # coeff buffer; contents overwritten below, so no transform needed
    if isa(dc, PencilArrays.PencilArray) && isa(sc, PencilArrays.PencilArray)
        copyto!(parent(dc), parent(sc))
    else
        copyto!(dc, sc)
    end
    dst.current_layout = :c

    if isa(dc, PencilArrays.PencilArray)
        _apply_spectral_cutoff_distributed!(
            dc, dst.bases, dealiasing_factor, dst.dtype)
    else
        nb = length(dst.bases)
        cutoffs = ntuple(nb) do i
            c = _axis_dealias_cutoff(dst.bases[i], dealiasing_factor)
            c === nothing ? size(dc, i) : c
        end
        bundle = _field_transform_bundle(dst)
        rfft_dims = ntuple(i -> isa(dst.bases[i], RealFourier) &&
                                _axis_uses_rfft(bundle, i), nb)
        apply_spectral_cutoff!(dc, cutoffs, rfft_dims)
    end
    return dst
end

"""
    evaluate_truncated_multiply_distributed(field1, field2, evaluator)

Multiply two MPI-distributed fields with 2/3-rule dealiasing.

Exact 3/2 zero-padding is not used under MPI because embedding the N-mode
spectrum into the 3N/2 padded spectrum requires cross-rank redistribution
(the original and padded PencilArrays decompose differently). The 2/3 rule is
purely local per rank: truncate both inputs to |k| ≤ N/3, multiply on the grid,
then truncate the product to |k| ≤ N/3. The product of two N/3-band fields has
support up to 2N/3; on the N grid those modes alias only into |k| ≥ N/3, so the
retained |k| ≤ N/3 band is alias-free — i.e. quadratic terms are dealiased
exactly within the retained band.
"""
# Rotating pool of product-result buffers: avoids a fresh ScalarField per product
# while still handing distinct buffers to callers that hold several products at
# once (e.g. cross product). 8 buffers ⇒ safe for realistic nested product depth;
# each buffer is fully overwritten before reuse.
const _NL_RESULT_POOL_SIZE = 8
const _NL_RESULT_IDX = Ref(0)

@inline function _nonlinear_temp_field_key(name::AbstractString, template::ScalarField)
    # Scratch PencilArrays and their transform plan must come from the exact
    # domain/dtype bundle, not merely an equal basis tuple. A live cached field
    # retains the bundle, so its object identity is safe for the key lifetime.
    # Domainless scalar tau fields have no plan; an evaluator belongs to one
    # Distributor, so name plus dtype uniquely identifies compatible 0-D scratch.
    template.domain === nothing && return string(name, "_0d_", template.dtype)
    bundle_id = objectid(_field_transform_bundle(template))
    return string(name, '_', bundle_id)
end

function _checkout_nl_result!(evaluator::NonlinearEvaluator, field1::ScalarField)
    i = _NL_RESULT_IDX[] % _NL_RESULT_POOL_SIZE
    _NL_RESULT_IDX[] += 1
    # Tuple key (no per-call string allocation; matches the tuple-key convention
    # used by `_get_padded_workspace!`).
    key = (i, objectid(_field_transform_bundle(field1)), field1.dtype, is_transposable_storage(field1))
    pool = evaluator.nl_result_pool
    cached = get(pool, key, nothing)
    cached === nothing || return cached
    field = is_transposable_storage(field1) ? copy(field1) :
            ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    field.name = "_nl_product"
    pool[key] = field
    return field
end

function evaluate_truncated_multiply_distributed(field1::ScalarField, field2::ScalarField,
                                                  evaluator::NonlinearEvaluator;
                                                  result_layout::Symbol=:g)
    factor = evaluator.dealiasing_factor

    # Truncate the inputs into reusable scratch fields (consumed within this call,
    # so sharing them across calls is safe — unlike `result`, which the caller may
    # hold alongside a second product, e.g. CrossProduct). This avoids the two
    # per-call `copy(field)` allocations.
    f1 = get!(() -> ScalarField(field1.dist, "_nl_trunc_f1", field1.bases, field1.dtype),
              evaluator.temp_fields, _nonlinear_temp_field_key("_nl_trunc_f1", field1))
    f2 = get!(() -> ScalarField(field1.dist, "_nl_trunc_f2", field1.bases, field1.dtype),
              evaluator.temp_fields, _nonlinear_temp_field_key("_nl_trunc_f2", field1))
    # Band-limit both inputs in coeff space, then bring them back to grid with a
    # SINGLE batched transpose instead of one per input (2 backwards → 1).
    _truncate_coeff_only!(f1, field1, factor)
    _truncate_coeff_only!(f2, field2, factor)
    _pencil_batched_backward!([f1, f2])

    # Multiply on the grid. `result` comes from a rotating buffer pool — distinct
    # buffers for consecutively-held products (e.g. cross product), no per-call alloc.
    result = _checkout_nl_result!(evaluator, field1)
    result_data = grid_data!(result)
    d1 = get_grid_data(f1)
    d2 = get_grid_data(f2)
    if isa(result_data, PencilArrays.PencilArray)
        parent(result_data) .= parent(d1) .* parent(d2)
    else
        result_data .= d1 .* d2
    end

    # Truncate the product to remove the aliased high modes. When the caller will
    # sum products in coeff (dot-product terms), keep the result in coeff and skip
    # the backward — the sum's single backward replaces the per-product ones.
    _dealias_truncate_field!(result, factor; final_layout=result_layout)
    return result
end
