"""Real grids and half spectra for serial real-valued padded products.

One shared half spectrum is consumed by each C2R transform before reuse. The two
real padded grids hold operands, with the product overwriting the first. Buffers
and plans belong to the evaluator's existing single-task execution context.
"""
mutable struct RealPaddedDealiasingWorkspace{R,C,PF,PB,SF,SB} <: AbstractNonlinearTransformConfig
    original_shape::Tuple{Vararg{Int}}
    padded_shape::Tuple{Vararg{Int}}
    fourier_dims::Vector{Int}
    padded1::R
    padded2::R
    padded_spectrum::C
    spectrum::C
    extended_grid::Union{Nothing,R}
    plan_forward::PF
    plan_backward::PB
    plan_spec_forward::SF
    plan_spec_backward::SB
    arch::AbstractArchitecture
    cached_operand::Any
end

# The workspace owns these input spectra and permits inverse FFTs to destroy
# them. cuFFT and rocFFT can skip their usual input-preservation copy.
_plan_padded_inverse(forward, spectrum, n, dims) = plan_irfft(spectrum, n, dims)
_padded_inverse!(dst, plan, src) = mul!(dst, plan, src)
_padded_plan_spectrum(forward, arch, ::Type{T}, shape) where T = zeros(arch, Complex{T}, shape...)

_padded_architecture_key(::CPU) = false
_padded_architecture_key(::GPU) = true

function _real_padded_fft_plans(arch::CPU, a, original, padhalf, half, dims)
    T = eltype(a)
    ps, s = zeros(arch, Complex{T}, padhalf...), zeros(arch, Complex{T}, half...)
    flags = FFTW.ESTIMATE | FFTW.UNALIGNED
    pf = FFTW.plan_rfft(a, dims; flags)
    pb = FFTW.plan_irfft(ps, size(a, first(dims)), dims; flags)
    sf = FFTW.plan_rfft(original, dims; flags)
    sb = FFTW.plan_irfft(s, size(original, first(dims)), dims; flags)
    return pf, pb, sf, sb, ps, s
end

function _real_padded_fft_plans(arch::GPU, a, original, padhalf, half, dims)
    T = eltype(a)
    pf, sf = plan_rfft(a, dims), plan_rfft(original, dims)
    # Reuse any spectrum already owned by a CUDA R2C plan.
    ps = _padded_plan_spectrum(pf, arch, T, padhalf)
    s = _padded_plan_spectrum(sf, arch, T, half)
    pb = _plan_padded_inverse(pf, ps, size(a, first(dims)), dims)
    sb = _plan_padded_inverse(sf, s, size(original, first(dims)), dims)
    return pf, pb, sf, sb, ps, s
end

function _get_real_padded_workspace!(ev, bases, ::Type{T}) where T
    key = (hash(bases), T, hash(:real_padded), _padded_architecture_key(ev.dist.architecture))
    haskey(ev.pencil_transforms, key) && return ev.pencil_transforms[key]
    shape, padded, dims, cosine_dims = _product_fft_layout(bases, ev.dealiasing_factor)
    isempty(dims) && return nothing
    length(shape) > 3 && return nothing
    rdim = first(dims)
    half(s) = ntuple(d -> d == rdim ? s[d] ÷ 2 + 1 : s[d], length(s))
    arch = ev.dist.architecture
    a, b = zeros(arch, T, padded...), zeros(arch, T, padded...)
    original = zeros(arch, T, shape...)
    pf, pb, sf, sb, ps, s = _real_padded_fft_plans(arch, a, original, half(padded), half(shape), dims)
    extended = isempty(cosine_dims) ? nothing : original
    ws = RealPaddedDealiasingWorkspace(shape, padded, dims, a, b, ps, s, extended, pf, pb, sf, sb, arch, nothing)
    ev.pencil_transforms[key] = ws
    return ws
end

# Each destination coefficient gathers one source, or writes the spectral gap.
# Even-N Nyquist planes split by 1/2 per padded axis, including intersections.
@kernel function _pad_real_spectrum_kernel!(out, @Const(src), shape, padded, fourier, rdim)
    I = @index(Global, NTuple)
    J = I
    weight = one(real(eltype(out)))
    valid = true
    for d in 1:length(shape)
        if fourier[d]
            k = I[d] - 1
            if d != rdim && k > padded[d] ÷ 2
                k -= padded[d]
            end
            valid &= abs(k) <= shape[d] ÷ 2
            nyq = iseven(shape[d]) && abs(k) == shape[d] ÷ 2
            if nyq && padded[d] > shape[d]
                weight *= real(eltype(out))(0.5)
            end
            j = nyq ? shape[d] ÷ 2 + 1 : (k < 0 ? shape[d] + k + 1 : k + 1)
            J = Base.setindex(J, j, d)
        end
    end
    @inbounds out[I...] = valid ? weight * src[J...] : zero(eltype(out))
end

# Gather a coefficient while folding downsampled Nyquist planes. A nonzero
# rdim identifies the reduced axis of a real FFT, zero denotes a full spectrum.
# When a grid axis grows, its old even Nyquist is dropped by the grid-resize
# convention. Nonlinear padding has a separate splitting convention.
@inline function _resized_fourier_coefficient(src, I, dstshape, srcshape, fourier, rdim)
    modes = I
    folds = 0
    for d in 1:length(I)
        fourier[d] || continue
        k = I[d] - 1
        if d != rdim && k > dstshape[d] ÷ 2
            k -= dstshape[d]
        end
        if dstshape[d] > srcshape[d] && abs(k) > (srcshape[d] - 1) ÷ 2
            return zero(eltype(src))
        end
        if srcshape[d] > dstshape[d] && iseven(dstshape[d]) && k == dstshape[d] ÷ 2
            folds |= 1 << (d - 1)
        end
        modes = Base.setindex(modes, k, d)
    end

    total = zero(eltype(src))
    images = folds
    while true
        # A negative frequency on the reduced axis lives in the conjugate
        # half. Reflect every Fourier axis to address it in stored memory.
        reflect = rdim != 0 && !iszero(images & (1 << (rdim - 1)))
        J = I
        for d in 1:length(I)
            fourier[d] || continue
            k = iszero(images & (1 << (d - 1))) ? modes[d] : -modes[d]
            k = reflect ? -k : k
            J = Base.setindex(J, k < 0 ? srcshape[d] + k + 1 : k + 1, d)
        end
        @inbounds value = src[J...]
        total += reflect ? conj(value) : value
        iszero(images) && break
        images = (images - 1) & folds
    end
    return total
end

@kernel function _truncate_real_spectrum_kernel!(out, @Const(src), shape, padded, fourier, rdim)
    I = @index(Global, NTuple)
    @inbounds out[I...] = _resized_fourier_coefficient(src, I, shape, padded, fourier, rdim)
end

@kernel function _resize_grid_spectrum!(out, @Const(src), dstshape, srcshape, scale, rdim)
    I = @index(Global, NTuple)
    fourier = ntuple(_ -> true, Val(length(I)))
    @inbounds out[I...] = scale * _resized_fourier_coefficient(src, I, dstshape, srcshape, fourier, rdim)
end

function _real_pad!(ws, src)
    fourier = ntuple(d -> d in ws.fourier_dims, length(ws.original_shape))
    launch!(ws.arch, _pad_real_spectrum_kernel!, ws.padded_spectrum, src,
            ws.original_shape, ws.padded_shape, fourier, first(ws.fourier_dims);
            ndrange=size(ws.padded_spectrum))
    return ws.padded_spectrum
end

function _real_truncate!(ws, dst)
    fourier = ntuple(d -> d in ws.fourier_dims, length(ws.original_shape))
    launch!(ws.arch, _truncate_real_spectrum_kernel!, dst, ws.padded_spectrum,
            ws.original_shape, ws.padded_shape, fourier, first(ws.fourier_dims);
            ndrange=size(dst))
    return dst
end

# irfft projects the self-conjugate boundary planes onto a real spectrum. Do
# the same when bypassing the grid round trip, including user-written spectra.
@kernel function _project_real_spectrum_kernel!(dst, @Const(src), shape, rdim)
    I = @index(Global, NTuple)
    J = ntuple(d -> d == rdim || I[d] == 1 ? I[d] : shape[d] + 2 - I[d], length(shape))
    boundary = I[rdim] == 1 || (iseven(shape[rdim]) && I[rdim] == shape[rdim] ÷ 2 + 1)
    @inbounds dst[I...] = boundary ? (src[I...] + conj(src[J...])) / 2 : src[I...]
end

function _real_operand_spectrum!(ws, field)
    if field.current_layout === :c && _padded_coefficient_compatible(field, size(ws.spectrum)) &&
       architecture(get_local_data(get_coeff_data(field))) == ws.arch
        launch!(ws.arch, _project_real_spectrum_kernel!, ws.spectrum,
                get_local_data(get_coeff_data(field)), ws.original_shape, first(ws.fourier_dims);
                ndrange=size(ws.spectrum))
    else
        ensure_layout!(field, :g)
        src = on_architecture(ws.arch, get_local_data(get_grid_data(field)))
        if ws.extended_grid !== nothing
            _extend_lobatto_grid!(ws.arch, ws.extended_grid, src)
            src = ws.extended_grid
        end
        mul!(ws.spectrum, ws.plan_spec_forward, src)
    end
    return ws.spectrum
end

function evaluate_padded_multiply(a::ScalarField, b::ScalarField, ev::NonlinearEvaluator,
                                  ws::RealPaddedDealiasingWorkspace;
                                  destination::Union{Nothing,ScalarField}=nothing,
                                  result_layout::Symbol=:g, operand_key=nothing)
    _real_operand_spectrum!(ws, a)
    _real_pad!(ws, ws.spectrum)
    _padded_inverse!(ws.padded1, ws.plan_backward, ws.padded_spectrum)
    if operand_key === nothing || ws.cached_operand !== operand_key
        ws.cached_operand = nothing
        _real_operand_spectrum!(ws, b)
        _real_pad!(ws, ws.spectrum)
        _padded_inverse!(ws.padded2, ws.plan_backward, ws.padded_spectrum)
        ws.cached_operand = operand_key
    end
    ws.padded1 .*= ws.padded2
    mul!(ws.padded_spectrum, ws.plan_forward, ws.padded1)
    _real_truncate!(ws, ws.spectrum)
    result = destination === nothing ? _checkout_nl_result!(ev, a) : destination
    T = result.dtype
    scale = prod(T(ws.padded_shape[d])/T(ws.original_shape[d]) for d in ws.fourier_dims)
    if result_layout === :c && _padded_coefficient_compatible(result, size(ws.spectrum))
        dst = get_local_data(get_coeff_data(result))
        @. dst = ws.spectrum * scale
        result.current_layout = :c
        return result
    end
    dst = get_local_data(grid_data!(result))
    grid = something(ws.extended_grid, dst)
    _padded_inverse!(grid, ws.plan_spec_backward, ws.spectrum)
    values = _lobatto_grid_view(grid, dst)
    @. dst = values * scale
    result.current_layout = :g
    ensure_layout!(result, result_layout)
    return result
end
