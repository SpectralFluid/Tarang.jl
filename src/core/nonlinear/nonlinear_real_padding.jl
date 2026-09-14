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
    plan_forward::PF
    plan_backward::PB
    plan_spec_forward::SF
    plan_spec_backward::SB
    arch::AbstractArchitecture
    cached_operand::Any
end

# The workspace owns these input spectra and permits inverse FFTs to destroy
# them. CUDA specializes these hooks to avoid its normal input-preservation copy.
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
    dims = findall(b -> b isa Union{RealFourier,ComplexFourier}, collect(bases))
    isempty(dims) && return nothing
    shape = map(b -> b.meta.size, bases)
    (length(shape) > 3 || minimum(shape[d] for d in dims) <= 4) && return nothing
    padded = ntuple(length(shape)) do d
        d in dims ? 2cld(ceil(Int, _axis_dealias_factor(bases[d], ev.dealiasing_factor)*shape[d]), 2) : shape[d]
    end
    rdim = first(dims)
    half(s) = ntuple(d -> d == rdim ? s[d] ÷ 2 + 1 : s[d], length(s))
    arch = ev.dist.architecture
    a, b = zeros(arch, T, padded...), zeros(arch, T, padded...)
    original = zeros(arch, T, shape...)
    pf, pb, sf, sb, ps, s = _real_padded_fft_plans(arch, a, original, half(padded), half(shape), dims)
    ws = RealPaddedDealiasingWorkspace(shape, padded, dims, a, b, ps, s, pf, pb, sf, sb, arch, nothing)
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

# Fold all Nyquist images. Negative reduced-axis frequencies are implicit:
# reflecting that axis requires conjugating and reflecting EVERY Fourier axis.
@kernel function _truncate_real_spectrum_kernel!(out, @Const(src), shape, padded, fourier, rdim)
    I = @index(Global, NTuple)
    total = zero(eltype(out))
    for mask in 0:((1 << length(shape)) - 1)
        K = I
        valid = true
        for d in 1:length(shape)
            negative = !iszero(mask & (1 << (d - 1)))
            if fourier[d]
                k = I[d] - 1
                if d != rdim && k > shape[d] ÷ 2
                    k -= shape[d]
                end
                split = iseven(shape[d]) && k == shape[d] ÷ 2 && padded[d] > shape[d]
                valid &= !negative || split
                K = Base.setindex(K, negative ? -k : k, d)
            else
                valid &= !negative
            end
        end
        if valid
            reflect = K[rdim] < 0
            J = I
            for d in 1:length(shape)
                if fourier[d]
                    k = reflect ? -K[d] : K[d]
                    J = Base.setindex(J, k < 0 ? padded[d] + k + 1 : k + 1, d)
                end
            end
            @inbounds value = src[J...]
            total += reflect ? conj(value) : value
        end
    end
    @inbounds out[I...] = total
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
    _padded_inverse!(dst, ws.plan_spec_backward, ws.spectrum)
    dst .*= scale
    result.current_layout = :g
    ensure_layout!(result, result_layout)
    return result
end
