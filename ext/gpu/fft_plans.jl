# Keep transformed axes adjacent and first in memory. rocFFT's strided batch
# interface cannot describe the two different batch strides around a middle axis.
struct PermutedFFTPlan{T,Inplace,P,A,B,D} <: AbstractFFTs.Plan{T}
    plan::P
    input::A
    output::B
    perm::D
    inverse_perm::D
end

function _permuted_plan(plan, input, output, perm, inplace)
    return PermutedFFTPlan{eltype(input),inplace,typeof(plan),typeof(input),typeof(output),typeof(perm)}(
        plan, input, output, perm, invperm(perm))
end
Base.size(p::PermutedFFTPlan) = map(i -> size(p.input, i), p.inverse_perm)
function LinearAlgebra.mul!(dest::AbstractArray, p::PermutedFFTPlan, src::AbstractArray)
    permutedims!(p.input, src, p.perm)
    mul!(p.output, p.plan, p.input)
    return permutedims!(dest, p.output, p.inverse_perm)
end
Base.:*(p::PermutedFFTPlan{T,true}, a::AbstractArray{T}) where T = mul!(a, p, a)

# A length-one real axis has only its DC coefficient. MPSGraph's inverse
# real FFT can return zero for this case. Convert the real axis directly and
# transform the remaining axes as complex, including for multidimensional padding.
struct SingletonRealFFTPlan{T,P,A,N} <: AbstractFFTs.Plan{T}
    plan::P
    work::A
    shape::NTuple{N,Int}
end
Base.size(p::SingletonRealFFTPlan) = p.shape
function _singleton_plan(::Type{T}, plan::P, work::A, shape::NTuple{N,Int}) where {T,P,A,N}
    return SingletonRealFFTPlan{T,P,A,N}(plan, work, shape)
end
function LinearAlgebra.mul!(dest::AbstractArray, p::SingletonRealFFTPlan{T}, src::AbstractArray) where {T<:Real}
    dest .= src
    isnothing(p.plan) || p.plan * dest
    return dest
end
function LinearAlgebra.mul!(dest::AbstractArray, p::SingletonRealFFTPlan{T}, src::AbstractArray) where {T<:Complex}
    if isnothing(p.plan)
        dest .= real.(src)
    else
        copyto!(p.work, src)
        p.plan * p.work
        dest .= real.(p.work)
    end
    return dest
end

function _plan_fft_region(a, dims; real_input=false, inplace=false)
    if real_input && size(a, first(dims)) == 1
        others = Tuple(d for d in dims if size(a, d) > 1)
        T = eltype(a)
        work = isempty(others) ? nothing : similar(a, complex(T))
        pf, pb = isempty(others) ? (nothing, nothing) : _plan_fft_region(work, others; inplace=true)
        return _singleton_plan(T, pf, nothing, size(a)), _singleton_plan(complex(T), pb, work, size(a))
    end
    perm = (dims..., (i for i in 1:ndims(a) if i ∉ dims)...)
    if issorted(perm) && inplace
        return AbstractFFTs.plan_fft!(a, dims), AbstractFFTs.plan_ifft!(a, dims)
    end
    staged = !issorted(perm)
    input = staged ? similar(a, map(i -> size(a, i), perm)) : a
    region = ntuple(identity, length(dims))
    forward = real_input ? AbstractFFTs.plan_rfft(input, region) : AbstractFFTs.plan_fft(input, region)
    shape = ntuple(i -> real_input && i == 1 ? Tarang.rfft_len(size(input,1)) : size(input,i), ndims(input))
    output = similar(input, complex(eltype(input)), shape)
    inverse = real_input ? AbstractFFTs.plan_irfft(output, size(input,1), region) : AbstractFFTs.plan_ifft(output, region)
    staged || return forward, inverse
    return _permuted_plan(forward, input, output, perm, inplace),
           _permuted_plan(inverse, output, input, perm, inplace)
end

# Both grid resizes and padded products own their spectra. A leading, nontrivial
# real transform can lend its plan buffer to the caller and consume it on inverse.
function _plan_real_workspace(arch, a, dims, halfshape)
    if size(a, first(dims)) > 1 && all(i -> dims[i] == i, eachindex(dims))
        region = ntuple(identity, length(dims))
        forward = AbstractFFTs.plan_rfft(a, region)
        spectrum = Tarang._padded_plan_spectrum(forward, arch, eltype(a), halfshape)
        inverse = Tarang._plan_padded_inverse(forward, spectrum, size(a, 1), region)
    else
        forward, inverse = _plan_fft_region(a, dims; real_input=true)
        spectrum = similar(a, complex(eltype(a)), halfshape)
    end
    return forward, inverse, spectrum
end
