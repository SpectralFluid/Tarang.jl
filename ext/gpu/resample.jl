# A resize and its reverse share two spectra. Keep the last pair for each
# element type and rank, so a sweep of resolutions has bounded scratch.
const _GRID_RESAMPLE_CACHE = Dict{Tuple,Any}()

function _grid_resample_plan(a::DeviceArray{T,N}, arch) where {T,N}
    dims = ntuple(identity, N)
    shape = size(a)
    if T <: Real
        halfshape = Base.setindex(shape, Tarang.rfft_len(first(shape)), 1)
        forward, inverse, spectrum = _plan_real_workspace(arch, a, dims, halfshape)
    else
        spectrum = similar(a)
        forward = DeviceFFT.plan_fft(a, dims)
        inverse = DeviceFFT.plan_ifft(spectrum, dims)
    end
    return (; shape, forward, inverse, spectrum)
end

function _resize_gpu_grid!(dst, src, output, input, arch)
    mul!(input.spectrum, input.forward, src)
    scale = real(eltype(src))(length(dst) / length(src))
    rdim = eltype(src) <: Real ? 1 : 0
    Tarang.launch!(arch, Tarang._resize_grid_spectrum!, output.spectrum, input.spectrum,
                   size(dst), size(src), scale, rdim; ndrange=size(output.spectrum))
    Tarang._padded_inverse!(dst, output.inverse, output.spectrum)
    return true
end

function Tarang.gpu_resample_grid_data!(dst::DeviceArray{T,N}, src::DeviceArray{T,N},
                                        old_shape::Tuple, new_shape::Tuple) where {T,N}
    arch = Tarang.architecture(src)
    DeviceRuntime.device(dst) == arch.device || throw(ArgumentError("Resampling requires arrays on the same GPU"))
    ensure_device!(arch)
    if size(dst) == size(src)
        copyto!(dst, src)
        return true
    end
    isempty(dst) && return true
    isempty(src) && throw(ArgumentError("Cannot resample an empty grid onto a nonempty grid"))
    cache = Tarang._gpu_workspace_cache(_GRID_RESAMPLE_CACHE)
    key = (arch.device, Tarang._gpu_workspace_queue(arch), T, N)
    pair = get(cache, key, nothing)
    if pair !== nothing
        a, b = pair
        a.shape == size(src) && b.shape == size(dst) && return _resize_gpu_grid!(dst, src, b, a, arch)
        b.shape == size(src) && a.shape == size(dst) && return _resize_gpu_grid!(dst, src, a, b, arch)
    end
    input, output = _grid_resample_plan(src, arch), _grid_resample_plan(dst, arch)
    cache[key] = (input, output)
    return _resize_gpu_grid!(dst, src, output, input, arch)
end
