Tarang.array_type(::GPU{NativeDevice}) = DeviceArray
Tarang.array_type(::GPU{NativeDevice}, ::Type{T}) where T = DeviceArray{T}
Tarang.is_gpu_array(::WrappedDeviceArray) = true
_device_storage(a::DeviceArray) = a
_device_storage(a::AbstractArray) = _device_storage(parent(a))
Tarang.architecture(a::WrappedDeviceArray) = GPU(DeviceRuntime.device(_device_storage(a)))
Tarang._device_cache_token(a::WrappedDeviceArray) = _device_token(DeviceRuntime.device(_device_storage(a)))
function _stream_cache_token(a::DeviceArray)
    ensure_device!(Tarang.architecture(a))
    return (Tarang._device_cache_token(a), _execution_queue())
end
Tarang._diff_matmul_cache_token(a::DeviceArray) = _stream_cache_token(a)
Tarang._deriv_fft_cache_token(a::DeviceArray) = _stream_cache_token(a)

function Tarang.ensure_device!(arch::GPU{NativeDevice})
    DeviceRuntime.device() == arch.device || DeviceRuntime.device!(arch.device)
    return nothing
end

function Tarang.synchronize(arch::GPU{NativeDevice})
    ensure_device!(arch)
    DeviceRuntime.synchronize()
    return nothing
end
Tarang.synchronize_device!(arch::GPU{NativeDevice}) = Tarang.synchronize(arch)
Tarang.unsafe_free!(::GPU{NativeDevice}, a::DeviceArray) = DeviceRuntime.unsafe_free!(a)

for op in (:zeros, :ones)
    @eval function Base.$op(arch::GPU{NativeDevice}, ::Type{T}, dims::Integer...) where T
        ensure_device!(arch)
        return DeviceRuntime.$op(T, dims...)
    end
end
Base.similar(arch::GPU{NativeDevice}, a::AbstractArray) = similar(arch, a, eltype(a), size(a))
Base.similar(arch::GPU{NativeDevice}, a::AbstractArray, T::Type) = similar(arch, a, T, size(a))
Base.similar(arch::GPU{NativeDevice}, a::AbstractArray, dims::Dims) = similar(arch, a, eltype(a), dims)
function Base.similar(arch::GPU{NativeDevice}, a::AbstractArray, ::Type{T}, dims::Dims) where T
    ensure_device!(arch)
    return DeviceArray{T}(undef, dims)
end

function Tarang.on_architecture(arch::GPU{NativeDevice}, a::Array)
    ensure_device!(arch)
    return DeviceArray(a)
end
Tarang.on_architecture(arch::GPU{NativeDevice}, a::AbstractArray) =
    Tarang.on_architecture(arch, convert(Array, Tarang.on_architecture(CPU(), a)))

function Tarang.on_architecture(::CPU, a::WrappedDeviceArray)
    source = Tarang.architecture(a)
    previous = DeviceRuntime.device()
    try
        ensure_device!(source)
        return Array(a isa DeviceArray ? a : copy(a))
    finally
        DeviceRuntime.device!(previous)
    end
end
function Tarang.on_architecture(arch::GPU{NativeDevice}, a::WrappedDeviceArray)
    if Tarang.architecture(a).device == arch.device
        ensure_device!(arch)
        return a isa DeviceArray ? a : copy(a)
    end
    return Tarang.on_architecture(arch, convert(Array, Tarang.on_architecture(CPU(), a)))
end
function Tarang.copy_to_device(a::AbstractArray, target::WrappedDeviceArray)
    previous = DeviceRuntime.device()
    try
        result = Tarang.on_architecture(Tarang.architecture(target), a)
        return result === a ? copy(result) : result
    finally
        DeviceRuntime.device!(previous)
    end
end
function Tarang.allocate_like(a::WrappedDeviceArray, ::Type{T}, dims...) where T
    previous = DeviceRuntime.device()
    try
        return zeros(Tarang.architecture(a), T, dims...)
    finally
        DeviceRuntime.device!(previous)
    end
end
Tarang.allocate_like(a::WrappedDeviceArray, dims...) = Tarang.allocate_like(a, eltype(a), dims...)
