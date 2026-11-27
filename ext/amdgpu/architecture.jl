Tarang._amdgpu_functional(::Val{:amdgpu}) = AMDGPU.functional() && AMDGPU.has_rocm_gpu()

function Tarang._gpu_device(::Val{:amdgpu}, device_id::Int)
    Tarang.has_amdgpu() || error("AMDGPU.jl could not initialize a ROCm device")
    devices = AMDGPU.devices()
    0 <= device_id < length(devices) || throw(ArgumentError(
        "AMD device_id=$device_id is outside 0:$(length(devices)-1)"))
    dev = devices[device_id + 1]
    AMDGPU.device!(dev)
    return dev
end

Tarang.device(::ROCGPU) = ROCBackend()
function Tarang._seed_device_rng!(arch::ROCGPU, seed::Integer)
    ensure_device!(arch)
    AMDGPU.seed!(seed % UInt64)
    return nothing
end
