function Tarang._amdgpu_functional(::Val{:amdgpu})
    AMDGPU.functional() || return false
    # With ROCm userspace but no usable device (login nodes, containers without
    # /dev/kfd), the HIP device count query throws instead of returning zero.
    try
        return AMDGPU.has_rocm_gpu()
    catch err
        err isa AMDGPU.HIP.HIPError || rethrow()
        @debug "ROCm device query failed; reporting no AMD GPU" exception=err
        return false
    end
end

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

# rocRAND has no complex normals and AMDGPU's `randn!` errors for complex arrays.
# Draw the real and imaginary parts as normals with variance 1/2 each, which is
# Julia's `randn` convention for complex numbers (unit total variance).
function Tarang._randn_device!(data::ROCArray{Complex{T}}) where {T<:AbstractFloat}
    parts = reinterpret(T, data)
    Random.randn!(parts)
    parts .*= inv(sqrt(T(2)))
    return data
end
