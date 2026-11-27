module TarangMetalExt

import AbstractFFTs, KernelAbstractions, Metal, Tarang
using LinearAlgebra
using KernelAbstractions: @kernel, @index, @Const
using Tarang: GPU, CPU, ScalarField, RealFourier, ComplexFourier, ChebyshevT,
              KernelOperation, launch!, ensure_device!, get_grid_data, get_coeff_data,
              set_grid_data!, set_coeff_data!

const NativeDevice = Metal.MTL.MTLDevice
const MetalGPU = GPU{NativeDevice}
const DeviceArray = Metal.MtlArray
const WrappedDeviceArray = Metal.WrappedMtlArray
const DeviceMatrix = Metal.MtlMatrix
const DeviceRuntime = Metal
const DeviceFFT = AbstractFFTs

Tarang._metal_functional(::Val{:metal}) = Metal.functional()
function Tarang._gpu_device(::Val{:metal}, device_id::Int)
    Tarang.has_metal() || error("Metal.jl could not initialize an Apple GPU")
    devices = Metal.devices()
    0 <= device_id < length(devices) || throw(ArgumentError(
        "Metal device_id=$device_id is outside 0:$(length(devices)-1)"))
    dev = devices[device_id + 1]
    Metal.device!(dev)
    return dev
end

Tarang.device(::MetalGPU) = Metal.MetalBackend()
Tarang._default_field_dtype(::MetalGPU) = Float32
function Tarang._validate_gpu_field_dtype(::MetalGPU, T::Type)
    T in (Float32, ComplexF32) || throw(ArgumentError(
        "Metal spectral fields require Float32 or ComplexF32, got $T"))
end
function Tarang._validate_distributor_size(::MetalGPU, ranks::Int)
    ranks == 1 || throw(ArgumentError("Metal fields require a single MPI rank"))
end
Tarang._device_aware_mpi(::MetalGPU) = false
Tarang._verify_device_aware_mpi(::MetalGPU) = false
function Tarang._seed_device_rng!(arch::MetalGPU, seed::Integer)
    ensure_device!(arch)
    Metal.seed!(seed % UInt64)
    return nothing
end

_device_token(dev) = (:metal, dev.registryID)
_execution_queue() = Metal.global_queue(Metal.device())
_current_device_id() = (_device_token(Metal.device()), _execution_queue())
_dct_cache_device_id(arch::MetalGPU) = (_device_token(arch.device), Metal.global_queue(arch.device))
Tarang._gpu_workspace_queue(arch::MetalGPU) = Metal.global_queue(arch.device)
_transform_cache(cache::Dict) = Tarang._gpu_workspace_cache(cache)

include("gpu/architecture.jl")
include("gpu/fft.jl")
include("gpu/resample.jl")
include("gpu/cheb_deriv.jl")
include("gpu/mixed_transforms.jl")

include("metal/solvers.jl")

function __init__()
    Tarang.MatSolvers.register_solver("metal_dense", MetalDenseLU)
end

end
