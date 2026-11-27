module TarangAMDGPUExt

import Tarang
import AMDGPU
import AbstractFFTs
import MPI
import Random
using LinearAlgebra
using SparseArrays
using AMDGPU: HIPDevice, ROCArray, AnyROCArray, ROCMatrix, ROCVector, ROCBackend
using Tarang: GPU, CPU, ScalarField, RealFourier, ComplexFourier, ChebyshevT,
              launch!, ensure_device!, get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!
import KernelAbstractions
using Tarang: KernelOperation
using KernelAbstractions: @kernel, @index, @Const

const ROCGPU = GPU{HIPDevice}
const DeviceArray = ROCArray
const WrappedDeviceArray = AnyROCArray
const DeviceMatrix = ROCMatrix
const NativeDevice = HIPDevice
const DeviceRuntime = AMDGPU
const DeviceFFT = AbstractFFTs

# rocFFT plans capture the HIP stream. Scratch must also be private to a task
# while its kernels are in flight, even when another task selects the same GPU.
Tarang._gpu_workspace_queue(::ROCGPU) = AMDGPU.stream()
_transform_cache(cache::Dict) = Tarang._gpu_workspace_cache(cache)
_current_device_id() = (AMDGPU.device_id(), AMDGPU.stream())
_dct_cache_device_id(arch::ROCGPU) = (AMDGPU.device_id(arch.device), AMDGPU.stream())

_device_token(dev) = (:amdgpu, AMDGPU.device_id(dev))
_execution_queue() = AMDGPU.stream()

include("amdgpu/architecture.jl")
include("gpu/architecture.jl")
include("gpu/fft.jl")
include("gpu/resample.jl")
# rocFFT's inverse can consume the half spectrum that its forward plan owns.
Tarang._padded_plan_spectrum(p::AMDGPU.rocFFT.rROCFFTPlan, ::ROCGPU, ::Type{T}, shape) where T = p.buffer
Tarang._plan_padded_inverse(p::AMDGPU.rocFFT.rROCFFTPlan, spectrum, n, dims) = inv(p)
function Tarang._padded_inverse!(dst::ROCArray{T}, plan::AbstractFFTs.ScaledPlan{Complex{T},P},
                                 src::ROCArray{Complex{T}}) where {T<:AbstractFloat,P<:AMDGPU.rocFFT.rROCFFTPlan}
    AMDGPU.rocFFT.assert_applicable(plan.p, src, dst)
    AMDGPU.rocFFT.unsafe_execute!(plan.p, src, dst)
    dst .*= plan.scale
    return dst
end
include("gpu/cheb_deriv.jl")
include("gpu/mixed_transforms.jl")
include("amdgpu/solvers.jl")
include("gpu/transpose.jl")
include("amdgpu/distributed.jl")

function __init__()
    Tarang.MatSolvers.register_solver("rocm_dense", ROCDenseLU)
end

end
