# ============================================================================
# GPU DCT-I / FFT plan support for Chebyshev and mixed bases
# ============================================================================
#
# Tarang's Chebyshev basis is DCT-I (REDFT00) on the Gauss-Lobatto grid. The
# on-device DCT-I lives in `cheb_deriv.jl` (`gpu_dct1_along_dim!`), built from
# the same verified symmetric-extension + rfft kernels as the Chebyshev
# derivative.
#
# This file used to also carry a complete second, DCT-**II**/III implementation
# (`GPUDCTPlan`, `OptimizedGPUDCTPlan`, `GPUDCTPlanDim`, the even-odd reorder
# kernels and their 1-D/3-D twiddle kernels, ~700 lines). It was the WRONG
# convention for this framework's Chebyshev basis, it had no caller in `src`,
# and it sat one autocomplete away from the DCT-I entry points — the exact
# mix-up that produced a real bug once already (see the DCT-II-vs-DCT-I fix in
# the project history). It has been removed. What remains here is the
# per-dimension FFT plan the mixed Fourier x Chebyshev driver actually uses,
# plus the shared plan/scratch caches.

# ============================================================================
# Dimension-by-Dimension FFT Plans (for mixed basis transforms)
# ============================================================================

"""
    GPUFFTPlanDim

GPU FFT plan for a specific dimension of a multi-dimensional array.
Used for mixed Fourier-Chebyshev transforms where we need separate
FFT along Fourier dimensions and DCT along Chebyshev dimensions.
"""
struct GPUFFTPlanDim{P, IP}
    plan::P
    iplan::IP
    full_size::Tuple{Vararg{Int}}
    transform_dim::Int
    is_real::Bool
end

"""
    plan_gpu_fft_dim(arch::GPU{CuDevice}, full_size::Tuple, T::Type, dim::Int; real_input::Bool=false)

Create a GPU FFT plan for a specific dimension of a multi-dimensional array.

# Arguments
- `arch`: GPU architecture
- `full_size`: Full array dimensions
- `T`: Element type
- `dim`: Dimension along which to transform (1-indexed)
- `real_input`: If true, create R2C/C2R plans for this dimension
"""
function plan_gpu_fft_dim(arch::GPU{CuDevice}, full_size::Tuple, T::Type, dim::Int; real_input::Bool=false)
    ensure_device!(arch)

    complex_T = T <: Complex ? T : Complex{T}
    ndims = length(full_size)

    @assert 1 <= dim <= ndims "Dimension $dim out of range for $(ndims)D array"

    if real_input
        # Real-to-complex FFT along specified dimension
        dummy_in = CUDA.zeros(T, full_size...)
        plan = CUFFT.plan_rfft(dummy_in, (dim,))
        return GPUFFTPlanDim(plan, inv(plan), full_size, dim, true)
    else
        # Complex-to-complex FFT along specified dimension
        dummy = CUDA.zeros(complex_T, full_size...)
        plan = CUFFT.plan_fft(dummy, (dim,))
        iplan = CUFFT.plan_ifft(dummy, (dim,))

        return GPUFFTPlanDim(plan, iplan, full_size, dim, false)
    end
end

# Fallback
plan_gpu_fft_dim(arch::GPU, full_size::Tuple, T::Type, dim::Int; real_input::Bool=false) =
    plan_gpu_fft_dim(GPU{CuDevice}(CUDA.device()), full_size, T, dim; real_input=real_input)

"""
    gpu_fft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim)

Execute forward FFT along a specific dimension.
"""
function gpu_fft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim)
    mul!(output, plan.plan, input)
    return output
end

"""
    gpu_ifft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim; destroy_input=false)

Execute inverse FFT along a specific dimension.

The cuFFT inverse preserves the input by default. Mixed transforms pass
`destroy_input=true` when their scratch spectrum can be consumed directly.
"""
function gpu_ifft_dim!(output::CuArray, input::CuArray, plan::GPUFFTPlanDim;
                       destroy_input::Bool=false)
    if plan.is_real && destroy_input
        return Tarang._padded_inverse!(output, plan.iplan, input)
    end
    return mul!(output, plan.iplan, input)
end

# ── Shared plan-cache lock and device identity ──────────────────────────────
# The DCT-II plan caches that used to live here went with the DCT-II
# implementation; the lock and the device-id helper stay because the scratch
# cache below (and `get_gpu_fft_plan`'s sibling logic) still need them.

const _GPU_DCT_PLAN_CACHE_LOCK = ReentrantLock()

# Key caches by the device the plan is actually built for. When `arch.device` is
# a concrete `CuDevice` use it directly (direct multi-GPU calls may target a
# device other than the current one); otherwise fall back to the current device.
# Mirrors `get_gpu_fft_plan`'s `GPU{CuDevice}` vs generic `GPU` split so the
# scratch and FFT caches agree on device identity.
_dct_cache_device_id(arch::GPU{CuDevice}) = CUDA.deviceid(arch.device)
_dct_cache_device_id(arch::GPU) = _current_device_id()

# ── Reusable scratch buffers for the complex / multi-dim DCT paths ───────────
# The pure-Chebyshev transform branches allocated fresh CuArrays every call:
# `real.(x)`/`imag.(x)` splits for complex data (DCT kernels need real input),
# and a new output array per dimension in the multi-dim ping-pong. For a field
# transformed every timestep this is 4–8 device allocations per transform. Since
# the DCT preserves array shape, we cache `count` reusable buffers per
# (device, shape, eltype, count) and the multi-dim loop alternates between two of
# them. Safe for serial single-GPU use: each transform copies its result into the
# field's coeff/grid array before returning, so the scratch is free for the next
# call (same assumption the plan work-buffer reuse already makes).
const _GPU_DCT_SCRATCH_CACHE = Dict{Tuple, Any}()

"""Get `count` reusable GPU buffers; `purpose` separates independent scratch users."""
function get_gpu_dct_scratch(arch::GPU, shape::NTuple{N,Int}, ::Type{T}, count::Int; purpose::Symbol=:default) where {N,T}
    # Preserve the original key for existing callers and cache inspection.
    base_key = (_dct_cache_device_id(arch), shape, T, count)
    key = purpose === :default ? base_key : (base_key..., purpose)
    buffers = lock(_GPU_DCT_PLAN_CACHE_LOCK) do
        get!(() -> CuArray{T,N}[CUDA.zeros(T, shape...) for _ in 1:count],
             _transform_cache(_GPU_DCT_SCRATCH_CACHE), key)
    end
    return buffers::Vector{CuArray{T,N}}   # function barrier: type-stable downstream
end

"""Clear all cached GPU DCT scratch buffers (thread-safe)."""
function clear_gpu_dct_scratch_cache!()
    lock(_GPU_DCT_PLAN_CACHE_LOCK) do
        empty!(_transform_cache(_GPU_DCT_SCRATCH_CACHE))
    end
end
