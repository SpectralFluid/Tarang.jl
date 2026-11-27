# ============================================================================
# GPU Transform Functions
# ============================================================================

# GPUTransformCache and GPU_TRANSFORM_CACHE are defined in transforms.jl

mutable struct FFT1DPlanCache
    plans::Dict{Tuple, Any}
    lock::ReentrantLock

    FFT1DPlanCache() = new(Dict{Tuple, Any}(), ReentrantLock())
end

const FFT_1D_PLAN_CACHE = FFT1DPlanCache()

function get_fft_1d_plan(size::Tuple, dim::Int, T::Type; inverse::Bool=false, device_id::Int=_current_device_id())
    # Include device ID in cache key for multi-GPU correctness
    key = (device_id, size, dim, T, inverse)
    return lock(FFT_1D_PLAN_CACHE.lock) do
        get!(_transform_cache(FFT_1D_PLAN_CACHE.plans), key) do
            # Ensure plan is created on the correct device
            prev_device = CUDA.device()
            CUDA.device!(CuDevice(device_id))
            try
                dummy = CUDA.zeros(T, size...)
                inverse ? CUFFT.plan_ifft(dummy, (dim,)) : CUFFT.plan_fft(dummy, (dim,))
            finally
                CUDA.device!(prev_device)
            end
        end
    end
end

# get_gpu_fft_plan and clear_gpu_transform_cache! are defined in transforms.jl

# ============================================================================
# GPU-aware Field Operations
# ============================================================================

"""
    allocate_gpu_data(arch::GPU{CuDevice}, dtype::Type, shape::Tuple)

Allocate array on the specific GPU device stored in the architecture.
Ensures correct device context for multi-GPU support.
"""
function allocate_gpu_data(arch::GPU{CuDevice}, dtype::Type, shape::Tuple)
    ensure_device!(arch)
    return CUDA.zeros(dtype, shape...)
end

# Fallback for generic GPU
function allocate_gpu_data(::GPU, dtype::Type, shape::Tuple)
    return CUDA.zeros(dtype, shape...)
end

# ============================================================================
# GPU-specific Array Allocation (zeros/ones/similar) with Device Context
# These are internal helpers - users should use Base.zeros(arch, ...) etc.
# ============================================================================
#
# The `_ext_` prefix is load-bearing. These are extension-LOCAL functions, and
# the bottom of this same file defines `Tarang._gpu_zeros` — a DIFFERENT generic
# function that happens to live in another module. When both were spelled
# `_gpu_zeros`, an unqualified call inside the extension silently resolved to
# whichever one you did not mean, with no error at either definition site.

# Internal helper functions (not exported, use Base.zeros/ones/similar instead)
function _ext_gpu_zeros(arch::GPU{CuDevice}, T::Type, dims...)
    ensure_device!(arch)
    return CUDA.zeros(T, dims...)
end

_ext_gpu_zeros(::GPU, T::Type, dims...) = CUDA.zeros(T, dims...)

function _ext_gpu_zeros(arr::CuArray, T::Type, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CUDA.zeros(T, dims...)
    finally
        # try/finally: an exception mid-allocation (e.g. OOM) must not strand the
        # task on the wrong device (matches Tarang.allocate_like below).
        CUDA.device!(prev_device)
    end
end

_ext_gpu_zeros(arr::CuArray, dims...) = _ext_gpu_zeros(arr, eltype(arr), dims...)

function _ext_gpu_ones(arch::GPU{CuDevice}, T::Type, dims...)
    ensure_device!(arch)
    return CUDA.ones(T, dims...)
end

_ext_gpu_ones(::GPU, T::Type, dims...) = CUDA.ones(T, dims...)

function _ext_gpu_ones(arr::CuArray, T::Type, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CUDA.ones(T, dims...)
    finally
        CUDA.device!(prev_device)
    end
end

_ext_gpu_ones(arr::CuArray, dims...) = _ext_gpu_ones(arr, eltype(arr), dims...)

function _ext_gpu_similar(arr::CuArray, T::Type, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CuArray{T}(undef, dims...)
    finally
        CUDA.device!(prev_device)
    end
end

_ext_gpu_similar(arr::CuArray, dims...) = _ext_gpu_similar(arr, eltype(arr), dims...)
_ext_gpu_similar(arr::CuArray) = _ext_gpu_similar(arr, eltype(arr), size(arr)...)

function _ext_gpu_fill(arch::GPU{CuDevice}, val, dims...)
    ensure_device!(arch)
    return CUDA.fill(val, dims...)
end

_ext_gpu_fill(::GPU, val, dims...) = CUDA.fill(val, dims...)

function _ext_gpu_fill(arr::CuArray, val, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(arr))
        return CUDA.fill(val, dims...)
    finally
        CUDA.device!(prev_device)
    end
end

# ============================================================================
# Data Transfer Helpers (attached to Tarang module)
# Users should use Tarang.on_architecture for the primary API
# ============================================================================

# Stub functions - these will be defined if Tarang exports them
# For now, keep as internal helpers that work via on_architecture

"""
    _to_gpu(a::Array)

Internal: Move array to GPU (current device).
Users should use `Tarang.on_architecture(GPU(), array)` instead.
"""
_to_gpu(a::Array) = CuArray(a)

"""
    _to_gpu(arch::GPU{CuDevice}, a::Array)

Internal: Move array to specific GPU device.
Users should use `Tarang.on_architecture(arch, array)` instead.
"""
function _to_gpu(arch::GPU{CuDevice}, a::Array)
    ensure_device!(arch)
    return CuArray(a)
end

"""
    _to_cpu(a::CuArray)

Internal: Move array to CPU.
Users should use `Tarang.on_architecture(CPU(), array)` instead.
"""
_to_cpu(a::CuArray) = Array(a)

# Internal convenience aliases - NOT exported
# Users should use Tarang.on_architecture(GPU(), array) or Tarang.on_architecture(CPU(), array) instead
to_gpu(a::Array) = _to_gpu(a)
to_gpu(arch::GPU{CuDevice}, a::Array) = _to_gpu(arch, a)
to_cpu(a::CuArray) = _to_cpu(a)

# ============================================================================
# Multi-dimensional FFT Kernels for Spectral Methods
# ============================================================================

"""
2D FFT forward transform on GPU
"""
function gpu_fft_2d_forward!(output::CuArray{Complex{T}}, input::CuArray{T}) where {T<:AbstractFloat}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=true)
    gpu_forward_fft!(output, input, plan)
    return output
end

"""
2D FFT backward transform on GPU
"""
function gpu_fft_2d_backward!(output::CuArray{T}, input::CuArray{Complex{T}}, n::Int) where {T<:AbstractFloat}
    out_size = (n, size(input)[2:end]...)
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, out_size, T; real_input=true)
    gpu_backward_fft!(output, input, plan)
    return output
end

"""
3D FFT forward transform on GPU (complex)
"""
function gpu_fft_3d_forward!(output::CuArray{T}, input::CuArray{T}) where {T<:Complex}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=false)
    gpu_forward_fft!(output, input, plan)
    return output
end

"""
3D FFT backward transform on GPU (complex)
"""
function gpu_fft_3d_backward!(output::CuArray{T}, input::CuArray{T}) where {T<:Complex}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=false)
    gpu_backward_fft!(output, input, plan)
    return output
end

# ============================================================================
# GPU Dealiasing Operations
# ============================================================================

"""
    _dealias_kmax(N, cutoff) -> Int

Max retained |mode| for an axis of a two-sided full FFT spectrum, using the SAME
rule as the CPU solver's `Tarang._axis_dealias_cutoff(basis, factor)` with
`factor = 1/keep_fraction`:

    kmax = min(floor(N / (2·factor)), (N − 1) ÷ 3)

The `(N − 1) ÷ 3` cap enforces `3·kmax < N`, so quadratic products (modes up to
`2·kmax`) cannot alias back into `[−kmax, kmax]` — see dealiasing.jl.

`cutoff` accepts EITHER convention, disambiguated by magnitude (they are
numerically disjoint — both mean "dealiasing off" only at exactly 1):
- a **retained fraction** ≤ 1 (this function's native units, e.g. `2/3`), or
- the codebase-wide `dealias` **padding factor** ≥ 1 (e.g. `3/2`), mapped through
  `1/cutoff` to the retained fraction.

This dual acceptance kills the silent no-op that passing the padding factor
`3/2` (or any value > 1) to a keep-fraction API would otherwise cause. `cutoff == 1`
disables dealiasing (all modes kept); we return `N` (no index has |mode| > N, so
nothing is zeroed), mirroring `_axis_dealias_cutoff` returning `nothing`.
"""
@inline function _dealias_kmax(N::Int, cutoff::Float64)
    keep_fraction = cutoff <= 1.0 ? cutoff : 1.0 / cutoff
    keep_fraction >= 1 && return N   # cutoff == 1 → keep every mode (dealias off)
    factor = 1.0 / keep_fraction
    return min(floor(Int, N / (2 * factor)), (N - 1) ÷ 3)
end

"""
Dealiasing kernel for 2D arrays (two-sided full FFT spectrum on BOTH axes):
zero every mode with |k| > kmax on either axis.

Index ↔ mode map per axis of length n: index i is mode i−1 on the positive side
or −(n−i+1) on the negative side. An entry is killed only when BOTH
interpretations exceed kmax, so the retained band is |k| ≤ kmax on each side
(conjugate-symmetric; DC at i==1 is always kept since kmax ≥ 0). The even-N
Nyquist index n÷2+1 has |k| = n/2 > kmax (kmax ≤ (n−1)÷3) and is always zeroed.
"""
@kernel function dealiasing_2d_kernel!(data, kmax_x::Int, kmax_y::Int, nx::Int, ny::Int)
    idx = @index(Global)
    j = ((idx - 1) ÷ nx) + 1
    i = ((idx - 1) % nx) + 1
    kill_x = (i - 1 > kmax_x) && (nx - i + 1 > kmax_x)
    kill_y = (j - 1 > kmax_y) && (ny - j + 1 > kmax_y)
    @inbounds data[i, j] = ifelse(kill_x || kill_y, zero(eltype(data)), data[i, j])
end

"""
Dealiasing kernel for 3D arrays (two-sided full FFT spectrum on all axes):
zero every mode with |k| > kmax on any axis. See dealiasing_2d_kernel! for the
index ↔ mode map.
"""
@kernel function dealiasing_3d_kernel!(data, kmax_x::Int, kmax_y::Int, kmax_z::Int,
                                        nx::Int, ny::Int, nz::Int)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    j = (((idx - 1) ÷ nx) % ny) + 1
    k = ((idx - 1) ÷ (nx * ny)) + 1
    kill_x = (i - 1 > kmax_x) && (nx - i + 1 > kmax_x)
    kill_y = (j - 1 > kmax_y) && (ny - j + 1 > kmax_y)
    kill_z = (k - 1 > kmax_z) && (nz - k + 1 > kmax_z)
    @inbounds data[i, j, k] = ifelse(kill_x || kill_y || kill_z, zero(eltype(data)), data[i, j, k])
end

"""
    create_dealiasing_mask_gpu(shape::Tuple, cutoff::Float64=2.0/3.0; eltype::Type=Float64)

Create a dealiasing mask on GPU: entries with |k| ≤ kmax on every axis are one,
all others zero, where kmax per axis follows the solver's
`Tarang._axis_dealias_cutoff` rule (`min(floor(N·cutoff/2), (N−1)÷3)` for the
default `cutoff = 2/3` retained fraction). `cutoff` accepts EITHER a retained
fraction ≤ 1 (e.g. `2/3`) OR the `dealias` padding factor ≥ 1 (e.g. `3/2`); see
`_dealias_kmax`. `cutoff == 1` disables dealiasing.

Every axis is treated as a TWO-SIDED full FFT spectrum (layout: DC, positive
freqs, [Nyquist,] negative freqs). rfft half-spectrum layouts are NOT supported
— a half-spectrum axis is indistinguishable from a short full spectrum by shape
alone, so it cannot be detected here; do not pass rfft output to this helper.

Supports 1–3 dimensions; other ranks throw an `ArgumentError`. `eltype` selects
the mask element type (default `Float64`).
"""
function create_dealiasing_mask_gpu(shape::Tuple, cutoff::Float64=2.0/3.0; eltype::Type=Float64)
    T = eltype
    nd = length(shape)
    1 <= nd <= 3 || throw(ArgumentError(
        "create_dealiasing_mask_gpu supports 1-, 2-, or 3-dimensional shapes, got ndims=$nd"))

    mask = CUDA.ones(T, shape...)
    kmaxs = map(n -> _dealias_kmax(n, cutoff), shape)

    if nd == 2
        nx, ny = shape
        arch = Tarang.architecture(mask)
        launch!(arch, dealiasing_2d_kernel!, mask, kmaxs[1], kmaxs[2], nx, ny;
                ndrange=nx*ny)
    elseif nd == 3
        nx, ny, nz = shape
        arch = Tarang.architecture(mask)
        launch!(arch, dealiasing_3d_kernel!, mask, kmaxs[1], kmaxs[2], kmaxs[3],
                nx, ny, nz; ndrange=nx*ny*nz)
    else
        # 1D: two-sided band |k| ≤ kmax → zero indices kmax+2 : n-kmax
        # (i−1 > kmax on the positive side AND n−i+1 > kmax on the negative side).
        n = shape[1]
        kmax = kmaxs[1]
        lo, hi = kmax + 2, n - kmax
        if lo <= hi
            fill!(view(mask, lo:hi), zero(T))
        end
    end

    return mask
end

"""
    apply_dealiasing_gpu!(data::CuArray, cutoff::Float64=2.0/3.0)

Zero spectral modes with |k| > kmax on any axis, in place, where kmax per axis
follows the solver's `Tarang._axis_dealias_cutoff` rule (see
`create_dealiasing_mask_gpu`). `cutoff` accepts EITHER a retained fraction ≤ 1
(e.g. `2/3`) OR the `dealias` padding factor ≥ 1 (e.g. `3/2`); `cutoff == 1`
disables dealiasing (see `_dealias_kmax`). All axes are treated as two-sided full
FFT spectra; rfft half-spectrum layouts are NOT supported (undetectable from the
array shape — for rfft/production dealiasing use `Tarang.apply_spectral_cutoff!`,
which is rfft-aware).
"""
function apply_dealiasing_gpu!(data::CuArray{T, 2}, cutoff::Float64=2.0/3.0) where T
    nx, ny = size(data)
    kmax_x = _dealias_kmax(nx, cutoff)
    kmax_y = _dealias_kmax(ny, cutoff)
    arch = Tarang.architecture(data)
    launch!(arch, dealiasing_2d_kernel!, data, kmax_x, kmax_y, nx, ny;
            ndrange=nx*ny)
    return data
end

function apply_dealiasing_gpu!(data::CuArray{T, 3}, cutoff::Float64=2.0/3.0) where T
    nx, ny, nz = size(data)
    kmax_x = _dealias_kmax(nx, cutoff)
    kmax_y = _dealias_kmax(ny, cutoff)
    kmax_z = _dealias_kmax(nz, cutoff)
    arch = Tarang.architecture(data)
    launch!(arch, dealiasing_3d_kernel!, data, kmax_x, kmax_y, kmax_z,
            nx, ny, nz; ndrange=nx*ny*nz)
    return data
end

function apply_dealiasing_gpu!(data::CuArray{T, 1}, cutoff::Float64=2.0/3.0) where T
    n = length(data)
    kmax = _dealias_kmax(n, cutoff)
    # Two-sided band |k| ≤ kmax → zero indices kmax+2 : n-kmax (see mask helper).
    lo, hi = kmax + 2, n - kmax
    if lo <= hi
        fill!(view(data, lo:hi), zero(T))
    end
    return data
end

# ============================================================================
# GPU Field Multiplication (Override for nonlinear terms)
# ============================================================================

"""
    gpu_multiply_fields!(result::CuArray, a::CuArray, b::CuArray)

GPU-accelerated pointwise multiplication using CUDA kernels.
"""
function Tarang.gpu_multiply_fields!(result::CuArray, a::CuArray, b::CuArray)
    launch!(architecture(result), mul_kernel!, result, a, b; ndrange=length(result))
    return result
end

# ============================================================================
# Distributed GPU FFT Implementation (CUFFT)
# ============================================================================

"""
    local_fft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)

Perform local FFT along dimension `dim` using CUFFT, on the device holding `data`.
"""
function Tarang.local_fft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)
    # A rank whose slab is empty (more ranks than points along an axis) has
    # nothing to transform, and cuFFT cannot plan an empty batch. Return the
    # empty result in the transform's element type so the rank's transpose
    # buffers match its peers'.
    isempty(data) && return similar(data, complex(float(eltype(data))))
    # CUFFT plans are device-specific: run on the DATA's device. This used to be
    # `CuDevice(dfft.config.device_id)`, i.e. the MPI rank.
    data_device = CUDA.device(data)
    Tarang._distributed_fft_device_id(CUDA.deviceid(data_device), dfft.config.device_id)
    prev_device = CUDA.device()
    try
        CUDA.device!(data_device)
        result = CUFFT.fft(data, (dim,))
        CUDA.synchronize()
        return result
    finally
        CUDA.device!(prev_device)
    end
end

"""
    local_ifft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)

Perform local inverse FFT along dimension `dim` using CUFFT, on the device holding `data`.
"""
function Tarang.local_ifft_dim!(data::CuArray, dim::Int, dfft::DistributedGPUFFT)
    isempty(data) && return similar(data, complex(float(eltype(data))))  # see local_fft_dim!
    data_device = CUDA.device(data)
    Tarang._distributed_fft_device_id(CUDA.deviceid(data_device), dfft.config.device_id)
    prev_device = CUDA.device()
    try
        CUDA.device!(data_device)
        result = CUFFT.ifft(data, (dim,))
        CUDA.synchronize()
        return result
    finally
        CUDA.device!(prev_device)
    end
end

"""
    gpu_fft_1d!(output::CuArray, input::CuArray, dim::Int)

1D FFT along specified dimension on GPU.
"""
function gpu_fft_1d!(output::CuArray, input::CuArray, dim::Int)
    # Derive device from input array for multi-GPU correctness
    input_device = CUDA.device(input)
    device_id = CUDA.deviceid(input_device)
    prev_device = CUDA.device()
    try
        CUDA.device!(input_device)
        plan = get_fft_1d_plan(size(input), dim, eltype(input); inverse=false, device_id=device_id)
        mul!(output, plan, input)
        return output
    finally
        CUDA.device!(prev_device)
    end
end

"""
    gpu_ifft_1d!(output::CuArray, input::CuArray, dim::Int)

1D inverse FFT along specified dimension on GPU.
"""
function gpu_ifft_1d!(output::CuArray, input::CuArray, dim::Int)
    # Derive device from input array for multi-GPU correctness
    input_device = CUDA.device(input)
    device_id = CUDA.deviceid(input_device)
    prev_device = CUDA.device()
    try
        CUDA.device!(input_device)
        plan = get_fft_1d_plan(size(input), dim, eltype(input); inverse=true, device_id=device_id)
        mul!(output, plan, input)
        return output
    finally
        CUDA.device!(prev_device)
    end
end

"""
    gpu_rfft!(output::CuArray{Complex{T}}, input::CuArray{T})

Real-to-complex FFT on GPU.
"""
function gpu_rfft!(output::CuArray{Complex{T}}, input::CuArray{T}) where {T<:AbstractFloat}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, size(input), T; real_input=true)
    gpu_forward_fft!(output, input, plan)
    return output
end

"""
    gpu_irfft!(output::CuArray{T}, input::CuArray{Complex{T}}, n::Int)

Complex-to-real inverse FFT on GPU.
"""
function gpu_irfft!(output::CuArray{T}, input::CuArray{Complex{T}}, n::Int) where {T<:AbstractFloat}
    arch = Tarang.architecture(output)
    plan = get_gpu_fft_plan(arch, (n, size(input)[2:end]...), T; real_input=true)
    gpu_backward_fft!(output, input, plan)
    return output
end

# ============================================================================
# GPU Memory Management
# ============================================================================

"""
    gpu_memory_info()

Get current GPU memory usage information.
"""
function gpu_memory_info()
    # CUDA.jl v5 API (the old `CUDA.Mem.info()` was restructured out).
    free  = CUDA.available_memory()
    total = CUDA.total_memory()
    used = total - free
    return (
        free_bytes = free,
        used_bytes = used,
        total_bytes = total,
        free_gb = free / 1e9,
        used_gb = used / 1e9,
        total_gb = total / 1e9,
        usage_percent = 100.0 * used / total
    )
end

"""
    check_gpu_memory(required_bytes::Int)

Check if there's enough GPU memory available.
"""
function check_gpu_memory(required_bytes::Int)
    info = gpu_memory_info()
    if info.free_bytes < required_bytes
        @warn "Insufficient GPU memory" required=required_bytes/1e9 available=info.free_gb
        return false
    end
    return true
end

# ============================================================================
# GPU-aware Array Allocation Helpers (for operators.jl compatibility)
# ============================================================================

"""
    allocate_like(a::CuArray, T::Type, dims...)

Allocate a zeros CuArray on the same device as the input CuArray.
Ensures correct device context for multi-GPU support.
"""
function Tarang.allocate_like(a::CuArray, T::Type, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(a))
        return CUDA.zeros(T, dims...)
    finally
        CUDA.device!(prev_device)
    end
end

function Tarang.allocate_like(a::CuArray, dims...)
    prev_device = CUDA.device()
    try
        CUDA.device!(CUDA.device(a))
        return CUDA.zeros(eltype(a), dims...)
    finally
        CUDA.device!(prev_device)
    end
end

# Device identity for workspace-cache keys (src fallback returns `nothing` for
# host arrays): two CuArrays on different devices must never share cached
# buffers or CUFFT plans.
Tarang._device_cache_token(x::CuArray) = CUDA.deviceid(CUDA.device(x))
# Derivative scratch is task-owned; separate streams/contexts within that task
# must not reuse an array while previous GPU work is still in flight.
Tarang._diff_matmul_cache_token(x::CuArray) =
    (CUDA.context(x), CUDA.deviceid(CUDA.device(x)), CUDA.stream())

"""
    copy_to_device(a::AbstractArray, target::CuArray)

Copy array `a` to GPU on the same device as target CuArray.
Ensures correct device context for multi-GPU support.
"""
function Tarang.copy_to_device(a::AbstractArray, target::CUDA.AnyCuArray)
    prev_device = CUDA.device()
    try
        result = Tarang.on_architecture(Tarang.architecture(target), a)
        return result === a ? copy(result) : result
    finally
        CUDA.device!(prev_device)
    end
end

function Tarang.copy_to_device(a::CuArray, target::CuArray)
    src_device = CUDA.device(a)
    dst_device = CUDA.device(target)

    if src_device == dst_device
        # Same device, just copy
        return copy(a)
    else
        # Cross-device copy: explicitly go through host memory to avoid
        # requiring P2P access between devices.
        prev_device = CUDA.device()
        try
            # 1. Set source device context and download to host
            CUDA.device!(src_device)
            host_data = Array(a)
            # 2. Set destination device context and upload from host
            CUDA.device!(dst_device)
            return CuArray(host_data)
        finally
            # 3. Restore caller's device context — even on exception (e.g. OOM
            #    during the upload), so the task is not stranded on the wrong
            #    device (matches Tarang.allocate_like).
            CUDA.device!(prev_device)
        end
    end
end

# ============================================================================
# GPU-native Random Number Generation for Stochastic Forcing
# ============================================================================

"""
    Tarang._try_gpu_rand!(phases::CuArray{T}) -> Bool

Legacy compatibility hook. Stochastic forcing now uses Tarang's seeded
counter-based device kernel and no longer calls this function.
"""
Tarang._try_gpu_rand!(phases::CuArray{T}) where {T<:AbstractFloat} = false

# ============================================================================
# GPU-Native Spectral Padding/Truncation for 3/2-Rule Dealiasing
# ============================================================================

# Ordinary CUDA C2R mul! preserves its input by copying through a plan-owned
# half spectrum. Here the evaluator owns and consumes its spectrum. Inverting
# its R2C plan shares that plan's buffer, which we use as the spectrum itself;
# execute that plan directly after size/alias validation. cuFFT's executor rebinds to the current task
# stream, including trailing non-transform axes. Keep a public-API fallback for
# CUDA versions without the trailing-axis executor.
function Tarang._plan_padded_inverse(forward::CUFFT.CuFFTPlan, spectrum, n, dims)
    isdefined(CUFFT, :unsafe_execute_trailing!) || return CUFFT.plan_irfft(spectrum, n, dims)
    return inv(forward)
end

function Tarang._padded_plan_spectrum(forward::CUFFT.CuFFTPlan, arch, ::Type{T}, shape) where T
    buffer = hasproperty(forward, :buffer) ? forward.buffer : nothing
    if buffer !== nothing && size(buffer) == shape && eltype(buffer) === Complex{T}
        return buffer
    end
    return CUDA.zeros(Complex{T}, shape...)
end

function Tarang._padded_inverse!(dst::CuArray{T}, plan::CUFFT.ScaledPlan{Complex{T},P},
                                  src::CuArray{Complex{T}}) where {T<:AbstractFloat,P<:CUFFT.CuFFTPlan}
    if isdefined(CUFFT, :unsafe_execute_trailing!)
        CUFFT.assert_applicable(plan.p, src, dst)
        CUFFT.unsafe_execute_trailing!(plan.p, src, dst)
        dst .*= plan.scale
        return dst
    end
    return mul!(dst, plan, src)
end

"""
Override _pad_spectral! for CuArray: uses fused GPU kernel instead of
multiple slice-based copies, reducing kernel launch overhead.
"""
function Tarang._pad_spectral!(padded::CuArray{Complex{T}}, spec_data::CuArray{Complex{T}},
                                original_shape::Tuple, padded_shape::Tuple,
                                fourier_dims::Vector{Int}, cosine_dims=fourier_dims) where T
    # Broadcasts, KA kernels and cuFFT remain ordered on this task's CUDA stream.
    # No host synchronization is needed between dependent operations. Pin the
    # device before the first broadcast as well as inside launch!.
    arch = Tarang.architecture(spec_data)
    ensure_device!(arch)
    fill!(padded, zero(Complex{T}))
    ndim = length(original_shape)

    # Route through `launch!` rather than constructing the kernel with a literal
    # `256`: a scalar workgroup is padded with ones by KernelAbstractions, so a
    # 2-D/3-D launch got a (256, 1[, 1]) block and masked off every thread past
    # the first dimension's extent. `workgroup_size` now splits the budget across
    # dimensions (src/core/architectures.jl).
    if ndim == 2
        N1, N2 = original_shape
        M1, M2 = padded_shape
        launch!(arch, pad_spectral_2d_kernel!, padded, spec_data, N1, N2, M1, M2,
                1 in fourier_dims, 2 in fourier_dims;
                ndrange=(N1, N2))
    elseif ndim == 3
        N1, N2, N3 = original_shape
        M1, M2, M3 = padded_shape
        launch!(arch, pad_spectral_3d_kernel!, padded, spec_data, N1, N2, N3, M1, M2, M3,
                1 in fourier_dims, 2 in fourier_dims, 3 in fourier_dims;
                ndrange=(N1, N2, N3))
    else
        # 1D: fall back to slice-based (fast enough for 1D)
        N = original_shape[1]
        M = padded_shape[1]
        if 1 in fourier_dims
            # Fourier: split the half-spectrum (low positive freqs + high-index negatives).
            Nh = N ÷ 2
            n_neg = N - Nh - 1
            padded[1:Nh+1] .= spec_data[1:Nh+1]
            if n_neg > 0
                padded[M-n_neg+1:M] .= spec_data[N-n_neg+1:N]
            end
        else
            # Non-Fourier (e.g. Chebyshev): coefficients run low→high order, so the
            # correct zero-padded embedding is a full leading copy (`padded` is zero-filled).
            padded[1:N] .= spec_data[1:N]
        end
    end

    Tarang._place_padded_nyquist!(padded, original_shape, padded_shape, fourier_dims, cosine_dims)
    return padded
end

"""
Override _truncate_spectral! for CuArray: uses fused GPU kernel.
"""
function Tarang._truncate_spectral!(result::CuArray{Complex{T}}, padded_spec::CuArray{Complex{T}},
                                     original_shape::Tuple, padded_shape::Tuple,
                                     fourier_dims::Vector{Int}, cosine_dims=fourier_dims) where T
    # Pin the current CUDA device to the array's device (see _pad_spectral! note).
    arch = Tarang.architecture(result)
    ensure_device!(arch)
    ndim = length(original_shape)

    Tarang._collect_truncated_nyquist!(padded_spec, original_shape, padded_shape, fourier_dims, cosine_dims)

    if ndim == 2
        N1, N2 = original_shape
        M1, M2 = padded_shape
        launch!(arch, truncate_spectral_2d_kernel!, result, padded_spec, N1, N2, M1, M2,
                1 in fourier_dims, 2 in fourier_dims;
                ndrange=(N1, N2))
    elseif ndim == 3
        N1, N2, N3 = original_shape
        M1, M2, M3 = padded_shape
        launch!(arch, truncate_spectral_3d_kernel!, result, padded_spec, N1, N2, N3, M1, M2, M3,
                1 in fourier_dims, 2 in fourier_dims, 3 in fourier_dims;
                ndrange=(N1, N2, N3))
    else
        # 1D: slice-based
        N = original_shape[1]
        M = padded_shape[1]
        if 1 in fourier_dims
            # Fourier: take the low positive freqs + the high-index negative-freq mirror.
            Nh = N ÷ 2
            n_neg = N - Nh - 1
            result[1:Nh+1] .= padded_spec[1:Nh+1]
            if n_neg > 0
                result[N-n_neg+1:N] .= padded_spec[M-n_neg+1:M]
            end
        else
            # Non-Fourier (e.g. Chebyshev): keep the leading N low-order coefficients.
            result[1:N] .= padded_spec[1:N]
        end
    end
end

# ============================================================================
# GPU MatSolvers Helper Implementations
# ============================================================================
# These extend the abstract function stubs defined in Tarang/src/tools/gpu_matsolvers.jl
# to provide CUDA-backed implementations for GPU matrix solvers.

using SparseArrays: SparseMatrixCSC

# `::Val{:cuda}` beats Tarang's `::Val` fallback, so these EXTEND rather than
# overwrite (a zero-argument method here would overwrite Tarang's).
Tarang._gpu_cusolver_module(::Val{:cuda}) = CUDA.CUSOLVER
Tarang._gpu_cusparse_module(::Val{:cuda}) = CUDA.CUSPARSE
Tarang._gpu_zeros(T::Type, dims...) = CUDA.zeros(T, dims...)
# Rank-generic: the previous pair was `::AbstractArray -> CuVector` plus an
# `::AbstractMatrix -> CuMatrix` override, so a 3-D argument picked the VECTOR
# method and died inside `CuVector`'s constructor with a shape error instead of
# doing the obvious thing.
Tarang._gpu_array(data::AbstractArray{<:Any,N}, T::Type) where {N} = CuArray{T,N}(data)
Tarang._gpu_sparse_csr(A::SparseMatrixCSC, T::Type) = CUDA.CUSPARSE.CuSparseMatrixCSR(SparseMatrixCSC{T, Int32}(A))
Tarang._gpu_axpy!(α, x, y) = CUDA.axpy!(α, x, y)
Tarang._is_gpu_array(a::CuArray) = true
Tarang._is_gpu_array(::Any) = false
Tarang._gpu_ilu0(A_csr) = CUDA.CUSPARSE.ilu02(A_csr)
Tarang._gpu_ic0(A_csr) = CUDA.CUSPARSE.ic02(A_csr)
