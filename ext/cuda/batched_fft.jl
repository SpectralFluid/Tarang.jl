import LinearAlgebra: mul!

# ============================================================================
# Batched FFT Support
# ============================================================================

"""
    BatchedGPUFFTPlan

FFT plan for batched transforms on multiple arrays simultaneously.
More efficient than individual FFTs for multi-field operations.
"""
struct BatchedGPUFFTPlan{P, IP, I, O}
    plan::P
    iplan::IP
    field_size::Tuple{Vararg{Int}}
    batch_size::Int
    is_real::Bool
    packed_input::I
    packed_output::O
    owner::WeakRef
    stream::CuStream
    device_id::Int
end

"""
    plan_batched_gpu_fft(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)

Create a batched GPU FFT plan for multiple fields.

# Arguments
- `arch`: GPU architecture
- `field_size`: Size of each individual field
- `T`: Element type
- `batch_size`: Number of fields to transform simultaneously
- `real_input`: If true, create real-to-complex plan

# Example
```julia
plan = plan_batched_gpu_fft(GPU(), (64, 64), Float64, 4; real_input=true)
```

**Important:** `field_size` should be the LOCAL field shape (what this process owns),
not the global domain size.
"""
function plan_batched_gpu_fft(arch::GPU{CuDevice}, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)
    # Ensure correct device for multi-GPU support
    ensure_device!(arch)
    _create_batched_fft_plan(field_size, T, batch_size, real_input)
end

# Fallback for generic GPU: delegates to device-specific version using current device,
# ensuring proper device context via ensure_device!
plan_batched_gpu_fft(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false) =
    plan_batched_gpu_fft(GPU{CuDevice}(CUDA.device()), field_size, T, batch_size; real_input=real_input)

# Internal helper to create batched FFT plan
function _create_batched_fft_plan(field_size::Tuple, T::Type, batch_size::Int, real_input::Bool)
    complex_T = T <: Complex ? T : Complex{T}

    # Create batched shape (field dims..., batch)
    batched_size = (field_size..., batch_size)

    if real_input
        # Real-to-complex batched FFT
        dummy_in = CUDA.zeros(T, batched_size...)
        # FFT over all dimensions except the last (batch dimension)
        fft_dims = ntuple(i -> i, length(field_size))
        plan = CUFFT.plan_rfft(dummy_in, fft_dims)

        # Inverse plan — rfft reduces the first dimension (FFTW/CUFFT convention)
        out_size = (div(field_size[1], 2) + 1, field_size[2:end]..., batch_size)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, field_size[1], fft_dims)

        return BatchedGPUFFTPlan(plan, iplan, field_size, batch_size, true,
            dummy_in, dummy_out, WeakRef(current_task()), CUDA.stream(), _current_device_id())
    else
        # Complex-to-complex batched FFT
        dummy = CUDA.zeros(complex_T, batched_size...)
        fft_dims = ntuple(i -> i, length(field_size))
        plan = CUFFT.plan_fft(dummy, fft_dims)
        iplan = CUFFT.plan_ifft(dummy, fft_dims)

        return BatchedGPUFFTPlan(plan, iplan, field_size, batch_size, false,
            dummy, similar(dummy), WeakRef(current_task()), CUDA.stream(), _current_device_id())
    end
end

# Batched FFT plan cache (Thread-Safe)
"""
Thread-safe cache for batched GPU FFT plans.
Uses a ReentrantLock to protect concurrent access from multiple Julia threads.
"""
struct BatchedFFTCache
    plans::WeakKeyDict{Task, Dict{Tuple, BatchedGPUFFTPlan}}
    lock::ReentrantLock
end

const BATCHED_FFT_CACHE = BatchedFFTCache(WeakKeyDict{Task, Dict{Tuple, BatchedGPUFFTPlan}}(), ReentrantLock())

"""
    _batched_plan_key(arch, field_size, T, batch_size, real_input)

Generate a cache key for batched FFT plans.
Includes device ID for multi-GPU support.
"""
_batched_plan_key(arch::GPU{CuDevice}, field_size::Tuple, T::Type, batch_size::Int, real_input::Bool) =
    (CUDA.deviceid(arch.device), field_size, T, batch_size, real_input)

_batched_plan_key(arch::GPU, field_size::Tuple, T::Type, batch_size::Int, real_input::Bool) =
    (_current_device_id(), field_size, T, batch_size, real_input)

"""
    get_batched_fft_plan(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)

Get or create a cached batched FFT plan (thread-safe).

**Important:** `field_size` should be the LOCAL field shape (what this process owns),
not the global domain size. Plans and packed buffers are cached per task and
(device, stream, size, type, batch_size, real_input). Tasks are weakly referenced
so short-lived task caches can be collected.
"""
function get_batched_fft_plan(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)
    ensure_device!(arch)
    key = (_batched_plan_key(arch, field_size, T, batch_size, real_input)..., CUDA.stream())
    lock(BATCHED_FFT_CACHE.lock) do
        plans = get!(BATCHED_FFT_CACHE.plans, current_task()) do
            Dict{Tuple, BatchedGPUFFTPlan}()
        end
        return get!(plans, key) do
            plan_batched_gpu_fft(arch, field_size, T, batch_size; real_input)
        end
    end
end

function _check_batched_fft_context(plan::BatchedGPUFFTPlan)
    plan.owner.value === current_task() || throw(ArgumentError(
        "Batched FFT plans own task-local buffers; obtain a plan in the executing task"))
    plan.device_id == _current_device_id() && plan.stream == CUDA.stream() || throw(ArgumentError(
        "Batched FFT plans own stream-local buffers; obtain a plan on the executing device and stream"))
    return nothing
end

# Backend-generic executor permits FFTW-backed parity/ownership tests without a
# CUDA device. Public CUDA entry points additionally enforce task/stream affinity.
function _execute_batched_fft!(outputs, inputs, fft_plan, packed_in, packed_out)
    batch_dim = ndims(packed_in)
    n = size(packed_in, batch_dim)
    length(inputs) == n && length(outputs) == n || throw(DimensionMismatch("FFT batch count"))
    input_shape = size(packed_in)[1:end-1]
    output_shape = size(packed_out)[1:end-1]
    for i in 1:n
        size(inputs[i]) == input_shape && size(outputs[i]) == output_shape ||
            throw(DimensionMismatch("FFT field shape does not match its plan"))
    end
    for i in 1:n
        selectdim(packed_in, batch_dim, i) .= inputs[i]
    end
    mul!(packed_out, fft_plan, packed_in)
    for i in 1:n
        outputs[i] .= selectdim(packed_out, ndims(packed_out), i)
    end
    return outputs
end

"""
    batched_fft!(outputs::Vector{<:CuArray}, inputs::Vector{<:CuArray}, plan::BatchedGPUFFTPlan)

Execute batched FFT on multiple fields simultaneously.
"""
function batched_fft!(outputs::Vector{<:CuArray}, inputs::Vector{<:CuArray}, plan::BatchedGPUFFTPlan)
    _check_batched_fft_context(plan)
    return _execute_batched_fft!(outputs, inputs, plan.plan, plan.packed_input, plan.packed_output)
end

"""
    batched_ifft!(outputs, inputs, plan::BatchedGPUFFTPlan)

Execute a batched inverse transform with retained packed buffers. Caller inputs
are copied before the potentially destructive C2R transform. A plan is local to
its creating Julia task and CUDA stream; request a plan in each execution context.
"""
function batched_ifft!(outputs::Vector{<:CuArray}, inputs::Vector{<:CuArray}, plan::BatchedGPUFFTPlan)
    _check_batched_fft_context(plan)
    return _execute_batched_fft!(outputs, inputs, plan.iplan, plan.packed_output, plan.packed_input)
end

"""
    clear_batched_fft_cache!()

Clear all cached batched FFT plans (thread-safe).
"""
function clear_batched_fft_cache!()
    lock(BATCHED_FFT_CACHE.lock) do
        empty!(BATCHED_FFT_CACHE.plans)
    end
end

# ============================================================================
# Stream-aware FFT Execution
# ============================================================================

"""
    gpu_fft_async!(output::CuArray, input::CuArray, plan; stream=nothing, synchronize=false)

Execute FFT asynchronously on specified stream.
The FFT is truly asynchronous - call CUDA.synchronize(stream) to wait for completion.

Uses CUDA.stream! context manager which properly integrates with CUDA.jl's
internal stream management (update_stream) so the plan correctly tracks
which stream it's executing on.
"""
function gpu_fft_async!(output::CuArray, input::CuArray, plan::GPUFFTPlan; stream=nothing, synchronize::Bool=false)
    if stream !== nothing
        CUDA.stream!(stream) do
            mul!(output, plan.plan, input)
        end
        synchronize && CUDA.synchronize(stream)
    else
        mul!(output, plan.plan, input)
    end
    return output
end

"""
    gpu_ifft_async!(output::CuArray, input::CuArray, plan; stream=nothing, synchronize=false)

Execute inverse FFT asynchronously on specified stream.
The FFT is truly asynchronous - call CUDA.synchronize(stream) to wait for completion.

Uses CUDA.stream! context manager which properly integrates with CUDA.jl's
internal stream management (update_stream) so the plan correctly tracks
which stream it's executing on.

For real plans (`plan.is_real`), the C2R (irfft) transform is DESTRUCTIVE on
its input buffer (cuFFT convention, same as FFTW). The input is copied into a
freshly allocated scratch before the `mul!` so the caller's coefficient buffer
is never corrupted — mirrors the guard in `gpu_backward_fft!` (a fresh
allocation, not the shared `get_gpu_dct_scratch` cache, because async/stream
execution must not race other users of the shared scratch).
"""
function gpu_ifft_async!(output::CuArray, input::CuArray, plan::GPUFFTPlan; stream=nothing, synchronize::Bool=false)
    if stream !== nothing
        CUDA.stream!(stream) do
            _gpu_ifft_exec!(output, input, plan)
        end
        synchronize && CUDA.synchronize(stream)
    else
        _gpu_ifft_exec!(output, input, plan)
    end
    return output
end

# Execute the inverse transform, guarding the caller's input against cuFFT's
# destructive C2R. C2C inverse is non-destructive — no copy needed.
function _gpu_ifft_exec!(output::CuArray, input::CuArray, plan::GPUFFTPlan)
    if plan.is_real
        scratch = similar(input)
        copyto!(scratch, input)
        mul!(output, plan.iplan, scratch)
    else
        mul!(output, plan.iplan, input)
    end
    return output
end

