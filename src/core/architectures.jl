"""
    Architectures - Backend abstraction for CPU and GPU execution

Architecture abstraction following Oceananigans.jl pattern for backend-agnostic computation.
This enables running the same code on CPU or GPU with minimal changes.

Key concepts:
- Architecture types (CPU, GPU) define the compute backend
- array_type() maps architecture to array constructors
- on_architecture() moves data between backends
- device() returns the KernelAbstractions backend

Usage:
    arch = CPU()           # Use CPU backend (default)
    arch = GPU()           # Use a loaded CUDA, AMDGPU or Metal backend

    # Create arrays on architecture
    a = zeros(arch, Float64, 100, 100)

    # Move data between architectures
    a_gpu = on_architecture(GPU(), a_cpu)
    a_cpu = on_architecture(CPU(), a_gpu)
"""

using KernelAbstractions

# ============================================================================
# Abstract Architecture Types
# ============================================================================

"""
    AbstractArchitecture

Abstract supertype for all compute architectures.
"""
abstract type AbstractArchitecture end

"""
    AbstractSerialArchitecture <: AbstractArchitecture

Architecture for single-device (non-distributed) computation.
"""
abstract type AbstractSerialArchitecture <: AbstractArchitecture end

# ============================================================================
# CPU Architecture
# ============================================================================

"""
    CPU <: AbstractSerialArchitecture

CPU architecture using Julia's native arrays and threading.
This is the default architecture.

# Example
```julia
dist = Distributor(coordsys; device=CPU())  # default, uses PencilArrays/PencilFFTs for MPI
```
"""
struct CPU <: AbstractSerialArchitecture end

# Note: The struct definition above provides the default CPU() constructor.
# No additional constructor needed for CPU since it has no fields.

"""
    device(::CPU)

Return the KernelAbstractions backend for CPU.
"""
device(::CPU) = KernelAbstractions.CPU()

"""
    array_type(::CPU)

Return the array type for CPU architecture.
"""
array_type(::CPU) = Array

"""
    array_type(::CPU, T::Type)

Return the concrete array type for CPU with element type T.
"""
array_type(::CPU, T::Type) = Array{T}

# ============================================================================
# GPU Architecture (placeholder - actual implementation via extension)
# ============================================================================

"""
    GPU <: AbstractSerialArchitecture

Select a GPU with `GPU(; backend=:auto, device_id=0)`. Load CUDA.jl for
NVIDIA, AMDGPU.jl for AMD ROCm, or Metal.jl for Apple silicon. Device IDs
start at zero. Automatic selection tries CUDA, AMDGPU, then Metal.

```julia
using Tarang, AMDGPU
dist = Distributor(coordsys; device=GPU(backend=:amdgpu, device_id=0))
```
"""
struct GPU{D} <: AbstractSerialArchitecture
    device::D
end

function _gpu_device(::Val{B}, device_id::Integer) where B
    B in (:cuda, :amdgpu, :metal) || throw(ArgumentError(
        "Unknown GPU backend $B. Choose :auto, :cuda, :amdgpu or :metal."))
    package = (cuda="CUDA", amdgpu="AMDGPU", metal="Metal")[B]
    error("GPU(backend=:$B) requires loading $package.jl with `using $package`.")
end

function GPU(; device_id::Int=0, backend::Symbol=:auto)
    device_id >= 0 || throw(ArgumentError("GPU device IDs start at zero, got $device_id"))
    if backend === :auto
        if has_cuda()
            backend = :cuda
        elseif has_amdgpu()
            backend = :amdgpu
        elseif has_metal()
            backend = :metal
        else
            error("GPU() requires a functional GPU. Load CUDA.jl for NVIDIA, AMDGPU.jl for AMD ROCm, or Metal.jl for Apple silicon.")
        end
    end
    return GPU(_gpu_device(Val(backend), device_id))
end

device(gpu::GPU) = gpu.device
array_type(arch::GPU) = error("No array backend is loaded for $arch")
array_type(arch::GPU, T::Type) = array_type(arch){T}

# ============================================================================
# Architecture Detection and Inference
# ============================================================================

"""
    architecture(a::Array)

Infer the architecture from an array type.
Returns CPU() for standard Julia arrays.
"""
architecture(::Array) = CPU()
architecture(::AbstractArray) = CPU()  # Default fallback

"""
    architecture(arch::AbstractArchitecture)

Return the architecture itself (identity).
"""
architecture(arch::AbstractArchitecture) = arch

# ============================================================================
# Data Movement Between Architectures
# ============================================================================

"""
    on_architecture(arch::AbstractArchitecture, data)

Move data to the specified architecture.
Returns the data on the target architecture.

# Arguments
- `arch`: Target architecture (CPU() or GPU())
- `data`: Data to move (Array, Tuple, NamedTuple, etc.)

# Returns
- Data converted to the target architecture's array type

# Example
```julia
a_cpu = rand(100, 100)
a_gpu = on_architecture(GPU(), a_cpu)
a_back = on_architecture(CPU(), a_gpu)
```
"""
function on_architecture end

# CPU to CPU: no-op
on_architecture(::CPU, a::Array) = a
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a

# Convert any array to CPU
function on_architecture(::CPU, a::AbstractArray)
    Array(a)
end

# Handle tuples recursively
function on_architecture(arch::AbstractArchitecture, t::Tuple)
    Tuple(on_architecture(arch, elem) for elem in t)
end

# Handle NamedTuples recursively
function on_architecture(arch::AbstractArchitecture, nt::NamedTuple)
    names = keys(nt)
    values = Tuple(on_architecture(arch, nt[k]) for k in names)
    NamedTuple{names}(values)
end

# Handle Nothing
on_architecture(::AbstractArchitecture, ::Nothing) = nothing

# Handle scalars (pass through)
on_architecture(::AbstractArchitecture, x::Number) = x
on_architecture(::AbstractArchitecture, x::String) = x
on_architecture(::AbstractArchitecture, x::Symbol) = x

"""
    to_architecture(arch::AbstractArchitecture, x)

Place `x` on `arch`'s device, as a no-op when `arch` is the host.

Distinct from [`on_architecture`](@ref) in exactly one respect: the `CPU`
fallback of `on_architecture` materializes `Array(x)` for any input outside its
`Array` and view-of-`Array` identity methods, so a host-resident wrapper such as
a `PermutedDimsArray` is copied. Call sites that only wanted "upload if this is a
GPU run" therefore wrapped it in
`is_gpu(arch) ? on_architecture(arch, x) : x` — putting a Bool branch in front of
an otherwise architecture-generic operation. This function performs that choice
by dispatch instead, so the host case is a method rather than a branch.

Use `on_architecture(CPU(), x)` when a dense host `Array` is genuinely required
(MPI buffers, file IO); use this when the host case should leave `x` untouched.
"""
to_architecture(::CPU, x) = x
to_architecture(arch::AbstractArchitecture, x) = on_architecture(arch, x)

# ============================================================================
# Array Allocation on Architecture
# ============================================================================

"""
    zeros(arch::AbstractArchitecture, T::Type, dims...)

Create a zero-filled array on the specified architecture.

# Example
```julia
a = zeros(CPU(), Float64, 100, 100)
a = zeros(GPU(), Float32, 100, 100)
```
"""
function Base.zeros(arch::CPU, T::Type, dims::Integer...)
    # For CPU, just use standard Julia zeros which creates Array{T}
    return zeros(T, dims...)
end

function Base.zeros(arch::GPU, T::Type, dims::Integer...)
    # GPU extensions specialize allocation for their native devices.
    AT = array_type(arch, T)
    return fill!(similar(AT, dims), zero(T))
end

function Base.zeros(arch::AbstractArchitecture, T::Type, dims::Tuple{Vararg{Integer}})
    zeros(arch, T, dims...)
end

"""
    ones(arch::AbstractArchitecture, T::Type, dims...)

Create an array of ones on the specified architecture.
"""
function Base.ones(arch::CPU, T::Type, dims::Integer...)
    # For CPU, just use standard Julia ones which creates Array{T}
    return ones(T, dims...)
end

function Base.ones(arch::GPU, T::Type, dims::Integer...)
    # Generic-GPU fallback (see zeros above): no array-type-first `ones` exists.
    AT = array_type(arch, T)
    return fill!(similar(AT, dims), one(T))
end

function Base.ones(arch::AbstractArchitecture, T::Type, dims::Tuple{Vararg{Integer}})
    ones(arch, T, dims...)
end

"""
    similar(arch::AbstractArchitecture, a::AbstractArray)

Create an uninitialized array with the same size on the specified architecture.
"""
function Base.similar(arch::CPU, a::AbstractArray)
    # For CPU, create a standard Array with same element type and size
    return similar(Array{eltype(a)}, size(a))
end

function Base.similar(arch::GPU, a::AbstractArray)
    # For GPU, use array_type to get CuArray type
    AT = array_type(arch, eltype(a))
    return similar(AT, size(a))
end

function Base.similar(arch::CPU, a::AbstractArray, T::Type)
    # For CPU, create a standard Array with specified element type
    return similar(Array{T}, size(a))
end

function Base.similar(arch::GPU, a::AbstractArray, T::Type)
    # For GPU, use array_type to get CuArray type
    AT = array_type(arch, T)
    return similar(AT, size(a))
end

# ============================================================================
# Architecture Utilities
# ============================================================================

"""
    is_gpu(arch::AbstractArchitecture)

Check if the architecture is a GPU.
"""
is_gpu(::CPU) = false
is_gpu(::GPU) = true

"""
    _uses_transpose_storage(arch::AbstractArchitecture, nprocs::Integer) -> Bool

Whether a field on this architecture, distributed across `nprocs` ranks, needs
`TransposableFieldStorage` (explicit-transpose GPU+MPI transforms) instead of
`SerialFieldStorage` (PencilFFTs / local transforms). Used by `ScalarField`'s
constructor. Defined by dispatch on the architecture TYPE rather than a
`is_gpu(arch) && nprocs > 1` Bool branch, for the same reason every other
backend choice in this file is: a future architecture that forgets to add a
method here fails loudly (`MethodError`) instead of silently taking the CPU
branch.
"""
_uses_transpose_storage(::GPU, nprocs::Integer) = nprocs > 1
_uses_transpose_storage(::CPU, nprocs::Integer) = false

"""
    has_cuda()

Return whether the loaded CUDA backend can execute on an NVIDIA GPU.
"""
has_cuda() = _cuda_functional(Val(:cuda))
_cuda_functional(::Any) = false

"""Return whether the loaded AMDGPU backend can execute on a ROCm device."""
has_amdgpu() = _amdgpu_functional(Val(:amdgpu))
_amdgpu_functional(::Any) = false

"""Return whether the loaded Metal.jl extension has a functional Apple GPU."""
has_metal() = _metal_functional(Val(:metal))
_metal_functional(::Any) = false

_default_field_dtype(::AbstractArchitecture) = Float64
_coordinate_array(arch::AbstractArchitecture, ::Type, values) = to_architecture(arch, values)
_coordinate_array(arch::GPU, T::Type, values) = on_architecture(arch, convert(AbstractArray{real(T)}, values))
_validate_gpu_field_dtype(::AbstractArchitecture, ::Type) = nothing
_validate_distributor_size(::AbstractArchitecture, ::Int) = nothing

"""
    is_gpu_array(a::AbstractArray)

Return whether an array resides on a GPU. The backend extensions recognize
their device arrays and views.
"""
is_gpu_array(::Array) = false
is_gpu_array(::AbstractArray) = false  # Default: not a GPU array

# ============================================================================
# GPU Compatibility Helpers
# ============================================================================

"""
    allocate_like(a::AbstractArray, T::Type, dims...)

Allocate a zeros array with the same backend (CPU/GPU) as the input array.
This ensures GPU arrays stay on GPU and CPU arrays stay on CPU.
"""
function allocate_like(a::AbstractArray, T::Type, dims...)
    # Default implementation for CPU arrays
    return zeros(T, dims...)
end

function allocate_like(a::AbstractArray, dims...)
    return allocate_like(a, eltype(a), dims...)
end

"""
    similar_zeros(a::AbstractArray)

Create a zero-filled array with same type, element type, and size as input.
Works for both CPU and GPU arrays.
"""
function similar_zeros(a::AbstractArray)
    result = similar(a)
    fill!(result, zero(eltype(a)))
    return result
end

"""
    similar_zeros(a::AbstractArray, dims...)

Create a zero-filled array with same type and element type as input, but different size.
Works for both CPU and GPU arrays.
"""
function similar_zeros(a::AbstractArray, dims...)
    result = similar(a, dims...)
    fill!(result, zero(eltype(a)))
    return result
end

"""
    similar_zeros(a::AbstractArray, T::Type, dims...)

Create a zero-filled array with same array type as input, but different element type and size.
Works for both CPU and GPU arrays.
"""
function similar_zeros(a::AbstractArray, T::Type, dims...)
    result = similar(a, T, dims...)
    fill!(result, zero(T))
    return result
end

"""
    copy_to_device(a::AbstractArray, target::AbstractArray)

Return a copy of `a` on the device that owns `target`.
"""
function copy_to_device(a::AbstractArray, target::AbstractArray)
    if is_gpu_array(a) && !is_gpu_array(target)
        return on_architecture(CPU(), a)
    end
    return copy(a)
end

"""
    _device_cache_token(x::AbstractArray)

Identify the device that owns `x` so caches can keep buffers and plans local to
each GPU. Host arrays use `nothing`.
"""
_device_cache_token(::AbstractArray) = nothing

# All readers and writers of basis.transforms must use the same lock: even a
# cache hit can race with a Dict rehash caused by another kind of cache entry.
const _BASIS_CACHE_LOCK = ReentrantLock()

"""Get or build a basis cache entry atomically. This protects the dictionary,
not subsequent use of mutable scratch stored in it."""
function _get_basis_cache!(build::F, basis, key) where {F}
    lock(_BASIS_CACHE_LOCK) do
        get!(build, basis.transforms, key)
    end
end

"""
    _get_device_basis_cache!(build_host, basis, tag::Symbol, data::AbstractArray, keyparts...)

Single idiom for device-resident arrays cached in `basis.transforms`: the key is
`(tag, keyparts..., _device_cache_token(data))`, so a multi-GPU process never
shares a cached buffer across devices, and every call site keys device identity
the same way. On a miss, `build_host()` produces the host array, which is
uploaded to `data`'s device WITHOUT eltype conversion — callers convert to the
target precision in `build_host` (include the eltype in `keyparts` when it can
vary). Cache lookup and construction are synchronized with every other
`basis.transforms` accessor through `_get_basis_cache!`.
"""
function _get_device_basis_cache!(build_host::Function, basis, tag::Symbol,
                                  data::AbstractArray, keyparts...)
    key = (tag, keyparts..., _device_cache_token(data))
    return _get_basis_cache!(basis, key) do
        host = build_host()
        dev = similar(data, eltype(host), size(host))
        copyto!(dev, host)
        dev
    end
end

# ============================================================================
# Architecture Array Creation
# ============================================================================

"""
    create_array(arch::AbstractArchitecture, T::Type, dims...)

Create a zero-filled array of type T with given dimensions on the specified architecture.
This is a convenience function that works like zeros(arch, T, dims...).

# Example
```julia
a = create_array(CPU(), Float64, 100, 100)
a = create_array(GPU(), ComplexF64, 64, 64)
```
"""
function create_array(arch::AbstractArchitecture, T::Type, dims::Integer...)
    return zeros(arch, T, dims...)
end

function create_array(arch::AbstractArchitecture, T::Type, dims::Tuple{Vararg{Integer}})
    return zeros(arch, T, dims...)
end

"""
    synchronize(arch::AbstractArchitecture)

Wait for pending operations on the selected GPU. CPU execution returns immediately.
"""
synchronize(::CPU) = nothing
synchronize(arch::GPU) = error("No synchronization backend is loaded for $arch")

"""
    unsafe_free!(arch::AbstractArchitecture, a)

Explicitly free GPU memory. No-op for CPU.
"""
unsafe_free!(::CPU, a) = nothing
unsafe_free!(arch::GPU, a) = error("No memory backend is loaded for $arch")

"""
    _seed_device_rng!(arch::AbstractArchitecture, seed::Integer)

Seed the generator that `rand!`/`randn!` draw from for arrays on `arch`.
Each GPU extension seeds its native generator. The calling code seeds the host
generator separately. Sequences from the same seed can differ between backends.
"""
_seed_device_rng!(::CPU, seed::Integer) = nothing
_seed_device_rng!(arch::GPU, seed::Integer) = throw(ArgumentError(
    "cannot seed the device RNG of $arch. For a reproducible fill on this backend, " *
    "use fill_random!(...; reproducible=true)."))

# ============================================================================
# Kernel Launch Utilities
# ============================================================================

"""
    launch_config(arch::AbstractArchitecture, n::Int)

Get optimal kernel launch configuration for n elements.
Returns (threads, blocks) for GPU, (n,) for CPU.
"""
function launch_config(::CPU, n::Int)
    return (n,)  # CPU uses all elements
end

function launch_config(::GPU, n::Int)
    threads = 256  # Standard CUDA threads per block
    blocks = cld(n, threads)
    return (threads, blocks)
end

"""
    workgroup_size(arch::AbstractArchitecture, N)

Get the workgroup size for KernelAbstractions kernels.
Supports integer `N` or tuple/Cartesian nd-ranges (following Oceananigans.jl).
"""
workgroup_size(::CPU, N::Integer) = min(N, 64)
workgroup_size(::GPU, N::Integer) = 256

_ndrange_length(N::Integer) = N
_ndrange_length(N::Tuple) = prod(N)
_ndrange_length(N::CartesianIndices) = length(N)
_ndrange_length(N::AbstractArray) = length(N)
_ndrange_length(N::AbstractRange) = length(N)

"""
    _split_workgroup(budget, dims) -> NTuple

Distribute a `budget`-thread workgroup across `dims`, first dimension fastest,
never giving a dimension more threads than it has work items. The product never
exceeds `budget` (each step spends what the previous one left).

This is what makes a multi-dimensional `ndrange` usable. `workgroup_size` used to
return the bare scalar `budget` for every ndrange, and KernelAbstractions pads a
scalar workgroup with ONES (`NDIteration.partition`) — so a 2-D launch got a
`(256, 1)` block. When the first dimension is shorter than 256 every block still
spawns 256 threads and masks the excess off: the DCT-I / Chebyshev-derivative
kernels launch over `(2(n-1), batch)`, so a Chebyshev axis of n = 64 ran at 49%
occupancy and n = 32 at 24%. Splitting the budget across dimensions keeps every
thread in the block on real work.
"""
@inline _split_workgroup(::Int, ::Tuple{}) = ()
@inline function _split_workgroup(budget::Int, dims::Tuple)
    w = clamp(dims[1], 1, budget)
    return (w, _split_workgroup(max(budget ÷ w, 1), Base.tail(dims))...)
end

workgroup_size(arch::AbstractArchitecture, ndrange::Tuple) =
    _split_workgroup(workgroup_size(arch, _ndrange_length(ndrange)), ndrange)

function workgroup_size(arch::AbstractArchitecture, ndrange)
    return workgroup_size(arch, _ndrange_length(ndrange))
end

"""
    launch!(arch_or_data, kernel, args...; ndrange, dependencies=nothing, wait=true, kwargs...)

Oceananigans-style helper to launch KernelAbstractions kernels on CPU or GPU.
- `arch_or_data`: Architecture (`CPU()`, `GPU()`) or an array (architecture inferred)
- `kernel`: KernelAbstractions `@kernel` function
- `ndrange`: Number of elements/work items (integer or tuple)
- `dependencies`: Optional kernel dependencies
- `wait`: If true (default), wait for the returned event before returning

Returns the KernelAbstractions event, even when `wait=true`, so callers can
chain launches using `dependencies` similar to Oceananigans.
"""

"""
    ensure_device!(::AbstractArchitecture)

Activate the selected GPU before kernel launches and allocations. CPU execution
returns immediately.
"""
ensure_device!(::AbstractArchitecture) = nothing

function launch!(arch::AbstractArchitecture, kernel, args...;
                 ndrange,
                 dependencies=nothing,
                 wait::Bool=true,
                 kwargs...)
    ensure_device!(arch)
    backend = device(arch)
    workgroup = workgroup_size(arch, ndrange)
    ka_kernel = kernel(backend, workgroup)
    if dependencies === nothing
        event = ka_kernel(args...; ndrange=ndrange, kwargs...)
    else
        event = ka_kernel(args...; ndrange=ndrange, dependencies=dependencies, kwargs...)
    end
    if wait && event !== nothing
        Base.wait(event)
    end
    return event
end

function launch!(data::AbstractArray, kernel, args...; kwargs...)
    arch = architecture(data)
    return launch!(arch, kernel, args...; kwargs...)
end

# ============================================================================
# Printing
# ============================================================================

Base.summary(::CPU) = "CPU"
Base.summary(gpu::GPU) = "GPU"

Base.show(io::IO, ::CPU) = print(io, "CPU()")
Base.show(io::IO, gpu::GPU) = print(io, "GPU(device=$(gpu.device))")

# ============================================================================
# Exports
# ============================================================================

export AbstractArchitecture, AbstractSerialArchitecture
export CPU, GPU
export device, array_type, architecture
export on_architecture
export is_gpu, has_cuda, has_amdgpu, has_metal, synchronize, unsafe_free!
export is_gpu_array, create_array
export allocate_like, similar_zeros, copy_to_device
export launch_config, workgroup_size, launch!, KernelOperation, ensure_device!

"""
    KernelOperation(kernel; ndrange_fn=args -> length(args[1]))

User-facing wrapper that couples a KernelAbstractions `@kernel` with a default
`ndrange` computation. Calling the object launches via `launch!`, so backend
logic always flows through the architecture abstraction.

Supports do-block syntax:
    op = KernelOperation(kernel) do args...
        length(args[1])
    end
"""
struct KernelOperation{F,K}
    ndrange_fn::F  # Computes default ndrange from kernel arguments (first for do-block syntax)
    kernel::K
end

default_ndrange_fn(args...) = length(args[1])

# Keyword constructor: KernelOperation(kernel; ndrange_fn=default_ndrange_fn)
KernelOperation(kernel; ndrange_fn=default_ndrange_fn) = KernelOperation(ndrange_fn, kernel)

function (op::KernelOperation)(arch_or_data, args...; ndrange=nothing, dependencies=nothing, wait::Bool=true, kwargs...)
    effective_ndrange = isnothing(ndrange) ? op.ndrange_fn(args...) : ndrange
    return launch!(arch_or_data, op.kernel, args...; ndrange=effective_ndrange,
                   dependencies=dependencies, wait=wait, kwargs...)
end
