# GPU API Reference

Load Tarang and the package for your GPU. Select the device on the distributor.

```julia
using Tarang, AMDGPU
arch = GPU(backend=:amdgpu, device_id=0)
dist = Distributor(coords; dtype=Float64, device=arch)
```

For NVIDIA, load CUDA.jl and select `GPU(backend=:cuda)`. For Apple silicon, load Metal.jl and select `GPU(backend=:metal)`. Metal fields use `Float32` or `ComplexF32` on a single rank. Device IDs start at zero. Automatic selection tries CUDA, AMDGPU and Metal in that order. Fields inherit their distributor's architecture.

## Architectures

```@docs
GPU
CPU
AbstractArchitecture
```

Useful public helpers include:

| Function | Purpose |
|----------|---------|
| `architecture(array)` | Return the array architecture |
| `is_gpu(arch)` | Test an architecture |
| `is_gpu_array(array)` | Test whether storage is device-resident |
| `has_cuda()` | Report whether Tarang can use CUDA |
| `has_amdgpu()` | Report whether Tarang can use an AMD ROCm device |
| `has_metal()` | Report whether Tarang can use an Apple Metal device |
| `ensure_device!(arch)` | Activate the architecture device |
| `synchronize(arch)` | Wait for queued work |
| `array_type(arch)` | Return the architecture array type |

## Allocation and explicit transfers

Architecture-aware allocation uses the architecture as the first argument:

```julia
a = zeros(GPU(), Float32, 128, 128)
b = ones(GPU(), Float32, 256)
c = create_array(GPU(), ComplexF32, 16, 16)
```

Use `on_architecture` for an explicit transfer:

```julia
a_host = on_architecture(CPU(), a)
a_device = on_architecture(GPU(), a_host)
```

`allocate_like`, `similar_zeros`, and `copy_to_device` infer the target from an
existing array. Each GPU package manages its own memory pool. `unsafe_free!` can release a large device buffer eagerly.

Explicit output, checkpoint, gather, and architecture-conversion APIs may copy
data to the host. GPU computation itself does not silently switch to CPU.

## Transform dispatch

| Function | Purpose |
|----------|---------|
| `forward_transform!(field)` | Grid to coefficient layout |
| `backward_transform!(field)` | Coefficient to grid layout |
| `gpu_fft_mode(field)` | Return `:auto`, `:gpu`, or `:cpu` |
| `set_gpu_fft_mode!(field, mode)` | Set the field transform mode |
| `should_use_gpu_fft(field)` | Report whether the device path is selected |
| `clear_gpu_workspaces!(dist)` | Release the distributor's cached local GPU transform plans and scratch buffers |

For a GPU field, `:auto` and `:gpu` both remain on-device for every array size.
`:cpu` is rejected. An unsupported transform raises an error instead of
downloading the field and calling FFTW.

`set_gpu_fft_min_elements!` and `gpu_fft_min_elements` are legacy CPU-field
preference controls; they do not affect GPU fields.

See [GPU Computing](../pages/gpu_computing.md#Transforms) for the supported
Fourier and Chebyshev combinations.

## Portable kernels

`KernelOperation` wraps a KernelAbstractions kernel and launches it through the
selected architecture.

```julia
using KernelAbstractions: @kernel, @index

@kernel function add_kernel!(c, a, b)
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
end

add = KernelOperation(add_kernel!) do c, a, b
    length(c)
end

arch = GPU()
a = ones(arch, Float64, 1024)
b = ones(arch, Float64, 1024)
c = similar(a)
add(arch, c, a, b)
```

The lower-level `launch!` accepts an architecture or an array and calls
`ensure_device!` before launching. Import KernelAbstractions names selectively;
both packages export a type named `CPU`.

## Distributed GPU transforms

The public distributed interface includes:

| Function or type | Purpose |
|------------------|---------|
| `check_cuda_aware_mpi()` | Check CUDA-aware MPI support |
| `MPI.has_rocm()` | Check ROCm-aware MPI support with MPI.jl |
| `TransposableField(field)` | Create distributed transpose storage |
| `distributed_forward_transform!(tf)` | Distributed forward transform |
| `distributed_backward_transform!(tf)` | Distributed inverse transform |
| `active_layout(tf)` | Return the active transpose layout |
| `current_data(tf)` | Return the active local buffer |
| `local_shape(tf, layout)` | Return a local layout shape |

Multi-GPU runs must select a device per MPI rank:

```julia
using MPI
comm = MPI.COMM_WORLD
node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
arch = GPU(backend=:amdgpu, device_id=MPI.Comm_rank(node))
MPI.free(node)
```

Launch one rank per visible GPU on each node for this example. GPU MPI communication requires an MPI library that accepts buffers from the selected backend. Pure complex-Fourier domains use `TransposableField`.
The limited distributed Fourier–Chebyshev DCT-I path is described in
[GPU Computing](../pages/gpu_computing.md#Multi-GPU-execution).

## Solver behavior

GPU runtime state and solver vectors stay on the device. Coupled IVPs and linear boundary-value problems select `:cuda_sparse` on NVIDIA, `:rocm_dense` on AMD and `:metal_dense` on Apple silicon. Metal solves each mode in single precision with Metal Performance Shaders. CUDA and ROCm also support batched mode solves. GPU NonlinearBoundaryValueProblem and EigenvalueProblem solves are currently unsupported.

Unsupported solver operations and factorization failures raise errors rather
than retrying with a CPU solver.

## Extension boundary

The GPU kernels and vendor-specific transform plans under `ext/` are implementation details.
Application code should use the public architecture, transform, solver, and
`KernelOperation` interfaces documented above.
