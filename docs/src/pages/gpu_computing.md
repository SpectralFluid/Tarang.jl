# GPU Computing

Tarang uses CUDA.jl for NVIDIA GPUs, AMDGPU.jl for AMD ROCm GPUs and Metal.jl for Apple silicon. Install the package for your hardware in the environment where you run Tarang. On AMD, select the backend explicitly with

```julia
using Tarang, AMDGPU
@assert has_amdgpu()
arch = GPU(backend=:amdgpu, device_id=0)
```

## Quick start

On NVIDIA, load `using Tarang, CUDA` and use `arch = GPU(backend=:cuda)`. Device IDs start at zero. `GPU()` tries CUDA, AMDGPU and Metal in that order. Fields and solvers inherit the architecture from their distributor.

```julia
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; device=arch)

xb = RealFourier(coords["x"]; size=256, bounds=(0.0, 2π))
yb = RealFourier(coords["y"]; size=256, bounds=(0.0, 2π))
domain = Domain(dist, (xb, yb))
u = ScalarField(domain, "u")

copyto!(get_grid_data(u), rand(dist.dtype, size(get_grid_data(u))))
forward_transform!(u)
backward_transform!(u)

@assert architecture(get_grid_data(u)) == arch
```

On Apple silicon, install Metal.jl in the same environment as Tarang. Keep that environment separate from CUDA.jl 5 and AMDGPU.jl 2.4–2.8 because their compiler dependencies conflict with Metal.jl 1.11. Metal fields default to `Float32`. Use `ComplexF32` with complex Fourier fields. Each Metal distributor uses one MPI rank.

```julia
using Tarang, Metal
@assert has_metal()
arch = GPU(backend=:metal)
domain = PeriodicDomain(64, 64; device=arch)
u = ScalarField(domain, "u")
set!(u, (x, y) -> sin(x) * cos(y))
coeff_data!(u)
```

Metal uses MPSGraph FFTs and the shared Chebyshev kernels. To check the Metal device and numerical results against FFTW and analytical solutions, run the hardware tests from a checkout.

```bash
julia --project=. test/run_metal_ci.jl --require-device
```

Use `CPU()` instead of `GPU()` to construct a CPU simulation.

## Strict GPU execution

A GPU field is never downloaded to run an unavailable CPU implementation.
Supported operations remain on the selected device; unsupported operations
raise an error. This applies to transforms, resampling, derivatives, solver
vectors, MPI communication, and stochastic phase generation.

Host transfers still occur when explicitly requested, for example:

```julia
host_data = on_architecture(CPU(), u["g"])
```

Output, checkpoints, gathers, and explicit architecture conversions are data
movement APIs, not computational fallbacks.

## Transforms

GPU transforms run on the device for every array size.

```julia
set_gpu_fft_mode!(u, :auto)  # default; device-resident
set_gpu_fft_mode!(u, :gpu)   # explicitly require the GPU backend
```

`set_gpu_fft_mode!(u, :cpu)` is rejected for GPU fields. The legacy global FFT
size threshold applies only to CPU-field preferences and cannot move a GPU field
to FFTW.

Current single-GPU transform support is:

| Bases | Support |
|-------|---------|
| Real or complex Fourier | Device FFT |
| 2D Fourier × Chebyshev | Device FFT plus DCT-I, including scaled grids |
| 3D Fourier × Chebyshev | Device FFT plus DCT-I for supported layouts |
| Pure Chebyshev, up to 3D | Device DCT-I, including scaled grids |
| Legendre transforms | Explicitly unsupported |

AMD uses rocFFT for Fourier transforms and the shared device kernels for Chebyshev DCT-I transforms and derivatives. Run the AMD validation from a checkout in an environment containing Tarang and AMDGPU.

```bash
julia --project=. test/run_amdgpu_ci.jl --require-device
```

The flag requires an actual ROCm device. Without it, the runner also works on a CPU machine and checks extension loading and the FFT axis layouts against FFTW. Launch the same runner with two or more MPI ranks to check the distributed pencil transforms.

Unsupported basis and layout combinations fail before entering the CPU
transform chain.

On CUDA, AMDGPU and Metal, field transforms reuse plans and scratch buffers
within each distributor, Julia task and GPU queue. Closing the distributor drops
its references to these resources. During a resolution sweep, call
`clear_gpu_workspaces!(dist)` between transforms to release cached scratch while
keeping the fields usable. The backend's memory pool may retain released memory.

Real periodic grid resizes use real FFTs and store only the independent half of
each spectrum. A resize and its reverse reuse the same pair of workspaces.
`set_scales!` still allocates the field's new grid whenever its shape changes.

CUDA async real inverse FFTs keep a cuFFT plan per Julia task and stream.
The input coefficients are preserved. CUDA.jl versions that expose a plan-owned
half-spectrum buffer can reuse that buffer for repeated calls.

### Complete 2D Fourier--Chebyshev path

The single-GPU 2D path supports `RealFourier` or `ComplexFourier` paired with
`ChebyshevT`, with either basis first. Scaled Chebyshev grids are transformed at
their nodal length and truncated or zero-padded on the device; Fourier scaling
used by 3/2-rule nonlinear products also remains device-resident. The focused
validation covers transforms, derivatives along both axes, dealiased products,
and a nonlinear wall-bounded RK222 InitialValueProblem with tau boundary conditions and CUDA
sparse subproblem solves.

Run the strict validation on an NVIDIA node from the repository root:

```bash
julia --project=. test/run_gpu_fc_2d.jl
```

This command prints CUDA device information and fails if CUDA is unavailable,
scalar indexing is attempted, a CPU/GPU value comparison fails, or the warmed
InitialValueProblem step performs a fresh device allocation.

The 2D validation runs on one GPU. For supported multi-rank 3D layouts,
use the distributed interfaces below. Concurrent transforms need separate
fields because a transform mutates its field's data and layout.

## 2D time stepping and solves

Pure-Fourier GPU IVPs use field-native stepping: explicit Runge-Kutta and the
matrix-free multistep field path run entirely on device arrays. When the
left-hand side has an implicit diagonal Fourier operator, `RK222()`, `RK443()`,
`SBDF2()` and the other IMEX Runge-Kutta tableaux (`RK111()`, `RKSMR()`,
`RKGFY()`) select their diagonal implementations automatically. The operator
is applied per mode in spectral space
on the device; the operator is read from the equation (Laplacian, hyper- or
fractional Laplacian, constant damping, derivatives of the stepped field) or
from an attached `SpectralLinearOperator`. Every other scheme refuses a
left-hand-side operator on a single GPU with an error that names these six,
because a pure-Fourier GPU solver assembles no global matrix to treat it with,
and refuses an attached operator with an `ArgumentError`.
The full per-architecture table is in
[Time Steppers](timesteppers.md#Where-each-scheme-runs).

Fourier–Chebyshev IVPs use per-mode coupled subproblems, batched across Fourier
modes by default on CUDA and ROCm (`batched_modes=false` opts out). Metal solves one Fourier mode at a time. With GPU fields,
`matsolver=:auto`, `:gpu`, and `:hybrid` select a backend-specific solver. Individual mode solves use `:cuda_sparse` on NVIDIA and `:rocm_dense` on AMD. On Metal, `:metal_dense` factors the equivalent real single-precision system with pivoted LU in Metal Performance Shaders. It retains the factors in native row-major layout and reuses the solve buffers for subsequent right-hand sides of the same shape. The AMD solver uses rocSOLVER for dense LU and rocSPARSE for sparse matrix products. A singular stage matrix raises a factorization error.

Most of this is covered without hardware: `test/test_gpu_timesteppers_jlarray.jl`
drives all twenty schemes on the single-GPU dispatch path using `JLArray` device
fields and a CPU stand-in for cuFFT (see
[Testing](testing.md#Testing-GPU-paths-without-a-GPU)).

Linear GPU boundary-value solves select the field's device solver automatically.
Their right-hand side and solution buffers stay on the field's device backend.
An explicit solver selection is also supported:

```julia
solver = BoundaryValueSolver(problem; matsolver=:rocm_dense)  # AMD
solver = BoundaryValueSolver(problem; matsolver=:metal_dense) # Apple
```

GPU NonlinearBoundaryValueProblem and EigenvalueProblem solves are not device-native yet and raise an unsupported
operation error.

## Multi-GPU execution

Assign one device to each MPI rank before allocating fields:

```julia
using Tarang, CUDA, MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
gpu_id = rank % CUDA.ndevices()
CUDA.device!(gpu_id)

arch = GPU(device_id=gpu_id)
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; comm=comm, dtype=ComplexF64, device=arch)
```

The example above uses CUDA-aware MPI. AMD uses ROCm-aware MPI for all distributed transposes. When a scheduler exposes one GPU per rank, each rank can use device ID zero. When all GPUs on a node are visible, use the node-local rank as the device ID.

```julia
@assert MPI.has_rocm()
node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
arch = GPU(backend=:amdgpu, device_id=MPI.Comm_rank(node))
MPI.free(node)
```

Pure complex-Fourier domains transform through the ordinary call. A field on a
GPU architecture with more than one rank is given `TransposableFieldStorage`
when it is constructed, and `forward_transform!` routes it through the explicit
transposes automatically:

```julia
bases = (
    ComplexFourier(coords["x"]; size=128, bounds=(0.0, 2π)),
    ComplexFourier(coords["y"]; size=128, bounds=(0.0, 2π)),
    ComplexFourier(coords["z"]; size=128, bounds=(0.0, 2π)),
)
field = ScalarField(Domain(dist, bases), "u")
forward_transform!(field)
backward_transform!(field)
```

The transpose workspace (buffers, counts, and two MPI sub-communicators) is
cached on the `Distributor` and shared by every field of the same shape and
element type, so allocating many fields does not allocate many communicators.
`close(dist)` frees them collectively.

Driving the wrapper by hand still works and is what to use if you need the
lower-level control:

```julia
tf = TransposableField(field)
distributed_forward_transform!(tf)
distributed_backward_transform!(tf)
```

These require the field to be in the matching layout first (`:g` for forward,
`:c` for backward); they raise rather than silently reading the stale buffer.

The distributed DCT-I path supports selected three-dimensional
Fourier–Chebyshev layouts. It requires at least one Fourier and one Chebyshev
axis. `RealFourier` is allowed only on the first axis and cannot be combined
with another Fourier axis. Other layouts, including pure Chebyshev, are rejected.
`forward_transform!` and `backward_transform!` select this path automatically
for such fields. AMD transposes exchange ROCArray buffers through MPI. NVIDIA uses NCCL for this path. Load NCCL.jl (`using NCCL`) on
every rank before the first transform (calling `init_nccl!` is not required).
Without it every rank raises the same error before any communication starts.
The NCCL communicators belong to the cached transform plans; `close(dist)` destroys
them explicitly — `ncclCommDestroy` is collective, so call it on every rank rather
than leaving it to garbage collection, which runs at a different moment on each
rank. A mesh may have more ranks than points along an axis: ranks whose block is
empty skip their local transforms but still take part in every exchange.
A hand-built multi-rank `TransposableField` refuses a real field with a
`RealFourier` axis, because its fixed-size buffers cannot hold the half spectrum.

## Custom kernels

Use the public architecture layer instead of importing implementation details
from `TarangCUDAExt`:

```julia
using KernelAbstractions: @kernel, @index

@kernel function scale_kernel!(y, x, α)
    i = @index(Global)
    @inbounds y[i] = α * x[i]
end

scale = KernelOperation(scale_kernel!) do y, x, α
    length(y)
end

arch = GPU()
x = ones(arch, Float64, 1024)
y = similar(x)
scale(arch, y, x, 2.0)
```

`KernelOperation` accepts an architecture or an array as its first argument;
passing `arch` makes the launch target explicit.

## Memory and profiling

Each backend manages its own device memory pool. On AMD, `AMDGPU.versioninfo()` reports the active devices and available ROCm libraries. On NVIDIA, CUDA.jl provides memory and profiling tools.

```julia
CUDA.memory_status()
CUDA.@profile begin
    forward_transform!(u)
    backward_transform!(u)
end
CUDA.reclaim()
```

See the [GPU API reference](../api/gpu.md) for the public architecture,
transform, and distributed interfaces.
