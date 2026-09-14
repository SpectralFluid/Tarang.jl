# Single-H200 optimization validation

Run from the checkout containing these changes, in a Julia environment containing
Tarang, CUDA 5, and the package's test dependencies. These commands require real
CUDA hardware and fail if it is unavailable:

```sh
julia --project=. test/run_single_gpu_optimizations.jl
julia --project=. test/run_gpu_fc_2d.jl
```

The first runner exercises real/complex nonlinear products, custom-stream
ordering, aliasing, device allocations, CFL buffer reuse, 3D mode batching, RK
stage fusion, batched FFT helpers, and optional iterative solvers. The second
checks complete Fourier–Chebyshev boundary-value and evolution paths.

To create an isolated test environment without modifying the package project,
run the following Julia commands from this checkout (the test target lists the
authoritative dependency inventory):

```julia
using Pkg
source = pwd()
Pkg.activate("/tmp/tarang-h200-tests")
Pkg.develop(path=source)
Pkg.add(["CUDA", "FFTW", "GPUArrays", "JLArrays"])
```

Then use `--project=/tmp/tarang-h200-tests` in the commands above and below.

## Repeatable benchmark

```sh
SHAPE=2048x2048 OUTPUT=h200-2048.toml julia --project=. scripts/benchmark_single_gpu.jl
SHAPE=128x128x128 OUTPUT=h200-128-3d.toml julia --project=. scripts/benchmark_single_gpu.jl
SHAPE=256x256x256 OUTPUT=h200-256-3d.toml julia --project=. scripts/benchmark_single_gpu.jl
```

The benchmark uses Float64, 3/2 dealiasing, and RK222 for a scalar nonlinear
advection–diffusion equation. It separately measures a padded product and a full
timestep. Defaults are ten warmup calls, five samples, twenty calls per sample,
and `dt=1e-4`; override `WARMUP`, `SAMPLES`, `STEPS`, or `DT` as needed. Increase
warmup if compilation or caches have not settled. Each timing sample waits for
GPU completion at its boundaries. Output and diagnostic reductions are outside
the timed blocks.

The TOML file contains milliseconds/call, cumulative device-allocation deltas per
call, explicit nonlinear scratch bytes, result-pool size, GPU name, and free
device memory after the run. Free memory includes allocator-pool effects and is
**not peak live memory**. Use a CUDA profiler for peak memory, kernel/FFT counts,
host waits, and transfer timelines. Compare full applications with their normal
output/diagnostic cadence as a separate measurement.

Start each configuration in a fresh process. Increase 3D size after checking its
complete memory footprint. A scalar benchmark fitting does not establish that a
multi-field Navier–Stokes or MHD run fits. Compare baseline and updated code with
the same grids, precision, equations, scheme, dt, and initial state; the benchmark
script can also be copied into the baseline checkout. Record both commits.

## What changed

| Area | Implementation | Scope or tradeoff |
| --- | --- | --- |
| Complex padded products | Two padded arrays and one original spectrum; product overwrites an operand | No precision or dealiasing change |
| Real padded products | Two padded real grids plus padded/original half spectra | cuFFT plan preservation buffers are reused as those spectra; owned inverse inputs are consumed |
| Nonlinear RHS | Writes directly into its destination | Public returned products remain independently owned |
| Repeated operands | One cached padded operand uses the existing second buffer | Valid only within a frozen compiled RHS equation evaluation; no extra padded grids |
| Coefficient layouts | Pure-Fourier compatible inputs/output avoid original-grid round trips | Mixed/scaled or incompatible storage uses the existing grid path; real boundary planes retain Hermitian projection |
| CUDA padding | Ordered device work replaces explicit host waits; truncation no longer clears a fully overwritten destination | Existing single-task nonlinear workspace contract applies |
| 3D coupled RK | Eligible two-Fourier-axis subproblems batch | Aggregate matrix/RK working-set cap retains per-mode fallback; this is not an unlimited dense batch |
| RK combinations | Fused explicit and serial diagonal-IMEX stage arithmetic | Coefficient order, tiny-dt contributions, ownership, and schemes preserved |
| CFL | Reuses one frequency grid per registered velocity and fuses component accumulation | Device reductions may still allocate small scratch and return a host scalar |
| Batched FFT helpers | Reuse packed input/output storage | Plans and buffers are bound to task/device/stream; fetch a plan in the context that executes it |
| CG/GMRES | Reusable `solve!` vectors, Krylov basis, and host workspace | Public `solve` returns owned results; each concurrent solve needs its own solver/preconditioner instance |

For 1024×1024×512, explicit complex nonlinear scratch decreases from **105 GiB to
62 GiB**. The real-input layout uses approximately **44.5 GiB**, including the
half-spectrum storage shared with supported CUDA 5 plans. At 2048² the
corresponding figures are **624 MiB → 352 MiB**, or approximately **248 MiB** for
real input. These estimates exclude solution fields, timestep history, cuFFT
internal work areas, and allocator reserve. CUDA versions without the internal
trailing-axis executor use the public inverse fallback, whose preservation
buffers/copies may add overhead; inspect the actual version and memory report.

The optimizations have CPU and hardware-gated regression coverage. CPU-backed
tests establish arithmetic and ownership behavior, not H200 speed or cost.
No measured H200 speedup is claimed. For your own equations, also compare boundary
residuals and relevant conservation/dissipation diagnostics over a meaningful
simulation interval.
