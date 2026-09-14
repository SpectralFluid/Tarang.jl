# CPU MPI Benchmark

Run from the repository root:

```sh
julia --project=. --startup-file=no scripts/benchmark_cpu_parallel.jl /tmp/tarang-cpu-n64
TARANG_BENCH_N=128 TARANG_BENCH_STEPS=100 TARANG_BENCH_PROFILE=1 julia --project=. --startup-file=no scripts/benchmark_cpu_parallel.jl /tmp/tarang-cpu-n128
```

The driver launches configurable MPI rank counts sequentially using MPI.jl's
configured launcher. Defaults are 1, 2, and 4 ranks with one Julia, FFTW, and BLAS
thread per rank. The workload is a periodic scalar nonlinear advection-diffusion
equation using SBDF2 and 3/2 Fourier padding, in either two or three dimensions.
It is a solver benchmark, not a substitute for timing a full application with
its velocity/pressure fields, boundaries, forcing, diagnostics and output.

Controls:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `TARANG_BENCH_N` | 64 | Grid points per dimension |
| `TARANG_BENCH_SHAPE` | `NxN` | Overrides N; e.g. `2048x2048` or `1024x1024x512` |
| `TARANG_BENCH_RANKS` | `1,2,4` | Ordered rank counts; the first is the comparison baseline |
| `TARANG_BENCH_THREADS` | 1 | Julia threads per worker launched by the driver |
| `TARANG_BENCH_FFTW_THREADS` | Julia benchmark threads | FFTW threads per rank |
| `TARANG_BENCH_BLAS_THREADS` | 1 | BLAS threads per rank |
| `TARANG_BENCH_MESH` | automatic | Explicit mesh, e.g. `40x25`; one rank count per driver invocation |
| `TARANG_BENCH_GATHER` | 1 in driver, 0 in external worker | Save and compare full fields; disable for large runs |
| `TARANG_BENCH_STEPS` | 30 | Timesteps per sample |
| `TARANG_BENCH_SAMPLES` | 3 | Number of timing samples |
| `TARANG_BENCH_PROFILE` | 0 | Set to 1 to collect separate CPU profiles |

Construction is warmed on a small problem large enough for the selected process
mesh, and the target problem takes ten
warmup steps before timing. Every sample starts after garbage collection and an
MPI barrier. Reported step time is the median of the samples, each taking the
maximum elapsed time across ranks. Allocations are also maximum per-rank values,
not cluster totals. Setup timing excludes package loading but includes target-size
planning and allocation; it is a single observation, not a robust setup estimate.

Peak RSS is the largest process high-water mark across ranks. It includes Julia,
compilation, and warmup, and is neither incremental solver memory nor simultaneous
total memory. Profiles run after measurements and the parity snapshot, so profiling
does not change the compared integration duration.

`ranks-N.toml` stores each run. With gathering enabled it includes the full final
field, and the driver fails if any field difference from the first run exceeds
`1e-9`. With gathering disabled, workers use reductions for the field sum,
squared norm and maximum magnitude. These diagnostics detect some discrepancies
but do not establish pointwise parity; reports explicitly say `diagnostics_only`.
External worker runs produce one rank report, without cross-run comparisons.
`summary.toml` stores driver-run metrics, speedup relative to the first rank count,
parallel efficiency, and relative core-time. The core-time proxy is rank count ×
maximum of Julia/FFTW/BLAS thread counts × elapsed time. It is neither measured
CPU consumption nor energy or billing; account separately for full-node allocation
and idle reserved cores. Optional `profile-N-rank-R.txt` files
contain flat sampling profiles; increase the step count for useful sample counts.

Use separate output directories for separate experiments. Repeat runs on an idle
machine and test production-sized grids before choosing a rank count. Use a new
output directory for each thread count, mesh, or repeated experiment. This is a
single-node strong-scaling experiment unless the configured launcher distributes
workers elsewhere; it does not establish multi-node scaling. The one-rank and
MPI solver execution paths differ, so the comparison measures the public API's
end-to-end behavior rather than an identical kernel under different rank counts.

## Rank and thread comparisons

Compare pure MPI and hybrid configurations within the same physical-core budget:

```sh
TARANG_BENCH_SHAPE=2048x2048 TARANG_BENCH_RANKS=1,2,4 TARANG_BENCH_THREADS=1 julia --project=. scripts/benchmark_cpu_parallel.jl results/2d-mpi
TARANG_BENCH_SHAPE=2048x2048 TARANG_BENCH_RANKS=1,2 TARANG_BENCH_THREADS=2 julia --project=. scripts/benchmark_cpu_parallel.jl results/2d-hybrid
```

The driver does not reserve or bind CPUs. Use your site's allocation and affinity
settings; ensure ranks × threads fits the assigned cores. More Julia threads
enable FFTW threading here but do not parallelize every per-mode solver loop.

## Externally launched cluster runs

Inside a scheduler allocation, use the site's MPI launcher matching the MPI
library selected by MPI.jl. Pass `--worker` to avoid nested MPI launchers. All
ranks must see the project and output directory. For example, with 1,000 allocated
cores across 40 nodes and one thread per rank:

```sh
TARANG_BENCH_MESH=40x25 TARANG_BENCH_FFTW_THREADS=1 mpiexec -n 1000 julia --threads=1 --project=. scripts/benchmark_cpu_parallel.jl --worker results/3d-1000-40x25 1024x1024x512 50 3
```

External workers default to no global gather. Their shape, steps and sample count
come from positional arguments; Julia threads come from `--threads`, FFTW threads
from `TARANG_BENCH_FFTW_THREADS`, and BLAS defaults to one. Repeat with fewer
nodes/ranks that fit memory, then compare median step time and **allocated
node-hours**. Test both `25x40` (the automatic 1,000-rank mesh) and `40x25`;
the best orientation depends on the rectangular grid and network. Treat 2D and
3D workloads independently; there is no measured 1,000-rank optimum yet.

Reports include the actual mesh and thread settings, rank-zero hostname, CPU
identifier, retained padded-workspace bytes summed across ranks and the maximum
on one rank. `padded_transposes_last_product` counts decomposition-changing
transposes; its companion element count is the global array volume involved,
not measured network bytes (self transfers and one-rank subcommunicators can
reduce network traffic). These counters cover the padded product, not all FFTs
or communication in a timestep.

## Local optimization measurements (2026-09-14)

On an Apple M2 with four MPI ranks and one Julia/FFTW/BLAS thread per rank,
a 128×128×64 padded product was measured before and after the optimization.
Three samples of 20 products followed ten warmup products; times are medians
of the slowest rank in each sample. Inputs were smooth real Fourier fields,
and the analytic product agreed within 2.0e-15 in both implementations.

| Metric | Before | After |
| --- | ---: | ---: |
| Median time per product | 95.37 ms | 85.88 ms |
| Retained padded scratch, all ranks | 688 MiB | 244 MiB |
| Allocations per product, maximum rank | 61.59 MiB | 4.21 MiB |
| Retained scratch arrays | 23 | 7 |
| Decomposition-changing transposes per product | 8 | 6 |

The scratch ratio is independent of grid size for this 3/2-padded pencil layout:
43 versus 15.25 full ComplexF64 grids. At 1024×1024×512, that extrapolates to
344 versus 122 GiB globally **for this workspace only**, excluding fields,
timestep history, FFT and communication buffers, and runtime overhead.
Timing samples vary; the roughly 10% reduction in this local product benchmark
does not predict full-application or 40-node speedup. Production rank/thread
selection remains a measurement task on the target cluster.

The full 2048×2048 SBDF2 benchmark (three samples of twenty timesteps) measured
670.50, 363.96 and 292.40 ms/step with one, two and four ranks after the change.
The earlier four-rank measurement was 354.62 ms/step, giving about an 18%
reduction in elapsed time. The four-rank speedup against its
contemporaneous one-rank baseline was 2.29×, with a 1.74× core-time proxy.
Final fields differed from the one-rank result by at most 6.2e-15. These are
single-node observations with timing variability, not cluster scaling results.
