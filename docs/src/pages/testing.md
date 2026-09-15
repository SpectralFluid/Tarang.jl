# Testing

Guide to running and writing tests for Tarang.jl.

## Running Tests

### Full Test Suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Specific Test File

Most feature test files can be included on their own. Check the file's imports
and setup first; registry checks and runner helpers need their driver context:

```bash
julia --project=. -e 'using Test, Tarang; include("test/test_solvers.jl")'
```

Files that emulate a GPU with `JLArrays` (`test_gpu_*_jlarray.jl`,
`test_2d_gpu_domain_compat.jl`, `test_gpu_2d_device_stack.jl`, ...) need the
test dependencies, which resolve only under `Pkg.test()`; run those through the
full suite, or from a scratch environment that `Pkg.develop`s this checkout and
adds `JLArrays`. The test directory has no standalone `Project.toml`;
`--project=test` does not provide the package test environment.

The optional and GPU groups are switched on with environment variables read by
`test/runtests.jl`:

| Variable | Effect |
|---|---|
| `TARANG_RUN_OPTIONAL_TESTS=true` | also run `OPTIONAL_TEST_FILES` |
| `TARANG_ONLY_OPTIONAL_TESTS=true` | run only `OPTIONAL_TEST_FILES` |
| `TARANG_RUN_GPU_TESTS=true` | also run `GPU_TEST_FILES` (they skip without CUDA) |

### With MPI

The multi-rank MPI tests each run in their own MPI world, via a driver
(CI exercises 1, 2, and 4 ranks):

```bash
julia --project=. test/run_mpi_ci.jl 4      # all MPI tests at 4 ranks
./test/run_mpi_tests.sh 4                    # convenience wrapper
```

### GPU

GPU tests need an NVIDIA GPU and run on JuliaGPU Buildkite CI (see
[Continuous Integration](#Continuous-Integration)). To run them locally on a
CUDA host:

```bash
julia --project=@v#.# -e 'using Pkg; Pkg.add("CUDA")' # keep this checkout clean
julia --project=. test/run_gpu_ci.jl                # single-process GPU tests
julia --project=. test/run_gpu_fc_2d.jl             # strict focused 2D FC validation
# distributed (NCCL) tests across, e.g., 2 GPUs:
TARANG_MPI_FILESET=distributed_gpu julia --project=. test/run_mpi_ci.jl 2
```

`test/run_gpu_fc_2d.jl` is intended for a single NVIDIA device on a cluster. It
requires functional CUDA, disables scalar indexing, prints CUDA device
information, and runs the CPU/GPU value and allocation checks for the complete
2D Fourier--Chebyshev path. It exits nonzero rather than skipping when CUDA is
missing. For ordinary CPU development, running
`test/test_gpu_fc_2d_complete.jl` directly is safe and reports one skipped
testset when no functional device is present.

## Test Structure

There is no hand-maintained list of test files in the runner. `test/file_lists.jl`
is the single registry, and every driver reads it:

| List | Run by |
|---|---|
| `TEST_FILES` | `Pkg.test()` on every CPU CI job |
| `OPTIONAL_TEST_FILES` | `Pkg.test()` with `TARANG_RUN_OPTIONAL_TESTS=true` |
| `GPU_TEST_FILES` | `test/run_gpu_ci.jl` on a CUDA host (Buildkite) |
| `MPI_TEST_FILES` | `test/run_mpi_ci.jl [nprocs]`, one MPI world per file |
| `DISTRIBUTED_GPU_TEST_FILES` | `TARANG_MPI_FILESET=distributed_gpu julia --project=. test/run_mpi_ci.jl 2` (CUDA + NCCL) |

`test_test_inventory.jl` rejects missing, unregistered, or untracked test files.
Register each new file in its execution groups; CPU/CUDA and serial/MPI groups
may overlap.

| Feature | Coverage |
|---|---|
| Stochastic forcing | `test_stochastic_forcing.jl` and `test_stochastic_checkpoint_restart.jl`: CPU and CUDA groups; restart also tests JLArrays staging |
| Distributed restart | `test_mpi_checkpoint_restart.jl`: multi-rank checkpoint and forcing restoration |
| LES closures | `test_les_models.jl`: CPU/CUDA comparisons with independent tensor contractions |
| AMD numerical range | `les_range_helpers.jl`: analytic small/large-gradient cases, 2D/3D, Float32/Float64, both clipping modes; shared by `test_les_models.jl` and `test_les_models_gpu_compat.jl` for CPU, CUDA, and JLArrays |

A skipped CUDA test does not verify hardware execution.

Beyond feature tests, several files are *ratchets* that pin a population the
codebase must not grow — `test_layout_discipline_ratchet.jl`,
`test_backend_dispatch_ratchet.jl`, `test_hasfield_ratchet.jl`,
`test_buffer_ownership_ratchet.jl`, `test_decomposition_convention.jl`, and the
JET/Aqua files. Their header comments explain what each count guards.

## Testing GPU paths without a GPU

The default CPU suite checks device behavior in three ways:

- **JLArrays:** CPU-backed device arrays exercise GPU dispatch with scalar
  indexing disabled. `test_gpu_timesteppers_jlarray.jl` checks every timestepper;
  `test_gpu_boundary_regressions_jlarray.jl` checks boundary solves and moving
  walls. Explicit host FFT and sparse-LU stand-ins support these tests; they do
  not validate cuFFT or CUDA solvers. Native coverage is in
  `test_gpu_fc_2d_complete.jl`.
- **KernelAbstractions CPU backend:** `test_gpu_dct1_kernels_cpu.jl`,
  `test_gpu_transpose_kernels_cpu.jl`, and `test_gpu_kernels_cpu.jl` launch the
  shared kernels over CPU arrays to check indexing and normalization.
- **GPU branches on CPU arrays:** `test_timestepper_boundaries.jl` forces
  `_gpu_subproblem_execution(sp)` to compare batched solves and constrained
  final updates with the CPU path.

Choose inputs that exercise the intended path: for example, nonzero boundary
conditions when testing boundary updates. Confirm that breaking the guarded
behavior makes the relevant assertion fail.

## Documentation code is tested

`test_webdocs_code.jl` parses every Julia, Bash, TOML, and Dockerfile fence under
`docs/src` on every run. A fence that does not parse fails the suite, so
pseudo-code belongs in a `text` fence. With `TARANG_RUN_WEBDOCS_EXAMPLES=true`
the self-contained Julia examples (fences containing `using Tarang`) are also
executed in fresh temporary directories; `TARANG_WEBDOCS_FILTER` narrows the run
to matching files.

## Writing Tests

Use self-contained setup and compare numerical results with an independent
reference. For example, this Fourier round-trip checks a nonconstant field:

```julia
using Test, Tarang

@testset "Fourier round-trip" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype=Float64)
    basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    field = ScalarField(dist, "f", (basis,), Float64)
    x, = local_grids(dist, basis)
    expected = sin.(x) .+ 0.25 .* cos.(2 .* x)

    ensure_layout!(field, :g)
    get_grid_data(field) .= expected
    ensure_layout!(field, :c)
    ensure_layout!(field, :g)

    @test get_grid_data(field) ≈ expected atol=1e-12
end
```

## Test Patterns

- Compare solver output with analytic solutions and check convergence under
  spatial or timestep refinement; see `test_rksmr_convergence.jl`.
- Compare CPU and device results with independent references, including nonzero
  inputs and limiting cases; see `test_les_models.jl`.
- Put collective MPI setup and teardown in each MPI test file and register it
  in `MPI_TEST_FILES`; the driver runs each file in a separate MPI world.

## Test Coverage

Collect coverage while running the suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'
```

The GitHub Actions CI workflow processes and uploads the resulting coverage.

## Continuous Integration

CPU tests run on **GitHub Actions** for every push and pull request:

- the default suite on Julia 1.10/1.11/1.12 across Linux, macOS, and Windows;
- the optional CPU feature tests (`TARANG_ONLY_OPTIONAL_TESTS=true`);
- the MPI suite via `test/run_mpi_ci.jl` at 1, 2, and 4 ranks.

GPU tests cannot run on GitHub-hosted runners (no NVIDIA GPU), so they run on
**Buildkite**, defined in `.buildkite/pipeline.yml`:

- a single-GPU job (`test/run_gpu_ci.jl`) on Julia 1.10/1.11/1.12.

Because CUDA is a *weak* dependency — which keeps CPU installs lean — that job
`Pkg.add`s CUDA before running instead of using the standard package test target.

The multi-GPU NCCL step is disabled in `.buildkite/pipeline.yml`. It requires
an agent with two physical GPUs and a `multigpu=true` tag. Its test files remain
registered and parse-checked by CPU CI.

### Which agent runs it

Use a self-hosted agent with an NVIDIA GPU. Select its queue through the
pipeline's `TARANG_GPU_QUEUE` environment variable:

| `TARANG_GPU_QUEUE` | Queue used |
|---|---|
| unset | `default` — a self-hosted agent that sets no queue |
| `juliagpu` | JuliaGPU's shared GPU pool |

Set it under Pipeline Settings > Environment Variables.

The agent must also carry a `cuda` tag, since the step requires `cuda: "*"`:

```bash
buildkite-agent start --tags "queue=default,cuda=true"
```

Without matching queue and CUDA tags, the job remains queued.

### When it runs

The branch guard in `.buildkite/pipeline.yml` selects which builds run GPU tests:

| Build source | Runs the GPU job on |
|---|---|
| push / pull request (Buildkite GitHub App) | `main` only |
| Buildkite UI, "New Build" button | any branch |
| GitHub Actions, the **GPU (Buildkite)** workflow | any branch |

To test a branch before merging, dispatch **GPU (Buildkite)** from GitHub Actions
with its branch name, or use **New Build** in Buildkite. The GitHub workflow
requires a `BUILDKITE_API_TOKEN` repository secret with `write_builds` scope;
`BUILDKITE_ORG` and `BUILDKITE_PIPELINE` repository variables override the default
pipeline. See `.github/workflows/gpu-buildkite.yml` for setup.

Filtered builds can finish green without running tests; check the job results.

Add `[skip tests]` to a commit message to suppress the GPU job regardless of how
the build was created.

### Reporting a GPU run by hand

`scripts/gpu_ci_report.sh` runs the suite locally and posts a GitHub commit
status, visible on the commit and its pull requests:

```bash
./scripts/gpu_ci_report.sh                 # test HEAD, post a gpu/cuda status
./scripts/gpu_ci_report.sh --sha 6a4da42   # require the checkout to be this commit
./scripts/gpu_ci_report.sh --no-status     # run only, post nothing
./scripts/gpu_ci_report.sh --gist          # also upload the log as a secret gist
```

The reporter requires a clean checkout, including no untracked files; `--sha`
must match `HEAD`. Posting status requires authenticated `gh` with `repo:status`
permission. Install CUDA in Julia's stacked default environment as shown above.
The script requires `CUDA.functional()` and reports a missing device as an error,
so skipped tests cannot produce a successful hardware status.

## See Also

- [Contributing](contributing.md): Development guidelines
- [Architecture](architecture.md): Code structure
