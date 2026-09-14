# CPU parallel efficiency implementation plan

**Goal:** Reduce distributed padded-product memory and communication, and provide reproducible rank/thread benchmarks for 2D and 3D runs.

**Architecture:** Retain the existing PencilArrays decomposition and Fourier padding rules. Reuse geometry-compatible scratch arrays only after their last read, multiply padded operands in place, and truncate axes in reverse padding order to communicate smaller arrays and finish closer to the original layout. Expose per-call workspace transpose counts and volume for regression tests and benchmarks. Keep runtime threading defaults unchanged until cluster measurements justify changing them.

**Tech stack:** Julia, MPI.jl, PencilArrays/PencilFFTs, FFTW, Test, TOML.

- [x] Add an MPI regression for a rectangular 3D product: analytic values, unchanged inputs, repeated calls, retained memory and communication budgets. Observe failure before implementation; register in the MPI test inventory.
- [x] Implement lifetime-aware scratch reuse and reverse truncation in `src/core/nonlinear/nonlinear_evaluation.jl`. Validate existing real/complex, slab/pencil, and mixed-basis padded-product cases.
- [x] Extend `scripts/benchmark_cpu_parallel.jl` and its documentation with configurable shapes, rank counts, thread counts, meshes, and an externally launched worker mode. Avoid global snapshots for large cluster cases; report distributed diagnostics and resource metrics instead. Preserve the default serial/MPI parity check for small local runs.
- [x] Run local before/after timing and memory probes, plus benchmark smoke tests. Document measured improvements separately from estimates and pending cluster validation.
- [x] Remove materialized 3D padding/truncation slices discovered by the allocation benchmark; validate with a failing allocation guard and CPU/device-reference round trips.
- [x] Independent code review, including scratch lifetimes, transpose completion, precision coverage, and benchmark reporting.

Integration target: the existing PR #134. The user will run the production cluster benchmarks.

Validation commands use Julia's configured `MPI.mpiexec()` to launch separate 2- and 4-rank processes with `--project=.`. New memory budgets must bound global retained buffer elements, not process RSS. Transpose budgets count changes in decomposition, excluding local permutation-only copies. Numerical regressions must cover changing operands and ensure scratch reuse cannot overwrite input fields or returned products.

The available host is an Apple M2 with 16 GiB RAM. Cluster runs on the user's 1,000 Intel cores / 40 nodes require external access; this work prepares their commands and does not claim those measurements were performed.
