# Single-GPU optimization implementation plan

> Implementation and review use the superpowers:subagent-driven-development workflow.

**Goal:** Address every opportunity in the single-H200 audit while preserving numerical methods and ownership contracts.

**Architecture:** Reuse nonlinear storage and accept explicit destinations; introduce real-input and coefficient-aware paths with complex fallback. Keep dependent CUDA operations stream ordered. Extend existing bounded mode batching, fuse stage combinations, and reuse CFL/FFT/iterative-solver execution buffers.

**Tech stack:** Julia, FFTW, CUDA.jl 5, KernelAbstractions, cuFFT/cuBLAS, MPI.

## Tasks and verification

- [x] Baseline: run nonlinear and spectral-padding tests in this isolated worktree.
- [x] Nonlinear memory: write storage-budget and alias/ownership regression tests; observe old implementation fail the budget; reduce six complex arrays to three and add destination output. Files: `src/core/nonlinear/nonlinear_padding.jl`, `nonlinear_evaluation.jl`, `src/core/solvers/lazy_rhs.jl`.
- [x] Transforms: test real/complex, odd/even, mixed axes, Nyquist corners, coefficient input/output; add reduced real FFT workspace and avoid round trips where field conventions permit. Cache repeated operands only within an explicit RHS scope with bounded storage.
- [x] CUDA padding: verify stream ordering against installed CUDA/KA implementation, remove redundant host waits and result clear, add strict CUDA regressions.
- [x] Mode batching/RK: extend gather/scatter eligibility to coupled 3D modes with bounded batches and fuse field-stage combinations. Test numerical parity, actual batching engagement and retained small-dt contributions. Files: mode-batch helpers and `src/core/timesteppers/`.
- [x] CFL: test warmed allocation and unchanged adaptive dt; cache/fuse frequency buffers in `src/extras/flow_tools/flow_tools_cfl.jl`.
- [x] Optional solvers/FFT: cache batched FFT execution buffers with clear concurrency ownership; reuse CG/GMRES buffers and in-place operations while preserving convergence semantics. Files: `ext/cuda/batched_fft.jl`, `src/tools/gpu_matsolvers.jl`.
- [x] Validation: run targeted CPU/JLArray, nonlinear ownership, timestep and MPI regressions; provide a strict CUDA runner and repeatable benchmark reporting time, allocation and memory. Actual H200 timings are user-run.
- [x] Review each component and the integrated diff for scope and correctness; resolve findings. Publish the reviewed changes on the separate `perf/single-gpu-nonlinear` branch and draft PR.

Each implementation task follows red/green testing. No GPU performance claim is supported by CPU-backed tests. Any architectural limitation discovered during implementation must be documented explicitly rather than silently changing precision, schemes, convergence criteria or supported mathematics.

## Local validation record

Validated on macOS with Julia 1.13; CUDA 5.11.3 loads but no CUDA device is available.

- Nonlinear storage, real-spectrum mappings, coefficient layouts, mixed precision, destination ownership, and scoped operand reuse pass CPU regressions.
- RK/mode-batch regression suite: 429 assertions pass, including 3D gather/scatter, aggregate memory caps, and extreme-range stage arithmetic.
- Optional iterative-solver/FFT workspace suite: 42 assertions pass; its hardware-only CUDA test skips explicitly.
- JLArray timestep parity, CFL and diffusive CFL, product ownership, and lazy transform-budget regressions pass.
- Two-rank MPI CFL and 3D mixed/pure-Fourier padded-product regressions pass.
- Test inventory, layout/architecture/layering, module structure, and ownership checks pass.
- The benchmark harness completes on CPU for 12×10 and 12×10×8 grids. These runs validate the harness only.
- Real CUDA tests remain pending. The strict runner fails without a functional device; use the commands in `scripts/benchmark_single_gpu.md` on the H200.

Component review findings were resolved: mixed-precision/cross-device operand conversion, cuFFT preservation-buffer sharing, real boundary-plane projection, weighted RK multiplication order, and batched FFT input promotion. Final integrated review found no remaining issues. CUDA runtime and H200 measurements remain explicitly pending in the draft PR.
