# BVP application examples

**Goal:** Add runnable Tarang Poisson and Lane–Emden
examples, with verified CairoMakie figures in the corresponding website sections.

**Architecture:** Keep the short analytic setup guides and add application
tutorials. Use Fourier–Chebyshev for mixed-boundary 2D Poisson. Reduce the
spherically symmetric Lane–Emden equation to a regular radial Chebyshev problem.
Explain the radial reduction and cite the numerical literature directly.

**Tools:** Tarang, Julia, the optional `docs/figures` CairoMakie environment,
Documenter, and existing web-documentation checks.

- [x] Implement `examples/bvp/poisson.jl` and `lane_emden.jl` with independent
  PDE/boundary checks and a reference-radius check for Lane–Emden.
- [x] Generate solution and diagnostic plots with `docs/figures/boundary_value.jl`;
  inspect both PNGs and provide SVG/PNG/CSV downloads.
- [x] Add tutorials and navigation links under the two BVP sections.
- [x] Run the examples, documentation checks, full site build, and local asset/link checks.
- [x] Commit the solver corrections and illustrated tutorials.
- [ ] Open a follow-up PR after #137; verify the published preview.

Poisson uses a periodic strip, mixed wall data, and filtered random forcing.
Lane–Emden uses
the n=3 nonzero branch with initial guess `5(1-r²)²` and verifies the recovered
radius against 6.896848619376960375454528. Check refinement in addition to the
residual so convergence to the trivial zero solution cannot pass validation.

Running Lane–Emden exposed a Newton linearization defect: base-state coefficients
must remain distinct from correction fields. The solver now snapshots the state
and materializes compound coefficients before assembling each Jacobian. MPI NCC
inspection also gathers logical global data and uses rank-local temporary
transforms. Validation passed 504 distinct CPU checks, 18 checks across two MPI
ranks, 40 documentation checks, and 1177 local links, images, and anchors.
