# Tutorials Overview

Start with the [problem-type guides](../pages/problems.md#Problem-Types) for a
complete IVP, linear BVP, nonlinear BVP, or EVP example. This directory groups
the longer tutorials by application and technique.

## Tutorial Path

### For Beginners

If you're new to spectral methods or Tarang.jl, start here:

1. **[First Steps](../getting_started/first_steps.md)** - Basic workflow and simple diffusion problem
2. **[2D Rayleigh-Bénard Convection](ivp_2d_rbc.md)** - Complete fluid dynamics example
3. **[Boundary Conditions](boundary_conditions.md)** - Master the boundary condition system

### Intermediate Topics

Once comfortable with basics:

4. **[Fluid Dynamics Examples](../examples/fluid_dynamics.md)** - Navier-Stokes, rotating convection, and turbulence examples
5. **[Surface Dynamics](surface_dynamics.md)** - Fractional Laplacian, surface quasi-geostrophic dynamics
6. **[3D Turbulent Flow](ivp_3d_turbulence.md)** - 3D problems with advanced parallelization
7. **[Analysis and Output](analysis_and_output.md)** - Data management and visualization
8. **[Eigenvalue Problems](eigenvalue_problems.md)** - Linear stability analysis

### Advanced Topics

For experienced users:

- **[Custom Operators](../pages/custom_operators.md)** - Define new differential operators
- **[Optimization Guide](../pages/optimization.md)** - Performance tuning

## Tutorial List

### Initial Value Problems (InitialValueProblem)

Time-evolution problems where you integrate PDEs forward in time.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [2D Rayleigh-Bénard](ivp_2d_rbc.md) | Thermal convection in 2D | Intermediate | Navier-Stokes, buoyancy, CFL |
| [Fluid Dynamics Examples](../examples/fluid_dynamics.md) | Navier-Stokes and turbulence examples | Intermediate | Shear flows, rotating convection, turbulence |
| [Surface Dynamics](surface_dynamics.md) | SQG and boundary-coupled dynamics | Advanced | Fractional Laplacian, surface dynamics, SQG inversion |
| [3D Taylor-Green Vortex](ivp_3d_turbulence.md) | 3D turbulence simulation | Advanced | 3D FFTs, energy spectra |
| [Channel Flow](../notebooks/channel_flow.md) | Turbulent channel flow | Advanced | Wall-bounded, statistics |

### Boundary Value Problems (BVP)

Steady-state problems with boundary conditions.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Poisson equation](../problems/linear_boundary_value.md) | Steady linear solve | Beginner | Tau constraints, analytic solution |
| [2D Poisson](lbvp_2d_poisson.md) | Periodic strip with mixed walls | Intermediate | Filtered forcing, solution heatmap, residual checks |

### Nonlinear Boundary Value Problems

[The nonlinear BVP guide](../problems/nonlinear_boundary_value.md) gives a
self-contained Newton solve, explains the initial guess and convergence controls,
and verifies the solution against an analytic profile.

The [Lane–Emden tutorial](nlbvp_lane_emden.md) develops a radial application,
checks the nonzero solution branch, and plots resolution refinement.

### Eigenvalue Problems (EigenvalueProblem)

Linear stability analysis and normal modes.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Stability Analysis](eigenvalue_problems.md) | Eigenvalue problem setup | Advanced | EigenvalueProblem, eigensolvers |

### Surface and Boundary Dynamics

Problems with dynamics confined to surfaces or boundaries.

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Surface Dynamics](surface_dynamics.md) | SQG, QG, boundary advection-diffusion | Advanced | Fractional Laplacian, coupled systems |

### Wave-Mean Flow Analysis

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Rotating Shallow Water](rotating_shallow_water.md) | Inertia-gravity wave filtering | Intermediate | Temporal filters, wave-mean separation |
| [GQL Approximation](../pages/gql_approximation.md) | Generalized Quasi-Linear methods | Advanced | Wavenumber cutoff, zonal jets |

### Special Topics

| Tutorial | Description | Complexity | Key Features |
|----------|-------------|------------|--------------|
| [Boundary Conditions](boundary_conditions.md) | All BC types and usage | Intermediate | Dirichlet, Neumann, Robin |
| [Analysis & Output](analysis_and_output.md) | Data management | Intermediate | NetCDF, analysis |

## Problem Types Explained

### Initial Value Problems (InitialValueProblem)

[Initial value problems](../problems/initial_value.md) evolve fields in time with
`InitialValueSolver`, a timestepper, and initial data.

### Boundary Value Problems (BVP)

[Linear BVPs](../problems/linear_boundary_value.md) solve a steady linear system.
[Nonlinear BVPs](../problems/nonlinear_boundary_value.md) use an initial guess and
Newton iteration. Both use `BoundaryValueSolver` and explicit boundary constraints.

### Eigenvalue Problems (EigenvalueProblem)

[Eigenvalue problems](../problems/eigenvalue.md) compute growth rates and modes
with `EigenvalueSolver`. The [stability tutorial](eigenvalue_problems.md)
extends the workflow to hydrodynamic examples.

## Choosing a Tutorial

### By Physics

**Fluid Dynamics**:
- Start: [2D Rayleigh-Bénard](ivp_2d_rbc.md)
- Intermediate: [Fluid Dynamics Examples](../examples/fluid_dynamics.md), [Surface Dynamics](surface_dynamics.md)
- Advanced: [3D Turbulence](ivp_3d_turbulence.md), [Channel Flow](../notebooks/channel_flow.md)

**Geophysical Flows**:
- Start: [Surface Dynamics](surface_dynamics.md) (SQG, QG)
- Intermediate: [Rotating Shallow Water](rotating_shallow_water.md) (wave-mean separation)
- Advanced: [GQL Approximation](../pages/gql_approximation.md) (zonal jets, turbulence closure)

**Heat Transfer**:
- Start: [First Steps](../getting_started/first_steps.md) (diffusion)
- Advanced: [2D Rayleigh-Bénard](ivp_2d_rbc.md) (convection)

**Stability Analysis**:
- Start: [Eigenvalue Problems](eigenvalue_problems.md)

### By Technique

**Want to learn**:
- **MPI parallelization** → [Running with MPI](../getting_started/running_with_mpi.md)
- **Boundary conditions** → [Boundary Conditions](boundary_conditions.md)
- **Output and analysis** → [Analysis and Output](analysis_and_output.md)
- **3D problems** → [3D Turbulence](ivp_3d_turbulence.md)

### By Complexity

**Beginner**: Basic concepts, single field
- [First Steps](../getting_started/first_steps.md)

**Intermediate**: Multiple fields, coupled equations
- [2D Rayleigh-Bénard](ivp_2d_rbc.md)
- [Fluid Dynamics Examples](../examples/fluid_dynamics.md)
- [Surface Dynamics](surface_dynamics.md)
- [Boundary Conditions](boundary_conditions.md)
- [Analysis and Output](analysis_and_output.md)

**Advanced**: 3D problems, advanced analysis
- [3D Turbulence](ivp_3d_turbulence.md)
- [Eigenvalue Problems](eigenvalue_problems.md)

## Common Patterns

### Setting Up a Simulation

Every tutorial follows this pattern. Here it is in full for 2D heat diffusion — the
smallest complete simulation Tarang can run:

```julia
# 1. MPI initialization (harmless in serial; needed for multi-rank runs)
using Tarang, MPI
MPI.Initialized() || MPI.Init()

# 2. Domain setup
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
domain = Domain(dist, (xbasis, ybasis))

# 3. Fields
u = ScalarField(domain, "u")

# 4. Problem
problem = InitialValueProblem([u])
add_parameters!(problem, nu=0.1)
add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")
# add_bc!(problem, "...")   # only for bounded (Chebyshev/Jacobi) directions

# 5. Initial condition — use `local_grids`, which returns THIS rank's slab
x, y = local_grids(dist, xbasis, ybasis)
ensure_layout!(u, :g)
get_grid_data(u) .= sin.(x) .* cos.(y')
ensure_layout!(u, :c)

# 6. Solver and time loop
solver = InitialValueSolver(problem, RK222(); dt=1e-3)
t_end = 0.5
while solver.sim_time < t_end
    step!(solver)
end
# max|u| = 0.90483742, exactly exp(-2*nu*t_end)
```

`MPI.Init()` defaults to `finalize_atexit=true`, so `MPI.Finalize()` runs on exit by itself —
you do not need a cleanup step.

!!! warning "`set!(field, ::Function)` is serial-only"
    `set!(u, (x, y) -> …)` builds the *global* meshgrid, so under MPI it throws
    `DimensionMismatch`. `local_grids(dist, bases...)` gives each rank its own slice and
    works identically at `np = 1, 2, 4`, which is why the pattern above uses it.

### Adding Analysis

Common analysis tasks, on a solver whose state has a scalar `T` and a `VectorField` `u`:

```julia
# Adaptive timestep: CFL takes the SOLVER, and velocities must be VectorFields
cfl = CFL(solver; initial_dt=1e-3, cadence=10, safety=0.4, max_dt=0.01)
add_velocity!(cfl, u)

# File output: base path FIRST, solver SECOND, then one task per quantity
handler = add_file_handler("outputs", solver; sim_dt=0.005, max_writes=10)
add_task!(handler, u; name="u")
add_task!(handler, T; name="T")

# `run!` applies the CFL controller and processes solver-registered handlers
run!(solver; stop_iteration=20, cfl=cfl)

# Custom diagnostics
function compute_diagnostics(u, T)
    ke = total_kinetic_energy(u)    # Float64: domain integral of ½|u|²
    temp_mean = integrate(T)        # Float64: quadrature-weighted integral
    return (ke=ke, temp=temp_mean)
end
```

Three things that trip people up here:

- Pass **field or operator objects** to `add_task!`, never strings. `add_task!(h, "u*u")` is
  accepted silently and then writes a scalar of zeros.
- The last component of the base path is the handler **name**, and each write set gets its own
  directory: `add_file_handler("outputs", …)` writes
  `outputs/outputs_s1/outputs_s1.nc`, and a new `_sN` set is started every `max_writes` writes
  (`outputs/outputs_s2/…`). Use `Tarang.current_file(handler)` to get the path in code. The
  variables live in NetCDF-4 **groups** (`vars`, `time`, `grids`), so a plain
  `NetCDF.ncread(file, "u")` will not find them — read them with
  `Tarang.group_ncread(file, "vars", "u")`. A `VectorField` task is stored with its component
  axis, e.g. `vars/u` has shape `(write, component, x, y)`.
- `add_file_handler` also has a `(path, dist, vars)` form. That one is **not** registered with
  the solver, so `run!` will not process it unless you also pass `outputs=[handler]`.

See [Analysis and Output](analysis_and_output.md) for the full output API.

## Getting Help

If you're stuck on a tutorial:

1. **Check the complete example** at the end of each tutorial
2. **Look at the source code** in the `examples/` directory
3. **Search the API reference** for function documentation
4. **Ask on GitHub Discussions** for community help

## Contributing Tutorials

We welcome tutorial contributions! See the [Contributing Guide](../pages/contributing.md) for details on:
- Tutorial format and style
- Adding Jupyter notebooks
- Including visualizations
- Testing tutorial code

## Next Steps

Ready to start? Pick a tutorial from the list above or continue with:
- [2D Rayleigh-Bénard Convection](ivp_2d_rbc.md) for a complete fluid dynamics example
- [Boundary Conditions](boundary_conditions.md) to master BC specification
- [Running with MPI](../getting_started/running_with_mpi.md) for parallel computing details
