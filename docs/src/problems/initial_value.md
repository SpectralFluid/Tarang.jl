# Initial Value Problems

Use `InitialValueProblem` to evolve fields from an initial state. Construct an
`InitialValueSolver` with a timestepper, then advance it with `step!` or `run!`.

## Formulation

Write the semidiscrete system as ``M\partial_t X+LX=F(X,t)``. For IMEX stepping,
put the time derivative and the linear terms to be treated implicitly on the
left; put the explicit terms on the right. Register coefficients and expression
substitutions with `add_parameters!` before naming them in equations.

A bounded direction also needs boundary conditions and appropriate tau unknowns.
Those constraints make the system a DAE; an initial condition should satisfy
them. Fully periodic diffusion, as below, needs no wall BCs or boundary taus.

## Example: periodic diffusion

This serial CPU example evolves ``u(x,0)=\sin x`` under
``\partial_t u=\nu\partial_x^2u`` and checks ``u(x,t)=e^{-\nu t}\sin x``.

```julia
using Tarang

coords = CartesianCoordinates("x")
dist = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
domain = Domain(dist, (xb,))
u = ScalarField(domain, "u")
x, = local_grids(dist, xb)
ensure_layout!(u, :g)
get_grid_data(u) .= sin.(x)

problem = InitialValueProblem([u])
add_parameters!(problem; nu=0.1)
add_equation!(problem, "dt(u) - nu*Δ(u) = 0")
solver = InitialValueSolver(problem, RK222(); dt=0.01)
run!(solver; stop_iteration=10, progress=false)

ensure_layout!(u, :g)
expected = exp(-0.1 * solver.sim_time) .* sin.(x)
@assert maximum(abs.(get_grid_data(u) .- expected)) < 1e-8
```

## Stepping and output

`step!(solver)` uses `solver.dt`; `step!(solver, dt)` also updates the stored
step size. `run!` manages stopping conditions, CFL control, callbacks, and
registered output handlers. The updated solution remains in the original fields;
`solver.sim_time` and `solver.iteration` record progress.

Choose the timestepper and backend together using the
[execution-support table](../pages/timesteppers.md#Where-each-scheme-runs).
See [Analysis & Output](../tutorials/analysis_and_output.md) for diagnostics and
[Running with MPI](../getting_started/running_with_mpi.md) for distributed setup.

## Bounded domains and tutorials

- [Tau method](../pages/tau_method.md): augmented equations and stage constraints.
- [Boundary conditions](../tutorials/boundary_conditions.md): fixed and moving walls.
- [2D Rayleigh–Bénard convection](../tutorials/ivp_2d_rbc.md): coupled velocity, pressure, and temperature.
- [3D turbulence](../tutorials/ivp_3d_turbulence.md): a periodic flow simulation.
- [Surface dynamics](../tutorials/surface_dynamics.md) and [rotating shallow water](../tutorials/rotating_shallow_water.md).

See [equation syntax](../pages/problems.md#Adding-Equations) and
[solver reference](../pages/solvers.md#InitialValueSolver) for shared APIs.
