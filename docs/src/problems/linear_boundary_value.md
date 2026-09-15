# Linear Boundary Value Problems

Use `LinearBoundaryValueProblem` for a steady linear system with prescribed
boundary conditions. `BoundaryValueSolver` assembles and solves the linear
system; no initial state or timestepper is required.

## Formulation

For ``Lu=f`` with boundary constraints ``Bu=g``, list the solution fields and
any tau unknowns in the problem constructor. Put the linear operator and tau
terms on the left, and prescribed forcing on the right. Add wall constraints
with `add_bc!`.

A scalar second-order equation on an interval needs two independent boundary
constraints and two tau DOFs in the formulation below. The two taus jointly
enforce both walls; neither is assigned to a particular wall.

## Example: Poisson equation

This serial CPU example solves ``u''=-2`` on ``[0,1]``, with ``u(0)=u(1)=0``.
The exact solution is ``u=z(1-z)``. With no tangential coordinate, the taus have
no bases; for a Fourier–Chebyshev domain, boundary taus retain the Fourier bases.

```julia
using Tarang

coords = CartesianCoordinates("z")
dist = Distributor(coords; dtype=Float64, device=CPU())
zb = ChebyshevT(coords["z"]; size=24, bounds=(0.0, 1.0))
domain = Domain(dist, (zb,))
u = ScalarField(domain, "u")
tau1 = ScalarField(dist, "tau1", (), Float64)
tau2 = ScalarField(dist, "tau2", (), Float64)
lift_basis = derivative_basis(zb, 2)

problem = LinearBoundaryValueProblem([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lift_basis, -1),
                         l2=lift(tau2, lift_basis, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
solver = BoundaryValueSolver(problem)
solve!(solver)

z, = local_grids(dist, zb)
ensure_layout!(u, :g)
@assert maximum(abs.(get_grid_data(u) .- z .* (1 .- z))) < 1e-12
@assert maximum(abs, get_grid_data(u)[[1, end]]) < 1e-12
```

## Results and constraints

`solve!` returns the solver and updates the original fields. Use
`ensure_layout!(u, :g)` before reading grid values. Check both the equation
residual and boundary values; a square matrix alone does not ensure a unique
solution. Problems with a nullspace may also need a gauge condition.

Spatial boundary data is supported. Steady solvers prepare boundary expressions
at `t=0`; a time-evolving boundary belongs in an
[initial value problem](initial_value.md).

## Further reading

- [Tau method](../pages/tau_method.md): residual corrections, lift representation, and gauges.
- [Boundary conditions](../tutorials/boundary_conditions.md): Dirichlet, Neumann, Robin, and vector constraints.
- [Nonlinear BVPs](nonlinear_boundary_value.md): when the equation depends nonlinearly on the unknown.
- [Solver reference](../pages/solvers.md#BoundaryValueSolver) and [GPU computing](../pages/gpu_computing.md): solver options and backend support.
