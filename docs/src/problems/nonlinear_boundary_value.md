# Nonlinear Boundary Value Problems

Use `NonlinearBoundaryValueProblem` for a steady nonlinear equation. The same
`BoundaryValueSolver` used for linear BVPs performs Newton iteration, starting
from the values stored in the solution fields.

## Formulation and initial guess

Write the supported linear part on the left and nonlinear terms on the right.
For ``Lu=F(u)``, the residual is ``R(u)=Lu-F(u)``. Newton iteration linearizes
this residual about the current state, solves for a correction, and updates the
fields. Tarang forms the symbolic Fréchet derivative and rebuilds the Jacobian
in its subproblem solve path.

Boundary constraints and tau unknowns follow the
[linear BVP formulation](linear_boundary_value.md). An initial guess is a
starting point for iteration, not a physical initial condition. Convergence can
depend on that guess and does not establish that the solution is unique.

## Example: a manufactured nonlinear equation

Choose ``g(z)=-2-[z(1-z)]^2`` so that ``u''=u^2+g`` has the known solution
``u=z(1-z)`` with homogeneous Dirichlet walls. This serial CPU example starts
Newton iteration from zero and checks the result.

```julia
using Tarang

coords = CartesianCoordinates("z")
dist = Distributor(coords; dtype=Float64, device=CPU())
zb = ChebyshevT(coords["z"]; size=24, bounds=(0.0, 1.0))
domain = Domain(dist, (zb,))
u = ScalarField(domain, "u")
g = ScalarField(domain, "g")
tau1 = ScalarField(dist, "tau1", (), Float64)
tau2 = ScalarField(dist, "tau2", (), Float64)
lift_basis = derivative_basis(zb, 2)

z, = local_grids(dist, zb)
expected = z .* (1 .- z)
ensure_layout!(g, :g)
get_grid_data(g) .= -2 .- expected.^2
ensure_layout!(u, :g)
get_grid_data(u) .= 0.0

problem = NonlinearBoundaryValueProblem([u, tau1, tau2])
add_parameters!(problem; g, l1=lift(tau1, lift_basis, -1),
                           l2=lift(tau2, lift_basis, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = u*u + g")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
solver = BoundaryValueSolver(problem; tolerance=1e-10, max_iterations=30)
solve!(solver)

ensure_layout!(u, :g)
@assert maximum(abs.(get_grid_data(u) .- expected)) < 1e-8
@assert maximum(abs, get_grid_data(u)[[1, end]]) < 1e-10
```

## Convergence and support

`tolerance` and `max_iterations` control Newton iteration. `solve!` updates the
fields and returns the solver, not a convergence flag. The subproblem Newton
path warns if it reaches its iteration limit without meeting tolerance; the
global fallback raises an error. Inspect convergence messages and validate the
residual and boundary values before using the result.

GPU nonlinear boundary-value solves are currently rejected. A working linear
GPU BVP does not imply nonlinear GPU support. See
[GPU computing](../pages/gpu_computing.md) for execution limits.

## Further reading

- [Tau method](../pages/tau_method.md) and [boundary conditions](../tutorials/boundary_conditions.md).
- [Problem API](../api/problems.md) and [solver reference](../pages/solvers.md#BoundaryValueSolver).
- [Eigenvalue problems](eigenvalue.md): analyze a linearized operator about a chosen state.
