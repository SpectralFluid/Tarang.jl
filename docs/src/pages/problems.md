# Problems

Problems define the PDE system to be solved, including equations and boundary conditions.

## Problem Types

Choose a guide by the result you need. Each includes a complete example,
validation checks, and links to the relevant solver and boundary-condition APIs.

| Result | Problem | Guide |
|---|---|---|
| Evolve a known initial state | `InitialValueProblem` | [Initial value problems](../problems/initial_value.md) |
| Solve a steady linear equation | `LinearBoundaryValueProblem` | [Linear boundary value problems](../problems/linear_boundary_value.md) |
| Find a nonlinear steady state from a guess | `NonlinearBoundaryValueProblem` | [Nonlinear boundary value problems](../problems/nonlinear_boundary_value.md) |
| Compute linear growth rates and modes | `EigenvalueProblem` | [Eigenvalue problems](../problems/eigenvalue.md) |

The sections below describe shared equation syntax and constraints. Backend
support depends on the problem and solver; see each guide before selecting a GPU
or MPI configuration.

### InitialValueProblem - Initial Value Problem

Use `InitialValueSolver` with a timestepper. The [IVP guide](../problems/initial_value.md)
covers initialization, time evolution, and output.

### LinearBoundaryValueProblem - Linear Boundary Value Problem

Use `BoundaryValueSolver` for a linear solve. The [linear BVP guide](../problems/linear_boundary_value.md)
shows a Poisson problem with explicit tau unknowns and boundary checks.

### NonlinearBoundaryValueProblem - Nonlinear Boundary Value Problem

Use `BoundaryValueSolver` with an initial guess and Newton controls. The
[nonlinear BVP guide](../problems/nonlinear_boundary_value.md) explains residuals
and checks a manufactured solution.

### EigenvalueProblem - Eigenvalue Problem

Use `EigenvalueSolver` for a linearized system. The [EVP guide](../problems/eigenvalue.md)
explains the growth-rate convention, algebraic constraints, and returned modes.

## Adding Equations

### Equation Syntax

An equation is a `"LHS = RHS"` string. For IMEX initial-value problems, put
implicitly treated linear terms on the LHS and explicit terms on the RHS. In a
nonlinear BVP the RHS is linearized during Newton iteration; it is not time-stepped.
See the problem-specific guides above for the appropriate formulation.

```julia
# scalar advection-diffusion: diffusion implicit, advection explicit
add_equation!(problem, "∂t(s) - nu*Δ(s) = -u⋅∇(s)")

# any number of linear terms may share the LHS
add_equation!(problem, "∂t(s) - nu*Δ(s) + ∂x(s) = -u⋅∇(s)")
```

### Equation Sizing

The solver automatically determines each equation's row count in the system matrix from the expression's output type (scalar, vector, tensor, etc.). Equations can be added in any order — no specific ordering is required.

```julia
# q, ψ: ScalarFields;  u: VectorField;  tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)
problem = InitialValueProblem([q, ψ, u, tau_ψ])
add_parameters!(problem, nu=1e-6)

# Any order is fine:
add_equation!(problem, "Δ(ψ) + tau_ψ - q = 0")           # scalar → D rows
add_equation!(problem, "u - skew(grad(ψ)) = 0")           # vector → 2D rows
add_equation!(problem, "∂t(q) + nu*Δ⁴(q) = -u⋅∇(q)")     # scalar → D rows
add_bc!(problem, "integ(ψ) = 0")                           # constraint → 1 row
```

### Supported Operations

- Derivatives: `∂x`, `∂y`, `∂z`, `∂t` (or `dt`), `Δ` (or `lap`), `∇`, `div`, `curl`,
  `Δ⁴` (hyperdiffusion), and the advection shorthand `u⋅∇(f)`
- Tensor/algebraic: `trace`, `skew`, `grad`, `lift`, `integ`
- Arithmetic: `+`, `-`, `*`, `/`, `^`
- Functions: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `tanh`
- Parameters: any name registered with `add_parameters!` (i.e. present in
  `problem.namespace`) — a bare Julia global is *not* visible to an equation string

### Parameters

Equation strings resolve names against `problem.namespace`. They cannot see plain Julia
globals, so every constant, field or operator an equation refers to must be registered
first — with `add_parameters!` (preferred) or by writing into the namespace directly.

```julia
add_parameters!(problem; nu=0.01, Ra=1e6, Pr=1.0)   # preferred
problem.namespace["nu"] = 0.01                       # equivalent, direct dict access

add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")
```

Keep parameter-times-field terms on the LHS: a linear term on the RHS (`= Ra*Pr*w`)
breaks the IMEX splitting, and `add_equation!` warns about it.

## First-Order Formulation

For problems involving Chebyshev bases, Tarang supports a first-order reduction (tau method) that replaces `Δ(f)` with `div(grad_f)` where `grad_f` includes a tau-lifting term. This ensures correct boundary condition enforcement.

```julia
# Derivative basis and lift closure
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)

# First-order gradient substitutions
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_b = grad(b) + ez * τ_lift(tau_b1)

# The substitutions are Julia objects, so they must be registered before the equation
# strings can name them
add_parameters!(problem, kappa=0.1, grad_u=grad_u, grad_b=grad_b, τ_lift=τ_lift)

# Equations use div(grad_f) instead of Δ(f)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")
```

Balance scalar tau DOFs with independent scalar constraints, counting vector
components and gauge conditions. The taus enforce the constraints jointly; their
names do not assign them to individual walls. The Rayleigh-Bénard pattern below
shows this formulation.

The advection operator `u⋅∇(f)` is automatically expanded component-wise to `Σᵢ uᵢ ∂ᵢf`, so it works for both scalar and vector fields without manual expansion.

## Boundary Conditions

Boundary conditions are declared with **`add_bc!`**, not `add_equation!`. The syntax
`field(coord=value)` is auto-detected and converted to the appropriate condition.

Two rules govern every BC:

1. **Each boundary condition needs its own `tau` variable.** A BC adds one row to the
   system, and the solver requires as many equations (PDEs + BCs) as variables. A wall
   problem with two conditions per field carries two tau variables per field, lifted into
   the bulk equation (see the [tau method](tau_method.md)); without them, solver
   construction throws `Problem validation failed: Number of equations ... does not match
   number of variables`.
2. **The value must be a literal or a registered parameter.** BC strings resolve names
   against `problem.namespace`, so a plain Julia global warns `Unknown variable` and is
   silently enforced as `0`. Use a literal, or register the name with `add_parameters!`.

Declaring a BC with `add_equation!` also registers it as an equation row, but it never
reaches the BC manager, so a space- or time-dependent value is never refreshed and is
enforced as zero. Always use `add_bc!`.

### Dirichlet (Value)

```julia
# field = value at location
add_bc!(problem, "T(z=0) = 1")   # T = 1 at z = 0
add_bc!(problem, "T(z=1) = 0")   # T = 0 at z = 1
```

### Neumann (Derivative)

```julia
# ∂field/∂z = value at location
add_bc!(problem, "∂z(T)(z=0) = 1")   # ∂T/∂z = 1 at z = 0
```

### Robin (Mixed)

```julia
# α*T + β*∂T/∂n = γ
add_bc!(problem, "1.0*T(z=0) + 1.0*∂z(T)(z=0) = 0")

# The same condition as a structured object: robin_bc(field, coord, position, α, β, γ)
add_bc!(problem, robin_bc("T", "z", 0.0, 1.0, 1.0, 0.0))
```

### No-Slip and Stress-Free Walls

```julia
add_bc!(problem, "u(z=0) = 0")       # no-slip:     u = 0 at a solid wall
add_bc!(problem, "∂z(u)(z=1) = 0")   # stress-free: ∂u/∂z = 0 at a free surface
```

### Named Helpers

The common physical conditions have wrappers that build the same BCs:

```julia
no_slip!(problem, "u", "z", 0.0)            # u = 0        at z = 0
free_slip!(problem, "u", "z", 1.0)          # ∂u/∂z = 0    at z = 1
fixed_value!(problem, "T", "z", 0.0, 1.0)   # T = 1        at z = 0
insulating!(problem, "T", "z", 1.0)         # ∂T/∂z = 0    at z = 1
```

### Gauge Constraints

A pressure-like field defined only up to a constant needs a gauge condition, which is
also declared with `add_bc!` and also consumes a tau variable:

```julia
add_bc!(problem, "integ(p) = 0")
```

## Problem Validation

`validate_problem` returns `true` for a well-posed system and otherwise throws an
`ArgumentError` listing every problem it found (missing variables or equations, a
mismatched equation count, invalid boundary conditions). It runs automatically inside
every solver constructor, so a badly-posed problem fails at solver construction — you
rarely need to call it yourself.

```julia
validate_problem(problem)   # true, or throws ArgumentError
```

The equation count check is exact for IVPs and EVPs (`#equations + #BCs == #variables`)
and a lower bound for BVPs (`#equations + #BCs >= #variables`). Boundary conditions are
merged into the equation list when the solver is built, which is why every BC must be
paired with a tau variable.

## Common Problem Patterns

All three wall-bounded patterns below share one skeleton: a periodic Fourier direction
`x`, a bounded Chebyshev direction `z`, one tau variable per boundary condition, and BCs
declared with `add_bc!`.

### Heat Equation

Two Dirichlet walls, so two tau variables lifted into the diffusion term. Lifting into a
second-order operator (`Δ`) uses `derivative_basis(zb, 2)` at lift orders `-1` and `-2`.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=8,  bounds=(0.0, 2π))
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
dom = Domain(dist, (xb, zb))

T    = ScalarField(dom, "T")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)   # one tau per BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = InitialValueProblem([T, tau1, tau2])
add_parameters!(problem; kappa=0.01, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "∂t(T) - kappa*Δ(T) + l1 + l2 = 0")
add_bc!(problem, "T(z=0) = 1")   # hot bottom
add_bc!(problem, "T(z=1) = 0")   # cold top

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
set!(T, 0.0)
run!(solver; stop_iteration=10, progress=false)
```

### Incompressible Navier-Stokes

Velocity is a single `VectorField` (not per-component scalars), and the equations use the
first-order tau form: `div(grad_u)` in place of `Δ(u)`, with `tau_u1` inside `grad_u` and
`tau_u2` lifted into the momentum equation. `tau_p` is the gauge unknown that pays for
`integ(p) = 0`.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 4.0), dealias=3/2)
zb = ChebyshevT(coords["z"];  size=12, bounds=(0.0, 1.0), dealias=3/2)
dom = Domain(dist, (xb, zb))

p = ScalarField(dom, "p")
u = VectorField(dom, "u")
tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xb,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xb,), Float64)

ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zb, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u     = grad(u) + ez * τ_lift(tau_u1)

problem = InitialValueProblem([p, u, tau_p, tau_u1, tau_u2])
add_parameters!(problem, nu=0.01, grad_u=grad_u, τ_lift=τ_lift)
add_equation!(problem, "trace(grad_u) + tau_p = 0")                                 # continuity
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) + τ_lift(tau_u2) = -u⋅∇(u)")  # momentum
add_bc!(problem, "u(z=0) = 0")     # no-slip walls
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")   # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-4)
```

### Rayleigh-Bénard

Navier-Stokes plus a buoyant temperature field: two more tau variables (`tau_T1`,
`tau_T2`) for the two temperature walls.

```julia
Rayleigh, Prandtl = 2e4, 1.0

# coords / dist / xb / zb / dom / p / u / tau_p / tau_u1 / tau_u2 / ez / τ_lift / grad_u
# exactly as in the Navier-Stokes pattern above
T      = ScalarField(dom, "T")
tau_T1 = ScalarField(dist, "tau_T1", (xb,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xb,), Float64)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = InitialValueProblem([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem, nu=Prandtl, buoy=Rayleigh*Prandtl, ez=ez,
                grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")
add_bc!(problem, "T(z=0) = 1")   # hot bottom
add_bc!(problem, "T(z=1) = 0")   # cold top
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-4)
```

!!! warning "Chebyshev advection is serial-only"
    The `-u⋅∇(u)` / `-u⋅∇(T)` terms differentiate along the Chebyshev axis on the explicit
    side, which a distributed run cannot do — each rank owns only part of that axis. Both
    patterns run in serial; under MPI the first step raises an error. See
    [Parallelism](parallelism.md).

### Poisson Equation (BVP)

A steady BVP needs the tau method. The 2D form below (`Δu = -2`, `u = 0` on both
`z` walls) is shown; a pure 1D Chebyshev Poisson works the same way — drop the `x`
axis and put the `tau` variables on `()`.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
dom = Domain(dist, (xb, zb))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LinearBoundaryValueProblem([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem); solve!(solver); ensure_layout!(u, :g)
```

## See Also

- [Solvers](solvers.md): Solving problems
- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): Detailed BC guide
- [API: Problems](../api/problems.md): Complete reference
