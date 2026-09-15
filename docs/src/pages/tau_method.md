# The Tau Method for Boundary Conditions

Tarang enforces boundary conditions using the **tau method**, a spectral technique that lets you solve PDEs with non-periodic boundary conditions without modifying the spectral basis. This page explains what the tau method is, how Tarang's implementation works, and how to write code that uses it correctly.

The formulation uses explicit tau variables: tau fields are added to the state vector, and `lift()` operators inject them into the equations as extra degrees of freedom to match boundary conditions.

## Why Do We Need the Tau Method?

Consider the steady diffusion problem:

```math
\frac{d^2 u}{dz^2} = f(z), \quad z \in [-1, 1], \quad u(-1) = u(+1) = 0.
```

In a Chebyshev spectral method, `u` is expanded as

```math
u(z) = \sum_{n=0}^{N-1} a_n\, T_n(z),
```

The second derivative has a two-dimensional nullspace (constant and linear
polynomials), so the interior equations alone do not determine all coefficients.
They also cannot represent an arbitrary degree-``N-1`` forcing: ``u_N''`` has
degree at most ``N-3``. A tau formulation adds two unknown residual amplitudes:

```math
\mathcal{L}[u_N] + \tau_1\phi_1(z) + \tau_2\phi_2(z) = f_N(z),
\qquad B u_N = g.
```

Here ``f_N`` is the discrete forcing, ``B`` evaluates the boundary constraints,
and the chosen polynomials ``\phi_k`` supply the extra columns. In coefficient
form the augmented system is

```math
\begin{pmatrix} L & P \\ B & 0 \end{pmatrix}
\begin{pmatrix} a \\ \tau \end{pmatrix}
= \begin{pmatrix} f \\ g \end{pmatrix}.
```

There are ``N+2`` equations and unknowns. The boundary rows constrain ``u_N``;
the solve determines its coefficients and both taus together. A square system
still requires independent constraints and suitable tau columns to be invertible.

**Tau terms modify the interior PDE residual:**
``\mathcal{L}[u_N]-f_N=-\sum_k\tau_k\phi_k``. High-degree polynomials extend
throughout the domain; they are not corrections localized at the walls. For a
convergent, well-resolved discretization the residual should decrease under
refinement, but choosing a high mode alone does not guarantee a small error.

When ``P`` selects particular coefficient rows, eliminating the tau unknowns
is equivalent to omitting those PDE rows and imposing the BCs instead. Explicit
tau variables and row replacement are two algebraic representations of that
formulation. See the [generalized tau-method reference](https://dedalus-project.readthedocs.io/en/latest/pages/tau_method.html).

## Quick-Start Example

This 2D Poisson example solves `Δu = -2`
with homogeneous Dirichlet BCs on the two `z` walls, whose exact solution is
`u = z(Lz - z)`. The domain has one separable (Fourier) `x` direction and one
bounded (Chebyshev) `z` direction; the Chebyshev direction is the coupled one
that needs tau corrections:

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))    # periodic (separable)
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))   # bounded  (coupled)
dom = Domain(dist, (xb, zb))

# State field
u = ScalarField(dom, "u")

# Tau fields — one per boundary condition. Each lives on the COMPLEMENT of the
# state's bases: the coupled (Chebyshev) z axis is dropped, leaving (xb,).
tau1 = ScalarField(dist, "tau1", (xb,), Float64)
tau2 = ScalarField(dist, "tau2", (xb,), Float64)

# The lift basis. This equation is written as a DIRECT second-order operator
# (`Δ(u)`), so the second-derivative basis is the matching one. The first-order
# `grad_u`/`grad_T` formulation further down uses `derivative_basis(zb, 1)`.
lb2 = derivative_basis(zb, 2)

problem = LinearBoundaryValueProblem([u, tau1, tau2])

# Register the lifts as named parameters, then reference them in the equation.
# lift(tau, lb2, -k) places tau at the k-th-from-last Chebyshev coefficient
# slot (-1 = last, -2 = second-to-last).
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")

add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(u, :g)
x, z = local_grids(dist, xb, zb)
@assert maximum(abs.(get_grid_data(u) .- z' .* (1 .- z'))) < 1e-12
@assert maximum(abs, get_grid_data(u)[:, [1, end]]) < 1e-12
```

The assertions check the analytic solution and both wall values. Roundoff-level
residuals depend on the numerical environment.

Notice that:

1. Tau fields (`tau1`, `tau2`) are **added to the state vector**, not computed from the state afterwards.
2. The lift terms are registered as named parameters via `add_parameters!` and then referenced by name (`l1`, `l2`) in the equation. Each lift uses the 3-arg form `lift(tau, lift_basis, -k)`.
3. The BCs are declared with `add_bc!` (`u(z=0) = 0`) — Tarang converts each into an algebraic row that constrains the boundary value of `u`.

The solver builds **one square tau subproblem per separable Fourier mode** and
determines `u`, `tau1`, and `tau2` **simultaneously** in a single linear solve.

!!! note "1D pure-Chebyshev BVP"
    A pure single-axis Chebyshev BVP (no Fourier direction) works the same way —
    drop the `x` axis and define the `tau` variables on `()`. The solver builds a
    single coupled tau subproblem over the Chebyshev spectrum.

## The `lift` Operator

`lift` is how a tau field enters an equation. Always use the explicit 3-argument
form, with the lift basis built from `derivative_basis`, choosing the derivative
order that matches the space the lifted term is added to:

```julia
lift(tau, derivative_basis(basis, order), -k)   # explicit lift basis
```

- **`tau`** is the tau field (a `ScalarField` or `VectorField` whose bases are a strict subset of the state field's bases — it's missing the coupled direction).
- **`derivative_basis(basis, order)`** is the *lift basis*. A direct second-order
  equation (`Δ(u)`) uses `order = 2`; the first-order `grad_u`/`grad_T`
  formulation used by the flow examples uses `order = 1`. Bind the result once
  and reuse it. See [Why still pass the derivative basis?](#Why-still-pass-the-derivative-basis?)
  below for what this argument does — and does not — control today.
- **`-k`** is an integer mode index with wraparound semantics: `-1` is the last coefficient slot of the coupled direction (`lift_mode = N-1` → Julia index `N`), `-2` is the second-to-last, `0` is the first, and so on. For a ChebyshevT state of size `N`, `-1` places the tau value at coefficient `N-1`. A second-order problem uses two taus with lift orders `-1` and `-2`.

### The 2-argument short form does not work

Use `lift(tau, basis, n)` explicitly. Automatic basis detection in
`lift(tau, n)` is unreliable with the distributor's cached layout keys and can
raise an error even after the full state field has been constructed. The parser
propagates this failure for equation strings; it does not replace a failed lift
with its bare operand.

Build the lift with an explicit basis, register it with `add_parameters!`, and
reference the parameter in the equation, as in the quick-start example.

### What `lift(tau, basis, n)` actually computes (solver view)

For a scalar tau, `subproblem_matrix(op::Lift, sp)` builds an ``N\times1``
sparse column with a unit entry at the selected coefficient row. Vector taus
produce one such column per component. A direct lift adds to one row per
component; composing it with a derivative, as in `div(grad_u)`, also differentiates
the lift column and can couple it to additional rows.

```math
\operatorname{Lift}(\tau,\cdot,n) \longrightarrow e_j\tau,
\qquad j=N+n+1\quad(n<0)
```

Here ``j`` is a Julia index: `-1` gives row `N`. This is a coefficient-space
injection, not a physical-space boundary impulse.

!!! note "Current matrix representation"
    `src/core/operators/matrices/matrices_subproblem_operators.jl` takes `N`
    from the subproblem's coupled basis and does not use `op.basis` to convert
    the lift column. At a fixed mode, passing the state basis or its first or
    second derivative basis therefore gives the same subproblem lift matrix.
    This implementation detail is not a mathematical equivalence between those
    polynomial bases. The standalone `evaluate_lift` path does use `op.basis`
    to construct its output field, so the basis argument is not universally ignored.

### Why still pass the derivative basis?

The examples use `derivative_basis(zb, 1)` for gradient substitutions and
`derivative_basis(zb, 2)` for direct second-order equations. This states the
intended output space and keeps basis-dependent operations explicit.
Mathematically, a Chebyshev-T derivative can be represented in Chebyshev-U;
changing the tau polynomial family generally changes the discretization.
Tarang's current subproblem columns do not implement that basis conversion,
so the syntax alone does not establish an ultraspherical representation or
its conditioning properties.

## Boundary Conditions as Algebraic Constraint Rows

When you write

```julia
add_bc!(problem, "u(z=-1) = 0")
```

Tarang produces a new equation row that looks like

```math
\sum_{n=0}^{N-1} a_n\, T_n(-1) \;=\; 0,
```

i.e. a linear combination of `u`'s coefficients that equals the boundary value. **This row is added to the system, not substituted in.** It's an *algebraic* equation — there is no time derivative, so it contributes nothing to the `M` matrix in the `M·dX/dt + L·X = F` formulation. In the linear-algebra picture:

- The PDE contributes `Nz` rows; the boundary conditions add constraint rows
- `tau_u1`, `tau_u2` contribute columns (one each, at the lift modes)
- BC rows are zero in `M` (no time derivative) → they're pure algebraic constraints
- The combined LHS matrix is square because `(PDE rows + BC rows) = (state cols + tau cols)`

Match scalar tau DOFs to the added scalar constraints, including vector components
and gauge conditions. This balances the dimensions; it does not guarantee full
rank. Missing BCs or unused tau columns can leave a non-square or singular system.

### DAE-style handling in InitialValueProblem steppers

Boundary constraints have zero rows in the mass matrix ``M``, making the
semidiscrete system a differential-algebraic equation. For an implicit RK stage
with nonzero diagonal coefficient ``a_{ii}``, the boundary part of
``(M+\Delta t\,a_{ii}L)X_i=R_i`` must use
``R_{i,BC}=\Delta t\,a_{ii}F_{BC}(t_i)`` to impose ``B X_i=g(t_i)``.
Tarang applies this through `apply_bc_override!`; multistep schemes use their
corresponding solve coefficient. Final-state constraint handling depends on the
scheme, including projection for weighted RK updates.

`sp.bc_rows` is a structural classification, not a scan for all zero mass rows.
For `Nz > 1`, `_is_bulk_eqn_size` classifies positive multiples of `Nz` as bulk;
other equation blocks are constraints. A full-size algebraic equation such as
continuity remains in the bulk. This is not a general detector for arbitrary
DAE constraints, and the size rule alone does not prove correct treatment of
inhomogeneous bulk algebraic equations.

## First-Order Formulation (Recommended)

The channel-flow examples use **first-order-style tau substitutions**: define an
augmented gradient expression and apply divergence to it. This does not introduce
a separately evolved gradient field. The derivative basis names the intended
space, subject to the matrix-representation limitation above:

```julia
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)

grad_u = grad(u) + ez * τ_lift(tau_u1)
```

`grad_u` includes the first tau correction; `div(grad_u)` applies its derivative
along with the Laplacian of `u`. Register the expressions so equations can name them:

```julia
add_parameters!(problem, nu=nu, grad_u=grad_u, τ_lift=τ_lift)
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) + τ_lift(tau_u2) = -u⋅∇(u)")
```

How the two formulations compare:

| Aspect | 2nd-order (`Δ(u)`) | First-order (`div(grad_u)`) |
|---|---|---|
| Tau DOFs for two scalar wall constraints | 2 lift terms in the second-order equation (modes `-1`, `-2`) | 2 tau fields: one in the gradient substitution and one in the evolution equation |
| Operator representation | Second derivative assembled directly | Composition of first-order gradient and divergence operators |
| Boundary enforcement | Two high-mode lift columns in the bulk equation | Split between the gradient substitution and the evolution equation |
| Recommended use | Compact scalar BVPs and small prototypes | Coupled IVPs and production flow problems |

The convention we use throughout Tarang's examples is:

- `tau_u1` — correction added inside `grad_u`, one-dimensional (xbasis only)
- `tau_u2` — correction added to the evolution equation directly, one-dimensional (xbasis only)

For a vector field `u`, both taus are VectorFields (one component per velocity
component). Their placement in different expressions does not assign them to
individual walls: both are determined jointly by the coupled system and its BCs.

## Worked Example: 2D Rayleigh–Bénard Convection

Here is the full first-order RBC setup, which is the `examples/ivp/rayleigh_benard_2d.jl` example in the repository. This is the **canonical pattern** to copy for any 2D channel-flow problem with non-periodic BCs.

```julia
using Tarang

# Domain and physical parameters
Lx, Lz = 4.0, 1.0
Nx, Nz = 256, 64
Rayleigh, Prandtl = 2e6, 1.0
nu   = Prandtl                 # viscous coefficient (diffusive-time scaling)
buoy = Rayleigh * Prandtl      # buoyancy forcing Ra·Pr

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)

domain = Domain(dist, (xbasis, zbasis))

# State variables: pressure, temperature, velocity
p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
u = VectorField(domain, "u")

# Boundary tau fields retain `(xbasis,)` and drop the coupled z direction.
# The pressure-gauge tau is spatially constant instead.
#
# tau_p:  scalar, no bases — gauge for the pressure constraint
# tau_T1: gradient-substitution correction for T
# tau_T2: evolution-equation correction for T
# tau_u1, tau_u2: ditto for each velocity component (so they are VectorFields)
tau_p  = ScalarField(dist, "tau_p",  (),         Float64)
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,),  Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,),  Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# First-order substitutions — FIRST-derivative lift basis
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

# Problem declaration includes ALL tau fields as state
problem = InitialValueProblem([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])

add_parameters!(problem,
    nu=nu, buoy=buoy, ez=ez,
    grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

# Equations
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem,
    "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

# Boundary conditions.
# NOTE the `$Lz` interpolation: a BC string is parsed by Tarang, not by Julia,
# so it cannot see the Julia binding `Lz`. Writing "T(z=Lz) = 0" drops the BC
# and the first solve fails with a DimensionMismatch. Interpolate, or write the
# number literally.
add_bc!(problem, "T(z=0) = 1")        # hot bottom wall
add_bc!(problem, "T(z=$Lz) = 0")      # cold top wall
add_bc!(problem, "u(z=0) = 0")        # no-slip
add_bc!(problem, "u(z=$Lz) = 0")      # no-slip
add_bc!(problem, "integ(p) = 0")      # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
```

**Pattern summary**:

- The five `add_bc!` calls expand to **7 scalar constraint rows**: two temperature rows, four velocity rows (two vector components at each of two walls), and one pressure-gauge row. The tau variables supply the matching seven scalar tau DOFs: one each from `tau_T1` and `tau_T2`, four from the two-component `tau_u1`/`tau_u2`, and one from `tau_p`.
- **Boundary taus retain only the tangential basis** `(xbasis,)`; the gauge tau has no bases.
- **`tau_p` is a 0-D scalar** (no bases). It contributes a one-DOF candidate column during each subproblem build, but valid-mode filtering removes it at non-DC modes, leaving the actual gauge correction only at DC.
- **The pressure gauge `integ(p) = 0`** is an algebraic constraint on the mean; it lives alongside the other BCs.

## Pressure Gauge and Valid-Mode Filtering

Pressure is defined up to a global constant. The condition `integ(p) = 0` fixes
that constant at the all-zero (DC) Fourier mode. At other Fourier modes the
integral vanishes identically and supplies no independent equation.

The spatially constant `tau_p` enters continuity directly:

```julia
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_bc!(problem, "integ(p) = 0")
```

The gauge equation fixes the mean pressure; `tau_p` supplies the extra unknown
needed to accommodate the compatibility constraint in continuity. It is not a
boundary lift. Declare it on `()`, while the wall taus retain the tangential
Fourier bases. See the [gauge-condition reference](https://dedalus-project.readthedocs.io/en/latest/pages/gauge_conditions.html).

In the raw subproblem matrices, a bases-free scalar contributes one candidate
column per Fourier mode. `build_matrices!` in
`src/core/subsystems/subproblem_matrix_build.jl` removes rows that are zero in
both `L` and `M`, pairing each with an unused one-DOF variable column selected by
the smallest combined column norm. This numerical heuristic is not an explicit
identification of a pressure-gauge variable. In the intended channel formulation,
filtering removes the redundant non-DC gauge row and its tau column; it does not
replace the need to specify independent BCs and a pressure gauge at DC.

## Number of Tau Terms

For the interval/channel formulations here, balance scalar tau DOFs with the
added scalar constraint rows (including gauge conditions). A vector-valued `add_bc!` call contributes one scalar row per component, and a vector tau field contributes the corresponding component-wise tau DOFs:

| Scalar differential order on an interval | Independent BCs | Tau DOFs |
|---|---|---|
| 1st (∂u/∂z) | 1 | 1 |
| 2nd (∂²u/∂z², via first-order form) | 2 | 2 (one in grad, one in evolution) |
| 4th (biharmonic) | 4 | 4 |

**In a 2D problem (x-Fourier, z-coupled)** each equation carries tau corrections only in the `z` direction — **not "2 per direction"**. The Fourier `x` direction is periodic and needs no tau terms. Only the coupled `z` direction contributes.

**Vector equations need vector tau fields.** In RBC, `u` is a 2-component velocity, so `tau_u1` and `tau_u2` are `VectorField`s (one scalar per component per Fourier mode).

## Time- and Space-Dependent BCs

Tarang supports boundary conditions whose value varies in time, space, or both. They are refreshed by the stepper:

- **Time-dependent BCs** (e.g. `T(z=0) = sin(t)`): the RK stepper re-evaluates the BC value at each stage time `t + c[i]*dt` to supply stage-consistent boundary data. This alone does not guarantee the
  method's formal convergence order for a constrained or stiff PDE; check timestep
  convergence for the problem being solved.
- **Space-dependent BCs** (e.g. `T(z=0) = sin(2*pi*x/4.0)`): at solver-build time the BC expression is evaluated on the **global** coordinate grid, and the resulting array is projected onto the Fourier modes via an unnormalized `FFTW.rfft`. Each subproblem picks its own mode from the cached coefficient array.
- **Space+time BCs** (e.g. `T(z=0) = sin(2*pi*x/4.0) * cos(2*pi*t)`): combined — re-projected on every stage.

You don't need to register coordinate fields manually. The solver auto-registers global grid arrays for every Fourier/Chebyshev axis under its element label (`"x"`, `"y"`, `"z"`, ...), so BC string expressions can reference those coordinates directly:

```julia
add_bc!(problem, "T(z=0) = 1 + 0.1*sin(2*pi*x/4.0)")   # Lx = 4.0, written literally
add_bc!(problem, "T(z=0) = sin(t)")                     # time-dependent
```

Both are enforced to machine precision — measured `max|T(z=0) − target|` of `2.2e-16`
(space) and `2.3e-16` (time) after 5 RK222 steps of a diffusion problem.

!!! warning "BC strings cannot see Julia variables"
    A BC string is parsed by Tarang's own expression parser, not by Julia, so it
    resolves only coordinate names (`x`, `y`, `z`), the time variable `t`, and names
    registered with `add_parameters!`. A bare Julia binding — `"T(z=0) = 1 + 0.1*sin(2*pi*x/Lx)"`
    with `Lx = 4.0` defined in your script — logs `Warning: Unknown variable: Lx` and is
    then **enforced as zero**, silently satisfying the wrong condition (measured error
    against the intended profile: `1.1`). Interpolate the value (`"…/$Lx)"`), write the
    literal, or register it with `add_parameters!`. The same applies to the *location*:
    use `"T(z=$Lz) = 0"`, or register `Lz` before using `"T(z=Lz) = 0"`.

`BoundaryValueSolver` prepares spatial BC expressions for both linear and
nonlinear boundary-value problems at construction, using `t = 0`. Initial-value
solvers additionally refresh moving boundary values during time stepping.
References to the boundary's normal coordinate use the wall position; tangential
coordinates use the global grid.

Under MPI, every rank evaluates the BC expression on the full (global) grid and computes a local FFT — no inter-rank communication is needed because all ranks produce identical coefficient arrays.

## Inspecting BC Satisfaction

After a solve, you can verify that the BCs are actually being enforced:

```julia
# After solve!()
ensure_layout!(u, :g)
u_data = get_grid_data(u)

# For Chebyshev-T, the Gauss-Lobatto grid includes the endpoints, and the grid
# runs from z_min to z_max.
# u_data[:, 1]    is the solution at z = z_min (first Cheb grid point)
# u_data[:, end]  is the solution at z = z_max (last Cheb grid point)
@assert maximum(abs.(u_data[:, 1]))   < 1e-10  "Bottom BC not satisfied"
@assert maximum(abs.(u_data[:, end])) < 1e-10  "Top BC not satisfied"
```

For a vector field, index a component: `get_grid_data(u.components[2])[:, 1]`.

Tau fields with coefficient storage can expose the corrections chosen by a
BVP solve or an IVP step. For example, the Fourier-dependent taus in the RK222
example below are scattered back to their fields. This is not a guarantee that
every timestepper exposes every internal stage tau: bases-free scalars can use
solver-owned storage, and final updates differ between schemes.

Continuing from the quick-start setup, this forced diffusion step demonstrates
nonzero IVP tau values while retaining homogeneous wall conditions:

```julia
forcing = ScalarField(dom, "forcing")
ensure_layout!(forcing, :g)
get_grid_data(forcing) .= sin.(7 .* z')
ensure_layout!(u, :g)
get_grid_data(u) .= 0

ivp = InitialValueProblem([u, tau1, tau2])
add_parameters!(ivp; forcing,
                l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(ivp, "∂t(u) - Δ(u) + l1 + l2 = forcing")
add_bc!(ivp, "u(z=0) = 0")
add_bc!(ivp, "u(z=1) = 0")
ivp_solver = InitialValueSolver(ivp, RK222(); dt=0.01)
step!(ivp_solver)

@info "Tau magnitudes" tau1=maximum(abs, get_coeff_data(tau1)) tau2=maximum(abs, get_coeff_data(tau2))
ensure_layout!(u, :g)
@assert maximum(abs, get_grid_data(u)[:, [1, end]]) < 1e-12
```

Tau amplitudes depend on polynomial normalization, equation scaling, resolution,
and (for IVPs) the time discretization. They are not standalone error estimates.
Check the physical PDE residual, boundary residuals, and convergence under
refinement. For vector taus, inspect the appropriate field component.

For time-dependent problems, check BC satisfaction inside a callback:

```julia
run!(solver;
     stop_iteration=30,
     callbacks=[on_interval(10) do s
         ensure_layout!(T, :g)
         T_data = get_grid_data(T)
         @info "T(z=0) residual: $(maximum(abs.(T_data[:, 1] .- 1.0)))"
     end])
```

## Common Pitfalls

### 1. Wrong number of tau fields

For a scalar second-order equation, two wall constraints and only one tau
produce `N+2` rows but `N+1` columns: a non-square, overdetermined system. Extra
unconstrained taus instead add columns and can make the system underdetermined.

**Fix**: count the scalar constraint rows — including gauge conditions like `integ(p) = 0`, and counting one row *per component* for a vector BC — and declare one tau DOF per row.

### 2. Missing `lift()` term in the equation

A tau field declared in `problem.variables` but never referenced in any equation contributes a zero column to the LHS matrix, making the system rank-deficient.

**Fix**: each tau must contribute to the intended implicit system. Boundary taus
usually enter through lifts; the pressure-gauge tau enters continuity directly.
An augmented-gradient substitution may carry the same tau into multiple
equations. Count independent columns, not textual occurrences of `lift()`.

### 3. Wrong lift basis — and what it does *not* cause

Use the explicit basis that describes the intended lift space. The current
subproblem matrix path produces the same column for several basis choices,
but standalone lift evaluation uses the selected basis. Do not generalize
matrix-column equality into a claim that polynomial family never affects
conditioning or accuracy. See the solver-view discussion above.

### 4. Tau field bases don't match

A tau field must live on the *complement* of the state field's bases — the state's bases **minus the coupled direction**. For a 2D state on `(xbasis, zbasis)`, the tau lives on `(xbasis,)`; for a 0-D gauge like `tau_p` it's `()`.

If you accidentally declare `tau_u1 = ScalarField(dist, "tau_u1", (xbasis, zbasis))` (same bases as `u`), the `lift` call itself still *constructs* — the mistake is not caught there. It surfaces at solver build as `Warning: Matrix is not square: rows=70, cols=180` (the tau contributes a whole Chebyshev spectrum of columns instead of one), and the solve then throws a `DimensionMismatch`.

**Fix**: drop the Chebyshev axis when declaring tau fields.

### 5. Non-square system at DC mode

Most non-square warnings at the DC Fourier mode come from a missing gauge BC like `integ(p) = 0`. The valid-mode filter pairs zero rows with eligible one-DOF columns; it cannot
replace a missing independent constraint.

**Fix**: for the incompressible formulation shown here, include `tau_p` in
continuity and `integ(p) = 0`. Other gauge freedoms require constraints appropriate
to their nullspaces.

### 6. BC F value not reaching the stepper

For inhomogeneous BCs like `T(z=0) = 1`, the stepper has to carry the value `1` through the stage RHS assembly. This happens automatically through `gather_alg_F!` + the `apply_bc_override!` path in `step_subproblem_rk!`, but only if the BC equation has a non-zero `F` expression in `equation_data[eq_idx]["F"]`.

**Symptom of the bug**: `max|T|` decays to zero over time even though `T(z=0) = 1` is declared, OR `max|T|` sticks at `1/γ = 2+√2 ≈ 3.414` (the classical 1/γ scaling factor from RK222's implicit coefficient).

**If you see this**: confirm you're on a recent version — this was a regression fixed after the subproblem-architecture rewrite. The `apply_bc_override!` override is what enforces `L_row·X = F_BC` at every stage regardless of accumulation history.

## Historical Note

The tau method was introduced by **Cornelius Lanczos** in 1938 as an approximation technique: rather than solving a PDE exactly, he sought polynomial approximations that satisfied the PDE with a small residual (the "tau error"). It was refined for spectral methods by **Steven Orszag**, **David Gottlieb**, and others in the 1970s–80s, and adapted to modern lift-based formulations by **Keaton Burns et al.** in the 2010s.

The name "tau" (τ) comes from Lanczos's notation for the residual/correction terms introduced when truncating the polynomial expansion and enforcing boundary conditions.

## References

### Textbooks

1. **Canuto, C., Hussaini, M. Y., Quarteroni, A., & Zang, T. A.** (2006). *Spectral Methods: Fundamentals in Single Domains*. Springer. — Rigorous treatment of tau and Galerkin methods.

2. **Boyd, J. P.** (2001). *Chebyshev and Fourier Spectral Methods* (2nd ed.). Dover. — Very readable; freely available online.

3. **Trefethen, L. N.** (2000). *Spectral Methods in MATLAB*. SIAM. — Practical, code-oriented.

4. **Peyret, R.** (2002). *Spectral Methods for Incompressible Viscous Flow*. Springer. — Detailed treatment of Navier–Stokes with spectral methods.

### Key papers

5. **Lanczos, C.** (1938). "Trigonometric interpolation of empirical and analytical functions." *Journal of Mathematics and Physics*, 17(1–4), 123–199. — Original tau method paper.

6. **Burns, K. J., Vasil, G. M., Oishi, J. S., Lecoanet, D., & Brown, B. P.** (2020). *Physical Review Research*, 2, 023068. — Modern lift-based tau methods.

7. **Orszag, S. A.** (1971). "Accurate solution of the Orr-Sommerfeld stability equation." *Journal of Fluid Mechanics*, 50(4), 689–703. — Classic application to hydrodynamic stability.

## See Also

- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): step-by-step BC examples
- [Bases](bases.md): spectral bases (Chebyshev, Fourier, Legendre, Jacobi)
- [Solvers](solvers.md): using InitialValueProblem / LinearBoundaryValueProblem / NonlinearBoundaryValueProblem solvers
- [API: Problems](../api/problems.md): programmatic API for adding equations and BCs
- [2D RBC Tutorial](../tutorials/ivp_2d_rbc.md): complete Rayleigh–Bénard convection walkthrough
