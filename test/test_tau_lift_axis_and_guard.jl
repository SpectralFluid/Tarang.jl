"""
Tau lift DOF bookkeeping and the duplicate-lift guard.

Two defects, both invisible to a boundary-value check because every failure mode
here keeps the walls at machine zero:

1. `subproblem_field_size` indexed `sp.group` with the FIELD's own basis index.
   A tau field carries only the TANGENTIAL bases, so index 1 is the separable
   axis only when the tangential axis happens to be axis 1 of the domain. With
   the Chebyshev axis first -- the layout distributed Cheb x Fourier REQUIRES --
   a `(xb,)` tau was counted as the full coefficient length, and the lift's Nz x 1
   column then hit `DimensionMismatch: second dimension of A, 1, does not match
   the first dimension of B, 5` with no mention of tau or axis order.

2. `_check_duplicate_tau_lifts` compared lift bases with `===`. `derivative_basis`
   returns a freshly built struct on every call, and `Basis` defines no structural
   `==`, so the ordinary spelling -- `derivative_basis(zb, 2)` written out at each
   lift -- walked straight past the guard into a singular per-mode system solved
   in the least-squares sense.

Uniquely-prefixed names (tla_*) -- the full suite shares the Main namespace.
"""

using Test
using Tarang
using MPI

MPI.Initialized() || MPI.Init()

# One heat-equation solver per axis ordering, with zero-dimensional taus so the
# problem always builds; the tangential tau is measured separately.
function tla_probe(order::Tuple; Nz=10, Nx=8, Ny=6)
    coords = CartesianCoordinates(order...)
    dist = Distributor(coords; comm=MPI.COMM_SELF, dtype=Float64, architecture=CPU())
    built = Dict("z" => ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0)),
                 "x" => RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π)))
    if "y" in order
        built["y"] = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2π))
    end
    bases = Tuple(built[c] for c in order)
    b = ScalarField(Domain(dist, bases), "b")
    t1 = ScalarField(dist, "tau1", (), Float64)
    t2 = ScalarField(dist, "tau2", (), Float64)
    tangential = Tuple(built[c] for c in order if c != "z")
    tt = ScalarField(dist, "tau_tangential", tangential, Float64)
    lb = derivative_basis(built["z"], 2)
    problem = InitialValueProblem([b, t1, t2])
    add_parameters!(problem; kappa=0.1, l1=lift(t1, lb, -1), l2=lift(t2, lb, -2))
    add_equation!(problem, "dt(b) - kappa*lap(b) + l1 + l2 = 0")
    add_bc!(problem, "b(z=0)=0")
    add_bc!(problem, "b(z=1)=0")
    solver = InitialValueSolver(problem, RK222(); dt=1e-3)
    sp = first(solver.problem.parameters["subproblems"])
    return sp, b, t1, tt
end

@testset "a tau field carries one DOF per subproblem in every axis ordering" begin
    # The separable axes are resolved by COORDINATE, so where the Chebyshev axis
    # sits in the ordering cannot change any field's per-subproblem DOF count.
    for order in (("x", "z"), ("z", "x"), ("x", "y", "z"), ("z", "x", "y"), ("x", "z", "y"))
        @testset "axes $(join(order, ","))" begin
            sp, b, t0, tt = tla_probe(order)
            @test Tarang.subproblem_field_size(sp, t0) == 1   # zero-dimensional tau
            @test Tarang.subproblem_field_size(sp, tt) == 1   # tangential-basis tau
            # The solved field still spans the full coupled (Chebyshev) axis.
            @test Tarang.subproblem_field_size(sp, b) == 10
        end
    end
end

@testset "a tangential-basis tau solves with the Chebyshev axis first" begin
    # Identical physics to the x,z ordering; only the axis order and the tau
    # declaration change. Chebyshev-first is the layout distributed Cheb x Fourier
    # requires, so this cell must work, not raise DimensionMismatch.
    Nz, Nx, kappa, nsteps, dt = 24, 8, 0.1, 100, 1e-3
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; comm=MPI.COMM_SELF, dtype=Float64, architecture=CPU())
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    b = ScalarField(Domain(dist, (zb, xb)), "b")
    set!(b, (z, x) -> sin(π * z))
    t1 = ScalarField(dist, "tau1", (xb,), Float64)
    t2 = ScalarField(dist, "tau2", (xb,), Float64)
    lb = derivative_basis(zb, 2)
    problem = InitialValueProblem([b, t1, t2])
    add_parameters!(problem; kappa=kappa, l1=lift(t1, lb, -1), l2=lift(t2, lb, -2))
    add_equation!(problem, "dt(b) - kappa*lap(b) + l1 + l2 = 0")
    add_bc!(problem, "b(z=0)=0")
    add_bc!(problem, "b(z=1)=0")
    solver = InitialValueSolver(problem, RK443(); dt=dt)
    for _ in 1:nsteps
        step!(solver)
    end
    grid = Array(grid_data!(b))
    zs = (1 .- cos.(π .* (0:Nz-1) ./ (Nz - 1))) ./ 2
    want = [exp(-kappa * π^2 * solver.sim_time) * sin(π * z) for z in zs, _ in 1:Nx]
    @test maximum(abs, grid .- want) < 1e-8
    @test maximum(abs, grid[1, :]) < 1e-12
    @test maximum(abs, grid[end, :]) < 1e-12
end

# Two lifts of DIFFERENT tau variables onto the SAME mode make those matrix
# columns identical, so every per-mode stage system is singular.
function tla_duplicate_problem(mk1, mk2)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; comm=MPI.COMM_SELF, dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
    u = ScalarField(dist, "u", (xb, zb), Float64)
    t1 = ScalarField(dist, "tau1", (), Float64)
    t2 = ScalarField(dist, "tau2", (), Float64)
    problem = LinearBoundaryValueProblem([u, t1, t2])
    add_parameters!(problem; l1=mk1(t1, zb), l2=mk2(t2, zb))
    add_equation!(problem, "lap(u) + l1 + l2 = 0")
    add_bc!(problem, "u(z=0)=0")
    add_bc!(problem, "u(z=1)=0")
    # The guard runs when the equation expressions are built, i.e. at solver build.
    return try
        solve!(BoundaryValueSolver(problem))
        nothing
    catch err
        err
    end
end

@testset "duplicate tau lifts are refused however the lift basis is spelled" begin
    shared = nothing
    # (a) one basis object reused -- the spelling `===` already caught
    refused_shared = tla_duplicate_problem(
        (t, zb) -> (shared = derivative_basis(zb, 2); lift(t, shared, -1)),
        (t, zb) -> lift(t, shared, -1))
    @test refused_shared isa Exception
    @test occursin("SAME mode", sprint(showerror, refused_shared))

    # (b) the ordinary spelling: derivative_basis written out at each lift, so
    # each call returns a fresh object and `===` is false.
    refused_fresh = tla_duplicate_problem(
        (t, zb) -> lift(t, derivative_basis(zb, 2), -1),
        (t, zb) -> lift(t, derivative_basis(zb, 2), -1))
    @test refused_fresh isa Exception
    @test occursin("SAME mode", sprint(showerror, refused_fresh))

    # The well-posed form must still be accepted -- the guard may not fire on
    # distinct modes, or every channel problem would stop building.
    accepted = tla_duplicate_problem(
        (t, zb) -> lift(t, derivative_basis(zb, 2), -1),
        (t, zb) -> lift(t, derivative_basis(zb, 2), -2))
    @test accepted === nothing
end
