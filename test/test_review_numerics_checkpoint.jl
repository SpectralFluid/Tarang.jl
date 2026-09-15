using Test
using Tarang
import NetCDF

@testset "Checkpoint clocks agree across every slab" begin
    clock = Dict("sim_time" => 1.0, "iteration" => 10, "dt" => 0.1)
    for kind in (:valid, :missing, :mismatch), key in keys(clock)
        @testset "$kind $key" begin
            mktempdir() do path
                for rank in 0:1
                    file = joinpath(path, "checkpoint_p$rank.nc")
                    Tarang.write_local_slab(file, "u", fill(Float64(rank + 1), 4),
                                           [4rank], [8])
                    attrs = copy(clock)
                    if rank == 1 && kind == :missing
                        delete!(attrs, key)
                    elseif rank == 1 && kind == :mismatch
                        attrs[key] = 2clock[key]
                    end
                    NetCDF.ncputatt(file, "global", attrs)
                end

                u = ScalarField(PeriodicDomain(8), "u")
                set!(u, -3.0)
                problem = InitialValueProblem([u])
                add_equation!(problem, "dt(u) = 0")
                solver = InitialValueSolver(problem, RK222(); dt=0.02)
                if kind == :valid
                    load_state!(solver, path)
                    @test (solver.sim_time, solver.iteration, solver.dt) == (1.0, 10, 0.1)
                    @test grid_data!(u) == vcat(ones(4), fill(2.0, 4))
                else
                    @test_throws ErrorException load_state!(solver, path)
                    @test (solver.sim_time, solver.iteration, solver.dt) == (0.0, 0, 0.02)
                    @test all(==(-3.0), grid_data!(u))
                end
            end
        end
    end
end

@testset "Fourier interpolation follows real versus complex storage" begin
    for N in (7, 8)
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=ComplexF64)
        basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (basis,), ComplexF64)
        f(x) = cis(x) + 0.4im * cis(-2x)
        set!(u, f)
        for x in (0.0, 0.3, 2.1)
            @test evaluate(interpolate(u, coords["x"], x)) ≈ f(x) atol=1e-12
        end
    end

    # The spectrum has the same length for rfft and fft at N=1 and N=2.
    for N in (1, 2), dtype in (Float64, ComplexF64)
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype)
        basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (basis,), dtype)
        value = dtype <: Complex ? 2.0 + 3im : 2.0
        set!(u, value)
        @test evaluate(interpolate(u, coords["x"], 0.3)) ≈ value atol=1e-12
    end

    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=ComplexF64)
    bases = Tuple(RealFourier(coords[name]; size=8, bounds=(0.0, 2π)) for name in ("x", "y"))
    u = ScalarField(dist, "u", bases, ComplexF64)
    f(x, y) = cis(x - 2y) + 0.25im
    set!(u, f)
    nodes = (0:7) .* (2π / 8)
    @test evaluate(interpolate(u, coords["x"], 0.3)) ≈ f.(0.3, nodes) atol=1e-12
    @test evaluate(interpolate(u, coords["y"], 0.3)) ≈ f.(nodes, 0.3) atol=1e-12
end

@testset "Integration uses the active scaled collocation grid" begin
    for B in (RealFourier, ChebyshevT, Legendre, ChebyshevU, Jacobi), scale in (0.75, 1.5)
        @testset "$B at scale $scale" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64)
            periodic = B == RealFourier
            bounds = periodic ? (0.0, 2π) : (-1.0, 1.0)
            basis = B(coords["x"]; size=8, bounds)
            u = ScalarField(dist, "u", (basis,), Float64)
            set!(u, periodic ? (x -> 2 + cos(x)) : (x -> 1 + x^2))
            set_scales!(u, scale)
            expected = periodic ? 4π : 8 / 3
            @test evaluate(integrate(u, coords["x"])) ≈ expected atol=1e-10
            @test evaluate(average(u, coords["x"])) ≈ expected / (bounds[2] - bounds[1]) atol=1e-10
        end
    end

    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    bx = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    bz = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
    u = ScalarField(dist, "u", (bx, bz), Float64)
    set!(u, (x, z) -> (2 + cos(x)) * (1 + z^2))
    set_scales!(u, (1.5, 1.25))
    @test evaluate(integrate(u, (coords["x"], coords["z"]))) ≈ 32π / 3 atol=1e-10
    reduced = evaluate(integrate(u, coords["x"]))
    @test reduced.scales == (1.25,)
    @test evaluate(interpolate(reduced, coords["z"], 0.3)) ≈ 4π * 1.09 atol=1e-10
end

@testset "Finite eigenvalues are independent of dimensional magnitude" begin
    for decay in (1.0, 1e10, 1e11)
        u = ScalarField(PeriodicDomain(8), "u")
        problem = EigenvalueProblem([u])
        add_equation!(problem, "dt(u) + $decay*u = 0")
        solver = EigenvalueSolver(problem; nev=2, which=:SM)
        values, vectors = solve!(solver)
        @test length(values) == 2
        @test size(vectors, 2) == 2
        @test values ≈ fill(ComplexF64(-decay), 2)
    end

    # Large physical eigenvalues must also survive a singular tau mass matrix.
    L = 1e-5
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype=Float64)
    basis = ChebyshevT(coords["z"]; size=24, bounds=(0.0, L))
    u = ScalarField(dist, "u", (basis,), Float64)
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    problem = EigenvalueProblem([u, tau1, tau2])
    lb = derivative_basis(basis, 2)
    add_parameters!(problem; l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2), L)
    add_equation!(problem, "dt(u) - lap(u) - l1 - l2 = 0")
    add_bc!(problem, "u(z=0) = 0")
    add_bc!(problem, "u(z=L) = 0")
    values, _ = solve!(EigenvalueSolver(problem; nev=3, which=:SM))
    @test length(values) == 3
    @test sort(real.(values); rev=true) ≈ [-(n * π / L)^2 for n in 1:3] rtol=1e-6
end
