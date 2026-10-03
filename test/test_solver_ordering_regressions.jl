using Test
using Tarang

function ordering_regression_solver(ts, equations; initial=(0.0, 0.0), forcing=false,
                                    interpreted=false, tau_position=nothing)
    domain = PeriodicDomain(8)
    u, v = ScalarField(domain, "u"), ScalarField(domain, "v")
    set!(u, initial[1]); set!(v, initial[2])
    variables = ScalarField[u, v]
    if tau_position !== nothing
        tau = ScalarField(u.dist, "tau", (), Float64)
        insert!(variables, tau_position, tau)
    end
    problem = InitialValueProblem(variables)
    foreach(eq -> add_equation!(problem, eq), equations)
    if forcing
        add_stochastic_forcing!(problem, :u,
            DeterministicForcing((x, t, p) -> (1 + t) .* cos.(x), (8,)))
        add_stochastic_forcing!(problem, :v,
            DeterministicForcing((x, t, p) -> (2 - t) .* sin.(x), (8,)))
    end
    solver = InitialValueSolver(problem, ts; dt=0.01)
    interpreted && (solver.rhs_plan = nothing)
    return solver, u, v
end

@testset "Global solvers preserve equation-space RHS" begin
    schemes = (RK111(), RK222(), RK443(), Tarang.RKGFY(), Tarang.RKSMR(),
               CNAB1(), CNAB2(), SBDF1(), SBDF2(), SBDF3(), SBDF4(),
               Tarang.MCNAB2(), Tarang.CNLF2(), ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
    @testset "$(nameof(typeof(ts)))" for ts in schemes
        # Includes a symmetric indefinite permutation mass matrix: its zero
        # diagonal must not trigger unpivoted sparse LDL.
        for scale in (1, 2)
            equations = ["dt(u) = 1", "$(scale)*dt(v) = $(2scale)"]
            solver, u, v = ordering_regression_solver(ts, reverse(equations))
            for dt in (0.01, 0.02, 0.015, 0.01)
                step!(solver, dt)
                @test Array(grid_data!(u)) ≈ fill(solver.sim_time, 8) atol=1e-12
                @test Array(grid_data!(v)) ≈ fill(2solver.sim_time, 8) atol=1e-12
            end
        end

        # Equivalent linear combinations of equations introduce coupled time
        # derivatives, including variables targeted by more than one RHS.
        reference, ru, rv = ordering_regression_solver(ts,
            ["dt(u) + u = 1", "dt(v) + v = 2"]; initial=(0.3, 0.7))
        coupled, cu, cv = ordering_regression_solver(ts,
            ["dt(u) + dt(v) + u + v = 3", "dt(u) - dt(v) + u - v = -1"];
            initial=(0.3, 0.7))
        for dt in (0.01, 0.02, 0.015, 0.01)
            step!(reference, dt); step!(coupled, dt)
            @test Array(grid_data!(cu)) ≈ Array(grid_data!(ru)) atol=1e-12
            @test Array(grid_data!(cv)) ≈ Array(grid_data!(rv)) atol=1e-12
        end
    end

    @testset "reordered nonlinear equations and registered forcing" begin
        equations = ["dt(u) + u = v*v", "2*dt(v) + v = u*u"]
        for ts in (RK222(), SBDF2(), ETD_CNAB2()), interpreted in (false, true)
            a, au, av = ordering_regression_solver(ts, equations;
                initial=(0.1, 0.2), forcing=true, interpreted)
            b, bu, bv = ordering_regression_solver(ts, reverse(equations);
                initial=(0.1, 0.2), forcing=true, interpreted)
            for dt in (0.01, 0.02, 0.015)
                step!(a, dt); step!(b, dt)
                @test Array(grid_data!(au)) ≈ Array(grid_data!(bu)) atol=1e-12
                @test Array(grid_data!(av)) ≈ Array(grid_data!(bv)) atol=1e-12
            end
        end
        solver, _, _ = ordering_regression_solver(RK222(),
            ["dt(u) + dt(v) = 3", "dt(u) - dt(v) = -1"]; forcing=true)
        @test_throws "ambiguous equation" step!(solver)
        @test solver.iteration == 0
    end

    @testset "coupled vector mass with scalar right-hand sides" begin
        for (f1, f2, du, dv) in ((0, 0, 0.0, 0.0), (1, 3, 2.0, -1.0))
            domain = PeriodicDomain(6, 6)
            u, v = VectorField(domain, "u"), VectorField(domain, "v")
            foreach(c -> set!(c, 1.0), u.components)
            foreach(c -> set!(c, 2.0), v.components)
            problem = InitialValueProblem([u, v])
            add_equation!(problem, "dt(u) + dt(v) = $f1")
            add_equation!(problem, "dt(u) - dt(v) = $f2")
            solver = InitialValueSolver(problem, RKSMR(); dt=0.01)
            step!(solver)
            for component in u.components
                @test maximum(abs, grid_data!(component) .- (1 + 0.01du)) < 1e-12
            end
            for component in v.components
                @test maximum(abs, grid_data!(component) .- (2 + 0.01dv)) < 1e-12
            end
        end
    end
end

@testset "Algebraic refresh preserves complete equations and evolved fields" begin
    for constraint in ("v - u - w = 0", "v - (u + w) = 0", "u + w - v = 0",
                       "2*v - 2*u - 2*w = 0", "v + w - u = 0")
        domain = PeriodicDomain(8)
        u, v, w = (ScalarField(domain, name) for name in ("u", "v", "w"))
        sign = constraint == "v + w - u = 0" ? -1.0 : 1.0
        set!(u, 0.0); set!(v, sign); set!(w, 1.0)
        problem = InitialValueProblem([u, v, w])
        foreach(eq -> add_equation!(problem, eq), ["dt(u) = v", constraint, "dt(w) = 0"])
        solver = InitialValueSolver(problem, RK222(); dt=0.01)
        for _ in 1:20
            step!(solver)
        end
        @test maximum(abs, Array(grid_data!(u)) .- sign * expm1(0.2)) < 5e-6
        @test Array(grid_data!(v)) ≈ Array(grid_data!(u)) .+ sign atol=1e-12
        @test Array(grid_data!(w)) ≈ ones(8) atol=1e-12
    end

    # Direct RHS refresh also preserves a nonzero algebraic RHS. The global
    # DAE stepper intentionally declines this projection, so inspect its RHS.
    solver, u, v = ordering_regression_solver(RK222(), ["dt(u) = v", "v - u = 2"];
                                              initial=(0.5, 2.5))
    rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)
    @test Array(grid_data!(rhs[1])) ≈ fill(2.5, 8) atol=1e-12
    @test Array(grid_data!(u)) ≈ fill(0.5, 8) atol=1e-12
end

@testset "Global vector unpack consumes every tau column" begin
    for ts in (RK222(), CNAB2(), SBDF2()), position in (2, 3)
        equations = ["dt(u) = 1", "tau = 0", "dt(v) = 2"]
        solver, u, v = ordering_regression_solver(ts, equations; tau_position=position)
        for dt in (0.01, 0.02, 0.015)
            step!(solver, dt)
            @test Array(grid_data!(u)) ≈ fill(solver.sim_time, 8) atol=1e-12
            @test Array(grid_data!(v)) ≈ fill(2solver.sim_time, 8) atol=1e-12
        end
    end
end
