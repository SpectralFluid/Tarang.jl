using Test
using Tarang

function tmg_solver(ts, equation; fields=1)
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    fill!(grid_data!(u), 1.0)
    variables = fields == 1 ? [u] : [u, ScalarField(domain, "v")]
    problem = InitialValueProblem(variables)
    add_equation!(problem, equation)
    fields == 1 || add_equation!(problem, "dt(v) = 0")
    return InitialValueSolver(problem, ts; dt=0.01), u
end

function tmg_refusal(solver, u, advance)
    before = copy(Array(grid_data!(u)))
    err = try
        advance(solver)
        nothing
    catch ex
        ex
    end
    @test err isa ArgumentError
    @test occursin("non-identity mass", err === nothing ? "" : sprint(showerror, err))
    @test Array(grid_data!(u)) == before
    @test solver.sim_time == 0
    @test solver.iteration == 0
end

@testset "Identity-mass field paths reject unsupported equations before stepping" begin
    @testset "DiagonalIMEX implicit and SBDF2 explicit paths" begin
        for ts in (Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443(), Tarang.DiagonalIMEX_SBDF2())
            for equation in ("2*dt(u) + u = 0", "0.5*dt(u) + dt(u) + u = 0")
                solver, u = tmg_solver(ts, equation)
                tmg_refusal(solver, u, step!)
                # Failed validation must not be memoized as success.
                tmg_refusal(solver, u, step!)
            end
        end
        solver, u = tmg_solver(Tarang.DiagonalIMEX_SBDF2(), "2*dt(u) = -u")
        tmg_refusal(solver, u, step!)
    end

    @testset "Explicit multistep field kernel" begin
        for (ts, method) in ((CNAB1(), :cnab1), (CNAB2(), :cnab2),
                             (SBDF1(), :sbdf1), (SBDF2(), :sbdf2),
                             (SBDF3(), :sbdf3), (SBDF4(), :sbdf4),
                             (Tarang.MCNAB2(), :cnab2), (Tarang.CNLF2(), :cnlf2))
            solver, u = tmg_solver(ts, "2*dt(u) = -u")
            tmg_refusal(solver, u, s -> Tarang._step_explicit_multistep_field!(
                Tarang._ensure_timestepper_state!(s, s.dt), s, method))
        end
    end

    @testset "Skipped global assembly does not imply identity mass" begin
        solver, u = tmg_solver(RK222(), "2*dt(u) = -u")
        tmg_refusal(solver, u, s -> Tarang._check_explicit_rk_mass_matrix!(
            Tarang._ensure_timestepper_state!(s, s.dt), s, nothing))
    end

    @testset "Explicit CPU fallback beside an algebraic constraint" begin
        # The constraint row makes the assembled M singular, so the CPU explicit RK
        # fallback has no mass factorization and takes k = F: `2*dt(u) = -u` stepped
        # as `dt(u) = -u` while the device refused it. With identity mass it must
        # step and leave the constrained variable refreshed (it used to stay 0).
        function tmg_constrained(ts, equation)
            domain = PeriodicDomain(8)
            u = ScalarField(domain, "u")
            v = ScalarField(domain, "v")
            fill!(grid_data!(u), 1.0)
            problem = InitialValueProblem([u, v])
            add_equation!(problem, equation)
            add_equation!(problem, "v - u = 0")
            return InitialValueSolver(problem, ts; dt=0.01), u, v
        end
        for ts in (Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443())
            solver, u, _ = tmg_constrained(ts, "2*dt(u) = -u")
            tmg_refusal(solver, u, step!)
            solver, u, v = tmg_constrained(ts, "dt(u) = -u")
            step!(solver)
            @test maximum(abs, Array(grid_data!(u)) .- exp(-0.01)) < 1e-6
            @test Array(grid_data!(v)) ≈ Array(grid_data!(u)) atol=1e-14
        end
    end

    @testset "Coupled time derivatives are not independent identity rows" begin
        solver, u = tmg_solver(Tarang.DiagonalIMEX_RK222(), "dt(u) + dt(v) + u = 0"; fields=2)
        tmg_refusal(solver, u, step!)
    end

    @testset "Equivalent identity expressions remain valid" begin
        for equation in ("dt(u) + u = 0", "1*dt(u) + u = 0",
                         "0.5*dt(u) + 0.5*dt(u) + u = 0")
            solver, u = tmg_solver(Tarang.DiagonalIMEX_RK443(), equation)
            step!(solver)
            @test maximum(abs, Array(grid_data!(u)) .- exp(-0.01)) < 1e-8
        end
    end

    @testset "Global mass solves remain available" begin
        for ts in (RK443(), ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
            solver, u = tmg_solver(ts, "2*dt(u) + u = 0")
            step!(solver)
            @test maximum(abs, Array(grid_data!(u)) .- exp(-0.005)) < 1e-8
        end
        solver, u = tmg_solver(Tarang.DiagonalIMEX_RK443(), "2*dt(u) = -u")
        step!(solver)
        @test maximum(abs, Array(grid_data!(u)) .- exp(-0.005)) < 1e-8
    end
end
