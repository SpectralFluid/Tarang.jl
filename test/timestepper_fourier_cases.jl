# The Float64 cases run on CPU, JLArray, CUDA and ROCm. Metal runs the shared
# variable-step and retry checks with Float32 fields.
using LinearAlgebra: SingularException

function check_fourier_timestepper_cases(arch)
    check_variable_timestep_cases(arch, Float64)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=arch)
    bases = map(c -> RealFourier(coords[c]; size=8, bounds=(0.0, 2π)), ("x", "y"))
    domain = Domain(dist, bases)
    xs = (2π/8) .* (0:7)

    @testset "Vector diffusion evolves each Fourier mode" begin
        initial = ([cos(2x) for x in xs, y in xs], [sin(y) for x in xs, y in xs])
        for ts in (RK222(), SBDF2())
            u = VectorField(domain, "u")
            for (component, values) in zip(u.components, initial)
                copyto!(grid_data!(component), values)
            end
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) - 0.5*lap(u) = 0")
            solver = InitialValueSolver(problem, ts; dt=0.01)
            for _ in 1:10
                step!(solver)
            end
            @test isempty(Tarang.compiled_problem(problem).identity_approx_ops)
            tol = ts isa RK222 ? 3e-6 : 3e-4
            for (f, values, rate) in zip(u.components, initial, (2.0, 0.5))
                @test maximum(abs, Array(grid_data!(f)) .- exp(-0.1rate) .* values) < tol
            end
            v = VectorField(domain, "v")
            @test Tarang._diagonal_Lhat_from_expr(lap(v), first(u.components)) === nothing
        end

        # The derivative and fractional Laplacian use the same component ordering.
        u = VectorField(domain, "u")
        for (component, values) in zip(u.components, initial)
            copyto!(grid_data!(component), values)
        end
        problem = InitialValueProblem([u])
        add_equation!(problem, "dt(u) + fraclap(u, 0.5) + d(u,x) = 0")
        step!(InitialValueSolver(problem, RK111(); dt=0.1))
        amp = inv(1 + 0.1 * (2 + 2im))
        @test Array(grid_data!(u.components[1])) ≈ [real(amp * cis(2x)) for x in xs, y in xs] atol=1e-12
        @test Array(grid_data!(u.components[2])) ≈ initial[2] ./ 1.1 atol=1e-12
    end

    @testset "Complex implicit coefficient retains phase" begin
        coord = CartesianCoordinates("x")
        cdist = Distributor(coord; dtype=ComplexF64, device=arch)
        basis = ComplexFourier(coord["x"]; size=8, bounds=(0.0, 2π))
        for ts in (RK222(), SBDF2())
            u = ScalarField(Domain(cdist, (basis,)), "u")
            fill!(grid_data!(u), 1)
            problem = InitialValueProblem([u])
            add_parameters!(problem; omega=1im)
            add_equation!(problem, "dt(u) + omega*u = 0")
            solver = InitialValueSolver(problem, ts; dt=0.02)
            for _ in 1:10
                step!(solver)
            end
            tol = ts isa RK222 ? 4e-6 : 4e-4
            @test maximum(abs, Array(grid_data!(u)) .- cis(-0.2)) < tol
        end
    end

    @testset "Singular Fourier steps preserve the accepted solution" begin
        function growth(ts)
            u = ScalarField(domain, "u")
            fill!(grid_data!(u), 1)
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) - u = 0")
            return InitialValueSolver(problem, ts; dt=0.1), u
        end
        for (ts, rejected_dt, first_dt) in ((RK111(), 1.0, 0.1), (RK443(), 2.0, 0.1),
                                             (SBDF2(), 1.0, 0.0), (SBDF2(), 1.5, 1.5))
            solver, u = growth(ts)
            control, reference = growth(ts)
            if first_dt > 0
                step!(solver, first_dt)
                step!(control, first_dt)
            end
            before = copy(Array(grid_data!(u)))
            @test_throws Union{ArgumentError,SingularException} step!(solver, rejected_dt)
            @test (solver.sim_time, solver.iteration, solver.dt) ==
                  (control.sim_time, control.iteration, control.dt)
            @test Array(grid_data!(u)) ≈ before atol=1e-13
            step!(solver, 0.2)
            step!(control, 0.2)
            @test Array(grid_data!(u)) ≈ Array(grid_data!(reference)) atol=1e-13
        end

        u = ScalarField(domain, "u")
        fill!(grid_data!(u), 1)
        p = InitialValueProblem([u])
        add_equation!(p, "dt(u) = 0")
        s = InitialValueSolver(p, RK111(); dt=0.1)
        op = SpectralLinearOperator(u, :laplacian)
        set_spectral_linear_operator!(s, op)
        step!(s)
        for invalid in (-10.0, Inf)
            fill!(op.coefficients, invalid)
            @test_throws ArgumentError step!(s)
            @test s.iteration == 1 && s.sim_time == 0.1
            @test all(isone, Array(grid_data!(u)))
        end
        fill!(op.coefficients, 0)
        step!(s)
        @test all(isone, Array(grid_data!(u)))
    end

    @testset "ETD startup and variable-step history at zero linear operator" begin
        for ts in (ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
            u = ScalarField(domain, "u")
            fill!(grid_data!(u), 1)
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) = -u*u")
            solver = InitialValueSolver(problem, ts; dt=0.1)
            expected = 1.0
            prev_f, prev_h = 0.0, 0.0
            for h in (0.1, 0.05, 0.08, 0.02)
                f = -expected^2
                if ts isa ETD_RK222 || iszero(prev_h)
                    expected += (h/2) * (f - (expected + h*f)^2)
                else
                    expected += h * (f + h/(2prev_h) * (f - prev_f))
                end
                prev_f, prev_h = f, h
                step!(solver, h)
                @test maximum(abs, Array(grid_data!(u)) .- expected) < 2e-13
            end
        end
    end
end

function check_variable_timestep_cases(arch, ::Type{T}) where {T<:AbstractFloat}
    dom = PeriodicDomain(8; device=arch, dtype=T)
    try
        @testset "Invalid timesteps leave the accepted state and clock intact" begin
            u = ScalarField(dom, "u")
            set!(u, zero(T))
            p = InitialValueProblem([u])
            add_equation!(p, "dt(u) = 1")
            for dt in (0.0, -0.01, NaN, Inf, -Inf)
                @test_throws ArgumentError InitialValueSolver(p, RK222(); dt)
            end
            s = InitialValueSolver(p, RK222(); dt=0.125)
            step!(s, 0.125f0)
            accepted = copy(Array(grid_data!(u)))
            history = copy(s.timestepper_state.dt_history)
            for (t, dt) in ((0.125, 0.0), (0.125, -0.01), (0.125, Inf), (0.125, NaN),
                            (1e12, eps(1e12)/4), (floatmax(Float64), floatmax(Float64)))
                s.sim_time = t
                @test_throws ArgumentError step!(s, dt)
                @test (s.sim_time, s.iteration, s.dt) == (t, 1, 0.125)
                @test s.timestepper_state.dt_history == history
                @test Array(grid_data!(u)) == accepted
            end
        end

        @testset "ETD stage failures and retries" begin
            for ts in (ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
                u = ScalarField(dom, "u")
                set!(u, zero(T))
                p = InitialValueProblem([u])
                add_equation!(p, "dt(u) = 0")
                fail_at = Ref(0.125)
                f = DeterministicForcing((x,t,p) -> begin
                    t == fail_at[] && error("forcing unavailable")
                    zero.(x) .+ T(1+t)
                end, (8,); architecture=arch, dtype=T)
                add_stochastic_forcing!(p, :u, f)
                s = InitialValueSolver(p, ts; dt=0.125)

                @test_throws "forcing unavailable" step!(s)
                @test (s.iteration, s.sim_time, s.timestepper_state) == (0, 0.0, nothing)
                @test all(iszero, Array(grid_data!(u)))
                fail_at[] = Inf
                step!(s)
                before = copy(Array(grid_data!(u)))
                fail_at[] = ts isa ETD_RK222 ? 0.375 : 0.125
                @test_throws "forcing unavailable" step!(s, 0.25)
                @test (s.iteration, s.sim_time, s.dt) == (1, 0.125, 0.125)
                @test Array(grid_data!(u)) == before

                fail_at[] = Inf
                for h in (0.0625, 0.125)
                    step!(s, h)
                    t = s.sim_time
                    @test maximum(abs, Array(grid_data!(u)) .- (t + t^2/2)) < 32eps(T)
                end
            end
        end

        @testset "CNLF2 with alternating steps" begin
            errors = map((16, 32, 64)) do n
                u = ScalarField(dom, "u")
                set!(u, zero(T))
                p = InitialValueProblem([u])
                add_equation!(p, "dt(u) = 0")
                f = DeterministicForcing((x,t,p) -> zero.(x) .+ T(t+t^2), (8,);
                                         architecture=arch, dtype=T)
                add_stochastic_forcing!(p, :u, f)
                s = InitialValueSolver(p, Tarang.CNLF2(); dt=1.0/n)
                for i in 1:n
                    step!(s, (isodd(i) ? 0.8 : 1.2)/n)
                end
                maximum(abs, Array(grid_data!(u)) .- 5/6)
            end
            @test all(log2.(errors[1:2] ./ errors[2:3]) .> 1.8)
        end
    finally
        close(dom.dist)
    end
end
