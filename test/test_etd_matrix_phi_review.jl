using Test
using Tarang
using LinearAlgebra
using SparseArrays

@testset "ETD matrix phi functions preserve singular couplings" begin
    @testset "nilpotent Jordan blocks" begin
        for T in (Float64, ComplexF64), n in (2, 3), dt in (1e-10, 0.1, 2.0, 100.0)
            A = zeros(T, n, n)
            for i in 1:n-1
                A[i, i + 1] = 1
            end
            original = copy(A)
            z = dt .* A
            identity = Matrix{T}(I, n, n)
            # A^3 = 0, so these Taylor polynomials are exact for either size.
            expected = (identity + z + z*z/2,
                        identity + z/2 + z*z/6,
                        identity/2 + z/6 + z*z/24)
            actual = Tarang.phi_functions_matrix(A, dt)
            for k in 1:3
                @test actual[k] ≈ expected[k] rtol=1e-12 atol=1e-13
            end
            @test A == original
        end
    end

    @testset "near-singular and nonnormal matrices" begin
        # Independent block-exponential oracle: its top row of n×n blocks is
        # [exp(z), phi1(z), phi2(z)], with no division or eigenvector inverse.
        for A in ([1e-12 1.0; 0.0 -1e-12],
                  [-0.3 1.0; 0.0 -0.3],
                  [-30.0 100.0; 0.0 -30.0],
                  ComplexF64[-0.3+0.4im 1.0; 0.0 -0.3+0.4im])
            n = size(A, 1)
            z = 0.25 .* A
            block = zeros(eltype(z), 3n, 3n)
            block[1:n, 1:n] .= z
            for i in 1:2n
                block[i, i + n] = 1
            end
            oracle = exp(block)
            actual = Tarang.phi_functions_matrix(A, 0.25)
            for k in 1:3
                @test actual[k] ≈ oracle[1:n, (k-1)*n+1:k*n] rtol=1e-12 atol=1e-13
            end
        end
    end

    @testset "singular diffusion and dense-size safeguard" begin
        Q = [1.0 1.0 1.0; 1.0 -1.0 0.0; 1.0 1.0 -2.0]
        Q = Matrix(qr(Q).Q)
        for eigenvalues in ([0.0, -2.0, -20.0], [0.0, -100.0, -1000.0]),
            basis in (Matrix{Float64}(I, 3, 3), Q)
            A = basis * Diagonal(eigenvalues) * basis'
            expected = (basis * Diagonal(exp.(eigenvalues)) * basis',
                        basis * Diagonal([iszero(z) ? 1.0 : expm1(z)/z for z in eigenvalues]) * basis',
                        basis * Diagonal([iszero(z) ? 0.5 : (expm1(z)-z)/z^2 for z in eigenvalues]) * basis')
            actual = Tarang.phi_functions_matrix(A, 1.0)
            for k in 1:3
                @test actual[k] ≈ expected[k] rtol=1e-11 atol=1e-12
            end
        end
        @test_throws ArgumentError Tarang.phi_functions_matrix(spzeros(4097, 4097), 0.1)
    end

    @testset "coupled ETD equations with time-dependent forcing" begin
        # u_t = v, v_t = t*cos(x), u(0)=0, v(0)=cos(x).
        # The linear operator has a nilpotent Jordan block in every
        # Fourier mode. Linear-in-time forcing also exercises phi1 and phi2.
        for timestepper in (ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
            domain = PeriodicDomain(8)
            u = ScalarField(domain, "u")
            v = ScalarField(domain, "v")
            set!(u, (x,) -> 0.0)
            set!(v, (x,) -> cos(x))
            problem = InitialValueProblem([u, v])
            add_equation!(problem, "dt(u) - v = 0")
            add_equation!(problem, "dt(v) = 0")
            forcing = DeterministicForcing((x, t, p) -> t .* cos.(x), (8,))
            add_stochastic_forcing!(problem, :v, forcing)
            solver = InitialValueSolver(problem, timestepper; dt=0.1)
            for dt in (0.1, 0.05, 0.15)
                step!(solver, dt)
            end
            t = solver.sim_time
            x = collect(0:7) .* (2π/8)
            @test u["g"] ≈ (t + t^3/6) .* cos.(x) atol=1e-11
            @test v["g"] ≈ (1 + t^2/2) .* cos.(x) atol=1e-11
            @test haskey(solver.timestepper_state.timestepper_data, :etd_phi)
        end
    end
end

@testset "Oversized ETD systems retain the implicit fallback" begin
    # 8192 real grid points produce 4097 Fourier coefficients, just beyond the
    # dense matrix-function limit. A non-identity mass checks M is retained by
    # the fallback, and changing dt exercises both multistep history branches.
    for ts in (ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
        domain = PeriodicDomain(8192)
        u = ScalarField(domain, "u")
        fill!(grid_data!(u), 1.0)
        problem = InitialValueProblem([u])
        add_equation!(problem, "2*dt(u) + 3*u = 0")
        solver = InitialValueSolver(problem, ts; dt=0.01)
        dts = (0.01, 0.015, 0.007, 0.02)
        amplitudes = [1.0]
        for (i, dt) in enumerate(dts)
            # RK222 and CNAB2 fall back to CNAB2 (CNAB1 startup). SBDF2's
            # ETDRK2 startup therefore uses CNAB1, then SBDF1 and SBDF2.
            current = amplitudes[end]
            expected = if !(ts isa ETD_SBDF2) || i == 1
                current * (2 - 1.5dt) / (2 + 1.5dt)
            elseif i == 2
                current * 2 / (2 + 3dt)
            else
                w = dt / dts[i - 1]
                a0 = (1 + 2w) / ((1 + w) * dt)
                a1 = -(1 + w) / dt
                a2 = w^2 / ((1 + w) * dt)
                (-2a1 * current - 2a2 * amplitudes[end - 1]) / (2a0 + 3)
            end
            step!(solver, dt)
            @test maximum(abs, grid_data!(u) .- expected) < 1e-11
            @test solver.iteration == i
            push!(amplitudes, expected)
        end
        cache = solver.timestepper_state.timestepper_data
        @test size(Tarang._get_problem_matrix(problem, "L_matrix")) == (4097, 4097)
        @test get(cache, :L_eff, nothing) === nothing
        @test !haskey(cache, :L_eff_neg)
        @test !haskey(cache, :etd_phi)

        # The preflight handles identity-without-M and rejects singular M
        # before invoking any fallback. Neither path may materialize L.
        calls = Ref(0)
        fallback = (_, _) -> (calls[] += 1)
        large_L = spzeros(4097, 4097)
        state = Tarang.TimestepperState(ts, 0.01, ScalarField[])
        @test Tarang._etd_oversized_fallback!(state, solver, large_L, nothing, fallback)
        @test calls[] == 1
        @test_throws "singular mass matrix" Tarang._etd_oversized_fallback!(
            state, solver, large_L, spzeros(4097, 4097), fallback)
        @test calls[] == 1
        @test get(state.timestepper_data, :L_eff, nothing) === nothing
        # A small operator still uses ETD rather than the implicit fallback.
        @test !Tarang._etd_oversized_fallback!(state, solver, spzeros(2, 2), nothing, fallback)
        @test calls[] == 1
    end
end
