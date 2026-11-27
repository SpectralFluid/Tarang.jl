"""
Test suite for DiagonalIMEX timestepper and SpectralLinearOperator.

Tests:
1. SpectralLinearOperator construction for Laplacian and hyperviscosity
2. SpectralLinearOperator coefficient values match expected -ν*k²
3. RK222 explicit fallback convergence (ODE: du/dt = -u, exact: exp(-t))
4. RK443 explicit fallback convergence
"""

using Test
using LinearAlgebra
using Tarang

import Tarang: SpectralLinearOperator

@testset "DiagonalIMEX and SpectralLinearOperator" begin

    @testset "SpectralLinearOperator construction — Laplacian" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        bases = (xb, yb)

        ν = 1e-3
        L = SpectralLinearOperator(dist, bases, :laplacian; ν=ν)

        @test L.operator_type == :laplacian

        # k=0 mode should have zero coefficient (no damping of mean)
        @test L.coefficients[1, 1] ≈ 0.0 atol=1e-14

        # Laplacian coefficients are ν*k² ≥ 0 (positive, used in IMEX as 1+dt*γ*L)
        @test all(real.(L.coefficients) .>= -1e-14)

        # Non-zero wavenumber should have positive coefficient
        @test L.coefficients[2, 1] > 0.0
    end

    @testset "SpectralLinearOperator default layout follows distributor dtype" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=ComplexF64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=10, bounds=(0.0, 2π))
        bases = (xb, yb)

        field = ScalarField(dist, "u", bases, ComplexF64)
        L = SpectralLinearOperator(dist, bases, :laplacian)

        @test size(L.coefficients) == size(Tarang.get_coeff_data(field)) == (8, 10)
    end

    @testset "SpectralLinearOperator construction — hyperviscosity" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        bases = (xb, yb)

        ν = 1e-4
        order = 2  # 4th-order hyperviscosity: ν*k⁴
        L = SpectralLinearOperator(dist, bases, :hyperviscosity; ν=ν, order=order)

        @test L.operator_type == :hyperviscosity

        # k=0 mode should have zero coefficient
        @test L.coefficients[1, 1] ≈ 0.0 atol=1e-14

        # Coefficients are ν*(k²)^order ≥ 0
        @test all(real.(L.coefficients) .>= -1e-14)

        # Higher wavenumbers should have larger coefficients (k⁴ grows fast)
        # Index i corresponds to wavenumber k=i-1, so index 4 (k=3) > index 2 (k=1)
        if size(L.coefficients, 1) > 4
            @test real(L.coefficients[4, 1]) > real(L.coefficients[2, 1])
        end
    end

    @testset "RK222 explicit convergence" begin
        # Solve du/dt = -u with u(0) = 1, exact solution: u(t) = exp(-t)
        # Test that halving dt improves error by ~4x (2nd order)
        T_final = 1.0

        function rk222_solve(dt)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            fill!(Tarang.get_grid_data(u), 1.0)  # u(0) = 1

            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) = -u")
            solver = InitialValueSolver(problem, RK222(); dt=dt)

            nsteps = round(Int, T_final / dt)
            for _ in 1:nsteps
                step!(solver)
            end

            ensure_layout!(u, :g)
            return Tarang.get_grid_data(u)[1]
        end

        # Two different timesteps
        dt1 = 0.05
        dt2 = 0.025
        u1 = rk222_solve(dt1)
        u2 = rk222_solve(dt2)
        exact = exp(-T_final)

        err1 = abs(u1 - exact)
        err2 = abs(u2 - exact)

        # 2nd order: error ratio should be ~4 when dt halved
        @test err2 > 0  # Sanity: error should not be zero or negative
        if err2 > 1e-14  # avoid division by zero near machine precision
            rate = log2(err1 / err2)
            @test rate > 1.5  # Should be ~2 for 2nd order
        end
    end

    @testset "RK222 explicit convergence with algebraic constraint" begin
        # A decoupled algebraic equation makes the mass matrix singular, but
        # it must not change the RK update order for the evolving variable.
        # Solve du/dt = -u with u(0) = 1, exact solution: u(t) = exp(-t).
        T_final = 1.0

        function rk222_constrained_solve(dt)
            domain = PeriodicDomain(4)
            u = ScalarField(domain, "u")
            lambda = ScalarField(domain, "lambda")
            set!(u, (x,) -> 1.0)

            problem = InitialValueProblem([u, lambda])
            add_equation!(problem, "∂t(u) = -u")
            add_equation!(problem, "lambda = 0")
            solver = InitialValueSolver(problem, RK222(); dt=dt)

            nsteps = round(Int, T_final / dt)
            for _ in 1:nsteps
                step!(solver)
            end

            ensure_layout!(u, :g)
            return Tarang.get_grid_data(u)[1]
        end

        dt1 = 0.05
        dt2 = 0.025
        u1 = rk222_constrained_solve(dt1)
        u2 = rk222_constrained_solve(dt2)
        exact = exp(-T_final)

        err1 = abs(u1 - exact)
        err2 = abs(u2 - exact)

        @test err2 > 0
        rate = log2(err1 / err2)
        @test rate > 1.5
    end

    @testset "RK111 explicit convergence" begin
        # 1st order Forward Euler: error ratio ~2 when dt halved
        T_final = 0.5

        function rk111_solve(dt)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            fill!(Tarang.get_grid_data(u), 1.0)

            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) = -u")
            solver = InitialValueSolver(problem, RK111(); dt=dt)

            nsteps = round(Int, T_final / dt)
            for _ in 1:nsteps
                step!(solver)
            end

            ensure_layout!(u, :g)
            return Tarang.get_grid_data(u)[1]
        end

        dt1 = 0.02
        dt2 = 0.01
        u1 = rk111_solve(dt1)
        u2 = rk111_solve(dt2)
        exact = exp(-T_final)

        err1 = abs(u1 - exact)
        err2 = abs(u2 - exact)

        @test err2 > 0  # Sanity: error should not be zero or negative
        if err2 > 1e-14
            rate = log2(err1 / err2)
            @test rate > 0.8  # Should be ~1 for 1st order
        end
    end

    @testset "Timestepper types exist" begin
        @test RK111() isa Tarang.TimeStepper
        @test RK222() isa Tarang.TimeStepper
        @test RK443() isa Tarang.TimeStepper
        @test Tarang.DiagonalIMEX_RK222() isa Tarang.TimeStepper
    end

    @testset "Multistep timestepper types exist" begin
        @test CNAB1() isa Tarang.TimeStepper
        @test CNAB2() isa Tarang.TimeStepper
        @test SBDF1() isa Tarang.TimeStepper
        @test SBDF2() isa Tarang.TimeStepper
        @test SBDF3() isa Tarang.TimeStepper
        @test SBDF4() isa Tarang.TimeStepper
    end

    @testset "DiagonalIMEX implicit stepping (attach SpectralLinearOperator)" begin
        # The explicit-convergence tests above never engage the implicit path:
        # step_diagonal_imex_*! only runs implicitly when a SpectralLinearOperator
        # is attached to the solver (otherwise it falls back to explicit RK). This
        # testset exercises the attach API and pins the implicit-stepping behavior.
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        Tarang.get_grid_data(u) .= cos.(2 .* collect(range(0, 2π, length=17))[1:16])

        L = SpectralLinearOperator(dist, (xb,), :laplacian; ν=0.5)
        @test L.operator_type == :laplacian

        # Operator attaches and is retrievable (this part works correctly).
        problem = InitialValueProblem([u])
        add_equation!(problem, "dt(u) = 0")
        solver = InitialValueSolver(problem, Tarang.DiagonalIMEX_RK222(); dt=0.005)
        Tarang.set_spectral_linear_operator!(solver, L)
        @test Tarang._get_spectral_linear_operator(solver) !== nothing

        # FIXED: the operator now sizes `coefficients` to the field's coefficient
        # array (rfft N÷2+1 = 9), not the basis grid size (16). spectral_operators.jl
        # derives coeff_shape from local_shape(Domain(dist,bases), :c), so it matches
        # the field layout and the implicit step no longer throws DimensionMismatch.
        ensure_layout!(u, :c)
        n_coeff = length(Tarang.get_coeff_data(u))
        @test length(L.coefficients) == n_coeff

        # The implicit diagonal-IMEX step integrates dt(u) = -ν k² u. With cos(2x),
        # ν=0.5 the decay rate is λ = ν k² = 2, so after T = 200·dt = 1.0 the
        # amplitude should be ≈ exp(-λ T) = exp(-2). This exercises the full
        # implicit path (step_diagonal_imex_rk222!) and pins the viscous rate.
        # (Made correct by the wavenumber alignment in spectral_operators.jl and
        # the :c-layout fix in step_diagonal_imex.jl.)
        ensure_layout!(u, :g)
        u0 = maximum(abs, Tarang.get_grid_data(u))
        for _ in 1:200
            step!(solver)
        end
        ensure_layout!(u, :g)
        uf = maximum(abs, Tarang.get_grid_data(u))
        @test uf < u0                                       # the mode decays
        @test isapprox(uf / u0, exp(-2.0); rtol=0.05)        # at the viscous rate λ = ν k² = 2
    end

    @testset "DiagonalIMEX implicit stepping — RK443 and SBDF2" begin
        # Same viscous-decay check for the other diagonal-IMEX steppers; all share
        # the implicit path fixed in step_diagonal_imex.jl / spectral_operators.jl.
        for ts in (Tarang.DiagonalIMEX_RK443(), Tarang.DiagonalIMEX_SBDF2())
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            Tarang.get_grid_data(u) .= cos.(2 .* collect(range(0, 2π, length=17))[1:16])
            L = SpectralLinearOperator(dist, (xb,), :laplacian; ν=0.5)
            problem = InitialValueProblem([u]); add_equation!(problem, "dt(u) = 0")
            solver = InitialValueSolver(problem, ts; dt=0.005)
            Tarang.set_spectral_linear_operator!(solver, L)
            for _ in 1:200
                step!(solver)
            end
            ensure_layout!(u, :g)
            uf = maximum(abs, Tarang.get_grid_data(u))
            @test isapprox(uf, exp(-2.0); rtol=0.05)   # decays at λ = ν k² = 2
        end
    end

    @testset "DiagonalIMEX L-stability in the stiff limit" begin
        # REGRESSION GUARD: the diagonal-IMEX RK steppers must include the
        # off-diagonal implicit tableau terms, otherwise R(z) → 1−1/γ (|R|>1 for
        # dt·λ ≳ 4.8) and high-wavenumber stiff modes BLOW UP instead of decaying.
        # The non-stiff tests above (z = dt·ν·k² ≈ 0.01) cannot catch this; this
        # one uses a deliberately stiff z = dt·ν·k² = 9.0. With the buggy
        # diagonal-only update the k=6 mode grows ~|−1.47|^20 ≈ 2000×; the correct
        # L-stable ESDIRK damps it to ≈ exp(−180) ≈ 0.
        for ts in (Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443())
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            xs = collect(range(0, 2π, length=17))[1:16]
            Tarang.get_grid_data(u) .= cos.(6 .* xs)        # stiff mode k=6, k²=36
            u0 = maximum(abs, Tarang.get_grid_data(u))

            L = SpectralLinearOperator(dist, (xb,), :laplacian; ν=1.0)
            problem = InitialValueProblem([u]); add_equation!(problem, "dt(u) = 0")
            dt = 0.25                                       # z = dt·ν·k² = 0.25·36 = 9 ≫ 4.8
            solver = InitialValueSolver(problem, ts; dt=dt)
            Tarang.set_spectral_linear_operator!(solver, L)
            for _ in 1:20                                   # T = 5.0
                step!(solver)
            end
            ensure_layout!(u, :g)
            uf = maximum(abs, Tarang.get_grid_data(u))
            @test isfinite(uf)
            @test uf < u0          # stays bounded (the buggy version grows)
            @test uf < 1e-2        # actually damped toward exp(−ν k² T) ≈ 0
        end
    end

    @testset "Attached operator: applied at the scheme's order, or refused" begin
        # dt(u) = 0 plus an attached ν·Laplacian (ν = 0.5): u0 = cos 2x decays as
        # exp(-2t)·cos 2x. Only RK222, RK443 and SBDF2 used to read the operator;
        # every other scheme returned u0 bit-unchanged with no error or warning.
        xs = collect(range(0, 2π, length=17))[1:16]
        function attached_setup(ts, dt)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            Tarang.get_grid_data(u) .= cos.(2 .* xs)
            problem = InitialValueProblem([u]); add_equation!(problem, "dt(u) = 0")
            solver = InitialValueSolver(problem, ts; dt=dt)
            return solver, u, SpectralLinearOperator(dist, (xb,), :laplacian; ν=0.5)
        end
        function attached_error(ts, dt; T=0.4)
            solver, u, L = attached_setup(ts, dt)
            Tarang.set_spectral_linear_operator!(solver, L)
            for _ in 1:round(Int, T / dt)
                step!(solver)
            end
            ensure_layout!(u, :g)
            return maximum(abs, Tarang.get_grid_data(u) .- exp(-2T) .* cos.(2 .* xs))
        end

        # Every IMEX RK tableau shares the per-mode stage solve; SBDF2 has its own.
        # RKSMR and RKGFY are second order in their implicit part. (Measured rates
        # 0.99, 2.00, 2.99, 2.00, 2.00, 2.02; an ignored operator errs by 0.55.)
        for (ts, order) in ((RK111(), 1), (RK222(), 2), (RK443(), 3),
                            (RKSMR(), 2), (Tarang.RKGFY(), 2), (SBDF2(), 2))
            coarse, fine = attached_error(ts, 0.02), attached_error(ts, 0.01)
            @test fine < 5e-3
            @test log2(coarse / fine) > order - 0.25
        end

        # The rest solve only the equations' own operator. Refused when attached,
        # naming the scheme and the ones that work; and, for an operator registered
        # as a problem parameter instead, at the first step, before anything moves.
        for ts in (CNAB1(), CNAB2(), SBDF1(), SBDF3(), SBDF4(), ETD_RK222(), ETD_CNAB2(),
                   ETD_SBDF2(), Tarang.MCNAB2(), Tarang.CNLF2())
            solver, u, L = attached_setup(ts, 0.01)
            err = try
                Tarang.set_spectral_linear_operator!(solver, L)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            message = err === nothing ? "" : sprint(showerror, err)
            @test occursin(string(nameof(typeof(ts))), message)
            @test occursin("RK111, RK222, RK443, RKGFY, RKSMR, SBDF2", message)
            @test Tarang._get_spectral_linear_operator(solver) === nothing

            solver.problem.parameters["spectral_linear_operator"] = L
            @test_throws ArgumentError step!(solver)
            ensure_layout!(u, :g)
            @test Tarang.get_grid_data(u) == cos.(2 .* xs)
            @test solver.iteration == 0
        end
    end

    @testset "Attached operator on a problem with per-mode subproblems is refused" begin
        # A Chebyshev axis gives RK222 per-mode subproblems, whose stage solve reads
        # the equations' own operator only: the attached one was silently dropped.
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
        b = ScalarField(Domain(dist, (xb, zb)), "b")
        set!(b, (x, z) -> cos(2x) * sin(π * z))
        tau1 = ScalarField(dist, "tau1", (xb,), Float64)
        tau2 = ScalarField(dist, "tau2", (xb,), Float64)
        lift_basis = derivative_basis(zb, 1)
        problem = InitialValueProblem([b, tau1, tau2])
        add_parameters!(problem; lift1=lift(tau1, lift_basis, -1), lift2=lift(tau2, lift_basis, -2))
        add_equation!(problem, "dt(b) - 0.01*lap(b) + lift1 + lift2 = 0")
        add_bc!(problem, "b(z=0) = 0")
        add_bc!(problem, "b(z=1) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=0.002)
        @test Tarang._timestepper_subproblems(solver) !== nothing
        L = SpectralLinearOperator(dist, (xb, zb), :laplacian; ν=5.0)
        err = try
            Tarang.set_spectral_linear_operator!(solver, L)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("non-Fourier axis", err === nothing ? "" : sprint(showerror, err))
    end
end
