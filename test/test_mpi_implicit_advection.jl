# Regression tests for the distributed diagonal-IMEX operator parser.
#
# Bug (2026-05-28): pure-Fourier MPI silently FROZE any field whose implicit linear
# operator contained a non-Laplacian term (e.g. first-derivative advection), because
# _diagonal_Lhat_from_expr returned `nothing` → the field was skipped every step.
#
# Unit/safety-net tests run SERIALLY (they exercise the parser directly, no MPI
# launcher needed). The end-to-end "field is not frozen" check is gated on nprocs>1.
using Test
using Tarang
import Tarang: _diagonal_Lhat_from_expr, d, wavenumbers_rfft, wavenumbers_fft

import MPI
MPI.Initialized() || MPI.Init()
const NP = MPI.Comm_size(MPI.COMM_WORLD)
const RANK = MPI.Comm_rank(MPI.COMM_WORLD)

function _mk_field(; np=1)
    coords = CartesianCoordinates("x", "y")
    dist = np > 1 ? Distributor(coords; mesh=(np,), dtype=Float64, architecture=CPU()) :
                    Distributor(coords; dtype=Float64, architecture=CPU())
    N = 8
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    q = ScalarField(domain, "q"); ensure_layout!(q, :c)
    return coords, dist, xb, yb, domain, q, N
end

if NP == 1
    @testset "diagonal L̂ parser: derivative multipliers (i·k)" begin
        coords, dist, xb, yb, domain, q, N = _mk_field()
        cd = get_coeff_data(q)

        # ∂x(q): first RealFourier axis is rfft layout, k = [0,1,…,N/2]. Multiplier i*k_x.
        Lhat = _diagonal_Lhat_from_expr(d(q, coords["x"]), q)
        @test Lhat !== nothing
        kx = wavenumbers_rfft(xb)
        expected = ComplexF64[im * kx[I[1]] for I in CartesianIndices(size(cd))]
        # On a real even grid the two Nyquist images have cancelling odd derivatives.
        expected[N ÷ 2 + 1, :] .= 0
        @test maximum(abs.(ComplexF64.(Lhat) .- expected)) < 1e-10

        # ∂y(q): second axis is full-FFT layout. Multiplier i*k_y on axis 2.
        Lhat_y = _diagonal_Lhat_from_expr(d(q, coords["y"]), q)
        @test Lhat_y !== nothing
        ky = wavenumbers_fft(yb)
        expected_y = ComplexF64[im * ky[I[2]] for I in CartesianIndices(size(cd))]
        expected_y[:, N ÷ 2 + 1] .= 0
        @test maximum(abs.(ComplexF64.(Lhat_y) .- expected_y)) < 1e-10
    end

    @testset "diagonal L̂ parser: real equation mixes diffusion + advection" begin
        coords, dist, xb, yb, domain, q, N = _mk_field()
        problem = InitialValueProblem([q])
        add_parameters!(problem, nu=0.05)
        add_equation!(problem, "dt(q) - nu*lap(q) - 0.3*d(q,x) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)  # populates equation_data
        Lexpr = get(solver.problem.equation_data[1], "L", nothing)
        @test Lexpr !== nothing
        Lhat = _diagonal_Lhat_from_expr(Lexpr, q)
        @test Lhat !== nothing                         # was `nothing` (freeze) before fix
        ch = ComplexF64.(Lhat)
        @test any(!iszero, real.(ch))                  # diffusion (−k²) present
        @test any(!iszero, imag.(ch))                  # advection (i·k_x) present
    end

    @testset "safety net: derivative of a DIFFERENT field is not diagonal" begin
        coords, dist, xb, yb, domain, q, N = _mk_field()
        p = ScalarField(domain, "p"); ensure_layout!(p, :c)
        # ∂x(p) inside q's implicit operator is a coupling term, NOT a scalar diagonal L̂.
        @test _diagonal_Lhat_from_expr(d(p, coords["x"]), q) === nothing
    end
else
    @testset "A singular zero mode rejects the Fourier step on all ranks" begin
        for (ts, h, first_h) in ((RK111(), 1.0, 0.06), (RK443(), 2.0, 0.06),
                                 (SBDF2(), 1.0, 0.0), (SBDF2(), 1.5, 1.5))
            function zero_mode_growth(name)
                _, _, _, _, domain, _, _ = _mk_field(np=NP)
                u = ScalarField(domain, name)
                fill!(parent(grid_data!(u)), 1)
                p = InitialValueProblem([u])
                add_equation!(p, "dt($name) + lap($name) - $name = 0")
                return InitialValueSolver(p, ts; dt=0.06), u
            end
            solver, u = zero_mode_growth("u")
            control, v = zero_mode_growth("v")
            if first_h > 0
                step!(solver, first_h)
                step!(control, first_h)
            end
            before = copy(parent(grid_data!(u)))
            @test_throws Exception step!(solver, h)
            @test (solver.iteration, solver.sim_time, solver.dt) ==
                  (control.iteration, control.sim_time, control.dt)
            @test parent(grid_data!(u)) ≈ before atol=1e-12
            step!(solver, 0.025)
            step!(control, 0.025)
            @test parent(grid_data!(u)) ≈ parent(grid_data!(v)) atol=1e-12
        end
    end

    @testset "MPI vector diffusion and complex phase" begin
        coords, dist, _, _, domain, _, N = _mk_field(np=NP)
        u = VectorField(domain, "u")
        gd = grid_data!(first(u.components))
        ax = Tarang.PencilArrays.pencil(gd).axes_local
        xs = (2π/N) .* (ax[1] .- 1)
        ys = (2π/N) .* (ax[2] .- 1)
        initial = ([cos(2x) for x in xs, y in ys], [sin(y) for x in xs, y in ys])
        for (component, values) in zip(u.components, initial)
            parent(grid_data!(component)) .= values
        end
        problem = InitialValueProblem([u])
        add_equation!(problem, "dt(u) - 0.5*lap(u) = 0")
        solver = InitialValueSolver(problem, RK111(); dt=0.1)
        step!(solver)
        for (component, values, rate) in zip(u.components, initial, (2.0, 0.5))
            @test parent(grid_data!(component)) ≈ values ./ (1 + 0.1rate) atol=1e-12
        end

        cdist = Distributor(coords; mesh=(NP,), dtype=ComplexF64)
        bases = map(c -> ComplexFourier(coords[c]; size=N, bounds=(0.0, 2π)), ("x", "y"))
        q = ScalarField(Domain(cdist, bases), "q")
        fill!(parent(grid_data!(q)), 1)
        oscillator = InitialValueProblem([q])
        add_parameters!(oscillator; omega=1im)
        add_equation!(oscillator, "dt(q) + omega*q = 0")
        step!(InitialValueSolver(oscillator, RK111(); dt=0.1))
        @test maximum(abs, parent(grid_data!(q)) .- inv(1 + 0.1im)) < 1e-12
    end

    @testset "MPI: implicit advection no longer freezes the field (rank=$RANK)" begin
        coords, dist, xb, yb, domain, q, N = _mk_field(np=NP)
        problem = InitialValueProblem([q])
        add_parameters!(problem, nu=0.05)
        add_equation!(problem, "dt(q) - nu*lap(q) - 0.3*d(q,x) - 0.2*d(q,y) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=2e-3)
        ensure_layout!(q, :g)
        dat = get_grid_data(q); raw = dat isa Tarang.PencilArrays.PencilArray ? parent(dat) : dat
        ax = dat isa Tarang.PencilArrays.PencilArray ? Tarang.PencilArrays.pencil(dat).axes_local : (1:N, 1:N)
        xs = [2π*(i-1)/N for i in ax[1]]; ys = [2π*(j-1)/N for j in ax[2]]
        raw .= @. sin(2xs - ys') + 0.5cos(xs + 3ys')
        ensure_layout!(q, :c)
        l2() = (ensure_layout!(q,:g); r=get_grid_data(q); rr=r isa Tarang.PencilArrays.PencilArray ? parent(r) : r;
                MPI.Allreduce(sum(abs2, rr), MPI.SUM, MPI.COMM_WORLD))

        n0 = l2()
        cd = get_coeff_data(q); snap = copy(cd isa Tarang.PencilArrays.PencilArray ? parent(cd) : cd)
        step!(solver, 2e-3)
        ensure_layout!(q, :c)
        cd2 = get_coeff_data(q); cur = cd2 isa Tarang.PencilArrays.PencilArray ? parent(cd2) : cd2
        dmax = MPI.Allreduce(maximum(abs.(cur .- snap)), MPI.MAX, MPI.COMM_WORLD)
        @test dmax > 1e-6                              # field actually evolves (was frozen → 0)
        for _ in 1:9; step!(solver, 2e-3); end
        @test l2() < n0 - 1e-6                         # diffusion decays L2 (frozen → unchanged)
    end
end
