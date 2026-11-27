"""
Every timestepper on the real single-GPU dispatch path, WITHOUT a GPU.

`test_gpu_timesteppers.jl` is the value-asserted, CUDA-gated GPU stepper suite.
This file drives the same
`step!` dispatch on `JLArray` device fields (GPUArrays' CPU-backed reference
device array): `plan_is_gpu`, `_field_uses_gpu` and
`_distributed_field_path_required` are all true, `allowscalar(false)` makes any
scalar indexing throw, and every stepper takes exactly the branch it takes on a
`CuArray` — the field-native explicit RK, the matrix-free multistep field path,
the serial diagonal-IMEX broadcasts, and the loud implicit-operator refusals.

What JLArray lacks is a device FFT. The transform backend below stands in for
cuFFT with a CPU twin field: it copies the device buffer to a host twin of the
same domain, runs Tarang's own CPU transform, and copies the result back. The
transform is therefore bit-identical to the CPU chain, which is what makes the
CPU-vs-device comparisons below exact rather than approximate.

Found by this harness before it was a test: MCNAB2 and CNLF2 silently ran as
first-order CNAB1 on every GPU/MPI explicit problem (their startup gate read
`length(state.history) < 2`, but the field path keeps a one-entry history).

Uniquely-prefixed names (gtj_*) — the full suite shares the Main namespace.
"""

using Test
using Tarang
using Printf
using Random
using LinearAlgebra

const _GTJ_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping JLArray timestepper test" err
    false
end

if _GTJ_OK
    const _GTJ = JLArrays.JLArray
    const _GTJ_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package. The same
    # five methods as the other JLArray test files, plus the identity/upload
    # methods the lazy RHS and dealiasing paths call on device data.
    Tarang.is_gpu_array(::_GTJ) = true
    Tarang.architecture(::_GTJ) = _GTJ_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _GTJ(a)
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::_GTJ) = a
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::AbstractArray) = _GTJ(Array(a))
    Tarang.copy_to_device(a::AbstractArray, ::_GTJ) = _GTJ(Array(a))
    Tarang.copy_to_device(a::_GTJ, ::_GTJ) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _GTJ
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _GTJ{T}

    # Padded products execute FFT plans directly, bypassing the field transform
    # hooks below. Give them the same explicit host stand-in: generic FFTW
    # dispatch on JLArray can impose host alignment requirements on separately
    # allocated device-proxy buffers (and some JLArrays versions reject it).
    struct GTJHostFFTPlan{P}
        host_plan::P
    end
    function LinearAlgebra.mul!(dst::_GTJ, plan::GTJHostFFTPlan, src::_GTJ)
        copyto!(dst, plan.host_plan * Array(src))
        return dst
    end
    function Tarang._real_padded_fft_plans(::Tarang.GPU{JLArrays.JLBackend},
                                           a, original, padhalf, half, dims)
        pf, pb, sf, sb, ps, s = Tarang._real_padded_fft_plans(
            CPU(), Array(a), Array(original), padhalf, half, dims)
        return GTJHostFFTPlan(pf), GTJHostFFTPlan(pb), GTJHostFFTPlan(sf),
               GTJHostFFTPlan(sb), _GTJ(ps), _GTJ(s)
    end

    # ---- cuFFT stand-in: a CPU twin field transformed by Tarang's CPU chain ----
    const _GTJ_TWINS = Dict{Any, Any}()
    function gtj_twin(field)
        get!(_GTJ_TWINS, (objectid(field.bases), field.dtype)) do
            cdist = Distributor(field.dist.coordsys; dtype=field.dtype, device=CPU())
            ScalarField(Domain(cdist, field.bases), "gtj_twin_" * field.name)
        end
    end
    function gtj_sync_scales!(twin, field)
        twin.scales == field.scales && return
        twin.current_layout = :c
        Tarang.preset_scales!(twin, field.scales)
    end
    # Copy `src` into the buffer selected by `getter`/`setter` on `dst`,
    # reallocating (via `make`) only when the shape or eltype differ.
    function gtj_copy_into!(getter, setter, make, dst, src)
        buf = getter(dst)
        if buf === nothing || size(buf) != size(src) || eltype(buf) != eltype(src)
            setter(dst, make(src))
        else
            copyto!(buf, src)
        end
    end
    function Tarang._gpu_forward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend},
                                                    field::Tarang.ScalarField)
        twin = gtj_twin(field)
        gtj_sync_scales!(twin, field)
        gtj_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, copy, twin,
                       Array(Tarang.get_grid_data(field)))
        twin.current_layout = :g
        Tarang.forward_transform!(twin)
        gtj_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, x -> _GTJ(copy(x)),
                       field, Tarang.get_coeff_data(twin))
        return true
    end
    function Tarang._gpu_backward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend}, field)
        twin = gtj_twin(field)
        gtj_sync_scales!(twin, field)
        gtj_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, copy, twin,
                       Array(Tarang.get_coeff_data(field)))
        twin.current_layout = :c
        Tarang.backward_transform!(twin)
        gtj_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, x -> _GTJ(copy(x)),
                       field, Tarang.get_grid_data(twin))
        return true
    end
end

@testset "GPU timestepper dispatch on JLArray device fields" begin
    if !_GTJ_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)

        gtj_grid(u) = (ensure_layout!(u, :g); Array(Tarang.get_grid_data(u)))

        function gtj_solver(arch, eqn, ts, dt, u0; N=16)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64, device=arch)
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
            u = ScalarField(Domain(dist, (xb,)), "u")
            xs = collect(range(0, 2π, length=N + 1))[1:N]
            ensure_layout!(u, :g)
            copyto!(Tarang.get_grid_data(u), u0.(xs))
            prob = InitialValueProblem([u])
            add_equation!(prob, eqn)
            return InitialValueSolver(prob, ts; dt), u, xs
        end

        function gtj_run(arch, eqn, ts, dts, u0)
            s, u, xs = gtj_solver(arch, eqn, ts, first(dts), u0)
            for d in dts
                step!(s, d)
            end
            return gtj_grid(u), xs
        end

        all_steppers = (RK111(), RK222(), RK443(), RKSMR(), Tarang.RKGFY(),
                        CNAB1(), CNAB2(), SBDF1(), SBDF2(), SBDF3(), SBDF4(),
                        ETD_RK222(), ETD_CNAB2(), ETD_SBDF2(), Tarang.MCNAB2(), Tarang.CNLF2(),
                        Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443(), Tarang.DiagonalIMEX_SBDF2())
        diagonal = (Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443(), Tarang.DiagonalIMEX_SBDF2())
        # Every IMEX RK tableau and SBDF2 solve a diagonal operator per mode.
        serial_diagonal = (RK111(), RK222(), RK443(), RKSMR(), Tarang.RKGFY(), SBDF2())
        implicit_supported = (serial_diagonal..., diagonal...)
        gtj_name(ts) = nameof(typeof(ts))
        gtj_supported(ts) = any(p -> typeof(p) === typeof(ts), implicit_supported)
        # RK111 is backward Euler: first order, ~1e-3 at these step sizes.
        gtj_tol(ts) = ts isa RK111 ? 2e-3 : 1e-4

        @testset "RK equilibrium is independent of time units" begin
            for arch in (CPU(), _GTJ_ARCH), ts in (RK222(), RK443()), dt in (0.1, 1e-15)
                rate = 0.1 / dt
                values, _ = gtj_run(arch, "dt(u) + $rate*u = $rate*u", ts,
                                    fill(dt, 3), x -> 1.0)
                @test values ≈ ones(16) atol=2e-12
            end
        end

        @testset "Reference steps on CPU and device" begin
            for arch in (CPU(), _GTJ_ARCH)
                for ts in (RK222(), Tarang.DiagonalIMEX_RK222())
                    values, _ = gtj_run(arch, "dt(u) = -u", ts, [0.1], x -> 1.0)
                    @test values ≈ fill(0.905, 16) atol=1e-13
                end
                for ts in (SBDF3(), SBDF4())
                    s, u, _ = gtj_solver(arch, "dt(u) = -u", ts, 0.1, x -> 1.0)
                    step!(s)
                    @test gtj_grid(u) ≈ fill(0.9, 16) atol=1e-13
                    step!(s)
                    @test gtj_grid(u) ≈ fill(0.8133333333333334, 16) atol=1e-13
                end
            end
        end

        # Which schemes are the same arithmetic on both architectures for an
        # explicit problem. ETD_* legitimately substitute (with a warning) an RK/
        # multistep step when there is no linear operator, so they are checked for
        # accuracy but not bit parity.
        parity_exact = (RK111(), RK222(), RK443(), RKSMR(), Tarang.RKGFY(),
                        CNAB1(), CNAB2(), SBDF1(), SBDF2(), SBDF3(), SBDF4(),
                        Tarang.MCNAB2(), Tarang.CNLF2(), diagonal...)

        @testset "explicit nonlinear problem: device == CPU, and startup-limited order" begin
            # dt(u) = -u², u0 ≡ 1 → 1/(1+t). Spatially exact, so the error is the
            # time-discretization error alone.
            expected_order = Dict(:RK111 => 1, :CNAB1 => 1, :SBDF1 => 1,
                                  :RK443 => 3, :RKSMR => 3,
                                  :SBDF3 => 2, :SBDF4 => 2)
            for ts in all_steppers
                T = 0.4
                errs = map((0.02, 0.01)) do dt
                    vals, _ = gtj_run(_GTJ_ARCH, "dt(u) = -u*u", ts, fill(dt, round(Int, T / dt)), x -> 1.0)
                    maximum(abs, vals .- 1 / (1 + T))
                end
                order = get(expected_order, gtj_name(ts), 2)
                rate = log2(errs[1] / errs[2])
                @test errs[2] < 5e-3
                # MCNAB2/CNLF2 used to fall to CNAB1 (rate 1) on every device step.
                @test rate > order - 0.5

                if any(p -> typeof(p) === typeof(ts), parity_exact)
                    dts = fill(0.01, 20)
                    dev, _ = gtj_run(_GTJ_ARCH, "dt(u) = -u*u", ts, dts, x -> 1 + 0.1cos(x))
                    cpu, _ = gtj_run(CPU(), "dt(u) = -u*u", ts, dts, x -> 1 + 0.1cos(x))
                    @test maximum(abs, dev .- cpu) < 1e-12
                end
            end
        end

        @testset "implicit operator: automatic diagonal dispatch" begin
            # dt(u) - 0.5 lap(u) = 0, u0 = cos 2x → e^{-2t} cos 2x.
            T = 0.2; dt = 0.005
            for ts in all_steppers
                if gtj_supported(ts)
                    dev, xs = gtj_run(_GTJ_ARCH, "dt(u) - 0.5*lap(u) = 0", ts, fill(dt, round(Int, T / dt)), x -> cos(2x))
                    cpu, _ = gtj_run(CPU(), "dt(u) - 0.5*lap(u) = 0", ts, fill(dt, round(Int, T / dt)), x -> cos(2x))
                    @test maximum(abs, dev .- exp(-2T) .* cos.(2 .* xs)) < gtj_tol(ts)
                    @test dev ≈ cpu atol=1e-12 rtol=1e-12
                else
                    s, _, _ = gtj_solver(_GTJ_ARCH, "dt(u) - 0.5*lap(u) = 0", ts, dt, x -> cos(2x))
                    @test_throws ErrorException step!(s)
                end
            end
        end

        @testset "Attached operators use the internal diagonal path" begin
            # RK111, RKSMR and RKGFY used to ignore an attached operator (u came back
            # bit-unchanged) on CPU and device alike.
            attached_cpu = Dict{Symbol, Vector{Float64}}()
            for arch in (CPU(), _GTJ_ARCH), ts in serial_diagonal
                s, u, xs = gtj_solver(arch, "dt(u) = 0", ts, 0.005, x -> cos(2x))
                L = SpectralLinearOperator(u.dist, u.bases, :laplacian; ν=0.5)
                set_spectral_linear_operator!(s, L)
                for _ in 1:40
                    step!(s)
                end
                @test typeof(s.timestepper_state.timestepper) === typeof(ts)
                @test haskey(s.timestepper_state.timestepper_data, :sdi_Lmap)
                actual = gtj_grid(u)
                @test maximum(abs, actual .- exp(-0.4) .* cos.(2 .* xs)) < gtj_tol(ts)
                if arch isa CPU
                    attached_cpu[gtj_name(ts)] = actual
                else
                    @test actual ≈ attached_cpu[gtj_name(ts)] atol=1e-12 rtol=1e-12
                end
                if ts isa SBDF2
                    # Independent scalar recurrence for u' = -2u, including
                    # the backward-Euler startup. Check the discrete solution
                    # separately from its pointwise truncation error above.
                    h = 2 * 0.005
                    previous, current = 1.0, 1 / (1 + h)
                    for _ in 2:40
                        previous, current = current, (4current - previous) / (3 + 2h)
                    end
                    @test maximum(abs, actual .- current .* cos.(2 .* xs)) < 1e-12
                end
            end

            # The other schemes solve only the equations' own operator. Refused on the
            # device as on the CPU — when attached, and at the first step when the
            # operator is registered as a problem parameter — before any state moves.
            for ts in all_steppers
                gtj_supported(ts) && continue
                s, u, xs = gtj_solver(_GTJ_ARCH, "dt(u) = 0", ts, 0.005, x -> cos(2x))
                L = SpectralLinearOperator(u.dist, u.bases, :laplacian; ν=0.5)
                @test_throws ArgumentError set_spectral_linear_operator!(s, L)
                s.problem.parameters["spectral_linear_operator"] = L
                @test_throws ArgumentError step!(s)
                @test gtj_grid(u) == cos.(2 .* xs)
                @test s.iteration == 0
            end
        end

        @testset "Attached operator with a gauge's 0-D tau variable" begin
            # ∂t(ζ) = 0 with ν·Laplacian attached (ν = 0.5), Δψ + τ - ζ = 0, ∫ψ = 0:
            # ζ = e^{-2t}cos 2x and ψ = -ζ/4. The operator used to be broadcast against
            # the tau variable's scalar storage too (DimensionMismatch), so no problem
            # with a gauge could attach one.
            for arch in (CPU(), _GTJ_ARCH), ts in (RK222(), SBDF2())
                coords = CartesianCoordinates("x")
                dist = Distributor(coords; dtype=Float64, device=arch)
                domain = Domain(dist, (RealFourier(coords["x"]; size=16, bounds=(0.0, 2π)),))
                ζ = ScalarField(domain, "ζ"); ψ = ScalarField(domain, "ψ")
                problem = InitialValueProblem([ζ, ψ, ScalarField(dist, "tau_ψ", (), Float64)])
                add_equation!(problem, "∂t(ζ) = 0")
                add_equation!(problem, "Δ(ψ) + tau_ψ - ζ = 0")
                add_bc!(problem, "integ(ψ) = 0")
                s = InitialValueSolver(problem, ts; dt=0.005)
                xs = collect(range(0, 2π, length=17))[1:16]
                copyto!(grid_data!(ζ), cos.(2 .* xs))
                set_spectral_linear_operator!(s, SpectralLinearOperator(ζ.dist, ζ.bases, :laplacian; ν=0.5))
                for _ in 1:40
                    step!(s)
                end
                @test maximum(abs, gtj_grid(ζ) .- exp(-0.4) .* cos.(2 .* xs)) < 1e-4
                @test maximum(abs, gtj_grid(ψ) .+ gtj_grid(ζ) ./ 4) < 1e-12
            end
        end

        @testset "GPU public schemes reject cross-field implicit coupling" begin
            for ts in serial_diagonal
                _, u, _ = gtj_solver(_GTJ_ARCH, "dt(u) = 0", ts, 0.01, x -> 1.0)
                v = ScalarField(Domain(u.dist, u.bases), "v")
                copyto!(grid_data!(v), ones(16))
                problem = InitialValueProblem([u, v])
                add_equation!(problem, "dt(u) + v = 0")
                add_equation!(problem, "dt(v) = 0")
                s = InitialValueSolver(problem, ts; dt=0.01)
                @test_throws ArgumentError step!(s)
                @test gtj_grid(u) == ones(16)
                @test s.sim_time == 0
                @test s.iteration == 0
            end
        end

        @testset "SBDF2 can switch from a matrix solve to an attached operator" begin
            s, u, xs = gtj_solver(CPU(), "dt(u) = 0", SBDF2(), 0.005, x -> cos(2x))
            step!(s)
            L = SpectralLinearOperator(u.dist, u.bases, :laplacian; ν=0.5)
            set_spectral_linear_operator!(s, L)
            step!(s)
            @test gtj_grid(u) ≈ cos.(2 .* xs) ./ 1.01 atol=1e-12
            @test s.iteration == 2
        end

        @testset "DiagonalIMEX: derivative-of-self implicit term on device" begin
            # dt(u) + 0.3 d(u,x) - 0.1 lap(u) = 0 → e^{-0.4t} cos(2(x - 0.3t)).
            # Exercises the device upload of the (ik)^n multiplier.
            T = 0.2; dt = 0.005
            for ts in implicit_supported
                dev, xs = gtj_run(_GTJ_ARCH, "dt(u) + 0.3*d(u,x) - 0.1*lap(u) = 0", ts, fill(dt, round(Int, T / dt)), x -> cos(2x))
                @test maximum(abs, dev .- exp(-0.4T) .* cos.(2 .* (xs .- 0.3T))) < gtj_tol(ts)
            end
        end

        @testset "nonidentity mass is refused before device state advances" begin
            for ts in all_steppers
                s, u, _ = gtj_solver(_GTJ_ARCH, "2*dt(u) = -u", ts, 0.01, x -> 1.0)
                @test Tarang.compiled_problem(s.problem).mass_matrix === nothing
                before = gtj_grid(u)
                err = try
                    step!(s)
                    nothing
                catch ex
                    ex
                end
                @test err isa ArgumentError
                @test occursin("non-identity mass", err === nothing ? "" : sprint(showerror, err))
                @test gtj_grid(u) == before
                @test s.sim_time == 0
                @test s.iteration == 0
            end
            for ts in diagonal
                s, u, _ = gtj_solver(_GTJ_ARCH, "2*dt(u) + u = 0", ts, 0.01, x -> 1.0)
                @test_throws ArgumentError step!(s)
                @test gtj_grid(u) == ones(16)
            end
        end

        @testset "variable dt: device == CPU" begin
            dts = repeat([0.01, 0.02], 10)
            for (ts, eqn) in ((CNAB2(), "dt(u) = -u*u"), (SBDF2(), "dt(u) = -u*u"),
                              (SBDF3(), "dt(u) = -u*u"), (SBDF4(), "dt(u) = -u*u"),
                              (Tarang.MCNAB2(), "dt(u) = -u*u"), (Tarang.CNLF2(), "dt(u) = -u*u"),
                              (SBDF2(), "dt(u) - 0.1*lap(u) = -u*u"),
                              (RK443(), "dt(u) - 0.1*lap(u) = -u*u"),
                              (Tarang.DiagonalIMEX_SBDF2(), "dt(u) - 0.1*lap(u) = -u*u"),
                              (Tarang.DiagonalIMEX_RK443(), "dt(u) - 0.1*lap(u) = -u*u"))
                dev, _ = gtj_run(_GTJ_ARCH, eqn, ts, dts, x -> 1 + 0.5cos(x))
                cpu, _ = gtj_run(CPU(), eqn, ts, dts, x -> 1 + 0.5cos(x))
                @test maximum(abs, dev .- cpu) < 1e-12
            end
        end

        @testset "registered forcings on device: stochastic == CPU, deterministic exact" begin
            # Stochastic: same seed ⇒ the counter-based phase kernel gives the same
            # realization on device and host, and the field paths agree to roundoff.
            function gtj_forced(arch, forcing_ctor, ts; N=16, dt=0.01, nst=20)
                coords = CartesianCoordinates("x")
                dist = Distributor(coords; dtype=Float64, device=arch)
                xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
                u = ScalarField(Domain(dist, (xb,)), "u")
                ensure_layout!(u, :g); fill!(Tarang.get_grid_data(u), 0.0)
                prob = InitialValueProblem([u]); add_equation!(prob, "dt(u) = 0")
                add_stochastic_forcing!(prob, :u, forcing_ctor(arch))
                s = InitialValueSolver(prob, ts; dt)
                for _ in 1:nst; step!(s); end
                return gtj_grid(u), s.sim_time
            end
            stoch(arch) = StochasticForcing(field_size=(16,), domain_size=(2π,), energy_injection_rate=0.1,
                                            k_forcing=4.0, dk_forcing=1.0, dt=0.01,
                                            rng=Random.MersenneTwister(3), architecture=arch)
            dev, _ = gtj_forced(_GTJ_ARCH, stoch, RK222())
            cpu, _ = gtj_forced(CPU(), stoch, RK222())
            @test maximum(abs, dev .- cpu) < 1e-12
            @test maximum(abs, dev) > 0

            # Deterministic: dt(u) = 0 with F = cos x integrates exactly to t·cos x; the
            # device path transforms the physical-space realization through the
            # device field (set_local_data! on device arrays + the device transform).
            det(arch) = DeterministicForcing((x, t, p) -> cos.(x), (16,); architecture=arch)
            dev, T = gtj_forced(_GTJ_ARCH, det, RK222())
            xs = collect(range(0, 2π, length=17))[1:16]
            @test maximum(abs, dev .- T .* cos.(xs)) < 1e-12
        end

        @testset "algebraic constraints are solved exactly per mode (CPU and device)" begin
            # ∂t(ζ) = -0.3φ (+ ν Δζ), ∂t(θ) = 0 and one constraint. Every form is
            # diagonal per mode, φ̂ = α(k²)ζ̂ + β(k²)θ̂. With ζ₀ = S = sin2x·cos3y
            # (k² = 13) and θ = 0.7C, C = cos(x+y) (k² = 2), the solution is closed
            # form: with γ(k²) = α + νk²/0.3 and r = β(2)/γ(2),
            #   ζ = a·S + b·C,  a = e^{-0.3γ(13)t},  b = 0.7r(e^{-0.3γ(2)t} - 1),
            #   φ = α(13)·a·S + (α(2)·b + 0.7β(2))·C,
            # so what is left is the time stepper's own error (≲ 2e-8 here). On a
            # device the old pattern-matching refresh was the only constraint solver
            # and dropped θ, ignored coefficients, never updated the Helmholtz form;
            # on the CPU it overwrote the stage solve's φ with that value right before
            # F was evaluated (ζ errors 3e-4 to 3e-3).
            nx, ny, nsteps, dt = 16, 12, 3, 1e-2
            T = nsteps * dt
            xs = collect(range(0, 2π, length=nx + 1))[1:nx]
            ys = collect(range(0, 2π, length=ny + 1))[1:ny]
            S = [sin(2x) * cos(3y) for x in xs, y in ys]
            C = [cos(x + y) for x in xs, y in ys]
            volume = (2π)^2

            function gtj_constraint_run(arch, constraints; tau=true, extra=(), ts=RK222(),
                                        evolution="∂t(ζ) = -0.3*φ", bc="integ(φ) = 0",
                                        init=Dict{String, Matrix{Float64}}())
                coords = CartesianCoordinates("x", "y")
                dist = Distributor(coords; dtype=Float64, device=arch)
                domain = Domain(dist, (RealFourier(coords["x"]; size=nx, bounds=(0.0, 2π)),
                                       RealFourier(coords["y"]; size=ny, bounds=(0.0, 2π))))
                fields = Dict(name => ScalarField(domain, name) for name in ("ζ", "θ", "φ", extra...))
                variables = ScalarField[fields["ζ"], fields["θ"], fields["φ"],
                                        (fields[name] for name in extra)...]
                tau && push!(variables, ScalarField(dist, "tau_φ", (), Float64))
                problem = InitialValueProblem(variables)
                add_equation!(problem, evolution)
                add_equation!(problem, "∂t(θ) = 0")
                foreach(c -> add_equation!(problem, c), constraints)
                tau && add_bc!(problem, bc)
                solver = InitialValueSolver(problem, ts; dt)
                copyto!(grid_data!(fields["ζ"]), S)
                copyto!(grid_data!(fields["θ"]), 0.7 .* C)
                for (name, values) in init
                    copyto!(grid_data!(fields[name]), values)
                end
                for _ in 1:nsteps
                    step!(solver)
                end
                return Dict(name => gtj_grid(field) for (name, field) in fields)
            end

            function gtj_constraint_exact(coeffs; ν=0.0)
                α13, _ = coeffs(13.0)
                α2, β2 = coeffs(2.0)
                γ13, γ2 = α13 + ν * 13 / 0.3, α2 + ν * 2 / 0.3
                a = exp(-0.3γ13 * T)
                b = 0.7 * (β2 / γ2) * (exp(-0.3γ2 * T) - 1)
                return a .* S .+ b .* C, α13 * a .* S .+ (α2 * b + 0.7β2) .* C
            end

            # (constraint, has tau + gauge, k² -> (α, β), CPU can run it)
            forms = (("Δ(φ) + tau_φ - ζ = 0",            true,  k2 -> (-1 / k2, 0.0),         true),
                     ("Δ(φ) + tau_φ - ζ - θ = 0",        true,  k2 -> (-1 / k2, -1 / k2),     true),
                     ("2*Δ(φ) + tau_φ - ζ = 0",          true,  k2 -> (-1 / 2k2, 0.0),        true),
                     ("Δ(φ) + tau_φ - 2*ζ = 0",          true,  k2 -> (-2 / k2, 0.0),         true),
                     ("φ - 0.5*Δ(φ) - ζ = 0",            false, k2 -> (1 / (1 + k2 / 2), 0.0), true),
                     ("φ - ζ - θ = 0",                   false, k2 -> (1.0, 1.0),             true),
                     ("fraclap(φ, 0.5) + tau_φ - ζ - θ = 0", true, k2 -> (1 / √k2, 1 / √k2), true),
                     ("Δ(φ) + tau_φ - Δ(ζ) = 0",         true,  k2 -> (1.0, 0.0),             true),
                     # A variable on the RHS of a constraint: the CPU global path
                     # refuses it, the refresh solves it.
                     ("Δ(φ) + tau_φ = ζ + θ",            true,  k2 -> (-1 / k2, -1 / k2),     false),
                     # No tau, no gauge: the k = 0 mode takes the zero-mean
                     # convention (the CPU matrix is singular there).
                     ("Δ(φ) - ζ = 0",                    false, k2 -> (-1 / k2, 0.0),         false))
            for (constraint, tau, coeffs, on_cpu) in forms
                ζex, φex = gtj_constraint_exact(coeffs)
                dev = gtj_constraint_run(_GTJ_ARCH, [constraint]; tau)
                @test maximum(abs, dev["φ"] .- φex) < 1e-7
                @test maximum(abs, dev["ζ"] .- ζex) < 1e-7
                if on_cpu
                    cpu = gtj_constraint_run(CPU(), [constraint]; tau)
                    @test maximum(abs, cpu["φ"] .- φex) < 1e-7
                    @test maximum(abs, cpu["ζ"] .- ζex) < 1e-7
                    @test maximum(abs, dev["φ"] .- cpu["φ"]) < 1e-12
                    @test maximum(abs, dev["ζ"] .- cpu["ζ"]) < 1e-12
                end
            end

            # Higher order and an implicit operator: RK443's error is far below the
            # tolerance, and with ν Δζ on the LHS the device's diagonal-IMEX step plus
            # refresh must match the CPU's coupled global IMEX solve.
            ζex, φex = gtj_constraint_exact(k2 -> (-1 / k2, -1 / k2))
            for arch in (CPU(), _GTJ_ARCH)
                run = gtj_constraint_run(arch, ["Δ(φ) + tau_φ - ζ - θ = 0"]; ts=RK443())
                @test maximum(abs, run["φ"] .- φex) < 1e-10
                @test maximum(abs, run["ζ"] .- ζex) < 1e-10
            end

            # No LHS operator: the diagonal-IMEX RK schemes step explicitly, and only
            # the refresh imposes the constraint. The CPU fallback (vector path, global
            # mass matrix) gave the constraint rows k = F = 0 and never refreshed them,
            # so φ stayed at its initial zero while ζ advanced; the device refreshed it.
            for ts in (Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443())
                dev = gtj_constraint_run(_GTJ_ARCH, ["Δ(φ) + tau_φ - ζ - θ = 0"]; ts)
                cpu = gtj_constraint_run(CPU(), ["Δ(φ) + tau_φ - ζ - θ = 0"]; ts)
                @test maximum(abs, cpu["φ"] .- φex) < 1e-7
                @test maximum(abs, cpu["ζ"] .- ζex) < 1e-7
                @test maximum(abs, dev["φ"] .- cpu["φ"]) < 1e-12
                @test maximum(abs, dev["ζ"] .- cpu["ζ"]) < 1e-12
            end
            implicit = "∂t(ζ) - 0.1*Δ(ζ) = -0.3*φ"
            ζex, φex = gtj_constraint_exact(k2 -> (-1 / k2, -1 / k2); ν=0.1)
            dev = gtj_constraint_run(_GTJ_ARCH, ["Δ(φ) + tau_φ - ζ - θ = 0"]; evolution=implicit)
            cpu = gtj_constraint_run(CPU(), ["Δ(φ) + tau_φ - ζ - θ = 0"]; evolution=implicit)
            @test maximum(abs, dev["ζ"] .- ζex) < 1e-5
            @test maximum(abs, dev["φ"] .- φex) < 1e-5
            @test maximum(abs, dev["ζ"] .- cpu["ζ"]) < 1e-12
            @test maximum(abs, dev["φ"] .- cpu["φ"]) < 1e-12

            # A nonzero gauge pins the MEAN of φ to c/|Ω| (device: the refresh is the
            # only solver; the mean of ζ then drifts by -0.3·c/|Ω|·t).
            ζex, φex = gtj_constraint_exact(k2 -> (-1 / k2, 0.0))
            dev = gtj_constraint_run(_GTJ_ARCH, ["Δ(φ) + tau_φ - ζ = 0"]; bc="integ(φ) = 2.0")
            @test maximum(abs, dev["φ"] .- (φex .+ 2 / volume)) < 1e-7
            @test maximum(abs, dev["ζ"] .- (ζex .- 0.3 * 2 * T / volume)) < 1e-7

            # Two algebraic variables coupled in two equations cannot be solved one
            # at a time. A device solver has no other constraint solver, so it must
            # refuse at construction, naming the equation, instead of stepping with
            # φ and χ stale. The CPU stage solve imposes them; its RHS refresh must
            # then leave them alone (it used to overwrite φ with ζ - χ_stale).
            coupled = ["φ + χ - ζ = 0", "φ - χ - θ = 0"]
            err = try
                gtj_constraint_run(_GTJ_ARCH, coupled; tau=false, extra=("χ",))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            message = err === nothing ? "" : sprint(showerror, err)
            @test occursin("φ + χ - ζ = 0", message)
            @test occursin("couples", message)
            ζex, φex = gtj_constraint_exact(k2 -> (0.5, 0.5))
            consistent = Dict("φ" => 0.5 .* (S .+ 0.7 .* C), "χ" => 0.5 .* (S .- 0.7 .* C))
            cpu = gtj_constraint_run(CPU(), coupled; tau=false, extra=("χ",), init=consistent)
            @test maximum(abs, cpu["ζ"] .- ζex) < 1e-7
            @test maximum(abs, cpu["φ"] .- φex) < 1e-7
            @test maximum(abs, cpu["χ"] .- (cpu["ζ"] .- φex)) < 1e-7
        end

        @testset "explicit diagonal-IMEX fallback: ψ and u refreshed, CPU == device" begin
            # 2D turbulence without a LHS operator: ζ evolves explicitly, ψ and u are
            # algebraic. On the CPU, DiagonalIMEX_RK222/RK443 left ψ and u at their
            # initial zeros in every stored state (output and CFL read u = 0) while the
            # device refreshed them. The reference is the same tableau on the CPU's
            # global-matrix path, whose stage solve imposes Δψ = ζ and u = skew∇ψ.
            function gtj_turbulence(arch, ts; N=16, nsteps=4)
                coords = CartesianCoordinates("x", "y")
                dist = Distributor(coords; dtype=Float64, device=arch)
                domain = Domain(dist, (RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2),
                                       RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)))
                ζ = ScalarField(domain, "ζ"); ψ = ScalarField(domain, "ψ")
                u = VectorField(domain, "u")
                problem = InitialValueProblem([ζ, ψ, u, ScalarField(dist, "tau_ψ", (), Float64)])
                add_equation!(problem, "∂t(ζ) = -u⋅∇(ζ) - 0.05*ζ")
                add_equation!(problem, "Δ(ψ) + tau_ψ - ζ = 0")
                add_equation!(problem, "u - skew(grad(ψ)) = 0")
                add_bc!(problem, "integ(ψ) = 0")
                solver = InitialValueSolver(problem, ts; dt=2e-3)
                xs = collect(range(0, 2π, length=N + 1))[1:N]
                copyto!(grid_data!(ζ), [sin(2x) * cos(3y) + 0.5cos(x + 2y) for x in xs, y in xs])
                for _ in 1:nsteps
                    step!(solver)
                end
                return [gtj_grid(f) for f in (ζ, ψ, u.components[1], u.components[2])]
            end
            for (ts, reference_ts) in ((Tarang.DiagonalIMEX_RK222(), RK222()),
                                       (Tarang.DiagonalIMEX_RK443(), RK443()))
                reference = gtj_turbulence(CPU(), reference_ts)
                cpu = gtj_turbulence(CPU(), ts)
                dev = gtj_turbulence(_GTJ_ARCH, ts)
                @test maximum(abs, cpu[2]) > 0.1 && maximum(abs, cpu[3]) > 0.1
                for (c, d, r) in zip(cpu, dev, reference)
                    @test maximum(abs, c .- r) < 1e-12
                    @test maximum(abs, d .- c) < 1e-12
                end
            end
        end
    end
end
