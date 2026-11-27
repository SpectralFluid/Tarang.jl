# Device paths that a Float32-only GPU (Apple Metal) exposes: a Float64 constant,
# weight or allocation that reaches a kernel compiles fine on CUDA and ROCm but
# is an InvalidIRError ("unsupported use of double value") on Metal. Each case
# compares the device result with the same Float32 computation on the CPU.
# Also covers the even-Nx RealFourier Nyquist mode, whose pressure constant is
# undetermined, so every GPU must factor a rank-deficient stage matrix.
#
# Caller loads Test, Tarang, LinearAlgebra and Random, and passes a GPU architecture.

_f32_host(a) = Array(a)
_f32_relerr(a, b) = maximum(abs, _f32_host(a) .- _f32_host(b)) / max(maximum(abs, _f32_host(b)), eps(Float32))

function _f32_velocity(arch; N=16)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float32, device=arch, comm=Tarang.MPI.COMM_SELF)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    dom = Domain(dist, (xb, yb))
    u = VectorField(dom, "u")
    set!(u.components[1], (x, y) -> sin(3x) * cos(2y))
    set!(u.components[2], (x, y) -> cos(x) + 0.5sin(5y))
    return dist, dom, u
end

# Run `f(arch)` on the device and on the CPU at Float32 and return both results.
function _f32_on_both(f, arch)
    return map((arch, CPU())) do a
        dist, dom, u = _f32_velocity(a)
        try
            f(dist, dom, u)
        finally
            close(dist)
        end
    end
end

function _f32_rayleigh_benard(arch, T; Nx, Nz=12, ts, steps=4)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=T, device=arch, comm=Tarang.MPI.COMM_SELF)
    try
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 4.0), dealias=3/2)
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0), dealias=3/2)
        dom = Domain(dist, (xb, zb))
        p, b, u = ScalarField(dom, "p"), ScalarField(dom, "b"), VectorField(dom, "u")
        tau_p = ScalarField(dist, "tau_p", (), T)
        tau_b1, tau_b2 = ScalarField(dist, "tau_b1", (xb,), T), ScalarField(dist, "tau_b2", (xb,), T)
        tau_u1 = VectorField(dist, coords, "tau_u1", (xb,), T)
        tau_u2 = VectorField(dist, coords, "tau_u2", (xb,), T)
        ex, ez = unit_vector_fields(coords, dist)
        lb = derivative_basis(zb, 1)
        τl(A) = lift(A, lb, -1)
        grad_u = grad(u) + ez * τl(tau_u1)
        grad_b = grad(b) + ez * τl(tau_b1)
        prob = InitialValueProblem([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2])
        add_parameters!(prob; buoy=1e4, ez, grad_u, grad_b, τl)
        add_equation!(prob, "trace(grad_u) + tau_p = 0")
        add_equation!(prob, "∂t(b) - div(grad_b) + τl(tau_b2) = -u⋅∇(b)")
        add_equation!(prob, "∂t(u) - div(grad_u) + ∇(p) - buoy*b*ez + τl(tau_u2) = -u⋅∇(u)")
        add_bc!(prob, "b(z=0) = 1"); add_bc!(prob, "b(z=1.0) = 0")
        add_bc!(prob, "u(z=0) = 0"); add_bc!(prob, "u(z=1.0) = 0")
        add_bc!(prob, "integ(p) = 0")
        solver = InitialValueSolver(prob, ts; dt=1e-3)
        g = Tarang.get_grid_coordinates(dom; on_device=false)
        copyto!(grid_data!(b), T.([1 - z + 1e-2 * z * (1 - z) * sin(π * x / 2) for x in g["x"], z in g["z"]]))
        for _ in 1:steps
            step!(solver)
        end
        return (b=_f32_host(grid_data!(b)), uz=_f32_host(grid_data!(u.components[2])))
    finally
        close(dist)
    end
end

function check_float32_device_paths(arch)
    @testset "Float32-only device paths" begin
        @testset "vorticity-streamfunction constraint refresh" begin
            # u - skew(grad(psi)) = 0 is refreshed in coefficient space with a sign flip.
            res = map((arch, CPU())) do a
                coords = CartesianCoordinates("x", "y")
                dist = Distributor(coords; dtype=Float32, device=a, comm=Tarang.MPI.COMM_SELF)
                try
                    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=3/2)
                    yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π), dealias=3/2)
                    dom = Domain(dist, (xb, yb))
                    q, psi, u = ScalarField(dom, "q"), ScalarField(dom, "psi"), VectorField(dom, "u")
                    tau = ScalarField(dist, "tau", (), Float32)
                    prob = InitialValueProblem([q, psi, u, tau])
                    add_parameters!(prob; nu=1e-3)
                    add_equation!(prob, "∂t(q) - nu*Δ(q) = -u⋅∇(q)")
                    add_equation!(prob, "Δ(psi) + tau - q = 0")
                    add_equation!(prob, "u - skew(grad(psi)) = 0")
                    add_bc!(prob, "integ(psi) = 0")
                    set!(q, (x, y) -> sin(2x) * cos(3y) + cos(x))
                    s = InitialValueSolver(prob, SBDF2(); dt=1e-2)
                    for _ in 1:3
                        step!(s)
                    end
                    _f32_host(grid_data!(u.components[1]))
                finally
                    close(dist)
                end
            end
            @test _f32_relerr(res[1], res[2]) < 1e-4
        end

        @testset "CFL timestep" begin
            dts = _f32_on_both(arch) do dist, dom, u
                q = ScalarField(dom, "q")
                prob = InitialValueProblem([q])
                add_equation!(prob, "dt(q) = 0")
                cfl = CFL(InitialValueSolver(prob, RK222(); dt=0.1); initial_dt=0.1, cadence=1,
                          safety=0.5, max_dt=1.0)
                add_velocity!(cfl, u)
                compute_timestep(cfl)
            end
            @test dts[1] ≈ dts[2] rtol=1e-5
        end

        @testset "spectra and flow diagnostics" begin
            for name in (:power_spectrum, :energy_spectrum, :enstrophy_spectrum)
                spectra = _f32_on_both(arch) do dist, dom, u
                    name === :power_spectrum ? power_spectrum(u.components[1]) :
                    name === :energy_spectrum ? energy_spectrum(u) : enstrophy_spectrum(u)
                end
                values = map(s -> hasproperty(s, :power) ? s.power : s.spectrum, spectra)
                @test _f32_relerr(values[1], values[2]) < 1e-4
            end
            for diagnostic in (streamfunction, kinetic_energy, enstrophy, u -> energy_dissipation_rate(u, 1e-3))
                fields = _f32_on_both(arch) do dist, dom, u
                    _f32_host(grid_data!(diagnostic(u)))
                end
                @test _f32_relerr(fields[1], fields[2]) < 1e-4
            end
        end

        @testset "field writes, fills and scalar arithmetic" begin
            dist, dom, u = _f32_velocity(arch)
            try
                q = ScalarField(dom, "q")
                fill_random!(q, "g")                                   # unseeded: backend RNG
                fill_random!(q, "g"; seed=3, reproducible=false)
                fill_random!(q, "g"; seed=3, distribution="uniform", reproducible=false)
                @test all(x -> -1 <= x <= 1, _f32_host(grid_data!(q)))
                set!(q, 1.5)
                @test all(==(1.5f0), _f32_host(grid_data!(q)))
                doubled = q * 2.0   # Julia promotion, narrowed where the device has no Float64
                @test doubled.dtype === (Tarang._device_supports_float64(arch) ? Float64 : Float32)
                @test all(==(3), _f32_host(grid_data!(doubled)))
                g = Tarang.get_grid_coordinates(dom; on_device=false)
                host = [sin(x) * cos(y) for x in g["x"], y in g["y"]]   # Float64 host data
                q["g"] = host
                @test _f32_host(grid_data!(q)) ≈ Float32.(host)
                @test Tarang.architecture(get_grid_data(q)) == arch
                buffer = similar(get_grid_data(q))
                Tarang.vectorized_axpy!(buffer, 2.0, get_grid_data(q), get_grid_data(q))
                @test _f32_host(buffer) ≈ 3 .* Float32.(host)
            finally
                close(dist)
            end
        end

        @testset "integration on Fourier × Chebyshev" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; dtype=Float32, device=arch, comm=Tarang.MPI.COMM_SELF)
            try
                xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
                zb = ChebyshevT(coords["z"]; size=17, bounds=(-1.0, 2.0))
                u = ScalarField(Domain(dist, (xb, zb)), "u")
                set!(u, (x, z) -> sin(2x) * z^3 + cos(x) * z + 1)
                @test evaluate(Tarang.integrate(u, coords)) ≈ 6π rtol=1e-5
                line = evaluate(Tarang.integrate(u, coords["z"]))
                @test line.dtype === Float32
                @test Tarang.architecture(get_grid_data(line)) == arch
            finally
                close(dist)
            end
        end

        @testset "deterministic forcing written with Float64 constants" begin
            forced = _f32_on_both(arch) do dist, dom, u
                q = ScalarField(dom, "q")
                set!(q, 0f0)
                f = DeterministicForcing((x, y, t, p) -> p[:A] .* sin.(x) .* cos.(y) .* cos(p[:ω] * t),
                                         (16, 16); parameters=Dict(:A => 0.1, :ω => 2.0),
                                         architecture=dist.architecture)
                prob = InitialValueProblem([q])
                add_equation!(prob, "dt(q) = 0")
                add_stochastic_forcing!(prob, :q, f)
                s = InitialValueSolver(prob, RK222(); dt=0.01)
                for _ in 1:3
                    step!(s)
                end
                _f32_host(grid_data!(q))
            end
            @test maximum(abs, forced[2]) > 0
            @test _f32_relerr(forced[1], forced[2]) < 1e-5
        end

        @testset "Nyquist pressure mode of an even-Nx channel" begin
            # The CPU factors this mode with sparse QR. The GPU pins the free unknown
            # and must reach the same fields to single-precision accuracy.
            for ts in (RK222(), SBDF2())
                reference = _f32_rayleigh_benard(CPU(), Float64; Nx=16, ts)
                device = _f32_rayleigh_benard(arch, Float32; Nx=16, ts)
                @test _f32_relerr(device.b, reference.b) < 1e-4
                @test _f32_relerr(device.uz, reference.uz) < 1e-2
            end
        end
    end
end
