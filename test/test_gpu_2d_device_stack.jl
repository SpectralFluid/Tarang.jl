"""
2D GPU stack device-safety + CPU/GPU value parity WITHOUT a GPU.

`test_2d_gpu_domain_compat.jl` covers the 2D pure-Fourier algebraic-constraint
refresh; this file covers the rest of the 2D GPU stack that had no coverage at
all: the stochastic-forcing device kernels (Hermitian symmetry, zero mode,
diagnostics), device-backed field construction/reductions, and the host→device
staging contract the coupled-subproblem `gather_alg_F!` relies on.

Method (same as test_2d_gpu_domain_compat.jl): run the REAL functions on
`JLArray` — GPUArrays' CPU-backed reference device array — with
`allowscalar(false)`, so any scalar `getindex`/`setindex!` throws exactly as it
would on a `CuArray`. A kernel that completes here is scalar-index-free; each
result is then compared against the CPU implementation, which is the oracle.

Beyond the two methods the older files patch, this one also supplies
`Tarang.array_type` for the JLArray backend — that is all it takes to build a
GPU-ARCHITECTURE `ScalarField` whose buffers are device arrays, which is what
lets the forcing/reduction paths be exercised end to end here.

The user-facing tools at the end — spectra, streamfunction, forcing registration
and `fill_random!` — run on a STRICT device (`strict_device_support.jl`): besides
scalar indexing, a host `Array` reaching a device kernel throws, as it does under
CUDA. Their transforms use a CPU-twin stand-in for cuFFT (bit-identical to the
CPU chain), so device results are compared to CPU ones exactly.
"""

using Test
using Random
using Tarang
using LinearAlgebra
using Statistics: mean, var

const _JLSTACK_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    @eval using FFTW
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping 2D GPU device-stack test" err
    false
end

if _JLSTACK_OK
    const _JLS = JLArrays.JLArray
    const _JLS_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package.
    Tarang.is_gpu_array(::_JLS) = true
    Tarang.architecture(::_JLS) = _JLS_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _JLS(a)
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::_JLS) = a
    Tarang.copy_to_device(a::AbstractArray, ::_JLS) = _JLS(Array(a))
    Tarang.copy_to_device(a::_JLS, ::_JLS) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _JLS
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _JLS{T}

    include("strict_device_support.jl")

    # ---- cuFFT stand-in: a CPU twin field transformed by Tarang's CPU chain ----
    const _JLS_TWINS = Dict{Any, Any}()
    function jls_twin(field)
        get!(_JLS_TWINS, (objectid(field.bases), field.dtype)) do
            cdist = Distributor(field.dist.coordsys; dtype=field.dtype, device=Tarang.CPU())
            ScalarField(Domain(cdist, field.bases), "jls_twin_" * field.name)
        end
    end
    function jls_sync_scales!(twin, field)
        twin.scales == field.scales && return
        twin.current_layout = :c
        Tarang.preset_scales!(twin, field.scales)
    end
    # Copy `src` into the buffer selected by `getter`/`setter` on `dst`,
    # reallocating (via `make`) only when the shape or eltype differ.
    function jls_copy_into!(getter, setter, make, dst, src)
        buf = getter(dst)
        if buf === nothing || size(buf) != size(src) || eltype(buf) != eltype(src)
            setter(dst, make(src))
        else
            copyto!(buf, src)
        end
    end
    function Tarang._gpu_forward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend},
                                                    field::Tarang.ScalarField)
        twin = jls_twin(field)
        jls_sync_scales!(twin, field)
        jls_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, copy, twin,
                       Array(Tarang.get_grid_data(field)))
        twin.current_layout = :g
        Tarang.forward_transform!(twin)
        jls_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, x -> _JLS(copy(x)),
                       field, Tarang.get_coeff_data(twin))
        return true
    end
    function Tarang._gpu_backward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend}, field)
        twin = jls_twin(field)
        jls_sync_scales!(twin, field)
        jls_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, copy, twin,
                       Array(Tarang.get_coeff_data(field)))
        twin.current_layout = :c
        Tarang.backward_transform!(twin)
        jls_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, x -> _JLS(copy(x)),
                       field, Tarang.get_grid_data(twin))
        return true
    end

    # Device-RNG seeding hook. JLArray's `randn!` draws from the HOST RNG, so this
    # method only records the seeds that reach it; the CUDA extension's method
    # seeds `CUDA.default_rng()`, the generator `randn!(::CuArray)` uses.
    const _JLS_DEVICE_SEEDS = Int[]
    Tarang._seed_device_rng!(::Tarang.GPU{JLArrays.JLBackend}, seed::Integer) =
        (push!(_JLS_DEVICE_SEEDS, seed); nothing)

    # 2D RealFourier field data on `arch` (host values copied in, never broadcast).
    function jls_domain(arch; Nx=16, Ny=12, Ly=2π)
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=arch)
        return Domain(dist, (RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π)),
                             RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly))))
    end
    function jls_set_grid!(field, values)
        ensure_layout!(field, :g)
        copyto!(get_grid_data(field), values)
        return field
    end
    function jls_flow(arch; Nx=16, Ny=12, Ly=2π)
        dom = jls_domain(arch; Nx, Ny, Ly)
        xs = [2π * (i - 1) / Nx for i in 1:Nx]
        ys = [Ly * (j - 1) / Ny for j in 1:Ny]
        ky = 2π / Ly
        u = VectorField(dom, "u")
        for (i, c) in enumerate(u.components)
            jls_set_grid!(c, [0.2i + 0.5cos(i * x) * sin(2ky * y) + 0.1sin(3x) for x in xs, y in ys])
        end
        f = jls_set_grid!(ScalarField(dom, "f"), [cos(x) * sin(3ky * y) + 0.05 for x in xs, y in ys])
        return u, f
    end
    jls_host_grid(f) = (ensure_layout!(f, :g); Array(get_grid_data(f)))
    jls_host_coeff(f) = (ensure_layout!(f, :c); Array(get_coeff_data(f)))
end

@testset "2D GPU stack device-safety (JLArray)" begin
    if !_JLSTACK_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)
        isdefined(@__MODULE__, :test_stochastic_cache_precision) || include("stochastic_cache_support.jl")
        test_stochastic_cache_precision(_JLS_ARCH)
        test_stochastic_cache_precision(_JLS_ARCH; separable=true)

        @testset "Hermitian symmetry: device kernel == CPU loop" begin
            # The forcing spectrum is a FULL complex spectrum (field_size, not an
            # rfft half), so the conjugate partner is index `n+2-i`. The GPU kernel
            # visits every (i,j) concurrently while the CPU version walks them in
            # order — if the two disagreed (or the kernel raced), the forcing would
            # stop being real-valued in grid space and the physics would be junk.
            for (nx, ny) in ((8, 8), (9, 7), (16, 12))
                Random.seed!(4)
                a = randn(ComplexF64, nx, ny)
                cpu = copy(a)
                dev = _JLS(copy(a))
                Tarang._enforce_hermitian_symmetry!(cpu, CPU())
                Tarang._enforce_hermitian_symmetry!(dev, _JLS_ARCH)
                @test Array(dev) == cpu
            end
        end

        @testset "Hermitian symmetry makes the grid field real" begin
            Random.seed!(5)
            dev = _JLS(randn(ComplexF64, 8, 8))
            Tarang._enforce_hermitian_symmetry!(dev, _JLS_ARCH)
            Tarang._set_zero_mode!(dev, _JLS_ARCH)
            @test maximum(abs, imag.(ifft(Array(dev)))) < 1e-12
        end

        @testset "_set_zero_mode! device == CPU" begin
            Random.seed!(6)
            a = randn(ComplexF64, 8, 8)
            cpu = copy(a)
            dev = _JLS(copy(a))
            Tarang._set_zero_mode!(cpu, CPU())
            Tarang._set_zero_mode!(dev, _JLS_ARCH)
            @test Array(dev) == cpu
        end

        @testset "GPU-architecture 2D field: buffers are device arrays" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; dtype=Float64, device=_JLS_ARCH)
            dom = Domain(dist, (RealFourier(coords["x"]; size=16, bounds=(0.0, 2π)),
                                RealFourier(coords["y"]; size=12, bounds=(0.0, 2π))))
            u = ScalarField(dom, "u")
            @test get_grid_data(u) isa _JLS
            @test get_coeff_data(u) isa _JLS
            @test size(get_grid_data(u)) == (16, 12)
            @test size(get_coeff_data(u)) == (9, 12)   # canonical rfft layout

            Random.seed!(7)
            host = randn(16, 12)
            get_grid_data(u) .= _JLS(host)
            # CFL-style reduction on device data — must not scalar-index
            @test maximum(abs, get_grid_data(u)) ≈ maximum(abs, host)
            @test sum(get_grid_data(u)) ≈ sum(host)
        end

        @testset "StochasticForcing 2D on a device architecture" begin
            fg = Tarang.StochasticForcing(; field_size=(16, 16), architecture=_JLS_ARCH,
                                          dtype=Float64, k_forcing=4.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            fc = Tarang.StochasticForcing(; field_size=(16, 16), architecture=CPU(),
                                          dtype=Float64, k_forcing=4.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            # The forcing spectrum is architecture-independent data
            @test Array(fg.spectrum) ≈ Array(fc.spectrum)

            Tarang.generate_forcing!(fg, 0.0)          # device kernels, no scalar indexing
            cf = fg.cached_forcing
            @test cf isa _JLS
            @test all(isfinite, Array(cf))
            # generate_forcing! ends with Hermitian symmetry + zero mode: the
            # realization must be a spectrum of a REAL field with no mean.
            @test abs(Array(cf)[1, 1]) == 0
            @test maximum(abs, imag.(ifft(Array(cf)))) < 1e-10
        end

        @testset "Forcing diagnostics: device path == CPU path" begin
            Random.seed!(8)
            fc = Tarang.StochasticForcing(; field_size=(8, 8), architecture=CPU(),
                                          dtype=Float64, k_forcing=3.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            Tarang.generate_forcing!(fc, 0.0)
            fg = Tarang.StochasticForcing(; field_size=(8, 8), architecture=_JLS_ARCH,
                                          dtype=Float64, k_forcing=3.0, dk_forcing=1.0,
                                          energy_injection_rate=0.1, dt=0.01)
            copyto!(fg.cached_forcing, _JLS(Array(fc.cached_forcing)))

            sol = randn(ComplexF64, 8, 8)
            prev = randn(ComplexF64, 8, 8)
            # one-argument form (Itô-style) and two-argument form (Stratonovich)
            @test Tarang._diagnostic_inner_product(fg, _JLS(copy(sol))) ≈
                  Tarang._diagnostic_inner_product(fc, copy(sol))
            @test Tarang._diagnostic_inner_product(fg, _JLS(copy(sol)), _JLS(copy(prev))) ≈
                  Tarang._diagnostic_inner_product(fc, copy(sol), copy(prev))
        end

        @testset "gather_alg_F! host→device staging contract" begin
            # `gather_alg_F!` builds the sparse BC/constraint vector on the host
            # (scalar writes must stay off device arrays) and uploads it with a
            # direct `copyto!`. Routing it through `_assign_to_buffer!` instead
            # killed every GPU coupled-subproblem step at the pre-stage gather,
            # because that helper's same-architecture guard refuses exactly this
            # transfer. Pin both halves of that contract.
            host = ComplexF64[1, 2, 3, 4]
            dev = _JLS(zeros(ComplexF64, 4))
            copyto!(dev, host)
            @test Array(dev) == host
            @test_throws Exception Tarang._assign_to_buffer!(_JLS(zeros(ComplexF64, 4)), host)
        end

        @testset "strict device rejects host arrays in kernels" begin
            # Guards the harness itself: were JLArray kernel arguments ever to stop
            # passing through Adapt, every strict testset below would pass vacuously.
            dev = _JLS(zeros(4))
            @test_throws HostArrayInKernel with_strict_device(() -> dev .+ ones(4))
            @test Array(dev .+ ones(4)) == ones(4)              # outside: plain JLArray
            @test Array(with_strict_device(() -> dev .+ _JLS(ones(4)))) == ones(4)
        end

        @testset "spectra on a strict device == CPU" begin
            # The shell binning, the full-spectrum dictionaries and the rfft weights
            # walked the density mode by mode: scalar indexing, which CUDA refuses.
            # Odd Nx (no Nyquist row) and Ly ≠ 2π cover both weight branches and
            # physical wavenumbers.
            for (Nx, Ny, Ly) in ((16, 12, 2π), (15, 10, 3.0))
                ud, fd = jls_flow(_JLS_ARCH; Nx, Ny, Ly)
                uc, fc = jls_flow(Tarang.CPU(); Nx, Ny, Ly)
                with_strict_device() do
                    for spectrum in (u -> energy_spectrum(u),
                                     u -> energy_spectrum(u; binning=LogBinning(bins_per_decade=4)),
                                     u -> enstrophy_spectrum(u))
                        d = spectrum(ud)
                        c = spectrum(uc)
                        @test d.k == c.k
                        @test d.bin_counts == c.bin_counts
                        @test d.power ≈ c.power rtol=1e-12 atol=1e-14
                    end
                    d = power_spectrum(fd)
                    c = power_spectrum(fc)
                    @test d.bin_counts == c.bin_counts
                    @test d.power ≈ c.power rtol=1e-12 atol=1e-14
                    for (d, c) in ((energy_spectrum(ud; radial_average=false),
                                    energy_spectrum(uc; radial_average=false)),
                                   (power_spectrum(fd; radial_average=false),
                                    power_spectrum(fc; radial_average=false)))
                        @test keys(d) == keys(c)
                        @test all(isapprox(d[k], c[k]; rtol=1e-12, atol=1e-14) for k in keys(c))
                    end
                    # The weighted spectral densities themselves stay on the device.
                    ke = calculate_spectral_kinetic_energy(ud)
                    @test ke isa _JLS
                    @test Array(ke) ≈ calculate_spectral_kinetic_energy(uc) rtol=1e-12 atol=1e-14
                    pw = calculate_spectral_power(fd)
                    @test pw isa _JLS
                    @test Array(pw) ≈ calculate_spectral_power(fc) rtol=1e-12 atol=1e-14
                end
            end
        end

        @testset "streamfunction on a strict device == CPU" begin
            # ψ̂ = -ω̂/k² was a per-mode loop: scalar setindex! on the device.
            ud, _ = jls_flow(_JLS_ARCH; Ly=3.0)
            uc, _ = jls_flow(Tarang.CPU(); Ly=3.0)
            ψd = with_strict_device(() -> streamfunction(ud))
            ψc = streamfunction(uc)
            @test get_coeff_data(ψd) isa _JLS
            @test jls_host_coeff(ψd) ≈ jls_host_coeff(ψc) rtol=1e-12 atol=1e-14
            @test jls_host_coeff(ψc)[1, 1] == 0                 # gauge: zero mean mode
            @test jls_host_grid(ψd) ≈ jls_host_grid(ψc) rtol=1e-12 atol=1e-14
        end

        @testset "forcing must match the field's architecture at registration" begin
            # A forcing built with the constructors' default architecture=CPU() and
            # registered on a GPU field handed host arrays to device kernels at the
            # first step: an error under CUDA, a host round trip per RHS at best.
            function jls_forced_problem(arch)
                u = jls_set_grid!(ScalarField(jls_domain(arch), "u"), zeros(16, 12))
                problem = InitialValueProblem([u])
                add_equation!(problem, "dt(u) = 0")
                return problem, u
            end
            stochastic(arch) = StochasticForcing(; field_size=(16, 12),
                energy_injection_rate=0.1, k_forcing=3.0, dk_forcing=1.0, dt=0.01,
                rng=MersenneTwister(3), architecture=arch)
            deterministic(arch) = DeterministicForcing(
                (x, y, t, p) -> cos.(x) .* sin.(2 .* y) .* (1 + t), (16, 12); architecture=arch)
            separable(arch) = SeparableStochasticForcing(; fourier_size=(16,),
                chebyshev_basis=ChebyshevT(CartesianCoordinates("z")["z"]; size=8, bounds=(0.0, 1.0)),
                chebyshev_profile=z -> z * (1 - z), energy_injection_rate=0.1,
                k_forcing=3.0, dk_forcing=1.0, dt=0.01, architecture=arch)

            for make in (stochastic, deterministic, separable),
                (field_arch, forcing_arch) in ((_JLS_ARCH, Tarang.CPU()), (Tarang.CPU(), _JLS_ARCH))
                problem, _ = jls_forced_problem(field_arch)
                forcing = make(forcing_arch)
                rng_before = forcing isa Tarang.StochasticForcingType ? copy(forcing.rng) : nothing
                err = try
                    add_stochastic_forcing!(problem, :u, forcing)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("architecture", sprint(showerror, err))
                @test isempty(problem.stochastic_forcings)
                if rng_before !== nothing
                    @test forcing.rng == rng_before             # a refused forcing is untouched
                end
            end
            # The constructors default to CPU(), so an unqualified forcing is refused
            # on a device field.
            problem, _ = jls_forced_problem(_JLS_ARCH)
            @test_throws ArgumentError add_stochastic_forcing!(problem, :u,
                DeterministicForcing((x, y, t, p) -> cos.(x) .* sin.(y), (16, 12)))

            # Matching architectures register and step on the strict device, == CPU.
            for make in (stochastic, deterministic)
                final = map((_JLS_ARCH, Tarang.CPU())) do arch
                    with_strict_device() do
                        problem, u = jls_forced_problem(arch)
                        add_stochastic_forcing!(problem, :u, make(arch))
                        solver = InitialValueSolver(problem, RK222(); dt=0.01)
                        for _ in 1:3
                            step!(solver)
                        end
                        jls_host_grid(u)
                    end
                end
                @test maximum(abs, final[2]) > 0.01             # the forcing was applied
                @test final[1] ≈ final[2] rtol=1e-12 atol=1e-14
            end
        end

        @testset "fill_random!: unit-variance complex normals; seed reaches the device RNG" begin
            # The counter-based device kernel drew BOTH components from N(0, 1), so a
            # complex coefficient fill had E|c|² = 2 against the host's 1 (2.08 vs
            # 1.08 on this grid). 33×48 samples put σ(mean |c|²) near 0.025.
            for arch in (_JLS_ARCH, Tarang.CPU())
                f = ScalarField(jls_domain(arch; Nx=64, Ny=48), "f")
                with_strict_device(() -> fill_random!(f, "c"; seed=7, distribution="normal"))
                c = Array(get_coeff_data(f))
                @test eltype(c) <: Complex
                @test isapprox(mean(abs2, c), 1; atol=0.15)
                @test isapprox(var(real.(c)), 0.5; atol=0.1)
                @test isapprox(var(imag.(c)), 0.5; atol=0.1)
                g = ScalarField(jls_domain(arch; Nx=64, Ny=48), "g")
                with_strict_device(() -> fill_random!(g, "g"; seed=7, distribution="normal"))
                @test isapprox(var(Array(get_grid_data(g))), 1; atol=0.15)   # real: unchanged
            end

            # reproducible=false: `Random.seed!` reaches only the host RNG, while
            # `randn!(::CuArray)` draws from CUDA's own generator, seeded from
            # RandomDevice, so every seeded GPU run differed. The seed now also goes
            # to the device generator through `_seed_device_rng!`.
            empty!(_JLS_DEVICE_SEEDS)
            f = ScalarField(jls_domain(_JLS_ARCH; Nx=64, Ny=48), "f")
            with_strict_device(() -> fill_random!(f, "c"; seed=11, reproducible=false))
            @test _JLS_DEVICE_SEEDS == [11]
            @test isapprox(mean(abs2, Array(get_coeff_data(f))), 1; atol=0.15)
            v = VectorField(jls_domain(_JLS_ARCH), "v")
            with_strict_device(() -> fill_random!(v, "g"; seed=5, reproducible=false))
            @test _JLS_DEVICE_SEEDS == [11, 5, 5 + 1000003]         # one seed per component
            # Unseeded, reproducible and host fills never touch the device generator.
            fill_random!(f, "g"; reproducible=false)
            fill_random!(f, "g"; seed=3)
            fill_random!(ScalarField(jls_domain(Tarang.CPU()), "h"), "g"; seed=3, reproducible=false)
            @test _JLS_DEVICE_SEEDS == [11, 5, 5 + 1000003]
            @test Tarang._seed_device_rng!(Tarang.CPU(), 3) === nothing
            # A GPU backend that cannot seed its generator refuses, rather than hand
            # back a "seeded" fill that will not repeat.
            @test_throws ArgumentError Tarang._seed_device_rng!(Tarang.GPU{Nothing}(nothing), 3)
        end
    end
end
