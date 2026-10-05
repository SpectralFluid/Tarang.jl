using Test
using Tarang
using FFTW

# Unit-step response of y'' + sqrt(2)*α*y' + α²*y = α², y(0)=y'(0)=0.
# This is independent of Tarang's ETD coefficient implementation.
_review_butterworth_step(α, t) =
    1 - exp(-α*t/sqrt(2)) * (cos(α*t/sqrt(2)) + sin(α*t/sqrt(2)))

@testset "Wave-mean filters follow changing timesteps" begin
    α = 1.3
    timesteps = (0.1, 0.9, 0.025)
    profile = [1.0, 2.0, 4.0]
    field = repeat(reshape(profile, 1, :), 8, 1)
    decomp = Tarang.WaveMeanDecomposition(size(field); α, horizontal_dims=(1,))
    flux_decomp = Tarang.WaveMeanDecomposition(size(field); α, horizontal_dims=(1,))
    forcing = Tarang.WaveInducedForcing(size(field); α, horizontal_dims=(1,))
    Tarang.add_field!(forcing, :u)
    Tarang.add_field!(forcing, :w)
    Tarang.add_flux!(forcing, :uw)
    wave = repeat(reshape(cos.(2pi .* (0:7) ./ 8), :, 1), 1, 3)

    elapsed = 0.0
    for dt in timesteps
        elapsed += dt
        response = _review_butterworth_step(α, elapsed)
        mean, _ = Tarang.decompose!(decomp, :u, field, dt)
        flux = Tarang.update_flux!(flux_decomp, :uw, 3 .* field, dt)
        @test mean ≈ response .* profile atol=2e-12
        @test flux ≈ 3response .* profile atol=2e-12

        # The public forcing wrapper must propagate the same changing dt to
        # both its mean and flux filters. A zero-mean cosine has mean square 1/2.
        Tarang.update!(forcing, Dict(:u => wave, :w => wave), dt)
        @test Tarang.get_mean(forcing, :u) ≈ zeros(3) atol=2e-12
        @test Tarang.get_flux(forcing, :uw) ≈ fill(response/2, 3) atol=2e-12
    end
end

@testset "GQL stresses contain only small-scale modes" begin
    shape = (16, 8, 4)
    x = reshape(2pi .* (0:shape[1]-1) ./ shape[1], :, 1, 1)
    z = reshape([-0.7, 0.1, 0.4, 1.2], 1, 1, :)
    α = 0.8
    function make_system(cutoff)
        sys = Tarang.GQLWaveMeanSystem(shape, (2pi, 2pi); Λ=cutoff, α)
        Tarang.add_field!(sys, :u)
        Tarang.add_field!(sys, :w)
        Tarang.add_flux!(sys, :uw)
        return sys
    end
    # Large-scale modes alone must never produce small-scale Reynolds stress,
    # even while the temporal mean is still relaxing from its zero initial state.
    large = Dict(:u => repeat(2 .+ z .+ cos.(x), 1, shape[2], 1),
                 :w => repeat(-1 .+ z .- cos.(x), 1, shape[2], 1))
    large_hat = Dict(name => rfft(values) for (name, values) in large)
    sys_large = make_system(2.0)
    Tarang.update!(sys_large, large_hat, large, 0.2)
    @test maximum(abs, Tarang.get_small(sys_large, :u)) < 1e-12
    @test Tarang.get_flux(sys_large, :uw) ≈ zeros(shape[3]) atol=1e-12

    # Both fields contain low kx=1 and high kx=4 modes. The high-mode product
    # has horizontal mean 3*(1+z)*(2+z), independently at each vertical point.
    # rfft/irfft must cover all axes to retain this vertical profile correctly.
    physical = Dict(:u => large[:u] .+ 2 .* cos.(4 .* x) .* (1 .+ z),
                    :w => large[:w] .+ 3 .* cos.(4 .* x) .* (2 .+ z))
    spectral = Dict(name => rfft(values) for (name, values) in physical)
    spectral_before = deepcopy(spectral)
    sys = make_system(2.0)
    elapsed = 0.0
    for dt in (0.1, 0.9)
        elapsed += dt
        Tarang.update!(sys, spectral, physical, dt)
        response = _review_butterworth_step(α, elapsed)
        expected_flux = 3 .* vec((1 .+ z) .* (2 .+ z)) .* response
        @test Tarang.get_flux(sys, :uw) ≈ expected_flux atol=3e-12
        @test Tarang.get_mean(sys, :u) ≈ vec(2 .+ z) .* response atol=3e-12
        @test Tarang.get_mean(sys, :w) ≈ vec(-1 .+ z) .* response atol=3e-12
        @test Tarang.get_large(sys, :u) .+ Tarang.get_small(sys, :u) ≈ spectral[:u]
    end
    @test spectral == spectral_before

    # Moving kx=4 into the large-scale partition removes its wave stress.
    all_large = make_system(4.0)
    Tarang.update!(all_large, spectral, physical, 1.0)
    @test Tarang.get_flux(all_large, :uw) ≈ zeros(shape[3]) atol=1e-12
end

@testset "Named NetCDF slice tasks write selected values" begin
    mktempdir() do path
        domain = PeriodicDomain(8, 6)
        field = ScalarField(domain, "u")
        set!(field, (x, y) -> sin(x) + 2cos(y))
        expected = copy(grid_data!(field))
        handler = Tarang.NetCDFFileHandler(joinpath(path, "slices"), field.dist,
                                           Dict("u" => field))
        Tarang.add_slice_task!(handler, field; slices=Dict(:x => 2), name="at_x")
        Tarang.add_slice_task!(handler, field; slices=(y=0.5,), name="at_y")
        Tarang.add_slice_task!(handler, field; slices=(x=2, y=3), name="point")
        @test Tarang.process!(handler; iteration=1)
        file = Tarang.current_file(handler)
        @test vec(Tarang.group_ncread(file, "vars", "at_x")) ≈ expected[2, :]
        # Fractional positions use round(value*(N-1))+1, preserving existing API.
        iy = round(Int, 0.5*(size(expected, 2)-1)) + 1
        @test vec(Tarang.group_ncread(file, "vars", "at_y")) ≈ expected[:, iy]
        @test only(Tarang.group_ncread(file, "vars", "point")) ≈ expected[2, 3]
        @test grid_data!(field) == expected
        close(handler)
    end
end
