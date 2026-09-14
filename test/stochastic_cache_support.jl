# Exercise the same timestamp contract on CPU, JLArrays, and CUDA when available.
function test_stochastic_cache_precision(architecture; separable=false)
    coords = CartesianCoordinates("z")
    zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
    for T in (Float32, Float64)
        @testset "forcing cache dtype=$T separable=$separable" begin
            kwargs = (; dtype=T, architecture, rng=MersenneTwister(42),
                       k_forcing=2.0, dt=2.0^-25)
            f = separable ? SeparableStochasticForcing(; fourier_size=(8,),
                chebyshev_basis=zb, chebyshev_profile=z -> 1.0, kwargs...) :
                StochasticForcing(; field_size=(8,), kwargs...)

            first_draw = Array(copy(generate_forcing!(f, 0.1)))
            rng_after_draw = copy(f.rng)
            @test Array(generate_forcing!(f, 0.1)) == first_draw
            @test rand(copy(f.rng), UInt64) == rand(copy(rng_after_draw), UInt64)
            @test Array(generate_forcing!(f, 0.1, 2)) == first_draw

            # Two different Float64 times that both round to 1.0f0 must
            # consume two RNG draws, even when the forcing arrays are Float32.
            a = Array(copy(generate_forcing!(f, 1.0 - 2.0^-25)))
            rng_before_next = copy(f.rng)
            b = Array(copy(generate_forcing!(f, 1.0)))
            @test a != b
            @test rand(copy(f.rng), UInt64) != rand(copy(rng_before_next), UInt64)
            @test Array(generate_forcing!(f, 1.0)) == b

            # Cache invalidation still regenerates a draw at the same time.
            set_dt!(f, 0.02)
            changed_dt = Array(copy(generate_forcing!(f, 1.0)))
            @test changed_dt != b
            @test Array(generate_forcing!(f, 1.0)) == changed_dt
            reset_forcing!(f)
            reset_draw = Array(copy(generate_forcing!(f, 1.0)))
            @test reset_draw != changed_dt
            @test Array(generate_forcing!(f, 1.0)) == reset_draw
            @test eltype(f.cached_forcing) == Complex{T}
        end
    end
end
