using Test, Tarang, Random

@testset "3D spectral padding does not materialize slices" begin
    rng = MersenneTwister(814)
    shape = (32, 24, 16)
    source = randn(rng, ComplexF64, shape)
    for dims in ([1], [2], [3], [1, 2, 3])
        padded_shape = ntuple(d -> d in dims ? 3 * shape[d] ÷ 2 : shape[d], 3)
        padded = zeros(ComplexF64, padded_shape)
        recovered = similar(source)
        Tarang._pad_spectral!(padded, source, shape, padded_shape, dims)
        Tarang._truncate_spectral!(recovered, padded, shape, padded_shape, dims)
        @test recovered ≈ source rtol=1e-14 atol=1e-14
        # Warmed helpers may allocate small metadata/views, never an O(grid) copy.
        @test (@allocated Tarang._pad_spectral!(padded, source, shape, padded_shape, dims)) < 32768
        @test (@allocated Tarang._truncate_spectral!(recovered, padded, shape, padded_shape, dims)) < 32768
        @test recovered ≈ source rtol=1e-14 atol=1e-14
    end
end
