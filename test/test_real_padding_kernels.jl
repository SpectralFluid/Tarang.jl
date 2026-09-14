using Test, Tarang, FFTW, Random

@testset "Half-spectrum maps match full-complex padding on every Fourier axis" begin
    for shape in ((9, 10), (10, 9, 8)), T in (Float32, Float64)
        rng = MersenneTwister(23)
        for mask in 1:(2^length(shape)-1)
            dims = findall(d -> !iszero(mask & (1 << (d-1))), 1:length(shape))
            rdim = first(dims)
            fourier = ntuple(d -> d in dims, length(shape))
            padshape = ntuple(d -> fourier[d] ? 2cld(ceil(Int, 1.5shape[d]), 2) : shape[d], length(shape))
            halfshape = ntuple(d -> d == rdim ? padshape[d] ÷ 2 + 1 : padshape[d], length(shape))
            a = randn(rng, T, shape)
            full = zeros(Complex{T}, padshape)
            Tarang._pad_spectral!(full, fft(a, dims), shape, padshape, dims)
            half = zeros(Complex{T}, halfshape)
            src = rfft(a, dims)
            Tarang.launch!(CPU(), Tarang._pad_real_spectrum_kernel!, half, src, shape, padshape, fourier, rdim;
                           ndrange=halfshape)
            keep = ntuple(d -> d == rdim ? (1:halfshape[d]) : Colon(), length(shape))
            @test half ≈ full[keep...] rtol=100eps(T) atol=100eps(T)

            # Include arbitrary transverse phases on every Nyquist intersection.
            padded_grid = randn(rng, T, padshape)
            full = fft(padded_grid, dims)
            half = rfft(padded_grid, dims)
            expected = zeros(Complex{T}, shape)
            Tarang._truncate_spectral!(expected, full, shape, padshape, dims)
            recovered = similar(src)
            Tarang.launch!(CPU(), Tarang._truncate_real_spectrum_kernel!, recovered, half,
                           shape, padshape, fourier, rdim; ndrange=size(recovered))
            keep = ntuple(d -> d == rdim ? (1:size(src, d)) : Colon(), length(shape))
            @test recovered ≈ expected[keep...] rtol=100eps(T) atol=100eps(T)
        end
    end
end

@testset "Real coefficient shortcut preserves arbitrary boundary-plane semantics" begin
    for shape in ((10, 9), (9, 10, 8))
        coords = CartesianCoordinates(("x", "y", "z")[1:length(shape)]...)
        dist = Distributor(coords; dtype=Float64)
        bases = ntuple(d -> RealFourier(coords[d]; size=shape[d], dealias=1.5), length(shape))
        domain = Domain(dist, bases)
        a, b = ScalarField(domain, "a"), ScalarField(domain, "b")
        rng = MersenneTwister(54)
        coeff_data!(a) .= randn(rng, ComplexF64, size(get_coeff_data(a)))
        coeff_data!(b) .= randn(rng, ComplexF64, size(get_coeff_data(b)))
        aa, bb = copy(a), copy(b)
        ev = Tarang._get_evaluator(dist)
        direct = evaluate_transform_multiply(a, b, ev; result_layout=:c)
        ensure_layout!(aa, :g); ensure_layout!(bb, :g)
        reference = evaluate_transform_multiply(aa, bb, ev)
        @test Array(grid_data!(direct)) ≈ Array(grid_data!(reference)) rtol=1e-12 atol=1e-12
    end
end
