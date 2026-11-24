using Test
using Tarang

@testset "Mixed transform stage shapes" begin
    @testset "Chebyshev first with both axes scaled" begin
        coords = CartesianCoordinates("z", "x")
        bases = (
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
            RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
        )

        ops, coeff_shape, _ = Tarang.forward_layout(bases, (14, 12), Float64)
        stages = Tarang.transform_stage_shapes(ops, (14, 12), (2, 1))

        @test coeff_shape == (9, 7)
        @test stages == [(14, 12), (14, 7), (9, 7)]
        @test reverse(stages) == [(9, 7), (14, 7), (14, 12)]
    end

    @testset "Fourier first and complex Fourier preserve the declared order" begin
        coords = CartesianCoordinates("x", "z")
        real_bases = (
            RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
        )
        real_ops, _, _ = Tarang.forward_layout(real_bases, (12, 14), Float64)
        @test Tarang.transform_stage_shapes(real_ops, (12, 14), (1, 2)) ==
              [(12, 14), (7, 14), (7, 9)]

        complex_bases = (
            ComplexFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
        )
        complex_ops, _, _ = Tarang.forward_layout(complex_bases, (12, 14), ComplexF64)
        @test Tarang.transform_stage_shapes(complex_ops, (12, 14), (1, 2)) ==
              [(12, 14), (12, 14), (12, 9)]

        reverse_coords = CartesianCoordinates("z", "x")
        reverse_complex_bases = (
            ChebyshevT(reverse_coords["z"]; size=9, bounds=(0.0, 1.0)),
            ComplexFourier(reverse_coords["x"]; size=8, bounds=(0.0, 2π)),
        )
        reverse_ops, _, _ = Tarang.forward_layout(
            reverse_complex_bases, (14, 12), ComplexF64)
        @test Tarang.transform_stage_shapes(reverse_ops, (14, 12), (2, 1)) ==
              [(14, 12), (14, 12), (9, 12)]
    end

    @testset "unscaled and malformed orders" begin
        coords = CartesianCoordinates("x", "z")
        bases = (
            RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
            ChebyshevT(coords["z"]; size=9, bounds=(0.0, 1.0)),
        )
        ops, _, _ = Tarang.forward_layout(bases, (8, 9), Float64)
        @test Tarang.transform_stage_shapes(ops, (8, 9), (1, 2)) ==
              [(8, 9), (5, 9), (5, 9)]

        @test_throws ArgumentError Tarang.transform_stage_shapes(ops, (8, 9), (1, 1))
        @test_throws ArgumentError Tarang.transform_stage_shapes(ops, (8, 9), (1,))
        @test_throws DimensionMismatch Tarang.transform_stage_shapes(ops, (8,), (1, 2))
    end
end

@testset "Transform axis prefix resizing" begin
    @testset "truncate and restore axis 2" begin
        source = reshape(collect(1:54), 6, 9)
        truncated = zeros(Int, 6, 5)

        @test Tarang._copy_axis_prefix!(truncated, source, 2) === truncated
        @test truncated == source[:, 1:5]

        restored = fill(-1, 6, 9)
        @test Tarang._zero_pad_axis_prefix!(restored, truncated, 2) === restored
        @test restored[:, 1:5] == truncated
        @test iszero(restored[:, 6:9])
    end

    @testset "complex data along axis 1" begin
        source = complex.(reshape(collect(1:24), 6, 4),
                          reshape(collect(25:48), 6, 4))
        truncated = zeros(Complex{Int}, 3, 4)
        Tarang._copy_axis_prefix!(truncated, source, 1)
        @test truncated == source[1:3, :]

        restored = fill(Complex{Int}(-1, -1), 6, 4)
        Tarang._zero_pad_axis_prefix!(restored, truncated, 1)
        @test restored[1:3, :] == truncated
        @test iszero(restored[4:6, :])
    end

    @testset "invalid resize shapes are rejected" begin
        @test_throws ArgumentError Tarang._copy_axis_prefix!(zeros(5, 4), zeros(3, 4), 1)
        @test_throws ArgumentError Tarang._zero_pad_axis_prefix!(zeros(3, 4), zeros(5, 4), 1)
        @test_throws DimensionMismatch Tarang._copy_axis_prefix!(zeros(3, 5), zeros(6, 4), 1)
        @test_throws DimensionMismatch Tarang._zero_pad_axis_prefix!(zeros(6, 5), zeros(3, 4), 1)
        @test_throws ArgumentError Tarang._copy_axis_prefix!(zeros(3, 4), zeros(6, 4), 3)
    end

    @testset "resize picks truncate or zero-pad from the lengths" begin
        source = reshape(collect(1.0:12.0), 3, 4)
        @test Tarang._resize_axis_prefix!(zeros(3, 2), source, 2) == source[:, 1:2]
        padded = Tarang._resize_axis_prefix!(fill(-1.0, 3, 6), source, 2)
        @test padded[:, 1:4] == source && iszero(padded[:, 5:6])
        @test Tarang._resize_axis_prefix!(zeros(3, 4), source, 2) == source
    end
end

# Backward transforms into a grid of another resolution resample the stored
# spectrum on EVERY Fourier axis by one shared rule (layout.jl). These pin the
# rule itself; test_cov_field_data_scales.jl pins it through set_scales!.
using FFTW
@testset "Backward spectral resampling rules" begin
    @testset "spectral_resample_ranges reproduces the grid resample_1d!" begin
        # resample_1d! special-cases one-point grids (it copies a sample), so
        # start at two points.
        for n_src in 2:13, n_dst in 2:13
            signal = randn(ComplexF64, n_src)
            expected = zeros(ComplexF64, n_dst)
            Tarang.resample_1d!(expected, signal)
            spectrum = fft(signal)
            r = Tarang.spectral_resample_ranges(n_src, n_dst)
            resampled = zeros(ComplexF64, n_dst)
            for (s, d) in ((r.pos_src, r.pos_dst), (r.neg_src, r.neg_dst),
                           (r.fold_src, r.fold_dst))
                resampled[d] .+= spectrum[s]
            end
            @test ifft(resampled .* (n_dst / n_src)) ≈ expected atol=1e-12
        end
    end

    @testset "half spectra keep the positive slice and owe the even fold" begin
        @test Tarang.half_spectrum_resample(16, 16) == (9, 0)
        @test Tarang.half_spectrum_resample(16, 24) == (8, 0)   # even base Nyquist dropped
        @test Tarang.half_spectrum_resample(15, 23) == (8, 0)   # odd: no Nyquist bin
        @test Tarang.half_spectrum_resample(16, 12) == (7, 7)   # even target: fold bin 7
        @test Tarang.half_spectrum_resample(16, 9) == (5, 0)
        plane = ComplexF64[1+2im 3-1im; 5+5im 7+0im]
        Tarang._fold_half_spectrum_nyquist!(plane, 1, 2)
        @test plane == ComplexF64[1+2im 3-1im; 10 14]
        @test Tarang._fold_half_spectrum_nyquist!(copy(plane), 1, 0) == plane
    end

    coords = CartesianCoordinates("x", "y", "z")
    rf = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    cf = ComplexFourier(coords["y"]; size=12, bounds=(0.0, 2π))
    cheb = ChebyshevT(coords["z"]; size=9, bounds=(-1.0, 1.0))

    @testset "backward_axis_op resamples a mismatched spectrum on any Fourier axis" begin
        @test Tarang.backward_axis_op(rf, 9, 24, true, true) ==
              Tarang.AxisOp(:irfft_resampled, 24, false, 13)
        @test Tarang.backward_axis_op(rf, 9, 12, true, true) ==
              Tarang.AxisOp(:irfft_resampled, 12, false, 7)
        @test Tarang.backward_axis_op(rf, 13, 24, true, true).op === :irfft
        @test Tarang.backward_axis_op(rf, 16, 24, true, false) ==
              Tarang.AxisOp(:ifft_resampled, 24, true, 24)
        @test Tarang.backward_axis_op(cf, 12, 18, true) ==
              Tarang.AxisOp(:ifft_resampled, 18, true, 18)
        @test Tarang.backward_axis_op(cf, 18, 18, true) == Tarang.AxisOp(:ifft, 18, true)
        # A full-length spectrum written directly keeps its C2C inverse ...
        @test Tarang.backward_axis_op(rf, 16, 16, true, true).op === :ifft
        # ... but a half spectrum of a third grid (24 or 25 points) is refused.
        @test_throws DimensionMismatch Tarang.backward_axis_op(rf, 13, 16, true, true)
    end

    @testset "whole-field rules and the one-pass resample" begin
        bases = (rf, cf, cheb)
        rules, grid_coeff_shape, nyquist =
            Tarang.backward_resample_rules(bases, (9, 12, 9), (12, 18, 14), Float64)
        @test rules == ((16, 12, :half), (12, 18, :c2c), (9, 9, :copy))
        @test grid_coeff_shape == (7, 18, 9)
        @test nyquist == (1, 7)
        unscaled, _, none = Tarang.backward_resample_rules(bases, (9, 12, 9), (16, 12, 9), Float64)
        @test all(rule -> rule[3] === :copy, unscaled) && none == (0, 0)

        source = randn(ComplexF64, 9, 12, 9)
        one_pass = Tarang._resample_fourier_spectrum!(zeros(ComplexF64, 7, 18, 9), source, rules)
        first_axis = Tarang._resample_spectrum_axis!(zeros(ComplexF64, 7, 12, 9), source,
                                                     1, 16, 12, true)
        both_axes = Tarang._resample_spectrum_axis!(zeros(ComplexF64, 7, 18, 9), first_axis,
                                                    2, 12, 18, false)
        @test one_pass ≈ both_axes
        @test_throws DimensionMismatch Tarang._resample_fourier_spectrum!(
            zeros(ComplexF64, 7, 18, 8), source, rules)
    end
end
