using Test
using Tarang

@testset "Spectral review regressions" begin
    @testset "Convert preserves stored-basis normalization" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords)
        z = coords["z"]
        bases = (Legendre(z; size=8, bounds=(-1.0, 1.0)),
                 ChebyshevT(z; size=8, bounds=(-1.0, 1.0)),
                 ChebyshevU(z; size=8, bounds=(-1.0, 1.0)),
                 Jacobi(z; a=0.0, b=0.0, size=8, bounds=(-1.0, 1.0)),
                 Jacobi(z; a=0.5, b=0.5, size=8, bounds=(-1.0, 1.0)))
        poly(z) = 1 + 0.4z - 0.7z^2 + 0.2z^3
        # Test both conversion directions and the equal-Jacobi-parameter case.
        for (i, j) in ((1, 2), (2, 1), (1, 4), (4, 1), (3, 5), (5, 3))
            f = ScalarField(dist, "f", (bases[i],), Float64)
            f["g"] .= poly.(vec(local_grid(bases[i], dist, 1)))
            converted = evaluate(Convert(f, bases[j]), :g)
            @test converted["g"] ≈ poly.(vec(local_grid(bases[j], dist, 1))) atol=2e-12
            @test evaluate(interpolate(converted, z, 0.3)) ≈ poly(0.3) atol=2e-12
            # Also exercise a warm conversion-matrix cache and coefficient output.
            converted_c = evaluate(Convert(f, bases[j]), :c)
            @test converted_c.current_layout == :c
            @test evaluate(interpolate(converted_c, z, -0.41)) ≈ poly(-0.41) atol=2e-12
        end
    end

    @testset "Scaled Fourier interpolation uses actual FFT lengths" begin
        coords = CartesianCoordinates("x")
        x = coords["x"]
        for N in (7, 8), (basis_type, dtype) in
            ((RealFourier, Float64), (RealFourier, ComplexF64), (ComplexFourier, ComplexF64)),
            initial_layout in (:g, :c)
            dist = Distributor(coords; dtype)
            basis = basis_type(x; size=N, bounds=(0.0, 2pi))
            f = ScalarField(dist, "f", (basis,), dtype)
            fun = dtype <: Real ? (x -> 1 + 0.3cos(2x) - 0.4sin(x)) :
                                  (x -> 1 + cis(-2x) + 0.25im * cis(x))
            f["g"] .= fun.(vec(local_grid(basis, dist, 1)))
            ensure_layout!(f, initial_layout)
            set_scales!(f, 1.5)
            for position in (0.3, 1.7)
                @test evaluate(interpolate(f, x, position)) ≈ fun(position) atol=3e-12
            end
        end

        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        bx = RealFourier(coords["x"]; size=7, bounds=(0.0, 2pi))
        by = RealFourier(coords["y"]; size=8, bounds=(0.0, 2pi))
        f = ScalarField(dist, "f", (bx, by), Float64)
        xs, ys = vec(local_grid(bx, dist, 1)), vec(local_grid(by, dist, 1))
        fun(x, y) = 1 + cos(2x) * sin(y) + 0.2sin(x + y)
        f["g"] .= fun.(xs, ys')
        set_scales!(f, (1.5, 2.0))
        xs, ys = vec(local_grid(bx, dist, 1.5)), vec(local_grid(by, dist, 2.0))
        along_x = evaluate(interpolate(f, coords["x"], 0.3))
        along_y = evaluate(interpolate(f, coords["y"], 0.4))
        @test size(along_x) == (16,)
        @test size(along_y) == (11,)
        @test along_x ≈ fun.(0.3, ys) atol=3e-12
        @test along_y ≈ fun.(xs, 0.4) atol=3e-12
    end

    @testset "Full FFT interpolation uses the negative even Nyquist mode" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=ComplexF64)
        for basis_type in (ComplexFourier, RealFourier), N in (7, 8)
            bx = basis_type(coords["x"]; size=N, bounds=(0.0, 2pi))
            by = basis_type(coords["y"]; size=8, bounds=(0.0, 2pi))
            mode = -(N ÷ 2)
            xs = vec(local_grid(bx, dist, 1))
            ys = vec(local_grid(by, dist, 1))
            f = ScalarField(dist, "f", (bx,), ComplexF64)
            f["g"] .= cis.(mode .* xs)
            @test evaluate(interpolate(f, coords["x"], pi/8)) ≈ cis(mode * pi/8) atol=3e-12
            f2 = ScalarField(dist, "f2", (bx, by), ComplexF64)
            f2["g"] .= cis.(mode .* xs .+ ys')
            @test evaluate(interpolate(f2, coords["x"], pi/8)) ≈
                cis.(mode * pi/8 .+ ys) atol=3e-12
        end
    end

    @testset "Polynomial interpolation preserves remaining scales" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords)
        bx = RealFourier(coords["x"]; size=7, bounds=(0.0, 2pi))
        bz = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
        f = ScalarField(dist, "f", (bx, bz), Float64)
        xs, zs = vec(local_grid(bx, dist, 1)), vec(local_grid(bz, dist, 1))
        f["g"] .= (1 .+ cos.(xs)) .* (1 .+ zs'.^2)
        set_scales!(f, (1.5, 1.5))
        reduced = evaluate(interpolate(f, coords["z"], 0.3))
        xs = vec(local_grid(bx, dist, 1.5))
        @test reduced.scales == (1.5,)
        @test size(reduced["g"]) == (11,)
        @test reduced["g"] ≈ (1 .+ cos.(xs)) .* 1.09 atol=3e-12
    end

    @testset "Scaled derivatives preserve resolution and layout" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        bx = RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
        f = ScalarField(dist, "f", (bx,), Float64)
        f["g"] .= sin.(vec(local_grid(bx, dist, 1)))
        set_scales!(f, 1.5)
        xs = vec(local_grid(bx, dist, 1.5))
        for input_layout in (:g, :c), output_layout in (:g, :c), order in (0, 1, 2)
            ensure_layout!(f, input_layout)
            df = evaluate(Differentiate(f, coords["x"], order), output_layout)
            @test df.current_layout == output_layout
            @test df.scales == f.scales
            expected = order == 0 ? sin.(xs) : order == 1 ? cos.(xs) : -sin.(xs)
            @test df["g"] ≈ expected atol=3e-12
        end
        zero_y = evaluate(Differentiate(f, coords["y"], 1), :c)
        @test zero_y.scales == f.scales
        @test size(zero_y["g"]) == (12,)
        @test all(iszero, zero_y["g"])
        # More calls than the rotating pool length, alternating resolutions,
        # catch stale metadata/buffers when a pooled field is reused.
        small = ScalarField(dist, "small", (bx,), Float64)
        small["g"] .= sin.(vec(local_grid(bx, dist, 1)))
        held = evaluate(Differentiate(f, coords["x"], 1), :c)
        held_coeffs = copy(held["c"])
        for i in 1:34
            # A period of three makes the same slot see both resolutions on
            # successive turns through the 16-slot pool.
            source = i % 3 == 0 ? f : small
            df = evaluate(Differentiate(source, coords["x"], 1))
            @test size(df["g"]) == size(source["g"])
            @test df["g"] ≈ cos.(vec(local_grid(bx, dist, source.scales[1]))) atol=3e-12
        end
        @test held["c"] == held_coeffs
        @test held["g"] ≈ cos.(xs) atol=3e-12

        by = RealFourier(coords["y"]; size=7, bounds=(0.0, 2pi))
        f2 = ScalarField(dist, "f2", (bx, by), Float64)
        xs, ys = vec(local_grid(bx, dist, 1)), vec(local_grid(by, dist, 1))
        f2["g"] .= sin.(xs) .* cos.(ys')
        set_scales!(f2, (1.5, 2.0))
        xs, ys = vec(local_grid(bx, dist, 1.5)), vec(local_grid(by, dist, 2.0))
        grad_f = evaluate(Gradient(f2, coords), :c)
        @test grad_f.components[1]["g"] ≈ cos.(xs) .* cos.(ys') atol=3e-12
        @test grad_f.components[2]["g"] ≈ -sin.(xs) .* sin.(ys') atol=3e-12
        df_vec = evaluate(Differentiate(grad_f, coords["x"], 1), :c)
        @test all(c -> c.current_layout == :c, df_vec.components)
        @test df_vec.components[1]["g"] ≈ -sin.(xs) .* cos.(ys') atol=3e-12
        @test df_vec.components[2]["g"] ≈ -cos.(xs) .* sin.(ys') atol=3e-12

        bz = ChebyshevT(coords["y"]; size=8, bounds=(-1.0, 1.0))
        poly = ScalarField(dist, "poly", (bz,), Float64)
        poly["g"] .= vec(local_grid(bz, dist, 1)).^3
        set_scales!(poly, 1.5)
        zs = vec(local_grid(bz, dist, 1.5))
        for layout in (:g, :c)
            dp = evaluate(Differentiate(poly, coords["y"], 1), layout)
            @test dp.scales == poly.scales
            @test dp["g"] ≈ 3 .* zs.^2 atol=2e-11
        end
    end
end
