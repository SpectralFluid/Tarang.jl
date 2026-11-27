# Shared CPU, device, and MPI checks against individual Fourier and Chebyshev modes.
function check_cartesian_products(arch, ::Type{R}, comm) where R
    coords = CartesianCoordinates("z", "x", "y")
    nz, nx, ny = 13, 12, 9
    xs = 2π .* (0:nx-1) ./ nx
    ys = 2π .* (0:ny-1) ./ ny
    cheb(m, j) = cospi(m * (nz-j) / (nz-1))
    for T in (R, Complex{R})
        dist = Distributor(coords; device=arch, dtype=T, comm)
        try
            basis = T <: Real ? RealFourier : ComplexFourier
            domain = Domain(dist, (ChebyshevT(coords["z"]; size=nz, dealias=1.5),
                                   basis(coords["x"]; size=nx), basis(coords["y"]; size=ny)))
            a, b = ScalarField(domain, "a"), ScalarField(domain, "b")
            function assign_grid!(field, values)
                grid = grid_data!(field)
                if grid isa Tarang.PencilArrays.PencilArray
                    local_view = Tarang.PencilArrays.global_view(grid)
                    for I in CartesianIndices(local_view)
                        local_view[I] = values[I]
                    end
                else
                    copyto!(grid, T.(values))
                end
            end
            for degree in (10, 12)
                if T <: Real
                    av = [cheb(degree,j)*(-1)^(i-1) for j in 1:nz, i in 1:nx, y in ys]
                    bv = [cheb(degree,j)*cos(x) for j in 1:nz, x in xs, y in ys]
                    expected = [cos(5x)/4 for j in 1:nz, x in xs, y in ys]
                else
                    av = [cheb(degree,j)*im*(-1)^(i-1)*cis(y) for j in 1:nz, i in 1:nx, y in ys]
                    bv = [cheb(degree,j)*cis(x) for j in 1:nz, x in xs, y in ys]
                    expected = [im*cis(-5x+y)/2 for j in 1:nz, x in xs, y in ys]
                end
                assign_grid!(a, av)
                assign_grid!(b, bv)
                for layout in (:g, :c)
                    p = Tarang.evaluate_transform_multiply(a, b, Tarang._get_evaluator(dist); result_layout=layout)
                    @test p.current_layout == layout
                    grid = grid_data!(p)
                    values = grid isa Tarang.PencilArrays.PencilArray ? Tarang._allgather_global_grid(grid, comm) : Array(grid)
                    @test maximum(abs, values .- expected) < 80eps(R)
                end
            end
        finally
            close(dist)
        end
    end
end

function check_cartesian_operators(arch, ::Type{T}) where T
    domain = PeriodicDomain(9, 12; L=(2π, 2π), arch, dtype=T)
    try
        u = ScalarField(domain, "u")
        coords = domain.dist.coordsys
        xs, ys = 2π .* (0:8) ./ 9, 2π .* (0:11) ./ 12
        copyto!(grid_data!(u), T.(cos.(xs .+ ys')))
        for x0 in (0.0, 0.31)
            slice = interpolate(u, coords["x"], x0)
            expected = cos.(x0 .+ ys)
            @test Array(evaluate(slice, :g)) ≈ expected atol=40eps(T)
            @test Array(evaluate(slice, :c)) ≈ Tarang.FFTW.fft(expected) atol=100eps(T)
        end
        copyto!(grid_data!(u), T.(cos.(xs) .* cos.(6 .* ys')))
        @test maximum(abs, Array(evaluate(interpolate(u, coords["y"], π/12)))) < 40eps(T)

        v = VectorField(domain, "v")
        for (j, component) in enumerate(v.components)
            copyto!(grid_data!(component), T.(j .* cos.(xs .+ ys')))
        end
        set_scales!(v, (1.5, 1.25))
        for layout in (:g, :c)
            result = evaluate(lap(v), layout)
            for (src, dst) in zip(v.components, result.components)
                @test dst.scales == src.scales
                @test maximum(abs, Array(grid_data!(dst)) .+ 2 .* Array(grid_data!(src))) < 600eps(T)
            end
        end
    finally
        close(domain.dist)
    end

    domain = ChannelDomain3D(9, 12, 9; Lx=2π, Ly=2π, Lz=6, arch, dtype=T)
    try
        @test domain.bases[3].meta.bounds == (-3.0, 3.0)
        @test Tarang.volume(domain) ≈ 24π^2
        u = ScalarField(domain, "u")
        set!(u, (x,y,z) -> cos(x+y)*(1+z^2))
        scales = (1.25, 1.5, 1.5)
        set_scales!(u, scales)
        nx, ny, nz = size(grid_data!(u))
        expected = [-2*cospi(2i/nx + 2j/ny) * (3cospi(k/(nz-1)))^2
                    for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1]
        for layout in (:g, :c)
            result = evaluate(lap(u), layout)
            @test result.scales == scales
            @test maximum(abs, Array(grid_data!(result)) .- expected) < (T === Float32 ? 2e-3 : 2e-10)
        end
        product = Tarang.evaluate_transform_multiply(u, u, Tarang._get_evaluator(domain.dist))
        @test product.scales == scales
        @test Array(grid_data!(product)) ≈ Array(grid_data!(u)).^2
    finally
        close(domain.dist)
    end

    # With Fourier padding disabled, only the Chebyshev factor controls projection.
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; device=arch, dtype=T, comm=Tarang.MPI.COMM_SELF)
    try
        bx = RealFourier(coords["x"]; size=7, dealias=1.0)
        for strength in (1.0, 1.5)
            bz = ChebyshevT(coords["z"]; size=13, dealias=strength)
            f = ScalarField(dist, "f", (bx, bz), T)
            data = T[cospi(10j/12) for i in 1:7, j in 0:12]
            copyto!(grid_data!(f), data)
            expected = strength == 1 ? data.^2 : fill(T(0.5), size(data))
            @test maximum(abs, Array(grid_data!(f*f)) .- expected) < 80eps(T)
        end
    finally
        close(dist)
    end
end
