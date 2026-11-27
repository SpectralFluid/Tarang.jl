# Shared CPU, device, and MPI checks against individual Fourier and Chebyshev modes.
function assign_cartesian_grid!(field, values)
    grid = grid_data!(field)
    if grid isa Tarang.PencilArrays.PencilArray
        owned = Tarang.PencilArrays.global_view(grid)
        for I in CartesianIndices(owned)
            owned[I] = values[I]
        end
    else
        copyto!(grid, field.dtype.(values))
    end
end

function cartesian_grid(field)
    grid = grid_data!(field)
    return grid isa Tarang.PencilArrays.PencilArray ? Tarang._allgather_global_grid(grid, field.dist.comm) : Array(grid)
end

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
                assign_cartesian_grid!(a, av)
                assign_cartesian_grid!(b, bv)
                for layout in (:g, :c)
                    p = Tarang.evaluate_transform_multiply(a, b, Tarang._get_evaluator(dist); result_layout=layout)
                    @test p.current_layout == layout
                    values = cartesian_grid(p)
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
        coords = domain.dist.coordsys
        pos = 0.37
        expected_wall = [(1+pos^2) * cospi(2i/nx + 2j/ny) for i in 0:nx-1, j in 0:ny-1]
        wall = evaluate(interpolate(u, coords["z"], pos), :c)
        @test wall.scales == scales[1:2]
        @test maximum(abs, Array(grid_data!(wall)) .- expected_wall) < 100eps(T)
        expected_tangent = [cos(pos + 2π*j/ny) * (1 + (3cospi(k/(nz-1)))^2)
                            for j in 0:ny-1, k in 0:nz-1]
        @test maximum(abs, Array(evaluate(interpolate(u, coords["x"], pos))) .- expected_tangent) < 200eps(T)
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

function check_mixed_cartesian_fields(arch, ::Type{T}, comm) where T
    for names in (("x", "y"), ("z", "x"), ("z", "x", "y"))
        @testset "Mixed fields on $names" begin
            coords = CartesianCoordinates(names...)
            dist = Distributor(coords; device=arch, dtype=T, comm)
            try
                sizes = Tuple(s == "x" ? 12 : 9 for s in names)
                bases = Tuple(s == "z" ? ChebyshevT(coords[s]; size=n, bounds=(-1., 1.), dealias=1.5) :
                                        RealFourier(coords[s]; size=n, bounds=(0., 2π), dealias=1.5)
                              for (s,n) in zip(names, sizes))
                domain = Domain(dist, bases)
                xaxis = findfirst(==("x"), names)
                xs = [2π*(I[xaxis]-1)/12 for I in CartesianIndices(sizes)]
                a = ScalarField(domain, "a")
                b = ScalarField(domain, "b"; dtype=Complex{T})
                assign_cartesian_grid!(a, cos.(6xs))
                assign_cartesian_grid!(b, cis.(xs))
                for (u,v) in ((a,b), (b,a))
                    for p in (u*v, evaluate(convection(u,v,:multiply), :c))
                        @test p.dtype === Complex{T}
                        @test maximum(abs, cartesian_grid(p) .- cis.(-5xs)./2) < 100eps(T)
                    end
                end
                if names == ("x", "y")
                    problem = InitialValueProblem([b])
                    add_parameters!(problem; a)
                    add_equation!(problem, "∂t(b) = a*b + (a+a)*d(b,x) + d(a,x)*b")
                    solver = InitialValueSolver(problem, RK222(); dt=1e-3)
                    @test solver.rhs_plan.is_compiled
                    rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
                    expected = (0.5 + im) .* cis.(-5xs)
                    @test maximum(abs, cartesian_grid(rhs) .- expected) < 150eps(T)
                end
                velocity = VectorField(domain, "velocity")
                for (j,c) in enumerate(velocity.components)
                    fill!(grid_data!(c), T(j == xaxis))
                end
                assign_cartesian_grid!(b, im .* sin.(xs))
                tracer = evaluate(advection(velocity, b), :c)
                @test maximum(abs, cartesian_grid(tracer) .- im .* cos.(xs)) < 100eps(T)

                if first(names) == "z"
                    # An asymmetric polynomial catches a spurious reversal of
                    # Chebyshev degrees when the real Fourier axis is collapsed.
                    nz = first(sizes)
                    z = [-cospi(j/(nz-1)) for j in 0:nz-1]
                    poly(z) = 1 + z + 0.5*(2z^2 - 1)
                    ys = length(names) == 3 ? 2π .* (0:sizes[3]-1) ./ sizes[3] : [0.0]
                    grid = [poly(z[I[1]]) * cos(xs[I] + ys[length(names) == 3 ? I[3] : 1])
                            for I in CartesianIndices(sizes)]
                    assign_cartesian_grid!(a, grid)
                    pos = 0.31
                    wall = interpolate(a, coords["z"], pos)
                    plane = [poly(pos) * cospi(2i/12 + y/π) for i in 0:11, y in ys]
                    expected = length(names) == 3 ? plane : vec(plane)
                    for layout in (:c, :g)
                        slice = evaluate(wall, layout)
                        @test slice.current_layout == layout
                        @test slice.dist.size == 1
                        @test maximum(abs, cartesian_grid(slice) .- expected) < 100eps(T)
                    end
                    tangent = interpolate(a, coords["x"], pos)
                    yz = poly.(z) .* transpose(cos.(pos .+ ys))
                    modes = zeros(nz)
                    modes[1:3] .= (1, 1, 0.5)
                    if length(names) == 3
                        coeffs = modes .* transpose(Tarang.FFTW.fft(cos.(pos .+ ys)))
                        @test maximum(abs, Array(evaluate(tangent, :c)) .- coeffs) < 300eps(T)
                        @test maximum(abs, Array(evaluate(tangent, :g)) .- yz) < 100eps(T)
                    else
                        @test maximum(abs, Array(evaluate(tangent, :c)) .- modes .* cos(pos)) < 100eps(T)
                        @test maximum(abs, Array(evaluate(tangent, :g)) .- vec(yz)) < 100eps(T)
                    end
                end
            finally
                close(dist)
            end
        end
    end
end

function check_scaled_cartesian_fields(arch, ::Type{T}) where T
    for names in (("x","y"), ("x","z"), ("x","y","z"))
        @testset "Scaled operators on $names" begin
            coords = CartesianCoordinates(names...)
            dist = Distributor(coords; device=arch, dtype=T, comm=Tarang.MPI.COMM_SELF)
            try
                bases = Tuple(s == "z" ? ChebyshevT(coords[s]; size=9) :
                                        RealFourier(coords[s]; size=11, bounds=(0., 2π)) for s in names)
                domain = Domain(dist, bases)
                u, v = ScalarField(domain, "u"), VectorField(domain, "v")
                sizes = size(grid_data!(u))
                wave = [sinpi(2*(I[1]-1)/sizes[1]) for I in CartesianIndices(sizes)]
                assign_cartesian_grid!(u, wave)
                assign_cartesian_grid!(v.components[1], 1 .+ wave)
                assign_cartesian_grid!(v.components[2], wave)
                scales = ntuple(d -> d == 1 ? 1.5 : 1.25, length(bases))
                coeff_data!(u)  # set_scales! may retain the preceding coefficient shape
                set_scales!(u, scales)
                set_scales!(v, scales)
                shape = size(get_grid_data(v.components[1]))
                xs = [2π*(I[1]-1)/shape[1] for I in CartesianIndices(shape)]
                wave, slope = sin.(xs), cos.(xs)
                advected = (1 .+ wave) .* slope
                tol = T === Float32 ? 2e-3 : 5e-11
                function check_field(f, expected, layout)
                    @test f.current_layout == layout
                    @test f.scales == scales
                    @test maximum(abs, cartesian_grid(f) .- expected) < tol
                end
                for layout in (:c,:g)
                    if names == ("x","y")
                        check_field(evaluate(fraclap(u, 0.5), layout), wave, layout)
                        check_field(evaluate(fraclap(v, 0.5), layout).components[1], wave, layout)
                    end
                    check_field(evaluate(CartesianLaplacian(u, coords), layout), .-wave, layout)
                    check_field(evaluate(CartesianLaplacian(v, coords), layout).components[2], .-wave, layout)
                    check_field(evaluate(CartesianDivergence(v), layout), slope, layout)
                    jac = evaluate(CartesianGradient(v, coords), layout)
                    check_field(evaluate(CartesianTrace(jac), layout), slope, layout)
                    vort = evaluate(curl(v), layout)
                    check_field(vort isa VectorField ? vort.components[3] : vort, slope, layout)
                    if length(bases) == 3
                        check_field(evaluate(CartesianCurl(v), layout).components[3], slope, layout)
                    end
                    check_field(evaluate(advection(v, u), layout), advected, layout)
                    check_field(evaluate(nonlinear_momentum(v), layout).components[2], advected, layout)
                end
            finally
                close(dist)
            end
        end
    end
end

function check_longwave_inverse(arch, ::Type{T}, comm) where T
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; device=arch, dtype=T, comm)
    try
        domain = Domain(dist, (RealFourier(coords["x"]; size=12, bounds=(0., 2π*1e8)),
                               RealFourier(coords["y"]; size=9, bounds=(0., 2π))))
        u = ScalarField(domain, "u")
        wave = [cospi(2i/12) for i in 0:11, j in 1:9]
        assign_cartesian_grid!(u, 2 .+ wave)
        op = invsqrtlap(u)
        @test maximum(abs, cartesian_grid(evaluate(op)) ./ 1e8 .- wave) < 60eps(T)
        rhs = ScalarField(domain, "rhs")
        expr = Tarang.translate_to_lazy(op, [u]; target=u)
        Tarang.evaluate_lazy!(rhs, expr, [u], Tarang.LazyWorkspace())
        @test maximum(abs, cartesian_grid(rhs) ./ 1e8 .- wave) < 60eps(T)
        if dist.size == 1
            diagonal = Array(Tarang._diagonal_Lhat_from_expr(op, u))
            @test iszero(diagonal[1,1])
            @test diagonal[2,1] ≈ 1e8 rtol=10eps(T)
            for matrix in (Tarang.subproblem_matrix(op, nothing),
                           Tarang._spectral_fractional_laplacian(u, -0.5, length(diagonal), length(diagonal)))
                @test iszero(matrix[1,1])
                @test matrix[2,2] ≈ 1e8 rtol=10eps(T)
            end
        end
    finally
        close(dist)
    end
end
