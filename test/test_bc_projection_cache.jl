using Test, Tarang, MPI

@testset "Boundary Fourier projections share unexpanded arrays" begin
    nx, ny = 8, 6
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; comm=MPI.COMM_SELF)
    bases = (RealFourier(coords["x"]; size=nx, bounds=(0.0, 2pi)),
             RealFourier(coords["y"]; size=ny, bounds=(0.0, 2pi)),
             ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0)))
    u = ScalarField(dist, "u", bases, Float64)
    problem = InitialValueProblem([u])
    # Projection needs the problem geometry and mode, without any assembled
    # PDE matrices. Each subproblem reads the same boundary array identity.
    subproblem(kx, ky) = Tarang.Subproblem((; problem), (), (kx, ky, nothing))
    cache = problem.compiled.caches.bc_rfft
    x_profile = reshape(cos.(2pi .* (0:nx-1) ./ nx), nx, 1)
    y_profile = reshape(sin.(2pi .* (0:ny-1) ./ ny), 1, ny)

    for (profile, varies_x) in ((x_profile, true), (y_profile, false))
        Tarang.invalidate_bc_array_cache!(problem)
        for ky in 0:ny-1, kx in 0:nx÷2
            expected = if varies_x
                kx == 1 && ky == 0 ? nx*ny/2 : 0.0
            elseif kx == 0 && ky == 1
                -im * nx*ny/2
            elseif kx == 0 && ky == ny-1
                im * nx*ny/2
            else
                0.0
            end
            @test Tarang._bc_array_projection(profile, subproblem(kx, ky)) ≈ expected atol=2e-13
        end
        @test length(cache) == 1
        @test haskey(cache, profile)
        @test length(cache[profile]) == 1
        coefficients = cache[profile][(nx, ny)]
        @test Tarang._get_or_compute_bc_array_coeffs!(profile, subproblem(0, 0), [nx, ny]) === coefficients
        @test size(profile) == (varies_x ? (nx, 1) : (1, ny))
    end

    @testset "geometry, identity, and refresh distinguish cached transforms" begin
        Tarang.invalidate_bc_array_cache!(problem)
        sp = subproblem(1, 0)
        first_coefficients = Tarang._get_or_compute_bc_array_coeffs!(x_profile, sp, [nx, ny])
        wider_coefficients = Tarang._get_or_compute_bc_array_coeffs!(x_profile, sp, [nx, ny+2])
        @test first_coefficients[2, 1] ≈ nx*ny/2
        @test wider_coefficients[2, 1] ≈ nx*(ny+2)/2
        @test size(wider_coefficients) == (nx÷2+1, ny+2)
        @test length(cache) == 1
        @test length(cache[x_profile]) == 2
        @test Tarang._get_or_compute_bc_array_coeffs!(x_profile, sp, [nx, ny]) === first_coefficients

        other_profile = copy(x_profile)
        other_coefficients = Tarang._get_or_compute_bc_array_coeffs!(other_profile, sp, [nx, ny])
        @test length(cache) == 2
        @test other_coefficients !== first_coefficients
        @test other_coefficients == first_coefficients

        x_profile .*= 2
        Tarang.invalidate_bc_array_cache!(problem)
        @test isempty(cache)
        @test Tarang._bc_array_projection(x_profile, sp) ≈ nx*ny atol=2e-13
        @test cache[x_profile][(nx, ny)] !== first_coefficients
    end
end
