using Test, Tarang, MPI, FFTW

function _bc_memory_fixture(nx=32, ny=48)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; comm=MPI.COMM_SELF)
    bases = (RealFourier(coords["x"]; size=nx, bounds=(0.0, 2pi)),
             RealFourier(coords["y"]; size=ny, bounds=(0.0, 2pi)),
             ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0)))
    problem = InitialValueProblem([ScalarField(dist, "u", bases, Float64)])
    sp = Tarang.Subproblem((; problem), (), (1, 1, nothing))
    return problem, sp, dist
end

function _bc_memory_sample(arr, sp, n)
    value = 0.0im
    for _ in 1:n
        value += Tarang._bc_array_projection(arr, sp)
    end
    return value
end

# Measure inside a specialized call so older Julia versions do not count a
# boxed ComplexF64 return at the testset's dynamically dispatched call site.
_bc_memory_sample_bytes(arr, sp, n) = @allocated _bc_memory_sample(arr, sp, n)

function _bc_memory_refresh(arr, sp, problem, n)
    value = 0.0im
    for _ in 1:n
        Tarang.invalidate_bc_array_cache!(problem)
        value += Tarang._bc_array_projection(arr, sp)
    end
    return value
end

@testset "Boundary projection workspaces and owned refresh values" begin
    problem, sp, dist = _bc_memory_fixture()
    shape = (32, 48)
    x = reshape(cos.(2pi .* (0:31) ./ 32), 32, 1)
    y = reshape(sin.(2pi .* (0:47) ./ 48), 1, 48)
    context = problem.compiled.caches
    coeffs(arr, dims=shape) = Tarang._get_or_compute_bc_array_coeffs!(arr, sp, dims)

    full_x, full_y = repeat(x, 1, shape[2]), repeat(y, shape[1], 1)
    for (arr, plane) in ((x, full_x), (y, full_y), (vec(x), full_x), (vec(y), full_y),
                         (x .+ y, full_x .+ full_y), ([2.0], fill(2.0, shape)),
                         (complex.(x, 2 .* x), (1+2im) .* full_x),
                         (complex.(x .+ y, x .- y), full_x .+ full_y .+ im .* (full_x .- full_y)),
                         (ComplexF64[2+3im], fill(2+3im, shape)))
        original = copy(arr)
        expected = eltype(arr) <: Complex ? FFTW.fft(plane) : FFTW.rfft(plane)
        actual = coeffs(arr)
        @test actual ≈ expected atol=1e-11
        @test actual === coeffs(arr, collect(shape))
        @test arr == original
        @test size(arr) == size(original)
        @test Tarang._bc_array_projection(arr, sp) ≈ expected[2,2] atol=1e-11
    end
    @test coeffs(complex.(x, 2 .* x))[2,1] ≈ (1+2im) * 32*48/2
    @test coeffs(ComplexF64[2+3im])[1,1] ≈ (2+3im) * prod(shape)

    first = coeffs(x)
    snapshot = copy(first)
    scratch = context.workspaces[:bc_fft_scratch][(Float64, shape)]
    other = coeffs(copy(x))
    @test other !== first
    @test other == first
    @test !Base.mightalias(other, first)
    wider = coeffs(x, (32, 50))
    @test wider[2,1] ≈ 32*50/2
    @test first == snapshot
    x .*= 2
    @test coeffs(x) === first # Refresh explicitly controls value lifetime.
    Tarang.invalidate_bc_array_cache!(problem)
    @test isempty(context.bc_rfft)
    refreshed = coeffs(x)
    @test refreshed !== first
    @test !Base.mightalias(refreshed, first)
    @test refreshed ≈ 2 .* snapshot
    @test first == snapshot
    @test context.workspaces[:bc_fft_scratch][(Float64, shape)] === scratch

    # Explicit singleton axes are unambiguous even when both lengths match.
    @test size(coeffs(reshape(collect(1.0:8.0), 1, 8), (8,8))) == (5,8)
    @test_throws ArgumentError Tarang._copy_bc_plane!(zeros(8,8), ones(8))
    @test_throws ArgumentError Tarang._copy_bc_plane!(zeros(shape), ones(7,3))
    @test_throws ArgumentError Tarang._bc_array_projection(ones(8), sp, (8,8), (1,1))
    @test_throws ArgumentError coeffs(ones(7,3))
    @test Tarang._copy_bc_plane!(zeros(6,4), reshape(collect(1:24), 4,6)) ==
          reshape(collect(1:24), 6,4)
    real3 = reshape(cos.(2pi .* (0:7) ./ 8), 1,8,1)
    @test coeffs(real3, (6,8,5)) ≈ FFTW.rfft(ones(6,1,5) .* real3)
    complex3 = complex.(real3, -real3)
    @test coeffs(complex3, (6,8,5)) ≈ FFTW.fft(ones(6,1,5) .* complex3)
    @test Tarang._sample_bc_coefficients(ComplexF64[1,2], (2,)) == 2
    @test Tarang._sample_bc_coefficients(fill(3.0im,2,3,4), (2,3,4)) == 3im
    @test Tarang._sample_bc_coefficients(fill(3.0im,2,3,4), (2,3,5)) == 0
    @test Tarang._sample_bc_coefficients(fill(3.0im,2,3,4), (2,3)) == 0

    # Workspace retention is bounded by geometry/type, never fresh BC identity.
    for n in 2:Tarang._BC_FFT_SCRATCH_CAPACITY+4
        coeffs([1.0], (n,7))
        @test length(context.workspaces[:bc_fft_scratch]) <= Tarang._BC_FFT_SCRATCH_CAPACITY
    end
    @test first == snapshot # Even evicted scratch cannot alter old values.

    _bc_memory_sample_bytes(y, sp, 10)
    _bc_memory_refresh(y, sp, problem, 3)
    sample_bytes = _bc_memory_sample_bytes(y, sp, 1000)
    refresh_bytes = @allocated _bc_memory_refresh(y, sp, problem, 10)
    @test sample_bytes <= 64_000
    # Independently owned refreshed coefficients are intentionally allocated;
    # no second full boundary plane or FFT output temporary should be needed.
    coefficient_bytes = (shape[1] ÷ 2 + 1) * shape[2] * sizeof(ComplexF64)
    @test refresh_bytes <= 10 * (coefficient_bytes + 6144)
    empty!(context)
    @test isempty(context.bc_rfft)
    @test isempty(context.workspaces)
    close(dist)
end
