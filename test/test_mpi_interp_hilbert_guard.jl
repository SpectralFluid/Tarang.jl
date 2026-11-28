# Fourier slices are replicated across ranks. Hilbert still requires a local axis.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI interp/Hilbert guard test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

@testset "Fourier slices and Hilbert on a decomposed axis (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    samples = PencilArrays.global_view(gd)
    for I in CartesianIndices(samples)
        x, y = (2π / 16) .* (Tuple(I) .- 1)
        samples[I] = cos(x+y) + sin(x)*cos(8y)
    end

    # Confirm a Fourier axis really is decomposed under MPI.
    ensure_layout!(u, :c)
    cd = get_coeff_data(u)
    decomp = cd isa PencilArrays.PencilArray ?
             collect(PencilArrays.decomposition(PencilArrays.pencil(cd))) : Int[]
    @test !isempty(decomp)

    x0 = 1.2
    y = (2π / 16) .* (0:15)
    expected_y = cos.(x0 .+ y) .+ sin(x0) .* cos.(8y)
    x_slice = interpolate(u, coords["x"], x0)
    @test evaluate(x_slice, :g) ≈ expected_y atol=2e-13
    @test evaluate(x_slice, :c) ≈ Tarang.FFTW.fft(expected_y) atol=2e-12

    # At half a y grid spacing the real Nyquist cosine vanishes. Every rank
    # receives the same x slice, including its phase in the half spectrum.
    y0 = π / 16
    expected = cos.((2π / 16) .* (0:15) .+ y0)
    slice = interpolate(u, coords["y"], y0)
    @test evaluate(slice, :g) ≈ expected atol=2e-13
    @test evaluate(slice, :c) ≈ Tarang.FFTW.rfft(expected) atol=2e-12

    # Hilbert on a decomposed Fourier axis must not silently produce a wrong field.
    @test_throws Exception Tarang.evaluate(Tarang.hilbert(u), :g)
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
