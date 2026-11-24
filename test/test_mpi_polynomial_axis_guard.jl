using Test
using MPI
MPI.Init()
using Tarang

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)

# The interpreted Chebyshev derivative is a DCT-I along its axis, so it is only correct on a
# rank that holds the whole axis. Plain-array distributed storage (GPU+MPI, or
# use_pencil_arrays=false on CPU) is the TransposableField ZLocal slab, whose LEADING axes are
# decomposed. The guard used to recognise only PencilArrays, so a split Chebyshev axis was
# differentiated slab by slab with no error: dx(x^3) was off by 2.35 (of 3) at np=2.
function guard_field(name, bases)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(nprocs, 1), dtype=ComplexF64, architecture=CPU(),
                       use_pencil_arrays=false)
    made = map(enumerate(bases)) do (i, (kind, n))
        coord = coords[("x", "y", "z")[i]]
        kind === :cheb ? ChebyshevT(coord; size=n, bounds=(-1.0, 1.0)) :
                         ComplexFourier(coord; size=n, bounds=(0.0, 2π))
    end
    u = ScalarField(dist, name, Tuple(made), ComplexF64)
    grids = map(g -> vec(collect(g)), local_grids(dist, made...))
    return coords, u, grids
end

@testset "polynomial derivative refuses a split axis on plain-array storage (np=$nprocs)" begin
    # Chebyshev on the trailing axis, which the ZLocal slab keeps whole: correct derivative.
    coords, u, (_, _, z) = guard_field("u_local", ((:fourier, 8), (:fourier, 8), (:cheb, 12)))
    ensure_layout!(u, :g)
    g = get_grid_data(u)
    g .= complex.(ones(size(g, 1), size(g, 2)) .* reshape(z, 1, 1, :) .^ 3)
    dz = get_grid_data(evaluate(Differentiate(u, coords["z"], 1), :g))
    @test maximum(abs.(dz .- 3 .* reshape(z, 1, 1, :) .^ 2)) < 1e-10

    # Chebyshev on the leading axis, which the ZLocal slab splits across ranks.
    coords, v, (x, _, _) = guard_field("v_split", ((:cheb, 16), (:fourier, 8), (:fourier, 8)))
    ensure_layout!(v, :g)
    h = get_grid_data(v)
    h .= complex.(reshape(x, :, 1, 1) .^ 3 .* ones(1, size(h, 2), size(h, 3)))
    if nprocs > 1
        @test size(h, 1) < 16
        @test_throws ErrorException evaluate(Differentiate(v, coords["x"], 1), :g)
    else
        dx = get_grid_data(evaluate(Differentiate(v, coords["x"], 1), :g))
        @test maximum(abs.(dx .- 3 .* reshape(x, :, 1, 1) .^ 2)) < 1e-10
    end
end
