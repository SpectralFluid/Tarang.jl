using Test, MPI, Tarang
using Tarang: PencilArrays

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const NP = MPI.Comm_size(COMM)

@testset "Padded-product workspace budget (np=$NP)" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    ns = (16, 16, 8)
    bs = Tuple(RealFourier(coords[c]; size=n, bounds=(0.0, 2pi), dealias=1.5)
               for (c, n) in zip(("x", "y", "z"), ns))
    u = ScalarField(dist, "u", bs, Float64)
    v = ScalarField(dist, "v", bs, Float64)
    ug, vg = grid_data!(u), grid_data!(v)
    @test (ug isa PencilArrays.PencilArray) == (NP > 1)
    ax = ug isa PencilArrays.PencilArray ? PencilArrays.pencil(ug).axes_local : axes(ug)
    x = reshape(2pi .* (collect(ax[1]) .- 1) ./ ns[1], :, 1, 1)
    y = reshape(2pi .* (collect(ax[2]) .- 1) ./ ns[2], 1, :, 1)
    z = reshape(2pi .* (collect(ax[3]) .- 1) ./ ns[3], 1, 1, :)
    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=1.5)
    buffers = nothing
    for rep in 1:4
        parent(ug) .= rep .* sin.(3 .* x) .+ cos.(y) .+ 0 .* z
        parent(vg) .= cos.(2 .* x) .+ sin.(z) .+ 0 .* y
        before_u, before_v = copy(parent(ug)), copy(parent(vg))
        product = Tarang.evaluate_transform_multiply(u, v, ev; result_layout=:g)
        # The fifth x harmonic is resolved; filtering it away would fail this.
        @test parent(grid_data!(product)) ≈ before_u .* before_v atol=1e-11
        @test parent(ug) == before_u
        @test parent(vg) == before_v
        if NP == 1
            # One-rank MPI uses the serial workspace, so no distributed scratch
            # or transpose counters should be created. Keep numerical checks above.
            @test isempty(Tarang._PADDED_DIST_WS_CACHE)
        else
            ws = only(values(Tarang._PADDED_DIST_WS_CACHE))
            retained = sum(prod(PencilArrays.size_global(PencilArrays.pencil(b))) for b in ws.buffers)
            # Previously 43 full complex grids. Reuse spent buffers and reverse the
            # truncation sweep to retain at most 16, including simultaneously live operands.
            @test retained <= 16 * prod(ns)
            if buffers !== nothing
                @test length(buffers) == length(ws.buffers)
                @test all(a === b for (a, b) in zip(buffers, ws.buffers))
            end
            buffers = copy(ws.buffers)
            @test hasproperty(ws, :transpose_count)
            if hasproperty(ws, :transpose_count)
                @test ws.transpose_count == 6
                @test ws.transpose_elements <= 12 * prod(ns)
                @test !any(ws.in_use)
            end
        end
    end
end

@testset "Padded scratch reuse across precisions and complex operands" begin
    for T in (Float32, ComplexF32, ComplexF64)
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; dtype=T, device=CPU())
        basis = T <: Complex ? ComplexFourier : RealFourier
        bs = Tuple(basis(coords[c]; size=12, bounds=(0.0, 2pi), dealias=1.5)
                   for c in ("x", "y", "z"))
        u, v = ScalarField(dist, "u", bs, T), ScalarField(dist, "v", bs, T)
        ug, vg = grid_data!(u), grid_data!(v)
        ax = ug isa PencilArrays.PencilArray ? PencilArrays.pencil(ug).axes_local : axes(ug)
        x = reshape(2pi .* (collect(ax[1]) .- 1) ./ 12, :, 1, 1)
        y = reshape(2pi .* (collect(ax[2]) .- 1) ./ 12, 1, :, 1)
        z = reshape(2pi .* (collect(ax[3]) .- 1) ./ 12, 1, 1, :)
        ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=1.5)
        for rep in 1:3
            phase = T <: Complex ? T(im) : T(1)
            parent(ug) .= rep .* sin.(x) .+ phase .* cos.(y) .+ 0 .* z
            parent(vg) .= cos.(2 .* x) .+ phase .* sin.(z) .+ 0 .* y
            a, b = copy(parent(ug)), copy(parent(vg))
            result = Tarang.evaluate_transform_multiply(u, v, ev; result_layout=:c)
            @test parent(grid_data!(result)) ≈ a .* b rtol=3e-6 atol=3e-6
            @test parent(ug) == a
            @test parent(vg) == b
        end
    end
end
