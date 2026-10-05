using Test, Tarang, MPI, PencilArrays

MPI.Initialized() || MPI.Init()
const _GX_COMM = MPI.COMM_WORLD
const _GX_NP = MPI.Comm_size(_GX_COMM)
const _GX_TO = PencilArrays.TimerOutputs
_GX_TO.enable_debug_timings(PencilArrays.Transpositions)

function _gx_calls(timer, label)
    function count_calls(node)
        total = 0
        for (name, child) in node["inner_timers"]
            total += name == label ? child["n_calls"] : count_calls(child)
        end
        total
    end
    count_calls(_GX_TO.todict(timer))
end

@testset "Packed MPI field exchanges ($_GX_NP ranks)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU(), mesh=(_GX_NP,))
    topology = PencilArrays.MPITopology(_GX_COMM, (_GX_NP,))
    source_pencil = PencilArrays.Pencil(topology, (11, 19), (2,))
    destination_pencil = PencilArrays.Pencil(source_pencil; decomp_dims=(1,),
                                            permute=PencilArrays.Permutation(2, 1))
    for T in (Float64, ComplexF32), extra in ((), (2,))
        sources = [PencilArrays.PencilArray{T}(undef, source_pencil, extra...) for _ in 1:3]
        destinations = [PencilArrays.PencilArray{T}(undef, destination_pencil, extra...) for _ in 1:3]
        expected = [similar(a) for a in destinations]
        for (i, source) in enumerate(sources)
            parent(source) .= reshape(T.(1:length(source)), size(parent(source))) .+ 1000i .+ 100MPI.Comm_rank(_GX_COMM)
        end
        originals = copy.(parent.(sources))
        timer = PencilArrays.timer(source_pencil)
        _GX_TO.reset_timer!(timer)
        for i in eachindex(sources)
            PencilArrays.transpose!(expected[i], sources[i]; method=PencilArrays.Transpositions.Alltoallv())
        end
        individual_calls = _gx_calls(timer, "MPI.Alltoallv!")
        _GX_TO.reset_timer!(timer)
        Tarang.group_pencil_transpose!(destinations, sources, dist)
        grouped_calls = _gx_calls(timer, "MPI.Alltoallv!")
        @test parent.(destinations) == parent.(expected)
        @test parent.(sources) == originals
        @test grouped_calls == 1
        @test individual_calls == 3grouped_calls
        workspace = only(values(Tarang._GROUPED_PENCIL_TRANSPOSE_CACHE[dist]))
        Tarang.group_pencil_transpose!(destinations, sources, dist)
        @test workspace in values(Tarang._GROUPED_PENCIL_TRANSPOSE_CACHE[dist])
        cache = Tarang._GROUPED_PENCIL_TRANSPOSE_CACHE[dist]
        for i in 1:(Tarang._GROUPED_PENCIL_TRANSPOSE_CACHE_CAP - length(cache))
            cache[(:capacity_probe, i)] = nothing
        end
        Tarang.group_pencil_transpose!(destinations, sources, dist)
        @test length(cache) == Tarang._GROUPED_PENCIL_TRANSPOSE_CACHE_CAP
        @test workspace in values(cache)
        empty!(cache)
    end
    close(dist)
    @test !haskey(Tarang._GROUPED_PENCIL_TRANSPOSE_CACHE, dist)
end

function _gx_fill!(field, seed)
    data = grid_data!(field)
    ranges = data isa PencilArrays.PencilArray ? PencilArrays.range_local(data) : axes(data)
    values = [sin(0.13sum(Tuple(index))) + seed*cos(0.19sum(Tuple(index)))
              for index in CartesianIndices(Tuple(ranges))]
    T = field.dtype
    if T <: Complex
        values = complex.(values, reverse(values))
    end
    copyto!(Tarang.get_local_data(data), T.(values))
end

@testset "Grouped Fourier and coupled transforms ($_GX_NP ranks)" begin
    for dimensions in (2, 3), mixed in (false, true), T in (Float64, ComplexF32)
        names = dimensions == 2 ? ("z", "x") : ("z", "x", "y")
        coords = CartesianCoordinates(names...)
        mesh = dimensions == 3 && _GX_NP == 4 ? (2, 2) : (_GX_NP,)
        dist = Distributor(coords; dtype=T, device=CPU(), mesh)
        bases = Tuple(mixed && i == 1 ? ChebyshevT(coords[name]; size=9) :
                      (T <: Real ? RealFourier : ComplexFourier)(coords[name]; size=11 + 2i)
                      for (i, name) in enumerate(names))
        domain = Domain(dist, bases)
        fields = [ScalarField(domain, "group_$i") for i in 1:3]
        references = [ScalarField(domain, "reference_$i") for i in 1:3]
        for i in 1:3
            _gx_fill!(fields[i], i)
            copyto!(Tarang.get_local_data(grid_data!(references[i])), Tarang.get_local_data(grid_data!(fields[i])))
        end
        originals = copy.(Tarang.get_local_data.(grid_data!.(fields)))
        foreach(f -> ensure_layout!(f, :c), references)
        Tarang.group_forward_transform!(fields)
        for i in 1:3
            @test Tarang.get_local_data(coeff_data!(fields[i])) ≈ Tarang.get_local_data(coeff_data!(references[i])) rtol=3e-5 atol=3e-5
        end
        Tarang.group_backward_transform!(fields)
        for i in 1:3
            @test Tarang.get_local_data(grid_data!(fields[i])) ≈ originals[i] rtol=3e-5 atol=3e-5
        end
        if _GX_NP > 1
            bundle = Tarang._field_transform_bundle(fields[1])
            workspace = only(values(bundle.pencil_work_cache[:grouped_fft]))
            @test workspace.forward_calls == 1
            @test workspace.backward_calls == 1
            @test workspace.fields_transformed == 6
            @test workspace.collective_exchanges > 0
            Tarang.group_forward_transform!(fields)
            @test workspace === only(values(bundle.pencil_work_cache[:grouped_fft]))
        end
        close(dist)
    end
end

@testset "Grouped transforms separate equal-sized domains" begin
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, device=CPU(), mesh=(_GX_NP,))
    xb = RealFourier(coords["x"]; size=15)
    domains = (Domain(dist, (RealFourier(coords["z"]; size=13), xb)),
               Domain(dist, (ChebyshevT(coords["z"]; size=13), xb)))
    fields = [ScalarField(domains[mod1(i, 2)], "domain_$i") for i in 1:4]
    references = [ScalarField(domains[mod1(i, 2)], "domain_reference_$i") for i in 1:4]
    for i in eachindex(fields)
        _gx_fill!(fields[i], i)
        copyto!(Tarang.get_local_data(grid_data!(references[i])), Tarang.get_local_data(grid_data!(fields[i])))
        ensure_layout!(references[i], :c)
    end
    Tarang.enable_transform_counts!(true)
    Tarang.reset_transform_counts!()
    try
        Tarang.group_forward_transform!(fields)
        if _GX_NP > 1
            @test Tarang.transform_counts().forward == 2
            @test Tarang.transform_counts().coupled_dct == 1
        end
    finally
        Tarang.enable_transform_counts!(false)
    end
    for i in eachindex(fields)
        @test Tarang.get_local_data(coeff_data!(fields[i])) ≈ Tarang.get_local_data(coeff_data!(references[i])) atol=1e-11
    end
    if _GX_NP > 1
        bundles = Tarang._field_transform_bundle.(fields[1:2])
        @test bundles[1] !== bundles[2]
        @test only(values(bundles[1].pencil_work_cache[:grouped_fft])) !==
              only(values(bundles[2].pencil_work_cache[:grouped_fft]))
    end
    close(dist)
    @test_throws ArgumentError Tarang.group_pencil_transpose!(PencilArrays.PencilArray[], PencilArrays.PencilArray[], dist)
end

_GX_TO.disable_debug_timings(PencilArrays.Transpositions)
