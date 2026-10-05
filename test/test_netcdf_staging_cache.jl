using Test, Tarang, MPI
MPI.Initialized() || MPI.Init()

@testset "NetCDF reusable output staging" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; comm=MPI.COMM_SELF, mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
    u = ScalarField(dist, "staging_u", (basis,), Float64)
    x = (0:7) .* (2pi/8)
    fine_x = (0:15) .* (2pi/16)
    mktempdir() do dir
        h = NetCDFFileHandler(joinpath(dir, "scaled"), dist, Dict("u" => u);
                              staging_max_entries=2)
        Tarang.add_task!(h, u; name="base")
        Tarang.add_task!(h, u; name="fine", scales=2)
        retained = Any[]
        Tarang.add_task!(h, u; name="callback", postprocess=data -> (push!(retained, data); data .*= 3))
        entry = nothing
        for record in 1:3
            # Replace storage as well as contents; cached host data must refresh.
            Tarang.set_grid_data!(u, record .+ cos.(x))
            u.current_layout = :g
            @test Tarang.process!(h; iteration=record, sim_time=0.1record)
            @test isempty(h._staging_cache.cpu_cache)
            @test get_grid_data(u) ≈ record .+ cos.(x)
            @test u.scales == (1.0,)
            current = only(values(h._staging_cache.rescaled))
            if entry !== nothing
                @test current.work === entry.work
                @test current.input_grid === entry.input_grid
                @test current.output_grid === entry.output_grid
            end
            entry = current
            file = Tarang.current_file(h)
            @test vec(Tarang.group_ncread(file, "vars", "base")[record, :]) ≈ record .+ cos.(x)
            @test vec(Tarang.group_ncread(file, "vars", "fine")[record, :]) ≈ record .+ cos.(fine_x) atol=2e-12
            @test vec(Tarang.group_ncread(file, "vars", "callback")[record, :]) ≈ 3 .* (record .+ cos.(x))
        end
        @test retained[1] !== retained[2]
        @test retained[1] ≈ 3 .* (1 .+ cos.(x))
        @test h._staging_cache.rescaled_bytes + h._staging_cache.buffer_bytes <= h._staging_cache.max_bytes
        @test length(h._staging_cache.rescaled) + length(h._staging_cache.buffers) <= 2
        close(h)
        @test isempty(h._staging_cache.rescaled)
        @test isempty(h._staging_cache.buffers)
        @test h._staging_cache.rescaled_bytes == h._staging_cache.buffer_bytes == 0

        h = NetCDFFileHandler(joinpath(dir, "coefficients"), dist, Dict("u" => u); precision=Float32)
        Tarang.add_task!(h, u; name="coeff", layout=:c)
        packed = nothing
        for record in 1:2
            grid_data!(u) .= record .+ cos.(x)
            ensure_layout!(u, :c)
            expected = copy(get_coeff_data(u))
            @test Tarang.process!(h; iteration=record, sim_time=0.1record)
            current = only(values(h._staging_cache.buffers))
            packed === nothing || @test current === packed
            packed = current
            written = Tarang.group_ncread(Tarang.current_file(h), "vars", "coeff")
            @test eltype(written) == Float32
            @test complex.(written[record, 1, :], written[record, 2, :]) ≈ expected rtol=1e-6
            @test get_coeff_data(u) == expected
        end
        close(h)
    end

    @testset "retention budget and precision conversion" begin
        cache = Tarang.NetCDFStagingCache(; max_bytes=128, max_entries=2)
        a = Tarang._output_buffer!(cache, :a, Float64, (8,))
        @test Tarang._output_buffer!(cache, :a, Float64, (8,)) === a
        Tarang._output_buffer!(cache, :b, Float64, (8,))
        Tarang._output_buffer!(cache, :c, Float64, (8,))
        @test length(cache.buffers) <= 2
        @test cache.buffer_bytes <= 128
        oversized = Tarang._output_buffer!(cache, :large, Float64, (32,))
        @test all(value !== oversized for value in values(cache.buffers))
        task = Dict{String, Any}("postprocess" => nothing)
        data = collect(1.0:8.0)
        @test first(Tarang._postprocess_task_data(task, data, Float64, cache)) === data
        first_output = first(Tarang._postprocess_task_data(task, data, Float32, cache))
        data .+= 10
        second_output = first(Tarang._postprocess_task_data(task, data, Float32, cache))
        @test first_output === second_output
        @test second_output == Float32.(data)
        empty!(cache)
        @test cache.buffer_bytes == cache.rescaled_bytes == 0
        @test isempty(cache.buffers) && isempty(cache.rescaled)
        @test_throws ArgumentError Tarang.NetCDFStagingCache(; max_bytes=-1)
        @test_throws ArgumentError Tarang.NetCDFStagingCache(; max_entries=-1)
        for limits in ((0, 2), (128, 0))
            disabled = Tarang.NetCDFStagingCache(; max_bytes=limits[1], max_entries=limits[2])
            Tarang._output_buffer!(disabled, :empty, Float64, (0,))
            Tarang._output_buffer!(disabled, :nonempty, Float64, (8,))
            @test isempty(disabled.buffers)
            scaled = Tarang._refresh_scaled_output!(disabled, u, (2.0,))
            ensure_layout!(scaled, :g)
            @test get_grid_data(scaled) ≈ 2 .+ cos.(fine_x) atol=2e-12
            @test isempty(disabled.rescaled)
        end
    end
    close(dist)
end

@testset "NetCDF component scratch refresh" begin
    domain = PeriodicDomain(4, 4)
    velocity = VectorField(domain, "velocity")
    stress = TensorField(domain, "stress")
    mktempdir() do dir
        handler = NetCDFFileHandler(joinpath(dir, "components"), domain.dist,
                                    Dict("v" => velocity, "s" => stress))
        Tarang.add_task!(handler, velocity; name="velocity")
        Tarang.add_task!(handler, stress; name="stress")
        buffers = nothing
        for record in 1:2
            for (index, component) in enumerate(velocity.components)
                grid_data!(component) .= 10record + index
            end
            for i in axes(stress.components, 1), j in axes(stress.components, 2)
                grid_data!(stress.components[i, j]) .= 100record + 10i + j
            end
            @test Tarang.process!(handler; iteration=record, sim_time=Float64(record))
            file = Tarang.current_file(handler)
            vdata = Tarang.group_ncread(file, "vars", "velocity")
            sdata = Tarang.group_ncread(file, "vars", "stress")
            for index in eachindex(velocity.components)
                @test all(vdata[record, index, :, :] .== 10record + index)
            end
            for i in axes(stress.components, 1), j in axes(stress.components, 2)
                @test all(sdata[record, i, j, :, :] .== 100record + 10i + j)
            end
            current = handler._staging_cache.buffers
            @test length(current) == 2
            buffers === nothing || @test all(current[key] === value for (key, value) in buffers)
            buffers = copy(current)
        end
        close(handler)
    end
    close(domain.dist)
end

@testset "NetCDF device host scratch refresh" begin
    jlarrays_available = try
        @eval using JLArrays, GPUArrays
        true
    catch
        false
    end
    if !jlarrays_available
        @test_skip "JLArrays unavailable"
    else
        # Match the native CUDA extension's device-array trait, including views.
        @eval Tarang.is_gpu_array(::JLArrays.JLArray) = true
        @eval Tarang.is_gpu_array(::SubArray{T,N,<:JLArrays.JLArray}) where {T,N} = true
        GPUArrays.allowscalar(false)
        cache = Tarang.NetCDFStagingCache()
        device_data = JLArrays.JLArray(reshape(collect(1.0:16.0), 4, 4))
        host = Tarang._stage_cpu_array!(cache, :device, device_data)
        @test host == Array(device_data)
        device_data .+= 20
        refreshed = Tarang._stage_cpu_array!(cache, :device, device_data)
        @test refreshed === host
        @test host == Array(device_data)
        view_data = view(device_data, 2:3, :)
        staged_view = Tarang._stage_cpu_array!(cache, :view, view_data)
        @test staged_view == Array(view_data)
        device_data .*= 2
        @test Tarang._stage_cpu_array!(cache, :view, view_data) === staged_view
        @test staged_view == Array(view_data)
    end
end
