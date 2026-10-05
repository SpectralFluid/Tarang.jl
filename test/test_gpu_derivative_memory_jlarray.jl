using Test, Tarang, LinearAlgebra
const _GDM_HAS_JL = try
    @eval using JLArrays, GPUArrays
    true
catch
    false
end
@testset "3D derivative workspace on JLArray" begin
    if !_GDM_HAS_JL
        @test_skip "JLArrays/GPUArrays unavailable"
    else
        GPUArrays.allowscalar(false)
        for T in (Float64,ComplexF64), axis in 1:3
            host=rand(T,7,9,11)
            matrix=rand(T,size(host,axis),size(host,axis))
            input=JLArrays.JLArray(host)
            D=JLArrays.JLArray(matrix)
            expected=mapslices(v -> matrix*v,host;dims=axis)
            Tarang._apply_1d_matrix!(input,D,axis,nothing)
            @test Array(input) ≈ expected
            workspace=Tarang._diff_matmul_workspace(input,axis)
            @test workspace[1] isa JLArrays.JLArray
            @test workspace[2] isa JLArrays.JLArray
            copyto!(input,host)
            Tarang._apply_1d_matrix!(input,D,axis,nothing)
            @test Tarang._diff_matmul_workspace(input,axis) === workspace
            @test Array(input) ≈ expected
        end
    end
end

@testset "JLArray field copies own both buffers" begin
    if !_GDM_HAS_JL
        @test_skip "JLArrays/GPUArrays unavailable"
    else
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}) = JLArrays.JLArray
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}, ::Type{T}) where T = JLArrays.JLArray{T}
        GPUArrays.allowscalar(false)
        coords=CartesianCoordinates("x")
        dist=Distributor(coords;device=GPU(JLArrays.JLBackend()))
        xb=RealFourier(coords["x"];size=9)
        for bases in ((),(xb,)), layout in (:g,:c), clone in (copy,deepcopy)
            field=ScalarField(dist,"copy",bases,Float64)
            field.current_layout=layout
            fill!(get_grid_data(field),2.0)
            fill!(get_coeff_data(field),3.0+im)
            live=layout === :g ? get_grid_data(field) : get_coeff_data(field)
            original=Array(live)
            copied=clone(field)
            @test get_grid_data(copied) isa JLArrays.JLArray
            @test get_coeff_data(copied) isa JLArrays.JLArray
            @test get_grid_data(copied) !== get_grid_data(field)
            @test get_coeff_data(copied) !== get_coeff_data(field)
            @test copied.current_layout === layout
            copied_live=layout === :g ? get_grid_data(copied) : get_coeff_data(copied)
            @test Array(copied_live) == original
            fill!(copied_live,9)
            @test Array(live) == original
        end
        close(dist)
    end
end
