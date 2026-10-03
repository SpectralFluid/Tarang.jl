using Test, Tarang, MPI, FFTW, Random
try
    using CUDA
catch
end
const _GM_HAS_CUDA = (@isdefined CUDA) && CUDA.functional()
if !_GM_HAS_CUDA
    @testset "GPU transform memory (requires CUDA)" begin
        @test_skip "CUDA is not functional on this host"
    end
else
    MPI.Initialized() || MPI.Init()
    # Honor per-rank visibility; otherwise map local MPI ranks across visible
    # devices before any allocation. Common launchers expose a local-rank key.
    local_rank = let rank=MPI.Comm_rank(MPI.COMM_WORLD)
        for name in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID")
            if haskey(ENV,name)
                rank=parse(Int,ENV[name])
                break
            end
        end
        rank
    end
    CUDA.device!(local_rank % length(CUDA.devices()))
    CUDA.allowscalar(false)
    const _GM_EXT = Base.get_extension(Tarang,:TarangCUDAExt)
    @testset "GPU C2C avoids unused staging grids" begin
        coords=CartesianCoordinates("x","y")
        dist=Distributor(coords;dtype=ComplexF64,device=GPU(),comm=MPI.COMM_SELF)
        bases=(ComplexFourier(coords["x"];size=17),ComplexFourier(coords["y"];size=19))
        f=ScalarField(Domain(dist,bases),"c2c_memory")
        original=rand(ComplexF64,17,19)
        copyto!(grid_data!(f),original)
        _GM_EXT.clear_gpu_dct_scratch_cache!()
        for _ in 1:3
            ensure_layout!(f,:c); ensure_layout!(f,:g)
        end
        @test isempty(_GM_EXT._GPU_DCT_SCRATCH_CACHE)
        CUDA.synchronize()
        before=CUDA.alloc_stats.alloc_bytes
        ensure_layout!(f,:c); ensure_layout!(f,:g)
        CUDA.synchronize()
        @test CUDA.alloc_stats.alloc_bytes == before
        @test Array(grid_data!(f)) ≈ original
        close(dist)
    end
    @testset "GPU mixed scratch is allocated only when used" begin
        coords=CartesianCoordinates("x","z")
        for T in (Float64,ComplexF64), mixed in (false,true), scaled in (false,true)
            _GM_EXT.clear_gpu_mixed_transform_cache!()
            bases=(mixed ? RealFourier(coords["x"];size=9) : ChebyshevT(coords["x"];size=9),
                   ChebyshevT(coords["z"];size=11))
            shape=scaled ? (9,15) : (9,11)
            plan=_GM_EXT.get_gpu_mixed_transform_plan(GPU(),bases,shape,T)
            input=CUDA.rand(T,shape...)
            output=CUDA.zeros(mixed || T <: Complex ? ComplexF64 : Float64,plan.coeff_shape...)
            recovered=similar(input)
            for _ in 1:3
                _GM_EXT.gpu_mixed_forward_transform!(output,input,plan)
                _GM_EXT.gpu_mixed_backward_transform!(recovered,output,plan)
            end
            !scaled && @test Array(recovered) ≈ Array(input) atol=1e-11
            projected=copy(recovered)
            _GM_EXT.gpu_mixed_forward_transform!(output,recovered,plan)
            _GM_EXT.gpu_mixed_backward_transform!(recovered,output,plan)
            @test Array(recovered) ≈ Array(projected) atol=1e-11
            for scratch in values(_GM_EXT.GPU_MIXED_SCRATCH_CACHE)
                if !mixed && T <: Real
                    @test scratch.complex_a === nothing
                    @test scratch.complex_b === nothing
                end
                if T <: Complex || mixed
                    @test scratch.real_input === nothing
                    @test scratch.real_output === nothing
                elseif !scaled
                    @test scratch.real_input === nothing
                end
            end
            CUDA.synchronize()
            before=CUDA.alloc_stats.alloc_bytes
            _GM_EXT.gpu_mixed_forward_transform!(output,input,plan)
            _GM_EXT.gpu_mixed_backward_transform!(recovered,output,plan)
            CUDA.synchronize()
            @test CUDA.alloc_stats.alloc_bytes == before
        end
    end
    @testset "Distributed complex DCT uses packed reusable workspace" begin
        for dim in 1:3
            input=CUDA.rand(ComplexF64,9,11,13)
            original=copy(input)
            output=similar(input); recovered=similar(input)
            for _ in 1:3
                _GM_EXT.local_dct1_along_dim!(output,input,dim,:forward)
                _GM_EXT.local_dct1_along_dim!(recovered,output,dim,:backward)
            end
            @test Array(recovered) ≈ Array(input) atol=1e-11
            @test Array(input) == Array(original)
            CUDA.synchronize()
            before=CUDA.alloc_stats.alloc_bytes
            _GM_EXT.local_dct1_along_dim!(output,input,dim,:forward)
            _GM_EXT.local_dct1_along_dim!(recovered,output,dim,:backward)
            CUDA.synchronize()
            @test CUDA.alloc_stats.alloc_bytes == before
        end
    end
    @testset "GPU derivative scratch and stream context" begin
        for T in (Float64,ComplexF64), axis in 1:3
            host=rand(T,7,9,11)
            matrix=rand(T,size(host,axis),size(host,axis))
            input=CuArray(host); D=CuArray(matrix)
            expected=mapslices(v -> matrix*v,host;dims=axis)
            Tarang._apply_1d_matrix!(input,D,axis,nothing)
            @test Array(input) ≈ expected
            workspace=Tarang._diff_matmul_workspace(input,axis)
            @test Tarang._diff_matmul_workspace(input,axis) === workspace
            @test workspace[1] isa CuArray
            @test workspace[2] isa CuArray
            CUDA.synchronize()
            before=CUDA.alloc_stats.alloc_bytes
            copyto!(input,host)
            Tarang._apply_1d_matrix!(input,D,axis,nothing)
            CUDA.synchronize()
            @test CUDA.alloc_stats.alloc_bytes == before
            token=Tarang._diff_matmul_cache_token(input)
            stream=CUDA.CuStream()
            CUDA.stream!(stream) do
                @test Tarang._diff_matmul_cache_token(input) != token
                @test Tarang._diff_matmul_workspace(input,axis) !== workspace
                copyto!(input,host)
                Tarang._apply_1d_matrix!(input,D,axis,nothing)
                CUDA.synchronize()
            end
            @test Array(input) ≈ expected
            @test Tarang._diff_matmul_cache_token(input) == token
        end
    end
    # Run under mpiexec on GPU workers to exercise real NCCL communication.
    # The host-emulated companion covers 1/2/4-rank uneven pack/unpack arithmetic.
    MPI.Initialized() || MPI.Init()
    @testset "Distributed GPU transform buffers and allocation reuse" begin
        if Tarang.nccl_available() || Tarang._try_load_nccl()
            np=MPI.Comm_size(MPI.COMM_WORLD)
            mesh=np == 4 ? (2,2) : (1,np)
            shape=(9,11,13)
            for kinds in ((:chebyshev,:chebyshev,:chebyshev),
                          (:complex_fourier,:chebyshev,:complex_fourier),
                          (:real_fourier,:chebyshev,:chebyshev))
                pencil=_GM_EXT.PencilDecomposition(shape,mesh,MPI.Comm_rank(MPI.COMM_WORLD),MPI.COMM_WORLD)
                plan=_GM_EXT._build_distributed_dct_plan(pencil,kinds,Float64)
                try
                    input=CUDA.rand(Float64,pencil.z_pencil_shape...)
                    output=CUDA.zeros(ComplexF64,plan.coeff_pencil.z_pencil_shape...)
                    recovered=similar(input)
                    for _ in 1:3
                        _GM_EXT.distributed_forward_dct!(output,input,plan)
                        _GM_EXT.distributed_backward_dct!(recovered,output,plan)
                    end
                    @test Array(recovered) ≈ Array(input) atol=1e-10
                    refs=copy(plan.scratch)
                    CUDA.synchronize()
                    before=CUDA.alloc_stats.alloc_bytes
                    _GM_EXT.distributed_forward_dct!(output,input,plan)
                    _GM_EXT.distributed_backward_dct!(recovered,output,plan)
                    CUDA.synchronize()
                    @test CUDA.alloc_stats.alloc_bytes == before
                    @test all(plan.scratch[k] === v for (k,v) in refs)
                    @test Array(recovered) ≈ Array(input) atol=1e-10
                finally
                    _GM_EXT.finalize_distributed_dct_plan!(plan)
                end
            end
        else
            @test_skip "NCCL.jl is required for the native distributed pipeline"
        end
    end
end
