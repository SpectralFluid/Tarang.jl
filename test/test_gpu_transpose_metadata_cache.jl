using Test, Tarang, MPI
import KernelAbstractions

# Execute the production cache/launcher code and KA kernels with host storage.
# This proves metadata reuse and numerical indexing; native allocation/stream
# behavior is checked separately below when CUDA hardware is available.
module GPUTransposeCacheEmulation
using Tarang: GPU
using KernelAbstractions
const CuArray = Array
module CUDA
using KernelAbstractions
const current_context = Ref(0)
const allocation_count = Ref(0)
context() = current_context[]
device(args...) = 0
device!(device) = nothing
function zeros(T, dims...)
    allocation_count[] += 1
    Base.zeros(T, dims...)
end
synchronize() = KernelAbstractions.synchronize(KernelAbstractions.CPU())
end
CUDABackend() = KernelAbstractions.CPU()
const uploads = Ref(0)
_to_gpu(x::Vector{Int}) = (uploads[] += 1; copy(x))
const _GPU_DCT_SCRATCH_CACHE = Dict{Tuple,Any}()
const _GPU_DCT_PLAN_CACHE_LOCK = ReentrantLock()
_dct_cache_device_id(arch) = 0

function definition(ex, name)
    ex isa Expr || return nothing
    if ex.head === :function
        sig = ex.args[1]
        sig isa Expr && sig.head === :where && (sig = sig.args[1])
        sig isa Expr && sig.head === :call && sig.args[1] === name && return ex
    end
    for arg in ex.args
        found = definition(arg, name)
        if found !== nothing
            # Keep @kernel around the selected production kernel definition.
            return ex.head === :macrocall && ex.args[1] === Symbol("@kernel") ? ex : found
        end
    end
    nothing
end
const root = dirname(@__DIR__)
const source = Meta.parseall(read(joinpath(root,"ext/cuda/transpose_kernels.jl"),String))
const nccl_source = Meta.parseall(read(joinpath(root,"ext/cuda/nccl_transpose.jl"),String))
Core.eval(@__MODULE__, definition(nccl_source, :_gpu_find_rank))
for name in (:pack_for_transpose_kernel_2d!, :pack_for_transpose_kernel_3d!,
             :unpack_from_transpose_kernel_2d!, :unpack_from_transpose_kernel_3d!,
             :_validated_chunk_size, :_gpu_transpose_chunk_plan,
             :gpu_pack_for_transpose!, :gpu_unpack_from_transpose!)
    Core.eval(@__MODULE__, definition(source,name))
end
const dct_source = Meta.parseall(read(joinpath(root,"ext/cuda/dct.jl"),String))
for name in (:get_gpu_dct_scratch, :clear_gpu_dct_scratch_cache!)
    Core.eval(@__MODULE__, definition(dct_source,name))
end
end
const _GTC = GPUTransposeCacheEmulation

@testset "Purpose-separated scratch retains only requested grids" begin
    arch = Tarang.GPU{Nothing}(nothing)
    _GTC.clear_gpu_dct_scratch_cache!()
    for T in (Float32,ComplexF64), shape in ((5,7),(5,7,9))
        legacy = _GTC.get_gpu_dct_scratch(arch,shape,T,1)
        fft = _GTC.get_gpu_dct_scratch(arch,shape,T,1;purpose=:transpose_fft)
        dct = _GTC.get_gpu_dct_scratch(arch,shape,T,1;purpose=:transpose_dct)
        @test length(legacy) == length(fft) == length(dct) == 1
        @test legacy[1] !== fft[1] !== dct[1]
        @test legacy[1] !== dct[1]
        @test _GTC.get_gpu_dct_scratch(arch,shape,T,1;purpose=:transpose_fft) === fft
        @test _GTC.get_gpu_dct_scratch(arch,shape,T,1;purpose=:transpose_dct) === dct
        @test haskey(_GTC._GPU_DCT_SCRATCH_CACHE,(0,shape,T,1))
    end
    @test _GTC.CUDA.allocation_count[] == 12
    _GTC.clear_gpu_dct_scratch_cache!()
    @test isempty(_GTC._GPU_DCT_SCRATCH_CACHE)
end

@testset "GPU transpose metadata reuse and numerical geometry" begin
    for T in (Float32,ComplexF64), shape in ((5,7),(5,7,9)), dim in eachindex(shape)
        data = reshape(T.(1:prod(shape)),shape)
        chunks = [1,0,shape[dim]-1] # Uneven peers, including an empty peer.
        counts = chunks .* (prod(shape) ÷ shape[dim])
        displs = cumsum([0;counts[1:end-1]])
        cache = Dict{Tuple,Any}()
        packed = zeros(T,length(data)); restored = similar(data)
        before = _GTC.uploads[]
        _GTC.gpu_pack_for_transpose!(packed,data,counts,displs,dim,3;metadata_cache=cache)
        # Independent rank-contiguous oracle; a copy-only pack cannot pass.
        expected = vcat(vec(copy(selectdim(data,dim,1:1))),
                        vec(copy(selectdim(data,dim,2:shape[dim]))))
        @test packed == expected
        _GTC.gpu_unpack_from_transpose!(restored,packed,counts,displs,dim,3;metadata_cache=cache)
        @test restored == data
        @test _GTC.uploads[] - before == 6
        retained = copy(cache)
        data .*= T(2)
        _GTC.gpu_pack_for_transpose!(packed,data,counts,displs,dim,3;metadata_cache=cache)
        _GTC.gpu_unpack_from_transpose!(restored,packed,counts,displs,dim,3;metadata_cache=cache)
        @test restored == data
        @test _GTC.uploads[] - before == 6
        @test all(cache[k] === v for (k,v) in retained)
        # Another workspace never borrows the first workspace's device vectors.
        other = Dict{Tuple,Any}()
        second = _GTC._gpu_transpose_chunk_plan(shape,counts,displs,dim,3,:pack,other)
        @test second[1] !== _GTC._gpu_transpose_chunk_plan(shape,counts,displs,dim,3,:pack,cache)[1]
    end
end

@testset "Geometry/context invalidation and bounded lifetime" begin
    cache = Dict{Tuple,Any}()
    plan(shape,counts,displs;cache=cache) =
        _GTC._gpu_transpose_chunk_plan(shape,counts,displs,2,length(counts),:pack,cache)
    counts = [4,8]; displs = [0,4]
    original = plan((2,6),counts,displs)
    @test original[1] == [2,4]
    counts .= [6,6]; displs .= [0,6]
    changed = plan((2,6),counts,displs)
    @test changed[1] == [3,3]
    @test changed[2] == displs
    @test original[1] == [2,4] # Cached buffers are immutable after publication.
    counts .= [4,8]; displs .= [0,4]
    @test plan((2,6),counts,displs) === original
    @test plan((4,3),counts,displs)[1] == [1,2] # Same element count, different shape.
    _GTC.CUDA.current_context[] = 1
    @test plan((2,6),counts,displs)[1] !== original[1]
    _GTC.CUDA.current_context[] = 0
    @test plan((2,6),counts,displs) === original
    @test_throws ArgumentError plan((3,4),counts,displs)
    for n in 1:40
        plan((2,n),[2n],[0])
        @test length(cache) <= 32
    end
    # Public async launchers retain their original uncached temporary path.
    empty!(cache)
    _GTC.gpu_pack_for_transpose!(zeros(12),ones(2,6),[4,8],[0,4],2,2;
                                 synchronize=false,metadata_cache=cache)
    _GTC.CUDA.synchronize()
    @test isempty(cache)
    # Empty ranks do not create metadata or launch a kernel.
    @test _GTC.gpu_pack_for_transpose!(Float64[],zeros(0,6),[0,0],[0,0],2,2;
                                      metadata_cache=cache) == Float64[]
    @test isempty(cache)
end

@testset "Workspace cache teardown and fallback errors" begin
    MPI.Initialized() || MPI.Init()
    coords = CartesianCoordinates("x","y")
    dist = Distributor(coords;dtype=ComplexF64,comm=MPI.COMM_SELF)
    f = ScalarField(Domain(dist,(ComplexFourier(coords[1];size=5),ComplexFourier(coords[2];size=7))),"meta")
    tf = TransposableField(f)
    tf.buffers.gpu_metadata[(:probe,)] = ones(3)
    close(tf)
    @test isempty(tf.buffers.gpu_metadata)
    @test_throws ErrorException Tarang.pack_for_transpose!(zeros(4),ones(2,2),[4],[0],2,1,
        Tarang.GPU{Nothing}(nothing);metadata_cache=Dict())
    @test_throws ErrorException Tarang.unpack_from_transpose!(zeros(2,2),ones(4),[4],[0],2,1,
        Tarang.GPU{Nothing}(nothing);metadata_cache=Dict())
    close(dist)
end

const _GTC_NATIVE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
@testset "Native CUDA transpose cache allocations" begin
    if !_GTC_NATIVE
        @test_skip "requires native CUDA hardware"
    else
        CUDA.allowscalar(false)
        ext = Base.get_extension(Tarang,:TarangCUDAExt)
        for T in (Float32,ComplexF64), shape in ((5,7),(5,7,9)), dim in eachindex(shape)
            original = reshape(T.(1:prod(shape)),shape)
            data=CuArray(original); packed=CUDA.zeros(T,length(data)); restored=similar(data)
            counts = [1,0,shape[dim]-1] .* (prod(shape)÷shape[dim])
            displs = cumsum([0;counts[1:end-1]])
            cache = Dict{Tuple,Any}()
            for _ in 1:2
                ext.gpu_pack_for_transpose!(packed,data,counts,displs,dim,3;metadata_cache=cache)
                ext.gpu_unpack_from_transpose!(restored,packed,counts,displs,dim,3;metadata_cache=cache)
            end
            CUDA.synchronize()
            before=CUDA.alloc_stats.alloc_bytes
            ext.gpu_pack_for_transpose!(packed,data,counts,displs,dim,3;metadata_cache=cache)
            ext.gpu_unpack_from_transpose!(restored,packed,counts,displs,dim,3;metadata_cache=cache)
            CUDA.synchronize()
            @test CUDA.alloc_stats.alloc_bytes == before
            @test Array(restored) == original
        end
        ext.clear_gpu_dct_scratch_cache!()
        data = CUDA.rand(ComplexF64,5,7,9)
        arch = Tarang.architecture(data)
        original = Array(data)
        for dim in 1:3
            Tarang.fft_in_dim!(data,dim,:forward,arch)
            Tarang.fft_in_dim!(data,dim,:backward,arch)
            @test Array(data) ≈ original
            Tarang.dct_in_dim!(data,dim,:forward,arch)
            Tarang.dct_in_dim!(data,dim,:backward,arch)
            @test Array(data) ≈ original
        end
        for purpose in (:transpose_fft,:transpose_dct)
            entry = ext.get_gpu_dct_scratch(arch,size(data),eltype(data),1;purpose)
            @test length(entry) == 1
            @test any(k -> length(k)==5 && last(k)===purpose,keys(ext._GPU_DCT_SCRATCH_CACHE))
        end
    end
end
