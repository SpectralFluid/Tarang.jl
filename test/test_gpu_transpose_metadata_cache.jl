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
_transform_cache(cache) = cache
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
const DeviceRuntime = CUDA
const DeviceArray = Array
const source = Meta.parseall(read(joinpath(root,"ext/gpu/transpose.jl"),String))
Core.eval(@__MODULE__, definition(source, :_gpu_find_rank))
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
# The DistributedGPUFFT pack/unpack overrides, renamed so they never extend Tarang.
import Tarang
rename(ex, from, to) = ex == from ? to :
    ex isa Expr ? Expr(ex.head, map(a -> rename(a, from, to), ex.args)...) : ex
function qualified_definition(ex, name)
    ex isa Expr || return nothing
    ex.head === :function && ex.args[1] isa Expr && ex.args[1].head === :call &&
        ex.args[1].args[1] == name && return ex
    for arg in ex.args
        found = qualified_definition(arg, name)
        found === nothing || return found
    end
    nothing
end
for (from, to) in ((:(Tarang._gpu_pack_for_transpose!), :distfft_pack!),
                   (:(Tarang._gpu_unpack_from_transpose!), :distfft_unpack!))
    Core.eval(@__MODULE__, rename(qualified_definition(source, from), from, to))
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

@testset "DistributedGPUFFT transposes with empty slabs ($P virtual ranks)" for (gshape, P, d) in (
        ((9, 11, 13), 4, 3), ((5, 6, 2), 3, 3), ((2, 5, 7), 3, 3), ((1, 7, 5), 3, 1),
        ((3, 4, 5), 4, 1), ((7, 2), 3, 2), ((6, 6, 1), 2, 3))
    # Production count functions and pack/unpack overrides on host arrays; the
    # MPI_Alltoallv between the P ranks is replayed here with explicit displs.
    # A slab with more ranks than points along `d` (or along the other split axis
    # after the transpose) is EMPTY: its counts divided by zero and that rank
    # threw while its peers waited. A decomposed axis of length 1 also needs the
    # direction passed explicitly — the shape cannot tell forward from reverse.
    MPI.Initialized() || MPI.Init()
    N = length(gshape)
    other = d == N ? 1 : N
    block(n, p, r) = (q = div(n, p); m = mod(n, p); f = r*q + min(r, m) + 1; f:(f + q + (r < m) - 1))
    slab(axis, r) = ntuple(i -> i == axis ? block(gshape[i], P, r) : (1:gshape[i]), N)
    G = reshape(ComplexF64.(1:prod(gshape)), gshape) .* (1 + 0.5im)
    configs = map(0:P-1) do r
        c = DistributedGPUConfig(MPI.COMM_SELF, gshape; decomp_dim=d, cuda_aware_mpi=true)
        c.rank, c.size = r, P
        c.local_shape = compute_local_shape(gshape, d, P, r)
        c
    end
    function alltoallv(sends, plans)
        recvs = [zeros(ComplexF64, sum(plan[2])) for plan in plans]
        for q in 1:P, p in 1:P
            n = plans[p][1][q]
            @test n == plans[q][2][p]      # every send meets a receive of its size
            src = plans[p][3][q]; dst = plans[q][4][p]
            recvs[q][dst+1:dst+n] .= sends[p][src+1:src+n]
        end
        recvs
    end
    data = [G[slab(d, r)...] for r in 0:P-1]
    @test count(isempty, data) == count(r -> isempty(block(gshape[d], P, r)), 0:P-1)
    tdims = [Tarang._compute_transposed_dims(size(data[i]), d, configs[i]) for i in 1:P]
    forward = [Tarang._compute_alltoall_counts(size(data[i]), d, configs[i]) for i in 1:P]
    sends = map(1:P) do i
        buf = zeros(ComplexF64, max(length(data[i]), prod(tdims[i])))
        _GTC.distfft_pack!(buf, data[i], d, configs[i]; reverse=false)
    end
    recvs = alltoallv(sends, forward)
    transposed = map(1:P) do i
        _GTC.distfft_unpack!(zeros(ComplexF64, tdims[i]), recvs[i], d, configs[i]; reverse=false)
    end
    @test all(transposed[r+1] == G[slab(other, r)...] for r in 0:P-1)

    reverse_plans = [Tarang._compute_alltoall_counts_reverse(tdims[i], d, configs[i]) for i in 1:P]
    sends = map(1:P) do i
        buf = zeros(ComplexF64, max(length(data[i]), prod(tdims[i])))
        _GTC.distfft_pack!(buf, transposed[i], d, configs[i]; reverse=true)
    end
    recvs = alltoallv(sends, reverse_plans)
    restored = map(1:P) do i
        _GTC.distfft_unpack!(zeros(ComplexF64, size(data[i])), recvs[i], d, configs[i]; reverse=true)
    end
    @test restored == data
end

@testset "DistributedGPUFFT transforms on the data's device, never the rank's" begin
    MPI.Initialized() || MPI.Init()
    config = DistributedGPUConfig(MPI.COMM_SELF, (8, 6, 4))
    @test config.device_id === nothing          # follow each array's own device
    @test DistributedGPUConfig(MPI.COMM_SELF, (8, 6, 4); device_id=1).device_id == 1
    for rank in 0:5                              # the rank plays no part
        config.rank = rank
        @test Tarang._distributed_fft_device_id(2, config.device_id) == 2
    end
    @test Tarang._distributed_fft_device_id(1, 1) == 1
    @test_throws ArgumentError Tarang._distributed_fft_device_id(1, 0)
    # The CUDA local transforms select the device from the data, not the config rank.
    utils = Meta.parseall(read(joinpath(dirname(@__DIR__), "ext", "cuda", "utils.jl"), String))
    for name in (:local_fft_dim!, :local_ifft_dim!)
        definition = _GTC.qualified_definition(utils, Expr(:., :Tarang, QuoteNode(name)))
        @test definition !== nothing
        code = string(definition)               # parsed code: comments are gone
        @test occursin("CUDA.device(data)", code)
        @test occursin("_distributed_fft_device_id", code)
        @test !occursin("CuDevice(", code)
    end
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
