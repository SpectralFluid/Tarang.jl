using Test, Tarang, MPI, FFTW, LinearAlgebra, Random

# Execute the CUDA pipeline's actual control flow and pack/unpack kernels on
# host arrays. This checks buffer ownership and uneven MPI layouts, not CUDA
# allocation statistics, streams, cuFFT work areas, or NCCL transport.
module GPUMemoryEmulation
using Tarang: GPU, CPU, ScalarField, RealFourier, ComplexFourier, ChebyshevT
using Tarang: get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!
import Tarang
using MPI, FFTW, LinearAlgebra, KernelAbstractions
const CuArray = Array
struct CuDevice end
module CUDA
using KernelAbstractions
zeros(T, dims...) = Base.zeros(T, dims...)
context() = :host_emulation
synchronize() = KernelAbstractions.synchronize(KernelAbstractions.CPU())
end
const CUFFT = FFTW
CUDABackend() = KernelAbstractions.CPU()
_current_device_id() = 0
ensure_device!(arch) = nothing
struct GPUFFTPlanDim{P,I}
    plan::P
    iplan::I
    is_real::Bool
end
function plan_gpu_fft_dim(arch, shape, T, dim; real_input=false)
    input = zeros(T, shape)
    plan = real_input ? plan_rfft(input, (dim,)) : plan_fft(input, (dim,))
    GPUFFTPlanDim(plan, inv(plan), real_input)
end
gpu_fft_dim!(out, input, plan) = mul!(out, plan.plan, input)
gpu_ifft_dim!(out, input, plan; destroy_input=false) = mul!(out, plan.iplan, input)
function gpu_dct1_along_dim!(out, input, dim, direction)
    n = size(input, dim)
    n <= 1 && return copyto!(out, input)
    function component(x)
        x = copy(x)
        if direction === :backward
            selectdim(x, dim, 1) .*= 2
            selectdim(x, dim, n) .*= 2
        end
        if direction === :backward
            for k in 2:2:n
                selectdim(x, dim, k) .*= -1
            end
        end
        y = FFTW.r2r(x, FFTW.REDFT00, (dim,))
        if direction === :forward
            y ./= n-1
            selectdim(y, dim, 1) .*= 0.5
            selectdim(y, dim, n) .*= 0.5
            for k in 2:2:n
                selectdim(y, dim, k) .*= -1
            end
        else
            y ./= 2
        end
        y
    end
    out .= eltype(input) <: Complex ? complex.(component(real.(input)), component(imag.(input))) : component(input)
    out
end
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "ext/cuda/mixed_transforms.jl"))
include(joinpath(ROOT, "ext/cuda/pencil.jl"))
include(joinpath(ROOT, "ext/cuda/nccl_transpose.jl"))
include(joinpath(ROOT, "ext/cuda/dct_distributed.jl"))
function nccl_alltoall!(send::Array, recv::Array, sc::Vector{Int}, rc::Vector{Int},
                         sd::Vector{Int}, rd::Vector{Int}, comm::MPI.Comm; my_rank::Int)
    MPI.Alltoallv!(MPI.VBuffer(send, sc, sd), MPI.VBuffer(recv, rc, rd), comm)
end
# CuArray reinterpret shares GPU storage directly; host Array reinterpret is
# a ReinterpretArray, which uses the same MPI wire-count contract.
function nccl_alltoall!(send::Base.ReinterpretArray, recv::Base.ReinterpretArray,
                         sc::Vector{Int}, rc::Vector{Int}, sd::Vector{Int}, rd::Vector{Int},
                         comm::MPI.Comm; my_rank::Int)
    # MPI.jl accepts Array storage but not a host ReinterpretArray pointer.
    # Transport copies are harness-only; production CuArray reinterpret aliases.
    wire_send, wire_recv = copy(send), copy(recv)
    MPI.Alltoallv!(MPI.VBuffer(wire_send, sc, sd), MPI.VBuffer(wire_recv, rc, rd), comm)
    copyto!(recv, wire_recv)
end
function buffer(pencil, T)
    n = maximum(prod, (pencil.x_pencil_shape, pencil.y_pencil_shape, pencil.z_pencil_shape))
    peers = maximum(pencil.proc_grid)
    comms = Tarang.NCCLSubComms()
    comms.initialized = true
    comms.row_comm, comms.col_comm = pencil.row_comm, pencil.col_comm
    NCCLTransposeBuffer{T}(zeros(T,n), zeros(T,n),
        [zeros(Int,peers) for _ in 1:10]..., zeros(Int,2peers), comms, pencil)
end
function distributed_plan(shape, mesh, kinds, T)
    pencil = PencilDecomposition(shape, mesh, MPI.Comm_rank(MPI.COMM_WORLD), MPI.COMM_WORLD)
    coeff_shape = (kinds[1] === :real_fourier ? shape[1]÷2+1 : shape[1], shape[2:3]...)
    cp = build_coeff_pencil(pencil, coeff_shape)
    work = [zeros(Complex{T}, s) for s in (pencil.x_pencil_shape, pencil.y_pencil_shape, pencil.z_pencil_shape)]
    DistributedDCTPlan{T}(pencil, cp, kinds, buffer(pencil,Complex{T}), work,
        Dict{NTuple{3,Int},Array{Complex{T},3}}(), Dict{Tuple{Int,Symbol},Any}())
end
# Load the actual field-buffer and C2C branch logic without requiring a CUDA
# device; only its FFT/scratch providers are replaced by host implementations.
function definition(ex, name)
    ex isa Expr || return nothing
    if ex.head === :function
        sig = ex.args[1]
        sig isa Expr && sig.head === :where && (sig=sig.args[1])
        sig isa Expr && sig.head === :call && sig.args[1] === name && return ex
    end
    for arg in ex.args
        found = definition(arg,name)
        found === nothing || return found
    end
    nothing
end
const scratch_requests = Tuple[]
function get_gpu_dct_scratch(arch, shape, T, count)
    push!(scratch_requests, (shape,T,count))
    [zeros(T,shape) for _ in 1:count]
end
get_gpu_fft_plan(arch, shape, T; real_input=false) = plan_ifft(zeros(T,shape))
gpu_backward_fft!(out,input,plan) = mul!(out,plan,input)
const field_plans = IdDict{Any,Any}()
get_or_create_distributed_dct_plan(field) = field_plans[field]
const transforms = Meta.parseall(read(joinpath(ROOT,"ext/cuda/transforms.jl"),String))
for name in (:_gpu_backward_c2c_fft!, :distributed_gpu_forward_transform!, :distributed_gpu_backward_transform!)
    Core.eval(@__MODULE__, definition(transforms,name))
end
end

MPI.Initialized() || MPI.Init()
const _GM = GPUMemoryEmulation
const _GM_NP = MPI.Comm_size(MPI.COMM_WORLD)

@testset "GPU lazy scratch control flow (host emulation)" begin
    arch = GPU{_GM.CuDevice}(_GM.CuDevice())
    coords = CartesianCoordinates("x","z")
    for T in (Float64,ComplexF64)
        dist = Distributor(coords; dtype=T, comm=MPI.COMM_SELF)
        bases = (ComplexFourier(coords["x"];size=7), ComplexFourier(coords["z"];size=9))
        f = ScalarField(Domain(dist,bases), "c2c")
        input = rand(ComplexF64,size(get_coeff_data(f)))
        empty!(_GM.scratch_requests)
        _GM._gpu_backward_c2c_fft!(f,arch,input,size(input))
        @test get_grid_data(f) ≈ (T <: Real ? real.(ifft(input)) : ifft(input))
        @test length(_GM.scratch_requests) == (T <: Real ? 1 : 0)
        T <: Real && @test only(_GM.scratch_requests)[3] == 1
        close(dist)
    end
    for T in (Float64,ComplexF64), mixed in (false,true), scaled in (false,true)
        bases = (mixed ? RealFourier(coords["x"];size=7) : ChebyshevT(coords["x"];size=7),
                 ChebyshevT(coords["z"];size=9))
        shape = scaled ? (7,13) : (7,9)
        plan = _GM.plan_gpu_mixed_transform(arch,bases,shape,T)
        _GM.clear_gpu_mixed_transform_cache!()
        input = rand(T,shape)
        output = zeros(mixed || T <: Complex ? ComplexF64 : Float64,plan.coeff_shape)
        recovered = zeros(T,shape)
        _GM.gpu_mixed_forward_transform!(output,input,plan)
        _GM.gpu_mixed_backward_transform!(recovered,output,plan)
        if !scaled
            @test recovered ≈ input
        end
        # After a truncating transform the represented field is a projection;
        # forward/backward again must preserve that projected field exactly.
        projected = copy(recovered)
        _GM.gpu_mixed_forward_transform!(output,recovered,plan)
        _GM.gpu_mixed_backward_transform!(recovered,output,plan)
        @test recovered ≈ projected
        for scratch in values(_GM.GPU_MIXED_SCRATCH_CACHE)
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
    end
end

@testset "Distributed CUDA pipeline reuse ($_GM_NP ranks, host kernels/MPI)" begin
    mesh = _GM_NP == 4 ? (2,2) : (1,_GM_NP)
    shape = (9,11,13) # Uneven in both decomposition directions.
    for kinds in ((:chebyshev,:chebyshev,:chebyshev),
                  (:complex_fourier,:chebyshev,:complex_fourier),
                  (:real_fourier,:chebyshev,:chebyshev))
        plan = _GM.distributed_plan(shape,mesh,kinds,Float64)
        pencil = plan.pencil
        # The public convenience API still uses buffer.pencil; production may
        # instead supply plan.coeff_pencil to the shared count implementation.
        for (direction,source,target,comm) in (
                (:z_to_y,pencil.z_pencil_shape,pencil.y_pencil_shape,pencil.row_comm),
                (:y_to_z,pencil.y_pencil_shape,pencil.z_pencil_shape,pencil.row_comm),
                (:y_to_x,pencil.y_pencil_shape,pencil.x_pencil_shape,pencil.col_comm),
                (:x_to_y,pencil.x_pencil_shape,pencil.y_pencil_shape,pencil.col_comm))
            _GM.compute_transpose_counts!(plan.transpose_buffer,direction)
            peers=MPI.Comm_size(comm)
            sc=plan.transpose_buffer.send_counts[1:peers]
            rc=plan.transpose_buffer.recv_counts[1:peers]
            @test sum(sc)==prod(source)
            @test sum(rc)==prod(target)
            @test plan.transpose_buffer.send_displs[1:peers]==cumsum([0;sc[1:end-1]])
            @test plan.transpose_buffer.recv_displs[1:peers]==cumsum([0;rc[1:end-1]])
        end
        @test_throws ErrorException _GM.compute_transpose_counts!(plan.transpose_buffer,:invalid)
        # Independent global data/oracle, sliced by the pencil's block ownership.
        full = rand(MersenneTwister(123),shape...)
        xrank,yrank = pencil.grid_coords
        function block(n,p,r)
            q,rem = divrem(n,p)
            first = r*q + min(r,rem) + 1
            first:(first+q+(r<rem)-1)
        end
        rx=block(shape[1],mesh[1],xrank); ry=block(shape[2],mesh[2],yrank)
        local_grid = copy(full[rx,ry,:])
        coefficients = zeros(ComplexF64,plan.coeff_pencil.z_pencil_shape)
        recovered = similar(local_grid)
        _GM.distributed_forward_dct!(coefficients,local_grid,plan)
        # CPU oracle applies this driver's axis order and matching half spectrum.
        oracle = complex.(full)
        for dim in 3:-1:1
            if kinds[dim] === :chebyshev
                tmp=similar(oracle); _GM.gpu_dct1_along_dim!(tmp,oracle,dim,:forward); oracle=tmp
            else
                oracle=fft(oracle,(dim,))
            end
        end
        kinds[1] === :real_fourier && (oracle=oracle[1:(shape[1]÷2+1),:,:])
        crx=block(size(oracle,1),mesh[1],xrank)
        @test coefficients ≈ oracle[crx,ry,:] atol=1e-10
        original_coeff=copy(coefficients)
        _GM.distributed_backward_dct!(recovered,coefficients,plan)
        @test recovered ≈ local_grid atol=1e-10
        @test coefficients == original_coeff
        buffers=copy(plan.scratch); fftplans=copy(plan.fft_plans)
        chunk_plans=copy(plan.transpose_buffer.chunk_plans)
        wire_plans=copy(plan.transpose_buffer.wire_plans)
        _GM_NP > 1 && @test !isempty(chunk_plans)
        _GM_NP > 1 && @test !isempty(wire_plans)
        _GM.distributed_forward_dct!(coefficients,local_grid,plan)
        _GM.distributed_backward_dct!(recovered,coefficients,plan)
        @test keys(plan.scratch) == keys(buffers)
        @test all(plan.scratch[k] === v for (k,v) in buffers)
        @test all(plan.fft_plans[k] === v for (k,v) in fftplans)
        @test keys(plan.transpose_buffer.chunk_plans) == keys(chunk_plans)
        @test all(plan.transpose_buffer.chunk_plans[k] === v for (k,v) in chunk_plans)
        @test keys(plan.transpose_buffer.wire_plans) == keys(wire_plans)
        @test all(plan.transpose_buffer.wire_plans[k] === v for (k,v) in wire_plans)
        @test recovered ≈ local_grid atol=1e-10
        if kinds[1] !== :real_fourier
            complex_grid=complex.(local_grid,local_grid.^2)
            preserved=copy(complex_grid)
            complex_recovered=similar(complex_grid)
            _GM.distributed_forward_dct!(coefficients,complex_grid,plan)
            _GM.distributed_backward_dct!(complex_recovered,coefficients,plan)
            @test complex_recovered ≈ complex_grid atol=1e-10
            @test complex_grid == preserved
        end
        if _GM_NP == 1 && all(==(:chebyshev), kinds)
            for T in (Float64, ComplexF64)
                coords = CartesianCoordinates("x","y","z")
                dist = Distributor(coords; dtype=T, comm=MPI.COMM_SELF)
                bases = ntuple(d -> ChebyshevT(coords[d];size=shape[d]),3)
                template = ScalarField(Domain(dist,bases), "reuse")
                # Distributed GPU coefficients are complex even for a real
                # Chebyshev field; use that storage contract in host emulation.
                storage = Tarang.SerialFieldStorage(CPU(), zeros(T,shape), zeros(ComplexF64,shape))
                f = ScalarField(template, storage)
                get_grid_data(f) .= T.(local_grid)
                _GM.field_plans[f] = plan
                _GM.distributed_gpu_forward_transform!(f)
                _GM.distributed_gpu_backward_transform!(f)
                g,c = get_grid_data(f),get_coeff_data(f)
                _GM.distributed_gpu_forward_transform!(f)
                _GM.distributed_gpu_backward_transform!(f)
                @test get_grid_data(f) === g
                @test get_coeff_data(f) === c
                @test g ≈ local_grid atol=1e-10
                delete!(_GM.field_plans,f)
                close(dist)
            end
        end
        # Standalone public transposes continue returning owned arrays.
        _GM.set_orientation!(pencil,:z_pencil)
        source=complex.(local_grid)
        first=Tarang.transpose_z_to_y!(plan.transpose_buffer,source,pencil)
        _GM.set_orientation!(pencil,:z_pencil)
        second=Tarang.transpose_z_to_y!(plan.transpose_buffer,source,pencil)
        @test first !== second
        @test first == second
        @test source == complex.(local_grid)
        _GM.free_pencil_decomposition!(pencil)
    end
end
