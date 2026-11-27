# MPS consumes row-major real matrices and zero-based pivots. Keep both in that
# representation from factorization through every solve.
struct MetalLUSolveWorkspace{F}
    rhs::F
    solution::F
    matrices::NTuple{4,Metal.MPS.MPSMatrix}
    kernel::Metal.MPS.MPSMatrixSolveLU
end

function MetalLUSolveWorkspace(factors, pivots, nrhs)
    m = size(factors, 1)
    rhs = similar(factors, nrhs, m)
    solution = similar(rhs)
    matrices = map(Metal.MPS.MPSMatrix, (factors, rhs, pivots, solution))
    kernel = Metal.MPS.MPSMatrixSolveLU(Metal.device(factors), false, m, nrhs)
    return MetalLUSolveWorkspace(rhs, solution, matrices, kernel)
end

mutable struct MetalDenseLU{D,F,P} <: Tarang.MatSolvers.AbstractMatSolver
    factors::F
    pivots::P
    workspace::Union{Nothing,MetalLUSolveWorkspace{F}}
    lock::ReentrantLock
end

@kernel function _pack_mps_matrix!(block, @Const(A))
    i, j = @index(Global, NTuple)
    @inbounds begin
        v = A[i,j]
        block[2j-1,2i-1] = real(v)
        block[2j,  2i-1] = -imag(v)
        block[2j-1,2i]   = imag(v)
        block[2j,  2i]   = real(v)
    end
end

Metal.@autoreleasepool function MetalDenseLU{D}(matrix::AbstractMatrix; kwargs...) where D
    n = LinearAlgebra.checksquare(matrix)
    dev = only(d for d in Metal.devices() if d.registryID == D)
    arch = GPU(dev)
    ensure_device!(arch)
    block = Metal.MtlMatrix{Float32,Metal.PrivateStorage}(undef, 2n, 2n)
    pivots = Metal.MtlMatrix{UInt32,Metal.PrivateStorage}(undef, 1, 2n)
    if n > 0
        A = matrix isa WrappedDeviceArray ? Tarang.on_architecture(arch, matrix) :
                                           Metal.MtlArray(convert(Matrix{ComplexF32}, matrix))
        launch!(arch, _pack_mps_matrix!, block, A; ndrange=(n,n))
        status = Metal.MPS.status_buffer()
        m = Metal.MPS.MPSMatrix(block)
        cmd = Metal.MTL.MTLCommandBuffer(Metal.global_queue(dev))
        Metal.MPS.encode!(cmd, Metal.MPS.MPSMatrixDecompositionLU(dev, 2n, 2n),
                          m, m, Metal.MPS.MPSMatrix(pivots), status)
        Metal.MTL.commit!(cmd)
        Metal.synchronize(cmd)
        LinearAlgebra.checknonsingular(Int(status[]))
    end
    return MetalDenseLU{D,typeof(block),typeof(pivots)}(block, pivots, nothing, ReentrantLock())
end
MetalDenseLU(matrix::AbstractMatrix; kwargs...) = MetalDenseLU{Metal.device().registryID}(matrix; kwargs...)

Tarang._subproblem_eltype(::MetalGPU) = ComplexF32
Tarang._matsolver_backend(::Type{<:MetalDenseLU}) = :metal
function Tarang._select_ivp_matsolver(choice, arch::MetalGPU, coupled::Bool)
    default = MetalDenseLU{arch.device.registryID}
    selected = Tarang._select_ivp_matsolver(choice, true, coupled; gpu_solver=default, backend=:metal)
    return Tarang._matsolver_backend(selected) === :metal ? default : selected
end

Metal.@autoreleasepool function Tarang.MatSolvers.solve!(dest, solver::MetalDenseLU, rhs)
    arch = Tarang.architecture(solver.factors)
    ensure_device!(arch)
    size(dest) == size(rhs) || throw(DimensionMismatch("Destination and RHS sizes differ"))
    ndims(rhs) in (1,2) || throw(DimensionMismatch("Expected a vector or matrix of right-hand sides"))
    2 * size(rhs, 1) == size(solver.factors, 1) || throw(DimensionMismatch("Invalid Metal LU RHS size"))
    isempty(rhs) && return dest
    local_rhs = rhs isa WrappedDeviceArray ? Tarang.on_architecture(arch, rhs) :
                                            Metal.MtlArray(ComplexF32.(rhs))
    direct = dest isa Metal.MtlArray{ComplexF32} && Metal.device(dest) == arch.device
    work = direct ? dest : Metal.MtlArray{ComplexF32}(undef, size(rhs))
    input = eltype(local_rhs) === ComplexF32 ? local_rhs : ComplexF32.(local_rhs)
    m, nrhs = size(solver.factors, 1), size(rhs, 2)
    lock(solver.lock) do
        ws = solver.workspace
        if isnothing(ws) || size(ws.rhs, 1) != nrhs
            ws = solver.workspace = MetalLUSolveWorkspace(solver.factors, solver.pivots, nrhs)
        end
        # Staging also permits overlapping input and output views.
        permutedims!(ws.rhs, reshape(reinterpret(Float32, input), m, nrhs), (2,1))
        dev = arch.device
        cmd = Metal.MTL.MTLCommandBuffer(Metal.global_queue(dev))
        Metal.MPS.encode!(cmd, ws.kernel, ws.matrices...)
        Metal.MTL.commit!(cmd)
        permutedims!(reshape(reinterpret(Float32, work), m, nrhs), ws.solution, (2,1))
        Metal.synchronize()
    end
    direct || copyto!(dest, Tarang.on_architecture(Tarang.architecture(dest), work))
    return dest
end
function Tarang.MatSolvers.solve(solver::MetalDenseLU, rhs)
    return Tarang.MatSolvers.solve!(similar(rhs, ComplexF32), solver, rhs)
end

function Tarang._subproblem_sparse_matrix(data::Metal.WrappedMtlArray{T,1}, matrix) where T
    ensure_device!(Tarang.architecture(data))
    return Metal.MtlArray(Matrix{ComplexF32}(matrix))
end
