# Per-mode stage matrices are assembled on the host. Carry the field's device
# in the selected solver type so later refactorizations target the same device.
struct ROCDenseLU{D} <: Tarang.MatSolvers.AbstractMatSolver
    factors::ROCMatrix{ComplexF64}
    pivots::ROCVector{Cint}
end

function ROCDenseLU{D}(matrix::AbstractMatrix; kwargs...) where D
    LinearAlgebra.checksquare(matrix)
    arch = GPU(AMDGPU.devices()[D])
    ensure_device!(arch)
    if matrix isa AnyROCArray
        factors = ROCMatrix{ComplexF64}(undef, size(matrix))
        factors .= Tarang.on_architecture(arch, matrix)
    else
        factors = ROCArray(convert(Matrix{ComplexF64}, matrix))
    end
    _, pivots, info = AMDGPU.rocSOLVER.getrf!(factors)
    iszero(info) || throw(SingularException(info))
    return ROCDenseLU{D}(factors, pivots)
end
ROCDenseLU(matrix::AbstractMatrix; kwargs...) =
    ROCDenseLU{AMDGPU.device_id()}(matrix; kwargs...)

Tarang._matsolver_backend(::Type{<:ROCDenseLU}) = :amdgpu
function Tarang._select_ivp_matsolver(choice, arch::ROCGPU, coupled::Bool)
    default = ROCDenseLU{AMDGPU.device_id(arch.device)}
    selected = Tarang._select_ivp_matsolver(choice, true, coupled; gpu_solver=default, backend=:amdgpu)
    return Tarang._matsolver_backend(selected) === :amdgpu ? default : selected
end

function Tarang.MatSolvers.solve!(dest, solver::ROCDenseLU, rhs)
    arch = Tarang.architecture(solver.factors)
    ensure_device!(arch)
    size(dest) == size(rhs) || throw(DimensionMismatch("Destination and RHS sizes differ"))
    local_rhs = Tarang.on_architecture(arch, rhs)
    direct = dest isa Union{ROCVector{ComplexF64},ROCMatrix{ComplexF64}} && AMDGPU.device(dest) == arch.device
    work = direct ? dest : ROCArray{ComplexF64}(undef, size(rhs))
    work === local_rhs || copyto!(work, Base.mightalias(work, local_rhs) ? copy(local_rhs) : local_rhs)
    AMDGPU.rocSOLVER.getrs!('N', solver.factors, solver.pivots, work)
    direct || copyto!(dest, Tarang.on_architecture(Tarang.architecture(dest), work))
    return dest
end
function Tarang.MatSolvers.solve(s::ROCDenseLU, rhs)
    dest = similar(rhs, ComplexF64)
    return Tarang.MatSolvers.solve!(dest, s, rhs)
end

function Tarang._subproblem_sparse_matrix(data::AMDGPU.AnyROCVector, matrix)
    ensure_device!(Tarang.architecture(data))
    return AMDGPU.rocSPARSE.ROCSparseMatrixCSR{eltype(matrix)}(sparse(matrix))
end

# rocSOLVER's strided interface operates directly on Tarang's (n,n,mode)
# matrix storage and (n,mode) RHS. Only the per-mode status leaves the GPU.
function Tarang._batched_factor_impl!(s::Tarang.BatchedDenseLU{<:ROCArray})
    s.factored = false
    A = s.A
    ensure_device!(Tarang.architecture(A))
    n, cols, modes = size(A)
    n == cols || throw(DimensionMismatch("Batched LU requires square matrices"))
    if s.backend_workspace !== A
        s.pivots = similar(A, Cint, n, modes)
        s.info = similar(A, Cint, modes)
        s.backend_workspace = A
    end
    if !isempty(A)
        AMDGPU.rocSOLVER.rocsolver_zgetrf_strided_batched(AMDGPU.rocBLAS.handle(), n, n,
            A, n, stride(A,3), s.pivots, n, s.info, modes)
        info = Array(s.info)
        bad = findfirst(!iszero, info)
        isnothing(bad) || error("BatchedDenseLU: singular stage matrix at mode $bad (info=$(info[bad]))")
    end
    s.factored = true
    return s
end
function Tarang._batched_solve_impl!(X::AbstractMatrix{ComplexF64}, s::Tarang.BatchedDenseLU{<:ROCArray},
                                     B::AbstractMatrix{ComplexF64})
    A = s.A
    n, _, modes = size(A)
    s.backend_workspace === A || error("Batched matrix storage changed. Refactor before solving.")
    size(X) == size(B) == (n, modes) || throw(DimensionMismatch("Invalid batched RHS shape"))
    X isa ROCMatrix && B isa ROCMatrix && AMDGPU.device(A) == AMDGPU.device(X) == AMDGPU.device(B) ||
        throw(ArgumentError("Batched solve requires matrices on the same AMD GPU"))
    ensure_device!(Tarang.architecture(A))
    X === B || copyto!(X, Base.mightalias(X, B) ? copy(B) : B)
    if !isempty(A)
        AMDGPU.rocSOLVER.rocsolver_zgetrs_strided_batched(AMDGPU.rocBLAS.handle(),
            AMDGPU.rocBLAS.rocblas_operation_none, n, 1, A, n, stride(A,3), s.pivots, n,
            X, n, stride(X,2), modes)
    end
    return X
end
