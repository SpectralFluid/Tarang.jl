# Batched dense LU uses persistent device pointer tables. CUDA's strided-batched
# convenience wrappers build and free these tables on every call, so the raw
# cuBLAS entry points below must own their metadata instead. The workspace also
# retains the pointed-to arrays and reuses pivot/status buffers across refactors.
# Numerical CUDA execution remains covered by hardware-gated tests.
#
# cuBLAS destroys `s.A`, unlike the CPU reference's copying `lu`. Callers must
# fully reassemble `s.A` before refactoring; see `BatchedDenseLU`'s lifecycle
# contract in src/tools/batched_matsolvers.jl.

mutable struct CUDABatchedLUWorkspace{A,P,I,AP}
    matrix::A
    matrix_ptrs::AP
    pivots::P
    info::I
    host_info::Vector{Cint}
    rhs::Any
    rhs_ptrs::Any
    solve_info::Base.RefValue{Cint}
    lock::ReentrantLock
end

function _cuda_batched_workspace!(s)
    A = s.A
    n, ncols, nmodes = size(A)
    n == ncols || throw(DimensionMismatch("BatchedDenseLU matrices must be square"))
    workspace = s.backend_workspace
    if workspace isa CUDABatchedLUWorkspace && workspace.matrix === A
        return workspace
    end
    workspace = CUDABatchedLUWorkspace(
        A, CUDA.CUBLAS.unsafe_strided_batch(A),
        CuArray{Cint}(undef, n, nmodes), CuArray{Cint}(undef, nmodes),
        Vector{Cint}(undef, nmodes), nothing, nothing, Ref{Cint}(0), ReentrantLock())
    s.backend_workspace = workspace
    return workspace
end

# Pointer tables hide their pointees from CUDA's memory tracking. Touch each
# array's managed pointer at every use so a changed task/stream waits on its
# previous work and records ownership for stream-ordered deallocation.
function _cuda_batched_factor!(s, workspace::CUDABatchedLUWorkspace)
    A = s.A
    n, _, nmodes = size(A)
    pointer(A)
    GC.@preserve A workspace begin
        CUDA.CUBLAS.cublasZgetrfBatched(CUDA.CUBLAS.handle(), n,
            workspace.matrix_ptrs, max(1, stride(A, 2)), workspace.pivots,
            workspace.info, nmodes)
    end
    copyto!(workspace.host_info, workspace.info)
    if any(!iszero, workspace.host_info)
        bad = findall(!iszero, workspace.host_info)
        error("BatchedDenseLU (GPU): singular stage matrix at mode(s) " *
              "$(bad) of $nmodes; cuBLAS info = $(workspace.host_info[bad]). " *
              "No CPU fallback is attempted — see the no-silent-fallback " *
              "contract (#74).")
    end
    s.pivots = workspace.pivots
    s.info = workspace.info
    s.factored = true
    return s
end

"""Factor fully reassembled matrices in place; retain and check every mode's status."""
function Tarang._batched_factor_impl!(s::Tarang.BatchedDenseLU{<:CuArray})
    s.factored = false
    return CUDA.context!(CUDA.context(s.A)) do
        workspace = _cuda_batched_workspace!(s)
        lock(workspace.lock) do
            _cuda_batched_factor!(s, workspace)
        end
    end
end

function _cuda_batched_solve!(X, s, B, workspace::CUDABatchedLUWorkspace)
    A = s.A
    workspace.matrix === A || error(
        "BatchedDenseLU (GPU): matrix storage changed; factor the new matrix before solving")
    n, _, nmodes = size(A)
    size(X) == size(B) == (n, nmodes) ||
        throw(DimensionMismatch("BatchedDenseLU RHS and destination must have size ($n, $nmodes)"))
    X isa CuMatrix{ComplexF64} || throw(ArgumentError(
        "BatchedDenseLU (GPU) requires a contiguous CUDA destination"))
    CUDA.context(X) == CUDA.context(A) || throw(ArgumentError(
        "BatchedDenseLU matrix and destination must belong to the same CUDA context"))
    if X !== B
        copyto!(X, Base.mightalias(X, B) ? copy(B) : B)
    end
    pointer(A)
    pointer(X)
    if workspace.rhs !== X
        # A matrix's last dimension is already the batch dimension: each RHS
        # column is contiguous, so no (n, 1, nmodes) reshape is needed.
        workspace.rhs_ptrs = CUDA.CUBLAS.unsafe_strided_batch(X)
        workspace.rhs = X
    end
    workspace.solve_info[] = 0
    GC.@preserve A X workspace begin
        CUDA.CUBLAS.cublasZgetrsBatched(CUDA.CUBLAS.handle(), 'N', n, 1,
            workspace.matrix_ptrs, max(1, stride(A, 2)), workspace.pivots,
            workspace.rhs_ptrs, max(1, stride(X, 2)), workspace.solve_info, nmodes)
    end
    # getrs reports invalid arguments through one host scalar. Singularity was
    # checked after getrf; neither failure may silently fall back to the CPU.
    workspace.solve_info[] == 0 || error(
        "BatchedDenseLU (GPU): getrs reported invalid arguments " *
        "(info = $(workspace.solve_info[])).")
    return X
end

"""Solve all modes into owned GPU storage, reusing pointer tables and pivots."""
function Tarang._batched_solve_impl!(X::AbstractMatrix{ComplexF64},
                                     s::Tarang.BatchedDenseLU{<:CuArray},
                                     B::AbstractMatrix{ComplexF64})
    return CUDA.context!(CUDA.context(s.A)) do
        workspace = s.backend_workspace
        lock(workspace.lock) do
            _cuda_batched_solve!(X, s, B, workspace)
        end
    end
end

# Dense per-mode solves may copy their result to a different GPU. Only a
# same-context contiguous destination can be passed directly to cuSOLVER.
Tarang._cudense_inplace_compatible(dest::CuVector, matrix::CuMatrix) =
    CUDA.context(dest) == CUDA.context(matrix)
Tarang._cudense_with_context(f::F, matrix::CuMatrix) where {F} =
    CUDA.context!(f, CUDA.context(matrix))
