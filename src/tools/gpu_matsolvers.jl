"""
GPU Matrix Solvers for Tarang.jl

This module provides GPU-accelerated linear solvers using CUDA.jl.
These can be beneficial for:
- Large problems (>500K unknowns)
- Iterative solvers with many RHS vectors
- Problems where matrix stays constant across many solves

## Solver Types

1. **CuSolverLU**: Direct LU factorization on GPU using cuSOLVER
   - Best for: Medium-large dense or sparse systems
   - Note: May be slower than CPU for small sparse matrices

2. **CuIterativeCG**: Conjugate Gradient on GPU
   - Best for: Large symmetric positive definite systems
   - Supports preconditioners: `:jacobi`, `:ilu0`, `:none`, or custom

3. **CuIterativeGMRES**: GMRES on GPU
   - Best for: Large non-symmetric systems
   - Supports preconditioners: `:jacobi`, `:ilu0`, `:none`, or custom

4. **CuDenseLU**: Dense LU on GPU
   - Best for: Small-medium dense systems
   - Very fast for matrices that fit in GPU memory

## Preconditioners

| Preconditioner | Description | Best For |
|----------------|-------------|----------|
| `:jacobi`      | Diagonal (Jacobi) preconditioner | Diagonally dominant matrices |
| `:ilu0`        | ILU(0) via cuSPARSE | Ill-conditioned matrices, strong off-diagonal coupling |
| `:ic0`         | IC(0) via cuSPARSE | Symmetric positive definite (SPD) matrices |
| `:none`        | No preconditioning | Well-conditioned problems |
| Custom function| User-defined M^{-1} | Specialized applications |

**ILU(0)** (Incomplete LU with zero fill-in) is significantly more effective than Jacobi
for tough matrices, often reducing iteration counts by 5-10x on ill-conditioned problems.

**IC(0)** (Incomplete Cholesky with zero fill-in) is the symmetric variant, optimal for
SPD matrices. It uses half the memory of ILU(0) and provides better numerical stability.

## When to Use GPU vs CPU Solvers

| Problem Size | Matrix Type | Recommended Solver |
|--------------|-------------|--------------------|
| < 10K        | Sparse      | CPU SparseLU       |
| < 10K        | Dense       | GPU CuDenseLU      |
| 10K - 100K   | Sparse      | CPU SparseLU or GPU CuIterative |
| 10K - 100K   | Dense       | GPU CuDenseLU      |
| > 100K       | Sparse      | GPU CuIterative    |
| > 100K       | Dense       | GPU CuDenseLU (if memory permits) |

## Usage

```julia
using CUDA
using Tarang

# Direct GPU solver
solver = CuSolverLU(A_sparse)
x = solve(solver, b)

# Iterative GPU solver with Jacobi preconditioner
solver = CuIterativeCG(A_sparse; preconditioner=:jacobi, tol=1e-10)
x = solve(solver, b)

# Iterative GPU solver with ILU(0) preconditioner (better for tough matrices)
solver = CuIterativeCG(A_sparse; preconditioner=:ilu0, tol=1e-10)
x = solve(solver, b)

# Iterative GPU solver with IC(0) preconditioner (optimal for SPD matrices)
solver = CuIterativeCG(A_spd; preconditioner=:ic0, tol=1e-10)
x = solve(solver, b)

# GMRES with ILU(0) for non-symmetric systems
solver = CuIterativeGMRES(A_sparse; preconditioner=:ilu0, tol=1e-10)
x = solve(solver, b)

# Dense GPU solver
solver = CuDenseLU(A_dense)
x = solve(solver, b)
```
"""

# Only load GPU solvers if CUDA is available
const CUDA_AVAILABLE = Ref(false)

# CUDA-specific bindings live in TarangCUDAExt. Core owns the solver contracts
# and algorithms, while these extension methods supply CUSOLVER/CUSPARSE only
# after CUDA.jl has actually loaded.
# Dispatch on `Val{:cuda}` rather than declaring method-less functions: the
# extension needs a signature it can EXTEND, and a zero-argument
# `Tarang._gpu_cusolver_module() = CUDA.CUSOLVER` in the extension would
# OVERWRITE a zero-argument definition here rather than add to it (there is no
# specificity gap between two zero-argument methods). The `::Val` fallback is
# strictly less specific than the extension's `::Val{:cuda}`, so both coexist.
# Mirrors `_cuda_functional(::Val{:cuda})`.
#
# The method-less form this replaces also left the call sites below with nothing
# to resolve, which is a MethodError to any caller — and to JET, which reported
# both of them.
_gpu_cusolver_module(::Val) = error(
    "CUSOLVER bindings require CUDA.jl to be loaded (load CUDA before using GPU solvers).")
_gpu_cusparse_module(::Val) = error(
    "CUSPARSE bindings require CUDA.jl to be loaded (load CUDA before using GPU solvers).")

_gpu_cusolver_module() = _gpu_cusolver_module(Val(:cuda))
_gpu_cusparse_module() = _gpu_cusparse_module(Val(:cuda))

_cusolver() = _gpu_cusolver_module()
_cusparse() = _gpu_cusparse_module()

"""
    _activate_gpu_solvers!()

Called by TarangCUDAExt.__init__ once the extension is loaded. The extension
owns all CUDA module bindings; this hook only flips availability and registers
the GPU solver constructors in the solver registry.
"""
function _activate_gpu_solvers!()
    CUDA_AVAILABLE[] = true
    _register_gpu_solvers()
    @debug "GPU matrix solvers enabled (CUDA extension active)"
    return nothing
end

"""
    _init_gpu_solvers!()

Legacy hook kept for the main module `__init__`. Extension loading always
happens AFTER the parent package's `__init__`, so `Base.get_extension` can
never see TarangCUDAExt here — actual activation is driven by the extension
itself via `_activate_gpu_solvers!`. This function is now a no-op.
"""
_init_gpu_solvers!() = nothing

# Import from parent MatSolvers module
using ..MatSolvers: AbstractMatSolver, register_solver, SOLVER_REGISTRY

# Import triangular types for ILU/IC preconditioner application
using LinearAlgebra: UnitLowerTriangular, LowerTriangular, UpperTriangular, ldiv!, tril

# ============================================================================
# GPU Helper Functions (Abstract CUDA-specific calls for consistency)
# ============================================================================
# These helpers centralize CUDA-specific operations, making the code:
# 1. More consistent with the architecture abstraction pattern
# 2. Easier to adapt for different GPU backends in the future
# 3. Cleaner and more maintainable

"""
    _gpu_zeros(T::Type, dims...)

Allocate a zero-filled GPU array. Wrapper around CUDA.zeros for consistency.
"""
function _gpu_zeros end

"""
    _gpu_array(data::AbstractArray, T::Type)

Convert array to GPU array with specified element type.
"""
function _gpu_array end

"""
    _gpu_sparse_csr(A::SparseMatrixCSC, T::Type)

Convert sparse matrix to GPU CSR format for cuSPARSE operations.
"""
function _gpu_sparse_csr end

"""
    _gpu_axpy!(α, x, y)

GPU AXPY operation: y = α*x + y. Uses CUDA.axpy! or custom kernel.
"""
function _gpu_axpy! end

"""
    _is_gpu_array(a)

Check if array is on GPU.
"""
function _is_gpu_array end

"""
    _gpu_ilu0(A_csr)

Compute ILU(0) factorization on GPU CSR matrix using cuSPARSE.
"""
function _gpu_ilu0 end

"""
    _gpu_ic0(A_csr)

Compute IC(0) (Incomplete Cholesky) factorization on GPU CSR matrix using cuSPARSE.
For symmetric positive definite matrices only.
"""
function _gpu_ic0 end

# GPU helper implementations (_gpu_zeros, _gpu_array, etc.) are now defined
# in ext/cuda/utils.jl via the CUDA extension, eliminating runtime @eval.

# ============================================================================
# Preconditioner infrastructure (must be defined before iterative solvers)
# ============================================================================

abstract type AbstractPreconditioner end

struct NoPreconditioner <: AbstractPreconditioner end

# Type parameter is the stored vector type (a CuVector on GPU). A field typed
# `Any` with an undeducible {T} made the struct unconstructible.
struct DiagonalPreconditioner{V} <: AbstractPreconditioner
    diag_inv::V
end

struct CustomPreconditioner{F} <: AbstractPreconditioner
    apply!::F
end

"""
    ILU0Preconditioner{T} <: AbstractPreconditioner

Incomplete LU factorization with zero fill-in (ILU(0)) preconditioner on GPU.
Uses cuSPARSE for efficient factorization and triangular solves.

ILU(0) is significantly more effective than Jacobi for:
- Ill-conditioned matrices
- Problems with strong off-diagonal coupling
- Reducing CG/GMRES iteration counts on tough matrices

The factorization computes L and U such that A ≈ LU, where:
- L is lower triangular with unit diagonal
- U is upper triangular
- Only nonzero positions from original A are filled

Application involves two triangular solves: z = U^{-1}(L^{-1}r)
"""
struct ILU0Preconditioner{T} <: AbstractPreconditioner
    # LU factors stored in CSR format (L below diagonal, U on and above diagonal)
    LU_csr::Any         # CuSparseMatrixCSR{T} with LU factors
    # Workspace for intermediate triangular solve result
    tmp::Any            # CuVector{T}
    # Matrix dimensions
    n::Int
end

"""
    IC0Preconditioner{T} <: AbstractPreconditioner

Incomplete Cholesky factorization with zero fill-in (IC(0)) preconditioner on GPU.
Uses cuSPARSE for efficient factorization and triangular solves.

IC(0) is the symmetric version of incomplete factorization, specifically for
symmetric positive definite (SPD) matrices. It is more efficient than ILU(0)
when the matrix is known to be SPD because:
- Only one triangular factor L is computed (A ≈ LL^T)
- Memory usage is halved compared to storing both L and U
- Better numerical stability for SPD systems

**Important**: The input matrix MUST be symmetric positive definite.
Using IC(0) on a non-SPD matrix may produce incorrect results or fail.

The factorization computes L such that A ≈ LL^T, where:
- L is lower triangular with explicit diagonal
- Only nonzero positions from the lower triangle of A are filled

Application involves two triangular solves: z = L^{-T}(L^{-1}r)
"""
struct IC0Preconditioner{T} <: AbstractPreconditioner
    # L factor stored in CSR format (lower triangular)
    L_csr::Any          # CuSparseMatrixCSR{T} with L factor
    # Workspace for intermediate triangular solve result
    tmp::Any            # CuVector{T}
    # Matrix dimensions
    n::Int
end

# ============================================================================
# GPU Dense LU Solver
# ============================================================================

"""
    CuDenseLU <: AbstractMatSolver

Dense LU factorization on GPU using cuSOLVER.
Efficient for small-medium dense matrices.

The matrix is kept on GPU, and factorization uses CUDA's cusolverDnXgetrf.
An exactly singular matrix (a zero pivot in `U`) throws `SingularException`
at construction, as the CPU `DenseLUSolver` does.
"""
struct CuDenseLU{T} <: AbstractMatSolver
    A_gpu::Any          # CuMatrix{T}
    lu_factor::Any      # Factorization result
    ipiv::Any           # CuVector{Int32} pivot indices
    workspace::Any      # GPU workspace buffer
    n::Int
end

function CuDenseLU(matrix::AbstractMatrix; kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)
    @assert size(matrix, 2) == n "Matrix must be square"

    # Transfer matrix to GPU using helper
    A_gpu = _gpu_array(matrix, T)

    # Allocate pivot vector on GPU using helper
    ipiv = _gpu_zeros(Int32, n)

    # Perform LU factorization on GPU
    # cusolverDnXgetrf computes LU factorization in-place
    _cudense_factor!(A_gpu, ipiv)

    return CuDenseLU{T}(A_gpu, nothing, ipiv, nothing, n)
end

# CUDA.jl's `CUSOLVER.getrf!` returns `(A, ipiv, info)` and raises only for
# info < 0 (an invalid argument, via `chkargsok`). info > 0 — U[info, info] is
# exactly zero — is an ordinary return, and ignoring it built a factor whose every
# solve came back Inf/NaN. Raise as the CPU `DenseLUSolver` does, so the stepper's
# singular-stage handling sees the same exception on both backends. (`getrs!`
# reports only invalid arguments, which CUDA.jl's `chkargsok` already raises.)
function _cudense_factor!(matrix, pivots)
    _, _, info = _cudense_getrf!(matrix, pivots)
    info > 0 && throw(SingularException(Int(info)))
    return matrix
end
_cudense_getrf!(matrix, pivots) = _cusolver().getrf!(matrix, pivots)

function MatSolvers.solve(s::CuDenseLU{T}, rhs::AbstractVector) where T
    length(rhs) == s.n || throw(DimensionMismatch("CuDenseLU right-hand side length"))
    return _cudense_with_context(s.A_gpu) do
        b_gpu = _is_gpu_array(rhs) && eltype(rhs) === T ? copy(rhs) : _gpu_array(rhs, T)
        _cudense_solve!(b_gpu, s.A_gpu, s.ipiv)
        b_gpu
    end
end

# Recover concrete array types from the backend-erased factor storage.
_cudense_solve!(dest, matrix, pivots) = _cusolver().getrs!('N', matrix, pivots, dest)
_cudense_with_context(f::F, matrix) where {F} = f()
_cudense_inplace_compatible(dest, matrix) = _is_gpu_array(dest)

function MatSolvers.solve!(dest::AbstractVector, s::CuDenseLU{T},
                           rhs::AbstractVector) where T
    length(rhs) == s.n && length(dest) == s.n ||
        throw(DimensionMismatch("CuDenseLU vector length"))
    if !_cudense_inplace_compatible(dest, s.A_gpu) || eltype(dest) != T ||
       (dest !== rhs && Base.mightalias(dest, rhs))
        return copyto!(dest, MatSolvers.solve(s, rhs))
    end
    return _cudense_with_context(s.A_gpu) do
        dest === rhs || copyto!(dest, rhs)
        _cudense_solve!(dest, s.A_gpu, s.ipiv)
        dest
    end
end

# ============================================================================
# GPU Sparse LU Solver (via cuSOLVER)
# ============================================================================

"""
    CuSparseLU <: AbstractMatSolver

GPU sparse direct solver.

- For `Float64` matrices, this uses a true sparse LU/refactorization path backed by
  cuSOLVER's host CSR LU analysis plus device `cusolverRF`.
- For other element types, it falls back to cuSOLVER's sparse QR solve path.

The RF backend is currently limited by CUDA.jl's bindings to real double-precision
matrices. Complex sparse systems therefore still use QR.
"""
mutable struct CuSparseRF{T}
    handle::Any
    A_rowptr::Any
    A_colind::Any
    A_vals::Any
    L_rowptr::Any
    L_colind::Any
    L_vals::Any
    U_rowptr::Any
    U_colind::Any
    U_vals::Any
    P::Any
    Q::Any
    n::Int
end

"""
    CuSparseLU{T} <: AbstractMatSolver

GPU sparse direct solver. Mutable so that `MatSolvers.refactor!` can swap
in a new numeric factorization while reusing the symbolic analysis when
the matrix sparsity pattern is unchanged — the common case for
IMEX-RK solvers that rebuild `(M + dt*a_ii*L)` with a new `dt`.

- **`:rf` backend** (Float64 only): full symbolic reuse via
  `cusolverRfResetValues` + `cusolverRfRefactor`. ~2× faster than a
  fresh rebuild when `dt` changes.
- **`:qr` backend** (complex or fallback): refactor updates
  `A_csr.nzVal` in place and re-calls `spqr_factorise` on the existing
  `SparseQR` handle so the cached symbolic analysis (rowPtr/colVal
  pattern, scratch buffer) is reused — only the Setup + numeric Factor
  phases run. Falls back to a full
  rebuild. Still safe to call; the interface matches the CPU path.
"""
mutable struct CuSparseLU{T} <: AbstractMatSolver
    A_csr::Any   # _cusparse().CuSparseMatrixCSR{T, Int32}
    factor::Any
    backend::Symbol
    tol::Float64
    reorder::Bool
    n::Int
    rhs_buffer::Any
    solution_buffer::Any
    temp_buffer::Any    # RF scratch uses the same storage that QR uses for its RHS.
end

function _host_csr(A::SparseMatrixCSC{T,<:Integer}) where T
    At = SparseMatrixCSC{T, Int32}(copy(transpose(A)))
    rowptr = Vector{Int32}(At.colptr)
    colind = Vector{Int32}(At.rowval)
    vals = Vector{T}(At.nzval)
    return rowptr, colind, vals
end

_supports_cusolver_rf(::Type{T}) where {T} = T == Float64

function _build_cusolver_rf(matrix::SparseMatrixCSC{Float64,Int32}; tol::Real=1e-12)
    # `checksquare` (LinearAlgebra) is not imported into this module — calling it here
    # threw UndefVarError, which the CuSparseLU constructor's try/catch silently swallowed,
    # so the cuSOLVER-RF path never ran (every Float64 sparse system fell back to complex QR).
    n = size(matrix, 1)
    size(matrix, 2) == n || throw(DimensionMismatch("cuSOLVER-RF requires a square matrix, got $(size(matrix))"))
    nnzA = Cint(nnz(matrix))
    rowptrA, colindA, valsA = _host_csr(matrix)

    sp_handle = _cusolver().sparse_handle()
    descA = _cusparse().CuMatrixDescriptor('G', 'L', 'N', 'O')
    info_ref = Ref{_cusolver().csrluInfoHost_t}()
    _cusolver().cusolverSpCreateCsrluInfoHost(info_ref)
    info = info_ref[]

    _cusolver().cusolverSpXcsrluAnalysisHost(sp_handle, Cint(n), nnzA, descA, rowptrA, colindA, info)

    internal_bytes = Ref{Csize_t}(0)
    workspace_bytes = Ref{Csize_t}(0)
    _cusolver().cusolverSpDcsrluBufferInfoHost(sp_handle, Cint(n), nnzA, descA,
                                                 valsA, rowptrA, colindA, info,
                                                 internal_bytes, workspace_bytes)
    workspace = Vector{UInt8}(undef, Int(workspace_bytes[]))
    _cusolver().cusolverSpDcsrluFactorHost(sp_handle, Cint(n), nnzA, descA,
                                             valsA, rowptrA, colindA, info,
                                             Float64(tol), pointer(workspace))

    singularity = Ref{Cint}(-1)
    _cusolver().cusolverSpDcsrluZeroPivotHost(sp_handle, info, Float64(tol), singularity)
    singularity[] >= 0 && throw(SingularException(Int(singularity[])))

    nnzL = Ref{Cint}(0)
    nnzU = Ref{Cint}(0)
    _cusolver().cusolverSpXcsrluNnzHost(sp_handle, nnzL, nnzU, info)

    P = Vector{Cint}(undef, n)
    Q = Vector{Cint}(undef, n)
    rowptrL = Vector{Cint}(undef, n + 1)
    colindL = Vector{Cint}(undef, Int(nnzL[]))
    valsL = Vector{Float64}(undef, Int(nnzL[]))
    rowptrU = Vector{Cint}(undef, n + 1)
    colindU = Vector{Cint}(undef, Int(nnzU[]))
    valsU = Vector{Float64}(undef, Int(nnzU[]))
    descL = _cusparse().CuMatrixDescriptor('G', 'L', 'U', 'O')
    descU = _cusparse().CuMatrixDescriptor('G', 'U', 'N', 'O')

    _cusolver().cusolverSpDcsrluExtractHost(sp_handle, P, Q,
                                              descL, valsL, rowptrL, colindL,
                                              descU, valsU, rowptrU, colindU,
                                              info, pointer(workspace))
    _cusolver().cusolverSpDestroyCsrluInfoHost(info)

    handle_ref = Ref{_cusolver().cusolverRfHandle_t}()
    _cusolver().cusolverRfCreate(handle_ref)
    handle = handle_ref[]
    _cusolver().cusolverRfSetMatrixFormat(handle,
        _cusolver().CUSOLVERRF_MATRIX_FORMAT_CSR,
        _cusolver().CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L)
    _cusolver().cusolverRfSetResetValuesFastMode(handle,
        _cusolver().CUSOLVERRF_RESET_VALUES_FAST_MODE_ON)

    A_rowptr_gpu = _gpu_array(rowptrA, Int32)
    A_colind_gpu = _gpu_array(colindA, Int32)
    A_vals_gpu = _gpu_array(valsA, Float64)
    L_rowptr_gpu = _gpu_array(rowptrL, Int32)
    L_colind_gpu = _gpu_array(colindL, Int32)
    L_vals_gpu = _gpu_array(valsL, Float64)
    U_rowptr_gpu = _gpu_array(rowptrU, Int32)
    U_colind_gpu = _gpu_array(colindU, Int32)
    U_vals_gpu = _gpu_array(valsU, Float64)
    P_gpu = _gpu_array(P, Int32)
    Q_gpu = _gpu_array(Q, Int32)

    _cusolver().cusolverRfSetupDevice(Cint(n), nnzA,
                                        A_rowptr_gpu, A_colind_gpu, A_vals_gpu,
                                        nnzL[], L_rowptr_gpu, L_colind_gpu, L_vals_gpu,
                                        nnzU[], U_rowptr_gpu, U_colind_gpu, U_vals_gpu,
                                        P_gpu, Q_gpu, handle)
    _cusolver().cusolverRfAnalyze(handle)
    _cusolver().cusolverRfRefactor(handle)

    rf = CuSparseRF{Float64}(handle,
                             A_rowptr_gpu, A_colind_gpu, A_vals_gpu,
                             L_rowptr_gpu, L_colind_gpu, L_vals_gpu,
                             U_rowptr_gpu, U_colind_gpu, U_vals_gpu,
                             P_gpu, Q_gpu, n)
    finalizer(rf) do _
        try
            _cusolver().cusolverRfDestroy(handle)
        catch err
            # A finalizer must never throw — an exception here would be raised on an
            # arbitrary task during GC. Record the leak instead of discarding it
            # silently; a cuSOLVER handle that will not destroy is worth knowing about.
            @debug "cusolverRfDestroy failed in finalizer; handle may leak" exception = err
        end
    end
    return rf
end

function _build_sparse_qr(A_csr, ::Type{T}, tol::Real) where {T}
    factor = _cusolver().SparseQR(A_csr)
    _cusolver().spqr_factorise(factor, A_csr, Float64(tol))
    return factor
end

function CuSparseLU(matrix::SparseMatrixCSC; tol::Real=1e-12, reorder::Bool=true, kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = eltype(matrix)
    if _supports_cusolver_rf(T)
        A = SparseMatrixCSC{Float64, Int32}(matrix)
        A_csr = _gpu_sparse_csr(A, Float64)
        try
            rf = _build_cusolver_rf(A; tol=tol)
            n = size(A, 1)
            work = _gpu_zeros(Float64, n)
            return CuSparseLU{Float64}(
                A_csr, rf, :rf, Float64(tol), reorder, n,
                work, _gpu_zeros(Float64, n), work,
            )
        catch err
            @warn "CuSparseLU: RF setup failed, falling back to sparse QR" exception=(err, catch_backtrace()) maxlog=1
        end
    end

    Tq = promote_type(T, ComplexF64)
    A_csr = _gpu_sparse_csr(matrix, Tq)
    qr_factor = _build_sparse_qr(A_csr, Tq, tol)
    n = size(matrix, 1)
    work = _gpu_zeros(Tq, n)
    return CuSparseLU{Tq}(
        A_csr, qr_factor, :qr, Float64(tol), reorder, n,
        work, _gpu_zeros(Tq, n), work,
    )
end

function CuSparseLU(matrix::AbstractMatrix; kwargs...)
    return CuSparseLU(sparse(matrix); kwargs...)
end

@inline _valid_cusparse_buffer(buffer, ::Type{T}, n::Int) where T =
    buffer !== nothing && length(buffer) == n && eltype(buffer) == T

function _cusparse_buffers!(s::CuSparseLU{T}) where T
    _valid_cusparse_buffer(s.rhs_buffer, T, s.n) ||
        (s.rhs_buffer = _gpu_zeros(T, s.n))
    _valid_cusparse_buffer(s.solution_buffer, T, s.n) ||
        (s.solution_buffer = _gpu_zeros(T, s.n))
    s.temp_buffer = s.rhs_buffer
    return s.rhs_buffer, s.solution_buffer, s.temp_buffer
end

function MatSolvers.solve!(dest::AbstractVector, s::CuSparseLU{T},
                           rhs::AbstractVector) where T
    length(rhs) == s.n || throw(DimensionMismatch(
        "CuSparseLU right-hand side has length $(length(rhs)); expected $(s.n)",
    ))
    length(dest) == s.n || throw(DimensionMismatch(
        "CuSparseLU destination has length $(length(dest)); expected $(s.n)",
    ))

    b_gpu, x_gpu, temp = _cusparse_buffers!(s)
    if s.backend === :rf
        rf = s.factor::CuSparseRF{Float64}
        # cusolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf): `XF` holds the RHS on
        # input and the solution on output; `Temp` is pure scratch. The old call passed
        # the RHS as `Temp` (clobbered) and an UNINITIALIZED buffer as `XF`, so the solve
        # read garbage. Put the RHS in XF (x_gpu) and give it a separate scratch buffer.
        copyto!(x_gpu, rhs)
        _cusolver().cusolverRfSolve(rf.handle, rf.P, rf.Q, Cint(1),
                                      temp, Cint(rf.n), x_gpu, Cint(rf.n))
    else
        copyto!(b_gpu, rhs)
        _cusolver().spqr_solve(s.factor, b_gpu, x_gpu)
    end
    copyto!(dest, x_gpu)
    return dest
end

function MatSolvers.solve(s::CuSparseLU{T}, rhs::AbstractVector) where T
    _, x_gpu, _ = _cusparse_buffers!(s)
    result = similar(x_gpu)
    return MatSolvers.solve!(result, s, rhs)
end

"""
    refactor!(solver::CuSparseLU, A::SparseMatrixCSC) -> solver

Update the GPU solver's numeric factorization to match the new matrix
`A`. When the sparsity pattern is unchanged and the backend supports it
(`:rf` / cusolverRF, Float64 only), this reuses the cached symbolic
analysis and runs only the numeric refactorization phase via
`cusolverRfResetValues` + `cusolverRfRefactor` — ~2× speedup on the LU
step compared to rebuilding from scratch.

For the `:qr` backend (complex matrices or Float64-RF-unavailable
fallback), the fast path updates `solver.A_csr.nzVal` in place (from
the transposed CSC of `A`) and re-calls `_cusolver().spqr_factorise`
on the existing `SparseQR` handle. This reuses the cached symbolic
analysis (row/column pattern and scratch buffer allocated inside the
`SparseQR` info struct) and runs only the Setup + numeric Factor
phases. Falls back to a full rebuild if the value update or refactor
raises — the interface matches the CPU path for drop-in compatibility.

### Pattern check

The fast path assumes `A` has the same sparsity pattern as the matrix
the solver was originally built from. Tarang's stepper satisfies this
because it updates `sp.LHS.nzval` in place; the pattern is identical by
construction. If you call `refactor!` with a differently-patterned
matrix, behavior is undefined — call `CuSparseLU(A)` instead to build
a fresh solver.
"""
function MatSolvers.refactor!(solver::CuSparseLU{T}, A::SparseMatrixCSC) where {T}
    if size(A, 1) != solver.n || size(A, 2) != solver.n
        # Dimension mismatch — fall back to full rebuild.
        new_solver = CuSparseLU(A; tol=solver.tol, reorder=solver.reorder)
        solver.A_csr = new_solver.A_csr
        solver.factor = new_solver.factor
        solver.backend = new_solver.backend
        solver.n = new_solver.n
        solver.rhs_buffer = new_solver.rhs_buffer
        solver.solution_buffer = new_solver.solution_buffer
        solver.temp_buffer = new_solver.temp_buffer
        return solver
    end

    if solver.backend === :rf
        # Symbolic-reuse fast path for Float64 + cusolverRF.
        try
            A_f64 = eltype(A) === Float64 ? A : SparseMatrixCSC{Float64, Int}(A)
            rf = solver.factor::CuSparseRF{Float64}

            # Update host-side CSR values (row-major, as expected by
            # cusolverRf), then upload to GPU.
            At = SparseMatrixCSC{Float64, Int32}(copy(transpose(A_f64)))
            host_vals = Vector{Float64}(At.nzval)
            copyto!(rf.A_vals, host_vals)

            # Reset matrix values in the RF handle and re-run numeric
            # factorization. The symbolic analysis (P, Q, L/U patterns)
            # is preserved from the original build.
            _cusolver().cusolverRfResetValues(
                Cint(solver.n), Cint(nnz(A_f64)),
                rf.A_rowptr, rf.A_colind, rf.A_vals,
                rf.P, rf.Q, rf.handle,
            )
            _cusolver().cusolverRfRefactor(rf.handle)
            return solver
        catch err
            @debug "refactor!(CuSparseLU, :rf) failed, rebuilding from scratch" exception=(err, catch_backtrace())
            # fall through to full rebuild below.
        end
    elseif solver.backend === :qr
        # Symbolic-reuse fast path for SparseQR.
        #
        # CUDA.jl's `SparseQR` struct caches its info handle and scratch
        # buffer; the symbolic analysis (row/col pattern and associated
        # internal state) is tied to that struct, not to the values. So
        # we can update `A_csr.nzVal` in place and re-call
        # `spqr_factorise`, which runs the Setup + numeric Factor phases
        # against the new values while reusing the cached analysis.
        try
            Tq = eltype(solver.A_csr.nzVal)
            A_typed = eltype(A) === Tq ? A : SparseMatrixCSC{Tq, Int}(A)

            # Transpose-then-convert trick: the nzval of
            # `transpose(A_typed)` as a CSC is exactly the row-major
            # value ordering that CSR expects (same permutation as the
            # original CSR build).
            At = SparseMatrixCSC{Tq, Int32}(copy(transpose(A_typed)))
            host_vals = Vector{Tq}(At.nzval)
            # Upload new values into the existing GPU CSR buffer in
            # place. `A_csr.nzVal` is a `CuVector{Tq}`; `copyto!` with a
            # host Vector uses the asynchronous H2D path.
            copyto!(solver.A_csr.nzVal, host_vals)

            # Re-run the numeric factorization on the existing
            # `SparseQR` handle, which preserves its cached info +
            # scratch buffer.
            _cusolver().spqr_factorise(solver.factor, solver.A_csr, Float64(solver.tol))
            return solver
        catch err
            @debug "refactor!(CuSparseLU, :qr) failed, rebuilding from scratch" exception=(err, catch_backtrace())
            # fall through to full rebuild below.
        end
    end

    # Full rebuild fallback (dimension change, or either fast path
    # raised). Swap the wrapper's fields in place so the caller's
    # handle stays valid.
    new_solver = CuSparseLU(A; tol=solver.tol, reorder=solver.reorder)
    solver.A_csr = new_solver.A_csr
    solver.factor = new_solver.factor
    solver.backend = new_solver.backend
    return solver
end

# ============================================================================
# GPU Iterative Solver: Conjugate Gradient
# ============================================================================

"""
    CuIterativeCG <: AbstractMatSolver

Conjugate Gradient solver on GPU.
Best for large symmetric positive definite systems.

The matrix-vector products are performed on GPU using cuSPARSE,
which is highly efficient for sparse matrices.

# Arguments
- `matrix`: Sparse matrix (must be SPD for convergence)
- `tol`: RELATIVE residual tolerance (default: 1e-10). The solve has converged when
  the CG residual satisfies `‖r‖ ≤ tol·‖b‖` (`r` is the recursively updated
  residual, equal to `b - A*x` in exact arithmetic). `b = 0` returns `x = 0`.
- `maxiter`: Maximum iterations (default: 1000)
- `preconditioner`: Preconditioner definition:
  - `:jacobi` (default): Diagonal preconditioner, good for diagonally dominant matrices.
    Zero diagonal entries (|aᵢᵢ| ≤ 1e-14) are treated as 1.
  - `:ilu0`: ILU(0) via cuSPARSE, much better for ill-conditioned matrices
  - `:ic0`: IC(0) via cuSPARSE, optimal for SPD matrices (recommended for CG)
  - `:none`: No preconditioning
  - `Vector`: Explicit diagonal inverse values
  - `Function`: Custom preconditioner `(out, r) -> out = M^{-1}*r`

Failure is an error, never a silently returned iterate: `solve` throws when the
tolerance is not met within `maxiter` iterations, or when CG breaks down
(`p'Ap` negligible relative to `‖p‖·‖Ap‖`, i.e. the operator is not
Hermitian positive definite).

# Example
```julia
# For well-conditioned SPD matrices
solver = CuIterativeCG(A; preconditioner=:jacobi)

# For ill-conditioned matrices (reduces iterations significantly)
solver = CuIterativeCG(A; preconditioner=:ilu0, tol=1e-12)

# For SPD matrices (most efficient, uses half the memory of ILU(0))
solver = CuIterativeCG(A; preconditioner=:ic0, tol=1e-12)
```
"""
mutable struct CuIterativeCG{T,P<:AbstractPreconditioner} <: AbstractMatSolver
    A_csr::Any              # CuSparseMatrixCSR for SpMV
    preconditioner::P
    tol::Float64
    maxiter::Int
    n::Int
    # Workspace vectors (pre-allocated for efficiency)
    r::Any                  # Residual
    p::Any                  # Search direction
    Ap::Any                 # Matrix-vector product
    z::Any                  # Preconditioned residual

    function CuIterativeCG{T,P}(A_csr, preconditioner::P, tol::Float64, maxiter::Int,
                                 n::Int, r, p, Ap, z) where {T, P<:AbstractPreconditioner}
        new{T,P}(A_csr, preconditioner, tol, maxiter, n, r, p, Ap, z)
    end
end

function CuIterativeCG(matrix::AbstractMatrix;
                       tol::Real=1e-10,
                       maxiter::Int=1000,
                       preconditioner=:jacobi,   # Symbol, vector of diagonal inverses, function or nothing
                       kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)

    # Convert to CSR on GPU using helper
    A_sparse = matrix isa SparseMatrixCSC ? matrix : sparse(matrix)
    A_csr = _gpu_sparse_csr(A_sparse, T)

    # Build preconditioner
    M_inv = _build_preconditioner(A_sparse, preconditioner, T)

    # Pre-allocate workspace on GPU using helper
    r = _gpu_zeros(T, n)
    p = _gpu_zeros(T, n)
    Ap = _gpu_zeros(T, n)
    z = _gpu_zeros(T, n)

    return CuIterativeCG{T, typeof(M_inv)}(A_csr, M_inv, Float64(tol), maxiter, n, r, p, Ap, z)
end

# Solver instances retain mutable workspaces; as with CuSparseLU, use one
# instance per concurrently executing solve. Returned `solve` values own data.
function MatSolvers.solve(s::CuIterativeCG{T}, rhs::AbstractVector) where T
    return MatSolvers.solve!(similar(s.r, T, s.n), s, rhs)
end

function MatSolvers.solve!(x::AbstractVector, s::CuIterativeCG, rhs::AbstractVector)
    length(rhs) == s.n && length(x) == s.n || throw(DimensionMismatch("CG vector length"))
    if (is_gpu_array(s.r) && !is_gpu_array(x)) || eltype(x) != eltype(s.r)
        return copyto!(x, MatSolvers.solve(s, rhs))
    end
    copyto!(s.r, rhs)  # must precede clearing x when rhs aliases the destination
    fill!(x, zero(eltype(x)))
    return _cg_solve_workspace!(x, s.A_csr, s.preconditioner, s.tol, s.maxiter,
                                s.r, s.p, s.Ap, s.z)
end

# ‖b‖ for the relative stopping tests of the iterative solvers. A non-finite norm
# must not reach them: `Inf ≤ tol·Inf` would report x = 0 as converged.
function _checked_rhs_norm(b, solver::AbstractString)
    bnorm = norm(b)
    isfinite(bnorm) || throw(ArgumentError(
        "$solver: the right-hand side contains non-finite values (‖b‖ = $bnorm)."))
    return bnorm
end

const _CG_OPERATOR_HINT =
    "CG requires a Hermitian positive-definite operator; coupled Chebyshev–Fourier " *
    "(tau) systems are non-Hermitian. Use a direct solver (CuSparseLU/CuDenseLU) or GMRES."

function _cg_solve_workspace!(x, A, preconditioner, tol, maxiter, r, p, Ap, z)
    # x = 0 on entry, so r = b.
    bnorm = _checked_rhs_norm(r, "CuIterativeCG")
    # b = 0 (a quiescent mode): x = 0 is exact, and iterating would divide 0 by 0.
    iszero(bnorm) && return x
    # Iterate on the unit right-hand side b/‖b‖ and rescale x on return, so the
    # stopping test ‖r‖ ≤ tol is RELATIVE to ‖b‖ and the inner products, which
    # are quadratic in the RHS amplitude, cannot underflow for a tiny ‖b‖. The
    # previous absolute test (`norm(r) < tol`) returned x = 0 untouched whenever
    # ‖b‖ < tol, and the error of every other solve scaled with ‖b‖.
    r ./= bnorm

    # Apply preconditioner: z = M^{-1} * r
    _apply_preconditioner!(z, preconditioner, r)

    # p = z
    copyto!(p, z)

    # rz_old = r' * z
    rz_old = dot(r, z)
    # ‖p‖²_M from the CG recurrence ‖p_{k+1}‖²_M = r'z + |β|²‖p_k‖²_M — a free
    # scale for the breakdown screen below.
    pnorm2_M = abs(rz_old)
    r_norm = one(bnorm)  # ‖r‖/‖b‖

    for iter in 1:maxiter
        # Ap = A * p
        mul!(Ap, A, p)

        # alpha = rz_old / (p' * Ap)
        pAp = dot(p, Ap)
        # Zero-curvature breakdown, judged RELATIVE to |p'Ap| ≤ ‖p‖·‖Ap‖. The
        # previous absolute |p'Ap| < 1e-30 depended on the scale of A and b (it
        # fired for every ‖b‖ ≲ 1e-15). The two extra norms are only computed
        # when the free ‖p‖²_M screen flags a candidate.
        ϵ = eps(real(typeof(pAp)))
        if abs(pAp) <= ϵ * pnorm2_M && abs(pAp) <= ϵ * norm(p) * norm(Ap)
            error("CuIterativeCG broke down at iteration $iter: p'Ap ≈ 0 relative to " *
                  "‖p‖·‖Ap‖. " * _CG_OPERATOR_HINT)
        end
        alpha = rz_old / pAp

        # x = x + alpha * p (using helper for GPU AXPY)
        axpy!(alpha, p, x)

        # r = r - alpha * Ap (using helper for GPU AXPY)
        axpy!(-alpha, Ap, r)

        # Check convergence
        r_norm = norm(r)
        if r_norm <= tol
            @debug "CG converged in $iter iterations, relative residual = $r_norm"
            x .*= bnorm
            return x
        end

        # z = M^{-1} * r
        _apply_preconditioner!(z, preconditioner, r)

        # rz_new = r' * z
        rz_new = dot(r, z)

        # beta = rz_new / rz_old
        beta = rz_new / rz_old

        # p = z + beta * p
        p .= z .+ beta .* p

        pnorm2_M = abs(rz_new) + abs2(beta) * pnorm2_M
        rz_old = rz_new
    end

    # Non-convergence is a HARD error, not a silent last-iterate return: CG only
    # converges for Hermitian positive-definite operators. The coupled
    # Fourier–Chebyshev implicit tau operator (M + dt·a_ii·L) is non-Hermitian/
    # indefinite (Chebyshev differentiation matrices are non-symmetric and the
    # tau/BC rows break symmetry), so returning the last iterate would feed a
    # silently wrong solution back into the timestep. Use a direct GPU solver
    # (CuSparseLU / CuDenseLU) or GMRES for such systems.
    error("CuIterativeCG did not converge in $(maxiter) iterations (relative residual " *
          "$(r_norm) > tol = $(tol)). " * _CG_OPERATOR_HINT)
end

_apply_preconditioner!(z, M_inv, r) = error("Unsupported preconditioner type $(typeof(M_inv))")

# ============================================================================
# GPU Iterative Solver: GMRES
# ============================================================================

struct _GMRESWorkspace{V,T,R}
    b::V
    r::V
    w::V
    tmp::V
    basis::Vector{V}
    H::Matrix{T}    # host Hessenberg, reduced in place to triangular R by Givens rotations
    g::Vector{T}    # host rotated residual vector; overwritten by the least-squares solution
    cs::Vector{R}   # host Givens cosines (real)
    sn::Vector{T}   # host Givens sines
end

function _gmres_workspace!(s, prototype)
    ws = s.workspace
    if ws === nothing || length(ws.b) != s.n || length(ws.basis) != s.restart + 1 ||
       is_gpu_array(ws.b) != is_gpu_array(prototype)
        T = eltype(s.A_csr)
        m = s.restart
        vec() = similar(prototype, T, s.n)
        s.workspace = _GMRESWorkspace(vec(), vec(), vec(), vec(),
            [vec() for _ in 1:(m+1)], zeros(T, m+1, m), zeros(T, m+1),
            zeros(real(T), m), zeros(T, m))
    end
    return s.workspace
end

"""
    CuIterativeGMRES <: AbstractMatSolver

GMRES solver on GPU.
Best for large non-symmetric systems.

Uses restarted GMRES with configurable restart parameter and RIGHT
preconditioning (`A·M⁻¹·u = b`, `x = M⁻¹·u`), so the residual GMRES minimizes
and monitors is the true residual `b - A*x`, not the preconditioned one.
Matrix-vector products use cuSPARSE for efficiency.

# Arguments
- `matrix`: Sparse matrix (can be non-symmetric)
- `tol`: RELATIVE tolerance on the true residual (default: 1e-10). The solve has
  converged when `‖b - A*x‖ ≤ tol·‖b‖`, recomputed explicitly after every restart
  cycle (including the last). `b = 0` returns `x = 0`.
- `maxiter`: Maximum total iterations, i.e. Arnoldi steps over all restart cycles
  (default: 1000)
- `restart`: Restart parameter (default: 30)
- `preconditioner`: Preconditioner definition:
  - `:jacobi` (default): Diagonal preconditioner, good for diagonally dominant matrices.
    Zero diagonal entries (|aᵢᵢ| ≤ 1e-14, e.g. tau/boundary rows) are treated as 1.
  - `:ilu0`: ILU(0) via cuSPARSE, much better for ill-conditioned matrices
  - `:ic0`: IC(0) via cuSPARSE, for SPD matrices only (use CG instead if SPD)
  - `:none`: No preconditioning
  - `Vector`: Explicit diagonal inverse values
  - `Function`: Custom preconditioner `(out, r) -> out = M^{-1}*r`

`solve` throws when the tolerance is not met within `maxiter` iterations; the
last iterate is never returned as if it were a solution.

# Example
```julia
# Standard GMRES with Jacobi preconditioning
solver = CuIterativeGMRES(A; preconditioner=:jacobi, restart=50)

# For tough non-symmetric systems (recommended)
solver = CuIterativeGMRES(A; preconditioner=:ilu0, tol=1e-12)
```
"""
mutable struct CuIterativeGMRES{T,P<:AbstractPreconditioner} <: AbstractMatSolver
    A_csr::Any              # CuSparseMatrixCSR
    preconditioner::P
    tol::Float64
    maxiter::Int
    restart::Int            # Restart parameter
    n::Int
    workspace::Any
    prototype::Any

    function CuIterativeGMRES{T,P}(A_csr, preconditioner::P, tol::Float64, maxiter::Int,
                                    restart::Int, n::Int, prototype=nothing) where {T, P<:AbstractPreconditioner}
        # A zero-length cycle never advances the iteration count: the solve would spin.
        restart >= 1 || throw(ArgumentError("CuIterativeGMRES: restart must be ≥ 1, got $restart"))
        new{T,P}(A_csr, preconditioner, tol, maxiter, restart, n, nothing, prototype)
    end
end

function CuIterativeGMRES(matrix::AbstractMatrix;
                          tol::Real=1e-10,
                          maxiter::Int=1000,
                          restart::Int=30,
                          preconditioner=:jacobi,   # Symbol, vector of diagonal inverses, function or nothing
                          kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)

    # Convert to CSR on GPU using helper
    A_sparse = matrix isa SparseMatrixCSC ? matrix : sparse(matrix)
    A_csr = _gpu_sparse_csr(A_sparse, T)

    M_inv = _build_preconditioner(A_sparse, preconditioner, T)

    return CuIterativeGMRES{T, typeof(M_inv)}(A_csr, M_inv, Float64(tol), maxiter, restart, n, _gpu_zeros(T, 0))
end

function MatSolvers.solve(s::CuIterativeGMRES{T}, rhs::AbstractVector) where T
    prototype = s.workspace === nothing ? s.prototype : s.workspace.b
    result = prototype === nothing ? _gpu_zeros(T, s.n) : similar(prototype, T, s.n)
    return MatSolvers.solve!(result, s, rhs)
end

function MatSolvers.solve!(x::AbstractVector, s::CuIterativeGMRES, rhs::AbstractVector)
    length(rhs) == s.n && length(x) == s.n || throw(DimensionMismatch("GMRES vector length"))
    prototype = s.prototype === nothing ? x : s.prototype
    ws = _gmres_workspace!(s, prototype)
    if (is_gpu_array(ws.b) && !is_gpu_array(x)) || eltype(x) != eltype(ws.b)
        return copyto!(x, MatSolvers.solve(s, rhs))
    end
    copyto!(ws.b, rhs)
    fill!(x, zero(eltype(x)))
    return _gmres_solve_workspace!(x, s.A_csr, s.preconditioner, s.tol, s.maxiter, s.restart, ws)
end

# Givens rotation G = [c s; -conj(s) c] (real c) with G·[f; h] = [ρ; 0], for the
# real, non-negative subdiagonal h = H[j+1, j] of the Arnoldi Hessenberg matrix.
@inline function _gmres_givens(f::T, h::Real) where {T<:Number}
    iszero(h) && return one(real(T)), zero(T), f
    af = abs(f)
    iszero(af) && return zero(real(T)), one(T), T(h)
    nrm = hypot(af, h)
    phase = f / af
    return af / nrm, phase * (h / nrm), phase * nrm
end

function _gmres_solve_workspace!(x, A, preconditioner, tol, maxiter, m, ws::_GMRESWorkspace)
    b, r, w, tmp = ws.b, ws.r, ws.w, ws.tmp
    V, H, g, cs, sn = ws.basis, ws.H, ws.g, ws.cs, ws.sn
    # RIGHT preconditioning: minimize ‖b - A·M⁻¹·u‖ and set x = M⁻¹·u, so the
    # least-squares residual |g[j+1]| is the TRUE residual ‖b - A*x‖ (in exact
    # arithmetic). The previous left-preconditioned form stopped on ‖M⁻¹(b - A*x)‖,
    # which cannot see rows that M⁻¹ shrinks — or, under the old zero-for-zero
    # Jacobi rule, deleted outright: the tau/BC rows of a tau system. It then
    # returned a wrong solution as "converged".
    # x = 0 on entry (solve! clears it), so r = b and `tol` is relative to ‖b‖.
    copyto!(r, b)
    bnorm = _checked_rhs_norm(r, "CuIterativeGMRES")
    threshold = tol * bnorm
    rnorm = bnorm
    total_iters = 0
    # Outer iteration (restart cycles). The EXPLICIT residual is tested before
    # every cycle and after the last one (b = 0 never iterates; a NaN residual
    # from a breakdown also ends the loop). The final cycle is truncated when
    # maxiter is not a multiple of restart, so maxiter < restart still iterates.
    while rnorm > threshold && total_iters < maxiter
        # Cycle length: full restart, or fewer on the final truncated cycle.
        mj = min(m, maxiter - total_iters)

        # Arnoldi process on A·M⁻¹, from v₁ = r/‖r‖
        V[1] .= r ./ rnorm
        fill!(g, zero(eltype(g)))
        g[1] = rnorm

        k = 0  # subspace dimension actually built this cycle
        for j in 1:mj
            k = j
            total_iters += 1

            # w = A * M^{-1} * v_j (right preconditioning)
            _apply_preconditioner!(tmp, preconditioner, V[j])
            mul!(w, A, tmp)

            # Modified Gram-Schmidt
            colnorm2 = zero(real(eltype(H)))
            for i in 1:j
                H[i, j] = dot(V[i], w)
                w .-= H[i, j] .* V[i]
                colnorm2 += abs2(H[i, j])
            end
            hnext = norm(w)

            # Apply the previous rotations to the new column, then annihilate its
            # subdiagonal; g holds the rotated right-hand side ‖r‖·e₁.
            for i in 1:(j - 1)
                hi = H[i, j]
                H[i, j] = cs[i] * hi + sn[i] * H[i + 1, j]
                H[i + 1, j] = -conj(sn[i]) * hi + cs[i] * H[i + 1, j]
            end
            cs[j], sn[j], H[j, j] = _gmres_givens(H[j, j], hnext)
            g[j + 1] = -conj(sn[j]) * g[j]
            g[j] = cs[j] * g[j]

            # |g[j+1]| is the residual norm of this cycle's least-squares iterate.
            # hnext ≈ 0 relative to ‖A·M⁻¹·v_j‖ is a happy breakdown: the Krylov
            # subspace is invariant, so stop before normalizing a null vector.
            if abs(g[j + 1]) <= threshold || j == mj ||
               hnext <= eps(real(eltype(H))) * sqrt(colnorm2 + hnext^2)
                break
            end

            V[j + 1] .= w ./ hnext
        end

        # Back substitution R·y = g[1:k] (small triangular system, on CPU);
        # y overwrites g.
        for i in k:-1:1
            acc = g[i]
            for l in (i + 1):k
                acc -= H[i, l] * g[l]
            end
            g[i] = acc / H[i, i]
        end

        # Update solution: x = x + M⁻¹·(V * y) (CPU y values avoid GPU scalar indexing)
        r .= g[1] .* V[1]
        for i in 2:k
            axpy!(g[i], V[i], r)
        end
        _apply_preconditioner!(tmp, preconditioner, r)
        x .+= tmp

        # Explicit true residual r = b - A*x: the convergence decision never rests
        # on the Arnoldi recurrence alone (rounding, near-breakdown).
        mul!(w, A, x)
        r .= b .- w
        rnorm = norm(r)
    end

    if rnorm <= threshold
        @debug "GMRES converged in $total_iters iterations, relative residual = $(rnorm / bnorm)"
        return x
    end
    # Non-convergence is a HARD error, as for CG: the last iterate of a stalled or
    # broken-down GMRES is not a solution, and the stepper must not consume it.
    error("CuIterativeGMRES did not converge in $(maxiter) iterations (restart = $m): " *
          "relative residual ‖b - A*x‖/‖b‖ = $(rnorm / bnorm) > tol = $(tol). Increase " *
          "maxiter or restart, use a stronger preconditioner (:ilu0), or use a direct " *
          "GPU solver (CuSparseLU/CuDenseLU).")
end

# ============================================================================
# Hybrid CPU-GPU Solver
# ============================================================================

"""
    HybridSolver <: AbstractMatSolver

Automatically selects CPU or GPU solver based on problem characteristics.

Decision criteria:
- Matrix size: GPU for n > threshold
- Matrix density: GPU for dense, CPU for very sparse
- Available GPU memory
"""
struct HybridSolver{T} <: AbstractMatSolver
    cpu_solver::Any
    gpu_solver::Any
    use_gpu::Bool
    threshold::Int
    mode::Symbol
end

function HybridSolver(matrix::AbstractMatrix;
                      threshold::Int=50000,
                      gpu_solver_type::Symbol=:cg,
                      mode::Symbol=:auto,
                      gpu_preconditioner=:jacobi,
                      kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)

    mode ∈ (:auto, :cpu, :gpu) || throw(ArgumentError("HybridSolver mode must be :auto, :cpu, or :gpu"))
    # Validate up front: an unrecognized type previously matched no construction
    # branch yet still set use_gpu=true with gpu_solver=nothing — the solver then
    # LOGGED "Using GPU" while silently solving on CPU.
    gpu_solver_type ∈ (:cg, :gmres, :lu) || throw(ArgumentError(
        "HybridSolver gpu_solver_type must be :cg, :gmres, or :lu, got $(repr(gpu_solver_type))"))

    # Build CPU solver (always available)
    cpu_solver = SparseLUSolver(matrix)

    # Decide whether to use GPU
    use_gpu = false
    gpu_solver = nothing

    use_gpu_requested = (mode === :gpu) || (mode === :auto && CUDA_AVAILABLE[] && n > threshold)

    if use_gpu_requested
        try
            if gpu_solver_type == :cg
                gpu_solver = CuIterativeCG(matrix; preconditioner=gpu_preconditioner, kwargs...)
            elseif gpu_solver_type == :gmres
                gpu_solver = CuIterativeGMRES(matrix; preconditioner=gpu_preconditioner, kwargs...)
            else  # :lu (validated above)
                gpu_solver = CuDenseLU(matrix)
            end
            use_gpu = true
            @info "HybridSolver: Using GPU $(gpu_solver_type) (mode=$(mode), n=$n, threshold=$threshold)"
        catch e
            # An EXPLICIT :gpu request must not silently degrade to CPU — that
            # is the implicit-CPU-fallback pattern the framework forbids
            # (see PR #74). Only :auto may fall back, and it says so.
            mode === :gpu && rethrow()
            @warn "HybridSolver(mode=:auto): GPU solver creation failed, using CPU solver instead" exception=(e, catch_backtrace()) maxlog=1
        end
    else
        @debug "HybridSolver: Using CPU solver (mode=$(mode), n=$n, threshold=$threshold)"
    end

    return HybridSolver{T}(cpu_solver, gpu_solver, use_gpu, threshold, mode)
end

function MatSolvers.solve(s::HybridSolver, rhs::AbstractVector)
    if s.use_gpu && s.gpu_solver !== nothing
        result = solve(s.gpu_solver, rhs)
        # Return as CPU array for compatibility using helper
        return _is_gpu_array(result) ? Array(result) : result
    else
        return solve(s.cpu_solver, rhs)
    end
end

# ============================================================================
# Registration
# ============================================================================

"""
    StrictGPUSolver

Registry factory for the solver name "gpu": constructs a `HybridSolver` pinned
to `mode=:gpu`, so a user who explicitly requested "gpu" either gets a working
GPU solver or a loud error — never a silent CPU solve. (Previously "gpu" mapped
to `HybridSolver` with its default `mode=:auto`, which silently ran 100% CPU
whenever CUDA was unavailable or the system was below the size threshold.)
Use the name "hybrid" for the auto CPU/GPU selection behavior.

The type itself is never instantiated — its constructor-style call returns the
underlying `HybridSolver` (the registry requires a `Type` entry).
"""
struct StrictGPUSolver <: AbstractMatSolver end
StrictGPUSolver(matrix::AbstractMatrix; kwargs...) = HybridSolver(matrix; kwargs..., mode=:gpu)

function _register_gpu_solvers()
    register_solver("cuda_dense", CuDenseLU)
    register_solver("cuda_lu", CuDenseLU)
    register_solver("cuda_sparse", CuSparseLU)
    register_solver("cuda_cg", CuIterativeCG)
    register_solver("cuda_gmres", CuIterativeGMRES)
    register_solver("gpu", StrictGPUSolver)   # explicit "gpu" = strict, no silent CPU
    register_solver("hybrid", HybridSolver)   # "hybrid" = auto CPU/GPU selection
    @debug "Registered GPU matrix solvers"
end

# Export GPU solvers
export CuDenseLU, CuSparseLU, CuIterativeCG, CuIterativeGMRES, HybridSolver
export CUDA_AVAILABLE

# ============================================================================
# Preconditioner helper functions
# ============================================================================

function _build_preconditioner(A::SparseMatrixCSC, preconditioner, T::Type)
    if preconditioner isa Symbol
        if preconditioner === :none
            return NoPreconditioner()
        elseif preconditioner === :jacobi
            # Transfer diagonal to GPU using helper
            return DiagonalPreconditioner(_gpu_array(_jacobi_diag_inv(A, T), T))
        elseif preconditioner === :ilu0
            return _build_ilu0_preconditioner(A, T)
        elseif preconditioner === :ic0
            return _build_ic0_preconditioner(A, T)
        else
            error("Unsupported preconditioner symbol $(preconditioner). Use :none, :jacobi, :ilu0, :ic0, a vector, or a custom function.")
        end
    elseif preconditioner === nothing
        return NoPreconditioner()
    elseif preconditioner isa AbstractVector
        length(preconditioner) == size(A, 1) || error("Preconditioner vector has wrong length")
        # Transfer preconditioner vector to GPU using helper
        return DiagonalPreconditioner(_gpu_array(preconditioner, T))
    elseif preconditioner isa Function
        return CustomPreconditioner(preconditioner)
    else
        error("Unsupported preconditioner type $(typeof(preconditioner))")
    end
end

"""
    _jacobi_diag_inv(A, T) -> Vector{T}

Host inverse diagonal for the `:jacobi` preconditioner. A (numerically) zero
diagonal entry maps to 1, so that row passes through UNSCALED. Tau/boundary rows
and saddle-point constraint rows have a structurally zero diagonal; mapping them
to 0, as before, made M⁻¹ singular — those rows dropped out of the preconditioned
system entirely and GMRES returned a wrong solution as converged.
"""
function _jacobi_diag_inv(A::AbstractMatrix, ::Type{T}) where {T}
    d = Vector(diag(A))  # dense: indexing the SparseVector diag(A) is a search per entry
    diag_inv = Vector{T}(undef, length(d))
    @inbounds for i in eachindex(diag_inv)
        diag_inv[i] = abs(d[i]) > 1e-14 ? T(1) / T(d[i]) : one(T)
    end
    return diag_inv
end

"""
    _build_ilu0_preconditioner(A::SparseMatrixCSC, T::Type)

Build an ILU(0) preconditioner using cuSPARSE.
Computes incomplete LU factorization with zero fill-in on GPU.
"""
function _build_ilu0_preconditioner(A::SparseMatrixCSC, T::Type)
    n = size(A, 1)
    @assert size(A, 2) == n "Matrix must be square for ILU(0)"

    # Convert to CSR format on GPU using helper
    A_csr = _gpu_sparse_csr(A, T)

    # Compute ILU(0) factorization using helper
    # ilu02 returns the matrix with L and U factors:
    # - L is stored below diagonal (with implicit unit diagonal)
    # - U is stored on and above diagonal
    LU_csr = _gpu_ilu0(A_csr)

    # Allocate workspace for triangular solve intermediate using helper
    tmp = _gpu_zeros(T, n)

    return ILU0Preconditioner{T}(LU_csr, tmp, n)
end

"""
    _build_ic0_preconditioner(A::SparseMatrixCSC, T::Type)

Build an IC(0) (Incomplete Cholesky) preconditioner using cuSPARSE.
Computes incomplete Cholesky factorization with zero fill-in on GPU.

**Important**: The input matrix must be symmetric positive definite (SPD).
The function extracts the lower triangular part of A for factorization.
"""
function _build_ic0_preconditioner(A::SparseMatrixCSC, T::Type)
    n = size(A, 1)
    @assert size(A, 2) == n "Matrix must be square for IC(0)"

    # For IC(0), we need to work with a symmetric matrix
    # cuSPARSE ic02 expects the lower triangular part in CSR format
    # Extract lower triangular part to ensure symmetry
    A_lower = tril(A)

    # Convert to CSR format on GPU using helper
    A_csr = _gpu_sparse_csr(A_lower, T)

    # Compute IC(0) factorization using helper
    # ic02 returns the matrix with L factor:
    # - L is stored in the lower triangular part with explicit diagonal
    # - A ≈ LL^T
    L_csr = _gpu_ic0(A_csr)

    # Allocate workspace for triangular solve intermediate using helper
    tmp = _gpu_zeros(T, n)

    return IC0Preconditioner{T}(L_csr, tmp, n)
end

function _apply_preconditioner!(z, ::NoPreconditioner, r)
    copyto!(z, r)
end

function _apply_preconditioner!(z, M::DiagonalPreconditioner, r)
    z .= M.diag_inv .* r
end

function _apply_preconditioner!(z, M::CustomPreconditioner, r)
    M.apply!(z, r)
end

"""
    _apply_preconditioner!(z, M::ILU0Preconditioner, r)

Apply ILU(0) preconditioner: z = (LU)^{-1} * r

This performs two triangular solves:
1. L * tmp = r  (lower triangular, unit diagonal)
2. U * z = tmp  (upper triangular)

Uses cuSPARSE triangular solves via ldiv! for GPU-accelerated operations.
For CSR format: L has unit diagonal (below), U has non-unit diagonal (on and above).
"""
function _apply_preconditioner!(z, M::ILU0Preconditioner{T}, r) where T
    # For CSR matrices from ilu02:
    # - L is lower triangular with implicit unit diagonal
    # - U is upper triangular with explicit diagonal

    # Step 1: Forward substitution - solve L * tmp = r
    # UnitLowerTriangular indicates L has unit diagonal (ones on diagonal, not stored)
    ldiv!(M.tmp, UnitLowerTriangular(M.LU_csr), r)

    # Step 2: Backward substitution - solve U * z = tmp
    # UpperTriangular indicates U has explicit diagonal values
    ldiv!(z, UpperTriangular(M.LU_csr), M.tmp)
end

"""
    _apply_preconditioner!(z, M::IC0Preconditioner, r)

Apply IC(0) preconditioner: z = (LL^T)^{-1} * r

This performs two triangular solves:
1. L * tmp = r   (forward substitution, lower triangular)
2. L^T * z = tmp (backward substitution, upper triangular)

Uses cuSPARSE triangular solves via ldiv! for GPU-accelerated operations.
For CSR format from ic02: L has explicit diagonal values.

Note: IC(0) is more efficient than ILU(0) for SPD matrices because:
- Only one factor needs to be stored (L instead of L and U)
- The transpose L^T can be computed implicitly from L
"""
function _apply_preconditioner!(z, M::IC0Preconditioner{T}, r) where T
    # For CSR matrices from ic02:
    # - L is lower triangular with explicit diagonal

    # Step 1: Forward substitution - solve L * tmp = r
    # LowerTriangular indicates L has explicit diagonal values
    ldiv!(M.tmp, LowerTriangular(M.L_csr), r)

    # Step 2: Backward substitution - solve L^T * z = tmp
    # For L^T, we use the transpose. With CSR, transpose gives us upper triangular.
    # cuSPARSE handles the transpose efficiently.
    ldiv!(z, LowerTriangular(M.L_csr)', M.tmp)
end
