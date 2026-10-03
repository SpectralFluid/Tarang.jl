# ── Batched dense LU over Fourier modes ──────────────────────────────────────
#
# Every mode's stage matrix `(M + dt*a_ii*L)` factored and solved in one call
# instead of one per mode. Dense rather than sparse because the per-mode
# matrices measure ~28% dense with full bandwidth (Chebyshev tau rows), so the
# sparse structure buys almost nothing while costing a per-mode launch.
#
# Partial pivoting can differ across modes despite their shared sparsity pattern.

"""
    BatchedDenseLU(A)

Factor and solve `A[:, :, m] * x = b` for every mode `m` in one call.
Fully overwrite `A` (see `batched_assemble_lhs!`) before each `batched_factor!`,
including after a timestep change. CUDA factors `A` in place; CPU currently
copies it, but callers must not rely on that. Refactoring existing LU factors
instead of a reassembled matrix can silently produce a wrong solution.

The storage parameter `AT` lets the CUDA extension specialize on
`BatchedDenseLU{<:CuArray}` without replacing the generic CPU method.
"""
mutable struct BatchedDenseLU{AT<:AbstractArray{ComplexF64, 3}}
    A::AT
    pivots::Any
    info::Any
    factored::Bool
    backend_workspace::Any
end

BatchedDenseLU(A::AbstractArray{ComplexF64, 3}) =
    BatchedDenseLU(A, nothing, nothing, false, nothing)

BatchedDenseLU(A::AbstractArray{ComplexF64, 3}, pivots, info, factored::Bool) =
    BatchedDenseLU(A, pivots, info, factored, nothing)

BatchedDenseLU{AT}(A::AT, pivots, info, factored::Bool) where {AT} =
    BatchedDenseLU{AT}(A, pivots, info, factored, nothing)

"""
    batched_factor!(s::BatchedDenseLU) -> s

LU-factor every mode in place.

Raises if any mode is singular, naming it. This check is not optional: the
batched LAPACK/CUBLAS entry points report per-matrix status in an `info` array
and return normally regardless, so an unchecked singular mode yields buffer
contents that read as a plausible solution and propagate through the timestep
undetected.
"""
function batched_factor!(s::BatchedDenseLU)
    return _batched_factor_impl!(s)
end

# CPU reference path. The GPU method is added by the CUDA extension.
function _batched_factor_impl!(s::BatchedDenseLU)
    A = s.A
    n, _, nmodes = size(A)
    facts = Vector{Any}(undef, nmodes)
    for m in 1:nmodes
        F = lu(view(A, :, :, m); check=false)
        if !issuccess(F)
            error("BatchedDenseLU: mode $m of $nmodes is singular " *
                  "(order $n). A singular stage matrix usually means the " *
                  "problem is under-constrained at this Fourier mode — check " *
                  "the tau/BC rows for that mode.")
        end
        facts[m] = F
    end
    s.pivots = facts
    s.info = zeros(Int, nmodes)
    s.factored = true
    return s
end

"""
    batched_solve!(X, s::BatchedDenseLU, B) -> X

Solve every mode against the stored factorization. `X` and `B` are
`(n, nmodes)`, column `m` being mode `m`. `X` may alias `B`.
"""
function batched_solve!(X::AbstractMatrix{ComplexF64}, s::BatchedDenseLU,
                        B::AbstractMatrix{ComplexF64})
    s.factored || error("BatchedDenseLU: batched_solve! called before " *
                        "batched_factor!; the factorization is stale or absent")
    return _batched_solve_impl!(X, s, B)
end

function _batched_solve_impl!(X::AbstractMatrix{ComplexF64}, s::BatchedDenseLU,
                              B::AbstractMatrix{ComplexF64})
    X === B || copyto!(X, B)
    for m in axes(X, 2)
        ldiv!(s.pivots[m], view(X, :, m))
    end
    return X
end
