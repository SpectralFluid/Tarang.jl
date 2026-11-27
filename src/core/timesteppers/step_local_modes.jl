# CPU workers only consume prepared per-subproblem arrays and factors. Shared
# fields, lazy RHS/BC caches, layout changes and MPI calls belong to the caller.
# One worker owns a subproblem for the whole phase, including mutable solver
# scratch. A phase completes before factors can be refactored or fields written.

function _thread_local_modes(setting::Union{Nothing, Bool}, subproblems,
                             matsolver=MatSolvers.SparseLUSolver)
    setting === false && return false
    nthreads = Threads.nthreads(:default)
    nthreads > 1 || return false
    if setting === nothing
        # External solver constructors may keep shared scratch; only explicit
        # opt-in promises that their independent factors can run concurrently.
        constructor = matsolver isa Tuple ? first(matsolver) : matsolver
        constructor in (MatSolvers.SparseLUSolver, MatSolvers.DenseLUSolver,
                        MatSolvers.BandedLUSolver, MatSolvers.BlockDiagonalSolver,
                        MatSolvers.SPQRSolver, MatSolvers.WoodburySolver) || return false
        # Never reconfigure a process-global BLAS pool during a timestep.
        BLAS.get_num_threads() == 1 || return false
    end
    modes = 0
    work = 0
    for sp in subproblems
        sp.M_min === nothing && continue
        sp.dist.architecture isa CPU || return false
        modes += 1
        work += length(sp.M_min)  # conservative dense triangular-solve proxy
    end
    modes >= 2 || return false
    setting === true && return true
    return modes >= max(8, 2nthreads) && work >= 32768nthreads
end

"""Run disjoint local mode jobs, joining every worker before returning/throwing.

Use a bounded number of contiguous chunks, not one task per Fourier mode. The
caller decides whether the workload merits threading and must finish all shared
cache/field preparation first. This helper never changes BLAS or FFTW threads.
"""
function _foreach_local_mode!(f::F, n::Int, threaded::Bool) where F
    if !threaded || n < 2 || Threads.nthreads(:default) == 1
        for i in 1:n
            f(i)
        end
        return nothing
    end
    nchunks = min(n, Threads.nthreads(:default))
    @sync for chunk in 1:nchunks
        first = fld((chunk - 1) * n, nchunks) + 1
        last = fld(chunk * n, nchunks)
        Threads.@spawn for i in first:last
            f(i)
        end
    end
    return nothing
end

# Factors can have different concrete types (LU, QR, Woodbury) at different
# modes. Keep the outer slots reusable and dispatch only at each solve call.
function _local_mode_factor_slots!(state::TimestepperState, n::Int)
    slots = get!(() -> Vector{Any}(undef, n), state.timestepper_data,
                 :_local_mode_factors)::Vector{Any}
    length(slots) == n || resize!(slots, n)
    return slots
end

# ──────────────────────────────────────────────────────────────────────────────
# Rank-deficient per-mode systems on GPU
# ──────────────────────────────────────────────────────────────────────────────

const _GPU_PINNED_LHS_KEYS = "_gpu_pinned_lhs_keys"

"""
    _gpu_singular_factorization(sp, solver_type, A, solver_kwargs, what) -> solver or nothing

Device counterpart of the CPU sparse-QR fallback for a rank-deficient per-mode
system. The common case is an incompressible problem at a RealFourier Nyquist
mode: ∂x vanishes there, so nothing determines the pressure's constant
coefficient and its column is empty. The free unknowns are pinned on the host
(`MatSolvers.pin_free_columns`) and the system is factored again with the device
solver, which then returns the CPU's basic solution for a consistent right-hand
side. Returns `nothing` when `A` has full rank, so the caller reports the
original error; raises if the pinned system still cannot be factored.
"""
function _gpu_singular_factorization(sp, solver_type, A, solver_kwargs, what::AbstractString)
    A isa SparseMatrixCSC || return nothing
    pinned = MatSolvers.pin_free_columns(A)
    pinned === nothing && return nothing
    solver = try
        MatSolvers.solver_instance(solver_type, pinned.matrix; solver_kwargs...)
    catch err
        error("GPU factorization of the $what for subproblem group=$(sp.group) failed " *
              "even after pinning its $(length(pinned.free_columns)) undetermined unknown(s); " *
              "CPU fallback is disabled. Original error: $(sprint(showerror, err))")
    end
    @warn "Singular $what for subproblem group=$(sp.group) on the GPU: " *
          "$(length(pinned.free_columns)) unknown(s) are not determined by the equations. " *
          "Solving the rest with those fixed at zero, as the CPU sparse-QR fallback does. " *
          "Right-hand-side consistency is not checked on the device." maxlog=1
    return solver
end

# A pinned factor was built from a patched pattern. The symbolic-reuse
# `refactor!` path would feed it the unpatched matrix, so pinned keys rebuild.
_gpu_lhs_pinned(sp, key) =
    (keys = get(sp.matrices, _GPU_PINNED_LHS_KEYS, nothing); keys !== nothing && key in keys)
_mark_gpu_lhs_pinned!(sp, key) =
    push!(get!(() -> Set{Any}(), sp.matrices, _GPU_PINNED_LHS_KEYS)::Set{Any}, key)

"""
    _structurally_rank_deficient(sp) -> Bool

Whether every stage matrix `M + c·L` of this subproblem leaves part of its
right-hand side space unreachable, whatever the step size — rank below the row
count at two unrelated coefficients. Such a mode is undetermined by the
formulation itself; a stage matrix that is singular only at one step size is
not. Probed once, on first need, and cached on the subproblem.
"""
function _structurally_rank_deficient(sp)
    cached = get(sp.matrices, "_structurally_rank_deficient", nothing)
    cached === nothing || return cached::Bool
    M, L = sp.M_min, sp.L_min
    deficient = M !== nothing && L !== nothing && all((0.0173, 0.2391)) do c
        A = SparseMatrixCSC{ComplexF64, Int}(M + c * L)
        rank(qr(A)) < size(A, 1)
    end
    sp.matrices["_structurally_rank_deficient"] = deficient
    return deficient
end
