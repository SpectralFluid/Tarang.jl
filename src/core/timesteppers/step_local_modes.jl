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
