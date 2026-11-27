# ============================================================================
# Additional IMEX Timestepper Step Functions
# ============================================================================
#
# The global fallback operates on one `(M, L)` matrix pair spanning all
# variables and modes, factorizing `(a*M + b*L)` once and reusing the LU across
# steps. This is the legacy path taken when no per-subproblem decomposition is
# available (see step_subproblem_rk.jl for the preferred per-Fourier-mode path).
# MCNAB2 and CNLF2 also use per-subproblem history when that decomposition is
# available, or field history for explicit GPU/MPI problems. A simpler scheme
# supplies their first step. Sign convention: `M dX/dt + L X = F`.

"""
    _prepare_global_matrix_timestep!(state, solver, method_name, fallback_name, fallback_step!)
        -> (L_matrix, M_matrix, used_fallback)

Shared entry guard for the implicit global-matrix steppers. Fetches the global
`L`/`M` matrices and, if either required matrix is missing, logs the reason and
runs `fallback_step!` (an explicit scheme). `used_fallback === true` signals the
caller to return immediately — the step has already been taken by the fallback.
"""
function _prepare_global_matrix_timestep!(state::TimestepperState,
                                          solver::InitialValueSolver,
                                          method_name::String,
                                          fallback_name::String,
                                          fallback_step!::F) where {F}
    L_matrix, M_matrix = _global_matrix_implicit_matrices(solver)
    reason = _global_matrix_implicit_missing_matrix_reason(L_matrix, M_matrix)
    if reason !== nothing
        _log_global_matrix_implicit_matrix_fallback(method_name, reason, fallback_name)
        fallback_step!(state, solver)
        return L_matrix, M_matrix, true
    end

    return L_matrix, M_matrix, false
end

"""
    Modified Crank-Nicolson Adams-Bashforth 2nd order step.

    Uses three implicit time levels. At constant timestep their weights are
    (9/16, 6/16, 1/16); explicit extrapolation uses AB2 weights (3/2, -1/2).
    Variable-step coefficients depend on the current/previous timestep ratio.
    """
function step_mcnab2!(state::TimestepperState, solver::InitialValueSolver)

    current_state = state.history[end]
    dt = state.dt

    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        # Subproblem steps keep one live field state and retain older M*X/L*X/F
        # values in per-mode rings. A state.history-length gate never exits
        # startup on this path; inspect the ring history instead.
        if _sp_multistep_history_depth(state) < 1
            step_cnab1!(state, solver)
            return
        end
        a, b, c = _mcnab2_coefs(dt, get_previous_timestep(state))
        step_subproblem_multistep!(state, solver, sps, a, b, c)
        return
    end

    # GPU / MPI without subproblems: no global matrix exists. With no implicit
    # operator the implicit stencil vanishes and MCNAB2 IS CNAB2, so take the
    # matrix-free field path (see step_multistep_field.jl). This must precede
    # the startup gate below: the field path keeps a one-entry `state.history`,
    # so `length(state.history) < 2` stayed true forever and every device step
    # silently ran as first-order CNAB1.
    _try_step_explicit_multistep_field!(state, solver, :cnab2) && return

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, :MX_history)
        state.timestepper_data[:MX_history] = Vector{Vector{ComplexF64}}()
        state.timestepper_data[:LX_history] = Vector{Vector{ComplexF64}}()
        state.timestepper_data[:F_history] = Vector{Vector{ComplexF64}}()
        state.timestepper_data[:iteration] = 0
    end

    iteration = state.timestepper_data[:iteration]

    # Check if we have enough history for MCNAB2
    if iteration < 1 || length(state.history) < 2
        @debug "MCNAB2 requires iteration >= 1, falling back to CNAB1"
        step_cnab1!(state, solver)
        state.timestepper_data[:iteration] += 1
        return
    end

    L_matrix, M_matrix, used_fallback =
        _prepare_global_matrix_timestep!(state, solver, "MCNAB2", "CNAB2", step_cnab2!)
    used_fallback && return

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)

    # Three-level MCNAB2 coefficients
    a, b, c = _mcnab2_coefs(dt_current, dt_previous)

    try
        X_current = _timestep_fields_vector!(state, :mcnab2_X_current_vec, current_state)

        MX_current = _timestep_matvec!(state, :mcnab2_MX_current_vec, M_matrix, X_current)
        LX_current = _timestep_matvec!(state, :mcnab2_LX_current_vec, L_matrix, X_current)

        F_current_vec = _timestep_global_rhs_vector!(state, :mcnab2_F_current_vec,
                                                       solver, current_state, solver.sim_time)

        MX_history = state.timestepper_data[:MX_history]::Vector{Vector{ComplexF64}}
        LX_history = state.timestepper_data[:LX_history]::Vector{Vector{ComplexF64}}
        F_history  = state.timestepper_data[:F_history]::Vector{Vector{ComplexF64}}

        # Build RHS: c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - b[1]*LX[0]
        rhs = _timestep_vector_buffer!(state, :mcnab2_rhs_vec, length(F_current_vec))
        @. rhs = c[2] * F_current_vec
        if !isempty(F_history)
            rhs .+= c[3] * F_history[1]
        end
        rhs .-= a[2] * MX_current
        rhs .-= b[2] * LX_current
        if length(b) >= 3 && !isempty(LX_history)
            rhs .-= b[3] * LX_history[1]
        end
        zero_rows = _global_zero_mass_rows!(state, M_matrix)
        _apply_global_algebraic_rhs!(rhs, state, solver.problem, zero_rows, b[1])

        # Build and solve LHS: (a[0]*M + b[0]*L) X = RHS
        # Cache single LU factorization; recompute only when (a[1],b[1]) changes.
        cache_key = (a[1], b[1])
        if get(state.timestepper_data, :mcnab2_lhs_key, nothing) !== cache_key
            state.timestepper_data[:mcnab2_lhs_factor] = _factorize_timestep_matrix(a[1] * M_matrix + b[1] * L_matrix)
            state.timestepper_data[:mcnab2_lhs_key] = cache_key
        end
        X_new = _timestep_vector_buffer!(state, :mcnab2_X_new_vec, length(rhs))
        _timestep_ldiv!(X_new, state.timestepper_data[:mcnab2_lhs_factor], rhs)

        # Update state
        _push_vector_state!(state.history, X_new, current_state, 4)
        _prepend_history_buffer!(MX_history, MX_current, 2)
        _prepend_history_buffer!(LX_history, LX_current, 2)
        _prepend_history_buffer!(F_history, F_current_vec, 2)
        state.timestepper_data[:iteration] += 1

        @debug "MCNAB2 step completed: dt=$dt_current, iteration=$(state.timestepper_data[:iteration])"

    catch e
        # GPU state: rethrow. Substituting another scheme here would SWALLOW the
        # framework's deliberate loud refusals (GPU vector-pack guard, GPU
        # factorization errors) and silently run CNAB2 in place of MCNAB2 every
        # step. MCNAB2 is a global-matrix method with no GPU path — on GPU use
        # CNAB2/SBDF2 (subproblem-backed) or DiagonalIMEX_SBDF2 explicitly.
        any(_field_uses_gpu, solver.state) && rethrow()
        @warn "MCNAB2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end
end

"""Crank–Nicolson/leapfrog over the interval from the previous to the next time."""
function step_cnlf2!(state::TimestepperState, solver::InitialValueSolver)
    dt = state.dt
    sps = _timestepper_subproblems(solver)
    if sps !== nothing
        _sp_multistep_history_depth(state) < 1 && return step_cnab1!(state, solver)
        a, b, c = _cnlf2_coefs(dt, get_previous_timestep(state))
        return step_subproblem_multistep!(state, solver, sps, a, b, c)
    end

    _try_step_explicit_multistep_field!(state, solver, :cnlf2) && return
    _init_global_multistep_history!(state, :iteration)
    if isempty(state.timestepper_data[:F_history])
        step_cnab1!(state, solver)
        state.timestepper_data[:iteration] += 1
        return
    end

    L, M, used_fallback =
        _prepare_global_matrix_timestep!(state, solver, "CNLF2", "RK222", step_rk222!)
    used_fallback && return
    a, b, c = _cnlf2_coefs(dt, get_previous_timestep(state))
    _global_multistep_core!(state, solver, last(state.history), L, M, a, b, c;
                            mx_depth=1, f_depth=1, state_cap=4, iter_key=:iteration, name="CNLF2")
    return nothing
end

"""
    step_rksmr!(state, solver)

Spalart–Moser–Rogers semi-implicit (IMEX) RK3 step. RKSMR now carries an ARK
Butcher tableau (see `RKSMR` in types.jl), so — like RKGFY — it
delegates to the generic IMEX driver `step_rk_imex!`, which treats the nonlinear
term `F` explicitly and the stiff linear operator `L` implicitly (M/L matrices,
per-mode subproblems, distributed diagonal IMEX). Failures propagate rather than
falling back to an explicit method that would drop `L`, `M`, and constraints.

Previously this was a fully-EXPLICIT SSP-RK3 that ignored `L` entirely (only
emitting a warning), so diffusion-dominated problems were integrated without any
implicit treatment and went unstable.
"""
function step_rksmr!(state::TimestepperState, solver::InitialValueSolver)
    step_rk_imex!(state, solver)
end

# `step_rkgfy!` is a named scheme entry point that carries no bespoke logic — it
# delegates to the generic `step_rk_imex!` so every IMEX RK variant shares one
# M/L-handling implementation and failure policy.
function step_rkgfy!(state::TimestepperState, solver::InitialValueSolver)
    step_rk_imex!(state, solver)
end
