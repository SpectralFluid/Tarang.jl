# Route large global systems before applying M⁻¹ to a dense copy of L. The
# distributed diagonal path has already returned at each call site. Validate M
# first so this size fallback never silently accepts an unsupported DAE.
function _etd_oversized_fallback!(state::TimestepperState, solver::InitialValueSolver,
                                   L_matrix, M_matrix, fallback)
    size(L_matrix, 1) <= _ETD_DENSE_MAX_SIZE && return false
    _get_etd_mass_factor!(state, M_matrix)
    if _has_registered_stochastic_forcing(solver)
        # An independent white-noise increment belongs to exactly one step.
        @warn "ETD matrix exponential exceeds the dense size limit; using one-step IMEX RK222 for stochastic forcing" n=size(L_matrix, 1) maxlog=1
        step_rk_imex!(state, solver; ts=_RK222_SINGLETON)
        return true
    end
    @warn "ETD matrix exponential exceeds the dense size limit; using $(nameof(fallback))" n=size(L_matrix, 1) maxlog=1
    fallback(state, solver)
    return true
end

"""Return the shared ETD matrix-function cache for the current operator and `dt`."""
function _get_etd_phi!(state::TimestepperState, L_linear, dt::Float64)
    cache = state.timestepper_data
    cache_dt = get(cache, :etd_phi_dt, nothing)
    cache_L = get(cache, :etd_phi_L_source, nothing)
    if !haskey(cache, :etd_phi) || cache_dt === nothing || cache_dt != dt ||
       cache_L !== L_linear
        cache[:etd_phi] = phi_functions_matrix(L_linear, dt)
        cache[:etd_phi_dt] = dt
        cache[:etd_phi_L_source] = L_linear
    end
    return cache[:etd_phi]::NTuple{3, Matrix{ComplexF64}}
end

"""Evaluate the zero-linear-operator ETD formulas directly on single-device fields."""
function _try_explicit_etd_field!(state::TimestepperState, solver::InitialValueSolver;
                                  multistep::Bool)
    plan_is_gpu(solver.execution_plan) && _explicit_multistep_field_eligible(solver) || return false
    current = last(state.history)
    dt = state.dt
    t = solver.sim_time
    F = _workspace_field_state!(state, :etd_field_F, current, 1)
    rhs = evaluate_rhs(solver, current, t)
    _copy_field_state!(F, rhs; preserve_layout=true)
    _release_rhs_buffer!(rhs, solver)
    candidate = _acquire_recycled_history_state!(state, :etd_field_recycled, current;
                                                  preserve_layout=true)
    previous = get(state.timestepper_data, :etd_field_previous_F, nothing)
    if multistep && previous !== nothing
        # φ₁(0) = 1 and φ₂(0) = 1/2 give variable-step Adams–Bashforth.
        w = dt / get_previous_timestep(state)
        _rk_combine_fields!(candidate, current, (F, previous), (dt * (1 + w/2), -dt * w/2))
    else
        # ETDRK2, including the first step of both multistep methods, is Heun's method.
        predictor = _workspace_field_state!(state, :etd_field_predictor, current, 2)
        _rk_combine_fields!(predictor, current, (F,), (dt,))
        state.current_substep = 2
        predicted_rhs = evaluate_rhs(solver, predictor, t + dt)
        _rk_combine_fields!(candidate, current, (F, predicted_rhs), (dt/2, dt/2))
        _release_rhs_buffer!(predicted_rhs, solver)
    end
    _refresh_algebraic_state!(solver.problem, candidate)
    if multistep
        # Commit the accepted derivative by swapping buffers. The previous
        # derivative stays available for a retry until the update succeeds.
        previous = _workspace_field_state!(state, :etd_field_previous_F, current, 3)
        state.timestepper_data[:etd_field_F] = previous
        state.timestepper_data[:etd_field_previous_F] = F
    end
    _push_recycled_history_state!(state, :etd_field_recycled, candidate)
    return true
end

step_etd_rk222!(state, solver) = _step_etd!(state, solver, false, step_cnab2!)
step_etd_cnab2!(state, solver) = _step_etd!(state, solver, true, step_cnab2!)
step_etd_sbdf2!(state, solver) = _step_etd!(state, solver, true, step_sbdf2!)

"""Advance ETD-RK2 or exponential Adams–Bashforth with the appropriate operator representation."""
function _step_etd!(state::TimestepperState, solver::InitialValueSolver, multistep::Bool, fallback)
    if _distributed_diagonal_imex_applicable(solver)
        return _step_distributed_diagonal_etd!(state, solver, multistep)
    end
    _try_explicit_etd_field!(state, solver; multistep) && return

    L = _get_problem_matrix(solver.problem, "L_matrix")
    M = _get_problem_matrix(solver.problem, "M_matrix")
    startup = length(state.history) == 1
    if L === nothing
        if !multistep || startup
            @warn "ETD linear matrix unavailable; using RK222" maxlog=1
            step_rk_imex!(state, solver; ts=_RK222_SINGLETON)
        else
            @warn "ETD linear matrix unavailable; using $(nameof(fallback))" maxlog=1
            fallback(state, solver)
        end
        return
    end
    _etd_oversized_fallback!(state, solver, L, M, startup ? step_cnab2! : fallback) && return

    L_rhs, mass_factor = _get_linear_operator_eff!(state, L, M)
    dt = state.dt
    t = solver.sim_time
    E, phi1, phi2 = _get_etd_phi!(state, L_rhs, dt)
    current = last(state.history)
    X = _timestep_fields_vector!(state, :etd_X, current)
    n = length(X)
    N = _timestep_vector_buffer!(state, :etd_N, n)
    candidate = _timestep_vector_buffer!(state, :etd_candidate, n)
    work = _timestep_vector_buffer!(state, :etd_work, n)
    difference = _timestep_vector_buffer!(state, :etd_difference, n)

    rhs = _timestep_global_rhs_vector!(state, :etd_rhs, solver, current, t)
    _apply_mass_inverse!(N, mass_factor, rhs)
    mul!(candidate, E, X)
    mul!(work, phi1, N)
    axpy!(dt, work, candidate)

    previous = get(state.timestepper_data, :etd_previous_N, nothing)::Union{Nothing,Vector{ComplexF64}}
    if multistep && previous !== nothing
        # Integrate the linear extrapolation of N over this step.
        @. difference = N - previous
        weight = dt * (dt / get_previous_timestep(state))
    else
        # Cox–Matthews ETD-RK2 also supplies the first step for ETD-AB2.
        predictor = _timestep_field_state!(state, :etd_predictor, current)
        vector_to_fields!(predictor, candidate, current)
        state.current_substep = 2
        rhs = _timestep_global_rhs_vector!(state, :etd_rhs, solver, predictor, t + dt)
        _apply_mass_inverse!(difference, mass_factor, rhs)
        difference .-= N
        weight = dt
    end
    mul!(work, phi2, difference)
    axpy!(weight, work, candidate)

    _push_vector_state!(state.history, candidate, current, get_max_timestep_history(state.timestepper))
    if multistep
        # Only an accepted step may replace the derivative used by the next step.
        state.timestepper_data[:etd_N] = _timestep_vector_buffer!(state, :etd_previous_N, n)
        state.timestepper_data[:etd_previous_N] = N
    end
    return nothing
end
