# Shared vector, linear solve, and history-buffer helpers for timestepper paths.

@inline function _timestep_ldiv!(dest::AbstractVector, lhs::F,
                                rhs::AbstractVector) where {F}
    ldiv!(dest, lhs, rhs)
    return dest
end

function _timestep_fields_vector!(state::TimestepperState, key::Symbol,
                                  fields::Vector{<:ScalarField})
    _ensure_coeff_layout!(fields)
    vector = _timestep_vector_buffer!(state, key, _fields_vector_size(fields))
    return fields_to_vector!(vector, fields)
end

struct GlobalRHSLayout
    rows::Vector{Tuple{Int,Int,Vector{Int}}}
    counts::Vector{Int}
    reusable::Bool
    size::Int
end

"""Equation-space layout for global M/L solves. State and equation order are
independent. A state-indexed RHS can be reused only if every evolution equation
has its own target block; coupled time derivatives require evaluating each
equation separately, since the state RHS plan overwrites shared targets."""
function _global_rhs_layout!(state::TimestepperState, solver::InitialValueSolver)
    get!(state.timestepper_data, :global_rhs_layout) do
        fields = solver.state
        rows = Tuple{Int,Int,Vector{Int}}[]
        counts = zeros(Int, length(fields))
        offset = 1
        reusable = true
        for eq in solver.problem.equation_data
            n = compute_field_size(eq)
            targets = _find_time_derivative_targets(eq.mass, fields, solver.problem.variables)
            push!(rows, (offset, n, targets))
            if !isempty(targets)
                reusable &= sum(i -> compute_field_vector_size(fields[i]), targets) == n
                for i in targets
                    counts[i] += 1
                end
            end
            offset += n
        end
        reusable &= all(c -> c <= 1, counts)
        GlobalRHSLayout(rows, counts, reusable, offset - 1)
    end::GlobalRHSLayout
end

"""Evaluate the explicit RHS in the row order of the global matrices.
Algebraic rows are zero here: implicit steppers impose those constraints
directly, rather than accumulating their values with explicit RK weights."""
function _evaluate_global_rhs!(dest::AbstractVector{ComplexF64},
                               state::TimestepperState, solver::InitialValueSolver,
                               fields::Vector{<:ScalarField}, t::Float64)
    layout = _global_rhs_layout!(state, solver)
    length(dest) == layout.size || throw(DimensionMismatch("Global RHS row count changed"))
    fill!(dest, 0)
    if layout.reusable
        rhs = evaluate_rhs(solver, fields, t)
        _ensure_coeff_layout!(rhs)
        for (offset, _, targets) in layout.rows
            for i in targets
                offset = _copy_field_data_to_vector!(dest, offset, rhs[i],
                                                     compute_field_vector_size(fields[i]))
            end
        end
        return dest
    end

    # A forcing is registered against a variable, not an equation. Its row is
    # unambiguous only when that variable belongs to one independent mass block.
    for (_, n, targets) in layout.rows
        for i in targets
            haskey(solver.problem.stochastic_forcings, i) || continue
            if layout.counts[i] != 1 ||
               sum(j -> compute_field_vector_size(fields[j]), targets) != n
                throw(ArgumentError("Registered forcing for '$(fields[i].name)' has an " *
                    "ambiguous equation in a coupled mass system. Express that forcing " *
                    "explicitly on each equation's right-hand side."))
            end
        end
    end
    _update_registered_forcings!(solver, t, solver.dt, DeterministicForcingType)
    _refresh_algebraic_state!(solver.problem, fields)
    sync_state_to_problem!(solver.problem, fields)
    try
        for (eq, (offset, n, targets)) in zip(solver.problem.equation_data, layout.rows)
            isempty(targets) && continue
            expr = eq.forcing_expr === nothing ? eq.forcing : eq.forcing_expr
            template = fields[first(targets)]
            value = evaluate_solver_expression(expr, solver.problem.variables; layout=:g, template)
            components = scalar_components(value)
            _ensure_coeff_layout!(components)
            written = sum(compute_field_vector_size, components)
            # A scalar constant/zero in a vector equation denotes that value in
            # every component. The scalar template above materializes it once.
            if 0 < written < n && n % written == 0 &&
               (is_zero_expression(expr) || _is_const_or_param(expr))
                components = repeat(components, n ÷ written)
                written = n
            end
            written == n || throw(DimensionMismatch(
                "Equation RHS has $written coefficients, but its matrix block has $n rows"))
            row = offset
            for field in components
                row = _copy_field_data_to_vector!(dest, row, field, compute_field_vector_size(field))
            end
            # Independent equations still retain their usual registered forcing
            # when another equation caused this per-equation evaluation path.
            row = offset
            for i in targets
                forcing = get(solver.problem.stochastic_forcings, i, nothing)
                if forcing !== nothing
                    cd = coeff_data!(fields[i])
                    values = _matched_forcing_view(forcing, cd)
                    values === nothing && throw(ArgumentError("Forcing size doesn't match RHS"))
                    view(dest, row:row + length(values) - 1) .+= vec(get_local_data(values))
                end
                row += compute_field_vector_size(fields[i])
            end
        end
    finally
        _alias_state_to_problem!(solver.problem, solver.state)
    end
    return dest
end

function _timestep_global_rhs_vector!(state::TimestepperState, key::Symbol,
                                      solver::InitialValueSolver,
                                      fields::Vector{<:ScalarField}, t::Float64)
    layout = _global_rhs_layout!(state, solver)
    dest = _timestep_vector_buffer!(state, key, layout.size)
    return _evaluate_global_rhs!(dest, state, solver, fields, t)
end

function _timestep_matvec!(state::TimestepperState, key::Symbol,
                           matrix::AbstractMatrix, vector::AbstractVector{ComplexF64})
    dest = _timestep_vector_buffer!(state, key, size(matrix, 1))
    mul!(dest, matrix, vector)
    return dest
end

function _prepend_history_buffer!(history::Vector{Vector{ComplexF64}},
                                  scratch::Vector{ComplexF64}, max_len::Int)
    max_len <= 0 && return history

    if length(history) >= max_len
        slot = pop!(history)
        if length(slot) != length(scratch)
            slot = Vector{ComplexF64}(undef, length(scratch))
        end
    else
        slot = Vector{ComplexF64}(undef, length(scratch))
    end

    copyto!(slot, scratch)
    pushfirst!(history, slot)
    return history
end
