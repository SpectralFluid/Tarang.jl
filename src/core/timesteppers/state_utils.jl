# ============================================================================
# State Manipulation Utilities (GPU-aware)
# ============================================================================

"""
    evaluate_rhs(solver::InitialValueSolver, state::Vector{<:ScalarField}, time::Float64)

Evaluate right-hand side of differential equations following Tarang pattern.

IMPORTANT: This function returns RHS fields matching the state vector structure,
not the equation structure. Each state field gets a corresponding RHS field.
Equations with time derivatives contribute their F_expr to the appropriate state field;
constraint equations (no dt) contribute zero.
"""
function evaluate_rhs(solver::InitialValueSolver, state::Vector{<:ScalarField}, time::Float64)
    _update_registered_forcings!(solver, time, solver.dt, DeterministicForcingType)
    strategy = _rhs_evaluation_strategy(solver)
    if strategy !== :interpreted
        # Not authoritative: this also runs on the implicit CPU paths, whose stage
        # solve owns the constraints the refresh cannot solve exactly.
        _refresh_algebraic_state!(solver.problem, state; authoritative=false)
    end

    rhs_fields = _evaluate_rhs_with_strategy(strategy, solver, state)
    rhs_fields !== nothing && return rhs_fields
    return _evaluate_rhs_interpreted(solver, state, time)
end

function _evaluate_rhs_interpreted(solver::InitialValueSolver,
                                   state::Vector{<:ScalarField},
                                   time::Float64)
    problem = solver.problem
    # Fallback: interpreted expression evaluation (original path)
    # Initialize RHS with zero fields for ALL state fields
    rhs = ScalarField[]
    for field in state
        push!(rhs, create_rhs_zero_field(field))
    end

    try
        # Sync current state into problem variables for expression evaluation
        sync_state_to_problem!(problem, state)

        # `equation_data` is filled by `build_matrix_expressions!`, which runs as part
        # of global-matrix assembly — the step a pure-Fourier GPU InitialValueProblem SKIPS. Treating an
        # empty IR as "no equations contribute" would leave `rhs` as the zero fields
        # allocated above and freeze the solution at its initial condition with nothing
        # raised. That is the bug fixed in `build_lazy_rhs_plan!`; this is the same hole
        # on the interpreted path, closed the same way. The build is matrix-free, and
        # the enclosing catch rethrows, so a parse failure surfaces instead of becoming
        # a zero RHS. It must precede the constraint refresh below, which reads the IR.
        if isempty(problem.equation_data) && !isempty(problem.equations)
            build_matrix_expressions!(problem)
        end

        # Refresh the algebraic variables F reads from the state being evaluated
        # (exact solves only; see `_refresh_algebraic_state!`).
        _solve_algebraic_constraints!(problem, state)

        # Update time parameter if it exists
        if hasfield(typeof(problem), :time) && problem.time !== nothing
            if problem.time isa ScalarField
                if get_grid_data(problem.time) !== nothing
                    fill!(grid_data!(problem.time), time)
                elseif get_coeff_data(problem.time) !== nothing
                    fill!(coeff_data!(problem.time), time)
                elseif hasfield(typeof(problem), :parameters)
                    problem.parameters["t"] = time
                end
            elseif hasfield(typeof(problem.time), :data)
                problem.time.data = time
            elseif hasfield(typeof(problem.time), :value)
                problem.time.value = time
            end
        end

        # Build mapping from state field index to equation F_expr
        # Only equations with time derivatives contribute to F
        if !isempty(problem.equation_data)
            for (eq_idx, eq_data) in enumerate(problem.equation_data)
                # Check if this equation has a time derivative (M term)
                M_expr = get(eq_data, "M", nothing)
                if M_expr === nothing || _is_zero_m_term(M_expr)
                    # Constraint equation - no contribution to explicit F
                    continue
                end

                # Find which state field(s) this equation's dt term targets
                target_indices = _find_time_derivative_targets(M_expr, state, problem.variables)
                if isempty(target_indices)
                    @debug "Equation $eq_idx has M term but no matching state field target"
                    continue
                end

                # Get F expression
                expr = if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
                    eq_data["F_expr"]
                else
                    get(eq_data, "F", ZeroOperator())
                end

                # Evaluate and assign to target state field(s)
                for state_idx in target_indices
                    if state_idx <= length(rhs)
                        template = state[state_idx]
                        try
                            rhs_field = evaluate_solver_expression(expr, problem.variables; layout=:g, template=template)
                            if isa(rhs_field, ScalarField)
                                ensure_layout!(rhs_field, :c)
                                rhs[state_idx] = rhs_field
                            elseif isa(rhs_field, VectorField)
                                # Vector RHS (e.g. -u⋅∇(u)) — extract the component
                                # matching this state field's position within the vector
                                comp_idx = state_idx - first(target_indices) + 1
                                if !(1 <= comp_idx <= length(rhs_field.components))
                                    error("RHS expression for state field $state_idx is a " *
                                          "$(length(rhs_field.components))-component VectorField, " *
                                          "but this field maps to component $comp_idx of it. " *
                                          "The equation's ∂t targets and its RHS rank disagree.")
                                end
                                comp = rhs_field.components[comp_idx]
                                ensure_layout!(comp, :c)
                                rhs[state_idx] = comp
                            else
                                error("RHS expression for state field $state_idx evaluated to " *
                                      "$(typeof(rhs_field)); expected a ScalarField or VectorField.")
                            end
                        catch e
                            # NEVER degrade a failed RHS evaluation to "RHS = 0". Zero is a
                            # physically meaningful right-hand side, so leaving it here
                            # silently replaces the equation with a different one — the run
                            # completes, reports success, and returns a wrong answer, and it
                            # takes down every other term in the same sum with it (the whole
                            # field's RHS stays at the zero it was initialized to). This
                            # extends the policy that used to apply only to
                            # UnrecognizedRHSExpression (typo'd operators) to every
                            # evaluation failure: an unsupported or malformed term must be
                            # reported, not quietly dropped. The catch survives only to
                            # name the offending state field before re-raising.
                            @error "Failed to evaluate the RHS expression for state field " *
                                   "$state_idx (expression type $(typeof(expr))). Aborting " *
                                   "rather than continuing with a zero RHS for this field."
                            rethrow()
                        end
                    end
                end
            end
        end

        @debug "Evaluated RHS for $(length(rhs)) state fields at time $time"

        # Add stochastic forcing if registered
        if hasfield(typeof(problem), :stochastic_forcings) && !isempty(problem.stochastic_forcings)
            for (var_idx, forcing) in problem.stochastic_forcings
                if var_idx <= length(rhs)
                    rhs_field = rhs[var_idx]
                    coeff_data = get_coeff_data(rhs_field)
                    if coeff_data !== nothing
                        F_view = _matched_forcing_view(forcing, coeff_data)
                        ensure_layout!(rhs_field, :c)
                        if F_view !== nothing
                            coeff_data .+= F_view
                            @debug "Added stochastic forcing to state field $var_idx"
                        else
                            throw(ArgumentError(
                                "Forcing size doesn't match RHS size $(size(coeff_data)) for state field $var_idx"))
                        end
                    end
                end
            end
        end

    catch e
        @error "RHS evaluation failed: $e"
        rethrow()
    finally
        # `sync_state_to_problem!` REBINDS `problem.variables` (it shares storage
        # now, it does not copy), and this path is handed the transient RK stage
        # field-set. Leaving the user's handles pointing at a recycled stage
        # buffer would make any diagnostic, output task or callback that reads a
        # problem variable mid-step — or after a throw — see stage data instead
        # of the solution. Put them back on the live state either way.
        _alias_state_to_problem!(problem, solver.state)
    end

    return rhs
end

"""
    _is_zero_m_term(M_expr)

Check if M term is effectively zero (no time derivative contribution).
"""
function _is_zero_m_term(M_expr)
    if M_expr === nothing
        return true
    elseif isa(M_expr, ZeroOperator)
        return true
    elseif isa(M_expr, Number) && M_expr == 0
        return true
    elseif isa(M_expr, Vector) && isempty(M_expr)
        return true
    end
    return false
end

"""
    _find_time_derivative_targets(M_expr, state, variables)

Find which state field indices are targeted by the time derivative in M_expr.
Returns a vector of state indices.
"""
function _find_time_derivative_targets(M_expr, state::Vector{<:ScalarField}, variables::Vector)
    targets = Int[]
    _collect_dt_targets!(targets, M_expr, state, variables)
    return unique(targets)
end

function _collect_dt_targets!(targets::Vector{Int}, expr, state::Vector{<:ScalarField}, variables::Vector)
    if expr === nothing
        return
    end

    if isa(expr, TimeDerivative)
        # Found a time derivative — extract the operand variable.
        # For VectorField operands (e.g. ∂t(u)), expand to ALL component indices:
        # each component has its own row block in the equation-space layout and
        # its own F contribution. Returning only the first index would leave
        # later components with zero F, matching the earlier silent-BC bug but
        # for vector PDE equations.
        operand = expr.operand
        if isa(operand, VectorField)
            first_idx = _find_state_index_for_operand(operand, state, variables)
            if first_idx !== nothing
                for c in 0:(length(operand.components) - 1)
                    push!(targets, first_idx + c)
                end
            end
        else
            state_idx = _find_state_index_for_operand(operand, state, variables)
            if state_idx !== nothing
                push!(targets, state_idx)
            end
        end
        return
    end

    # Recurse into composite expressions
    if isa(expr, Vector)
        for item in expr
            _collect_dt_targets!(targets, item, state, variables)
        end
    elseif hasfield(typeof(expr), :operand)
        _collect_dt_targets!(targets, getfield(expr, :operand), state, variables)
    elseif hasfield(typeof(expr), :left)
        _collect_dt_targets!(targets, getfield(expr, :left), state, variables)
    end
    if hasfield(typeof(expr), :right)
        _collect_dt_targets!(targets, getfield(expr, :right), state, variables)
    end
end

"""
    evaluate_rhs_buffered(solver::InitialValueSolver, state::Vector{<:ScalarField}, time::Float64)

Evaluate the RHS using a reusable result buffer when the compiled lazy plan is
available. Callers must consume the returned fields immediately and must not
store the vector or its fields across later RHS evaluations.
"""
function evaluate_rhs_buffered(
    solver::InitialValueSolver,
    state::Vector{<:ScalarField},
    time::Float64,
)
    _update_registered_forcings!(solver, time, solver.dt, DeterministicForcingType)
    strategy = _rhs_evaluation_strategy(solver; buffered=true)
    if strategy !== :interpreted
        _refresh_algebraic_state!(solver.problem, state; authoritative=false)
    end

    rhs_fields = _evaluate_rhs_with_strategy(strategy, solver, state)
    rhs_fields !== nothing && return rhs_fields
    return _evaluate_rhs_interpreted(solver, state, time)
end

"""
    _refresh_algebraic_state!(problem, state; authoritative=true) -> Bool

Re-solve the algebraic variables of `state` from its evolved fields, exactly (see
"Algebraic-constraint refresh" below). With `authoritative=true` — the default,
for the steppers that have no implicit stage solve and call this after each step —
an algebraic equation the refresh cannot solve exactly is an error. The RHS
evaluation passes `authoritative=false`: it also runs on the CPU global-matrix and
subproblem paths, whose stage solve owns such equations, so they are left
untouched there. Either way nothing is written that is not an exact solution.

The problem variables are left aliased to `state`, which the refresh writes in place.
"""
function _refresh_algebraic_state!(problem::Problem, state::Vector{<:ScalarField};
                                   authoritative::Bool=true)
    _has_algebraic_constraints(problem) || return false
    plan = _constraint_refresh_plan!(problem, state)
    authoritative && _require_solved_constraints(plan)
    isempty(plan.solves) && return false

    sync_state_to_problem!(problem, state)
    _apply_constraint_plan!(plan, problem, state)
    return true
end

function _has_algebraic_constraints(problem::Problem)
    isempty(problem.equation_data) && return false

    for eq_data in problem.equation_data
        M_expr = get(eq_data, "M", nothing)
        if M_expr === nothing || _is_zero_m_term(M_expr)
            return true
        end
    end
    return false
end

"""
    _find_state_index_for_operand(operand, state, variables)

Find the state field index corresponding to an operand (variable or field).
Returns nothing if not found.
"""
function _find_state_index_for_operand(operand, state::Vector{<:ScalarField}, variables::Vector)
    # Direct ScalarField match
    if isa(operand, ScalarField)
        for (i, s) in enumerate(state)
            if s === operand || s.name == operand.name
                return i
            end
        end
    end

    # VectorField component - need to find which component and map to state
    if isa(operand, VectorField)
        # VectorField components are expanded in state, return first component index
        state_idx = 1
        for var in variables
            if var === operand || (isa(var, VectorField) && var.name == operand.name)
                return state_idx
            end
            if isa(var, ScalarField)
                state_idx += 1
            elseif isa(var, VectorField)
                state_idx += length(var.components)
            elseif isa(var, TensorField)
                state_idx += length(vec(var.components))
            end
        end
    end

    # String name lookup
    if isa(operand, AbstractString)
        for (i, s) in enumerate(state)
            if s.name == operand
                return i
            end
        end
    end

    return nothing
end

"""
    _zero_array!(a)

Fill an array with zeros, handling PencilArray wrappers.
"""
@inline function _zero_array!(a::AbstractArray)
    if isa(a, PencilArrays.PencilArray)
        fill!(parent(a), zero(eltype(a)))
    else
        fill!(a, zero(eltype(a)))
    end
end

"""
    _zero_like(a)

Create a zero-filled array with the same type, size, and structure as `a`.
Uses `similar` + `fill!` instead of `copy` + `fill!` to avoid copying data
that will be immediately overwritten with zeros.
"""
@inline function _zero_like(a::AbstractArray)
    z = similar(a)
    _zero_array!(z)
    return z
end

"""Create a zero RHS field matching the template field properties.

    Uses the global FieldPool (via checkout_or_alloc) when available to avoid
    per-RHS-evaluation allocations.  Falls back to direct ScalarField allocation
    when no pool is active.

    IMPORTANT: In MPI mode the pool field is constructed from the same dist, so
    it already carries the correct PencilArray decomposition structure — no
    need for similar()-based copying.
    """
function create_rhs_zero_field(template_field::ScalarField)

    # Skip 0D fields (tau variables) which have no spatial data
    if isempty(template_field.bases)
        return ScalarField(template_field.dist, "rhs_$(template_field.name)", template_field.bases, template_field.dtype)
    end

    # Check out (or allocate) a field from the pool
    field = checkout_or_alloc(template_field.bases, template_field.dtype, template_field.dist)
    field.current_layout = template_field.current_layout

    # Zero the data arrays so stale pool data does not leak into the RHS
    gd = get_grid_data(field)
    if gd !== nothing
        fill!(gd, zero(eltype(gd)))
    end
    cd = get_coeff_data(field)
    if cd !== nothing
        fill!(cd, zero(eltype(cd)))
    end

    # Copy scale information from template
    if template_field.scales !== nothing
        field.scales = template_field.scales
    end

    return field
end

# ── Layout policy for state arithmetic ──────────────────────────────────────
#
# Every helper below computes a LINEAR COMBINATION of states, and a linear
# combination commutes with the spectral transform: it is equally valid in grid
# space or coefficient space, provided every operand is in the SAME one. These
# helpers used to force `:g` unconditionally, so handing them coefficient-space
# fields paid a backward transform per operand on entry — and, because the RHS
# evaluators want `:c`, a forward transform right back on the next call.
#
# Measured directly (transform COUNTERS, not a stopwatch —
# `Tarang.enable_transform_counts!`): `axpy_state!` on two `:c` fields used to
# report `(forward = 0, backward = 2)` and leave both operands `:g`. It now
# reports zero and leaves them `:c`.
#
# Scope, stated honestly: in the three stepper paths that were measured end to
# end (explicit RK443, SBDF2, DiagonalIMEX_SBDF2 on the distributed field path)
# the per-step transform count is UNCHANGED, because those callers happen to
# reach these helpers with their fields already in `:g`. The transforms those
# steppers do pay come from the lazy RHS's own coefficient round trip, not from
# here. So this removes a latent cost and makes the helpers layout-honest; it is
# not, on today's call sites, a measured end-to-end speedup.
#
# `_arith_layout` returns `:c` only when EVERY operand is already there with a
# usable, mutually compatible coefficient buffer, and `:g` otherwise. So it can
# only ever remove a transform the grid-forcing version already paid; it never
# introduces one. When the buffers do not line up (mixed layouts, a missing
# coeff buffer, a shape/eltype mismatch) it falls back to exactly the old
# behaviour.

# The arithmetic kernels below are split by ELEMENT TYPE, and the call sites
# branch on `layout` so that each branch calls `get_grid_data` or
# `get_coeff_data` DIRECTLY. Both halves matter:
#
#   * A `layout === :c ? get_coeff_data(f) : get_grid_data(f)` helper returns a
#     Union of the grid and coefficient buffer types, which makes every
#     downstream broadcast dynamically dispatched. That is not free and it is not
#     size-independent: `test_multistep_field_path.jl`'s "steady-state cost is
#     O(1) in the grid size" caught it immediately (464 vs 768 bytes/step,
#     because the n>100 `@turbo` branch and the small-n broadcast branch box
#     differently).
#   * `@turbo` (LoopVectorization) does not vectorize `Complex`, and must never
#     see device memory. Coefficient buffers are complex, so they take the plain
#     broadcast; the real host path keeps the original BLAS/`@turbo`/broadcast
#     ladder unchanged.

"""`y .+= a*x` — real host arrays keep the BLAS/`@turbo`/broadcast ladder."""
@inline function _sa_axpy!(y_data::AbstractArray{T}, a::Float64,
                           x_data::AbstractArray{T}) where {T<:AbstractFloat}
    if is_gpu_array(x_data)
        y_data .+= a .* x_data
        return y_data
    end
    n = length(x_data)
    if n > 2000 && x_data isa StridedArray && y_data isa StridedArray
        BLAS.axpy!(a, x_data, y_data)
    elseif n > 100
        a_local = a
        @turbo for j in eachindex(y_data, x_data)
            y_data[j] += a_local * x_data[j]
        end
    else
        y_data .+= a .* x_data
    end
    return y_data
end

"""`y .+= a*x` — complex (coefficient) and device arrays: plain broadcast."""
@inline function _sa_axpy!(y_data, a::Float64, x_data)
    y_data .+= a .* x_data
    return y_data
end

"""`dest .= x .+ a*y` — real host arrays keep the original ladder."""
@inline function _sa_add_scaled!(dest::AbstractArray{T}, x_data::AbstractArray{T},
                                 y_data::AbstractArray{T}, a::Float64) where {T<:AbstractFloat}
    if is_gpu_array(x_data)
        dest .= x_data .+ a .* y_data
        return dest
    end
    n = length(x_data)
    if n > 2000 && x_data isa StridedArray && y_data isa StridedArray && dest isa StridedArray
        copyto!(dest, x_data)
        BLAS.axpy!(a, y_data, dest)
    elseif n > 100
        a_local = a
        @turbo for j in eachindex(dest, x_data, y_data)
            dest[j] = x_data[j] + a_local * y_data[j]
        end
    else
        dest .= x_data .+ a .* y_data
    end
    return dest
end

"""`dest .= x .+ a*y` — complex (coefficient) and device arrays."""
@inline function _sa_add_scaled!(dest, x_data, y_data, a::Float64)
    dest .= x_data .+ a .* y_data
    return dest
end

"""`dest .= α*a .+ β*b` — real host arrays keep the original ladder."""
@inline function _sa_lincomb!(dest::AbstractArray{T}, α::Float64, a_data::AbstractArray{T},
                              β::Float64, b_data::AbstractArray{T}) where {T<:AbstractFloat}
    if is_gpu_array(a_data)
        dest .= α .* a_data .+ β .* b_data
        return dest
    end
    if length(a_data) > 100
        α_local, β_local = α, β
        @turbo for j in eachindex(dest, a_data, b_data)
            dest[j] = α_local * a_data[j] + β_local * b_data[j]
        end
    else
        dest .= α .* a_data .+ β .* b_data
    end
    return dest
end

"""`dest .= α*a .+ β*b` — complex (coefficient) and device arrays."""
@inline function _sa_lincomb!(dest, α::Float64, a_data, β::Float64, b_data)
    dest .= α .* a_data .+ β .* b_data
    return dest
end

"""
    _arith_layout(fields::Tuple) -> Symbol

`:c` when every field in `fields` is already in coefficient layout with a
coefficient buffer, and all those buffers agree in size and element type; `:g`
otherwise. See the comment block above for why either answer is correct and why
this can only save work.
"""
function _arith_layout(fields::Tuple)
    ref = nothing
    for f in fields
        f.current_layout === :c || return :g
        cd = get_coeff_data(f)
        cd === nothing && return :g
        if ref === nothing
            ref = cd
        elseif size(cd) != size(ref) || eltype(cd) != eltype(ref)
            return :g
        end
    end
    return ref === nothing ? :g : :c
end

"""
    add_scaled_state(state1::Vector{<:ScalarField}, state2::Vector{<:ScalarField}, scale::Float64)

Compute state1 + scale * state2 (GPU-aware)
"""
function add_scaled_state(state1::Vector{<:ScalarField}, state2::Vector{<:ScalarField}, scale::Float64)
    n = length(state1)
    result = Vector{ScalarField}(undef, n)

    @inbounds for i in 1:n
        field1 = state1[i]
        field2 = state2[i]

        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(field1.bases)
            result[i] = ScalarField(field1.dist, field1.name, field1.bases, field1.dtype)
            continue
        end

        # Use copy() to preserve PencilArray structure in MPI mode
        new_field = copy(field1)

        layout = _arith_layout((field1, field2, new_field))
        ensure_layout!(field1, layout)
        ensure_layout!(field2, layout)
        ensure_layout!(new_field, layout)

        # Branch on layout so each side calls the accessor DIRECTLY and keeps
        # concrete buffer types (see the note above `_sa_axpy!`).
        if layout === :c
            d1, d2, dn = get_coeff_data(field1), get_coeff_data(field2), get_coeff_data(new_field)
            d1 !== nothing && d2 !== nothing && dn !== nothing &&
                _sa_add_scaled!(dn, d1, d2, scale)
        else
            d1, d2, dn = get_grid_data(field1), get_grid_data(field2), get_grid_data(new_field)
            d1 !== nothing && d2 !== nothing && dn !== nothing &&
                _sa_add_scaled!(dn, d1, d2, scale)
        end
        result[i] = new_field
    end

    return result
end

"""
    add_scaled_state!(dest::Vector{<:ScalarField}, state1::Vector{<:ScalarField},
                      state2::Vector{<:ScalarField}, scale::Float64)

In-place version: dest = state1 + scale * state2 (GPU-aware)
"""
function add_scaled_state!(dest::Vector{<:ScalarField}, state1::Vector{<:ScalarField},
                           state2::Vector{<:ScalarField}, scale::Float64)
    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        dest_field = dest[i]

        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(field1.bases)
            continue
        end

        layout = _arith_layout((field1, field2, dest_field))
        ensure_layout!(field1, layout)
        ensure_layout!(field2, layout)
        ensure_layout!(dest_field, layout)

        if layout === :c
            d1, d2, dd = get_coeff_data(field1), get_coeff_data(field2), get_coeff_data(dest_field)
            (d1 === nothing || d2 === nothing || dd === nothing) && continue
            _sa_add_scaled!(dd, d1, d2, scale)
        else
            d1, d2, dd = get_grid_data(field1), get_grid_data(field2), get_grid_data(dest_field)
            (d1 === nothing || d2 === nothing || dd === nothing) && continue
            _sa_add_scaled!(dd, d1, d2, scale)
        end
    end
end

"""
    axpy_state!(scale::Float64, x::Vector{<:ScalarField}, y::Vector{<:ScalarField})

In-place AXPY: y = y + scale * x (GPU-aware)
"""
function axpy_state!(scale::Float64, x::Vector{<:ScalarField}, y::Vector{<:ScalarField})
    for i in eachindex(x, y)
        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(x[i].bases)
            continue
        end

        layout = _arith_layout((x[i], y[i]))
        ensure_layout!(x[i], layout)
        ensure_layout!(y[i], layout)

        if layout === :c
            xd, yd = get_coeff_data(x[i]), get_coeff_data(y[i])
            (xd === nothing || yd === nothing) && continue
            _sa_axpy!(yd, scale, xd)
        else
            xd, yd = get_grid_data(x[i]), get_grid_data(y[i])
            (xd === nothing || yd === nothing) && continue
            _sa_axpy!(yd, scale, xd)
        end
    end
end

"""
    linear_combination_state!(dest::Vector{<:ScalarField}, α::Float64, a::Vector{<:ScalarField},
                              β::Float64, b::Vector{<:ScalarField})

In-place linear combination: dest = α*a + β*b (GPU-aware)
"""
function linear_combination_state!(dest::Vector{<:ScalarField}, α::Float64, a::Vector{<:ScalarField},
                                   β::Float64, b::Vector{<:ScalarField})
    for i in eachindex(dest, a, b)
        # Skip 0D fields (tau variables) which have no spatial data
        if isempty(a[i].bases)
            continue
        end

        layout = _arith_layout((a[i], b[i], dest[i]))
        ensure_layout!(a[i], layout)
        ensure_layout!(b[i], layout)
        ensure_layout!(dest[i], layout)

        if layout === :c
            ad, bd, dd = get_coeff_data(a[i]), get_coeff_data(b[i]), get_coeff_data(dest[i])
            (ad === nothing || bd === nothing || dd === nothing) && continue
            _sa_lincomb!(dd, α, ad, β, bd)
        else
            ad, bd, dd = get_grid_data(a[i]), get_grid_data(b[i]), get_grid_data(dest[i])
            (ad === nothing || bd === nothing || dd === nothing) && continue
            _sa_lincomb!(dd, α, ad, β, bd)
        end
    end
end

"""
    copy_state(state::Vector{<:ScalarField})

Create a deep copy of state
"""
function copy_state(state::Vector{F}) where {F<:ScalarField}
    if F <: ScalarField{<:Any, <:SerialFieldStorage}
        return _copy_serial_state(state)
    end

    n = length(state)
    new_state = Vector{ScalarField}(undef, n)

    @inbounds for i in 1:n
        field = state[i]
        if isempty(field.bases)
            new_state[i] = ScalarField(field.dist, field.name, field.bases, field.dtype)
        else
            new_state[i] = copy(field)
        end
    end

    return new_state
end

function _copy_serial_state(state::Vector{F}) where {F<:ScalarField{<:Any, <:SerialFieldStorage}}
    n = length(state)
    new_state = Vector{F}(undef, n)

    @inbounds for i in 1:n
        field = state[i]
        if isempty(field.bases)
            new_state[i] = ScalarField(field.dist, field.name, field.bases, field.dtype)
        else
            new_state[i] = copy(field)
        end
    end

    return new_state
end

"""
    _timestep_field_state!(state, key, template)

Return a reusable field-state workspace matching `template`.
"""
function _timestep_field_state!(state::TimestepperState, key::Symbol,
                                template::Vector{F}) where {F<:ScalarField}
    cached = get(state.timestepper_data, key, nothing)
    if cached isa Vector{F} && length(cached) == length(template)
        return cached
    end

    fields = copy_state(template)
    state.timestepper_data[key] = fields
    return fields
end

"""
    _push_vector_state!(history, vector, template, max_len)

Write `vector` into a recycled state slot and append it to `history`.
When the history is already at capacity, the oldest state is reused instead
of allocating a fresh deep copy every step.
"""
function _push_vector_state!(history::Vector{V}, vector::AbstractVector{<:Number},
                             template::V, max_len::Int) where {V<:Vector{<:ScalarField}}
    max_len > 0 || throw(ArgumentError("max_len must be positive"))

    new_state = if length(history) >= max_len
        popfirst!(history)
    else
        copy_state(template)
    end

    vector_to_fields!(new_state, vector, template)
    push!(history, new_state)
    return new_state
end

# Forcing → RHS coefficient view. Every concrete forcing type supplies its own
# `_matched_forcing_view` (diagnostics.jl for the stochastic
# types, deterministic.jl for DeterministicForcing). There is
# deliberately NO untyped fallback: the one that used to live here sliced
# `forcing.cached_forcing` to the target's size whatever the array held, which
# is how a registered DeterministicForcing had its physical-grid values added
# to the coefficient array as though they were spectral coefficients — a wrong
# field with no error. An unknown forcing type is a MethodError, on purpose.
_matched_forcing_view(::Nothing, target) = nothing

# ============================================================================
# Algebraic-constraint refresh
# ============================================================================
#
# An algebraic equation (no ∂t) determines a diagnostic variable — a
# streamfunction, a velocity, an inverted field — from the evolved state. The
# refresh below re-solves those variables from the current state, before every
# RHS evaluation and after every step of a stepper that has no implicit stage
# solve.
#
# Who relies on it:
#   * A pure-Fourier GPU problem, the matrix-free MPI path and the diagonal-IMEX /
#     explicit field steppers have NO other constraint solver; the refresh is
#     authoritative there (`authoritative=true`, the post-step call sites).
#   * The CPU global-matrix and subproblem steppers impose every constraint in
#     the stage solve, so the stages they produce are already consistent. There
#     the RHS-evaluation refresh (`authoritative=false`) only repairs a state that
#     did not come out of a solve — the initial condition, whose diagnostic
#     variables are usually left at zero, or a state edited between steps —
#     before F reads it. Without it the first stage of the first step evaluates F
#     on that inconsistent state: a first-order error that the measured 2D
#     Poisson case shows at ~1e-4 against an exact 6e-12.
#
# Because of that second role the refresh may never write an inexact value: it
# either solves an equation EXACTLY or leaves it untouched, and "untouched" is a
# loud error wherever the refresh is authoritative. The pattern matcher this
# replaces did neither — it used the first source field it found, ignored
# coefficients, recognised only a bare Laplacian, and swallowed failures at
# @debug. On a device problem that dropped θ from `Δφ = ζ + θ` and never updated
# `2Δφ = ζ` or `φ - Δφ/2 = ζ`; on the CPU it then overwrote the stage solve's
# correct φ with that wrong one right before F was evaluated.
#
# Solved exactly, per Fourier mode and without scalar indexing:
#
#     L̂ ⊙ v̂  +  Σⱼ M̂ⱼ ⊙ x̂ⱼ  +  Σ cₜ·τₜ  +  (other terms)  =  F̂
#
#   * `v`, the target, is the one algebraic variable the equation determines. Its
#     operator L̂ must be diagonal in Fourier space: any sum of constant multiples
#     of the identity, Laplacian, fractional/hyper-Laplacian and derivatives of
#     `v` — the operators `_accumulate_diagonal_L!` symbolizes for the
#     diagonal-IMEX steppers. A plain constant multiple of `v` works on any basis.
#   * `xⱼ` are evolved fields or algebraic variables an earlier equation solved.
#     Diagonal operators on them become multipliers M̂ⱼ; anything else (products,
#     `skew(grad(ψ))`, parameter fields) is evaluated with the RHS evaluator.
#   * `τₜ` are 0-D tau variables. They touch only k = 0 and carry no storage, so
#     v̂(k=0) comes from the gauge row instead, as in the coupled implicit solve:
#     `integ(v) = c` pins the mean of `v` to c/|Ω|. With neither tau nor gauge, a
#     mode where L̂ vanishes (k = 0 of a Laplacian) is set to zero — the
#     zero-mean convention of the inverse Laplacian. L̂ vanishing anywhere else
#     means the operator is singular, which is refused.
#   * Equations are solved in dependency order (a velocity after the
#     streamfunction it differentiates); a vector target (`u - skew(grad(ψ)) = 0`)
#     is evaluated from its right-hand side.
#
# Anything else is reported by name. With `authoritative=false` those equations
# are left to the stage solve: "not mine to refresh" is an explicit outcome of
# the plan, not an exception swallowed along the way.

"""One contribution `weight ⊙ x̂` to a constraint's right-hand side, where
`x = state[index]`. `weight` is a real scalar or a complex multiplier array on
the target's local coefficient grid."""
struct _ConstraintInput
    index::Int
    weight::Any
end

"""Exact refresh of one algebraic equation for its target variable."""
struct _ConstraintSolve
    equation::Int
    label::String
    target::Int                           # state index (first component of a vector)
    target_var::Operand                   # the problem variable being solved for
    neg_inv::Any                          # scalar target: -1/L̂ — a scalar, or an array zero on pinned modes
    pin::Any                              # `nothing`, or the gauge-pinned k = 0 coefficients
    inputs::Vector{_ConstraintInput}
    constant::Any                         # `nothing`, or the coefficients of the constant terms
    generic::Vector{Tuple{Float64, Any}}  # (coefficient, term) evaluated at every refresh
    vector_expr::Any                      # vector target: its value, or `nothing` for zero
end

"""An algebraic equation the refresh cannot solve exactly, and why."""
struct _UnsolvedConstraint
    equation::Int
    label::String
    reason::String
end

struct _ConstraintRefreshPlan
    equation_ids::Vector{UInt}
    rhs_ids::Vector{UInt}    # see `_constraint_rhs_id`
    nstate::Int
    solves::Vector{_ConstraintSolve}
    unsolved::Vector{_UnsolvedConstraint}
end

"""A linear term `coefficient * term` of a constraint, with the problem variables
`term` references."""
struct _ConstraintTerm
    coefficient::Float64
    term::Any
    variables::Vector{Operand}
end

"""
    _solve_algebraic_constraints!(problem, state)

Refresh every algebraic variable the constraint plan can solve exactly, leaving
the rest to the caller's implicit solve (`authoritative=false` semantics). The
caller must already have aliased the problem variables to `state`
(`sync_state_to_problem!`).
"""
function _solve_algebraic_constraints!(problem::Problem, state::Vector{<:ScalarField})
    _has_algebraic_constraints(problem) || return nothing
    _apply_constraint_plan!(_constraint_refresh_plan!(problem, state), problem, state)
    return nothing
end

"""Raise, naming each equation, when the refresh cannot solve every algebraic
equation of `problem` exactly. Called at construction for solvers that have no
implicit stage solve, so an unsupported constraint fails before the first step."""
function _require_refreshable_constraints!(problem::Problem, state::Vector{<:ScalarField})
    _has_algebraic_constraints(problem) || return nothing
    _require_solved_constraints(_constraint_refresh_plan!(problem, state))
    return nothing
end

function _require_solved_constraints(plan::_ConstraintRefreshPlan)
    isempty(plan.unsolved) && return nothing
    details = join(("  * `$(u.label)`: $(u.reason)" for u in plan.unsolved), "\n")
    throw(ArgumentError(
        "No implicit stage solve imposes this problem's algebraic equations (pure-Fourier " *
        "GPU, matrix-free MPI, or a diagonal-IMEX / explicit field stepper), so the " *
        "spectral constraint refresh is their only solver, and it cannot solve these " *
        "exactly:\n$details\n" *
        "The refresh solves an equation for ONE algebraic variable whose operator is " *
        "diagonal in Fourier space (sums of constant multiples of the identity, Laplacian, " *
        "fractional/hyper-Laplacian and derivatives of that variable), plus any linear " *
        "combination of other fields, constants, and 0-D tau variables backed by a gauge " *
        "row such as `integ(v) = 0`. Rewrite the constraint in that form, or use a CPU " *
        "solver whose global-matrix or subproblem stage solve imposes it. (Such equations " *
        "used to be skipped silently, leaving the variable stale or wrong.)"))
end

"""The cached refresh plan for `problem`, rebuilt when its parsed equations change."""
function _constraint_refresh_plan!(problem::Problem, state::Vector{<:ScalarField})
    cache = compiled_problem(problem).caches.workspaces
    plan = get(cache, :algebraic_constraint_refresh, nothing)
    if plan isa _ConstraintRefreshPlan && _constraint_plan_matches(plan, problem, state)
        return plan
    end
    plan = _build_constraint_refresh_plan(problem, state)
    cache[:algebraic_constraint_refresh] = plan
    return plan
end

function _constraint_plan_matches(plan::_ConstraintRefreshPlan, problem::Problem,
                                  state::Vector{<:ScalarField})
    plan.nstate == length(state) || return false
    equations = problem.equation_data
    length(equations) == length(plan.equation_ids) || return false
    @inbounds for i in eachindex(equations)
        objectid(equations[i]) == plan.equation_ids[i] || return false
        _constraint_rhs_id(equations[i]) == plan.rhs_ids[i] || return false
    end
    return true
end

"""Identity of an algebraic equation's right-hand side. A time- or space-dependent
boundary condition is re-applied by storing a NEW `F`/`F_expr` in the same
`EquationIR` (`_write_bc_value_to_eq!`), so a gauge value or constant the plan
folded in would otherwise go stale without the equation itself changing."""
_constraint_rhs_id(eq::EquationIR) = _is_zero_m_term(eq.mass) ?
    hash(objectid(eq.forcing), objectid(eq.forcing_expr)) : UInt(0)

_constraint_label(eq, e::Int) = string(something(get(eq, "equation_string", nothing), "equation $e"))

_is_tau_variable(x) = x isa ScalarField && isempty(x.bases)

"""Flatten `expr` into `(coefficient, term)` pairs. Sums, differences, negation
and products/quotients with a literal scalar are distributed; every other node
is a term."""
function _flatten_constraint_terms!(terms::Vector{Tuple{Float64, Any}}, expr, coefficient::Float64)
    (expr === nothing || expr isa ZeroOperator) && return terms
    if expr isa AddOperator
        _flatten_constraint_terms!(terms, expr.left, coefficient)
        _flatten_constraint_terms!(terms, expr.right, coefficient)
    elseif expr isa SubtractOperator
        _flatten_constraint_terms!(terms, expr.left, coefficient)
        _flatten_constraint_terms!(terms, expr.right, -coefficient)
    elseif expr isa NegateOperator
        _flatten_constraint_terms!(terms, expr.operand, -coefficient)
    elseif expr isa MultiplyOperator && _as_diagonal_scalar(expr.left) !== nothing
        _flatten_constraint_terms!(terms, expr.right, coefficient * _as_diagonal_scalar(expr.left))
    elseif expr isa MultiplyOperator && _as_diagonal_scalar(expr.right) !== nothing
        _flatten_constraint_terms!(terms, expr.left, coefficient * _as_diagonal_scalar(expr.right))
    elseif expr isa DivideOperator && _as_diagonal_scalar(expr.right) !== nothing
        denominator = _as_diagonal_scalar(expr.right)
        (isfinite(denominator) && !iszero(denominator)) || throw(ArgumentError(
            "Algebraic constraint divides by the scalar $denominator."))
        _flatten_constraint_terms!(terms, expr.left, coefficient / denominator)
    else
        push!(terms, (coefficient, expr))
    end
    return terms
end

"""Terms of `L - F = 0` for one parsed equation."""
function _constraint_equation_terms(eq::EquationIR, variables::Vector{<:Operand})
    flat = Tuple{Float64, Any}[]
    _flatten_constraint_terms!(flat, eq.linear, 1.0)
    rhs = eq.forcing_expr === nothing ? eq.forcing : eq.forcing_expr
    _flatten_constraint_terms!(flat, rhs, -1.0)
    return _ConstraintTerm[_ConstraintTerm(c, t, _detect_equation_variables(t, variables))
                           for (c, t) in flat]
end

function _integrates_all_axes(op::Integrate, field::ScalarField)
    coords = op.coord isa Tuple ? op.coord : (op.coord,)
    names = Set(String(c.name) for c in coords)
    return names == Set(String(b.meta.element_label) for b in field.bases)
end

"""`(variable, value)` when the equation is a gauge row `a*integ(v) + constants = 0`
over every axis of `v`, i.e. it fixes `integ(v) = value`; the reason it cannot be
used when it has that shape but a non-constant value; `nothing` otherwise."""
function _constraint_gauge(terms::Vector{_ConstraintTerm})
    target = nothing
    scale = 0.0
    constant = 0.0
    nonconstant = nothing
    for t in terms
        if isempty(t.variables)
            s = _as_diagonal_scalar(t.term)
            s === nothing ? (nonconstant = t.term) : (constant += t.coefficient * s)
        elseif target === nothing && t.term isa Integrate && length(t.variables) == 1
            v = only(t.variables)
            (v isa ScalarField && !isempty(v.bases) && t.term.operand isa ScalarField &&
             _operand_matches_variable(t.term.operand, v) && _integrates_all_axes(t.term, v)) ||
                return nothing
            target = v
            scale = t.coefficient
        else
            return nothing
        end
    end
    (target === nothing || iszero(scale)) && return nothing
    nonconstant === nothing ||
        return "a gauge row must fix `integ($(target.name))` to a constant, and " *
               "`$(expression_to_string(nonconstant))` is not one"
    return (target, -constant / scale)
end

"""Reduce a plan-build statistic over the field's ranks (identity in serial)."""
function _constraint_allreduce(value, op, dist::Distributor)
    (dist.size > 1 && MPI.Initialized()) || return value
    return MPI.Allreduce(value, op, dist.comm)
end

"""Coefficients of the constant field 1 on `field`'s layout. Every multiple of a
constant is represented through these, so no transform normalisation is assumed."""
function _unit_constant_coefficients(field::ScalarField)
    unit = _constant_field_from_template(field, 1.0; layout=:g)
    return _local_coeff(coeff_data!(unit))
end

"""Scratch field shaped and NAMED like `variable`, so the diagonal symbolization
matches operands by name without touching (and transforming) the live field."""
_constraint_template(variable::ScalarField) =
    ScalarField(variable.dist, variable.name, variable.bases, variable.dtype)

_add_constraint_weights(a::Real, b::Real) = a + b
_add_constraint_weights(a, b) = a .+ b

"""Build the exact per-mode solve of equation `e` for the scalar target `v`, or
return the reason it cannot be built."""
function _scalar_constraint_solve(e::Int, label::String, terms::Vector{_ConstraintTerm},
                                  v::ScalarField, first_index::IdDict{Any, Int},
                                  gauges::IdDict{Any, Tuple{Int, Float64}},
                                  consumed::Set{Int})
    target_terms = Tuple{Float64, Any}[]
    inputs = Tuple{Float64, Any, ScalarField}[]
    generic = Tuple{Float64, Any}[]
    constant = 0.0
    has_tau = false
    for t in terms
        vars = t.variables
        if isempty(vars)
            s = _as_diagonal_scalar(t.term)
            s === nothing ? push!(generic, (t.coefficient, t.term)) :
                            (constant += t.coefficient * s)
        elseif any(x -> x === v, vars)
            length(vars) == 1 ||
                return "the term `$(expression_to_string(t.term))` mixes `$(v.name)` with other variables"
            push!(target_terms, (t.coefficient, t.term))
        elseif any(_is_tau_variable, vars)
            (length(vars) == 1 && t.term isa ScalarField) ||
                return "the 0-D variable in `$(expression_to_string(t.term))` enters " *
                       "through an operator; only a bare tau term is supported"
            has_tau |= !iszero(t.coefficient)
        elseif length(vars) == 1 && only(vars) isa ScalarField
            push!(inputs, (t.coefficient, t.term, only(vars)))
        else
            push!(generic, (t.coefficient, t.term))   # vector-valued or nonlinear in known fields
        end
    end

    fourier = _constraint_all_fourier_bases(v)
    template = fourier ? _constraint_template(v) : nothing
    k2 = nothing
    k0 = nothing
    if fourier
        coeff_data!(template)
        k2 = _local_coeff(compute_wavenumber_squared_grid(template))
        k0 = k2 .== 0
    end

    # The target's operator L̂: a scalar for a constant multiple of `v`, else a
    # diagonal multiplier over the local |k|² grid.
    m = if all(term isa ScalarField for (_, term) in target_terms)
        s = sum(first, target_terms)
        iszero(s) && return "the coefficients of `$(v.name)` cancel"
        s
    else
        fourier || return "`$(v.name)` has a non-Fourier basis, so only a constant multiple " *
                          "of it can be solved for"
        multiplier = similar_zeros(k2, ComplexF64, size(k2)...)
        for (c, term) in target_terms
            _accumulate_diagonal_L!(multiplier, k2, term, c, template) ||
                return "`$(expression_to_string(term))` is not diagonal in Fourier space"
        end
        multiplier
    end

    # Modes where L̂ vanishes. Only k = 0 may be one (then the mean is a gauge);
    # anywhere else the operator is singular. Reduced over ranks so every rank
    # reaches the same decision.
    zero_modes = nothing
    mean_free = false
    if m isa AbstractArray
        local_scale = isempty(m) ? 0.0 : Float64(maximum(abs, m))
        scale = _constraint_allreduce(local_scale, max, v.dist)
        zero_modes = abs.(m) .<= 1e-13 * scale
        singular = _constraint_allreduce(count(zero_modes .& .!k0), +, v.dist)
        singular == 0 ||
            return "its operator on `$(v.name)` vanishes at $singular nonzero wavenumber(s), so it is singular"
        mean_free = _constraint_allreduce(any(zero_modes .& k0), |, v.dist)
    end

    gauge = get(gauges, v, nothing)
    pinned = false
    pin_value = 0.0
    if gauge !== nothing
        fourier || return "a gauge row is only supported for a purely Fourier `$(v.name)`"
        (has_tau || mean_free) ||
            return "the constraint and the gauge row both fix the mean of `$(v.name)`; " *
                   "add a tau variable to the constraint"
        pinned = true
        pin_value = gauge[2]
        push!(consumed, gauge[1])
    elseif has_tau
        return "it has a tau variable but no gauge row such as `integ($(v.name)) = 0` " *
               "fixes the mean of `$(v.name)`"
    elseif mean_free
        pinned = true   # zero-mean convention
    end

    neg_inv = if m isa AbstractArray
        inverse = similar(m)
        inverse .= ifelse.(zero_modes .| (pinned .& k0), zero(ComplexF64), .-inv.(m))
        inverse
    elseif pinned
        inverse = similar_zeros(k2, ComplexF64, size(k2)...)
        inverse .= ifelse.(k0, zero(ComplexF64), ComplexF64(-1 / m))
        inverse
    else
        -1 / m
    end

    pin = nothing
    if pinned && !iszero(pin_value)
        volume = prod(b.meta.bounds[2] - b.meta.bounds[1] for b in v.bases)
        unit = _unit_constant_coefficients(v)
        pin = similar_zeros(k2, ComplexF64, size(k2)...)
        pin .= ifelse.(k0, (pin_value / volume) .* unit, zero(ComplexF64))
    end

    constant_coefficients = iszero(constant) ? nothing :
                            constant .* _unit_constant_coefficients(v)

    # Inputs: a field with the target's layout enters with a scalar weight, or a
    # multiplier when its operator is diagonal; anything else is evaluated.
    weights = IdDict{Any, Any}()
    order = ScalarField[]
    for (c, term, x) in inputs
        weight = nothing
        if x.dist === v.dist && x.bases == v.bases
            if term isa ScalarField
                weight = c
            elseif fourier
                input_template = _constraint_template(x)
                coeff_data!(input_template)
                multiplier = similar_zeros(k2, ComplexF64, size(k2)...)
                _accumulate_diagonal_L!(multiplier, k2, term, c, input_template) &&
                    (weight = multiplier)
            end
        end
        if weight === nothing
            push!(generic, (c, term))
        elseif haskey(weights, x)
            weights[x] = _add_constraint_weights(weights[x], weight)
        else
            weights[x] = weight
            push!(order, x)
        end
    end
    constraint_inputs = _ConstraintInput[_ConstraintInput(first_index[x], weights[x]) for x in order]

    return _ConstraintSolve(e, label, first_index[v], v, neg_inv, pin, constraint_inputs,
                            constant_coefficients, generic, nothing)
end

"""`coefficient * term`, without a multiplication node for ±1."""
_scaled_constraint_term(coefficient::Float64, term) =
    coefficient == 1 ? term :
    coefficient == -1 ? NegateOperator(term) :
    MultiplyOperator(ConstantOperator(coefficient), term)

"""Build the refresh of equation `e` for the vector target `v`: only a constant
multiple of `v` itself, with the rest evaluated (exact for the spectral operators
it contains, e.g. `skew(grad(ψ))`). Returns the reason when that form does not apply."""
function _vector_constraint_solve(e::Int, label::String, terms::Vector{_ConstraintTerm},
                                  v::VectorField, first_index::IdDict{Any, Int})
    a = 0.0
    rest = Tuple{Float64, Any}[]
    for t in terms
        vars = t.variables
        if any(x -> x === v, vars)
            (length(vars) == 1 && t.term isa VectorField) ||
                return "only a constant multiple of the vector `$(v.name)` can be solved " *
                       "for; `$(expression_to_string(t.term))` is not one"
            a += t.coefficient
        elseif isempty(vars)
            s = _as_diagonal_scalar(t.term)
            (s !== nothing && iszero(s)) ||
                return "the term `$(expression_to_string(t.term))` is not a vector field"
        elseif any(_is_tau_variable, vars)
            return "the 0-D variable in `$(expression_to_string(t.term))` cannot enter a " *
                   "vector equation"
        else
            push!(rest, (t.coefficient, t.term))
        end
    end
    iszero(a) && return "the coefficients of `$(v.name)` cancel"
    expr = isempty(rest) ? nothing :
           combine_operators(Any[_scaled_constraint_term(-c / a, term) for (c, term) in rest])
    return _ConstraintSolve(e, label, first_index[v], v, nothing, nothing, _ConstraintInput[],
                            nothing, Tuple{Float64, Any}[], expr)
end

"""Classify every algebraic equation of `problem`: the exact solves, in dependency
order, and the equations the refresh cannot solve."""
function _build_constraint_refresh_plan(problem::Problem, state::Vector{<:ScalarField})
    equations = problem.equation_data
    variables = problem.variables
    plan = _ConstraintRefreshPlan(UInt[objectid(eq) for eq in equations],
                                  UInt[_constraint_rhs_id(eq) for eq in equations],
                                  length(state), _ConstraintSolve[], _UnsolvedConstraint[])

    # Flattened state index of each variable (first component of a vector) and
    # which state entries are time-stepped.
    first_index = IdDict{Any, Int}()
    next = 1
    for var in variables
        first_index[var] = next
        # Mirrors `collect_state_fields`, which gives other operands no state entry.
        var isa Union{ScalarField, VectorField, TensorField} &&
            (next += length(scalar_components(var)))
    end
    evolved = falses(length(state))
    for eq in equations
        _is_zero_m_term(eq.mass) && continue
        for i in _find_time_derivative_targets(eq.mass, state, variables)
            1 <= i <= length(state) && (evolved[i] = true)
        end
    end

    algebraic = Int[e for (e, eq) in enumerate(equations) if _is_zero_m_term(eq.mass)]
    isempty(algebraic) && return plan
    terms = Dict{Int, Vector{_ConstraintTerm}}(
        e => _constraint_equation_terms(equations[e], variables) for e in algebraic)

    gauges = IdDict{Any, Tuple{Int, Float64}}()
    gauge_rows = Int[]
    unusable_gauges = Dict{Int, String}()
    for e in algebraic
        gauge = _constraint_gauge(terms[e])
        gauge === nothing && continue
        if gauge isa String
            unusable_gauges[e] = gauge
            continue
        end
        haskey(gauges, gauge[1]) && continue   # a second gauge on one variable stays unsolved
        gauges[gauge[1]] = (e, gauge[2])
        push!(gauge_rows, e)
    end

    solved = IdDict{Any, Nothing}()
    unknown(x) = (x isa VectorField || x isa TensorField ||
                  (x isa ScalarField && !isempty(x.bases))) &&
                 !evolved[first_index[x]] && !haskey(solved, x)
    unknowns_of(e) = unique(objectid,
                            Operand[x for t in terms[e] for x in t.variables if unknown(x)])

    # Peel off equations that determine exactly one not-yet-solved algebraic
    # variable; each solve can unblock the equations that read its variable.
    pending = Int[e for e in algebraic if !(e in gauge_rows) && !haskey(unusable_gauges, e)]
    done = Set{Int}()
    consumed = Set{Int}()
    reasons = Dict{Int, String}()
    progress = true
    while progress
        progress = false
        for e in pending
            e in done && continue
            unknowns = unknowns_of(e)
            length(unknowns) == 1 || continue
            v = only(unknowns)
            label = _constraint_label(equations[e], e)
            result = if v isa ScalarField
                _scalar_constraint_solve(e, label, terms[e], v, first_index, gauges, consumed)
            elseif v isa VectorField
                _vector_constraint_solve(e, label, terms[e], v, first_index)
            else
                "the tensor variable `$(_operand_name(v))` cannot be refreshed"
            end
            if result isa _ConstraintSolve
                push!(plan.solves, result)
                solved[v] = nothing
                push!(done, e)
                delete!(reasons, e)
                progress = true
            else
                reasons[e] = result
            end
        end
    end

    for e in pending
        e in done && continue
        # A row on 0-D variables and constants only determines tau variables,
        # which carry no storage: there is nothing to refresh.
        all(t -> all(_is_tau_variable, t.variables), terms[e]) && continue
        unknowns = unknowns_of(e)
        reason = get(reasons, e, nothing)
        if reason === nothing
            reason = isempty(unknowns) ?
                "it determines no algebraic variable the refresh could update (its " *
                "variables are time-stepped, 0-D, or fixed by another equation), so only " *
                "an implicit solve can impose it" :
                "it couples the algebraic variables " *
                join(("`$(_operand_name(x))`" for x in unknowns), ", ") *
                ", which no sequence of single-variable solves determines"
        end
        push!(plan.unsolved, _UnsolvedConstraint(e, _constraint_label(equations[e], e), reason))
    end
    for e in gauge_rows
        e in consumed && continue
        push!(plan.unsolved, _UnsolvedConstraint(e, _constraint_label(equations[e], e),
            "it fixes the mean of a variable no refreshable constraint solves"))
    end
    for (e, reason) in sort!(collect(unusable_gauges); by=first)
        push!(plan.unsolved, _UnsolvedConstraint(e, _constraint_label(equations[e], e), reason))
    end
    return plan
end

"""Run every exact solve of `plan` on `state` (problem variables already aliased to it)."""
function _apply_constraint_plan!(plan::_ConstraintRefreshPlan, problem::Problem,
                                 state::Vector{<:ScalarField})
    for solve in plan.solves
        if solve.target_var isa VectorField
            _apply_vector_constraint!(solve, problem, state)
        else
            _apply_scalar_constraint!(solve, problem, state)
        end
    end
    return state
end

"""Coefficient buffer the solve overwrites in full: the existing one when it has
the right shape (no transform of the stale value), else allocated by a transform."""
function _constraint_target_coefficients!(target::ScalarField, shape)
    cd = get_coeff_data(target)
    if cd === nothing || (shape !== nothing && size(_local_coeff(cd)) != shape)
        cd = coeff_data!(target)
    end
    return _local_coeff(cd)
end

_constraint_shape(neg_inv::AbstractArray) = size(neg_inv)
_constraint_shape(::Any) = nothing

# Function barriers: the plan stores weights untyped, these see concrete arrays.
function _constraint_accumulate!(dest::AbstractArray, weight, x::AbstractArray, first::Bool)
    size(x) == size(dest) || throw(DimensionMismatch(
        "constraint refresh: input coefficients $(size(x)) vs target $(size(dest))"))
    first ? (dest .= weight .* x) : (dest .+= weight .* x)
    return dest
end

_constraint_finish!(dest::AbstractArray, neg_inv, ::Nothing) = (dest .*= neg_inv; dest)
_constraint_finish!(dest::AbstractArray, neg_inv, pin::AbstractArray) =
    (dest .= neg_inv .* dest .+ pin; dest)

# The common one-input case (Δψ + τ - ζ = 0) in a single fused pass.
function _constraint_single!(dest::AbstractArray, neg_inv, weight, x::AbstractArray, pin)
    size(x) == size(dest) || throw(DimensionMismatch(
        "constraint refresh: input coefficients $(size(x)) vs target $(size(dest))"))
    if pin === nothing
        dest .= neg_inv .* (weight .* x)
    else
        dest .= neg_inv .* (weight .* x) .+ pin
    end
    return dest
end

function _apply_scalar_constraint!(solve::_ConstraintSolve, problem::Problem,
                                   state::Vector{<:ScalarField})
    target = state[solve.target]
    dest = _constraint_target_coefficients!(target, _constraint_shape(solve.neg_inv))
    if length(solve.inputs) == 1 && isempty(solve.generic) && solve.constant === nothing
        input = only(solve.inputs)
        x = _local_coeff(coeff_data!(state[input.index]))
        _constraint_single!(dest, solve.neg_inv, input.weight, x, solve.pin)
    else
        first = true
        for input in solve.inputs
            x = _local_coeff(coeff_data!(state[input.index]))
            _constraint_accumulate!(dest, input.weight, x, first)
            first = false
        end
        for (c, term) in solve.generic
            value = evaluate_solver_expression(term, problem.variables; layout=:c,
                                               template=solve.target_var)
            value isa ScalarField || throw(ArgumentError(
                "constraint refresh of `$(solve.label)`: the term " *
                "`$(expression_to_string(term))` evaluated to $(typeof(value)), not a " *
                "scalar field"))
            _constraint_accumulate!(dest, c, _local_coeff(coeff_data!(value)), first)
            first = false
        end
        if solve.constant !== nothing
            _constraint_accumulate!(dest, 1.0, solve.constant, first)
            first = false
        end
        first && fill!(dest, zero(eltype(dest)))
        _constraint_finish!(dest, solve.neg_inv, solve.pin)
    end
    target.current_layout = :c
    return target
end

function _apply_vector_constraint!(solve::_ConstraintSolve, problem::Problem,
                                   state::Vector{<:ScalarField})
    target = solve.target_var::VectorField
    expr = solve.vector_expr
    ncomp = length(target.components)
    if expr === nothing
        for i in 1:ncomp
            component = state[solve.target + i - 1]
            fill!(_constraint_target_coefficients!(component, nothing), 0)
            component.current_layout = :c
        end
        return target
    end
    _try_refresh_fourier_streamfunction_velocity!(target, expr) && return target
    value = evaluate_solver_expression(expr, problem.variables; layout=:c)
    (value isa VectorField && length(value.components) == ncomp) || throw(ArgumentError(
        "constraint refresh of `$(solve.label)`: `$(expression_to_string(expr))` evaluated " *
        "to $(typeof(value)), not a $ncomp-component vector field"))
    for i in 1:ncomp
        component = value.components[i]
        coeff_data!(component)
        copy_field_data!(state[solve.target + i - 1], component; preserve_layout=true)
    end
    return target
end

@inline function _constraint_all_fourier_bases(field::ScalarField)
    !isempty(field.bases) && all(is_fourier_axis, field.bases)
end

@inline _coefficient_array(data::PencilArrays.PencilArray) = parent(data)
@inline _coefficient_array(data::AbstractArray) = data

"""Refresh `u = skew(grad(psi))` in existing 2D pure-Fourier buffers."""
function _try_refresh_fourier_streamfunction_velocity!(target, expr)
    target isa VectorField || return false
    expr isa Skew || return false
    gradient = expr.operand
    gradient isa Gradient || return false
    source = gradient.operand
    source isa ScalarField || return false
    length(target.components) == 2 || return false
    length(gradient.coordsys.names) == 2 || return false
    _constraint_all_fourier_bases(source) || return false
    all(comp -> comp.bases == source.bases, target.components) || return false

    source_coeff = coeff_data!(source)
    source_coeff === nothing && return false
    source_data = _coefficient_array(source_coeff)

    # skew(grad(psi)) = (-dpsi/dy, dpsi/dx)
    derivative_coords = (gradient.coordsys[gradient.coordsys.names[2]],
                         gradient.coordsys[gradient.coordsys.names[1]])
    signs = (-1.0, 1.0)
    @inbounds for i in 1:2
        component = target.components[i]
        component_coeff = coeff_data!(component)
        component_coeff === nothing && return false
        copyto!(_coefficient_array(component_coeff), source_data)
        axis = _resolve_diff_axis(derivative_coords[i], component.bases)
        axis == 0 && return false
        _apply_lazy_diff_coeff!(component, 1, axis)
        signs[i] == 1.0 || (_coefficient_array(component_coeff) .*= signs[i])
        component.current_layout = :c
    end
    return true
end

"""
Build the k² array of a field's local coefficient grid.

The k² grid is a constant of the domain/decomposition, so it is cached (per rank,
keyed by basis sizes + local coefficient shape + element type) and reused. Callers
(the lazy fractional Laplacian) only read it, so sharing is safe.
"""
const _POISSON_K2_CACHE = Dict{Tuple, Any}()

function _build_k_squared(field::ScalarField)
    coeff_data = coeff_data!(field)

    if coeff_data === nothing
        return nothing
    end

    # Get local shape
    local_shape = if isa(coeff_data, PencilArrays.PencilArray)
        size(parent(coeff_data))
    else
        size(coeff_data)
    end

    T = real(eltype(coeff_data))
    # Include the storage backend in the cache key.  A CPU k² array cannot be
    # broadcast with CuArray coefficients, and rebuilding/copying k² at every RK
    # stage would add an unnecessary host-to-device transfer.
    backend = (typeof(coeff_data), architecture(coeff_data))
    basis_key = Tuple(b === nothing ? nothing :
                      (typeof(b), b.meta.size, b.meta.bounds) for b in field.bases)
    key = (basis_key, local_shape, T, backend)
    cached = get(_POISSON_K2_CACHE, key, nothing)
    cached !== nothing && return cached

    if is_gpu_array(coeff_data)
        k_squared = compute_wavenumber_squared_grid(field)
        _POISSON_K2_CACHE[key] = k_squared
        return k_squared
    end

    k_squared = zeros(T, local_shape)

    # Get wavenumber contributions from each basis
    for (dim, basis) in enumerate(field.bases)
        k_array = _get_wavenumber_array_for_poisson(field, dim, local_shape)
        if k_array !== nothing
            for idx in CartesianIndices(k_squared)
                k_squared[idx] += k_array[idx]^2
            end
        end
    end

    _POISSON_K2_CACHE[key] = k_squared
    return k_squared
end

"""
Get wavenumber array for a specific dimension, handling MPI offsets.
"""
function _get_wavenumber_array_for_poisson(field::ScalarField, dim::Int, local_shape)
    if dim > length(field.bases)
        return nothing
    end

    basis = field.bases[dim]
    coeff_data = get_coeff_data(field)
    is_mpi = isa(coeff_data, PencilArrays.PencilArray)
    global_shape = is_mpi ? PencilArrays.size_global(coeff_data) : size(coeff_data)
    dim <= length(global_shape) || return zeros(local_shape)

    if isa(basis, RealFourier)
        k_global = _axis_uses_rfft(_field_transform_bundle(field), dim) ?
                   wavenumbers_rfft(basis) : wavenumbers_fft(basis)
    elseif isa(basis, ComplexFourier)
        k_global = wavenumbers(basis)
    else
        return zeros(local_shape)
    end

    # Initialize wavenumber array
    k_array = zeros(local_shape)

    if is_mpi
        pencil = PencilArrays.pencil(coeff_data)
        local_axes = pencil.axes_local
        perm = PencilArrays.permutation(coeff_data)
        # Tuple(NoPermutation()) is `nothing`, NOT identity — guard before findfirst.
        perm_raw = Tuple(perm)
        perm_tuple = perm_raw === nothing ? ntuple(identity, ndims(coeff_data)) : perm_raw
        physical_axis = findfirst(==(dim), perm_tuple)
        if physical_axis === nothing
            physical_axis = dim
        end

        if dim > length(local_axes)
            return zeros(local_shape)
        end

        global_start = first(local_axes[dim])

        for idx in CartesianIndices(k_array)
            local_idx = Tuple(idx)[physical_axis]
            global_idx = global_start + local_idx - 1
            k_array[idx] = k_global[global_idx]
        end
    else
        for idx in CartesianIndices(k_array)
            global_idx = Tuple(idx)[dim]
            k_array[idx] = k_global[global_idx]
        end
    end

    return k_array
end
