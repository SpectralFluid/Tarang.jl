"""
    Build system matrices for problem following structure.
    Following subsystems:build_subproblem_matrices (subsystems:72-81) and
    Subproblem.build_matrices (subsystems:497-576).
    """
function build_matrices(problem::Problem)
    
    if length(problem.equations) == 0
        throw(ArgumentError("No equations specified"))
    end
    
    # Build matrix expressions from equations (following problems:_build_matrix_expressions)
    build_matrix_expressions!(problem)
    
    # Compute field sizes for each equation and variable
    eqn_sizes = [compute_field_size(eq_data) for eq_data in problem.equation_data]
    var_sizes = [compute_field_size(var) for var in problem.variables]

    total_rows = sum(eqn_sizes)  # Total rows
    total_cols = sum(var_sizes)  # Total columns

    @debug "Building matrices: equations=$total_rows, variables=$total_cols"

    # Matrix names to build (following convention)
    matrix_names = ["M", "L"]  # M = mass matrix, L = stiffness matrix

    # Build sparse matrices following subsystems:513-537 pattern
    matrices = Dict{String, Any}()
    for name in matrix_names
        # Collect sparse matrix entries (ComplexF64 for spectral methods)
        data, rows, cols = ComplexF64[], Int[], Int[]
        
        i0 = 0  # Row offset
        for (eq_idx, eq_data) in enumerate(problem.equation_data)
            eqn_size = eqn_sizes[eq_idx]
            if eqn_size > 0 && check_equation_condition(eq_data)
                # Get expression matrix blocks for this equation
                expr = get_matrix_expression(eq_data, name)
                if expr !== nothing && !is_zero_expression(expr)
                    # Build expression matrices for each variable
                    j0 = 0  # Column offset
                    for (var_idx, var) in enumerate(problem.variables)
                        var_size = var_sizes[var_idx]
                        if var_size > 0
                            # Get matrix block for this variable
                            block = build_expression_matrix_block(expr, var, eqn_size, var_size)
                            if !isempty(block.nzval)
                                # Add to sparse matrix data
                                # SparseMatrixCSC stores: rowval (row indices), colptr (column pointers), nzval (values)
                                # We need to expand colptr to get column indices for each non-zero
                                block_rows, block_cols, block_vals = findnz(block)
                                append!(data, block_vals)
                                append!(rows, i0 .+ block_rows)
                                append!(cols, j0 .+ block_cols)
                            end
                        end
                        j0 += var_size
                    end
                end
            end
            i0 += eqn_size
        end
        
        # Create sparse matrix
        if !isempty(data)
            # Filter small entries (following entry_cutoff pattern)
            entry_cutoff = 1e-14
            significant = abs.(data) .>= entry_cutoff
            data = data[significant]
            rows = rows[significant]
            cols = cols[significant]
            
            matrices[name] = sparse(rows, cols, data, total_rows, total_cols)
        else
            # Empty matrix
            matrices[name] = spzeros(ComplexF64, total_rows, total_cols)
        end

        @debug "Built matrix $name: size=($total_rows, $total_cols), nnz=$(nnz(matrices[name]))"
    end
    
    # Build forcing vector (RHS terms)
    F_vector = build_forcing_vector(problem, eqn_sizes, total_rows)
    
    # Return matrices in standard format
    L_matrix = matrices["L"]
    M_matrix = matrices["M"] 
    
    # Only log on rank 0 to avoid repeated messages
    if length(problem.variables) > 0 && problem.variables[1].dist.rank == 0
        @info "Matrix building completed: L=$(size(L_matrix)), M=$(size(M_matrix)), F=$(length(F_vector))"
    end

    return L_matrix, M_matrix, F_vector
end

"""
    Build matrix expressions from parsed equations.
    Following problems:_build_matrix_expressions patterns.
    """
function build_matrix_expressions!(problem::Problem)
    
    problem.equation_data = Dict{String, Any}[]
    
    for (i, equation_str) in enumerate(problem.equations)
        try
            # Parse equation
            lhs, rhs = parse_equation(equation_str, problem.namespace)
            
            # Build matrix expressions following problem type
            eq_data = build_equation_expressions(lhs, rhs, problem.variables)
            eq_data["equation_index"] = i
            eq_data["equation_string"] = equation_str
            if !haskey(eq_data, "equation_size")
                vars = get(eq_data, "equation_variables", get(eq_data, "variables", problem.variables))
                if isa(vars, Vector)
                    eq_sz = sum(field_dofs(var) for var in vars)
                    if eq_sz <= 0
                        eq_sz = sum(field_dofs(var) for var in problem.variables)
                    end
                    eq_data["equation_size"] = eq_sz
                end
            end

            push!(problem.equation_data, eq_data)
            
        catch e
            @error "Failed to build matrix expressions for equation $i: $equation_str" exception=e
            # Create fallback equation data
            fallback_data = Dict(
                "M" => nothing,
                "L" => UnknownOperator(equation_str),
                "F" => ZeroOperator(),
                "equation_index" => i,
                "equation_string" => equation_str,
                "equation_size" => 0
            )
            push!(problem.equation_data, fallback_data)
        end
    end
end

"""
    Build matrix expressions from LHS and RHS operators.
    Following _build_matrix_expressions patterns.
    """
function build_equation_expressions(lhs, rhs, variables::Vector)
    
    eq_data = Dict{String, Any}()
    
    # Split LHS into mass matrix (time derivatives) and stiffness matrix (spatial) terms
    # Following IVP pattern: M.dt(X) + L.X = F (problems:328)
    M_terms, L_terms = split_time_spatial_operators(lhs)
    
    # Store matrix expressions
    eq_data["M"] = combine_operators(M_terms)      # Mass matrix terms
    eq_data["L"] = combine_operators(L_terms)      # Stiffness matrix terms  
    eq_data["F"] = rhs                             # Forcing terms

    # Determine which variables participate in this equation
    eq_vars = _detect_equation_variables(lhs, variables)
    if isempty(eq_vars)
        # Some constraint equations (e.g., BCs) only reference variables on RHS
        eq_vars = _detect_equation_variables(rhs, variables)
    end
    if isempty(eq_vars)
        # Fall back to all variables to keep matrix sizes consistent
        eq_vars = copy(variables)
    end

    eq_data["equation_variables"] = eq_vars
    eq_size = sum(field_dofs(var) for var in eq_vars)
    if eq_size <= 0
        eq_size = sum(field_dofs(var) for var in variables)
    end
    eq_data["equation_size"] = eq_size
    
    # Metadata
    eq_data["variables"] = variables
    eq_data["lhs"] = lhs
    eq_data["rhs"] = rhs
    
    return eq_data
end

function _detect_equation_variables(expr, variables::Vector{<:Operand})
    found = Operand[]
    _collect_equation_variables!(found, expr, variables)
    return found
end

function _collect_equation_variables!(found::Vector{Operand}, expr, variables::Vector{<:Operand})
    expr === nothing && return

    if isa(expr, ScalarField) || isa(expr, VectorField) || isa(expr, TensorField)
        for var in variables
            if _operand_matches_variable(expr, var)
                _maybe_add_variable!(found, var)
                break
            end
        end
    end

    if hasfield(typeof(expr), :left)
        _collect_equation_variables!(found, getfield(expr, :left), variables)
    end
    if hasfield(typeof(expr), :right)
        _collect_equation_variables!(found, getfield(expr, :right), variables)
    end
    if hasfield(typeof(expr), :operand)
        _collect_equation_variables!(found, getfield(expr, :operand), variables)
    end
    if hasfield(typeof(expr), :operands)
        ops = getfield(expr, :operands)
        if ops !== nothing
            for op in ops
                _collect_equation_variables!(found, op, variables)
            end
        end
    end
    if hasfield(typeof(expr), :array)
        _collect_equation_variables!(found, getfield(expr, :array), variables)
    end
    if hasfield(typeof(expr), :indices)
        idxs = getfield(expr, :indices)
        if idxs !== nothing
            for idx in idxs
                _collect_equation_variables!(found, idx, variables)
            end
        end
    end
    if hasfield(typeof(expr), :base)
        _collect_equation_variables!(found, getfield(expr, :base), variables)
    end
    if hasfield(typeof(expr), :exponent)
        _collect_equation_variables!(found, getfield(expr, :exponent), variables)
    end
end

function _maybe_add_variable!(found::Vector{Operand}, var::Operand)
    for existing in found
        if existing === var
            return
        end
        existing_name = _operand_name(existing)
        new_name = _operand_name(var)
        if existing_name !== nothing && existing_name == new_name
            return
        end
    end
    push!(found, var)
end

@inline function _operand_matches_variable(expr, var::Operand)
    (expr === var) && return true
    expr_name = _operand_name(expr)
    var_name = _operand_name(var)
    return expr_name !== nothing && expr_name == var_name
end

@inline function _operand_name(var)
    return hasfield(typeof(var), :name) ? getfield(var, :name) : nothing
end

"""
    Split operator into time derivative (mass matrix) and spatial (stiffness) terms.
    Following operators split pattern.
    """
function split_time_spatial_operators(operator)
    
    M_terms = []  # Time derivative terms
    L_terms = []  # Spatial terms
    empty_namespace = Dict{String, Any}()
    
    if isa(operator, TimeDerivative)
        # Pure time derivative
        push!(M_terms, operator)
        
    elseif isa(operator, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Pure spatial operator
        push!(L_terms, operator)
        
    elseif isa(operator, AddOperator)
        # Split addition terms recursively
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)
        append!(M_terms, left_M)
        append!(L_terms, left_L)
        append!(M_terms, right_M)
        append!(L_terms, right_L)
        
    elseif isa(operator, SubtractOperator)
        # Split subtraction terms recursively
        # For A - B, we split both and negate the right side terms
        left_M, left_L = split_time_spatial_operators(operator.left)
        right_M, right_L = split_time_spatial_operators(operator.right)

        # Add left terms directly
        append!(M_terms, left_M)
        append!(L_terms, left_L)

        # Right terms need negation - wrap in NegateOperator or multiply by -1
        for term in right_M
            push!(M_terms, NegateOperator(term))
        end
        for term in right_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, NegateOperator)
        inner_M, inner_L = split_time_spatial_operators(operator.operand)
        for term in inner_M
            push!(M_terms, NegateOperator(term))
        end
        for term in inner_L
            push!(L_terms, NegateOperator(term))
        end

    elseif isa(operator, MultiplyOperator)
        coeff = nothing
        inner = nothing

        if _is_constant_coefficient_strict(operator.left, empty_namespace) &&
           !_is_constant_coefficient_strict(operator.right, empty_namespace)
            coeff = operator.left
            inner = operator.right
        elseif _is_constant_coefficient_strict(operator.right, empty_namespace) &&
               !_is_constant_coefficient_strict(operator.left, empty_namespace)
            coeff = operator.right
            inner = operator.left
        end

        if inner !== nothing
            scaled = MultiplyOperator(coeff, inner)
            if isa(inner, TimeDerivative)
                push!(M_terms, scaled)
            elseif isa(inner, Union{Laplacian, Gradient, Divergence, Differentiate}) || hasfield(typeof(inner), :name)
                push!(L_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end

    elseif isa(operator, DivideOperator)
        if _is_constant_coefficient_strict(operator.right, empty_namespace)
            scaled = DivideOperator(operator.left, operator.right)
            if isa(operator.left, TimeDerivative)
                push!(M_terms, scaled)
            else
                push!(L_terms, scaled)
            end
        else
            push!(L_terms, operator)
        end
        
    elseif hasfield(typeof(operator), :name)
        # Direct variable reference -> identity in L
        push!(L_terms, operator)
        
    else
        # Other operators go to L by default
        push!(L_terms, operator)
    end
    
    return M_terms, L_terms
end

"""Combine operator terms into single expression"""
function combine_operators(terms::Vector)
    if isempty(terms)
        return ZeroOperator()
    elseif length(terms) == 1
        return terms[1]
    else
        # Combine with addition
        result = terms[1]
        for i in 2:length(terms)
            result = AddOperator(result, terms[i])
        end
        return result
    end
end

# Supporting functions for matrix building

function field_dofs(field::ScalarField)
    if get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif get_grid_data(field) !== nothing
        return length(get_grid_data(field))
    else
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    end
end

field_dofs(field::VectorField) = sum(field_dofs(comp) for comp in field.components)
field_dofs(field::TensorField) = sum(field_dofs(comp) for comp in vec(field.components))

"""Compute size (degrees of freedom) of field or equation data"""
function compute_field_size(field_or_data)
    if isa(field_or_data, Dict)
        if haskey(field_or_data, "equation_size")
            return field_or_data["equation_size"]
        elseif haskey(field_or_data, "equation_variables")
            vars = field_or_data["equation_variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        elseif haskey(field_or_data, "variables")
            vars = field_or_data["variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        end
        return 0
    elseif isa(field_or_data, ScalarField)
        return field_dofs(field_or_data)
    elseif isa(field_or_data, VectorField) || isa(field_or_data, TensorField)
        return field_dofs(field_or_data)
    elseif hasfield(typeof(field_or_data), :buffers) && get_coeff_data(field_or_data) !== nothing
        return length(get_coeff_data(field_or_data))
    elseif hasfield(typeof(field_or_data), :buffers) && get_grid_data(field_or_data) !== nothing
        return length(get_grid_data(field_or_data))
    else
        return 0
    end
end

"""
    Check if equation should be included in matrix assembly.

    An equation is included if:
    1. It has valid matrix expressions (M, L, or F)
    2. It is marked as enabled (if "enabled" key exists)
    3. It has a valid condition (if "condition" key exists)
    4. It references at least one problem variable
    5. The equation is well-formed (not flagged as invalid)

    Following patterns where equations can be conditionally
    included/excluded based on wavenumber, problem parameters, etc.
    """
function check_equation_condition(eq_data::Dict)

    # Check if equation is explicitly disabled
    if haskey(eq_data, "enabled") && !eq_data["enabled"]
        @debug "Equation excluded: explicitly disabled" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has a condition function that evaluates to false
    if haskey(eq_data, "condition")
        condition = eq_data["condition"]
        if isa(condition, Bool)
            if !condition
                @debug "Equation excluded: condition is false" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        elseif isa(condition, Function)
            # Condition is a function - evaluate it
            try
                result = condition(eq_data)
                if !result
                    @debug "Equation excluded: condition function returned false" eq_index=get(eq_data, "equation_index", 0)
                    return false
                end
            catch e
                @warn "Equation condition evaluation failed, including equation" exception=e
            end
        end
    end

    # Check if equation is flagged as invalid
    if get(eq_data, "is_invalid", false)
        @debug "Equation excluded: flagged as invalid" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has any matrix content
    has_M = haskey(eq_data, "M") && !is_zero_expression(eq_data["M"])
    has_L = haskey(eq_data, "L") && !is_zero_expression(eq_data["L"])
    has_F = haskey(eq_data, "F") && !is_zero_expression(eq_data["F"])

    if !has_M && !has_L && !has_F
        @debug "Equation excluded: no matrix content (M, L, F all zero/missing)" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check equation size
    eq_size = get(eq_data, "equation_size", 0)
    if eq_size <= 0
        @debug "Equation excluded: equation_size <= 0" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check wavenumber conditions (for spectral problems)
    if haskey(eq_data, "valid_modes")
        valid_modes = eq_data["valid_modes"]
        current_mode = get(eq_data, "current_mode", nothing)
        if current_mode !== nothing && !in(current_mode, valid_modes)
            @debug "Equation excluded: mode not in valid_modes" current_mode valid_modes
            return false
        end
    end

    # Check for wavenumber-based conditions (k=0 special handling, etc.)
    if haskey(eq_data, "exclude_k_zero") && eq_data["exclude_k_zero"]
        wavenumber = get(eq_data, "wavenumber", nothing)
        if wavenumber !== nothing
            # Check if all wavenumber components are zero
            if isa(wavenumber, Number) && wavenumber == 0
                @debug "Equation excluded: k=0 mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            elseif isa(wavenumber, Tuple) && all(k -> k == 0, wavenumber)
                @debug "Equation excluded: k=(0,...,0) mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        end
    end

    # Check for gauge conditions (pressure gauge, etc.)
    if haskey(eq_data, "is_gauge_condition") && eq_data["is_gauge_condition"]
        # Gauge conditions may have special handling
        gauge_mode = get(eq_data, "gauge_mode", nothing)
        current_mode = get(eq_data, "current_mode", nothing)

        if gauge_mode !== nothing && current_mode !== nothing
            if gauge_mode != current_mode
                # Only include gauge condition for specific mode
                return false
            end
        end
    end

    # Check if this is a boundary condition equation
    if haskey(eq_data, "is_boundary_condition") && eq_data["is_boundary_condition"]
        # Boundary conditions are always included if they're valid
        bc_valid = get(eq_data, "bc_valid", true)
        if !bc_valid
            @debug "Equation excluded: boundary condition marked invalid"
            return false
        end
    end

    # All checks passed
    return true
end

"""
    Check if equation data is structurally valid.
    Returns (is_valid::Bool, error_message::Union{String,Nothing})
    """
function is_equation_valid(eq_data::Dict)

    # Must have equation string
    if !haskey(eq_data, "equation_string")
        return (false, "Missing equation_string")
    end

    # Must have LHS
    if !haskey(eq_data, "lhs")
        return (false, "Missing LHS expression")
    end

    # Check for parse errors
    if haskey(eq_data, "parse_error")
        return (false, "Parse error: $(eq_data["parse_error"])")
    end

    # Check LHS structure if we have the expression
    lhs = eq_data["lhs"]
    if lhs !== nothing
        is_valid_lhs, lhs_info = is_proper_lhs_structure(lhs)
        if !is_valid_lhs
            return (false, "Invalid LHS structure: $(lhs_info[:error_message])")
        end
    end

    return (true, nothing)
end

"""
    Set a condition for equation inclusion in matrix assembly.
    """
function set_equation_condition!(eq_data::Dict, condition::Union{Bool, Function})
    eq_data["condition"] = condition
end

"""Enable an equation for matrix assembly."""
function enable_equation!(eq_data::Dict)
    eq_data["enabled"] = true
end

"""Disable an equation from matrix assembly."""
function disable_equation!(eq_data::Dict)
    eq_data["enabled"] = false
end

"""
    Set the valid wavenumber modes for this equation.
    The equation will only be included for these modes.
    """
function set_valid_modes!(eq_data::Dict, modes::Union{Vector, Set, AbstractRange})
    eq_data["valid_modes"] = Set(modes)
end

"""
    Exclude this equation from k=0 (homogeneous) mode.
    Useful for gauge conditions in incompressible flow problems.
    """
function exclude_k_zero!(eq_data::Dict, exclude::Bool=true)
    eq_data["exclude_k_zero"] = exclude
end

"""Get matrix expression from equation data"""
function get_matrix_expression(eq_data::Dict, matrix_name::String)
    return get(eq_data, matrix_name, nothing)
end

"""Check if expression is effectively zero"""
function is_zero_expression(expr)
    return isa(expr, ZeroOperator) || expr === nothing
end

@inline _zero_block(eqn_size::Int, var_size::Int) = spzeros(ComplexF64, eqn_size, var_size)

function _identity_block(eqn_size::Int, var_size::Int; scale::Number=1.0)
    if eqn_size == 0 || var_size == 0
        return _zero_block(eqn_size, var_size)
    end
    diag_len = min(eqn_size, var_size)
    vals = fill(ComplexF64(scale), diag_len)
    return spdiagm(eqn_size, var_size, 0 => vals)
end

"""
    Build matrix block for expression acting on variable.
    Following expression_matrices pattern.
    """
function build_expression_matrix_block(expr, var, eqn_size::Int, var_size::Int)
    
    if isa(expr, TimeDerivative) && _operand_matches_variable(expr.operand, var)
        # Time derivative of this variable -> identity block
        return _identity_block(eqn_size, var_size)

    elseif isa(expr, Laplacian) && _operand_matches_variable(expr.operand, var)
        # Laplacian: ∇² = Σ_i ∂²/∂x_i²
        # In spectral space for Fourier bases: Δ̂ = -|k|² (diagonal)
        # For Chebyshev/Legendre: use second derivative matrix D²
        # Here we return a diagonal approximation using -|k|² scaling
        # The actual matrix construction happens in subsystems.jl via expression_matrices
        return _identity_block(eqn_size, var_size; scale=-1.0)

    elseif isa(expr, Union{Gradient, Divergence, Differentiate}) && _operand_matches_variable(expr.operand, var)
        # First-order spatial derivatives
        # Gradient/Differentiate: ∂/∂x_i -> ik_i in Fourier, D matrix in Chebyshev
        # Divergence: ∇·v = Σ_i ∂v_i/∂x_i
        # Returns identity matrix here as the marker for variable participation.
        # Actual spectral differentiation matrices with basis-specific coefficients
        # are constructed in operators.jl and subsystems.jl during system assembly.
        return _identity_block(eqn_size, var_size)

    elseif _operand_matches_variable(expr, var)
        # Direct variable reference -> identity
        return _identity_block(eqn_size, var_size)
        
    elseif isa(expr, AddOperator)
        # Sum of operators
        left_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        right_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        return left_block + right_block
        
    elseif isa(expr, SubtractOperator)
        # Difference of operators
        left_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
        right_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
        return left_block - right_block
        
    elseif isa(expr, MultiplyOperator)
        # Constant multiplication (either side)
        if isa(expr.right, ConstantOperator) || isa(expr.right, Number)
            coeff = coerce_constant_value(expr.right)
            base_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
            return ComplexF64(coeff) * base_block
        elseif isa(expr.left, ConstantOperator) || isa(expr.left, Number)
            coeff = coerce_constant_value(expr.left)
            base_block = build_expression_matrix_block(expr.right, var, eqn_size, var_size)
            return ComplexF64(coeff) * base_block
        else
            @debug "Non-constant multiplication in matrix block: $(typeof(expr.left)) * $(typeof(expr.right))"
            return _zero_block(eqn_size, var_size)
        end
        
    elseif isa(expr, DivideOperator)
        # Constant division
        if isa(expr.right, ConstantOperator) || isa(expr.right, Number)
            denom = coerce_constant_value(expr.right)
            base_block = build_expression_matrix_block(expr.left, var, eqn_size, var_size)
            return (ComplexF64(1) / ComplexF64(denom)) * base_block
        else
            @debug "Non-constant division in matrix block: $(typeof(expr.right))"
            return _zero_block(eqn_size, var_size)
        end
        
    elseif isa(expr, NegateOperator)
        return -build_expression_matrix_block(expr.operand, var, eqn_size, var_size)
        
    elseif isa(expr, ConstantOperator)
        # Constant expression -> zero block (constants don't depend on variables)
        return _zero_block(eqn_size, var_size)
        
    elseif isa(expr, ZeroOperator)
        # Zero expression -> zero block
        return _zero_block(eqn_size, var_size)

    elseif isa(expr, Interpolate) && _operand_matches_variable(expr.operand, var)
        # BC interpolation constraint: field evaluated at boundary
        # Return identity block to mark variable participation in this BC equation
        return _identity_block(eqn_size, var_size)

    else
        # Unknown expression -> zero block
        @debug "Unknown expression type for matrix block: $(typeof(expr))"
        return _zero_block(eqn_size, var_size)
    end
end

"""Build forcing vector from RHS terms"""
function build_forcing_vector(problem::Problem, eqn_sizes::Vector{Int}, total_size::Int)
    
    F_vector = zeros(ComplexF64, total_size)
    
    i0 = 0
    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        eqn_size = eqn_sizes[eq_idx]
        if eqn_size > 0
            rhs_expr = get(eq_data, "F", ZeroOperator())
            
            # Evaluate RHS expression to get forcing values
            if isa(rhs_expr, ConstantOperator)
                F_vector[i0+1:i0+eqn_size] .= rhs_expr.value
            elseif isa(rhs_expr, ZeroOperator)
                F_vector[i0+1:i0+eqn_size] .= 0.0
            else
                # Complex RHS expressions would need proper evaluation
                @debug "Complex RHS expression not fully supported: $(typeof(rhs_expr))"
                F_vector[i0+1:i0+eqn_size] .= 0.0
            end
        end
        i0 += eqn_size
    end
    
    return F_vector
end

# Legacy functions (kept for compatibility)

"""
    Process LHS operator and extract contributions to system matrices.
    Following pattern where time derivatives go to M_matrix,
    spatial operators go to L_matrix.
    """
function process_lhs_operator!(L_matrix::Matrix, M_matrix::Matrix, lhs_op, eq_idx::Int, variables::Vector)
    
    if isa(lhs_op, TimeDerivative)
        # Time derivative terms go to mass matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            M_matrix[eq_idx, var_idx] = 1.0
        else
            @debug "Unknown variable in time derivative"
        end
        
    elseif isa(lhs_op, Union{Laplacian, Gradient, Divergence, Differentiate})
        # Spatial operators go to linear operator matrix
        var_idx = find_variable_index(lhs_op.operand, variables)
        if var_idx !== nothing
            # Store operator type marker - actual spectral matrix coefficients
            # are computed during subproblem matrix assembly based on basis type
            if isa(lhs_op, Laplacian)
                L_matrix[eq_idx, var_idx] = -1.0  # Typical Laplacian sign
            else
                L_matrix[eq_idx, var_idx] = 1.0
            end
        else
            @debug "Unknown variable in spatial operator"
        end
        
    elseif isa(lhs_op, AddOperator)
        # Recursively process addition terms
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.right, eq_idx, variables)
        
    elseif isa(lhs_op, SubtractOperator)
        # Process left term normally, right term with negative sign
        process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        # Would need to negate contributions from right side
        # This requires more sophisticated matrix coefficient tracking
        @debug "Subtraction operator needs more sophisticated handling"
        
    elseif isa(lhs_op, MultiplyOperator)
        # Handle coefficient multiplication
        if isa(lhs_op.right, ConstantOperator)
            coeff = lhs_op.right.value
            # Apply coefficient to left operand contributions
            # This would require modifying matrix entries by coefficient
            @debug "Coefficient multiplication needs coefficient tracking: $coeff"
            process_lhs_operator!(L_matrix, M_matrix, lhs_op.left, eq_idx, variables)
        else
            @debug "General multiplication not yet supported"
        end
        
    elseif hasfield(typeof(lhs_op), :name)
        # Direct variable reference
        var_idx = find_variable_index(lhs_op, variables)
        if var_idx !== nothing
            L_matrix[eq_idx, var_idx] = 1.0
        end
        
    elseif isa(lhs_op, ZeroOperator)
        # Zero contribution
        nothing
        
    elseif isa(lhs_op, ConstantOperator)
        # Constant terms shouldn't appear in LHS typically
        @debug "Constant term in LHS: $(lhs_op.value)"
        
    else
        @debug "Unhandled LHS operator type: $(typeof(lhs_op))"
    end
end

"""
    Process RHS operator and extract contributions to forcing vector.
    Following pattern where RHS represents known terms/forcing.

    Recursively evaluates composite operators (Add, Subtract, Multiply) to
    compute the scalar forcing value for each equation.
    """
function process_rhs_operator!(F_vector::Vector, rhs_op, eq_idx::Int, variables::Vector)

    if isa(rhs_op, ConstantOperator)
        # Constant forcing term
        F_vector[eq_idx] = rhs_op.value

    elseif isa(rhs_op, ZeroOperator)
        # Zero RHS (homogeneous equation)
        F_vector[eq_idx] = 0.0

    elseif isa(rhs_op, AddOperator)
        # Sum of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value + right_value

    elseif isa(rhs_op, SubtractOperator)
        # Difference of RHS terms: recursively evaluate left and right
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        right_value = evaluate_rhs_scalar(rhs_op.right, variables)
        F_vector[eq_idx] = left_value - right_value

    elseif isa(rhs_op, MultiplyOperator)
        # Product of RHS terms
        left_value = evaluate_rhs_scalar(rhs_op.left, variables)
        if isa(rhs_op.right, Number)
            F_vector[eq_idx] = left_value * rhs_op.right
        else
            right_value = evaluate_rhs_scalar(rhs_op.right, variables)
            F_vector[eq_idx] = left_value * right_value
        end

    elseif isa(rhs_op, Number)
        # Direct numeric value
        F_vector[eq_idx] = Float64(real(rhs_op))

    elseif isa(rhs_op, String) && (rhs_op == "0" || rhs_op == "zero")
        # String representation of zero
        F_vector[eq_idx] = 0.0

    else
        @debug "Unhandled RHS operator type: $(typeof(rhs_op)), using zero"
        F_vector[eq_idx] = 0.0
    end
end

"""
    evaluate_rhs_scalar(op, variables::Vector) -> Float64

Recursively evaluate an operator expression to obtain a scalar value.
Used for extracting forcing terms from composite RHS expressions.

Returns the scalar value of the expression, or 0.0 for unhandled types.
"""
function evaluate_rhs_scalar(op, variables::Vector)
    if isa(op, ConstantOperator)
        return Float64(op.value)

    elseif isa(op, ZeroOperator)
        return 0.0

    elseif isa(op, Number)
        return Float64(real(op))

    elseif isa(op, AddOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val + right_val

    elseif isa(op, SubtractOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        right_val = evaluate_rhs_scalar(op.right, variables)
        return left_val - right_val

    elseif isa(op, MultiplyOperator)
        left_val = evaluate_rhs_scalar(op.left, variables)
        if isa(op.right, Number)
            return left_val * op.right
        else
            right_val = evaluate_rhs_scalar(op.right, variables)
            return left_val * right_val
        end

    elseif isa(op, ScalarField)
        # For field-valued RHS, we need to evaluate at specific points
        # For now, return the mean value if available
        if get_grid_data(op) !== nothing && length(get_grid_data(op)) > 0
            return real(sum(get_grid_data(op)) / length(get_grid_data(op)))
        elseif get_coeff_data(op) !== nothing && length(get_coeff_data(op)) > 0
            # First coefficient is often the mean for spectral methods
            # Use GPU-safe indexing to avoid scalar indexing on GPU arrays
            if is_gpu_array(get_coeff_data(op))
                # Copy first element to CPU to avoid GPU scalar indexing
                first_coef = Array(@view get_coeff_data(op)[1:1])[1]
            else
                first_coef = get_coeff_data(op)[1]
            end
            return real(first_coef)
        else
            return 0.0
        end

    elseif isa(op, String)
        # Try to parse as number
        if op == "0" || op == "zero"
            return 0.0
        end
        try
            return parse(Float64, op)
        catch
            return 0.0
        end

    else
        @debug "evaluate_rhs_scalar: unhandled type $(typeof(op)), returning 0.0"
        return 0.0
    end
end

"""Find index of variable in problem variable list"""
function find_variable_index(operand, variables::Vector)
    
    # Handle direct variable reference
    for (i, var) in enumerate(variables)
        if operand === var
            return i
        end
    end
    
    # Handle by name if operand has name field
    if hasfield(typeof(operand), :name)
        for (i, var) in enumerate(variables)
            if hasfield(typeof(var), :name) && operand.name == var.name
                return i
            end
        end
    end
    
    return nothing
end

