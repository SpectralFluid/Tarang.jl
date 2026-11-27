"""
    Fractional Laplacian operators

This file contains fractional Laplacian evaluation, wavenumber-grid helpers,
and the matrix-interface methods used by implicit solvers.
"""

# ============================================================================
# Fractional Laplacian Evaluation
# ============================================================================

"""
    evaluate_fractional_laplacian(frac_lap::FractionalLaplacian, layout::Symbol=:g)

Evaluate fractional Laplacian operator: (-Delta)^alpha

In spectral space, this multiplies each Fourier coefficient by |k|^(2*alpha),
where k = sqrt(k1^2 + k2^2 + ...) is the wavenumber magnitude.

For alpha > 0: High-order dissipation (smoothing)
For alpha < 0: Inverse operation (integration/smoothing)
For alpha = 1: Negative Laplacian, (-Δ)^1 = -Δ (note: opposite sign from Laplacian())
For alpha = 1/2: Square root Laplacian (SQG dissipation)
For alpha = -1/2: Inverse square root (SQG streamfunction from buoyancy)

Note: For alpha < 0, the k=0 mode is handled specially to avoid division by zero.
The k=0 mode is set to zero (removes mean).
"""
function evaluate_fractional_laplacian(frac_lap::FractionalLaplacian, layout::Symbol=:g)
    operand = frac_lap.operand
    alpha = frac_lap.α

    if isa(operand, ScalarField)
        return evaluate_scalar_fractional_laplacian(operand, alpha, layout)
    elseif isa(operand, VectorField)
        return evaluate_vector_fractional_laplacian(operand, alpha, layout)
    else
        throw(ArgumentError("Fractional Laplacian not implemented for $(typeof(operand))"))
    end
end

function evaluate_scalar_fractional_laplacian(operand::ScalarField, alpha::Float64, layout::Symbol)
    # Fractional Laplacian is only defined for Fourier bases (wavenumber-based).
    # For Chebyshev/Legendre bases, the wavenumber grid would be all zeros,
    # producing a silently wrong zero result.
    # The wavenumber-based fractional Laplacian can only represent Fourier axes. A
    # non-Fourier (Chebyshev/Legendre) axis contributes 0 to |k|² and would be SILENTLY
    # dropped, producing a wrong operator on a mixed domain (e.g. returning only the
    # horizontal part of -Δ). Require every axis to be Fourier rather than just one.
    all_fourier = all(isa(b, Union{RealFourier, ComplexFourier}) for b in operand.bases)
    if !all_fourier
        throw(ArgumentError("Fractional Laplacian requires ALL axes to be Fourier bases. " *
            "On a mixed Fourier/Chebyshev(Legendre) domain the non-Fourier direction would be " *
            "silently dropped; use a matrix-based approach for those axes instead."))
    end

    # Create result field
    alpha_str = alpha >= 0 ? "$(alpha)" : "m$(abs(alpha))"
    result = _allocate_field_arithmetic_result(operand, operand.dtype, "fraclap$(alpha_str)_$(operand.name)")

    # Work on a copy to avoid mutating the caller's field layout
    work = copy(operand)
    _has_nonunit_arithmetic_scale(work) && ensure_layout!(work, :g)
    ensure_layout!(work, :c)
    ensure_layout!(result, :c)

    # Get wavenumber grids for each Fourier basis
    k_squared_total = compute_wavenumber_squared_grid(work)

    alpha_native = eltype(k_squared_total)(alpha)
    get_local_data(get_coeff_data(result)) .= get_local_data(get_coeff_data(work)) .*
        _fractional_multiplier.(get_local_data(k_squared_total), alpha_native)

    # Transform to requested layout if needed
    if layout == :g
        ensure_layout!(result, :g)
    end

    return result
end

function evaluate_vector_fractional_laplacian(operand::VectorField, alpha::Float64, layout::Symbol)
    alpha_str = alpha >= 0 ? "$(alpha)" : "m$(abs(alpha))"
    result = VectorField(operand.dist, operand.coordsys, "fraclap$(alpha_str)_$(operand.name)",
                        operand.bases, operand.dtype)

    for (i, comp) in enumerate(operand.components)
        result.components[i] = evaluate_scalar_fractional_laplacian(comp, alpha, layout)
    end

    return result
end

"""
    compute_wavenumber_squared_grid(field::ScalarField)

Compute |k|^2 = k1^2 + k2^2 + ... for each point in spectral space.
Supports both CPU and GPU arrays.

Returns an array with the same shape as get_coeff_data(field) containing
the squared wavenumber magnitude at each spectral coefficient location.
"""
function compute_wavenumber_squared_grid(field::ScalarField)
    cd = get_coeff_data(field)
    T = real(eltype(cd))
    k_squared = fill!(similar(cd, T), zero(T))
    isempty(field.bases) && return k_squared
    shape, ranges, order = _coefficient_geometry(field, cd)
    bundle = _field_transform_bundle(field)
    for (axis, basis) in enumerate(field.bases)
        basis isa FourierBasis || continue
        half = _axis_uses_rfft(bundle, axis)
        n = shape[axis]
        k0 = 2π / (basis.meta.bounds[2] - basis.meta.bounds[1])
        # A full FFT wraps its upper bins to negative frequencies. A half
        # spectrum is nonnegative for both even and odd physical grid sizes.
        k = [k0 * (half || i <= cld(n, 2) ? i - 1 : i - 1 - n) for i in ranges[axis]]
        add_wavenumber_squared_contribution!(get_local_data(k_squared), k,
                                            findfirst(==(axis), order), length(shape))
    end
    return k_squared
end

# Only the constant Fourier mode is singular for an inverse power.
@inline _fractional_multiplier(k2, alpha) = iszero(k2) && alpha < 0 ? zero(k2) : k2^alpha

"""
Add k^2 contribution from one axis to the total wavenumber grid.
Works with both CPU and GPU arrays.
"""
function add_wavenumber_squared_contribution!(k_squared::AbstractArray, k_axis_cpu::AbstractVector{<:Real}, axis::Int, ndims::Int)
    # Create shape for broadcasting
    shape = ones(Int, ndims)
    shape[axis] = length(k_axis_cpu)

    # Move k_axis to the same device as k_squared if needed
    if is_gpu_array(k_squared)
        k_axis = copy_to_device(convert(AbstractVector{eltype(k_squared)}, k_axis_cpu), k_squared)
    else
        k_axis = k_axis_cpu
    end

    # Reshape k_axis for broadcasting
    k_reshaped = reshape(k_axis, Tuple(shape))

    # Add k^2 contribution
    k_squared .+= k_reshaped.^2
end

# ============================================================================
# Fractional Laplacian - Matrix methods
# ============================================================================

"""
    matrix_dependence(op::FractionalLaplacian, vars...)

Determine which variables this fractional Laplacian operator depends on.
"""
function matrix_dependence(op::FractionalLaplacian, vars...)
    result = falses(length(vars))
    for (i, var) in enumerate(vars)
        if op.operand === var || (hasfield(typeof(op.operand), :name) &&
                                   hasfield(typeof(var), :name) &&
                                   op.operand.name == var.name)
            result[i] = true
        end
    end
    return result
end

"""
    matrix_coupling(op::FractionalLaplacian, vars...)

Fractional Laplacian only couples a variable to itself.
"""
function matrix_coupling(op::FractionalLaplacian, vars...)
    return matrix_dependence(op, vars...)
end

"""
    subproblem_matrix(op::FractionalLaplacian, subproblem)

Build the sparse matrix representation of fractional Laplacian for implicit solvers.
"""
function subproblem_matrix(op::FractionalLaplacian, subproblem)
    operand = op.operand
    alpha = op.α

    if isa(operand, ScalarField)
        n = prod(size(get_coeff_data(operand)))
    else
        throw(ArgumentError("subproblem_matrix for FractionalLaplacian only implemented for ScalarField"))
    end

    # Compute wavenumber squared grid
    k_squared = compute_wavenumber_squared_grid(operand)

    # Convert to CPU if on GPU
    if is_gpu_array(k_squared)
        k_squared = Array(k_squared)
    end

    k_squared_flat = vec(k_squared)

    diag_entries = _fractional_multiplier.(k_squared_flat, alpha)
    return spdiagm(0 => diag_entries)
end

"""Check conditions for fractional Laplacian."""
function check_conditions(op::FractionalLaplacian)
    operand = op.operand

    if isa(operand, ScalarField)
        if operand isa ScalarField
            layout = operand.current_layout
            if layout == :c
                return get_coeff_data(operand) !== nothing
            elseif layout == :g
                return get_grid_data(operand) !== nothing
            end
        end
    elseif isa(operand, VectorField)
        for comp in operand.components
            if !check_conditions(FractionalLaplacian(comp, op.α))
                return false
            end
        end
    end

    return true
end

"""Enforce conditions for fractional Laplacian."""
function enforce_conditions(op::FractionalLaplacian)
    operand = op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :c)
    elseif isa(operand, VectorField)
        for comp in operand.components
            ensure_layout!(comp, :c)
        end
    end
end

"""Fractional Laplacian is linear."""
is_linear(op::FractionalLaplacian) = true

"""Return effective derivative order: 2*alpha."""
operator_order(op::FractionalLaplacian) = 2 * op.α

# ============================================================================
# TransposeComponents Matrix Methods
# ============================================================================

"""
    _transpose_matrix(op::TransposeComponents)

Build permutation matrix for transposing tensor components.
"""
function _transpose_matrix(op::TransposeComponents)
    operand = op.operand
    i0, i1 = op.indices

    if !isa(operand, TensorField)
        throw(ArgumentError("_transpose_matrix requires TensorField"))
    end

    dim = size(operand.components, 1)
    n = dim * dim

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # For a rank-2 tensor T_{ij}, transposing indices (i0, i1) maps T_{ij} -> T_{ji}.
    # In flat storage: input at i*dim+j goes to output at j*dim+i.
    # This correctly swaps the index POSITIONS (not values independently).
    for i in 0:(dim-1)
        for j in 0:(dim-1)
            input_idx = i * dim + j + 1
            output_idx = j * dim + i + 1
            push!(rows, output_idx)
            push!(cols, input_idx)
            push!(vals, 1.0)
        end
    end

    return sparse(rows, cols, vals, n, n)
end

"""Build operator matrix for TransposeComponents."""
function subproblem_matrix(op::TransposeComponents, subproblem)
    transpose_mat = _transpose_matrix(op)

    operand = op.operand
    coeff_size = 1
    if isa(operand, TensorField) && length(operand.components) > 0
        first_comp = operand.components[1,1]
        if first_comp !== nothing && hasfield(typeof(first_comp), :bases)
            for basis in first_comp.bases
                coeff_size *= basis.meta.size
            end
        end
    end

    eye = sparse(I, coeff_size, coeff_size)

    return kron(transpose_mat, eye)
end

check_conditions(op::TransposeComponents) = true
enforce_conditions(op::TransposeComponents) = nothing

function matrix_dependence(op::TransposeComponents, vars...)
    if hasmethod(matrix_dependence, Tuple{typeof(op.operand), typeof.(vars)...})
        return matrix_dependence(op.operand, vars...)
    end
    return falses(length(vars))
end

function matrix_coupling(op::TransposeComponents, vars...)
    if hasmethod(matrix_coupling, Tuple{typeof(op.operand), typeof.(vars)...})
        return matrix_coupling(op.operand, vars...)
    end
    return falses(length(vars))
end

is_linear(op::TransposeComponents) = true
