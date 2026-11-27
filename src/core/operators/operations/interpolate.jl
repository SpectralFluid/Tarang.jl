# Interpolation evaluation plus Clenshaw-based polynomial reconstruction helpers.

# ============================================================================
# Interpolation Evaluation
# ============================================================================

"""
    evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)

Evaluate interpolation operator at a specific position along a coordinate.
Following operators Interpolate implementation.

For Fourier bases: uses spectral interpolation (sum of modes)
For Jacobi bases: contracts coefficients with the polynomial values at the requested point.
"""
function evaluate_interpolate(interp_op::Interpolate, layout::Symbol=:g)
    operand = interp_op.operand
    coord = interp_op.coord
    position = interp_op.position

    # Accept composite expressions: reduce the operand tree to a scalar field first.
    if !isa(operand, ScalarField)
        operand = evaluate(operand)
    end
    if !isa(operand, ScalarField)
        throw(ArgumentError("interpolate: operand must reduce to a scalar field, got $(typeof(operand))"))
    end

    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        throw(ArgumentError("Coordinate $(coord.name) not found in field bases"))
    end

    basis = operand.bases[basis_index]

    # Scaled fields can still carry coefficients from their previous grid
    # resolution (set_scales! preserves coefficients when :c is current).
    # Normalize through the current grid so all Fourier axes below use the
    # actual FFT lengths, including the parity of an RFFT axis.
    scaled = operand.scales !== nothing && any(!isone, operand.scales)
    if scaled
        ensure_layout!(operand, :g)
    end
    grid_shape = get_scaled_shape(operand)

    # Work in coefficient space for spectral interpolation.
    ensure_layout!(operand, :c)

    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        return interpolate_fourier(operand, basis, basis_index, position, layout; grid_shape)
    elseif isa(basis, JacobiBasis)
        return interpolate_jacobi(operand, basis, basis_index, position, layout)
    else
        throw(ArgumentError("Interpolation not implemented for basis type $(typeof(basis))"))
    end
end

"""
    interpolate_fourier(field, basis, axis, position, layout)

Interpolate Fourier field at a specific position using spectral reconstruction.
f(x) = Sigma c_k exp(i k x) for ComplexFourier
f(x) = a_0 + Sigma (a_k cos(kx) + b_k sin(kx)) for RealFourier
"""
function interpolate_fourier(field::ScalarField, basis::FourierBasis, axis::Int, position::Real, layout::Symbol;
                             grid_shape::Tuple=get_scaled_shape(field))
    N = grid_shape[axis]
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    x0 = basis.meta.bounds[1]

    # Normalize position to [0, L)
    x = mod(position - x0, L)

    bundle = _field_transform_bundle(field)
    uses_rfft = _axis_uses_rfft(bundle, axis)
    weights = uses_rfft ? _fourier_interp_weights(basis, N, L, x) :
                          _fourier_full_weights(N, L, x; real_field=field.dtype <: Real)
    data = _contract_slice(field, axis, weights)
    length(field.bases) == 1 && return field.dtype <: Real ? real(data) : data

    remaining = Tuple(d for d in eachindex(field.bases) if d != axis)
    bases = map(d -> field.bases[d], remaining)
    if uses_rfft
        # The doubled positive modes stand for conjugate pairs across the
        # remaining Fourier axes. Polynomial degree is unchanged by conjugation.
        spectrum = similar(data)
        reflect = map(b -> b isa FourierBasis, bases)
        launch!(architecture(data), _real_slice_spectrum!, spectrum, data, reflect; ndrange=size(data))
        data = spectrum
    end
    layout === :c && return data

    # Preserve the parent's storage convention. After collapsing its half
    # spectrum, surviving Fourier axes still contain both frequency signs.
    half_axis = findfirst(d -> _axis_uses_rfft(bundle, d), remaining)
    if all(b -> b isa FourierBasis, bases)
        if half_axis === nothing
            grid = ifft(data)
            return field.dtype <: Real ? real.(grid) : grid
        end
        full_axes = Tuple(d for d in eachindex(bases) if d != half_axis)
        tmp = isempty(full_axes) ? data : ifft(data, full_axes)
        return irfft(tmp, grid_shape[remaining[half_axis]], half_axis)
    end
    dtype = any(b -> b isa FourierBasis, bases) && half_axis === nothing ? complex(field.dtype) : field.dtype
    reduced = _field_from_slice(field, remaining, data, dtype)
    grid = grid_data!(reduced)
    return field.dtype <: Real && eltype(grid) <: Complex ? real.(grid) : grid
end

@kernel function _real_slice_spectrum!(out, @Const(data), reflect)
    I = @index(Global, NTuple)
    J = ntuple(d -> reflect[d] && I[d] != 1 ? size(data, d) + 2 - I[d] : I[d], Val(length(I)))
    @inbounds out[I...] = (data[I...] + conj(data[J...])) / 2
end

# Contract the owned coefficients before communicating. Only the reduced
# slice is replicated, in logical axis order, on every participating rank.
function _contract_slice(field, axis, weights)
    coeffs = get_coeff_data(field)
    shape, ranges, order = _coefficient_geometry(field, coeffs)
    local_data = get_local_data(coeffs)
    physical_axis = findfirst(==(axis), order)
    w = on_architecture(architecture(local_data), convert(Vector{eltype(coeffs)}, weights[ranges[axis]]))
    if ndims(coeffs) == 1
        value = sum(local_data .* w)
        return field.dist.size == 1 ? value : MPI.Allreduce(value, +, field.dist.comm)
    end

    wshape = ntuple(d -> d == physical_axis ? length(w) : 1, ndims(coeffs))
    piece = dropdims(sum(local_data .* reshape(w, wshape); dims=physical_axis); dims=physical_axis)
    field.dist.size == 1 && return piece
    remaining = Tuple(d for d in eachindex(shape) if d != axis)
    physical = filter(!=(axis), order)
    logical_order = map(d -> findfirst(==(d), physical), remaining)
    # Stage only the slice for MPI implementations without GPU-aware collectives.
    result = zeros(eltype(piece), map(d -> shape[d], remaining))
    host_piece = on_architecture(CPU(), piece)
    view(result, map(d -> ranges[d], remaining)...) .= PermutedDimsArray(host_piece, logical_order)
    MPI.Allreduce!(result, +, field.dist.comm)
    return on_architecture(field.dist.architecture, result)
end

function _field_from_slice(field, remaining, data, dtype)
    dist = field.dist.size == 1 ? field.dist : _serial_replica_distributor(field.dist, dtype)
    bases = map(d -> field.bases[d], remaining)
    result = ScalarField(dist, "interp_$(field.name)", bases, dtype)
    scales = field.scales === nothing ? ntuple(_ -> 1.0, length(bases)) : map(d -> field.scales[d], remaining)
    preset_scales!(result, scales)
    # When the last Fourier axis is removed, the polynomial series is real.
    coeff_type = eltype(get_coeff_data(result))
    coeffs = coeff_type <: Real && eltype(data) <: Complex ? real.(data) : data
    set_coeff_data!(result, coeffs)
    result.current_layout = :c
    return result
end

"""
    _fourier_interp_weights(basis, N, L, x)

Build spectral interpolation weight vector for Fourier basis at position x.
"""
function _fourier_interp_weights(basis::FourierBasis, N::Int, L::Real, x::Real)
    k0 = 2π / L
    if isa(basis, RealFourier)
        # Complex weights for the unnormalized rfft half-spectrum (length N÷2+1).
        # Applied to the complex coefficient array; real() is taken afterwards.
        # Encodes the 1/N normalization and the factor-2 on interior modes.
        half = N ÷ 2 + 1
        weights = zeros(ComplexF64, half)
        weights[1] = 1.0 / N  # DC
        nyquist = iseven(N)
        for i in 2:half
            k = i - 1
            weights[i] = nyquist && i == half ? cos(k0 * k * x) / N : (2 / N) * cis(k0 * k * x)
        end
        return weights
    else  # ComplexFourier
        return _fourier_full_weights(N, L, x)
    end
end

"""
    _fourier_full_weights(N, L, x; real_field=false)

Interpolation weights for a FULL (unnormalized, FFT-ordered) complex Fourier
spectrum of length `N`: `w_k = e^{i k k0 x} / N` with `k` in FFT order
(`0,…,N÷2-1,-N÷2,…,-1` for even N). Used for ComplexFourier axes and for RealFourier axes
stored as a full FFT (any RealFourier axis other than the operand's first).
For real fields on even grids, the Nyquist bin uses the cosine interpolant.
"""
function _fourier_full_weights(N::Int, L::Real, x::Real; real_field::Bool=false)
    k0 = 2π / L
    weights = zeros(ComplexF64, N)
    for i in 1:N
        k = i <= (N + 1) ÷ 2 ? i - 1 : i - N - 1
        weights[i] = cis(k0 * k * x) / N
    end
    if real_field && iseven(N)
        weights[N ÷ 2 + 1] = cospi(N * x / L) / N
    end
    return weights
end

"""
    interpolate_jacobi(field, basis, axis, position, layout)

Evaluate the polynomial expansion along one axis, preserving the other axes.
"""
function interpolate_jacobi(field::ScalarField, basis::JacobiBasis, axis::Int, position::Real, layout::Symbol)
    a, b = basis.meta.bounds
    modes = 0:(basis.meta.size - 1)
    x = clamp(2 * (position - a) / (b - a) - 1, -1, 1)
    weights = if basis isa ChebyshevT
        cos.(modes .* acos(x))
    elseif basis isa Legendre
        # Use the same orthonormal polynomials as the inverse transform.
        polynomials = vec(build_legendre_polynomials(length(modes), [Float64(x)]))
        polynomials .*= sqrt.((2 .* modes .+ 1) ./ 2)
    else
        vec(evaluate_basis(basis, [clamp(position, a, b)], modes))
    end
    data = _contract_slice(field, axis, weights)
    length(field.bases) == 1 && return data
    remaining = Tuple(d for d in eachindex(field.bases) if d != axis)
    result = _field_from_slice(field, remaining, data, field.dtype)
    ensure_layout!(result, layout)
    return result
end

# ============================================================================
# Clenshaw Algorithm Implementations
# ============================================================================

"""
    clenshaw_chebyshev_t(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev T polynomial sum.
T_n(x) satisfies: T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
"""
function clenshaw_chebyshev_t(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence: b_k = c_k + 2x b_{k+1} - b_{k+2}
    b_k2 = 0.0  # b_{n+1}
    b_k1 = 0.0  # b_{n}

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step: f(x) = c_0 + x*b_1 - b_2
    return coeffs[1] + x * b_k1 - b_k2
end

"""
    clenshaw_chebyshev_u(coeffs, x)

Clenshaw algorithm for evaluating Chebyshev U polynomial sum.
U_n(x) satisfies: U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
"""
function clenshaw_chebyshev_u(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Same recurrence as T_n
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        b_k = coeffs[k] + 2x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # For U_n: f(x) = c_0 * U_0(x) + b_1 * U_1(x) - b_2 * U_0(x)
    # where U_0(x) = 1, U_1(x) = 2x
    return coeffs[1] + 2x * b_k1 - b_k2
end

"""
    clenshaw_legendre(coeffs, x)

Clenshaw algorithm for evaluating Legendre polynomial sum.
P_n(x) satisfies: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
"""
function clenshaw_legendre(coeffs::AbstractVector, x::Real)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Backward recurrence for Legendre
    b_k2 = 0.0
    b_k1 = 0.0

    for k in n:-1:2
        # Clenshaw backward recurrence: b_k = c_k + α_k * b_{k+1} - C_{k+1} * b_{k+2}
        # where α_k = (2k+1)/(k+1) * x and C_k = k/(k+1) for Legendre
        deg = k - 1  # 0-indexed polynomial degree
        alpha = (2*deg + 1) / (deg + 1) * x
        beta = (deg + 1) / (deg + 2)  # C_{deg+1}, NOT C_{deg}
        b_k = coeffs[k] + alpha * b_k1 - beta * b_k2
        b_k2 = b_k1
        b_k1 = b_k
    end

    # Final step
    return coeffs[1] + x * b_k1 - 0.5 * b_k2
end

"""
    clenshaw_jacobi(coeffs, x, a, b)

Clenshaw algorithm for evaluating general Jacobi polynomial sum.
"""
function clenshaw_jacobi(coeffs::AbstractVector, x::Real, a::Float64, b::Float64)
    n = length(coeffs)
    if n == 0
        return 0.0
    elseif n == 1
        return coeffs[1]
    end

    # Use direct evaluation for general Jacobi (less efficient but correct)
    result = 0.0
    for k in 1:n
        result += coeffs[k] * jacobi_polynomial(k-1, a, b, x)
    end
    return result
end

"""
    jacobi_polynomial(n, a, b, x)

Evaluate Jacobi polynomial P_n^{(a,b)}(x) using three-term recurrence.
"""
function jacobi_polynomial(n::Int, a::Float64, b::Float64, x::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return 0.5 * (a - b + (a + b + 2) * x)
    end

    p_km2 = 1.0
    p_km1 = 0.5 * (a - b + (a + b + 2) * x)

    for k in 2:n
        # Three-term recurrence for Jacobi polynomials
        k_f = Float64(k)
        a1 = 2 * k_f * (k_f + a + b) * (2*k_f + a + b - 2)
        a2 = (2*k_f + a + b - 1) * (a^2 - b^2)
        a3 = (2*k_f + a + b - 2) * (2*k_f + a + b - 1) * (2*k_f + a + b)
        a4 = 2 * (k_f + a - 1) * (k_f + b - 1) * (2*k_f + a + b)

        p_k = ((a2 + a3 * x) * p_km1 - a4 * p_km2) / a1
        p_km2 = p_km1
        p_km1 = p_k
    end

    return p_km1
end
