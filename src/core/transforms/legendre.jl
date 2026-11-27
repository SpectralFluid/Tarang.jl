"""
    Legendre and Jacobi matrix transforms.

The base matrices are built during planning. Matrices for scaled grids use
the field's collocation nodes while retaining the basis coefficient count.
The matrix applications run on CPU and write through the transform stage buffers.
"""

const MatrixTransform = Union{LegendreTransform, JacobiTransform}

function _scaled_matrix_transform_matrices(basis::Legendre, grid_size::Int, coeff_size::Int)
    nodes, weights = gauss_jacobi_quadrature(grid_size, 0.0, 0.0)
    return _legendre_transform_matrices(nodes, weights, coeff_size)
end

function _scaled_matrix_transform_matrices(basis::JacobiBasis, grid_size::Int, coeff_size::Int)
    # The rational scale makes ceil(basis_size * scale) exactly grid_size.
    native_nodes = _native_grid(basis, grid_size // basis.meta.size)
    nodes = _basis_grid_in_problem_coords(basis, native_nodes)
    return _jacobi_transform_matrices(basis, nodes, coeff_size)
end

function _matrix_transform_matrices(transform::MatrixTransform, grid_size::Int)
    if grid_size == transform.grid_size
        return (transform.forward_matrix, transform.backward_matrix)
    end
    basis = transform.basis
    coeff_size = transform.coeff_size
    return _get_basis_cache!(basis, (:scaled_matrix_transform, grid_size, coeff_size)) do
        _scaled_matrix_transform_matrices(basis, grid_size, coeff_size)
    end
end

function _apply_matrix_transform(data::AbstractArray, axis::Int, mat)
    # Legendre transforms are matrix-backed and CPU-only today. GPU input is
    # REJECTED (loud error) by `_execute_on_cpu` — CPU staging fallback is
    # disabled; a GPU-arch Legendre field already errors at the core dispatcher
    # ("No on-device forward transform supports field ...") before reaching here.
    return _execute_on_cpu(data) do host_data
        if axis > ndims(host_data)
            return host_data
        end

        if mat === nothing
            return host_data
        end

        output_size = size(mat, 1)
        out_shape = ntuple(i -> i == axis ? output_size : size(host_data, i), ndims(host_data))
        real_type = real(eltype(host_data))
        out_eltype = eltype(host_data) <: Complex ? eltype(host_data) : real_type
        out = zeros(out_eltype, out_shape)

        other_dims = Tuple(filter(i -> i != axis, 1:ndims(host_data)))

        # Reuse global work buffers for each one-dimensional slice so repeated
        # multi-dimensional transforms do not allocate scratch vectors.
        wm = get_global_workspace()
        temp_real = get_workspace!(wm, real_type, (output_size,))
        temp_imag = eltype(host_data) <: Complex ? get_workspace!(wm, real_type, (output_size,)) : nothing

        if isempty(other_dims)
            # One-dimensional case: apply the quadrature matrix directly.
            mul!(temp_real, mat, real(host_data))
            if temp_imag === nothing
                copyto!(out, temp_real)
            else
                mul!(temp_imag, mat, imag(host_data))
                out .= complex.(temp_real, temp_imag)
            end
        else
            # Multi-dimensional case: transform along the selected axis by
            # applying the same matrix to each orthogonal slice.
            for (slice_in, slice_out) in zip(eachslice(host_data, dims=other_dims), eachslice(out, dims=other_dims))
                mul!(temp_real, mat, real(slice_in))
                if temp_imag === nothing
                    slice_out .= temp_real
                else
                    mul!(temp_imag, mat, imag(slice_in))
                    slice_out .= complex.(temp_real, temp_imag)
                end
            end
        end

        release_workspace!(wm, temp_real)
        if temp_imag !== nothing
            release_workspace!(wm, temp_imag)
        end

        return out
    end
end

function _legendre_forward(data::AbstractArray, transform::MatrixTransform)
    forward, _ = _matrix_transform_matrices(transform, size(data, transform.axis))
    return _apply_matrix_transform(data, transform.axis, forward)
end

function _legendre_backward(data::AbstractArray, transform::MatrixTransform,
                            grid_size::Int=transform.grid_size)
    _, backward = _matrix_transform_matrices(transform, grid_size)
    return _apply_matrix_transform(data, transform.axis, backward)
end

_apply_forward(current, t::MatrixTransform) = _legendre_forward(current, t)
_apply_backward(current, t::MatrixTransform) = _legendre_backward(current, t)

function _forward_output_spec(in::AbstractArray, transform::MatrixTransform)
    out_shape = _replace_axis_shape(size(in), transform.axis, transform.coeff_size)
    return (out_shape, eltype(in))
end

function _bwd_rfft_target(field::ScalarField, transform::MatrixTransform)
    scale = _axis_scale(field.scales, transform.axis)
    return ceil(Int, scale * transform.grid_size)
end

function _backward_output_spec(in::AbstractArray, transform::MatrixTransform,
                               grid_size::Int=transform.grid_size)
    out_shape = _replace_axis_shape(size(in), transform.axis, grid_size)
    return (out_shape, eltype(in))
end

function _apply_forward!(out::AbstractArray, in::AbstractArray, transform::MatrixTransform)
    copyto!(out, _legendre_forward(in, transform))
    return out
end

function _apply_backward!(out::AbstractArray, in::AbstractArray, transform::MatrixTransform,
                          grid_size::Int=transform.grid_size)
    copyto!(out, _legendre_backward(in, transform, grid_size))
    return out
end
