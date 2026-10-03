"""
    Field component buffers

This file contains structure-of-arrays helpers for stacking and unstacking
vector and tensor field component data.
"""

# ---------------------------------------------------------------------------
# Vector field component buffers (structure-of-arrays helpers)
# ---------------------------------------------------------------------------

"""
    stack_components(vf::VectorField; layout::Symbol=:g, arch::AbstractArchitecture=vf.dist.architecture, force::Bool=false)

Build (or reuse) a contiguous buffer containing all vector components stacked along the
first dimension. This provides an easy structure-of-arrays view that is convenient for
GPU kernels expecting component-major memory layout. For PencilArray storage, the buffer
contains the local slab for each component and can be created on CPU or GPU (data is
copied from the host to the requested architecture).
"""
function stack_components(vf::VectorField; layout::Symbol=:g,
                           arch::AbstractArchitecture=vf.dist.architecture,
                           force::Bool=false)
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for stack_components"))
    isempty(vf.components) && throw(ArgumentError("VectorField has no components"))

    return _stack_component_buffers(vf, layout, arch, force)
end

"""
    unstack_components!(vf::VectorField, buffer; layout::Union{Symbol,Nothing}=vf.buffer_layout)

Scatter a stacked buffer back into the vector field components.
"""
function unstack_components!(vf::VectorField, buffer::AbstractArray; layout::Union{Symbol,Nothing}=vf.buffer_layout)
    layout === nothing && throw(ArgumentError("Cannot unstack components without a known layout"))
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for unstack_components!"))
    size(buffer, 1) == length(vf.components) || throw(ArgumentError("Component count mismatch: expected $(length(vf.components)), got $(size(buffer, 1))"))

    return _unstack_component_buffers!(vf, buffer, layout)
end

"""
    stack_tensor_components(tf::TensorField; layout::Symbol=:g, arch::AbstractArchitecture=tf.dist.architecture, force::Bool=false)

Stack tensor components (matrix of `ScalarField`s) into a structure-of-arrays buffer of shape
`(dim, dim, ...)` where additional dimensions correspond to the underlying scalar data. For
PencilArray storage, the buffer contains only the local slab and can be created on CPU or GPU
(data is copied from the host when needed).
"""
function stack_tensor_components(tf::TensorField; layout::Symbol=:g,
                                  arch::AbstractArchitecture=tf.dist.architecture,
                                  force::Bool=false)
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for stack_tensor_components"))

    dim = tf.coordsys.dim
    dim == size(tf.components, 1) == size(tf.components, 2) || throw(ArgumentError("TensorField component matrix mismatch"))

    return _stack_component_buffers(tf, layout, arch, force)
end

"""
    unstack_tensor_components!(tf::TensorField, buffer; layout::Union{Symbol,Nothing}=tf.buffer_layout)

Scatter stacked tensor buffer back into component fields.
"""
function unstack_tensor_components!(tf::TensorField, buffer::AbstractArray; layout::Union{Symbol,Nothing}=tf.buffer_layout)
    layout === nothing && throw(ArgumentError("Cannot unstack tensor components without layout information"))
    layout in (:g, :c) || throw(ArgumentError("Unsupported layout $layout for unstack_tensor_components!"))

    dim = tf.coordsys.dim
    size(buffer, 1) == dim && size(buffer, 2) == dim || throw(ArgumentError("Tensor buffer must have leading dimensions ($dim, $dim)"))

    return _unstack_component_buffers!(tf, buffer, layout)
end

# Preserve the tensor traversal order and the leading component dimensions.
_component_buffer_indices(vf::VectorField) = CartesianIndices(vf.components)
function _component_buffer_indices(tf::TensorField)
    dim = tf.coordsys.dim
    return (CartesianIndex(index[2], index[1]) for index in CartesianIndices((dim, dim)))
end
_component_buffer_slice(buffer, index::CartesianIndex{1}) = selectdim(buffer, 1, index[1])
_component_buffer_slice(buffer, index::CartesianIndex{2}) = selectdim(selectdim(buffer, 2, index[2]), 1, index[1])
_component_buffer_data(component, layout) = layout == :g ? get_grid_data(component) : get_coeff_data(component)

function _stack_component_buffers(field, layout, arch, force)
    using_pencils = is_pencil_storage(field)
    indices = _component_buffer_indices(field)
    for index in indices
        component = field.components[index]
        ensure_layout!(component, layout)
        if !using_pencils
            synchronize_field_architecture!(component; arch=arch,
                                            move_grid = layout == :g,
                                            move_coefficients = layout == :c)
        end
    end

    sample = _component_buffer_data(field.components[1], layout)
    sample isa AbstractArray || throw(ArgumentError(field isa VectorField ?
        "stack_components requires array-backed components, got $(typeof(sample))" :
        "Tensor components must be array-backed"))
    local_sample = using_pencils ? get_local_data(sample) : sample
    local_sample isa AbstractArray || throw(ArgumentError(field isa VectorField ?
        "Unable to obtain local data for stacking" : "Unable to obtain local tensor data for stacking"))

    buffer_shape = (size(field.components)..., size(local_sample)...)
    buffer_eltype = eltype(local_sample)
    needs_new = force || field.component_buffer === nothing ||
                size(field.component_buffer) != buffer_shape ||
                eltype(field.component_buffer) != buffer_eltype ||
                architecture(field.component_buffer) != arch

    if needs_new
        field.component_buffer = zeros(arch, buffer_eltype, buffer_shape...)
    end

    for index in indices
        src = _component_buffer_data(field.components[index], layout)
        src_local = using_pencils ? get_local_data(src) : src
        copyto!(_component_buffer_slice(field.component_buffer, index), src_local)
    end

    field.buffer_layout = layout
    field.buffer_architecture = arch
    return field.component_buffer
end

function _unstack_component_buffers!(field, buffer, layout)
    using_pencils = is_pencil_storage(field)
    buffer_arch = architecture(buffer)
    indices = _component_buffer_indices(field)

    for index in indices
        component = field.components[index]
        ensure_layout!(component, layout)
        slice_view = _component_buffer_slice(buffer, index)
        dest = _component_buffer_data(component, layout)
        if using_pencils
            dest = get_local_data(dest)
            if buffer_arch != CPU()
                copyto!(dest, on_architecture(CPU(), slice_view))
            else
                dest .= slice_view
            end
        else
            dest .= slice_view
        end
    end

    if !using_pencils
        for index in indices
            synchronize_field_architecture!(field.components[index]; arch=buffer_arch,
                                            move_grid = layout == :g,
                                            move_coefficients = layout == :c)
        end
    end

    field.component_buffer = buffer
    field.buffer_layout = layout
    field.buffer_architecture = buffer_arch
    return field
end

"""
    change_scales!(lf::LockedField, new_scales)

Attempt to change scales on a locked field.
Only succeeds if new_scales matches locked scales or locked scales is nothing.
"""
