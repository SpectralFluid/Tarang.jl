"""
    Transform Grouped - Grouped transform operations

This file contains grouped transform operations that process multiple fields
at once for better efficiency (following GROUP_TRANSFORMS pattern).
"""

# ============================================================================
# Grouped Transforms (Following GROUP_TRANSFORMS)
# ============================================================================

"""
    GroupedTransformConfig

Configuration for grouped transform operations.
"""
mutable struct GroupedTransformConfig
    enabled::Bool                    # Whether to use grouped transforms
    min_fields::Int                  # Minimum number of fields to trigger grouping
    max_batch_size::Int             # Maximum number of fields per batch
    batch_buffer::Union{Nothing, AbstractArray}  # Reusable buffer for batching

    function GroupedTransformConfig()
        new(true, 2, 32, nothing)
    end
end

const GROUPED_TRANSFORM_CONFIG = GroupedTransformConfig()

"""
    set_group_transforms!(enabled::Bool; min_fields::Int=2, max_batch_size::Int=32)

Enable or disable grouped transforms. When enabled, multiple fields are transformed
together in batches for improved efficiency.

Following GROUP_TRANSFORMS configuration.
"""
function set_group_transforms!(enabled::Bool; min_fields::Int=2, max_batch_size::Int=32)
    GROUPED_TRANSFORM_CONFIG.enabled = enabled
    GROUPED_TRANSFORM_CONFIG.min_fields = min_fields
    GROUPED_TRANSFORM_CONFIG.max_batch_size = max_batch_size
    return nothing
end

"""
    group_forward_transform!(fields::Vector{<:ScalarField})

Apply forward transforms to multiple fields. Compatible CPU/MPI fields share
a cached packed transform and MPI exchanges. Other fields use their ordinary
per-field transform paths.
"""
function group_forward_transform!(fields::Vector{<:ScalarField})
    return _group_field_transforms!(fields, true)
end

"""
    group_backward_transform!(fields::Vector{<:ScalarField})

Apply backward transforms to multiple fields. Compatible CPU/MPI fields share
a cached packed transform and MPI exchanges. Other fields use their ordinary
per-field transform paths.
"""
function group_backward_transform!(fields::Vector{<:ScalarField})
    return _group_field_transforms!(fields, false)
end

# A trailing local batch dimension coalesces MPI exchanges while preserving the
# base plan's Fourier axes, real half spectrum, and memory permutations. Scratch
# belongs to the exact domain/dtype bundle and is used only by the coordinator.
mutable struct GroupedPencilFFTWorkspace{P,G,C,S,F,B}
    plan::P
    grid::G
    coefficients::C
    solve::S
    to_solve::F
    from_solve::B
    forward_calls::Int
    backward_calls::Int
    fields_transformed::Int
    collective_exchanges::Int
    exchanges_per_transform::Int
end

function _packed_transpose_path(destination, source)
    sp = PencilArrays.pencil(source)
    dp = PencilArrays.pencil(destination)
    source_decomp = collect(PencilArrays.decomposition(sp))
    target_decomp = collect(PencilArrays.decomposition(dp))
    path = PencilArrays.Transpositions.Transposition[]
    current = source
    function hop!(decomposition)
        pencil = PencilArrays.Pencil(PencilArrays.topology(sp), PencilArrays.size_global(sp),
            Tuple(decomposition); permute=PencilArrays.NoPermutation())
        next = PencilArrays.PencilArray{eltype(source)}(undef, pencil,
                                                      PencilArrays.extra_dims(source)...)
        push!(path, PencilArrays.Transpositions.Transposition(next, current;
            method=PencilArrays.Transpositions.Alltoallv()))
        current = next
        source_decomp = decomposition
    end
    if count(source_decomp .!= target_decomp) > 1
        for slot in eachindex(target_decomp)
            source_decomp[slot] == target_decomp[slot] && continue
            if target_decomp[slot] in source_decomp
                local_axis = first(setdiff(1:length(PencilArrays.size_global(sp)), source_decomp))
                hop!([axis == target_decomp[slot] ? local_axis : axis for axis in source_decomp])
            end
            hop!([i == slot ? target_decomp[i] : source_decomp[i] for i in eachindex(source_decomp)])
        end
    end
    push!(path, PencilArrays.Transpositions.Transposition(destination, current;
        method=PencilArrays.Transpositions.Alltoallv()))
    return path
end

_packed_path_exchanges(path) = count(t -> t.dim !== nothing, path)

function _build_grouped_fft_workspace(bundle::TransformPlanBundle, nfields::Int)
    plan = PencilFFTs.PencilFFTPlan(bundle.pencil_fft_input,
        _pencil_transform_tuple(bundle.forward_ops), real(bundle.dtype);
        extra_dims=(nfields,), transpose_method=PencilArrays.Transpositions.Alltoallv())
    grid = PencilFFTs.allocate_input(plan)
    coefficients = PencilFFTs.allocate_output(plan)
    solve = bundle.pencil_solve === nothing ? nothing :
        PencilArrays.PencilArray{eltype(coefficients)}(undef, bundle.pencil_solve, nfields)
    to_solve = solve === nothing ? () : _packed_transpose_path(solve, coefficients)
    from_solve = solve === nothing ? () : _packed_transpose_path(coefficients, solve)
    fft_exchanges = count(2:length(plan.plans)) do i
        PencilArrays.decomposition(plan.plans[i-1].pencil_out) !=
            PencilArrays.decomposition(plan.plans[i].pencil_in)
    end
    nexchanges = fft_exchanges + _packed_path_exchanges(to_solve) + _packed_path_exchanges(from_solve)
    return GroupedPencilFFTWorkspace(plan, grid, coefficients, solve, to_solve, from_solve,
                                    0, 0, 0, 0, nexchanges)
end

function _grouped_fft_workspace!(bundle::TransformPlanBundle, nfields::Int)
    cache = get!(() -> Dict{Int, Any}(), bundle.pencil_work_cache, :grouped_fft)
    cached = get(cache, nfields, nothing)
    cached === nothing || return cached
    # A full batch and one remainder suffice for a fixed field set. Bound
    # retention when callers change batch size or reuse a domain for other sets.
    length(cache) >= 2 && empty!(cache)
    workspace = _build_grouped_fft_workspace(bundle, nfields)
    cache[nfields] = workspace
    return workspace
end

function _same_pencil_geometry(a, b)
    return PencilArrays.topology(a) === PencilArrays.topology(b) &&
           PencilArrays.size_global(a) == PencilArrays.size_global(b) &&
           PencilArrays.decomposition(a) == PencilArrays.decomposition(b) &&
           PencilArrays.permutation(a) == PencilArrays.permutation(b)
end

function _can_group_pencil_field(field::ScalarField, bundle::TransformPlanBundle)
    field.dist.size > 1 && field.dist.architecture isa CPU || return false
    bundle.pencil_fft_plan === nothing && return false
    bundle.pencil_fft_input === nothing && return false
    bundle.pencil_fft_output === nothing && return false
    all(isone, field.scales) || return false
    grid, coeff = get_grid_data(field), get_coeff_data(field)
    grid isa PencilArrays.PencilArray && coeff isa PencilArrays.PencilArray || return false
    parent(grid) isa Array && parent(coeff) isa Array || return false
    isempty(PencilArrays.extra_dims(grid)) && isempty(PencilArrays.extra_dims(coeff)) || return false
    return _same_pencil_geometry(PencilArrays.pencil(grid), bundle.pencil_fft_input) &&
           _same_pencil_geometry(PencilArrays.pencil(coeff), bundle.pencil_fft_output)
end

function _execute_grouped_fft!(workspace::GroupedPencilFFTWorkspace, fields,
                                bundle::TransformPlanBundle, forward::Bool)
    _count_transform!(forward ? :forward : :backward)
    packed_input = parent(forward ? workspace.grid : workspace.coefficients)
    for (slot, field) in enumerate(fields)
        source = parent(forward ? get_grid_data(field) : get_coeff_data(field))
        copyto!(selectdim(packed_input, ndims(packed_input), slot), source)
    end
    if forward
        mul!(workspace.coefficients, workspace.plan, workspace.grid)
    end
    if workspace.solve !== nothing
        _count_transform!(:coupled_dct)
        foreach(PencilArrays.transpose!, workspace.to_solve)
        if forward
            _solve_layout_forward_transform!(workspace.solve, bundle)
        else
            _solve_layout_backward_transform!(workspace.solve, bundle)
        end
        foreach(PencilArrays.transpose!, workspace.from_solve)
    end
    if !forward
        ldiv!(workspace.grid, workspace.plan, workspace.coefficients)
    end
    packed_output = parent(forward ? workspace.coefficients : workspace.grid)
    for (slot, field) in enumerate(fields)
        destination = parent(forward ? get_coeff_data(field) : get_grid_data(field))
        source = selectdim(packed_output, ndims(packed_output), slot)
        if eltype(destination) <: Real && eltype(source) <: Complex
            destination .= real.(source)
        else
            copyto!(destination, source)
        end
        field.current_layout = forward ? :c : :g
    end
    workspace.forward_calls += forward
    workspace.backward_calls += !forward
    workspace.fields_transformed += length(fields)
    workspace.collective_exchanges += workspace.exchanges_per_transform
    return nothing
end

function _group_field_transforms!(fields, forward::Bool; mpi_only::Bool=false)
    target = forward ? :c : :g
    config = GROUPED_TRANSFORM_CONFIG
    if !config.enabled || length(fields) < config.min_fields
        mpi_only || foreach(f -> ensure_layout!(f, target), fields)
        return nothing
    end
    # Ordered vectors keep collective order identical on all ranks. The keys
    # use exact bundle identity within a rank; membership, unlike local sizes,
    # depends only on the replicated domain/type metadata.
    groups = Vector{Vector{ScalarField}}()
    keys = UInt[]
    for field in fields
        field.current_layout === target && continue
        if isempty(field.bases) || !(field.dist.architecture isa CPU) || field.dist.size <= 1
            mpi_only || ensure_layout!(field, target)
            continue
        end
        bundle = _field_transform_bundle(field)
        if !_can_group_pencil_field(field, bundle)
            mpi_only || ensure_layout!(field, target)
            continue
        end
        key = objectid(bundle)
        index = findfirst(==(key), keys)
        if index === nothing
            push!(keys, key)
            push!(groups, ScalarField[field])
        elseif !any(f -> f === field, groups[index])
            push!(groups[index], field)
        end
    end
    for group in groups
        max_batch = max(1, config.max_batch_size)
        for first in 1:max_batch:length(group)
            batch = view(group, first:min(first + max_batch - 1, length(group)))
            if length(batch) < config.min_fields
                mpi_only || foreach(f -> ensure_layout!(f, target), batch)
                continue
            end
            bundle = _field_transform_bundle(batch[1])
            workspace = _grouped_fft_workspace!(bundle, length(batch))
            _execute_grouped_fft!(workspace, batch, bundle, forward)
        end
    end
    return nothing
end

"""Precompute compatible CPU/MPI RHS coefficients before per-field consumers.
Serial, device, scaled, and singleton fields retain their existing lazy paths."""
function _group_rhs_coefficients!(fields)
    length(fields) >= GROUPED_TRANSFORM_CONFIG.min_fields || return nothing
    isempty(fields) && return nothing
    fields[1].dist.size > 1 && fields[1].dist.architecture isa CPU || return nothing
    return _group_field_transforms!(fields, true; mpi_only=true)
end
