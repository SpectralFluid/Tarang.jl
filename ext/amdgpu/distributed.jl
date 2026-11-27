Tarang._device_aware_mpi(::ROCGPU) = MPI.has_rocm()
Tarang._verify_device_aware_mpi(arch::ROCGPU) = Tarang._device_aware_mpi(arch)

const _LOCAL_FFT_PLANS = Dict{Tuple,Any}()
function _local_fft_plan(arch, shape, T, dim; real_input=false)
    ensure_device!(arch)
    key = (_current_device_id(), shape, T, dim, real_input)
    return get!(_transform_cache(_LOCAL_FFT_PLANS), key) do
        plan_gpu_fft_dim(arch, shape, T, dim; real_input)
    end
end

function Tarang.fft_in_dim!(data::ROCArray{T}, dim::Int, direction::Symbol, arch::ROCGPU;
                            plan=nothing) where {T<:Complex}
    isempty(data) && return data
    p = _local_fft_plan(arch, size(data), T, dim)
    scratch = only(get_gpu_dct_scratch(arch, size(data), T, 1; purpose=:transpose_fft))
    if direction === :forward
        gpu_fft_dim!(scratch, data, p)
    elseif direction === :backward
        gpu_ifft_dim!(scratch, data, p)
    else
        throw(ArgumentError("Unknown FFT direction $direction"))
    end
    return copyto!(data, scratch)
end
function Tarang.dct_in_dim!(data::ROCArray, dim::Int, direction::Symbol, arch::ROCGPU; grid_size=nothing)
    isnothing(grid_size) || grid_size == size(data, dim) || throw(DimensionMismatch(
        "A distributed DCT-I must span the complete local axis"))
    ensure_device!(arch)
    return gpu_dct1_along_dim!(data, data, dim, direction)
end
for (name, direction) in ((:local_fft_dim!, :forward), (:local_ifft_dim!, :backward))
    @eval function Tarang.$name(data::ROCArray, dim::Int, dfft::Tarang.DistributedGPUFFT)
        arch = Tarang.architecture(data)
        Tarang._distributed_fft_device_id(AMDGPU.device_id(arch.device) - 1, dfft.config.device_id)
        ensure_device!(arch)
        result = complex.(data)
        return Tarang.fft_in_dim!(result, dim, $(QuoteNode(direction)), arch)
    end
end

# The full and half spectra have different Z pencils but the same transverse
# partition in XLocal. Transform x while every column still contains real grid
# values. The remaining Fourier/DCT stages then operate on the reduced pencil.
function _half_pencil_transform!(arch, field, grid_ws, coeff_ws, real_x, p, direction)
    g, c = grid_ws.buffers, coeff_ws.buffers
    xgrid = g.x_local_data
    grid = get_grid_data(field)
    coeff = get_coeff_data(field)
    function stage!(a, axis)
        isempty(a) && return
        Tarang.transform_in_dim!(a, axis, direction, field.bases[axis], arch)
    end

    if direction === :forward
        g.active_layout[] = Tarang.ZLocal
        g.z_local_data .= grid
        Tarang.transpose_z_to_y!(grid_ws)
        Tarang.transpose_y_to_x!(grid_ws)
        if !isempty(real_x)
            real_x .= real.(xgrid)
            gpu_fft_dim!(c.x_local_data, real_x, p)
        end
        c.active_layout[] = Tarang.XLocal
        Tarang.transpose_x_to_y!(coeff_ws)
        stage!(c.y_local_data, 2)
        Tarang.transpose_y_to_z!(coeff_ws)
        stage!(c.z_local_data, 3)
        copyto!(coeff, c.z_local_data)
    elseif direction === :backward
        c.active_layout[] = Tarang.ZLocal
        copyto!(c.z_local_data, coeff)
        stage!(c.z_local_data, 3)
        Tarang.transpose_z_to_y!(coeff_ws)
        stage!(c.y_local_data, 2)
        Tarang.transpose_y_to_x!(coeff_ws)
        if !isempty(real_x)
            gpu_ifft_dim!(real_x, c.x_local_data, p)
            xgrid .= real_x
        end
        g.active_layout[] = Tarang.XLocal
        Tarang.transpose_x_to_y!(grid_ws)
        Tarang.transpose_y_to_z!(grid_ws)
        grid .= real.(g.z_local_data)
    else
        throw(ArgumentError("Unknown transform direction $direction"))
    end
    return true
end

function Tarang._distributed_gpu_transform!(arch::ROCGPU, field::ScalarField, direction::Symbol)
    field.dist.size > 1 || return false
    return Tarang._with_gpu_workspace(_distributed_roc_transform!, field.dist, arch, field, direction)
end
function _distributed_roc_transform!(arch::ROCGPU, field::ScalarField, direction::Symbol)
    dist = field.dist
    MPI.Allreduce(Tarang._device_aware_mpi(arch), &, dist.comm) ||
        throw(ArgumentError("Distributed AMD fields require ROCm-aware MPI on every rank"))
    Tarang._uses_distributed_gpu_dct(field) || return false
    field.dtype <: Real && first(field.bases) isa RealFourier || return false
    ensure_device!(arch)
    dist.closed && throw(ArgumentError("Cannot transform a field on a closed Distributor"))
    shape = Tarang.global_shape(field.domain)
    coeffshape = Tarang.layout_coefficient_shape(field.bases, field.dtype)
    function workspace(kind, shape)
        get!(dist.transpose_workspace_cache, (kind, shape, field.dtype)) do
            Tarang._build_transpose_workspace(field, shape, nothing)
        end
    end
    g = workspace(:rocfft_grid, shape)
    c = workspace(:rocfft_coeff, coeffshape)
    valid = size(get_grid_data(field)) == size(g.buffers.z_local_data) &&
            size(get_coeff_data(field)) == size(c.buffers.z_local_data)
    MPI.Allreduce(valid, &, dist.comm) || throw(DimensionMismatch(
        "The distributed ROCm field buffers must match the grid and coefficient pencils"))
    xshape = size(g.buffers.x_local_data)
    real_x = only(get_gpu_dct_scratch(arch, xshape, field.dtype, 1; purpose=:rfft_pencil))
    p = isempty(real_x) ? nothing : _local_fft_plan(arch, xshape, field.dtype, 1; real_input=true)
    return _half_pencil_transform!(arch, field, g, c, real_x, p, direction)
end
