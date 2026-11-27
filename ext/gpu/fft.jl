include("fft_plans.jl")

struct GPUFFTPlanDim{F,I}
    forward::F
    inverse::I
    is_real::Bool
end

function plan_gpu_fft_dim(arch::GPU{NativeDevice}, shape::Tuple, T::Type, dim::Int; real_input=false)
    ensure_device!(arch)
    return _plan_fft_axis(arch, shape, T, dim, real_input)
end

function _plan_fft_axis(arch, shape, T, dim, real_input)
    input = zeros(arch, real_input ? real(T) : complex(T), shape)
    forward, inverse = _plan_fft_region(input, (dim,); real_input)
    return GPUFFTPlanDim(forward, inverse, real_input)
end

gpu_fft_dim!(dest, src, p::GPUFFTPlanDim) = mul!(dest, p.forward, src)
gpu_ifft_dim!(dest, src, p::GPUFFTPlanDim; destroy_input=false) = mul!(dest, p.inverse, src)

function Tarang._device_complex_fft_plans!(a::DeviceArray, dims)
    ensure_device!(Tarang.architecture(a))
    return _plan_fft_region(a, dims; inplace=true)
end

function Tarang._real_padded_fft_plans(arch::GPU{NativeDevice}, a, original, padhalf, half, dims)
    ensure_device!(arch)
    pf, pb, ps = _plan_real_workspace(arch, a, dims, padhalf)
    sf, sb, s = _plan_real_workspace(arch, original, dims, half)
    return pf, pb, sf, sb, ps, s
end

const _RESAMPLE_SCRATCH = Dict{Tuple,Any}()
function get_gpu_dct_scratch(arch::GPU{NativeDevice}, shape, ::Type{T}, count; purpose=:default) where T
    ensure_device!(arch)
    key = (_current_device_id(), shape, T, count, purpose)
    return get!(_transform_cache(_RESAMPLE_SCRATCH), key) do
        [DeviceArray{T}(undef, shape) for _ in 1:count]
    end
end

function _check_field_transform(arch, field, data, direction)
    ensure_device!(arch)
    data isa DeviceArray && DeviceRuntime.device(data) == arch.device ||
        Tarang.gpu_transform_unsupported(field, direction, "field storage belongs to a different device")
    field.dist.size == 1 ||
        Tarang.gpu_transform_unsupported(field, direction, "a local GPU FFT cannot transform an MPI-distributed field")
    !isempty(field.bases) && all(b -> b isa Union{RealFourier,ComplexFourier,ChebyshevT}, field.bases) ||
        Tarang.gpu_transform_unsupported(field, direction, "this GPU transform supports Fourier and ChebyshevT bases")
    Tarang.should_use_gpu_fft(field, Tarang.get_scaled_shape(field))
end

function Tarang._gpu_forward_transform_backend!(arch::GPU{NativeDevice}, field::ScalarField)
    grid = get_grid_data(field)
    _check_field_transform(arch, field, grid, :forward)
    plan = get_gpu_mixed_transform_plan(arch, field.bases, size(grid), eltype(grid))
    T = Tarang.coefficient_eltype(field.domain, field.dtype)
    coeff = get_coeff_data(field)
    if !(coeff isa DeviceArray{T}) || size(coeff) != plan.coeff_shape || DeviceRuntime.device(coeff) != arch.device
        coeff = DeviceArray{T}(undef, plan.coeff_shape)
        set_coeff_data!(field, coeff)
    end
    gpu_mixed_forward_transform!(coeff, grid, plan)
    return true
end
function Tarang._gpu_backward_transform_backend!(arch::GPU{NativeDevice}, field::ScalarField)
    coeff = get_coeff_data(field)
    _check_field_transform(arch, field, coeff, :backward)
    return _gpu_backward_staged!(field, arch, coeff, Tarang.get_scaled_shape(field))
end
