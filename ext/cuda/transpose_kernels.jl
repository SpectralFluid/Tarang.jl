# ============================================================================
# GPU FFT in dimension
# ============================================================================

"""
Override fft_in_dim! for GPU arrays using CUFFT.

Accepts the `plan` keyword its callers always pass (see `transform_in_dim!` in
src/core/transpose/transforms.jl) but ignores it: the plan handed down
may be a CPU FFTW plan or a placeholder, so the cached CUFFT plan is used instead.
"""
function Tarang.fft_in_dim!(data::CuArray, dim::Int, direction::Symbol, arch::Tarang.GPU;
                            plan=nothing)
    # A rank whose block in this layout is empty (more ranks than points along an
    # axis) has nothing to transform; cuFFT cannot plan an empty batch.
    isempty(data) && return data
    # Pin the device to the DATA's device, and key the plan cache on it too. The
    # `device_id` keyword defaults to whatever device happens to be current,
    # which in a multi-GPU run is not necessarily where `data` lives — the plan
    # would then be built on, and executed against, the wrong device. The
    # sibling `gpu_fft_1d!` has always passed it explicitly; this did not.
    data_device = CUDA.device(data)
    device_id = CUDA.deviceid(data_device)
    prev_device = CUDA.device()
    CUDA.device!(data_device)
    try
        # Use cached CUFFT plan to avoid expensive plan creation per call
        cufft_plan = get_fft_1d_plan(size(data), dim, eltype(data);
                                     inverse=(direction != :forward), device_id=device_id)
        # `data .= cufft_plan * data` allocated a whole device array per call.
        # C2C is non-destructive, so transform into cached scratch and copy back.
        scratch = get_gpu_dct_scratch(Tarang.architecture(data), size(data), eltype(data), 1;
                                      purpose=:transpose_fft)[1]
        mul!(scratch, cufft_plan, data)
        copyto!(data, scratch)
        CUDA.synchronize()
    finally
        CUDA.device!(prev_device)
    end
    return data
end

# ============================================================================
# GPU DCT in dimension (for Chebyshev transforms)
# ============================================================================

"""
Override dct_in_dim! for GPU arrays.

Must match the CPU distributed reference `Tarang.dct_in_dim!(…, ::CPU)` in
src/core/transpose/transforms.jl EXACTLY, and both produce Tarang's CANONICAL
Chebyshev coefficients (those of `_chebyshev_forward` / `_chebyshev_backward`):
  forward:  REDFT00, 1/(N-1) normalization, half-weight at both endpoints,
            odd-degree sign flip
  backward: odd-degree sign flip, double both endpoint coefficients, REDFT00,
            divide by 2

That is precisely the verified building block `gpu_dct1_along_dim!`
(ext/gpu/cheb_deriv.jl), which realises the flip as a grid reversal
(`REDFT00(reverse(x))[k] = (-1)^k REDFT00(x)[k]`), so it is used unmodified.
(This override used to undo that flip to match a CPU reference that lacked it;
the two agreed with each other while both handed every coefficient consumer a
mirrored field.)
"""
function Tarang.dct_in_dim!(data::CuArray{T,N}, dim::Int, direction::Symbol, arch::Tarang.GPU) where {T,N}
    n = size(data, dim)
    # n <= 1 is the identity; an empty block (over-decomposed mesh) has nothing to transform.
    (n <= 1 || isempty(data)) && return data

    RT = T <: Complex ? real(T) : T
    if N > 3 || !(RT <: AbstractFloat)
        error("GPU DCT-I supports real or complex floating-point arrays with at most " *
              "three dimensions; got $(typeof(data)). CPU fallback is disabled.")
    end

    ensure_device!(Tarang.architecture(data))

    # Reshape 1D/2D input to 3D (CuArray reshape shares storage, so writing
    # through data3 writes data). `dim` stays valid: trailing dims are appended.
    data3 = N == 3 ? data : reshape(data, size(data)..., ntuple(_ -> 1, 3 - N)...)

    # Separate purpose avoids aliasing the nested DCT/FFT scratch without
    # retaining unused arrays just to distinguish the cache key.
    out3 = get_gpu_dct_scratch(Tarang.architecture(data), size(data3), T, 1;
                               purpose=:transpose_dct)[1]

    # Complex data goes through `gpu_dct1_along_dim!`'s complex method, which
    # packs re/im as extra batch columns and runs ONE batched DCT-I. The previous
    # version split into `real.(data)` / `imag.(data)`, recursed twice (each
    # recursion allocating its own `similar` and syncing), and recombined —
    # roughly six device allocations and three device-wide syncs per axis.
    # Transforming into the scratch leaves the caller's buffer untouched if the
    # transform throws.
    if direction == :forward || direction == :backward
        gpu_dct1_along_dim!(out3, data3, dim, direction)
    else
        error("direction must be :forward or :backward, got $direction")
    end
    copyto!(data3, out3)

    CUDA.synchronize()
    return data
end

# ============================================================================
# Exports
# ============================================================================

export pack_for_transpose_kernel_3d!, pack_for_transpose_kernel_2d!
export unpack_from_transpose_kernel_3d!, unpack_from_transpose_kernel_2d!
export copy_to_buffer_kernel!, copy_from_buffer_kernel!
export apply_fft_normalization_kernel!
export gpu_pack_for_transpose!, gpu_unpack_from_transpose!
export GPU_PACK_3D_OP, GPU_UNPACK_3D_OP, GPU_PACK_2D_OP, GPU_UNPACK_2D_OP
