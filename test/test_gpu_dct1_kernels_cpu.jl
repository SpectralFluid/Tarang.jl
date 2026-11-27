"""
Value tests for the GPU DCT-I / Chebyshev-derivative kernels AND for the full
Fourier-Fourier / Fourier-Chebyshev device transform drivers — WITHOUT a GPU.

The kernels in `ext/cuda/cheb_deriv.jl` are KernelAbstractions kernels, so the
very same kernel objects the CUDA path launches on a `CUDABackend()` also run on
`KernelAbstractions.CPU()` over plain `Array`s (CUDA.jl loads fine on machines
with no NVIDIA GPU). Only the cuFFT `mul!` is device-specific; FFTW's
`plan_rfft` has identical semantics, so composing the REAL kernels with FFTW
reproduces the device pipeline value-for-value.

The final block goes past the kernels and replays the DRIVERS
(`gpu_mixed_forward_transform!` / `_gpu_forward_transform_impl!`) end to end for
every FF and FC layout in 2D and 3D, so the stage order, the per-axis rfft-vs-fft
choice, the Chebyshev truncate/zero-pad and the permuted-axis DCT are all pinned
against the CPU chain — coverage the 2D-only kernel pins never reached.

These pins matter because the kernels were rewritten from one-thread-per-COLUMN
(with a serial inner loop) to one-thread-per-ELEMENT, four pairs were fused
(reverse+extension, prescale+extension, extract+normalize, extract+finalize),
and the complex path switched from `real./imag./complex.` splits to a packed
re/im batch. A wrong index in any of them is a silent value bug on hardware
this CI does not have.
"""

using Test
using Tarang
using KernelAbstractions
using FFTW
import LinearAlgebra: mul!

mutable struct _DCTCountedFFT{P}
    plan::P
    calls::Int
end
function mul!(output, plan::_DCTCountedFFT, input)
    plan.calls += 1
    return mul!(output, plan.plan, input)
end

const _CUDA_LOADED = try
    @eval using CUDA
    true
catch err
    @info "CUDA.jl unavailable; skipping GPU DCT-I kernel value tests" err
    false
end

# Independent statement of Tarang's Chebyshev DCT-I convention
# (chebyshev.jl): forward = REDFT00 · 1/(N-1) · ½-endpoints · odd-flip.
function _ref_cheb_forward(x::Vector{Float64})
    n = length(x)
    c = FFTW.r2r(x, FFTW.REDFT00) ./ (n - 1)
    c[1] /= 2
    c[end] /= 2
    return c .* [iseven(k) ? 1.0 : -1.0 for k in 0:n-1]
end

# Backward = odd-flip · ×2-endpoints · REDFT00 · ½ (the inverse of the above).
function _ref_cheb_backward(c::Vector{Float64})
    y = c .* [iseven(k) ? 1.0 : -1.0 for k in 0:n_ref_len(c)-1]
    y[1] *= 2
    y[end] *= 2
    return FFTW.r2r(y, FFTW.REDFT00) ./ 2
end
n_ref_len(c) = length(c)

@testset "GPU DCT-I kernels on the KA CPU backend" begin
    if !_CUDA_LOADED
        @test_skip "CUDA.jl not loadable in this environment"
    else
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @test ext !== nothing
        backend = KernelAbstractions.CPU()

        runk(kern, args...; ndrange) = begin
            kern(backend, 64)(args...; ndrange=ndrange)
            KernelAbstractions.synchronize(backend)
        end

        # Compose the actual ext kernels with FFTW standing in for cuFFT —
        # mirrors `_dct1_batch_mat!` exactly (in-place capable: out may === in).
        function emu_dct1!(out::Matrix{Float64}, mat::Matrix{Float64}, direction::Symbol)
            n, batch = size(mat)
            M = 2 * (n - 1)
            work_ext = zeros(M, batch)
            work_cx = zeros(ComplexF64, div(M, 2) + 1, batch)
            @assert size(work_cx, 1) == n
            p = FFTW.plan_rfft(work_ext, 1)
            if direction === :forward
                runk(ext._dct1_reverse_ext_kernel!, work_ext, mat, n, batch; ndrange=(M, batch))
                mul!(work_cx, p, work_ext)
                runk(ext._dct1_extract_normalize_kernel!, out, work_cx, n, batch,
                     1.0 / (n - 1); ndrange=(n, batch))
            else
                runk(ext._dct1_prescale_ext_kernel!, work_ext, mat, n, batch; ndrange=(M, batch))
                mul!(work_cx, p, work_ext)
                runk(ext._dct1_extract_finalize_kernel!, out, work_cx, n, batch; ndrange=(n, batch))
            end
            return out
        end
        emu_dct1(mat, direction) = emu_dct1!(similar(mat), mat, direction)

        @testset "forward DCT-I matches the CPU convention (n=$n)" for n in (9, 16)
            batch = 5
            mat = randn(n, batch)
            got = emu_dct1(mat, :forward)
            for j in 1:batch
                @test got[:, j] ≈ _ref_cheb_forward(mat[:, j]) atol = 1e-13
            end
        end

        @testset "backward inverts forward, and matches the convention (n=$n)" for n in (9, 16)
            batch = 4
            mat = randn(n, batch)
            fwd = emu_dct1(mat, :forward)
            @test emu_dct1(fwd, :backward) ≈ mat atol = 1e-12
            for j in 1:batch
                @test emu_dct1!(similar(mat), fwd, :backward)[:, j] ≈
                      _ref_cheb_backward(fwd[:, j]) atol = 1e-12
            end
        end

        @testset "in-place: out === in is safe" begin
            n, batch = 12, 3
            mat = randn(n, batch)
            expect = emu_dct1(mat, :forward)
            work = copy(mat)
            emu_dct1!(work, work, :forward)
            @test work ≈ expect atol = 1e-13
        end

        @testset "plain extension kernel (derivative step 5) still mirrors correctly" begin
            n, batch = 9, 3
            M = 2 * (n - 1)
            inp = randn(n, batch)
            work = fill(NaN, M, batch)
            runk(ext._dct1_ext_kernel!, work, inp, n, batch; ndrange=(M, batch))
            for j in 1:batch
                @test work[1:n, j] == inp[:, j]
                for k in 1:n-2
                    @test work[n+k, j] == inp[n-k, j]
                end
            end
        end

        @testset "complex pack/unpack kernels" begin
            n, batch = 8, 6
            cx = randn(ComplexF64, n, batch)
            packed = fill(NaN, n, 2batch)
            runk(ext._cheb_pack_reim_kernel!, packed, cx, n, batch; ndrange=(n, 2batch))
            @test packed[:, 1:batch] == real.(cx)
            @test packed[:, batch+1:2batch] == imag.(cx)
            back = zeros(ComplexF64, n, batch)
            runk(ext._cheb_unpack_reim_kernel!, back, packed, n, batch; ndrange=(n, batch))
            @test back == cx
        end

        @testset "fused axis permutation and complex packing" begin
            for T in (Float32,Float64), shape in ((5,7), (4,5,7), (1,5,3), (3,1,7)), axis in 2:length(shape)
                cx = randn(Complex{T}, shape)
                n, batch = size(cx, axis), length(cx) ÷ size(cx, axis)
                packed = fill(T(NaN), n, 2batch)
                perm = (axis, (d for d in 1:length(shape) if d != axis)...)
                reference = reshape(permutedims(cx, perm), n, batch)
                runk(ext._cheb_pack_permuted_reim_kernel!, packed, cx, n, batch,
                     stride(cx, axis); ndrange=(n,batch))
                @test packed[:,1:batch] == real.(reference)
                @test packed[:,batch+1:end] == imag.(reference)
                restored = similar(cx)
                runk(ext._cheb_unpack_permuted_reim_kernel!, restored, packed, n, batch,
                     stride(cx, axis); ndrange=(n,batch))
                @test restored == cx
            end
        end

        # Full derivative pipeline — the ext kernels composed exactly as
        # `_apply_gpu_cheb_deriv_nth!` does, FFTW in place of cuFFT.
        function emu_cheb_deriv_nth!(out::Matrix{T}, mat::Matrix{T}, scale::Float64,
                                      order::Int) where T
            n, batch = size(mat)
            M = 2 * (n - 1)
            work_ext = zeros(T, M, batch)
            work_cx = zeros(Complex{T}, n, batch)
            work_real = zeros(T, n, batch)
            work_deriv = zeros(T, n, batch)
            p = _DCTCountedFFT(FFTW.plan_rfft(work_ext, 1),0)
            workspace = (;work_ext,work_cx,work_real,work_deriv,rfft_plan=p)
            ext._cheb_deriv_pipeline!(mat,out,scale,order,workspace,Tarang.CPU())
            @test p.calls == (order == 0 ? 0 : 2)
            return out
        end
        emu_cheb_deriv_1!(out, mat, scale) = emu_cheb_deriv_nth!(out, mat, scale, 1)

        @testset "derivative pipeline matches chebyshev_derivative_1d! (n=$n)" for n in (9, 17)
            batch = 4
            scale = 2.0 / 3.7
            mat = randn(n, batch)
            got = emu_cheb_deriv_1!(similar(mat), mat, scale)
            for j in 1:batch
                expect = similar(mat[:, j])
                Tarang.chebyshev_derivative_1d!(expect, mat[:, j], scale)
                @test got[:, j] ≈ expect atol = 1e-11
            end
            # Retain the old two-round-trip result as an independent low-order reference.
            got2 = emu_cheb_deriv_1!(similar(mat), got, scale)
            @test emu_cheb_deriv_nth!(similar(mat), mat, scale, 2) ≈ got2 atol=1e-9
            for j in 1:batch
                e1 = similar(mat[:, j]); Tarang.chebyshev_derivative_1d!(e1, mat[:, j], scale)
                e2 = similar(e1);        Tarang.chebyshev_derivative_1d!(e2, e1, scale)
                @test got2[:, j] ≈ e2 atol = 1e-9
            end
        end

        @testset "higher derivatives stay in coefficient space, with alias-safe output" begin
            for T in (Float32,Float64), n in (9,12)
                scale = 2.0 / 3.7
                x = T.(-cos.(pi .* (0:n-1) ./ (n-1)))
                degree = min(8,n-1)
                mat = hcat(x.^degree, T(0.3) .* x.^(degree-1))
                original = copy(mat)
                orders = T === Float64 ? (0,1,2,3,4,8,n) : (0,1,2,3,n)
                for order in orders
                    exact = hcat([order > d ? zeros(T,n) :
                                  T(amplitude * prod(d-order+1:d) * scale^order) .* x.^(d-order)
                                  for (d,amplitude) in ((degree,1.0),(degree-1,0.3))]...)
                    tolerance = (T === Float64 ? 2e-8 : 2e-3) * max(1,maximum(abs,exact))
                    got = emu_cheb_deriv_nth!(similar(mat), mat, scale, order)
                    @test got ≈ exact atol=tolerance rtol=0
                    @test mat == original
                    inplace = copy(mat)
                    @test emu_cheb_deriv_nth!(inplace, inplace, scale, order) === inplace
                    @test inplace ≈ got atol=tolerance rtol=0
                end
            end
        end

        @testset "complex nonleading derivatives use fused packing, including in-place" begin
            shape = (5,9,8)
            scale = 2.0 / 3.7
            for axis in 2:3, order in (0,1,2,3,9)
                n, batch = shape[axis], prod(shape) ÷ shape[axis]
                nodes = -cos.(pi .* (0:n-1) ./ (n-1))
                x = reshape(nodes, ntuple(d -> d == axis ? n : 1,3))
                amplitude = [(1+0.7im) * (1+sum(I[d]/10 for d in 1:3 if d != axis))
                             for I in CartesianIndices(shape)]
                original = amplitude .* x.^5
                exact = order > 5 ? zero(original) :
                        amplitude .* (prod(6-order:5) * scale^order) .* x.^(5-order)
                for inplace in (false,true)
                    input = copy(original)
                    output = inplace ? input : similar(input)
                    packed = zeros(n,2batch)
                    runk(ext._cheb_pack_permuted_reim_kernel!,packed,input,n,batch,
                         stride(input,axis);ndrange=(n,batch))
                    emu_cheb_deriv_nth!(packed,packed,scale,order)
                    runk(ext._cheb_unpack_permuted_reim_kernel!,output,packed,n,batch,
                         stride(input,axis);ndrange=(n,batch))
                    @test output ≈ exact atol=1e-8 rtol=0
                    inplace || @test input == original
                end
            end
        end

        # End-to-end: emulate the REWRITTEN mixed 2D RealFourier×Chebyshev
        # forward driver (rfft on dim 1, packed-complex DCT-I on dim 2, last
        # stage writing the coefficient buffer) and compare against the CPU
        # transform chain, bit-for-bit at machine precision.
        @testset "2D RF×Cheb forward: emulated device driver == CPU chain" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=9, bounds=(-1.0, 1.0))
            u = ScalarField(dist, "u", (xb, zb), Float64)
            g = Tarang.get_grid_data(u)
            g .= randn(size(g))
            grid = Array(copy(g))
            u.current_layout = :g
            Tarang.forward_transform!(u)
            cpu_coeffs = Array(copy(Tarang.get_coeff_data(u)))

            # Stage 1: rfft along dim 1 (unnormalized, like cuFFT)
            spec = FFTW.rfft(grid, 1)              # (9, 9) complex
            # Stage 2: packed-complex DCT-I along dim 2 via the ext kernels
            n2 = size(spec, 2)
            b2 = size(spec, 1)
            cperm = permutedims(spec, (2, 1))       # (n2, b2)
            packed = zeros(n2, 2b2)
            runk(ext._cheb_pack_reim_kernel!, packed, cperm, n2, b2; ndrange=(n2, 2b2))
            emu_dct1!(packed, packed, :forward)
            runk(ext._cheb_unpack_reim_kernel!, cperm, packed, n2, b2; ndrange=(n2, b2))
            coeffs = permutedims(cperm, (2, 1))

            @test coeffs ≈ cpu_coeffs atol = 1e-12
        end

        # ── N-D device DRIVER emulation: FF and FC in 2D and 3D ─────────────
        # The 2D block above pins one layout. The device drivers
        # (`gpu_mixed_forward_transform!` / `_gpu_forward_transform_impl!`) also
        # have to get the STAGE ORDER, the per-axis rfft-vs-fft choice, the
        # Chebyshev truncate/zero-pad, and the permuted-axis DCT right in 3D and
        # for every basis position — none of which the 2D case exercises. These
        # replay the driver logic with the SHARED layout rules
        # (`Tarang.forward_layout` / `Tarang.transform_stage_shapes` — the very
        # functions the CUDA plan calls) and the REAL ext kernels, and compare to
        # the CPU chain.
        function emu_dct1_dim!(out::Array{Float64,N}, inp::Array{Float64,N},
                               dim::Int, dir::Symbol) where {N}
            n = size(inp, dim)
            n <= 1 && return (out === inp || copyto!(out, inp); out)
            batch = prod(size(inp)) ÷ n
            if dim == 1
                emu_dct1!(reshape(out, n, batch), reshape(inp, n, batch), dir)
            else
                other = ntuple(i -> i < dim ? i : i + 1, N - 1)
                perm = (dim, other...)
                pshape = ntuple(i -> size(inp, perm[i]), N)
                wp = permutedims(inp, perm)
                mat = reshape(wp, n, batch)
                emu_dct1!(mat, mat, dir)
                permutedims!(out, reshape(mat, pshape), invperm(perm))
            end
            return out
        end

        function emu_dct1_dim!(out::Array{ComplexF64,N}, inp::Array{ComplexF64,N},
                               dim::Int, dir::Symbol) where {N}
            n = size(inp, dim)
            n <= 1 && return (out === inp || copyto!(out, inp); out)
            batch = prod(size(inp)) ÷ n
            packed = zeros(n, 2batch)
            if dim == 1
                cin = reshape(inp, n, batch); cout = reshape(out, n, batch)
                runk(ext._cheb_pack_reim_kernel!, packed, cin, n, batch; ndrange=(n, 2batch))
                emu_dct1!(packed, packed, dir)
                runk(ext._cheb_unpack_reim_kernel!, cout, packed, n, batch; ndrange=(n, batch))
            else
                runk(ext._cheb_pack_permuted_reim_kernel!, packed, inp, n, batch,
                     stride(inp,dim); ndrange=(n,batch))
                emu_dct1!(packed, packed, dir)
                runk(ext._cheb_unpack_permuted_reim_kernel!, out, packed, n, batch,
                     stride(inp,dim); ndrange=(n,batch))
            end
            return out
        end

        _axis_idx(nd, dim, r) = ntuple(i -> i == dim ? r : Colon(), nd)
        _stage_order(bases) = vcat(sort([d for (d, b) in enumerate(bases) if !(b isa ChebyshevT)]),
                                   [d for (d, b) in enumerate(bases) if b isa ChebyshevT])

        function emu_mixed_forward(bases, grid)
            ops, coeff_shape, _ = Tarang.forward_layout(bases, size(grid), eltype(grid))
            order = _stage_order(bases)
            stages = Tarang.transform_stage_shapes(ops, size(grid), order)
            cur = grid
            for (si, dim) in enumerate(order)
                want = stages[si + 1]
                if bases[dim] isa ChebyshevT
                    cur = emu_dct1_dim!(similar(cur), cur, dim, :forward)
                    size(cur) == want ||
                        (cur = cur[_axis_idx(ndims(cur), dim, 1:want[dim])...])
                else
                    cur = ops[dim].op === :rfft ? FFTW.rfft(cur, dim) :
                          FFTW.fft(eltype(cur) <: Complex ? cur : complex.(cur), dim)
                end
                @assert size(cur) == want "forward stage $si (dim $dim): $(size(cur)) != $want"
            end
            @assert size(cur) == coeff_shape
            return cur
        end

        function emu_mixed_backward(bases, coeffs, grid_shape, dtype)
            ops, _, _ = Tarang.forward_layout(bases, grid_shape, dtype)
            order = _stage_order(bases)
            stages = Tarang.transform_stage_shapes(ops, grid_shape, order)
            cur = coeffs
            for si in reverse(eachindex(order))
                dim = order[si]
                want = stages[si]
                if bases[dim] isa ChebyshevT
                    inp = cur
                    if size(cur) != want
                        padded = zeros(eltype(cur), want...)
                        padded[_axis_idx(length(want), dim, 1:size(cur, dim))...] .= cur
                        inp = padded
                    end
                    cur = emu_dct1_dim!(similar(inp), inp, dim, :backward)
                else
                    cur = ops[dim].op === :rfft ? FFTW.irfft(cur, want[dim], dim) :
                          FFTW.ifft(eltype(cur) <: Complex ? cur : complex.(cur), dim)
                end
                @assert size(cur) == want "backward stage $si (dim $dim): $(size(cur)) != $want"
            end
            return dtype <: Real && eltype(cur) <: Complex ? real.(cur) : cur
        end

        emu_ff_forward(bases, grid) =
            Tarang.forward_axis_op(bases[1], size(grid, 1), eltype(grid) <: Complex).op === :rfft ?
                FFTW.rfft(grid) : FFTW.fft(complex.(grid))

        function emu_ff_backward(bases, coeffs, grid_shape, dtype)
            if bases[1] isa RealFourier && !(dtype <: Complex)
                op = Tarang.backward_axis_op(bases[1], size(coeffs, 1), grid_shape[1], true)
                op.op === :irfft && return FFTW.irfft(coeffs, op.out_len)
            end
            r = FFTW.ifft(eltype(coeffs) <: Complex ? coeffs : complex.(coeffs))
            return dtype <: Real ? real.(r) : r
        end

        _mkb(c, nm, kind, n) =
            kind === :RF ? RealFourier(c[nm]; size=n, bounds=(0.0, 2π)) :
            kind === :CF ? ComplexFourier(c[nm]; size=n, bounds=(0.0, 2π)) :
                           ChebyshevT(c[nm]; size=n, bounds=(-1.0, 1.0))

        _ND_CASES = [
            ("FF 2D RF*RF",        (:RF, :RF),          (16, 16), Float64),
            ("FF 2D RF*RF odd",    (:RF, :RF),          (15, 9),  Float64),
            ("FF 2D CF*RF",        (:CF, :RF),          (12, 16), Float64),
            ("FF 3D RF*CF*RF",     (:RF, :CF, :RF),     (8, 6, 10), Float64),
            ("FF 3D CF*CF*CF",     (:CF, :CF, :CF),     (6, 6, 6), ComplexF64),
            ("FC 2D RF*CHEB",      (:RF, :CHEB),        (16, 9),  Float64),
            ("FC 2D CHEB*RF",      (:CHEB, :RF),        (9, 16),  Float64),
            ("FC 2D CF*CHEB",      (:CF, :CHEB),        (12, 9),  ComplexF64),
            ("FC 3D RF*RF*CHEB",   (:RF, :RF, :CHEB),   (8, 8, 9), Float64),
            ("FC 3D CHEB*RF*RF",   (:CHEB, :RF, :RF),   (9, 8, 8), Float64),
            ("FC 3D RF*CHEB*RF",   (:RF, :CHEB, :RF),   (8, 9, 8), Float64),
            ("FC 3D RF*CHEB*CHEB", (:RF, :CHEB, :CHEB), (8, 9, 7), Float64),
            ("FC 3D CHEB*CHEB*RF", (:CHEB, :CHEB, :RF), (7, 9, 8), Float64),
        ]

        @testset "device driver == CPU chain: $label" for (label, kinds, sizes, T) in _ND_CASES
            names = ("x", "y", "z")[1:length(kinds)]
            coords = CartesianCoordinates(names...)
            dist = Distributor(coords; mesh=(1,), dtype=T)
            bases = ntuple(i -> _mkb(coords, names[i], kinds[i], sizes[i]), length(kinds))
            u = ScalarField(dist, "u", bases, T)
            ensure_layout!(u, :g)
            g = Tarang.get_grid_data(u)
            grid = T <: Complex ? randn(ComplexF64, size(g)...) : randn(size(g)...)
            g .= grid
            u.current_layout = :g
            Tarang.forward_transform!(u)
            cpu_c = Array(copy(Tarang.get_coeff_data(u)))
            Tarang.backward_transform!(u)
            ensure_layout!(u, :g)
            cpu_g = Array(copy(Tarang.get_grid_data(u)))

            has_cheb = any(k -> k === :CHEB, kinds)
            emu_c = has_cheb ? emu_mixed_forward(bases, grid) : emu_ff_forward(bases, grid)
            emu_g = has_cheb ? emu_mixed_backward(bases, emu_c, size(grid), T) :
                               emu_ff_backward(bases, emu_c, size(grid), T)
            @test size(emu_c) == size(cpu_c)
            @test isapprox(emu_c, cpu_c; atol=1e-12)
            @test isapprox(emu_g, cpu_g; atol=1e-12)
        end
    end
end

# ── TransposableField's GPU per-axis DCT (`Tarang.dct_in_dim!` for CuArray) ──
#
# The PRODUCTION override from ext/cuda/transpose_kernels.jl runs here on host
# arrays together with the PRODUCTION `gpu_dct1_along_dim!` (and its permutation /
# packed re-im helpers and plan cache) from ext/cuda/cheb_deriv.jl: CuArray ===
# Array, kernels on the KA CPU backend, FFTW's rfft standing in for cuFFT's. Only
# device plumbing is shimmed, so this needs no CUDA.jl. The override used to undo
# `gpu_dct1_along_dim!`'s odd-degree flip to match a CPU reference that lacked it;
# both handed every coefficient consumer the mirrored field. Pinned to the
# canonical convention through the CPU reference, which the support tests pin
# against the serial ChebyshevTransform.
module GPUDctInDimEmulation
import Tarang as RealTarang
using LinearAlgebra, FFTW, KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
module Tarang   # absorbs cheb_deriv.jl's `Tarang._gpu_chebyshev_deriv!` method
    import Tarang as T0
    const architecture = T0.architecture
    const ScalarField = T0.ScalarField
    const get_grid_data = T0.get_grid_data
    const set_grid_data! = T0.set_grid_data!
    function _gpu_chebyshev_deriv! end
end
const launch! = RealTarang.launch!
const CuArray = Array
const CuMatrix = Matrix
module CUDA
    zeros(T, dims...) = Base.zeros(T, dims...)
    device() = 0
    synchronize() = nothing
end
module CUFFT
    using FFTW
    # cuFFT, unlike FFTW, cannot plan an empty batch (CUDA.jl passes batch 0 to
    # cufftXtMakePlanMany -> CUFFT_INVALID_SIZE); the stand-in refuses it too.
    function plan_rfft(x, region)
        isempty(x) && error("cuFFT cannot plan an empty batch (size $(size(x)))")
        return FFTW.plan_rfft(x, region; flags=FFTW.ESTIMATE | FFTW.UNALIGNED)
    end
end
ensure_device!(arch) = nothing
get_gpu_dct_scratch(arch, shape, T, count; purpose=:default) = [zeros(T, shape) for _ in 1:count]
const _ROOT = dirname(@__DIR__)
include(joinpath(_ROOT, "ext", "cuda", "cheb_deriv.jl"))

# Production override, renamed so it never extends the real `Tarang.dct_in_dim!`.
function _find_override(ex)
    ex isa Expr || return nothing
    if ex.head === :function
        sig = ex.args[1]
        call = sig isa Expr && sig.head === :where ? sig.args[1] : sig
        call isa Expr && call.head === :call && call.args[1] == :(Tarang.dct_in_dim!) && return ex
    end
    for a in ex.args
        found = _find_override(a)
        found === nothing || return found
    end
    return nothing
end
let def = _find_override(Meta.parseall(read(joinpath(_ROOT, "ext", "cuda", "transpose_kernels.jl"), String)))
    def === nothing && error("GPU dct_in_dim! override not found in ext/cuda/transpose_kernels.jl")
    def = deepcopy(def)
    call = def.args[1].args[1]          # signature carries `where {T,N}`
    call.args[1] = :gpu_dct_in_dim!
    call.args[end] = :arch              # was `arch::Tarang.GPU`
    Core.eval(@__MODULE__, def)
end
end # GPUDctInDimEmulation

@testset "GPU dct_in_dim! override (KA CPU, FFTW) == canonical CPU dct_in_dim!" begin
    emu = GPUDctInDimEmulation
    for shape in ((9,), (8,), (7, 6), (6, 9), (5, 8, 7), (8, 5, 6)),
        T in (Float64, Float32, ComplexF64), dim in eachindex(shape)
        x = randn(T, shape...)
        tol = real(T) === Float32 ? 1e-5 : 1e-12
        gpu = emu.gpu_dct_in_dim!(copy(x), dim, :forward, nothing)
        cpu = Tarang.dct_in_dim!(copy(x), dim, :forward, Tarang.CPU())
        @test gpu ≈ cpu atol = tol rtol = 0
        @test emu.gpu_dct_in_dim!(copy(gpu), dim, :backward, nothing) ≈ x atol = tol rtol = 0
        @test emu.gpu_dct_in_dim!(copy(gpu), dim, :backward, nothing) ≈
              Tarang.dct_in_dim!(copy(gpu), dim, :backward, Tarang.CPU()) atol = tol rtol = 0
    end
    # Analytic anchor independent of both implementations: u = T0 + T1 + T2 on
    # Tarang's -cos grid has canonical coefficients [1, 1, 1, 0, ...] — the
    # override used to return [1, -1, 1, 0, ...].
    for N in (8, 9)
        x = -cos.(π .* (0:N-1) ./ (N - 1))
        u = reshape(1 .+ x .+ (2 .* x .^ 2 .- 1), 1, N, 1)     # along dim 2 of a 3D array
        c = emu.gpu_dct_in_dim!(copy(u), 2, :forward, nothing)
        @test vec(c) ≈ [1.0, 1.0, 1.0, zeros(N - 3)...] atol = 1e-12
    end
end

# ── Scaled backward through the REAL staged device driver (host emulation) ───
# `ext/cuda/mixed_transforms.jl` and `_gpu_backward_staged!` (ext/cuda/transforms.jl)
# run verbatim on host arrays; only cuFFT, the DCT-I kernels, device scratch and
# device ids are stubbed (the kernels are pinned above). This covers the step the
# replays above cannot see: base coefficients left by a scale change resampled
# into the plan's layout, the rfft-stage Nyquist fold of a downsample, and the
# Chebyshev pad/truncate — the paths a GPU `set_scales!` takes.
module ScaledBackwardEmulation
using Tarang: GPU, ScalarField, RealFourier, ComplexFourier, ChebyshevT
using Tarang: get_grid_data, set_grid_data!
import Tarang
using FFTW, LinearAlgebra
const CuArray = Array
struct CuDevice end
module CUDA
zeros(T, dims...) = Base.zeros(T, dims...)
deviceid(_) = 0
device() = nothing
end
_current_device_id() = 0
ensure_device!(arch) = nothing
struct GPUFFTPlanDim{P,I}
    plan::P
    iplan::I
    is_real::Bool
end
function plan_gpu_fft_dim(arch, shape, T, dim; real_input=false)
    input = zeros(T, shape)
    plan = real_input ? plan_rfft(input, (dim,)) : plan_fft(input, (dim,))
    return GPUFFTPlanDim(plan, inv(plan), real_input)
end
gpu_fft_dim!(out, input, plan) = mul!(out, plan.plan, input)
gpu_ifft_dim!(out, input, plan; destroy_input=false) =
    mul!(out, plan.iplan, destroy_input ? input : copy(input))
get_gpu_dct_scratch(arch, shape, T, count; purpose=:default) = [zeros(T, shape) for _ in 1:count]
# Tarang's DCT-I convention (chebyshev.jl): forward = REDFT00 · 1/(N-1) ·
# ½-endpoints · odd-flip; backward is its inverse.
function gpu_dct1_along_dim!(out, input, dim, direction)
    n = size(input, dim)
    n <= 1 && return copyto!(out, input)
    function component(x)
        x = copy(x)
        if direction === :backward
            selectdim(x, dim, 1) .*= 2
            selectdim(x, dim, n) .*= 2
            for k in 2:2:n
                selectdim(x, dim, k) .*= -1
            end
        end
        y = FFTW.r2r(x, FFTW.REDFT00, (dim,))
        if direction === :forward
            y ./= n - 1
            selectdim(y, dim, 1) .*= 0.5
            selectdim(y, dim, n) .*= 0.5
            for k in 2:2:n
                selectdim(y, dim, k) .*= -1
            end
        else
            y ./= 2
        end
        return y
    end
    out .= eltype(input) <: Complex ?
           complex.(component(real.(input)), component(imag.(input))) : component(input)
    return out
end
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "ext/cuda/mixed_transforms.jl"))
function _definition(ex, name)
    ex isa Expr || return nothing
    if ex.head === :function
        sig = ex.args[1]
        sig isa Expr && sig.head === :where && (sig = sig.args[1])
        sig isa Expr && sig.head === :call && sig.args[1] === name && return ex
    end
    for arg in ex.args
        found = _definition(arg, name)
        found === nothing || return found
    end
    return nothing
end
Core.eval(@__MODULE__, _definition(Meta.parseall(read(joinpath(ROOT, "ext/cuda/transforms.jl"),
                                                     String)), :_gpu_backward_staged!))
end

@testset "Scaled transforms through the staged device driver (host emulation)" begin
    E = ScaledBackwardEmulation
    arch = Tarang.GPU{E.CuDevice}(E.CuDevice())
    kinds_basis(c, nm, kind, n) =
        kind === :RF ? RealFourier(c[nm]; size=n, bounds=(0.0, 2π)) :
        kind === :CF ? ComplexFourier(c[nm]; size=n, bounds=(0.0, 2π)) :
                       ChebyshevT(c[nm]; size=n, bounds=(-1.0, 1.0))
    function emu_field(kinds, sizes, T)
        names = ("x", "y", "z")[1:length(kinds)]
        coords = CartesianCoordinates(names...)
        dist = Distributor(coords; mesh=(1,), dtype=T)
        bases = ntuple(i -> kinds_basis(coords, names[i], kinds[i], sizes[i]), length(kinds))
        return ScalarField(dist, "u", bases, T)
    end
    layouts = [((:RF, :CHEB), (16, 17), Float64), ((:CHEB, :RF), (17, 15), Float64),
               ((:CF, :CHEB), (12, 9), Float64), ((:CF, :CHEB), (12, 9), ComplexF64),
               ((:RF, :RF, :CHEB), (8, 8, 9), Float64), ((:RF, :CHEB, :CHEB), (8, 9, 7), Float64),
               ((:CHEB, :CHEB), (9, 8), Float64), ((:CHEB,), (9,), Float64),
               ((:RF, :RF), (16, 12), Float64), ((:RF, :CF), (16, 12), Float64),
               ((:RF,), (16,), Float64)]

    @testset "base coefficients onto the scaled grid: $kinds $sizes $T s=$s" for
            (kinds, sizes, T) in layouts, s in (1.5, 0.75)
        cpu = emu_field(kinds, sizes, T)
        dev = emu_field(kinds, sizes, T)
        ensure_layout!(cpu, :g)
        get_grid_data(cpu) .= randn(T, size(get_grid_data(cpu))...)
        forward_transform!(cpu)               # base spectrum, Nyquist content included
        ensure_layout!(dev, :c)
        copyto!(get_coeff_data(dev), get_coeff_data(cpu))
        stored = copy(get_coeff_data(dev))
        preset_scales!(cpu, s); backward_transform!(cpu)
        preset_scales!(dev, s)
        @test E._gpu_backward_staged!(dev, arch, get_coeff_data(dev), get_scaled_shape(dev))
        @test size(get_grid_data(dev)) == get_scaled_shape(dev)
        @test isapprox(get_grid_data(dev), get_grid_data(cpu); atol=1e-12)
        @test get_coeff_data(dev) == stored   # the resample never writes the stored spectrum
    end

    @testset "forward from a scaled grid keeps basis-size Chebyshev coefficients" begin
        # Upscaled grids truncate, downscaled ones zero-pad (as the CPU chain);
        # the Fourier axes keep the grid's own spectrum, e.g. (13, 17) at 3/2.
        for (kinds, sizes, T) in layouts[1:8], s in (1.5, 0.75)
            cpu = emu_field(kinds, sizes, T)
            preset_scales!(cpu, s)
            ensure_layout!(cpu, :g)
            grid = randn(T, size(get_grid_data(cpu))...)
            get_grid_data(cpu) .= grid
            forward_transform!(cpu)
            plan = E.plan_gpu_mixed_transform(arch, cpu.bases, size(grid), T)
            coeffs = zeros(eltype(get_coeff_data(cpu)), plan.coeff_shape)
            E.gpu_mixed_forward_transform!(coeffs, grid, plan)
            @test size(coeffs) == size(get_coeff_data(cpu))
            @test isapprox(coeffs, get_coeff_data(cpu); atol=1e-12)
        end
    end
end

# An over-decomposed mesh (more ranks than points along an axis) hands some rank
# an EMPTY block whose transform axis is still long. Its batch is zero, and cuFFT
# cannot plan that: the rank errored while its peers waited in the next transpose.
@testset "empty blocks build no cuFFT plan and launch nothing" begin
    emu = GPUDctInDimEmulation
    plans_before = length(emu._GPU_CHEB_DERIV_CACHE)
    for shape in ((0, 6, 5), (7, 0, 5), (7, 6, 0)), T in (Float64, ComplexF64), dim in 1:3
        shape[dim] > 1 || continue
        x = zeros(T, shape)
        out = zeros(T, shape)
        @test emu.gpu_dct1_along_dim!(out, x, dim, :forward) === out
        @test emu.gpu_dct1_along_dim!(out, x, dim, :backward) === out
        @test emu.gpu_dct_in_dim!(x, dim, :backward, nothing) === x
        deriv! = T <: Real ? emu._gpu_cheb_deriv_into! : emu._gpu_cheb_deriv_complex_into!
        @test deriv!(out, x, dim, 1, 1.0) === out
    end
    @test length(emu._GPU_CHEB_DERIV_CACHE) == plans_before
end
