"""
    Transform layout — the single source of truth for spectral array layout.

Every transform backend needs the same four answers for each axis: which
operation runs, how long the axis becomes, whether the data turns complex, and
(backward) whether the stored spectrum must be zero-padded first. Those rules
used to be written out separately in `domain.jl` (`coefficient_shape`), in the
CPU stage specs (`fourier.jl` / `chebyshev.jl`), and again in
the CUDA extension (`plan_gpu_mixed_transform` plus the branch chains in
`_gpu_{forward,backward}_transform_impl!`). Three hand-maintained copies of one
rule set is where the layout bugs came from:

  * the GPU mixed plan derived its coefficient shape from the GRID shape, so a
    scaled Chebyshev axis silently kept every grid mode while the CPU chain
    truncated to the basis size;
  * a `ComplexFourier` first axis with a real dtype was handled by the GPU
    (promote to complex) and by the out-of-place CPU helper (`FFTW.fft`), but not
    by the in-place CPU stage, which had no rule for it at all.

The functions here state each rule once. Callers ask for an `AxisOp` instead of
re-deriving `div(N, 2) + 1` — that expression appears at 60+ sites across the
package, and every one of them is a place the copies can drift apart.

The per-axis layout rules are pure, allocation-free, and return isbits values,
so they are safe on the zero-allocation transform hot path. Plan-construction
helpers below may allocate metadata outside that hot path.
"""

"""
    AxisOp

What one transform stage does to one axis.

- `op` — `:rfft`, `:fft`, `:irfft`, `:irfft_resampled`, `:ifft`, `:ifft_resampled`,
  `:dct1_forward`, `:dct1_backward`, or `:none`
- `out_len` — the axis length AFTER the stage
- `out_complex` — whether the stage's output is complex
- `pad_len` — `:irfft_resampled` / `:ifft_resampled` only: the spectral length the
  stored spectrum is zero-padded or truncated to before the inverse (`0` otherwise)
"""
struct AxisOp
    op::Symbol
    out_len::Int
    out_complex::Bool
    pad_len::Int
end

AxisOp(op::Symbol, out_len::Int, out_complex::Bool) = AxisOp(op, out_len, out_complex, 0)

"""
    rfft_len(n) -> Int

Length of the non-redundant half-spectrum of a real signal of length `n`. THE
definition of the rfft layout rule; prefer it over writing `div(n, 2) + 1`.
"""
@inline rfft_len(n::Integer) = div(n, 2) + 1

# `is_fourier_basis` is defined once, in basis/core.jl. A second definition
# lived here and disagreed with it about an absent axis; see the note there.
#
# `first_fourier_axis(bases)` also lived here — "axis index of the first Fourier
# basis, or 0". It had no callers: planning.jl computes a LOCAL variable
# of the same name from its own `fourier_axes` list, which reads like a call and is
# not one. The rule it documented is still true and still enforced there — only
# this axis can be halved, because after any FFT the data is complex, so every
# later Fourier axis runs a full-size C2C transform (serial FFTW dispatches on
# element type; the MPI PencilFFTs plan does RFFT only on the first Fourier axis),
# which is why coefficient shapes agree in serial and distributed runs.

# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

"""
    forward_axis_op(basis, in_len, in_complex) -> AxisOp

The forward stage for one axis, given the CURRENT axis length and whether the
data arriving at this stage is already complex.

`RealFourier` halves the axis only when it still sees real data — that is what
makes "only the first Fourier axis is halved" fall out rather than being a
special case. `ChebyshevT` maps the (possibly scaled) grid length to the basis
size: the DCT-I runs on the full grid and the result is truncated to
`basis.meta.size` coefficients, which is exactly the step the GPU mixed driver
was missing for scaled fields.
"""
@inline function forward_axis_op(basis, in_len::Int, in_complex::Bool)
    if isa(basis, RealFourier)
        return in_complex ? AxisOp(:fft, in_len, true) :
                            AxisOp(:rfft, rfft_len(in_len), true)
    elseif isa(basis, ComplexFourier)
        # Real input is promoted to complex first (FFTW's in-place `mul!` cannot
        # do that itself; the allocating `FFTW.fft` and cuFFT both can).
        return AxisOp(:fft, in_len, true)
    elseif isa(basis, ChebyshevT)
        return AxisOp(:dct1_forward, basis.meta.size, in_complex)
    else
        return AxisOp(:none, in_len, in_complex)
    end
end

"""
    forward_layout(bases, grid_shape, dtype) -> (ops, coeff_shape, coeff_complex)

Walk `bases` in axis order and return the per-axis `AxisOp`s, the resulting
coefficient shape, and whether the coefficients are complex. `grid_shape` is the
LOCAL, possibly scaled grid — pass `basis.meta.size` per axis for the canonical
(unscaled) layout.
"""
function forward_layout(bases, grid_shape::Tuple, dtype::Type)
    n = length(bases)
    ops = Vector{AxisOp}(undef, n)
    shape = collect(Int, grid_shape)
    complex_now = dtype <: Complex
    for ax in 1:n
        op = forward_axis_op(bases[ax], shape[ax], complex_now)
        ops[ax] = op
        shape[ax] = op.out_len
        complex_now = op.out_complex
    end
    return ops, tuple(shape...), complex_now
end

"""
    transform_stage_shapes(ops, input_shape, order) -> Vector{Tuple}

Return the input shape followed by the shape after each axis operation in
`order`. This lets a backend execute the shared `AxisOp` rules in a different
order (for example, Fourier axes before Chebyshev axes on CUDA) without
re-deriving intermediate buffer sizes.

`order` must be a permutation of every axis exactly once. The helper runs only
while constructing a transform plan, so clarity is preferred over hot-path
allocation constraints.
"""
function transform_stage_shapes(ops, input_shape::Tuple, order)
    n = length(ops)
    length(input_shape) == n || throw(DimensionMismatch(
        "received $n axis operations for an input shape with " *
        "$(length(input_shape)) dimensions"))

    axes_order = collect(Int, order)
    if length(axes_order) != n || sort(axes_order) != collect(1:n)
        throw(ArgumentError(
            "transform order must contain every axis 1:$n exactly once, got " *
            "$(Tuple(axes_order))"))
    end

    shape = collect(Int, input_shape)
    stages = Tuple[Tuple(shape)]
    for axis in axes_order
        shape[axis] = ops[axis].out_len
        push!(stages, Tuple(shape))
    end
    return stages
end

function _validate_axis_prefix_shapes(dst::AbstractArray, src::AbstractArray,
                                      axis::Int, operation::Symbol)
    ndims(dst) == ndims(src) || throw(DimensionMismatch(
        "$operation requires arrays with the same dimensionality, got " *
        "$(ndims(dst))D and $(ndims(src))D"))
    1 <= axis <= ndims(src) || throw(ArgumentError(
        "resize axis must be between 1 and $(ndims(src)), got $axis"))
    for dim in 1:ndims(src)
        dim == axis && continue
        size(dst, dim) == size(src, dim) || throw(DimensionMismatch(
            "$operation may change only axis $axis, but axis $dim has sizes " *
            "$(size(dst, dim)) and $(size(src, dim))"))
    end
    return nothing
end

"""Copy the leading part of `src` along `axis` into the shorter `dst`."""
function _copy_axis_prefix!(dst::AbstractArray, src::AbstractArray, axis::Int)
    _validate_axis_prefix_shapes(dst, src, axis, :truncate)
    size(dst, axis) <= size(src, axis) || throw(ArgumentError(
        "truncate destination axis $axis has length $(size(dst, axis)), larger than " *
        "source length $(size(src, axis))"))
    indices = ntuple(dim -> dim == axis ? (1:size(dst, axis)) : Colon(), ndims(src))
    @views dst .= src[indices...]
    return dst
end

"""Zero `dst`, then copy `src` into its leading entries along `axis`."""
function _zero_pad_axis_prefix!(dst::AbstractArray, src::AbstractArray, axis::Int)
    _validate_axis_prefix_shapes(dst, src, axis, :zero_pad)
    size(dst, axis) >= size(src, axis) || throw(ArgumentError(
        "zero-pad destination axis $axis has length $(size(dst, axis)), smaller than " *
        "source length $(size(src, axis))"))
    fill!(dst, zero(eltype(dst)))
    indices = ntuple(dim -> dim == axis ? (1:size(src, axis)) : Colon(), ndims(dst))
    @views dst[indices...] .= src
    return dst
end

"""
Truncate or zero-pad `src` into `dst` along `axis`, whichever the lengths call
for — a Chebyshev coefficient axis meeting an upscaled grid (pad) or a
downscaled one (truncate), as the CPU chain does with `min(coeff, grid)` modes.
"""
_resize_axis_prefix!(dst::AbstractArray, src::AbstractArray, axis::Int) =
    size(dst, axis) < size(src, axis) ? _copy_axis_prefix!(dst, src, axis) :
                                        _zero_pad_axis_prefix!(dst, src, axis)

"""
    layout_coefficient_shape(bases, dtype) -> Tuple

Canonical (unscaled) coefficient shape implied by `bases` and `dtype`. This is
the rule `Domain`'s `coefficient_shape` reports and every buffer is sized from.
"""
function layout_coefficient_shape(bases, dtype::Type)
    grid = ntuple(i -> bases[i].meta.size, length(bases))
    _, coeff_shape, _ = forward_layout(bases, grid, dtype)
    return coeff_shape
end

# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

"""
    backward_axis_op(basis, in_len, grid_len, in_complex) -> AxisOp

The inverse stage for one axis: `in_len` is the stored spectral length and
`grid_len` the target (possibly scaled) grid length.

The `RealFourier` classification order matters and is shared by both backends:
test the direct half-spectrum FIRST (a pure shape heuristic misreads `N = 1` and
`N = 2`, where `rfft_len(N) == N`), then the BASE half-spectrum of a scaled
field, and only then fall through to a complex (C2C) spectrum. Guessing wrong
here stores a wrong-length or wrong-typed array instead of raising.

A spectrum that does not fit the grid is resampled onto it — never inverted at its
stored length, which produced a grid of the wrong shape on every Fourier axis but
the first. A base half-spectrum becomes `:irfft_resampled`; a C2C spectrum of any
other length becomes `:ifft_resampled`, from exactly that length (a full spectrum
has one bin per point it was formed on, so this is unambiguous). The index map is
`spectral_resample_ranges`.
"""
@inline function backward_axis_op(basis, in_len::Int, grid_len::Int, in_complex::Bool)
    if isa(basis, RealFourier)
        base_len = basis.meta.size
        if in_len == rfft_len(grid_len)
            return AxisOp(:irfft, grid_len, false)
        elseif in_len == rfft_len(base_len)
            # Scaled field: zero-pad (finer grid) or truncate (coarser grid) the
            # base half-spectrum to the grid's half-spectrum length.
            return AxisOp(:irfft_resampled, grid_len, false, rfft_len(grid_len))
        else
            # Complex spectrum at full length (e.g. the axis was transformed C2C
            # because it was not the first Fourier axis).
            return _c2c_backward_axis_op(in_len, grid_len)
        end
    elseif isa(basis, ComplexFourier)
        return _c2c_backward_axis_op(in_len, grid_len)
    elseif isa(basis, ChebyshevT)
        # Zero-pads (or truncates) the coefficient axis to the grid length.
        return AxisOp(:dct1_backward, grid_len, in_complex)
    else
        return AxisOp(:none, in_len, in_complex)
    end
end

@inline _c2c_backward_axis_op(in_len::Int, grid_len::Int) =
    in_len == grid_len ? AxisOp(:ifft, in_len, true) :
                         AxisOp(:ifft_resampled, grid_len, true, grid_len)

"""
    backward_axis_op(basis, in_len, grid_len, in_complex, uses_rfft) -> AxisOp

Canonical inverse classification when the forward plan is known.  A
`RealFourier` basis does not by itself imply an RFFT: complex-valued fields and
later Fourier stages use a full C2C transform.  That distinction cannot be
recovered from array shape for one- and two-point axes because
`rfft_len(grid_len) == grid_len` there.

Knowing the axis is an RFFT axis also makes an unmatched half-spectrum an error
rather than a guess. Its length is `rfft_len` of the grid it was formed on: the
target grid, the basis size, or a third grid whose parity the length does not
record (`2L - 2` and `2L - 1` points share `rfft_len`). A full-length spectrum
(`in_len == grid_len`, coefficients written directly) keeps its C2C inverse.
"""
@inline function backward_axis_op(basis, in_len::Int, grid_len::Int,
                                  in_complex::Bool, uses_rfft::Bool)
    if isa(basis, RealFourier)
        uses_rfft || return _c2c_backward_axis_op(in_len, grid_len)
        base_len = basis.meta.size
        in_len == rfft_len(grid_len) || in_len == rfft_len(base_len) ||
            in_len == grid_len || _unresolvable_half_spectrum(in_len, grid_len, base_len)
    end
    return backward_axis_op(basis, in_len, grid_len, in_complex)
end

@noinline _unresolvable_half_spectrum(in_len::Int, grid_len::Int, base_len::Int) =
    throw(DimensionMismatch(
        "a RealFourier half-spectrum of length $in_len fits neither the $grid_len-point " *
        "grid (length $(rfft_len(grid_len))) nor the $base_len-point basis (length " *
        "$(rfft_len(base_len))); it was formed on a grid of $(2in_len - 2) or " *
        "$(2in_len - 1) points, which the stored length cannot tell apart. Transform " *
        "back to grid space at the scales the coefficients were formed on, then " *
        "change scales with set_scales!."))

# ---------------------------------------------------------------------------
# Spectral resampling — a backward transform into a grid of another resolution
# ---------------------------------------------------------------------------

"""
    spectral_resample_ranges(n_src, n_dst) -> NamedTuple of UnitRanges

Index map carrying a full FFT spectrum formed on `n_src` uniform points to the
spectrum the same field has on `n_dst` points, as `(pos_src, pos_dst)`,
`(neg_src, neg_dst)` and `(fold_src, fold_dst)` slices (empty when unused). This is
the full-spectrum convention used for backward transforms and grid resizing. The CPU grid resample
`resample_1d!` follows it.

- Upsampling from even `n_src` DROPS the source Nyquist bin: it stands for both
  `±n_src/2`, and copying it to one side breaks Hermitian symmetry.
- Downsampling to even `n_dst` FOLDS the `-n_dst/2` image into the new Nyquist
  bin (`fold_*`, accumulated onto the positive copy) — the value the truncated
  series takes on the coarser grid. Odd lengths have no Nyquist bin.

Values must also be scaled by `n_dst/n_src` (unnormalised FFTs: the inverse
divides by the new length). A half spectrum uses only the positive slice; see
`half_spectrum_resample`.
"""
@inline function spectral_resample_ranges(n_src::Int, n_dst::Int)
    if n_dst == n_src
        return (pos_src=1:n_src, pos_dst=1:n_dst, neg_src=1:0, neg_dst=1:0,
                fold_src=1:0, fold_dst=1:0)
    elseif n_dst > n_src
        h = div(n_src, 2)
        npos = iseven(n_src) ? h : h + 1
        nneg = n_src - h - 1
        return (pos_src=1:npos, pos_dst=1:npos,
                neg_src=(n_src - nneg + 1):n_src, neg_dst=(n_dst - nneg + 1):n_dst,
                fold_src=1:0, fold_dst=1:0)
    else
        h = div(n_dst, 2)
        nneg = n_dst - h - 1
        fold = iseven(n_dst)
        return (pos_src=1:(h + 1), pos_dst=1:(h + 1),
                neg_src=(n_src - nneg + 1):n_src, neg_dst=(n_dst - nneg + 1):n_dst,
                fold_src=fold ? ((n_src - h + 1):(n_src - h + 1)) : (1:0),
                fold_dst=fold ? ((h + 1):(h + 1)) : (1:0))
    end
end

"""
    half_spectrum_resample(n_src, n_dst) -> (ncopy, nyquist)

The rfft-axis form of `spectral_resample_ranges`: copy the leading `ncopy` bins of
the half spectrum of an `n_src`-point signal into that of an `n_dst`-point one.
`nyquist > 0` (downsampling to an even `n_dst`) is the new Nyquist bin, which still
needs its `-n_dst/2` image added by `_fold_half_spectrum_nyquist!`.
"""
@inline function half_spectrum_resample(n_src::Int, n_dst::Int)
    n_dst == n_src && return (rfft_len(n_src), 0)
    ncopy = length(spectral_resample_ranges(n_src, n_dst).pos_src)
    return (ncopy, n_dst < n_src && iseven(n_dst) ? rfft_len(n_dst) : 0)
end

"""
    _resample_spectrum_axis!(dst, src, axis, n_src, n_dst, half) -> dst

Write into `dst` the spectrum that `src` (formed on `n_src` points along `axis`)
has on `n_dst` points — `spectral_resample_ranges`, or `half_spectrum_resample`
for an rfft (`half`) axis — scaled by `n_dst/n_src`; other axes are copied as
they are. A `half` axis still owes its Nyquist fold. Broadcasts over views only,
so device arrays work too.
"""
function _resample_spectrum_axis!(dst::AbstractArray, src::AbstractArray, axis::Int,
                                  n_src::Int, n_dst::Int, half::Bool)
    fill!(dst, zero(eltype(dst)))
    s = real(eltype(dst))(n_dst / n_src)
    if half
        ncopy, _ = half_spectrum_resample(n_src, n_dst)
        _add_axis_slab!(dst, 1:ncopy, src, 1:ncopy, axis, s)
    else
        r = spectral_resample_ranges(n_src, n_dst)
        _add_axis_slab!(dst, r.pos_dst, src, r.pos_src, axis, s)
        _add_axis_slab!(dst, r.neg_dst, src, r.neg_src, axis, s)
        _add_axis_slab!(dst, r.fold_dst, src, r.fold_src, axis, s)
    end
    return dst
end

# Accumulates: the fold slice lands on the bin the positive copy already wrote.
@inline function _add_axis_slab!(dst::AbstractArray, dst_range::UnitRange{Int},
                                 src::AbstractArray, src_range::UnitRange{Int},
                                 axis::Int, s)
    isempty(dst_range) && return dst
    view(_axis_as_middle(dst, axis), :, dst_range, :) .+=
        view(_axis_as_middle(src, axis), :, src_range, :) .* s
    return dst
end

# `A` as a (before, axis, after) array sharing its memory: a slab along a runtime
# `axis` is then one concrete view type, so the copies compile once instead of
# dispatching per call (`selectdim` builds a differently typed view per axis).
@inline function _axis_as_middle(A::AbstractArray, axis::Int)
    before = 1
    for d in 1:(axis - 1)
        before *= size(A, d)
    end
    after = 1
    for d in (axis + 1):ndims(A)
        after *= size(A, d)
    end
    return reshape(A, before, size(A, axis), after)
end

"""
    _fold_half_spectrum_nyquist!(data, axis, bin) -> data

Finish a half-spectrum downsample to an even grid (`half_spectrum_resample`): add
the `-n/2` image to the new Nyquist `bin`. For a real field that image is the
bin's own conjugate WHILE every other axis holds a real-linear image of the data
(grid values or Chebyshev coefficients) — true at the inverse-rfft stage of both
backends, which invert the other Fourier axes first. In a full multi-axis spectrum
the image sits at mirrored wavenumbers instead. `bin == 0` is a no-op.
"""
function _fold_half_spectrum_nyquist!(data::AbstractArray, axis::Int, bin::Int)
    bin > 0 || return data
    plane = view(_axis_as_middle(data, axis), :, bin:bin, :)
    plane .= plane .+ conj.(plane)
    return data
end

"""
    backward_resample_rules(bases, coeff_shape, grid_shape, dtype)
        -> (rules, grid_coeff_shape, nyquist)

Whole-field form of `backward_axis_op` for a backend that resamples every axis of
the stored spectrum up front and then inverts at `grid_shape`: per-axis
`(n_src, n_dst, mode)` rules for `_resample_fourier_spectrum!` (`mode` is `:copy`,
`:c2c` or `:half`), the coefficient shape the grid implies (`forward_layout`), and
`nyquist = (axis, bin)` — the half-spectrum fold that must wait until the other
Fourier axes are inverted (`(0, 0)` if none). RFFT axes are the forward plan's, so
this classifies exactly as the CPU chain does stage by stage.
"""
function backward_resample_rules(bases, coeff_shape::Tuple, grid_shape::Tuple, dtype::Type)
    n = length(bases)
    (length(coeff_shape) == n && length(grid_shape) == n) || throw(DimensionMismatch(
        "$n bases, coefficient shape $coeff_shape and grid shape $grid_shape disagree"))
    fwd_ops, grid_coeff_shape, _ = forward_layout(bases, grid_shape, dtype)
    rules = ntuple(n) do a
        b = bases[a]
        (isa(b, RealFourier) || isa(b, ComplexFourier)) ||
            return (coeff_shape[a], coeff_shape[a], :copy)
        op = backward_axis_op(b, coeff_shape[a], grid_shape[a], true,
                              fwd_ops[a].op === :rfft)
        op.op === :irfft_resampled && return (b.meta.size, grid_shape[a], :half)
        op.op === :ifft_resampled && return (coeff_shape[a], grid_shape[a], :c2c)
        return (coeff_shape[a], coeff_shape[a], :copy)
    end
    nyquist = (0, 0)
    for a in 1:n
        if rules[a][3] === :half
            bin = half_spectrum_resample(rules[a][1], rules[a][2])[2]
            bin > 0 && (nyquist = (a, bin))
        end
    end
    return rules, grid_coeff_shape, nyquist
end

"""
    _resample_fourier_spectrum!(dst, src, rules) -> dst

All axes of `_resample_spectrum_axis!` in one tensor-product pass, driven by
`backward_resample_rules`: the same index maps and `n_dst/n_src` scaling, and the
half-spectrum Nyquist fold likewise left to the caller. Broadcasts over views only.
"""
function _resample_fourier_spectrum!(dst::AbstractArray, src::AbstractArray, rules)
    N = ndims(dst)
    (ndims(src) == N && length(rules) == N) || throw(DimensionMismatch(
        "resampling a $(ndims(src))D spectrum into $(N)D with $(length(rules)) axis rules"))
    slices = Vector{Vector{NTuple{2,UnitRange{Int}}}}(undef, N)
    scale = 1.0
    for a in 1:N
        n_src, n_dst, mode = rules[a]
        src_len, dst_len = mode === :half ? (rfft_len(n_src), rfft_len(n_dst)) :
                           (n_src, n_dst)
        (size(src, a) == src_len && size(dst, a) == dst_len) || throw(DimensionMismatch(
            "axis $a ($mode, $n_src → $n_dst points) needs lengths $src_len → " *
            "$dst_len, got $(size(src, a)) → $(size(dst, a))"))
        if mode === :copy
            slices[a] = [(1:src_len, 1:dst_len)]
            continue
        end
        scale *= n_dst / n_src
        if mode === :half
            ncopy = half_spectrum_resample(n_src, n_dst)[1]
            slices[a] = [(1:ncopy, 1:ncopy)]
        else
            r = spectral_resample_ranges(n_src, n_dst)
            slices[a] = [p for p in ((r.pos_src, r.pos_dst), (r.neg_src, r.neg_dst),
                                     (r.fold_src, r.fold_dst)) if !isempty(p[1])]
        end
    end
    fill!(dst, zero(eltype(dst)))
    s = real(eltype(dst))(scale)
    # Accumulate: fold slices overlap the positive copy on the new Nyquist bins.
    for combo in Iterators.product(slices...)
        @views dst[ntuple(a -> combo[a][2], N)...] .+= src[ntuple(a -> combo[a][1], N)...] .* s
    end
    return dst
end

"""
    scaled_chebyshev_axis(bases, grid_shape) -> Union{Nothing, Tuple{Int,Int,Int}}

First Chebyshev axis whose grid length differs from its basis size, as
`(axis, grid_len, basis_len)`, or `nothing`. A backend without a Chebyshev
truncation/zero-pad stage must refuse such a field rather than return the
untruncated spectrum — see `forward_axis_op`.
"""
function scaled_chebyshev_axis(bases, grid_shape::Tuple)
    for (ax, b) in enumerate(bases)
        if isa(b, ChebyshevT) && ax <= length(grid_shape) && grid_shape[ax] != b.meta.size
            return (ax, grid_shape[ax], b.meta.size)
        end
    end
    return nothing
end
