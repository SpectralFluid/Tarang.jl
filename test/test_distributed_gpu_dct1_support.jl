"""
Tests for axis_kinds and distributed_gpu_supported predicate.

axis_kinds classifies each basis in a tuple as :real_fourier, :complex_fourier,
or :chebyshev. distributed_gpu_supported decides whether a 3D bases tuple is
eligible for the distributed GPU Chebyshev transform path:
  - must be 3D
  - must have at least one ChebyshevT axis
  - every RealFourier axis must be on dim 1 (the framework convention)
  - RealFourier on dim 1 must not be combined with a Fourier transverse axis
    (the backward Hermitian expansion cannot place conjugate partners at the
    flipped transverse wavenumber — such layouts are rejected explicitly)
"""

using Test
using Tarang
using Tarang: axis_kinds, distributed_gpu_supported

@testset "axis_kinds + distributed_gpu_supported" begin
    # Build three real basis objects using the canonical constructor API.
    # Each basis needs a Coordinate from CartesianCoordinates.
    coords = CartesianCoordinates("x", "y", "z")

    rf = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    cf = ComplexFourier(coords["y"]; size=8, bounds=(0.0, 2π))
    cb = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))

    # axis_kinds returns a tuple of symbols
    @test axis_kinds((rf, cf, cb)) == (:real_fourier, :complex_fourier, :chebyshev)

    # Not supported: RealFourier on dim 1 with a Fourier transverse axis. The
    # forward pipeline completes, but the backward Hermitian expansion needs
    # conjugate partners at the FLIPPED transverse wavenumber and hard-errors
    # (guard in distributed_backward_dct!) — the predicate rejects the layout
    # up front so it fails explicitly before the first backward transform.
    @test distributed_gpu_supported((rf, cf, cb)) == false

    # Supported: RealFourier on dim 1 with only Chebyshev transverse axes
    @test distributed_gpu_supported((rf, cb, cb)) == true

    # Supported: no RealFourier at all, has ChebyshevT, 3D
    @test distributed_gpu_supported((cf, cf, cb)) == true

    # Not yet supported: all-Chebyshev Float64 fields have real coefficient
    # storage, while the distributed DCT driver currently emits complex buffers.
    @test distributed_gpu_supported((cb, cb, cb)) == false

    # Not supported: RealFourier on dim 2 (must be dim 1)
    @test distributed_gpu_supported((cf, rf, cb)) == false

    # Not supported: RealFourier on dim 3 (must be dim 1)
    @test distributed_gpu_supported((cb, cb, rf)) == false

    # Not supported: no ChebyshevT axis
    @test distributed_gpu_supported((rf, cf, cf)) == false

    # Not supported: only 2D
    @test distributed_gpu_supported((rf, cb)) == false

    @testset "GPU+MPI domain validation matches the transform predicate" begin
        @test Tarang.validate_mpi_fourier_only(
            (rf, cb, cb), 2; use_pencil_arrays=false,
        )
        @test_throws ErrorException Tarang.validate_mpi_fourier_only(
            (rf, cf, cb), 2; use_pencil_arrays=false,
        )
        @test_throws ErrorException Tarang.validate_mpi_fourier_only(
            (cb, cb, cb), 2; use_pencil_arrays=false,
        )
    end
end

# ---------------------------------------------------------------------------
# Helper: build a length-N vector that is a valid Hermitian (real-signal) spectrum.
# DC (and Nyquist for even N) are real; positive-freq entries are arbitrary complex;
# negative-freq entries are their conjugates.  The index convention matches the
# expansion: X[N-k+2] = conj(X[k]) for k = 2 … (N - div(N,2)).
# ---------------------------------------------------------------------------
function _make_hermitian(N)
    full = zeros(ComplexF64, N)
    full[1] = real(randn())                         # DC real
    kmax = iseven(N) ? div(N,2)-1 : div(N,2)        # last non-Nyquist positive freq
    for k in 2:(kmax+1)
        full[k] = randn(ComplexF64)
        full[N-k+2] = conj(full[k])
    end
    if iseven(N)
        full[div(N,2)+1] = real(randn())            # Nyquist real
    end
    return full
end

@testset "hermitian half->full expansion" begin
    using Tarang: _hermitian_full_from_half
    for N in (8, 7, 16, 9, 4, 5)
        full = _make_hermitian(N)
        half = full[1:div(N,2)+1]
        @test _hermitian_full_from_half(half, N) ≈ full
    end
end

# ── The refusal REASON must be derived from the predicate, not restated ──────
#
# `distributed_gpu_supported` is now defined AS
# `_distributed_gpu_dct_unsupported_reason(bases) === nothing`, because the two
# had drifted: the hand-written refusal text said "every RealFourier axis on dim
# 1", which `(RealFourier, ComplexFourier, ChebyshevT)` satisfies — yet that
# layout is refused (the backward Hermitian expansion cannot place the dim-1
# half-spectrum's conjugate partners at the flipped transverse wavenumber). A
# reader who followed the advice hit the identical message again.
@testset "refusal reason agrees with the predicate and is actionable" begin
    coords = CartesianCoordinates("x", "y", "z")
    names = ("x", "y", "z")
    mk(i, kind) = kind === :RF ? RealFourier(coords[names[i]]; size=8, bounds=(0.0, 2π)) :
                  kind === :CF ? ComplexFourier(coords[names[i]]; size=8, bounds=(0.0, 2π)) :
                                 ChebyshevT(coords[names[i]]; size=8, bounds=(-1.0, 1.0))

    # An INDEPENDENT restatement of the documented rule, so a refactor of the
    # reason function cannot quietly change which layouts are accepted.
    function expected_supported(kinds)
        length(kinds) == 3 || return false
        any(==(:CHEB), kinds) || return false
        any(k -> k in (:RF, :CF), kinds) || return false
        any(i -> kinds[i] === :RF, 2:3) && return false
        kinds[1] === :RF && any(k -> k in (:RF, :CF), kinds[2:3]) && return false
        return true
    end

    for k1 in (:RF, :CF, :CHEB), k2 in (:RF, :CF, :CHEB), k3 in (:RF, :CF, :CHEB)
        kinds = (k1, k2, k3)
        bases = ntuple(i -> mk(i, kinds[i]), 3)
        reason = Tarang.distributed_gpu_unsupported_reason(bases)
        want = expected_supported(kinds)
        @test distributed_gpu_supported(bases) == want
        @test (reason === nothing) == want
        want || @test !isempty(reason)
    end

    # The layout that exposed the drift: it obeys "every RealFourier axis on dim
    # 1" and is still refused, so the reason must NOT tell the reader to move a
    # RealFourier axis to dim 1.
    drift = (mk(1, :RF), mk(2, :CF), mk(3, :CHEB))
    @test !distributed_gpu_supported(drift)
    drift_reason = Tarang.distributed_gpu_unsupported_reason(drift)
    @test occursin("dim 1 is RealFourier", drift_reason)
    @test !occursin("move it to dim 1", drift_reason)

    # "move it to dim 1" is only ever offered when doing so actually helps, i.e.
    # when the RealFourier axis is the only Fourier axis.
    @test occursin("move it to dim 1",
                   Tarang.distributed_gpu_unsupported_reason((mk(1, :CHEB), mk(2, :RF), mk(3, :CHEB))))
    @test !occursin("move it to dim 1",
                    Tarang.distributed_gpu_unsupported_reason((mk(1, :CF), mk(2, :RF), mk(3, :CHEB))))

    # Following the advice must reach a SUPPORTED layout, not another refusal.
    @test distributed_gpu_supported((mk(1, :CF), mk(2, :CF), mk(3, :CHEB)))
    @test distributed_gpu_supported((mk(1, :RF), mk(2, :CHEB), mk(3, :CHEB)))

    # Non-3D and unsupported basis families report their own specific reason.
    @test occursin("3D", Tarang.distributed_gpu_unsupported_reason((mk(1, :RF), mk(2, :CHEB))))
    @test occursin("Chebyshev", Tarang.distributed_gpu_unsupported_reason((mk(1, :CF), mk(2, :CF), mk(3, :CF))))
end

# ── TransposableField's Chebyshev transform is the CANONICAL one ─────────────
#
# `dct_in_dim!` (the per-axis DCT of the TransposableField path) used to be plain
# REDFT00 without the odd-degree sign flip that `_chebyshev_forward` applies.
# Tarang's grid is -cos(πj/(N-1)), the reverse of REDFT00's, so its coefficients
# were those of f(-x): u = T0+T1+T2 came out as [1,-1,1,0]. Forward→backward
# round trips hid it (both directions shared the error) while every consumer of
# the stored coefficients — tau/BC rows, interpolation, output, solver matrices —
# saw a mirrored field. Pinned against an analytic field AND the serial transform.
_cheb_flip(n) = [iseven(k) ? 1.0 : -1.0 for k in 0:n-1]
function _canonical_cheb_forward(v::AbstractVector{<:Real})
    n = length(v)
    c = Tarang.FFTW.r2r(Vector{Float64}(v), Tarang.FFTW.REDFT00) ./ (n - 1)
    c[1] /= 2
    c[end] /= 2
    return c .* _cheb_flip(n)
end
_canonical_cheb_forward(v::AbstractVector{<:Complex}) =
    complex.(_canonical_cheb_forward(real.(v)), _canonical_cheb_forward(imag.(v)))

@testset "TransposableField dct_in_dim! == canonical Chebyshev transform" begin
    @testset "analytic T0+T1+T2 (N=$N)" for N in (5, 8, 9, 16)
        x = -cos.(π .* (0:N-1) ./ (N - 1))          # Tarang's Gauss-Lobatto grid
        u = 1 .+ x .+ (2 .* x .^ 2 .- 1)             # T0 + T1 + T2
        c = copy(u)
        Tarang.dct_in_dim!(c, 1, :forward, CPU())
        @test c ≈ [1.0, 1.0, 1.0, zeros(N - 3)...] atol = 1e-12
        Tarang.dct_in_dim!(c, 1, :backward, CPU())
        @test c ≈ u atol = 1e-12                     # the grid itself, not its mirror
    end

    @testset "$(length(shape))D $shape $T" for shape in ((9,), (8,), (7, 6), (6, 9), (5, 8, 7), (8, 5, 6)),
                                               T in (Float64, ComplexF64)
        names = ("x", "y", "z")[1:length(shape)]
        coords = CartesianCoordinates(names...)
        dist = Distributor(coords; dtype=T)
        bases = ntuple(i -> ChebyshevT(coords[names[i]]; size=shape[i], bounds=(-1.0, 1.0)),
                       length(shape))
        field = ScalarField(dist, "dct_in_dim_ref", bases, T)
        grid = randn(T, shape...)
        Tarang.get_grid_data(field) .= grid
        field.current_layout = :g
        forward_transform!(field)
        serial = copy(Tarang.get_coeff_data(field))

        # Every axis through dct_in_dim! == the serial ChebyshevTransform chain.
        all_axes = copy(grid)
        for d in eachindex(shape)
            Tarang.dct_in_dim!(all_axes, d, :forward, CPU())
        end
        @test all_axes ≈ serial atol = 1e-12
        # Canonical coefficients go back to the grid (they used to land on the
        # axis-reversed grid).
        back = copy(serial)
        for d in reverse(eachindex(shape))
            Tarang.dct_in_dim!(back, d, :backward, CPU())
        end
        @test back ≈ grid atol = 1e-12
        # Single axes against the documented formula; backward inverts forward.
        for d in eachindex(shape)
            one_axis = copy(grid)
            Tarang.dct_in_dim!(one_axis, d, :forward, CPU())
            @test one_axis ≈ mapslices(_canonical_cheb_forward, grid; dims=d) atol = 1e-12
            Tarang.dct_in_dim!(one_axis, d, :backward, CPU())
            @test one_axis ≈ grid atol = 1e-12
        end
        close(dist)
    end
end

# ── Routing and rejection rules, as pure functions ──────────────────────────
#
# Every multi-rank GPU field carries TransposableFieldStorage, so the dedicated
# multi-GPU DCT-I backend must be chosen BEFORE the TransposableField early
# return in forward_transform!/backward_transform! — and identically on every
# rank, since both paths enter collectives.
struct _RoutingProbeDevice end
struct _RoutingProbeArchitecture <: Tarang.AbstractArchitecture
    calls::Vector{Symbol}
end
Tarang._distributed_gpu_transform!(arch::_RoutingProbeArchitecture, field::ScalarField,
                                   direction::Symbol) = (push!(arch.calls, direction); true)

@testset "multi-GPU DCT-I routing and rejection rules" begin
    coords = CartesianCoordinates("x", "y", "z")
    names = ("x", "y", "z")
    mk(i, kind) = kind === :RF ? RealFourier(coords[names[i]]; size=8, bounds=(0.0, 2π)) :
                  kind === :CF ? ComplexFourier(coords[names[i]]; size=8, bounds=(0.0, 2π)) :
                                 ChebyshevT(coords[names[i]]; size=8, bounds=(-1.0, 1.0))
    gpu = Tarang.GPU{_RoutingProbeDevice}(_RoutingProbeDevice())

    @testset "routing predicate" begin
        for k1 in (:RF, :CF, :CHEB), k2 in (:RF, :CF, :CHEB), k3 in (:RF, :CF, :CHEB)
            bases = (mk(1, k1), mk(2, k2), mk(3, k3))
            supported = distributed_gpu_supported(bases)
            @test Tarang._uses_distributed_gpu_dct(gpu, 2, bases) == supported
            @test Tarang._uses_distributed_gpu_dct(gpu, 4, bases) == supported
            @test !Tarang._uses_distributed_gpu_dct(gpu, 1, bases)      # one rank: no transposes
            @test !Tarang._uses_distributed_gpu_dct(CPU(), 4, bases)    # CPU: PencilArrays/TransposableField
        end
        @test !Tarang._uses_distributed_gpu_dct(gpu, 2, (mk(1, :RF), mk(2, :CHEB)))
    end

    @testset "plan axis kinds follow the field dtype" begin
        rf = (mk(1, :RF), mk(2, :CHEB), mk(3, :CHEB))
        @test Tarang._distributed_gpu_dct_axis_kinds(rf, Float64) == (:real_fourier, :chebyshev, :chebyshev)
        @test Tarang._distributed_gpu_dct_axis_kinds(rf, Float32) == (:real_fourier, :chebyshev, :chebyshev)
        # A complex field's RealFourier axis is full-length C2C. The plan used
        # to halve it anyway (dtype-blind axis_kinds) and swap in a wrongly
        # shaped coefficient buffer: round-trip error 0.44-0.57.
        @test Tarang._distributed_gpu_dct_axis_kinds(rf, ComplexF64) == (:complex_fourier, :chebyshev, :chebyshev)
        @test Tarang.axis_kinds(rf) == (:real_fourier, :chebyshev, :chebyshev)  # basis-only classification

        # The coefficient shape the plan writes is the one Tarang allocates, for
        # every supported layout and dtype — this is what the plan cache keys on.
        for k1 in (:RF, :CF, :CHEB), k2 in (:RF, :CF, :CHEB), k3 in (:RF, :CF, :CHEB),
            T in (Float64, Float32, ComplexF64)
            bases = (mk(1, k1), mk(2, k2), mk(3, k3))
            if distributed_gpu_supported(bases)
                kinds = Tarang._distributed_gpu_dct_axis_kinds(bases, T)
                plan_coeff = (kinds[1] === :real_fourier ? Tarang.rfft_len(8) : 8, 8, 8)
                @test plan_coeff == Tarang.layout_coefficient_shape(bases, T)
                @test all(i -> (kinds[i] === :chebyshev) == (bases[i] isa ChebyshevT), 1:3)
            else
                @test_throws ArgumentError Tarang._distributed_gpu_dct_axis_kinds(bases, T)
            end
        end
    end

    @testset "TransposableField buffer rule rejects rfft-halved axes" begin
        reason = Tarang._transposable_field_unsupported_reason
        rf = (mk(1, :RF), mk(2, :CHEB), mk(3, :CHEB))
        @test occursin("RealFourier", reason(rf, Float64))
        @test occursin("half spectrum", reason(rf, Float32))
        @test reason(rf, ComplexF64) === nothing        # full-length C2C: representable
        @test reason((mk(1, :CF), mk(2, :CF), mk(3, :CHEB)), Float64) === nothing
        @test reason((mk(1, :CF), mk(2, :CF), mk(3, :CF)), Float64) === nothing

        # Constructor and transforms refuse BEFORE any collective. dist.size is
        # faked to 2 after construction: if validation did not fire first, the
        # constructor would go on to split communicators for a 2-rank topology.
        dist = Distributor(coords; dtype=Float64, comm=Tarang.MPI.COMM_SELF, use_pencil_arrays=false)
        rf_field = ScalarField(dist, "rf_transposable", rf, Float64)
        cf_field = ScalarField(dist, "cf_transposable", (mk(1, :CF), mk(2, :CHEB), mk(3, :CHEB)), Float64)
        tf = TransposableField(cf_field)   # one rank: no layout validation, nothing to transpose
        dist.size = 2
        try
            err = try
                TransposableField(rf_field)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("cannot transform field \"rf_transposable\"", sprint(showerror, err))
            @test occursin("half spectrum", sprint(showerror, err))

            # A cached workspace is repointed to other fields (transpose_workspace!):
            # the transform itself must refuse the rfft-halved layout too.
            tf.field = rf_field
            @test_throws ErrorException distributed_forward_transform!(tf)
            rf_field.current_layout = :c
            @test_throws ErrorException distributed_backward_transform!(tf)
        finally
            dist.size = 1
            close(tf)
        end
        close(dist)
    end

    @testset "forward/backward_transform! consult the device hook before the TransposableField path" begin
        dist = Distributor(coords; dtype=Float64, comm=Tarang.MPI.COMM_SELF, use_pencil_arrays=false)
        bases = (mk(1, :RF), mk(2, :CHEB), mk(3, :CHEB))
        host = ScalarField(dist, "probe_host", bases, Float64)
        storage = Tarang.TransposableFieldStorage(CPU(), Tarang.get_grid_data(host),
                                                  Tarang.get_coeff_data(host))
        field = ScalarField(dist, "probe", bases, Float64, storage)
        @test Tarang.is_transposable_storage(field)
        probe = _RoutingProbeArchitecture(Symbol[])
        dist.architecture = probe
        try
            forward_transform!(field)
            @test probe.calls == [:forward]
            @test field.current_layout === :c
            backward_transform!(field)
            @test probe.calls == [:forward, :backward]
            @test field.current_layout === :g
            # The TransposableField path was never entered.
            @test isempty(dist.transpose_workspace_cache)
        finally
            dist.architecture = CPU()
        end
        close(dist)
    end
end
