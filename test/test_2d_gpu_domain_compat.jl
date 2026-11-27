"""
Device-safety of the 2D pure-Fourier GPU algebraic-constraint refresh WITHOUT a GPU.

A pure-Fourier GPU InitialValueProblem (e.g. the 2D vorticity–streamfunction turbulence example)
builds no global matrix; it refreshes its algebraic constraints — the
streamfunction Poisson solve `ψ̂ = -q̂/k²`, the |k|² grid, the spectral
`skew(grad)` velocity — spectrally on-device every RHS. The transforms between
those steps need cuFFT, but the refresh kernels themselves are broadcasts and
reductions on coefficient arrays.

This runs those real kernels on `JLArray` (GPUArrays' CPU-backed reference GPU
array) with `allowscalar(false)`, so any scalar getindex/setindex! throws exactly
as on a CuArray — a kernel that completes here is scalar-index-free — and checks
each against a CPU reference. Runs on any machine; no GPU needed.
"""

using Test
using Random
using Tarang
using Statistics: mean

const _JL2D_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    @eval using FFTW
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping 2D GPU refresh device-safety test" err
    false
end

if _JL2D_OK
    include("strict_device_support.jl")
    const _JLA2D = JLArrays.JLArray
    # Wire JLArray into Tarang's GPU dispatch so the REAL refresh functions treat
    # it as device memory — mirrors ext/cuda/architecture.jl and ext/cuda/utils.jl.
    # Test-scoped; JLArray is used by nothing else in the suite.
    const _JL2D_ARCH = Tarang.GPU(JLArrays.JLBackend())
    Tarang.is_gpu_array(::_JLA2D) = true
    Tarang.architecture(::_JLA2D) = _JL2D_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _JLA2D(a)
    Tarang.copy_to_device(a::AbstractArray, ::_JLA2D) = _JLA2D(Array(a))
    Tarang.copy_to_device(a::_JLA2D, ::_JLA2D) = copy(a)
end

@testset "2D pure-Fourier GPU refresh device-safety (JLArray)" begin
    if !_JL2D_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)
        Random.seed!(1)

        @testset "constraint solve ψ̂ = -q̂/k² (k=0 pinned, no scalar write)" begin
            # Δψ - q = 0: L̂ = -k², input weight -1, so the refresh multiplies by
            # -1/L̂ = 1/k² with the gauge (k=0) entry of that multiplier zeroed.
            N = 8
            q = ComplexF64.(randn(N, N)) .+ im .* randn(N, N)
            k2 = abs.(randn(N, N)); k2[1, 1] = 0.0        # k=0 (mean) mode
            ref = ifelse.(k2 .> 1e-12, .-q ./ k2, zero(ComplexF64))
            neg_inv = ifelse.(k2 .> 0, complex.(1 ./ k2), zero(ComplexF64))

            dest = _JLA2D(zeros(ComplexF64, N, N))
            Tarang._constraint_single!(dest, _JLA2D(neg_inv), -1.0, _JLA2D(copy(q)),
                                       nothing)           # real function, on device
            @test Array(dest) ≈ ref rtol=1e-12
            @test Array(dest)[1, 1] == 0.0 + 0.0im        # gauge / k=0 mode exactly zero
        end

        @testset "2D k² grid (rfft first axis × full-fft second axis)" begin
            N = 8; k0 = 1.0
            kx = k0 .* collect(0:(N ÷ 2))                  # rfft: length N/2+1
            ky = k0 .* Float64.([0:(N ÷ 2 - 1); -(N ÷ 2):-1])  # fft: length N
            ref = [kx[i]^2 + ky[j]^2 for i in 1:length(kx), j in 1:length(ky)]

            ksq = _JLA2D(zeros(Float64, length(kx), length(ky)))
            Tarang.add_wavenumber_squared_contribution!(ksq, kx, 1, 2)   # real function
            Tarang.add_wavenumber_squared_contribution!(ksq, ky, 2, 2)
            @test Array(ksq) ≈ ref rtol=1e-12
        end

        @testset "Poisson-solve-into broadcast (both sign conventions)" begin
            # the shape of the constraint refresh's per-mode solve (state_utils.jl):
            #   dest .= ifelse.(k2 .> 1e-12, sign .* src ./ k2, 0)
            N = 8
            src = ComplexF64.(randn(N, N ÷ 2 + 1)) .+ im .* randn(N, N ÷ 2 + 1)
            k2 = abs.(randn(N, N ÷ 2 + 1)); k2[1, 1] = 0.0
            for was_negated in (true, false)
                s = was_negated ? -1.0 : 1.0
                ref = ifelse.(k2 .> 1e-12, s .* src ./ k2, zero(ComplexF64))
                dest = _JLA2D(similar(src)); srcd = _JLA2D(copy(src)); k2d = _JLA2D(copy(k2))
                dest .= ifelse.(k2d .> 1e-12, s .* srcd ./ k2d, zero(eltype(dest)))
                @test Array(dest) ≈ ref rtol=1e-12
                @test Array(dest)[1, 1] == 0.0 + 0.0im
            end
        end

        @testset "spectral derivative multiply (skew/grad core)" begin
            # ∂/∂x sin(3x) = 3 cos(3x) via full FFT along axis 1.
            N = 16; L = 2π; m = 3
            x = (0:N-1) .* (L / N)
            f = [sin(m * x[i]) for i in 1:N, j in 1:N]
            fhat = fft(f, 1)
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; comm=Tarang.MPI.COMM_SELF, use_pencil_arrays=false)
            xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, L))

            dhat = _JLA2D(copy(fhat))
            with_strict_device() do
                Tarang._apply_spectral_derivative_distributed!(dhat, xb, 1, 1, dist;
                                                                uses_rfft=false)
            end
            got = real.(ifft(Array(dhat), 1))
            @test got ≈ [m * cos(m * x[i]) for i in 1:N, j in 1:N] rtol=1e-10
        end

        @testset "device reductions (energy/enstrophy diagnostics)" begin
            ux = randn(16, 16); uy = randn(16, 16); q = randn(16, 16)
            E_ref = 0.5 * mean(@. ux^2 + uy^2)
            Z_ref = 0.5 * mean(q .^ 2)
            uxd = _JLA2D(ux); uyd = _JLA2D(uy); qd = _JLA2D(q)
            @test 0.5 * mean(@. uxd^2 + uyd^2) ≈ E_ref rtol=1e-12
            @test 0.5 * mean(qd .^ 2) ≈ Z_ref rtol=1e-12
        end

        @testset "2D axis-matrix application remains device-resident" begin
            M = [2.0 -1.0; 0.5 3.0]
            A = reshape(collect(1.0:8.0), 2, 4)
            Ad = _JLA2D(A)
            got = Tarang.apply_dense_along_axis(M, Ad, 1)
            @test got isa _JLA2D
            @test Array(got) ≈ M * A
        end

        @testset "low-level dense algebra refuses CPU result staging" begin
            M = [2.0 -1.0; 0.5 3.0]
            x = _JLA2D([1.0, 2.0])
            y = _JLA2D(zeros(2))
            Tarang.fast_matvec!(y, Tarang.DenseMatVec(M), x)
            @test y isa _JLA2D
            @test Array(y) ≈ M * [1.0, 2.0]
            @test_throws ArgumentError Tarang.fast_matvec!(
                zeros(2), Tarang.DenseMatVec(M), x)

            A = _JLA2D([1.0 2.0; 3.0 4.0])
            B = _JLA2D([2.0 0.0; 1.0 2.0])
            C = _JLA2D(zeros(2, 2))
            Tarang.fast_matmat!(C, Tarang.DenseDenseMatMat(), A, B)
            @test C isa _JLA2D
            @test Array(C) ≈ Array(A) * Array(B)
            @test_throws ArgumentError Tarang.fast_matmat!(
                zeros(2, 2), Tarang.DenseDenseMatMat(), A, B)
        end

        @testset "subproblem buffers refuse architecture staging" begin
            src = _JLA2D(ComplexF64[1, 2])
            dest = _JLA2D(zeros(ComplexF64, 2))
            Tarang._assign_to_buffer!(dest, src)
            @test Array(dest) == ComplexF64[1, 2]
            @test_throws ErrorException Tarang._assign_to_buffer!(
                zeros(ComplexF64, 2), src)
            @test_throws ErrorException Tarang._assign_from_buffer!(
                src, zeros(ComplexF64, 2))
        end

        @testset "2D Hilbert multipliers remain device-resident" begin
            coords = Tarang.CartesianCoordinates("x", "y")
            xb = Tarang.RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            yb = Tarang.RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
            coeff = ComplexF64.(randn(5, 8), randn(5, 8))
            ref = copy(coeff)
            got = _JLA2D(copy(coeff))

            Tarang._apply_hilbert_spectral!(ref, (xb, yb))
            Tarang._apply_hilbert_spectral!(got, (xb, yb))

            @test got isa _JLA2D
            @test Array(got) ≈ ref
        end

        @testset "reproducible random fill is generated on-device" begin
            a = _JLA2D(zeros(Float64, 4, 3))
            b = similar(a)
            Tarang._fill_random_reproducible_device!(
                Tarang.architecture(a), a, 42, (0, 0), size(a), "normal", 0.5)
            Tarang._fill_random_reproducible_device!(
                Tarang.architecture(b), b, 42, (0, 0), size(b), "normal", 0.5)
            @test Array(a) == Array(b)
            @test all(isfinite, Array(a))
            @test any(!iszero, Array(a))
        end

        @testset "unsupported GPU resampling fails instead of staging through CPU" begin
            old = _JLA2D(ones(Float64, 4, 4))
            new = _JLA2D(zeros(Float64, 4, 4, 1))
            @test_throws ErrorException Tarang.resample_grid_data!(
                new, old, size(old), size(new))
        end

        GPUArrays.allowscalar(true)   # restore for later tests in the process
    end
end
