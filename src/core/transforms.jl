"""
Spectral transform classes with PencilFFTs integration

This module provides spectral transforms for various bases:
- Fourier (FFT/RFFT via FFTW and PencilFFTs)
- Chebyshev (DCT-I via FFTW)
- Legendre (Gauss-Legendre quadrature)

## File Organization

This module is split into multiple files for maintainability:
- layout.jl: THE layout rules (axis ops, shapes, eltypes) both backends share
- types.jl: Core type definitions + in-place dispatch protocol
- planning.jl: Transform planning (builds 1D FFTW plans)
- gpu.jl: Serial CPU + GPU dispatch (`forward_transform!`)
- fourier.jl: Fourier transform execution (in-place and legacy)
- chebyshev.jl: Chebyshev transform execution (in-place and legacy)
- legendre.jl: Legendre transform execution
- transposable.jl: TransposableField transform planning
"""

# PencilFFTs, FFTW, LinearAlgebra, SparseArrays already in Tarang.jl

# Include all the split files
include("transforms/layout.jl")
include("transforms/types.jl")
include("transforms/planning.jl")
include("transforms/gpu.jl")
include("transforms/fourier.jl")
include("transforms/chebyshev.jl")
include("transforms/fft_dct.jl")
include("transforms/legendre.jl")
include("transforms/transposable.jl")
include("transforms/grouped.jl")

# ============================================================================
# Exports
# ============================================================================

# Export abstract type
export Transform

# Export transform types
export PencilFFTTransform, FourierTransform, ChebyshevTransform, LegendreTransform

# Export main transform planning function
export plan_transforms!

# TransposableField transform planning
export plan_transposable_transforms!, setup_distributed_transforms!

# Export forward/backward transform functions
export forward_transform!, backward_transform!

# Export setup functions
export setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!
export setup_fftw_transform!
export setup_chebyshev_transform!, setup_legendre_transform!

# Export Legendre quadrature functions
export compute_legendre_quadrature
export evaluate_legendre_and_derivative
export build_legendre_polynomials

