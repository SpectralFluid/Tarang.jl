"""
    Derivative evaluation functions

This file contains all differentiation implementations including:
- Gradient, divergence, and differentiate evaluators
- Fourier derivative functions (distributed and local, CPU and GPU)
- Chebyshev derivative functions
- Legendre derivative functions
- Matrix application helpers
"""

# LinearAlgebra, SparseArrays, FFTW already in Tarang.jl


# Runtime map:
#   eval.jl         — gradient/divergence evaluators and Differentiate dispatch
#   fourier.jl      — Fourier derivative implementations for local and distributed layouts
#   polynomial.jl   — Chebyshev and Legendre derivative implementations
#   matrix_apply.jl — dense/sparse matrix application helpers along arbitrary axes

include("derivatives/eval.jl")
include("derivatives/fourier.jl")
include("derivatives/polynomial.jl")
include("derivatives/matrix_apply.jl")
