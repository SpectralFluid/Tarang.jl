"""
    Tensor operations and special operators

This file contains evaluation and matrix helpers for tensor-style operators.
"""


# Runtime map:
#   basic.jl                — trace, skew, and transpose-components evaluation
#   curl_laplacian.jl       — curl and standard Laplacian evaluation
#   fractional_laplacian.jl — fractional Laplacian evaluation and matrix methods
#   misc.jl                 — outer-product and AdvectiveCFL utilities

include("tensor/basic.jl")
include("tensor/curl_laplacian.jl")
include("tensor/fractional_laplacian.jl")
include("tensor/misc.jl")
