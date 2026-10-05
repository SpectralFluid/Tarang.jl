"""
    Problem matrix building

This file contains solver-facing matrix construction for parsed problems.
"""


# Runtime map:
#   build.jl         — top-level matrix assembly and equation-expression construction
#   expr_analysis.jl — equation-variable detection, DOF inference, and operator splitting
#   support.jl       — size helpers, equation conditions, and shared small utilities
#   spectral.jl      — spectral operator blocks and expression-matrix assembly
#   legacy.jl        — forcing-vector construction and legacy compatibility helpers

include("problem_matrices/build.jl")
include("problem_matrices/expr_analysis.jl")
include("problem_matrices/support.jl")
include("problem_matrices/spectral.jl")
include("problem_matrices/legacy.jl")
