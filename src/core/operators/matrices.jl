"""
    Expression matrices

This file contains sparse operator-matrix construction for implicit solvers.
"""


# Runtime map:
#   expression.jl          — compositional expression_matrices entry points and linearity helpers
#   subproblem_helpers.jl  — subproblem-specific basis, operand, and mode helpers
#   subproblem_operators.jl — subproblem_matrix implementations and remaining operator expression fallbacks
#   builders.jl            — low-level differentiation and lift matrix builders

include("matrices/expression.jl")
include("matrices/subproblem_helpers.jl")
include("matrices/subproblem_operators.jl")
include("matrices/builders.jl")
