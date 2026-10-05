"""
Problem definitions and equation parsing for Tarang.jl

Split into focused sub-files:
- ir.jl: equation IR, compiled artifacts, and runtime-cache lifecycle
- types.jl: InitialValueProblem, LinearBoundaryValueProblem, NonlinearBoundaryValueProblem, EigenvalueProblem definitions and constructors
- parsing.jl: Expression parsing and evaluation
- matrices.jl: Matrix building for solvers
- utils.jl: Validation, substitution, introspection, exports
"""

include("problems/ir.jl")
include("problems/types.jl")
include("problems/parsing.jl")
include("problems/matrices.jl")
include("problems/utils.jl")
