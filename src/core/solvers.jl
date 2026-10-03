"""
Solver implementations for Tarang.jl

Split into focused sub-files:
- execution_plan.jl: runtime path facts resolved once at construction
- types.jl: Solver definitions and constructors
- state_vectors.jl: Field/vector transport for matrix solver paths
- stepping.jl: Time stepping, BVP/EigenvalueProblem solve
- lazy_rhs.jl: Type-specialized lazy RHS evaluation with broadcasting fusion
- rhs_runtime.jl: RHS evaluation strategy selection
- utils.jl: Diagnostics and exports

Checkpoint save/load moved to src/tools/solver_checkpoint.jl: it reads and writes
NetCDF, whose slab layer loads after the whole core stack.
"""

include("solvers/execution_plan.jl")
include("solvers/types.jl")
include("solvers/state_vectors.jl")
include("solvers/compiled_rhs.jl")
include("solvers/lazy_rhs.jl")
include("solvers/rhs_runtime.jl")
include("solvers/runtime.jl")
include("solvers/stepping.jl")
include("solvers/utils.jl")
