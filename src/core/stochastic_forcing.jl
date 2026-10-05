"""
Stochastic and deterministic forcing.

Split into `core/forcing/` — this file only sets the load order, which matters:
the types must exist before the generation/application methods that dispatch on
them, and the exports come last.
"""

include("forcing/types.jl")
include("forcing/generation.jl")
include("forcing/application.jl")
include("forcing/diagnostics.jl")
include("forcing/deterministic.jl")
