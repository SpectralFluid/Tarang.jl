"""
Field types and operations for Tarang.jl

Split into focused sub-files:
- types.jl: Type definitions (ScalarField, VectorField, TensorField)
- data.jl: Data access, allocation, component operations
- layout.jl: Layout transitions, transforms, field operations
- exports.jl: Export declarations
"""

include("field/types.jl")
include("field/data.jl")
include("field/layout.jl")
include("field/exports.jl")
