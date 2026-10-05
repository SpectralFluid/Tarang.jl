"""
    Various operator operations

This file contains evaluation functions for:
- Interpolate, integrate, average, lift, convert
- GeneralFunction and UnaryGridFunction
- Grid and coeff conversion
- Component extraction (component, radial, angular, azimuthal)
"""


# Runtime map:
#   interpolate.jl  — interpolation evaluation and Clenshaw helpers
#   integrate.jl    — integration and averaging evaluation
#   lift_convert.jl — lift and basis-conversion evaluation
#   misc.jl         — general functions, layout conversion, components, copy, and Hilbert transform

include("operations/interpolate.jl")
include("operations/integrate.jl")
include("operations/lift_convert.jl")
include("operations/misc.jl")
