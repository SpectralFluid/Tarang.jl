"""
    Field layout operations

This file contains layout transitions, field-level access helpers, arithmetic,
and shape/filter utilities for field types.
"""


# Runtime map:
#   access.jl           — CPU/local data access and layout transitions
#   operations.jl       — random fill, integration, and Vector/Tensor field API helpers
#   arithmetic.jl       — scalar-field arithmetic (NetCDF I/O moved
#                                      to tools/field_netcdf_io.jl, which loads
#                                      after the slab layer it depends on)
#   filters_shapes.jl   — spectral filtering and global/local shape helpers
#   vectorized.jl       — vectorized array kernels, fast_axpy!, and unit-vector helpers

include("field_layout/access.jl")
include("field_layout/operations.jl")
include("field_layout/arithmetic.jl")
include("field_layout/filters_shapes.jl")
include("field_layout/vectorized.jl")
