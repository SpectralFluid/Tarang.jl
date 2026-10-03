"""
    Field data operations

This file contains data access, allocation, scaling, and storage helpers for
field types.
"""


# Runtime map:
#   access_locked.jl     — `has`, LockedField, and access/property shims
#   component_buffers.jl — stacked component-buffer helpers
#   copy_alloc.jl        — copy/deepcopy, raw storage accessors, and allocation
#   distributor_utils.jl — local/global size and index helpers
#   scales.jl            — scale changes, resampling, and local-data helpers

include("field_data/access_locked.jl")
include("field_data/component_buffers.jl")
include("field_data/copy_alloc.jl")
include("field_data/distributor_utils.jl")
include("field_data/scales.jl")
