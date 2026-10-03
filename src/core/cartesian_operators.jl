"""
Cartesian coordinate system specific operators

This module implements Cartesian-specific operator variants following the
MultiClass dispatch pattern. Each operator has:
- CartesianX version for Cartesian coordinates
- DirectProductX version for direct product coordinate systems
- Matrix operation methods for implicit solvers
- Layout condition check/enforce patterns

Key features:
- CartesianComponent: Extract vector component by coordinate
- CartesianGradient: Gradient in Cartesian coordinates
- CartesianDivergence: Divergence in Cartesian coordinates
- CartesianCurl: Curl in 3D Cartesian coordinates
- CartesianLaplacian: Laplacian in Cartesian coordinates
- CartesianTrace: Trace of tensor in Cartesian coordinates
- CartesianSkew: Skew operation for 2D vectors
"""

# LinearAlgebra, SparseArrays already in Tarang.jl


# Runtime map:
#   core.jl         — common operator traits, component extraction, and tensor helpers
#   differential.jl — Cartesian gradient/divergence/curl/laplacian/trace/skew operators
#   evaluation.jl   — evaluation helpers for Cartesian operators
#   direct_product.jl — direct-product operator variants and their structure
#   dispatch.jl     — dispatch glue, fallbacks, and evaluate integration

include("cartesian_operators/core.jl")
include("cartesian_operators/differential.jl")
include("cartesian_operators/evaluation.jl")
include("cartesian_operators/direct_product.jl")
include("cartesian_operators/dispatch.jl")

# ============================================================================
# Exports
# ============================================================================

# Export types
export AbstractLinearOperator, OperatorConditions,
       CartesianComponent, CartesianGradient, CartesianDivergence, CartesianCurl,
       CartesianLaplacian, CartesianTrace, CartesianSkew,
       DirectProductGradient, DirectProductDivergence, DirectProductLaplacian,
       DirectProductTrace, DirectProductCurl, DirectProductComponent

# Export functions
export cartesian_component, get_tensorsig, get_scalar_size,
       matrix_dependence, matrix_coupling, subproblem_matrix,
       check_conditions, enforce_conditions, operate,
       evaluate_cartesian_component, evaluate_cartesian_gradient,
       evaluate_cartesian_divergence, evaluate_cartesian_curl,
       evaluate_cartesian_laplacian, evaluate_cartesian_trace,
       evaluate_cartesian_skew, evaluate_direct_product_curl,
       field_subtract,
       dispatch_cartesian_operator
