# Strict entry point: a machine without CUDA must fail, never report a skipped pass.
using CUDA
CUDA.functional() || error("A functional CUDA GPU is required")
CUDA.allowscalar(false)
CUDA.versioninfo()
ENV["TARANG_REQUIRE_CUDA"] = "true"
include("test_gpu_batch_stream_ordering.jl")
include("test_gpu_solver_workspace_reuse.jl")
include("test_gpu_cheb_pipeline_cuda.jl")
include("test_gpu_transpose_metadata_cache.jl")
include("test_gpu_nonlinear_optimizations.jl")
include("test_gpu_rk_optimizations_cuda.jl")
include("test_gpu_timesteppers.jl")
include("test_gpu_optional_workspaces_cuda.jl")
