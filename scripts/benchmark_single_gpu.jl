#=
Single-device nonlinear-product and full-step benchmark.

SHAPE=2048x2048 julia --project=. scripts/benchmark_single_gpu.jl
SHAPE=128x128x128 STEPS=20 SAMPLES=5 OUTPUT=h200-128.toml julia --project=. scripts/benchmark_single_gpu.jl
DEVICE=cpu is a small-grid harness check, not an H200 performance proxy.
Use a Julia environment containing Tarang and CUDA for DEVICE=gpu (the default).
=#
using Tarang, Statistics, TOML, LinearAlgebra
if lowercase(get(ENV, "DEVICE", "gpu")) == "gpu"
    using CUDA
end

function benchmark_single_device()
    shape = Tuple(parse.(Int, split(get(ENV, "SHAPE", "512x512"), 'x')))
    length(shape) in (2, 3) && all(>(4), shape) || error("SHAPE must contain two or three dimensions, all > 4")
    device_name = lowercase(get(ENV, "DEVICE", "gpu"))
    device_name in ("gpu", "cpu") || error("DEVICE must be gpu or cpu")
    gpu = device_name == "gpu"
    if gpu
        CUDA.functional() || error("A functional CUDA GPU is required; this benchmark never silently falls back")
        CUDA.allowscalar(false)
        CUDA.versioninfo()
    end
    steps = parse(Int, get(ENV, "STEPS", "20"))
    samples = parse(Int, get(ENV, "SAMPLES", "5"))
    warmup = parse(Int, get(ENV, "WARMUP", "10"))
    minimum((steps, samples, warmup)) > 0 || error("STEPS, SAMPLES, WARMUP must be positive")
    dt = parse(Float64, get(ENV, "DT", "0.0001"))
    dt > 0 || error("DT must be positive")
    coords = CartesianCoordinates(("x", "y", "z")[1:length(shape)]...)
    dist = Distributor(coords; dtype=Float64, device=gpu ? GPU() : CPU())
    bases = ntuple(d -> RealFourier(coords[d]; size=shape[d], bounds=(0.0, 2π), dealias=1.5), length(shape))
    domain = Domain(dist, bases)
    q = ScalarField(domain, "q")
    initial = ones(Float64, shape)
    for d in eachindex(shape)
        axis = reshape(sin.(2π .* (0:shape[d]-1) ./ shape[d]),
                       ntuple(j -> j == d ? shape[d] : 1, length(shape)))
        initial .*= axis
    end
    initial .*= 0.1
    copyto!(grid_data!(q), initial)
    problem = InitialValueProblem([q])
    add_parameters!(problem; nu=0.01)
    add_equation!(problem, "dt(q) - nu*lap(q) = -q*∂x(q)")
    solver = InitialValueSolver(problem, RK222(); dt)
    output = ScalarField(domain, "product")
    sync_device() = gpu ? CUDA.synchronize() : nothing
    allocated_device() = gpu ? Int(CUDA.alloc_stats.alloc_bytes) : 0

    function measure(f)
        for _ in 1:warmup
            f()
        end
        sync_device()
        times = Float64[]
        allocations = Int[]
        for _ in 1:samples
            sync_device()
            before = allocated_device()
            elapsed = @elapsed begin
                for _ in 1:steps
                    f()
                end
                sync_device()
            end
            push!(times, 1000elapsed/steps)
            push!(allocations, allocated_device()-before)
        end
        return Dict("median_ms" => median(times), "sample_ms" => times,
                    "max_device_alloc_bytes_per_call" => maximum(allocations)/steps)
    end

    product = measure(() -> Tarang._dealiased_lazy_product!(output, q, q))
    timestep = measure(() -> step!(solver, dt))
    ev = Tarang._get_evaluator(dist)
    buffers = Any[]
    for ws in values(ev.pencil_transforms.padded_dealiasing), k in fieldnames(typeof(ws))
        x = getfield(ws, k)
        x isa AbstractArray && eltype(x) <: Number && push!(buffers, x)
    end
    scratch_bytes = sum(sizeof, unique(objectid, buffers))
    sync_device()
    qgrid = grid_data!(q)
    finite = all(isfinite, qgrid)
    finite || error("Benchmark solution contains nonfinite values")
    report = Dict("device" => device_name, "julia_version" => string(VERSION),
                  "shape" => collect(shape), "precision" => "Float64", "scheme" => "RK222",
                  "dt" => dt, "warmup" => warmup, "steps_per_sample" => steps, "samples" => samples,
                  "product" => product, "step" => timestep,
                  "nonlinear_scratch_bytes" => scratch_bytes,
                  "nonlinear_result_pool_fields" => length(ev.nl_result_pool),
                  "final_sim_time" => solver.sim_time, "final_max_abs" => Float64(maximum(abs, qgrid)))
    if gpu
        report["gpu_name"] = CUDA.name(CUDA.device())
        report["free_device_bytes_after_run"] = Int(CUDA.available_memory())
    end
    path = get(ENV, "OUTPUT", "single-gpu-benchmark.toml")
    open(path, "w") do io
        TOML.print(io, report)
    end
    TOML.print(stdout, report)
    println("\nSaved ", abspath(path))
    return report
end

benchmark_single_device()
