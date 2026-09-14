# Run: julia --project=. scripts/benchmark_cpu_parallel.jl [output_directory]
# See benchmark_cpu_parallel.md for grid, rank, thread and external-launch controls.
using MPI
using Tarang
using FFTW
using LinearAlgebra
using Statistics
using TOML
using Printf
using Logging
using Profile
include("cpu_parallel_config.jl")

function benchmark_problem(shape; mesh=nothing)
    names = ("x", "y", "z")[1:length(shape)]
    coords = CartesianCoordinates(names...)
    dist = Distributor(coords; dtype=Float64, device=CPU(), mesh=mesh)
    bases = Tuple(RealFourier(coords[c]; size=n, bounds=(0.0, 2pi)) for (c,n) in zip(names, shape))
    q = ScalarField(Domain(dist, bases), "q")
    data = grid_data!(q)
    axes_global = data isa Tarang.PencilArrays.PencilArray ?
                  Tarang.PencilArrays.pencil(data).axes_local : map(n -> 1:n, shape)
    axes_data = ntuple(length(shape)) do d
        reshape(2pi .* (collect(axes_global[d]) .- 1) ./ shape[d],
                ntuple(i -> i == d ? length(axes_global[d]) : 1, length(shape)))
    end
    x, y = axes_data[1:2]
    parent(data) .= 0.2 .* sin.(x) .* cos.(y) .+ 0.1 .* cos.(2 .* x .+ y)
    length(shape) == 3 && (parent(data) .*= 1 .+ 0.2 .* cos.(axes_data[3]))
    problem = InitialValueProblem([q])
    add_equation!(problem, "dt(q) - 0.02*lap(q) = -q*d(q,x)")
    solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)
    return solver, q
end

function advance_benchmark!(solver, steps)
    for _ in 1:steps
        step!(solver, 1e-3)
    end
end

function benchmark_worker(output, config)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    ranks = MPI.Comm_size(comm)
    shape, steps, samples = config.shape, config.steps, config.samples
    FFTW.set_num_threads(config.fftw_threads)
    BLAS.set_num_threads(config.blas_threads)
    global_logger(ConsoleLogger(stderr, Logging.Warn))
    rank == 0 && mkpath(output)
    MPI.Barrier(comm)
    mesh = config.mesh === nothing && ranks > 1 ?
           (length(shape) == 2 ? (ranks,) : Tarang.create_2d_process_mesh(ranks)) : config.mesh

    # Compile on a small grid that still supports every transposed decomposition.
    warm_n = mesh === nothing ? 8 : max(8, 2 * maximum(mesh))
    warm_solver, _ = benchmark_problem(ntuple(_ -> warm_n, length(shape)); mesh=mesh)
    advance_benchmark!(warm_solver, 5)
    GC.gc()
    MPI.Barrier(comm)
    setup = @timed benchmark_problem(shape; mesh=mesh)
    solver, q = setup.value
    setup_seconds = MPI.Allreduce(setup.time, MPI.MAX, comm)
    setup_bytes = MPI.Allreduce(setup.bytes, MPI.MAX, comm)
    advance_benchmark!(solver, 10)

    seconds = Float64[]
    bytes = Float64[]
    gc_seconds = Float64[]
    for _ in 1:samples
        GC.gc()
        MPI.Barrier(comm)
        measurement = @timed advance_benchmark!(solver, steps)
        push!(seconds, MPI.Allreduce(measurement.time, MPI.MAX, comm) / steps)
        push!(bytes, MPI.Allreduce(measurement.bytes, MPI.MAX, comm) / steps)
        push!(gc_seconds, MPI.Allreduce(measurement.gctime, MPI.MAX, comm) / steps)
    end
    peak_rss = MPI.Allreduce(Sys.maxrss(), MPI.MAX, comm)
    local_grid = parent(grid_data!(q))
    diagnostics = [MPI.Allreduce(sum(local_grid), MPI.SUM, comm),
                   MPI.Allreduce(sum(abs2, local_grid), MPI.SUM, comm),
                   MPI.Allreduce(maximum(abs, local_grid), MPI.MAX, comm)]
    all(isfinite, diagnostics) || error("Non-finite final field diagnostics")
    workspaces = [ws for (key, ws) in Tarang._PADDED_DIST_WS_CACHE if first(key) == objectid(q.dist)]
    scratch_bytes = sum((sum(length(parent(b)) * sizeof(eltype(b)) for b in ws.buffers) for ws in workspaces); init=0)
    scratch_global = MPI.Allreduce(scratch_bytes, MPI.SUM, comm)
    scratch_max = MPI.Allreduce(scratch_bytes, MPI.MAX, comm)
    final_grid = config.gather ? Tarang.gather_array(q.dist, grid_data!(q)) : nothing
    if rank == 0
        result = Dict(
            "julia_version" => string(VERSION), "ranks" => ranks, "shape" => collect(shape),
            "cpu" => Sys.CPU_NAME, "rank_zero_host" => MPI.Get_processor_name(),
            "mesh" => q.dist.mesh === nothing ? Int[] : collect(q.dist.mesh),
            "threads_per_rank" => Threads.nthreads(), "fftw_threads" => FFTW.get_num_threads(),
            "blas_threads" => BLAS.get_num_threads(), "steps_per_sample" => steps, "samples" => samples,
            "setup_seconds" => setup_seconds, "setup_allocated_bytes_max_rank" => setup_bytes,
            "seconds_per_step_max_rank" => seconds, "allocated_bytes_per_step_max_rank" => bytes,
            "gc_seconds_per_step_max_rank" => gc_seconds, "peak_rss_bytes_max_rank" => peak_rss,
            "assembled_global_matrices" => solver.execution_plan.assembled_global_matrices,
            "sim_time" => solver.sim_time, "field_diagnostics" => diagnostics,
            "gathered_field" => config.gather,
            "padded_scratch_bytes_global" => scratch_global,
            "padded_scratch_bytes_max_rank" => scratch_max,
            "padded_transposes_last_product" => sum((ws.transpose_count for ws in workspaces); init=0),
            "padded_transpose_global_elements_last_product" => sum((ws.transpose_elements for ws in workspaces); init=0),
            "core_seconds_per_step_proxy" => ranks * max(Threads.nthreads(), FFTW.get_num_threads(), BLAS.get_num_threads()) * median(seconds),
        )
        config.gather && (result["final_grid"] = vec(Array(final_grid)))
        open(joinpath(output, "ranks-$ranks.toml"), "w") do io
            TOML.print(io, result)
        end
    end

    if get(ENV, "TARANG_BENCH_PROFILE", "0") == "1"
        Profile.clear()
        MPI.Barrier(comm)
        Profile.@profile advance_benchmark!(solver, steps)
        open(joinpath(output, "profile-$ranks-rank-$rank.txt"), "w") do io
            Profile.print(io; format=:flat, sortedby=:count)
        end
    end
    MPI.Barrier(comm)
end

function benchmark_driver(output, config)
    mkpath(output)
    project = dirname(@__DIR__)
    shape_text = join(config.shape, 'x')
    println("CPU MPI benchmark: $shape_text, $(config.samples) samples of $(config.steps) steps; output: $output")
    println("Setup is JIT-warmed. RSS includes compilation and process overhead.")
    reference = nothing
    baseline = 0.0
    reports = Dict{String, Any}[]
    baseline_cost = 0.0
    for ranks in config.ranks
        command = `$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --threads=$(config.threads) --startup-file=no --project=$project $(@__FILE__) --worker $output $shape_text $(config.steps) $(config.samples)`
        run(addenv(command, "TARANG_FFTW_THREADS" => string(config.fftw_threads),
                   "OPENBLAS_NUM_THREADS" => string(config.blas_threads),
                   "TARANG_BENCH_GATHER" => config.gather ? "1" : "0"))
        report = TOML.parsefile(joinpath(output, "ranks-$ranks.toml"))
        field = pop!(report, "final_grid", nothing)
        compared = config.gather ? field : report["field_diagnostics"]
        if reference === nothing
            reference = compared
            baseline = median(report["seconds_per_step_max_rank"])
            baseline_cost = report["core_seconds_per_step_proxy"]
        end
        max_error = maximum(abs.(compared .- reference))
        if config.gather
            max_error <= 1e-9 || error("Field mismatch at $ranks ranks: $max_error")
        else
            isapprox(compared, reference; rtol=1e-10, atol=1e-9) || error("Distributed diagnostics mismatch at $ranks ranks")
        end
        report["comparison"] = config.gather ? "full_field" : "diagnostics_only"
        report["max_abs_difference_vs_baseline"] = max_error
        report["baseline_ranks"] = first(config.ranks)
        report["speedup_vs_baseline"] = baseline / median(report["seconds_per_step_max_rank"])
        report["relative_core_time_proxy"] = report["core_seconds_per_step_proxy"] / baseline_cost
        report["parallel_efficiency_vs_baseline"] = 1 / report["relative_core_time_proxy"]
        push!(reports, report)
        @printf("%d ranks: setup %.4f s; step %.3f ms; alloc %.1f KiB/step; peak RSS %.1f MiB/rank; speedup %.2fx; error %.3g\n",
                ranks, report["setup_seconds"], 1e3 * median(report["seconds_per_step_max_rank"]),
                median(report["allocated_bytes_per_step_max_rank"]) / 1024,
                report["peak_rss_bytes_max_rank"] / 2.0^20, report["speedup_vs_baseline"], max_error)
        flush(stdout)
    end
    open(joinpath(output, "summary.toml"), "w") do io
        TOML.print(io, Dict("runs" => reports))
    end
end

if !isempty(ARGS) && ARGS[1] == "--worker"
    length(ARGS) == 5 || error("Usage: --worker OUTPUT SHAPE STEPS SAMPLES")
    env = copy(ENV)
    env["TARANG_BENCH_SHAPE"] = occursin('x', ARGS[3]) ? ARGS[3] : "$(ARGS[3])x$(ARGS[3])"
    env["TARANG_BENCH_STEPS"], env["TARANG_BENCH_SAMPLES"] = ARGS[4], ARGS[5]
    benchmark_worker(abspath(ARGS[2]), cpu_benchmark_config(env; worker=true))
else
    output = isempty(ARGS) ? mktempdir(; prefix="tarang-cpu-parallel-", cleanup=false) : abspath(ARGS[1])
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("Use --worker under an external MPI launcher")
    benchmark_driver(output, cpu_benchmark_config())
end
