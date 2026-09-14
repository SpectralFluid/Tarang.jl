"""Parse CPU benchmark controls without starting MPI or allocating a problem."""
function cpu_benchmark_config(env=ENV; worker=false)
    positive(key, default) = begin
        n = tryparse(Int, get(env, key, string(default)))
        n !== nothing && n > 0 || throw(ArgumentError("$key must be a positive integer"))
        n
    end
    intlist(value, separator) = begin
        ns = tryparse.(Int, split(value, separator))
        all(n -> n !== nothing && n > 0, ns) || throw(ArgumentError("Invalid positive integer list: $value"))
        Int[n for n in ns]
    end
    n = positive("TARANG_BENCH_N", 64)
    shape = Tuple(intlist(get(env, "TARANG_BENCH_SHAPE", "$(n)x$(n)"), 'x'))
    length(shape) in (2, 3) && minimum(shape) >= 8 ||
        throw(ArgumentError("TARANG_BENCH_SHAPE must have 2 or 3 dimensions, each >= 8"))
    ranks = intlist(get(env, "TARANG_BENCH_RANKS", "1,2,4"), ',')
    allunique(ranks) || throw(ArgumentError("TARANG_BENCH_RANKS must not repeat ranks"))
    threads = positive("TARANG_BENCH_THREADS", 1)
    fftw_threads = positive("TARANG_BENCH_FFTW_THREADS", threads)
    blas_threads = positive("TARANG_BENCH_BLAS_THREADS", 1)
    gather_setting = get(env, "TARANG_BENCH_GATHER", worker ? "0" : "1")
    gather_setting in ("0", "1") || throw(ArgumentError("TARANG_BENCH_GATHER must be 0 or 1"))
    mesh = haskey(env, "TARANG_BENCH_MESH") ? Tuple(intlist(env["TARANG_BENCH_MESH"], 'x')) : nothing
    !worker && mesh !== nothing && length(ranks) != 1 &&
        throw(ArgumentError("An explicit mesh requires one rank count per driver invocation"))
    return (; shape, ranks, threads, fftw_threads, blas_threads, mesh,
             steps=positive("TARANG_BENCH_STEPS", 30),
             samples=positive("TARANG_BENCH_SAMPLES", 3), gather=gather_setting == "1")
end
