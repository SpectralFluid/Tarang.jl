using Test
using MPI
using Tarang

struct DCTCloseProbeArchitecture <: Tarang.AbstractArchitecture
    calls::Base.RefValue{Int}
end

function Tarang._close_backend_plan_caches!(dist::Distributor, arch::DCTCloseProbeArchitecture)
    arch.calls[] += 1
    return nothing
end

@testset "Distributor close dispatches backend cleanup without replacement" begin
    dist = Distributor(CartesianCoordinates("x"))
    calls = Ref(0)
    dist.architecture = DCTCloseProbeArchitecture(calls)
    close(dist)
    close(dist)
    @test calls[] == 1
    @test dist.closed
end

const TRANSFORMS_SOURCE = joinpath(@__DIR__, "..", "ext", "cuda", "transforms.jl")
const DCT_SOURCE = joinpath(@__DIR__, "..", "ext", "cuda", "dct_distributed.jl")
const PENCIL_SOURCE = joinpath(@__DIR__, "..", "ext", "cuda", "pencil.jl")
const GPU_DISTRIBUTED_SOURCE = joinpath(@__DIR__, "..", "src", "core", "gpu_distributed", "transform.jl")
const DISTRIBUTOR_SOURCE = joinpath(@__DIR__, "..", "src", "core", "distributor", "core.jl")

function _definition_name(expr)
    expr isa Expr || return nothing
    signature = if expr.head == :function
        expr.args[1]
    elseif expr.head == :(=)
        expr.args[1]
    else
        return nothing
    end
    signature isa Expr && signature.head == :where && (signature = signature.args[1])
    signature isa Expr && signature.head == :call || return nothing
    callee = signature.args[1]
    callee isa Symbol && return callee
    if callee isa Expr && callee.head == :. && callee.args[end] isa QuoteNode
        return callee.args[end].value
    end
    return nothing
end

function _find_definition(expr, name::Symbol)
    _definition_name(expr) == name && return expr
    expr isa Expr || return nothing
    for arg in expr.args
        found = _find_definition(arg, name)
        found === nothing || return found
    end
    return nothing
end

function _calls(expr, name::Symbol)
    expr isa Expr || return false
    expr.head == :call && expr.args[1] == name && return true
    return any(arg -> _calls(arg, name), expr.args)
end

"""Like `_calls`, but also matches a module-qualified call such as `Tarang.name(...)`."""
function _calls_qualified(expr, name::Symbol)
    expr isa Expr || return false
    if expr.head == :call
        callee = expr.args[1]
        callee == name && return true
        callee isa Expr && callee.head == :. && callee.args[end] == QuoteNode(name) && return true
    end
    return any(arg -> _calls_qualified(arg, name), expr.args)
end

@testset "distributed CUDA DCT cache context" begin
    transforms_ast = Meta.parseall(read(TRANSFORMS_SOURCE, String))

    # Evaluate the actual pure production helpers in a tiny harness. This avoids
    # loading CUDA/NCCL and therefore works on CPU-only CI workers.
    harness = Module(:DistributedDCTCacheContextHarness)
    Core.eval(harness, :(using MPI))
    for name in (:_compute_proc_grid, :_distributed_dct_proc_grid,
                 :_distributed_dct_plan_cache_key,
                 :_distributed_dct_comm_token)
        definition = _find_definition(transforms_ast, name)
        @test definition !== nothing
        Core.eval(harness, definition)
    end

    proc_grid = getfield(harness, :_distributed_dct_proc_grid)
    @test proc_grid((; size=8, mesh=(4, 2))) == (4, 2)
    @test proc_grid((; size=8, mesh=nothing)) == (2, 4)
    # A 1-D mesh (P,) decomposes only dim 1 of a 3D GPU+MPI field
    # (decomposed_axes), so the pencil grid is (P, 1). The square-ish heuristic
    # gave (1, 2) at P = 2: pencil blocks that matched no field buffer.
    @test proc_grid((; size=2, mesh=(2,))) == (2, 1)
    @test proc_grid((; size=6, mesh=(6,))) == (6, 1)
    @test proc_grid((; size=2, mesh=(1, 2))) == (1, 2)

    key = getfield(harness, :_distributed_dct_plan_cache_key)
    shape = (32, 48, 64)
    axes = (:chebyshev, :chebyshev, :chebyshev)
    baseline = key(shape, (2, 4), Float64, axes, :comm_a, 0)

    @test key(shape, (4, 2), Float64, axes, :comm_a, 0) != baseline
    @test key(shape, (2, 4), Float64, axes, :comm_b, 0) != baseline
    @test key(shape, (2, 4), Float64, axes, :comm_a, 1) != baseline
    @test key(shape, (2, 4), Float32, axes, :comm_a, 0) != baseline
    # A complex field's RealFourier dim 1 resolves to a different (full-length)
    # kind, so real and complex fields never share a plan.
    rf_axes = (:real_fourier, :chebyshev, :chebyshev)
    @test key(shape, (2, 4), Float64, rf_axes, :comm_a, 0) !=
          key(shape, (2, 4), Float64, (:complex_fourier, :chebyshev, :chebyshev), :comm_a, 0)
    @test _calls_qualified(_find_definition(transforms_ast, :get_or_create_distributed_dct_plan),
                           :_distributed_gpu_dct_axis_kinds)
    @test !_calls_qualified(_find_definition(transforms_ast, :get_or_create_distributed_dct_plan),
                            :axis_kinds)

    # The plan pre-flight, on host arrays (CuArray === Array in the harness): it
    # must refuse a missing NCCL with a clear message (instead of an unreachable
    # path or a rank-local transform) and refuse buffers whose blocks differ from
    # what the plan reads/writes, collectively (one Allreduce) before any transpose.
    Core.eval(harness, :(import Tarang))
    Core.eval(harness, :(using Tarang: get_grid_data, get_coeff_data))
    Core.eval(harness, :(const CuArray = Array))
    pencil_defs = Meta.parseall(read(PENCIL_SOURCE, String))
    for name in (:compute_pencil_shapes, :column_major_grid_coords)
        definition = _find_definition(pencil_defs, name)
        @test definition !== nothing
        Core.eval(harness, definition)
    end
    preflight_def = _find_definition(transforms_ast, :_distributed_dct_preflight)
    @test preflight_def !== nothing
    Core.eval(harness, preflight_def)
    preflight = getfield(harness, :_distributed_dct_preflight)
    get_plan = _find_definition(transforms_ast, :get_or_create_distributed_dct_plan)
    @test _calls(get_plan, :_distributed_dct_preflight)

    coords = CartesianCoordinates("x", "y", "z")
    pdist = Distributor(coords; dtype=Float64, comm=MPI.COMM_SELF)
    pbases = (RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
              ChebyshevT(coords["y"]; size=5, bounds=(-1.0, 1.0)),
              ChebyshevT(coords["z"]; size=6, bounds=(-1.0, 1.0)))
    pfield = ScalarField(pdist, "preflight", pbases, Float64)   # grid (8,5,6), coeffs (5,5,6)
    previous_nccl = Tarang._NCCL_MODULE[]
    try
        Tarang._NCCL_MODULE[] = nothing
        if !Tarang._try_load_nccl()
            err = try
                preflight(pfield, (8, 5, 6), (1, 1), rf_axes)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test err !== nothing && occursin("NCCL.jl is not loaded", sprint(showerror, err))
            @test err !== nothing && occursin("using NCCL", sprint(showerror, err))
        end
        Tarang._NCCL_MODULE[] = Module(:NCCL)     # NCCL "loaded": resolved on demand
        @test preflight(pfield, (8, 5, 6), (1, 1), rf_axes) === nothing
        # A plan that keeps the full dim 1 (a complex field's kinds) does not fit
        # this real field's half-spectrum buffer: refused, never swapped.
        err = try
            preflight(pfield, (8, 5, 6), (1, 1), (:complex_fourier, :chebyshev, :chebyshev))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test err !== nothing && occursin("coefficient block of size (5, 5, 6)", sprint(showerror, err))
    finally
        Tarang._NCCL_MODULE[] = previous_nccl
        close(pdist)
    end

    comm_token = getfield(harness, :_distributed_dct_comm_token)
    @test comm_token(MPI.COMM_SELF) != comm_token(MPI.COMM_WORLD)

    # Pin the call-site wiring as well as helper behavior: all three execution
    # contexts must participate when the production cache key is constructed.
    get_plan = _find_definition(transforms_ast, :get_or_create_distributed_dct_plan)
    @test get_plan !== nothing
    @test _calls(get_plan, :_distributed_dct_proc_grid)
    @test _calls(get_plan, :_distributed_dct_comm_token)
    @test _calls(get_plan, :_dct_cache_device_id)
    @test _calls(get_plan, :_distributed_dct_plan_cache_key)

    clear_cache = _find_definition(transforms_ast, :clear_distributed_dct_plan_cache!)
    @test clear_cache !== nothing
    @test _calls(clear_cache, :_finalize_distributed_dct_plans!)
    @test _calls(clear_cache, :_distributed_dct_teardown_order)
    @test _calls(clear_cache, :empty!)
    finalize_plans = _find_definition(transforms_ast, :_finalize_distributed_dct_plans!)
    @test finalize_plans !== nothing
    @test _calls(finalize_plans, :finalize_distributed_dct_plan!)
    # Teardown follows build order, so the builder must record it.
    @test _calls(get_plan, :push!)

    # Explicit cache teardown must release the MPI row/column communicators held
    # by each pencil; relying on a later GC finalizer makes communicator lifetime
    # nondeterministic and can retain obsolete contexts after a cache clear.
    dct_ast = Meta.parseall(read(DCT_SOURCE, String))
    finalize_plan = _find_definition(dct_ast, :finalize_distributed_dct_plan!)
    @test finalize_plan !== nothing
    @test _calls(finalize_plan, :free_pencil_decomposition!)

    pencil_ast = Meta.parseall(read(PENCIL_SOURCE, String))
    constructor = _find_definition(pencil_ast, :PencilDecomposition)
    @test constructor !== nothing
    @test !_calls(constructor, :finalizer)
    @test _calls(constructor, :_split_pencil_subcommunicators)

    close_pencil = _find_definition(pencil_ast, :close)
    @test close_pencil !== nothing
    @test _calls(close_pencil, :free_pencil_decomposition!)

    # The Distributor owns every transpose workspace. DistributedGPUTransform
    # must borrow through transpose_workspace! rather than construct (and then
    # have to close) a TransposableField of its own.
    gpu_distributed_ast = Meta.parseall(read(GPU_DISTRIBUTED_SOURCE, String))
    setup_workspace = _find_definition(gpu_distributed_ast, :setup_transposable_workspace!)
    @test setup_workspace !== nothing
    @test _calls(setup_workspace, :transpose_workspace!)
    @test !_calls(setup_workspace, :TransposableField)

    # close(dist) releases backend plan caches through a hook the CUDA
    # extension implements for its distributed DCT plans.
    distributor_ast = Meta.parseall(read(DISTRIBUTOR_SOURCE, String))
    close_dist = _find_definition(distributor_ast, :close)
    @test close_dist !== nothing
    @test _calls(close_dist, :_close_backend_plan_caches!)
    ext_hook = _find_definition(transforms_ast, :_close_backend_plan_caches!)
    @test ext_hook !== nothing
    @test _calls(ext_hook, :_finalize_distributed_dct_plans!)
    @test _calls(ext_hook, :_distributed_dct_teardown_order)
    @test _calls(ext_hook, :_distributed_dct_comm_token)

    # The split helper is dependency-injected so a failure after the first
    # communicator can be tested without MPI ranks or a GPU.
    split_definition = _find_definition(pencil_ast, :_split_pencil_subcommunicators)
    @test split_definition !== nothing
    if split_definition !== nothing
        Core.eval(harness, split_definition)
        split_calls = Ref(0)
        freed = Any[]
        splitter = (_comm, _color, _key) -> begin
            split_calls[] += 1
            split_calls[] == 1 ? :row_comm : error("injected second split failure")
        end
        @test_throws ErrorException getfield(
            harness, :_split_pencil_subcommunicators)(
                :world, 0, 0; splitter=splitter, freer=comm -> push!(freed, comm))
        @test freed == [:row_comm]
    end
end

# ---------------------------------------------------------------------------
# Explicit, collective NCCL teardown. `ncclCommDestroy` is an intra-node
# collective, but NCCL.jl only registers it as a GC finalizer, so merely
# dropping references let each rank destroy whenever its GC ran — a rank could
# block in ncclCommDestroy while a peer waited in an MPI collective. A mock
# NCCL module (Tarang resolves NCCL.jl from the session) records the sequence.
# ---------------------------------------------------------------------------
struct TeardownMockDevice end
const TEARDOWN_EVENTS = Any[]
Tarang.synchronize(::Tarang.GPU{TeardownMockDevice}) = (push!(TEARDOWN_EVENTS, :sync); nothing)
module TeardownMockNCCL
    import ..TeardownMockDevice, ..TEARDOWN_EVENTS
    device(comm) = TeardownMockDevice()
    function destroy(comm)
        push!(TEARDOWN_EVENTS, (:destroy, comm))
        comm === :bad && error("injected destroy failure")
        return nothing
    end
end

@testset "NCCL communicators are destroyed explicitly, in a fixed order" begin
    MPI.Initialized() || MPI.Init()
    previous = Tarang._NCCL_MODULE[]
    try
        Tarang._NCCL_MODULE[] = TeardownMockNCCL
        subcomms(row, col) = (s = Tarang.NCCLSubComms(); s.initialized = true;
                              s.row_comm = row; s.col_comm = col; s)

        # Device work first, then (barrier on the parent communicator) row, column.
        s = subcomms(:row, :col)
        empty!(TEARDOWN_EVENTS)
        Tarang.finalize_nccl_subcomms!(s; barrier_comm=MPI.COMM_SELF)
        @test TEARDOWN_EVENTS == [:sync, :sync, (:destroy, :row), (:destroy, :col)]
        @test !s.initialized && s.row_comm === nothing && s.col_comm === nothing
        # Idempotent: a repeated call (or the later GC finalizer) destroys nothing.
        empty!(TEARDOWN_EVENTS)
        Tarang.finalize_nccl_subcomms!(s; barrier_comm=MPI.COMM_SELF)
        @test isempty(TEARDOWN_EVENTS)

        # A single-rank direction carries no communicator and is skipped.
        empty!(TEARDOWN_EVENTS)
        Tarang.finalize_nccl_subcomms!(subcomms(nothing, :col))
        @test TEARDOWN_EVENTS == [:sync, (:destroy, :col)]

        # A failed row destroy must not skip the column destroy its peers are
        # already waiting in; the failure surfaces afterwards.
        s = subcomms(:bad, :col)
        empty!(TEARDOWN_EVENTS)
        @test_throws ErrorException Tarang.finalize_nccl_subcomms!(s)
        @test TEARDOWN_EVENTS == [:sync, :sync, (:destroy, :bad), (:destroy, :col)]
        @test !s.initialized

        # The init_nccl! communicator: destroyed (not just dropped), config reset.
        Tarang.NCCL_CONFIG.initialized = true
        Tarang.NCCL_CONFIG.comm_handle = :world
        Tarang.NCCL_CONFIG.mpi_comm = MPI.COMM_SELF
        empty!(TEARDOWN_EVENTS)
        Tarang.finalize_nccl!()
        @test TEARDOWN_EVENTS == [:sync, (:destroy, :world)]
        @test !Tarang.NCCL_CONFIG.initialized
        @test Tarang.NCCL_CONFIG.comm_handle === nothing
        @test Tarang.NCCL_CONFIG.mpi_comm === nothing
    finally
        Tarang._NCCL_MODULE[] = previous
        Tarang.NCCL_CONFIG.initialized = false
        Tarang.NCCL_CONFIG.comm_handle = nothing
        Tarang.NCCL_CONFIG.mpi_comm = nothing
    end

    # The CUDA transpose buffer passes its parent communicator for the barrier.
    nccl_ast = Meta.parseall(read(joinpath(@__DIR__, "..", "ext", "cuda", "nccl_transpose.jl"), String))
    finalize_buffer = _find_definition(nccl_ast, :finalize_nccl_transpose!)
    @test finalize_buffer !== nothing
    @test _calls_qualified(finalize_buffer, :finalize_nccl_subcomms!)
    @test occursin("barrier_comm", string(finalize_buffer))
end

@testset "cached distributed DCT plans are torn down in build order" begin
    transforms_ast = Meta.parseall(read(TRANSFORMS_SOURCE, String))
    h = Module(:DistributedDCTTeardownHarness)
    Core.eval(h, quote
        import Tarang
        using Tarang: GPU
        struct CuDevice end
        using MPI
        const DISTRIBUTED_DCT_PLAN_CACHE = Dict{Tuple,Any}()
        const DISTRIBUTED_DCT_PLAN_BUILD_ORDER = Tuple[]
        const DISTRIBUTED_DCT_PLAN_LOCK = ReentrantLock()
        const FINALIZED = Any[]
        function finalize_distributed_dct_plan!(plan)
            push!(FINALIZED, plan)
            plan === :bad && error("injected plan teardown failure")
            return nothing
        end
    end)
    for name in (:_distributed_dct_teardown_order, :_finalize_distributed_dct_plans!,
                 :clear_distributed_dct_plan_cache!, :_distributed_dct_comm_token)
        definition = _find_definition(transforms_ast, name)
        @test definition !== nothing
        Core.eval(h, definition)
    end
    # The close(dist) hook, renamed so it never extends the real Tarang function.
    hook = deepcopy(_find_definition(transforms_ast, :_close_backend_plan_caches!))
    hook.args[1].args[1] = :close_hook
    Core.eval(h, hook)

    teardown_order = getfield(h, :_distributed_dct_teardown_order)
    kinds = (:real_fourier, :chebyshev, :chebyshev)
    key(i, token, device) = ((8, 6, i), (2, 1), Float64, kinds, token, device)
    # The cache Dict hashes process-local communicator tokens and device ids, so
    # its iteration order differs between ranks; the teardown order must not.
    for device in 0:15
        built = [key(i, token, device) for (i, token) in ((1, :a), (2, :b), (3, :a), (4, :a), (5, :b))]
        rank_cache = Dict(k => i for (i, k) in enumerate(built))
        @test teardown_order(built, rank_cache) == built
        @test teardown_order(built, rank_cache; comm_token=:a) == built[[1, 3, 4]]
        # Keys no longer cached (or never cached) are skipped.
        delete!(rank_cache, built[3])
        @test teardown_order(vcat(built, [key(9, :a, device)]), rank_cache) == built[[1, 2, 4, 5]]
    end

    cache = getfield(h, :DISTRIBUTED_DCT_PLAN_CACHE)
    build_order = getfield(h, :DISTRIBUTED_DCT_PLAN_BUILD_ORDER)
    finalized = getfield(h, :FINALIZED)
    function build!(entries)
        empty!(cache); empty!(build_order); empty!(finalized)
        for (k, plan) in entries
            cache[k] = plan
            push!(build_order, k)
        end
    end

    # clear_distributed_dct_plan_cache!: build order; a failing plan does not stop
    # the plans after it (peers expect those collective teardowns); rethrown last.
    build!([key(1, :a, 3) => :p1, key(2, :b, 3) => :bad, key(3, :a, 3) => :p3, key(4, :b, 3) => :p4])
    err = try
        getfield(h, :clear_distributed_dct_plan_cache!)()
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("injected plan teardown failure", err.msg)
    @test finalized == [:p1, :bad, :p3, :p4]
    @test isempty(cache) && isempty(build_order)

    # close(dist): only this communicator's plans, in build order; the rest stay.
    MPI.Initialized() || MPI.Init()
    dist = Distributor(CartesianCoordinates("x"); comm=MPI.COMM_SELF)
    mine = getfield(h, :_distributed_dct_comm_token)(dist.comm)
    build!([key(1, mine, 0) => :m1, key(2, :other, 0) => :o2, key(3, mine, 0) => :m3])
    getfield(h, :close_hook)(dist, Tarang.GPU(h.CuDevice()))
    @test finalized == [:m1, :m3]
    @test collect(keys(cache)) == [key(2, :other, 0)]
    @test build_order == [key(2, :other, 0)]
    close(dist)
end
