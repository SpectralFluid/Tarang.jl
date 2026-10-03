using Test, Tarang, MPI, PencilArrays

MPI.Initialized() || MPI.Init()
const _GR_COMM = MPI.COMM_WORLD
const _GR_NP = MPI.Comm_size(_GR_COMM)
const _GR_TO = PencilArrays.TimerOutputs
_GR_TO.enable_debug_timings(PencilArrays.Transpositions)

function _gr_transpose_calls(timer)
    function count_calls(node)
        sum((name == "transpose!" ? child["n_calls"] : count_calls(child)
             for (name, child) in node["inner_timers"]); init=0)
    end
    count_calls(_GR_TO.todict(timer))
end

function _gr_run(enabled, timestepper)
    Tarang.set_group_transforms!(enabled)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU(), mesh=(_GR_NP,))
    xb = RealFourier(coords["x"]; size=12, bounds=(0.0, 2pi), dealias=1.0)
    yb = RealFourier(coords["y"]; size=14, bounds=(0.0, 2pi), dealias=1.0)
    domain = Domain(dist, (xb, yb))
    u, v = ScalarField(domain, "u"), ScalarField(domain, "v")
    for (i, field) in enumerate((u, v))
        data = grid_data!(field)
        ranges = data isa PencilArrays.PencilArray ? PencilArrays.range_local(data) : axes(data)
        values = [0.2i + 0.07sin(2pi*(ix-1)/12) + 0.04cos(2pi*(iy-1)/14)
                  for ix in ranges[1], iy in ranges[2]]
        copyto!(Tarang.get_local_data(data), values)
    end
    problem = InitialValueProblem([u, v])
    add_equation!(problem, "∂t(u) - 0.1*lap(u) = -u*u")
    add_equation!(problem, "∂t(v) - 0.2*lap(v) = -v*v")
    solver = InitialValueSolver(problem, timestepper; dt=0.002)
    step!(solver, 0.002) # warm plans, history, and scratch before counting
    bundle = Tarang._field_transform_bundle(u)
    timer = bundle.pencil_fft_input === nothing ? nothing : PencilArrays.timer(bundle.pencil_fft_input)
    timer === nothing || _GR_TO.reset_timer!(timer)
    for _ in 1:3
        step!(solver, 0.002)
    end
    calls = timer === nothing ? 0 : _gr_transpose_calls(timer)
    workspaces = get(bundle.pencil_work_cache, :grouped_fft, Dict())
    batched_calls = sum(w.forward_calls for w in values(workspaces); init=0)
    result = [copy(Tarang.get_local_data(grid_data!(field))) for field in (u, v)]
    close(dist)
    return result, calls, batched_calls
end

@testset "Production multi-field RHS communication batching ($_GR_NP ranks)" begin
    original = Tarang.GROUPED_TRANSFORM_CONFIG.enabled
    try
        for timestepper in (RK222(), SBDF2())
            baseline, separate_calls, no_batches = _gr_run(false, timestepper)
            actual, batched_calls, batches = _gr_run(true, timestepper)
            @test actual ≈ baseline atol=2e-12 rtol=2e-12
            @test no_batches == 0
            if _GR_NP > 1
                @test batches > 0
                @test batched_calls < separate_calls
                MPI.Comm_rank(_GR_COMM) == 0 && @info "RHS transpose calls" timestepper separate_calls batched_calls batches
            else
                @test batches == 0
            end
        end
    finally
        Tarang.set_group_transforms!(original)
        _GR_TO.disable_debug_timings(PencilArrays.Transpositions)
    end
end
