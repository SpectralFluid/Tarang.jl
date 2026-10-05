module TimestepWorkspace3DTests
using Test, Tarang

@testset "Owned timestep workspaces grow only when requested" begin
    domain = PeriodicDomain(8, 6, 5)
    u, v = [ScalarField(domain, name) for name in ("u", "v")]
    u["g"] .= 3
    state = Tarang.TimestepperState(RK222(), 0.01, [u, v]; workspace_sets=0)
    @test isempty(state.workspace_fields)
    @test !state.workspace_allocated
    @test_throws ArgumentError Tarang.get_workspace_field!(state, u, 0)
    @test_throws ArgumentError Tarang.TimestepperState(RK222(), 0.01, [u]; workspace_sets=-1)
    last = Tarang.get_workspace_field!(state, v, 4)
    @test getproperty.(state.workspace_fields, :name) == ["u", "v", "u", "v"]
    @test state.workspace_allocated
    @test Tarang.get_workspace_field!(state, v, 4) === last
    @test length(state.workspace_fields) == 4
    fill!(get_grid_data(state.workspace_fields[1]), 99)
    @test state.history[end][1]["g"] == fill(3.0, 8, 6, 5)
    @test u["g"] == fill(3.0, 8, 6, 5)
    for i in 1:4, j in 1:i-1
        @test get_grid_data(state.workspace_fields[i]) !== get_grid_data(state.workspace_fields[j])
        @test get_coeff_data(state.workspace_fields[i]) !== get_coeff_data(state.workspace_fields[j])
    end
    reserved = Tarang.TimestepperState(RK222(), 0.01, [u, v])
    @test length(reserved.workspace_fields) == 2Tarang._workspace_count(RK222())
    close(domain.dist)
end

function mixed_solver(device, ts; matsolver=nothing)
    cs = CartesianCoordinates("x", "y", "z")
    dist = Distributor(cs; dtype=Float64, device)
    xb = RealFourier(cs["x"]; size=6, bounds=(0.0, 2pi))
    yb = RealFourier(cs["y"]; size=5, bounds=(0.0, 2pi))
    zb = ChebyshevT(cs["z"]; size=12, bounds=(0.0, 1.0))
    domain = Domain(dist, (xb, yb, zb))
    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xb, yb), Float64)
    tau2 = ScalarField(dist, "tau2", (xb, yb), Float64)
    ez = last(unit_vector_fields(cs, dist))
    lb = derivative_basis(zb, 1)
    tau_lift(a) = lift(a, lb, -1)
    grad_b = grad(b) + ez*tau_lift(tau1)
    problem = InitialValueProblem([b, tau1, tau2])
    add_parameters!(problem; grad_b, tau_lift)
    add_equation!(problem, "dt(b) - 0.1*div(grad_b) + tau_lift(tau2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    mesh = Tarang.get_grid_coordinates(domain; on_device=false)
    initial = [0.1sin(x)*cos(y)*sin(pi*z) for x in mesh["x"], y in mesh["y"], z in mesh["z"]]
    copyto!(get_grid_data(b), initial)
    kwargs = matsolver === nothing ? (;) : (; matsolver)
    solver = InitialValueSolver(problem, ts; dt=1e-3, batched_modes=false, kwargs...)
    solver, b
end

@testset "3D mixed solvers need no generic field workspaces" begin
    for ts in (RK222(), SBDF2())
        solver, b = mixed_solver(CPU(), ts)
        # Compare the lazy reservation with the old eager reservation. Each
        # solver owns a distinct problem so stage or tau caches cannot alias.
        reference, rb = mixed_solver(CPU(), ts)
        reference.timestepper_state = Tarang.TimestepperState(ts, 1e-3, reference.state)
        for dt in (1e-3, 1e-3, 7e-4)
            step!(solver, dt)
            step!(reference, dt)
            @test b["c"] ≈ rb["c"] atol=2e-12 rtol=3e-10
            @test isempty(solver.timestepper_state.workspace_fields)
        end
        @test !solver.timestepper_state.workspace_allocated
        @test all(isfinite, b["c"])
        close(b.dist); close(rb.dist)
    end
end

const _BCJL_OK = try
    @eval using JLArrays, GPUArrays
    true
catch
    false
end
if _BCJL_OK
    include("gpu_boundary_jlarray_support.jl")
    GPUArrays.allowscalar(false)
    @testset "3D mixed device stages agree with CPU without generic workspaces" begin
        cpu, cb = mixed_solver(CPU(), RK222())
        gpu, gb = mixed_solver(_BCJL_ARCH, RK222(); matsolver=BoundaryJLHostLU)
        for dt in (1e-3, 1e-3, 7e-4)
            step!(cpu, dt); step!(gpu, dt)
            @test Array(gb["c"]) ≈ cb["c"] atol=2e-12 rtol=3e-10
            @test isempty(gpu.timestepper_state.workspace_fields)
        end
        @test get_coeff_data(gb) isa JLArray
        close(cb.dist); close(gb.dist)
    end

    @testset "3D device field RK retains required workspace slots" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; dtype=Float64, device=_BCJL_ARCH)
        bases = ntuple(i -> RealFourier(coords[("x", "y", "z")[i]];
                                        size=(8, 7, 6)[i]), 3)
        q = ScalarField(Domain(dist, bases), "q")
        q["g"] .= 1
        problem = InitialValueProblem([q])
        add_equation!(problem, "dt(q) = q")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)
        state = Tarang._ensure_timestepper_state!(solver, 1e-3)
        @test isempty(state.workspace_fields)
        step!(solver)
        owned = copy(state.workspace_fields)
        @test length(owned) == state.timestepper.stages + 1
        for _ in 1:3
            step!(solver)
            @test all(a === b for (a, b) in zip(owned, state.workspace_fields))
            @test length(state.workspace_fields) == length(owned)
        end
        @test Array(q["g"]) ≈ fill(exp(solver.sim_time), 8, 7, 6) rtol=3e-9
        close(dist)
    end
else
    @testset "3D device timestep workspace ownership" begin
        @test_skip "JLArrays/GPUArrays unavailable"
    end
end
end
