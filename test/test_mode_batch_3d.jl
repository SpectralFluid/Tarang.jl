using Test
using Tarang

function _optimized_channel_solver_3d(; nx=6, ny=5, nz=8, dt=1e-3, ts=RK222(),
                                     device=Tarang.CPU(), kwargs...)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=Float64, device=device)
    xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2π), dealias=3 / 2)
    ybasis = RealFourier(coords["y"]; size=ny, bounds=(0.0, 2π), dealias=3 / 2)
    zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xbasis, ybasis, zbasis))

    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xbasis, ybasis), Float64)
    tau2 = ScalarField(dist, "tau2", (xbasis, ybasis), Float64)
    _, _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    tau_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * tau_lift(tau1)

    problem = InitialValueProblem([b, tau1, tau2])
    add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
    add_equation!(problem,
                  "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 1 + sin(t)")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, ts; dt, kwargs...)
    return solver, b
end

function _check_3d_mode_batch_parity(; device=CPU())
    for ts in (RK111(), RK222(), RK443())
        ref, ref_b = _optimized_channel_solver_3d(; ts, batched_modes=false)
        bat, bat_b = _optimized_channel_solver_3d(; ts, device,
            batched_modes=device isa CPU ? true : nothing)
        host = [0.1*sin(2pi*i/6)*cos(2pi*j/5)*sin(pi*(k-1)/7)
                for i in 1:6, j in 1:5, k in 1:8]
        ref_b["g"] = host
        bat_b["g"] = host
        for dt in (1e-3, 1e-3, 7e-4)
            step!(ref, dt)
            step!(bat, dt)
        end
        plan = get(bat.timestepper_state.timestepper_data, :_sp_rk_mode_batches, nothing)
        @test plan !== nothing
        plan === nothing && continue
        @test isempty(plan.leftovers)
        @test sum(b -> b.nmodes, plan.batches) == 20
        ensure_layout!(ref_b, :g)
        ensure_layout!(bat_b, :g)
        @test Array(get_grid_data(bat_b)) ≈ Array(get_grid_data(ref_b)) rtol=2e-10 atol=2e-11
        # Every field, including the reduced-rank Fourier tau fields, agrees
        # with the established per-mode gather at asymmetric Fourier sizes.
        fields = bat.timestepper_state.timestepper_data[:_sp_state_fields][2]
        foreach(f -> ensure_layout!(f, :c), fields)
        sps = bat.problem.compiled.subproblems
        for (batch, ws) in zip(plan.batches, plan.workspaces)
            device isa CPU || @test Tarang.is_gpu_array(ws.X0)
            Tarang._batched_gather_state!(ws.X0, ws, batch, fields)
            packed = Array(ws.X0)
            for (m, index) in enumerate(batch.sp_indices)
                @test packed[:, m] ≈ Array(Tarang.gather_inputs(sps[index], fields))
            end
            before = [copy(Array(get_coeff_data(f))) for f in fields]
            Tarang._batched_scatter_state!(ws, batch, fields, ws.X0)
            for (field, original) in zip(fields, before)
                @test Array(get_coeff_data(field)) ≈ original
            end
        end
    end
end

@testset "3D coupled RK batching parity and gather/scatter" begin
    _check_3d_mode_batch_parity()
end

@testset "3D batching cap includes RK working storage" begin
    solver, _ = _optimized_channel_solver_3d(; batched_modes=true)
    step!(solver)
    sps = solver.problem.compiled.subproblems
    fields = solver.timestepper_state.timestepper_data[:_sp_state_fields][2]
    foreach(f -> ensure_layout!(f, :c), fields)
    indices = first(values(Tarang.bucket_subproblems(sps)))
    # A cap that can hold only the matrix pack must not allocate additional
    # full state, equation, and stage matrices beyond that cap.
    solver.base.batched_modes_max_bytes = Tarang.mode_batch_bytes(sps[first(indices)], length(indices))
    plan = Tarang._build_batched_rk_plan(solver, sps, fields)
    @test plan === nothing
end

@testset "RK cap is shared across distinct matrix buckets" begin
    solver, _ = _optimized_channel_solver_3d(; batched_modes=true)
    step!(solver)
    sps = solver.problem.compiled.subproblems
    fields = solver.timestepper_state.timestepper_data[:_sp_state_fields][2]
    foreach(f -> ensure_layout!(f, :c), fields)
    live = [sp for sp in sps if sp.M_min !== nothing]
    for sp in live[end-1:end]
        matrix = copy(sp.LHS)
        row = findfirst(r -> iszero(matrix[r, 1]), axes(matrix, 1))
        @test row !== nothing
        matrix[row, 1] = 1.0
        sp.LHS = matrix
    end
    buckets = collect(values(Tarang.bucket_subproblems(sps)))
    @test length(buckets) == 2
    costs = [Tarang.mode_batch_bytes(sps[first(ids)], length(ids)) +
             Tarang._mode_batch_rk_workspace_bytes(sps[first(ids)], length(ids), fields, solver.timestepper.stages)
             for ids in buckets]
    solver.base.batched_modes_max_bytes = maximum(costs)
    plan = Tarang._build_batched_rk_plan(solver, sps, fields)
    @test plan !== nothing
    @test length(plan.batches) == 1
    @test !isempty(plan.leftovers)
    retained = sum(Tarang.mode_batch_bytes(sps[first(b.sp_indices)], b.nmodes) +
                   Tarang._mode_batch_rk_workspace_bytes(sps[first(b.sp_indices)], b.nmodes, fields, solver.timestepper.stages)
                   for b in plan.batches)
    @test retained <= solver.base.batched_modes_max_bytes
end
