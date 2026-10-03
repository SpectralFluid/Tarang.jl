using Test, Tarang, LinearAlgebra, SparseArrays, MPI
MPI.Initialized() || MPI.Init()

# A real dense solve records which task constructed and used each factor.
# This detects a production path that accidentally bypasses the worker phase,
# while checking that lazy factor construction remains on the coordinator.
struct LocalModeProbeSolver <: Tarang.MatSolvers.AbstractMatSolver
    inner::Tarang.MatSolvers.DenseLUSolver{ComplexF64}
    constructed_by::Task
    solved_by::Base.RefValue{Union{Nothing,Task}}
end
function LocalModeProbeSolver(matrix::AbstractMatrix; kwargs...)
    LocalModeProbeSolver(Tarang.MatSolvers.DenseLUSolver(matrix), current_task(),
                         Ref{Union{Nothing,Task}}(nothing))
end
function Tarang.MatSolvers.solve!(dest, factor::LocalModeProbeSolver, rhs)
    factor.solved_by[] = current_task()
    yield()
    Tarang.MatSolvers.solve!(dest, factor.inner, rhs)
end

# The same file runs under one or several MPI ranks. Both solvers use the same
# decomposition; every comparison is local and every rank checks its own data.
function local_mode_thread_case(ts, threaded;
                               reverse_axes=MPI.Comm_size(MPI.COMM_WORLD) > 1, matsolver=:sparse,
                               nx=32, nz=18, nonlinear=false)
    coords = reverse_axes ? CartesianCoordinates("z", "x") : CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2pi), dealias=3/2)
    zb = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    bases = reverse_axes ? (zb, xb) : (xb, zb)
    domain = Domain(dist, bases)
    u = ScalarField(domain, "u")
    source = ScalarField(domain, "source")
    # Distributed 1D Fourier domains are unsupported; the MPI per-mode runtime
    # keeps these scalar taus in each subproblem's stash (as in existing tests).
    tau_bases = dist.size > 1 ? () : (xb,)
    tau1 = ScalarField(dist, "tau1", tau_bases, Float64)
    tau2 = ScalarField(dist, "tau2", tau_bases, Float64)
    phi(x,z) = 1 + z + cos(x)*exp(z) + 0.3*cos(2x)*exp(2z)
    profile = reverse_axes ? ((z,x)->phi(x,z)) : phi
    mesh = Tarang.create_meshgrid(domain; on_device=false)
    values = profile.((mesh[b.meta.element_label] for b in bases)...)
    for field in (u,source)
        data = grid_data!(field)
        if data isa Tarang.PencilArrays.PencilArray
            parent(data) .= values[Tarang.PencilArrays.pencil(data).axes_local...]
        else
            data .= values
        end
    end
    initial = copy(parent(grid_data!(u)))
    problem = InitialValueProblem([u,tau1,tau2])
    lb = derivative_basis(zb, 2)
    add_parameters!(problem; source, l1=lift(tau1,lb,-1), l2=lift(tau2,lb,-2))
    add_equation!(problem, "dt(u) - 0.2*lap(u) + l1 + l2 = " *
                  (nonlinear ? "source - 0.01*u*d(u,x)" : "source"))
    add_bc!(problem, "u(z=0)=(1+t)*(1+cos(x)+0.3*cos(2*x))")
    add_bc!(problem, "u(z=1)=(1+t)*(2+$(exp(1.0))*cos(x)+$(0.3exp(2))*cos(2*x))")
    solver = InitialValueSolver(problem,ts; dt=0.01, threaded_modes=threaded,
                                batched_modes=false, matsolver)
    return solver, u, initial
end

@testset "Local mode scheduler ownership and failures" begin
    counts = zeros(Int, 29)
    owner = current_task()
    jobs = Vector{Task}(undef, length(counts))
    Tarang._foreach_local_mode!(length(counts), true) do i
        counts[i] += 1
        jobs[i] = current_task()
        yield() # Migration/yielding must not cause duplicate ownership.
    end
    @test counts == ones(Int, length(counts))
    if Threads.nthreads(:default) > 1
        @test all(task -> task !== owner, jobs)
        @test length(unique(jobs)) == min(length(counts), Threads.nthreads(:default))
    else
        @test all(task -> task === owner, jobs)
    end
    Tarang._foreach_local_mode!(length(counts), false) do i
        @test current_task() === owner
    end
    completed = zeros(Int, 8)
    err = try
        Tarang._foreach_local_mode!(8, true) do i
            i == 1 && error("local-mode-probe")
            completed[i] = 1
        end
        nothing
    catch caught
        caught
    end
    @test err !== nothing
    @test occursin("local-mode-probe", sprint(showerror,err))
    # A thrown worker error is joined before returning: no worker keeps writing.
    saved = copy(completed)
    yield()
    @test completed == saved
end

@testset "Automatic local threading respects work and BLAS budgets" begin
    mode = (M_min=spzeros(ComplexF64,256,256), dist=(architecture=CPU(),))
    modes = ntuple(_ -> mode, max(8,2Threads.nthreads(:default)))
    previous_blas_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        @test Tarang._thread_local_modes(nothing,modes) == (Threads.nthreads(:default) > 1)
        @test !Tarang._thread_local_modes(nothing,(mode,))
        @test !Tarang._thread_local_modes(nothing,modes,LocalModeProbeSolver)
        @test Tarang._thread_local_modes(true,modes,LocalModeProbeSolver) == (Threads.nthreads(:default) > 1)
        BLAS.set_num_threads(2)
        @test !Tarang._thread_local_modes(nothing,modes)
        @test Tarang._thread_local_modes(true,modes) == (Threads.nthreads(:default) > 1)
        @test !Tarang._thread_local_modes(false,modes)
        non_cpu = (M_min=mode.M_min, dist=(architecture=nothing,))
        @test !Tarang._thread_local_modes(true,(mode,non_cpu))
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end
end

@testset "Production factors stay local to one worker" begin
    owner = current_task()
    for ts in (RK222(),SBDF2()), threaded in (false,true)
        solver,_,_ = local_mode_thread_case(ts,threaded;matsolver=LocalModeProbeSolver)
        for _ in 1:2
            step!(solver,0.01)
            factors = solver.timestepper_state.timestepper_data[:_local_mode_factors]
            @test all(factor -> factor isa LocalModeProbeSolver, factors)
            @test all(factor -> factor.constructed_by === owner, factors)
            if threaded && Threads.nthreads(:default) > 1
                @test all(factor -> factor.solved_by[] !== nothing &&
                                    factor.solved_by[] !== owner, factors)
            else
                @test all(factor -> factor.solved_by[] === owner, factors)
            end
            @test length(unique(objectid(factor.solved_by) for factor in factors)) == length(factors)
        end
    end
end

@testset "Threaded RK/multistep variable dt and moving boundaries" begin
    # Mixed MPI domains require coupled axes first; serial supports both orders.
    axis_orders = MPI.Comm_size(MPI.COMM_WORLD) > 1 ? (true,) : (false,true)
    for ts in (RK222(), RK443(), RKSMR(), SBDF2(), CNAB2()), reverse_axes in axis_orders
        @testset "$(typeof(ts)) reverse_axes=$reverse_axes" begin
            serial,u_serial,_ = local_mode_thread_case(ts,false;reverse_axes)
            parallel,u,initial = local_mode_thread_case(ts,true;reverse_axes)
            sps = parallel.problem.compiled.subproblems
            @test !Tarang._thread_local_modes(false,sps)
            @test Tarang._thread_local_modes(true,sps) == (Threads.nthreads(:default) > 1)
            @test !Tarang._thread_local_modes(nothing,sps) # small problems stay serial
            for dt in (0.01,0.02,0.015,0.01,0.012)
                step!(serial,dt)
                step!(parallel,dt)
                data = parent(grid_data!(u))
                @test data ≈ parent(grid_data!(u_serial)) atol=2e-11 rtol=2e-11
                @test maximum(abs,data .- (1+parallel.sim_time).*initial) < 2e-8
                for (field, reference) in zip(parallel.state,serial.state)
                    @test parent(coeff_data!(field)) ≈ parent(coeff_data!(reference)) atol=2e-10 rtol=2e-10
                end
            end
            # Workspaces persist and each local mode owns a distinct RHS/result.
            cache = parallel.timestepper_state.timestepper_data
            key = ts isa Union{SBDF2,CNAB2} ? :_sp_ms_solutions : :_sp_rk_solutions
            solutions = cache[key]
            active = [i for i in eachindex(sps) if sps[i].M_min !== nothing]
            @test length(unique(objectid(solutions[i]) for i in active)) == length(active)
            saved = [solutions[i] for i in active]
            step!(parallel,0.012)
            @test all(solutions[i] === buffer for (i,buffer) in zip(active,saved))
        end
    end
end

@testset "Threaded mode solves preserve nonlinear RHS evaluation" begin
    for ts in (RK222(),SBDF2())
        serial,u_serial,_ = local_mode_thread_case(ts,false;nonlinear=true)
        parallel,u,_ = local_mode_thread_case(ts,true;nonlinear=true)
        for dt in (0.001,0.002,0.0015,0.001)
            step!(serial,dt); step!(parallel,dt)
            @test parent(grid_data!(u)) ≈ parent(grid_data!(u_serial)) atol=2e-11 rtol=2e-11
        end
    end
end
