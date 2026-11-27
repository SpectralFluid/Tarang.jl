function check_steady_boundary(arch; T=Float64, expected_solver=nothing, alloc_bytes=nothing)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; device=arch, dtype=T, comm=MPI.COMM_SELF)
    try
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=10, bounds=(0.0, 1.0))
        u = ScalarField(Domain(dist, (xb, zb)), "u")
        t1 = ScalarField(dist, "t1", (xb,), T)
        t2 = ScalarField(dist, "t2", (xb,), T)
        db = derivative_basis(zb, 2)
        problem = LinearBoundaryValueProblem([u, t1, t2])
        add_parameters!(problem; l1=lift(t1, db, -1), l2=lift(t2, db, -2))
        add_equation!(problem, "-lap(u) + l1 + l2 = 0")
        add_bc!(problem, "u(z=0) = 1")
        add_bc!(problem, "u(z=1) = 2")
        solver = BoundaryValueSolver(problem; batched_modes=false)
        if expected_solver !== nothing
            @test solver.base.matsolver === expected_solver
            @test_throws ArgumentError BoundaryValueSolver(problem; matsolver=:cuda_sparse)
        end
        solve!(solver)
        z = (1 .- cospi.((0:9) ./ 9)) ./ 2
        values = grid_data!(u)
        @test Tarang.architecture(values) == arch
        @test Array(values) ≈ [1+v for i in 1:8, v in z] atol=max(2e-8, 300eps(T))
        dz = Tarang.evaluate_differentiate(Differentiate(u, coords["z"], 1), :g)
        @test Array(get_grid_data(dz)) ≈ ones(8,10) atol=max(2e-8, 300eps(T))

        initial = T[cospi(i / 4) * sinpi(v) for i in 0:7, v in z]
        ivp = InitialValueProblem([u, t1, t2])
        add_parameters!(ivp; l1=lift(t1, db, -1), l2=lift(t2, db, -2))
        add_equation!(ivp, "dt(u) - 0.2*lap(u) + l1 + l2 = 0")
        add_bc!(ivp, "u(z=0) = 0")
        add_bc!(ivp, "u(z=1) = 0")
        for (ts, batched) in ((RK222(), true), (SBDF2(), false))
            copyto!(grid_data!(u), initial)
            fill!(grid_data!(t1), 0)
            fill!(grid_data!(t2), 0)
            ivs = InitialValueSolver(ivp, ts; dt=0.002, batched_modes=batched)
            for _ in 1:3
                step!(ivs)
            end
            if alloc_bytes !== nothing && ts isa RK222
                Tarang.synchronize(arch)
                before = alloc_bytes()
                step!(ivs)
                Tarang.synchronize(arch)
                @test alloc_bytes() == before
            end
            expected = exp(-0.2 * (1 + π^2) * ivs.sim_time) .* initial
            # SBDF2 starts with a first-order step. Check pointwise solution error.
            @test maximum(abs, Array(grid_data!(u)) .- expected) < 3e-5
        end
    finally
        close(dist)
    end
end
