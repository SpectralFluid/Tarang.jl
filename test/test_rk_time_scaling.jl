using Test, Tarang

# Rescaling rate and dt inversely must preserve the numerical trajectory.
# Large RHS values make dt*A significant even when dt*A itself is tiny.
function rkscale_solver(ts, dt; coupled=false, batched=false, equilibrium=true)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
    zb = ChebyshevT(coords["z"]; size=6, bounds=(0.0, 1.0))
    u = ScalarField(Domain(dist, coupled ? (xb, zb) : (xb,)), "u")
    # Populate every Fourier mode so a leftover mode in a partial batch cannot
    # pass trivially by staying zero. No spatial derivatives enter this ODE.
    xs = (0:7) .* (2pi/8)
    profile = [1 + 0.05sum(cos(k*x) for k in 1:4) for x in xs]
    zs = (1 .- cos.(pi .* (0:5) ./ 5)) ./ 2
    initial = coupled ? profile .* reshape(1 .+ 0.2zs, 1, :) : profile
    copyto!(grid_data!(u), initial)
    problem = InitialValueProblem([u])
    add_parameters!(problem; rate=0.1/dt, fraction=equilibrium ? 1.0 : 0.4)
    # There are no spatial derivatives, so the Chebyshev case needs no BC/tau
    # rows. It exercises the subproblem update with a regular mass matrix.
    add_equation!(problem, "dt(u) + rate*u = fraction*rate*u")
    return InitialValueSolver(problem, ts; dt, batched_modes=batched), u, initial
end

function rkscale_weighted()
    ts = RK222()
    # A consistent first-order quadrature using the same stages, but a separate
    # final update. Public tableaux otherwise all take the last stage directly.
    ts.b_explicit .= [1.0, 0.0, 0.0]
    return ts
end

function rkscale_partial_batch!(solver)
    sps = solver.problem.compiled.subproblems
    indices = first(values(Tarang.bucket_subproblems(collect(sps))))
    @test length(indices) >= 3
    state = solver.timestepper_state
    fields = state.timestepper_data[:_sp_state_fields][2]
    foreach(f -> ensure_layout!(f, :c), fields)
    batch = Tarang.build_mode_batch(sps, indices[1:end-1]; like=ComplexF64[])
    plan = Tarang._build_batched_rk_plan(solver, sps, fields; batches=[batch])
    @test plan !== nothing
    @test plan.leftovers == [indices[end]]
    state.timestepper_data[:_sp_rk_mode_batches] = plan
    state.timestepper_data[:_sp_rk_mode_batches_key] = sps
end

@testset "RK trajectories are invariant under a change of time units" begin
    for (path, coupled, batched) in (("global", false, false),
                                     ("subproblem", true, false),
                                     ("batched", true, true),
                                     ("partial batch", true, true))
        for constructor in (RK111, RK222, RK443, RKSMR, Tarang.RKGFY, rkscale_weighted)
            @testset "$path / $constructor" begin
                for equilibrium in (true, false)
                    results = map((0.1, 1e-15)) do dt
                        solver, u, initial = rkscale_solver(constructor(), dt; coupled, batched, equilibrium)
                        for n in 1:3
                            step!(solver)
                            path == "partial batch" && n == 1 && rkscale_partial_batch!(solver)
                        end
                        @test (Tarang._timestepper_subproblems(solver) !== nothing) == coupled
                        @test !isempty(Tarang.active_mode_batches(solver)) == batched
                        values = Array(grid_data!(u))
                        equilibrium && @test values ≈ initial atol=2e-12
                        values
                    end
                    @test results[1] ≈ results[2] atol=2e-12 rtol=2e-12
                end
            end
        end
    end
end
