using Test
using Tarang

const _RK_OPT_REQUIRE_CUDA = lowercase(get(ENV, "TARANG_REQUIRE_CUDA", "false")) in ("1", "true", "yes")
const _RK_OPT_HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if !_RK_OPT_HAS_CUDA
    _RK_OPT_REQUIRE_CUDA && error("RK optimization validation requires a functional CUDA device")
    @testset "Single-GPU RK optimizations" begin
        @test_skip "CUDA not functional on this host"
    end
else
    CUDA.allowscalar(false)
    isdefined(@__MODULE__, :_check_3d_mode_batch_parity) || include("test_mode_batch_3d.jl")
    @testset "CUDA 3D coupled RK defaults to batching" begin
        _check_3d_mode_batch_parity(; device=GPU())
    end
    @testset "CUDA fused RK device allocation and parity" begin
        for T in (Float32, Float64, ComplexF64)
            base = CUDA.fill(T(0.125), 4096)
            terms = (CUDA.fill(T(1e8), 4096), CUDA.fill(T(-2e8), 4096), CUDA.fill(T(NaN), 4096))
            weights = (1e-16, -2e-16, 0.0)
            dest = similar(base)
            expected = Array(base)
            for (w, term) in zip(weights, terms)
                iszero(w) || (expected .+= w .* Array(term))
            end
            CUDA.@sync Tarang._rk_combine_arrays!(dest, base, terms, weights)
            @test Array(dest) ≈ expected
            before = CUDA.alloc_stats.alloc_bytes
            CUDA.@sync Tarang._rk_combine_arrays!(dest, base, terms, weights)
            after = CUDA.alloc_stats.alloc_bytes
            @test after == before
            @test Array(dest) ≈ expected
        end
    end
    @testset "CUDA field-native explicit and diagonal IMEX RK parity" begin
        for ts in (RK222(), DiagonalIMEX_RK222(), DiagonalIMEX_RK443())
            solvers = []
            fields = []
            for device in (CPU(), GPU())
                coords = CartesianCoordinates("x")
                dist = Distributor(coords; dtype=Float64, device)
                xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi))
                q = ScalarField(Domain(dist, (xb,)), "q")
                q["g"] = cos.(2pi .* (0:15) ./ 16)
                problem = InitialValueProblem([q])
                add_equation!(problem, ts isa RK222 ? "dt(q) = q" : "dt(q) - lap(q) = 0.25*q")
                push!(solvers, InitialValueSolver(problem, ts; dt=1e-3))
                push!(fields, q)
            end
            for solver in solvers, _ in 1:3
                if ts isa RK222
                    state = Tarang._ensure_timestepper_state!(solver, 1e-3)
                    tableau = state.timestepper
                    Tarang._step_explicit_rk_gpu!(state, solver,
                        tableau.A_explicit, tableau.b_explicit, tableau.c_explicit)
                    Tarang._sync_solver_from_timestepper!(solver)
                    solver.sim_time += state.dt
                else
                    step!(solver)
                end
            end
            foreach(f -> ensure_layout!(f, :g), fields)
            @test get_grid_data(fields[2]) isa CUDA.CuArray
            @test Array(get_grid_data(fields[2])) ≈ Array(get_grid_data(fields[1])) rtol=2e-11 atol=2e-12
        end
    end

end
