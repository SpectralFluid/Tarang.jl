using Test, Tarang, FFTW, Random

const _SG_CUDA_OK = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if !_SG_CUDA_OK
    lowercase(get(ENV, "TARANG_REQUIRE_CUDA", "false")) in ("true", "1", "yes") &&
        error("Single-GPU optimization validation requires functional CUDA")
    @testset "GPU nonlinear optimizations" begin
        @test_skip "No functional CUDA device"
    end
else
    CUDA.allowscalar(false)
    @testset "GPU nonlinear products remain stream ordered" begin
        for shape in ((9, 10), (10, 9, 8)), T in (Float32, Float64, ComplexF64), mixed in (false, true)
            names = ("x", "y", "z")[1:length(shape)]
            coords = CartesianCoordinates(names...)
            bases = ntuple(length(shape)) do d
                mixed && d == 1 ? ChebyshevT(coords[d]; size=shape[d]) :
                    (T <: Real ? RealFourier : ComplexFourier)(coords[d]; size=shape[d], dealias=1.5)
            end
            function fields(device)
                dist = Distributor(coords; dtype=T, device)
                domain = Domain(dist, bases)
                return ntuple(i -> ScalarField(domain, "stream_$i"), 3)
            end
            ca, cb, co = fields(CPU())
            ga, gb, go = fields(GPU())
            rng = MersenneTwister(411)
            aa, bb = randn(rng, T, shape), randn(rng, T, shape)
            copyto!(grid_data!(ca), aa); copyto!(grid_data!(cb), bb)
            copyto!(grid_data!(ga), aa); copyto!(grid_data!(gb), bb)
            Tarang._dealiased_lazy_product!(co, ca, cb)
            reference = Array(grid_data!(co))
            # Build plans on the original task stream, then execute on another.
            Tarang._dealiased_lazy_product!(go, ga, gb)
            CUDA.synchronize()
            stream = CUDA.CuStream()
            CUDA.stream!(stream) do
                for i in 1:12
                    get_grid_data(ga) .*= T(0.99)
                    Tarang._dealiased_lazy_product!(go, ga, gb)
                    get_grid_data(go) .*= T(2)
                end
                # One wait at the end; no host read or barrier between products.
                CUDA.synchronize()
            end
            tol = 500eps(real(T))
            @test Array(grid_data!(go)) ≈ reference .* (T(2)*T(0.99)^12) rtol=tol atol=tol
            ev = Tarang._get_evaluator(ga.dist)
            @test isempty(ev.nl_result_pool)
            if T <: Real && isdefined(CUDA.CUFFT, :unsafe_execute_trailing!)
                ws = Tarang._get_padded_workspace!(ev, ga.bases, T; real_input=true)
                @test ws.plan_backward.p.buffer === ws.padded_spectrum || ws.plan_backward.p.buffer === nothing
                @test ws.plan_spec_backward.p.buffer === ws.spectrum || ws.plan_spec_backward.p.buffer === nothing
                @test ws.plan_forward.buffer === ws.padded_spectrum || ws.plan_forward.buffer === nothing
                @test ws.plan_spec_forward.buffer === ws.spectrum || ws.plan_spec_forward.buffer === nothing
            end
            for _ in 1:3
                Tarang._dealiased_lazy_product!(go, ga, gb)
            end
            CUDA.synchronize()
            before = CUDA.alloc_stats.alloc_bytes
            Tarang._dealiased_lazy_product!(go, ga, gb)
            CUDA.synchronize()
            @test CUDA.alloc_stats.alloc_bytes == before
            # Alias destination/input after both operands have been consumed.
            reference = Array(grid_data!(go))
            Tarang._dealiased_lazy_product!(ga, ga, gb)
            CUDA.synchronize()
            @test Array(grid_data!(ga)) ≈ reference rtol=tol atol=tol
        end
    end

    @testset "GPU CFL reuses device frequency storage" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=GPU())
        bases = ntuple(d -> RealFourier(coords[d]; size=64, bounds=(0.0, 2π)), 2)
        domain = Domain(dist, bases)
        u = VectorField(domain, "u")
        q = ScalarField(domain, "q")
        fill!(grid_data!(u.components[1]), 1.0)
        fill!(grid_data!(u.components[2]), 2.0)
        problem = InitialValueProblem([q])
        add_equation!(problem, "dt(q)=0")
        cfl = CFL(InitialValueSolver(problem, RK111()); initial_dt=1.0,
                  min_change=0.0, threshold=0.0)
        add_velocity!(cfl, u)
        for _ in 1:3
            Tarang.compute_timestep(cfl)
        end
        buffer = cfl.frequency_buffers[u]
        @test buffer isa CUDA.CuArray
        CUDA.synchronize()
        before = CUDA.alloc_stats.alloc_bytes
        dt = Tarang.compute_timestep(cfl)
        CUDA.synchronize()
        # Reduction scratch may remain, but no new full-grid component arrays.
        @test CUDA.alloc_stats.alloc_bytes - before < sizeof(buffer)
        @test cfl.frequency_buffers[u] === buffer
        dx, dy = Tarang.grid_spacing(domain)
        @test dt ≈ cfl.safety / (1/dx + 2/dy)
        fill!(grid_data!(u.components[2]), 4.0)
        @test Tarang.compute_timestep(cfl) ≈ cfl.safety / (1/dx + 4/dy)
    end

    @testset "Real products preserve CPU/GPU operand conversion" begin
        coords = CartesianCoordinates("x", "y")
        bases = ntuple(d -> RealFourier(coords[d]; size=8, dealias=1.5), 2)
        cpu = ScalarField(Domain(Distributor(coords; device=CPU()), bases), "cpu")
        gpu = ScalarField(Domain(Distributor(coords; device=GPU()), bases), "gpu")
        fill!(grid_data!(cpu), 2.0); fill!(grid_data!(gpu), 3.0)
        for (a, b) in ((cpu, gpu), (gpu, cpu))
            ensure_layout!(a, :c); ensure_layout!(b, :c)
            out = evaluate_transform_multiply(a, b, Tarang._get_evaluator(a.dist))
            @test Array(grid_data!(out)) ≈ fill(6.0, 8, 8)
        end
    end
end
