# julia --project=<environment with Tarang and Metal> test/run_metal_ci.jl --require-device
using Test, Tarang, Metal, LinearAlgebra
using Random: MersenneTwister, rand!
using Tarang: MPI, FFTW, RKGFY, MCNAB2, CNLF2

include("gpu_boundary_cases.jl")
include("gpu_workspace_cases.jl")
include("stochastic_forcing_layout_cases.jl")
include("gpu_float32_cases.jl")
include("timestepper_fourier_cases.jl")

@testset "Apple Metal" begin
    ext = Base.get_extension(Tarang, :TarangMetalExt)
    @test ext !== nothing
    @test Tarang.MatSolvers.get_solver(:metal_dense) === ext.MetalDenseLU
    @test which(Tarang._gpu_forward_transform_backend!, Tuple{ext.MetalGPU,ScalarField}).module === ext
    @test has_metal() == Metal.functional()
    if !has_metal()
        @test_throws ErrorException GPU(backend=:metal)
        "--require-device" in ARGS && error("A functional Metal device is required")
        @test_skip has_metal()
    else
        arch = GPU(backend=:metal)
        check_variable_timestep_cases(arch, Float32)
        check_forcing_grid_registration(arch)
        check_gpu_workspace_cases(arch, ext, () -> Metal.alloc_stats.alloc_bytes)
        check_float32_device_paths(arch)
        @testset "Complex LU on the device" begin
            A = ComplexF32[0 2+im; 3-im 1]
            b = ComplexF32[4+im, 1-im]
            m = Tarang.MatSolvers.solver_instance(:metal_dense, A)
            sol = Tarang.MatSolvers.solve(m, on_architecture(arch, b))
            @test sol isa Metal.MtlArray{ComplexF32}
            @test A * Array(sol) ≈ b
            rhs = on_architecture(arch, hcat(b, 2b))
            Tarang.MatSolvers.solve!(rhs, m, rhs)
            @test A * Array(rhs) ≈ hcat(b, 2b)
            before = Metal.alloc_stats.alloc_bytes
            Tarang.MatSolvers.solve!(rhs, m, rhs)
            Metal.synchronize()
            @test Metal.alloc_stats.alloc_bytes == before
            @test A * (A * Array(rhs)) ≈ hcat(b, 2b)
            empty_lu = Tarang.MatSolvers.solver_instance(:metal_dense, zeros(ComplexF32, 0, 0))
            @test isempty(Tarang.MatSolvers.solve(empty_lu, Metal.zeros(ComplexF32, 0)))
            @test_throws Union{SingularException,ZeroPivotException} Tarang.MatSolvers.solver_instance(:metal_dense, zeros(ComplexF32, 2, 2))
        end
        @testset "Periodic constraint and nonlinear evolution" begin
            dom = PeriodicDomain(12; device=arch)
            u, phi = ScalarField(dom, "u"), ScalarField(dom, "phi")
            tau = ScalarField(dom.dist, "tau")
            set!(u, sin)
            initial = Array(grid_data!(u))
            problem = InitialValueProblem([u, phi, tau])
            add_equation!(problem, "dt(u) = 0")
            add_equation!(problem, "lap(phi) + tau - u = 0")
            add_bc!(problem, "integ(phi) = 0")
            step!(InitialValueSolver(problem, RK222(); dt=0.01))
            @test Array(grid_data!(phi)) ≈ -initial atol=2f-5

            set!(u, 1f0)
            nonlinear = InitialValueProblem([u])
            add_equation!(nonlinear, "dt(u) = -u*u")
            s = InitialValueSolver(nonlinear, ETD_RK222(); dt=0.01)
            for _ in 1:3
                step!(s)
            end
            @test maximum(abs, Array(grid_data!(u)) .- inv(1 + s.sim_time)) < 3f-6
            close(dom.dist)
        end
        @testset "Coupled boundary and time-dependent solves" begin
            check_steady_boundary(arch; T=Float32, expected_solver=ext.MetalDenseLU{arch.device.registryID},
                                  alloc_bytes=() -> Metal.alloc_stats.alloc_bytes)
        end
        @testset "Device and precision" begin
            @test Tarang.device(arch) isa Metal.MetalBackend
            @test GPU().device == arch.device || has_cuda() || has_amdgpu()
            @test_throws ArgumentError GPU(backend=:metal, device_id=length(Metal.devices()))
            a = on_architecture(arch, reshape(Float32.(1:12), 3, 4))
            @test a isa Metal.MtlArray{Float32}
            @test architecture(view(a, :, 2:3)) == arch
            @test on_architecture(CPU(), view(a, :, 2:3)) == reshape(Float32.(4:9), 3, 2)
            @test on_architecture(CPU(), view(a, 1:2:3, :)) == reshape(Float32.(1:12), 3, 4)[1:2:3, :]
            @test_throws ErrorException on_architecture(arch, Float64[1,2])
            @test Array(a .* a .+ 1f0) == reshape(Float32.(1:12).^2 .+ 1, 3, 4)
            Tarang._seed_device_rng!(arch, 31)
            seeded = Array(rand!(a))
            Tarang._seed_device_rng!(arch, 31)
            @test Array(rand!(a)) == seeded
            forcing = StochasticForcing(; field_size=(12,), k_forcing=2, rng=MersenneTwister(7), architecture=arch)
            values = Tarang.generate_forcing!(forcing, 0.0)
            @test values isa Metal.MtlArray{ComplexF32}
            @test iszero(first(Array(values))) && any(!iszero, Array(values))
            @test_throws ArgumentError Tarang._validate_distributor_size(arch, 2)
            @test_throws ArgumentError PeriodicDomain(8; device=arch, dtype=Float64)
            dist = Distributor(CartesianCoordinates("x"); device=arch)
            @test dist.dtype === Float32
            @test_throws ArgumentError ScalarField(dist, "u", (), ComplexF64)
            close(dist)
        end
        @testset "Multidimensional real FFT with singleton reduced axis" begin
            a = randn(MersenneTwister(85), Float32, 1, 4, 5)
            g = on_architecture(arch, a)
            pf, pb = ext._plan_fft_region(g, (1,3); real_input=true)
            c, restored = similar(g, ComplexF32), similar(g)
            mul!(c, pf, g)
            reference = FFTW.rfft(a, (1,3))
            @test Array(c) ≈ reference rtol=2f-5
            mul!(restored, pb, c)
            @test Array(restored) ≈ a rtol=2f-5
            @test Array(c) ≈ reference rtol=2f-5
        end
        @testset "Complex Fourier field" begin
            coord = CartesianCoordinates("x")
            dist = Distributor(coord; device=arch, dtype=ComplexF32)
            basis = ComplexFourier(coord["x"]; size=11)
            u = ScalarField(Domain(dist, (basis,)), "u")
            initial = randn(MersenneTwister(56), ComplexF32, 11)
            copyto!(grid_data!(u), initial)
            @test Array(coeff_data!(u)) ≈ FFTW.fft(initial) rtol=2f-5
            @test Array(grid_data!(u)) ≈ initial rtol=2f-5
            close(dist)
        end
        for shape in ((1,), (1,5), (7,), (8,9), (9,6,5))
            @testset "Fourier transforms $shape" begin
                dom = PeriodicDomain(shape...; device=arch)
                u = ScalarField(dom, "u")
                data = randn(MersenneTwister(23), Float32, shape)
                copyto!(grid_data!(u), data)
                coeff = coeff_data!(u)
                @test coeff isa Metal.MtlArray{ComplexF32}
                @test Array(coeff) ≈ FFTW.rfft(data) rtol=2f-5
                @test Array(grid_data!(u)) ≈ data rtol=2f-5
                close(dom.dist)
            end
        end
        @testset "Fourier–Chebyshev transform and derivatives" begin
            dom = ChannelDomain(9, 10; arch, Lz=1.0)
            u = ScalarField(dom, "u")
            x = Float32.(2π .* (0:8) ./ 9)
            z = (1f0 .- cospi.((0:9) ./ 9f0)) ./ 2f0
            initial = [sin(xx) * zz^2 for xx in x, zz in z]
            set!(u, (x,z) -> sin(x)*z^2)
            @test Array(grid_data!(u)) ≈ initial rtol=2f-5
            ref = ScalarField(ChannelDomain(9, 10; dtype=Float32, Lz=1.0), "ref")
            copyto!(grid_data!(ref), initial)
            @test Array(coeff_data!(u)) ≈ coeff_data!(ref) rtol=2f-5
            @test Array(grid_data!(u)) ≈ initial rtol=2f-5
            dz = Tarang.evaluate_differentiate(Differentiate(u, dom.dist.coordsys["z"], 1), :g)
            @test Array(grid_data!(dz)) ≈ [2zz*sin(xx) for xx in x, zz in z] atol=1f-4
            grids = Tarang.get_grid_coordinates(dom)
            @test grids["z"] isa Metal.MtlArray{Float32}
            @test Array(grids["z"]) ≈ z
            products = map((ref, u)) do f
                evaluator = Tarang.NonlinearEvaluator(f.dist; dealiasing_factor=3/2)
                product = Tarang.evaluate_transform_multiply(f, f, evaluator; result_layout=:g)
                @test architecture(grid_data!(product)) == f.dist.architecture
                Array(grid_data!(product))
            end
            @test products[2] ≈ products[1] atol=3f-5
            for f in (u, ref)
                preset_scales!(f, (4/3, 3/2))
                ensure_layout!(f, :g)
            end
            scaled = randn(MersenneTwister(82), Float32, size(grid_data!(ref)))
            for f in (u, ref)
                copyto!(grid_data!(f), scaled)
            end
            @test Array(coeff_data!(u)) ≈ coeff_data!(ref) rtol=4f-5
            @test Array(grid_data!(u)) ≈ grid_data!(ref) rtol=4f-5
            close(dom.dist)
            close(ref.dist)
        end
        for ts in (RK111(), RK222(), RK443(), RKSMR(), RKGFY(), SBDF2(),
                   CNAB1(), CNAB2(), MCNAB2(), CNLF2(), SBDF1(), SBDF3(), SBDF4(),
                   ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
            @testset "Periodic diffusion with $(typeof(ts))" begin
                dom = PeriodicDomain(12; device=arch)
                u = ScalarField(dom, "u")
                initial = sinpi.(Float32.(0:11) ./ 6f0)
                copyto!(grid_data!(u), initial)
                p = InitialValueProblem([u])
                if !(ts isa Union{RK111,RK222,RK443,RKSMR,RKGFY,SBDF2})
                    add_equation!(p, "dt(u) = 0.5*lap(u)")
                else
                    add_equation!(p, "dt(u) - 0.5*lap(u) = 0")
                end
                s = InitialValueSolver(p, ts; dt=0.01)
                for _ in 1:4
                    step!(s)
                end
                @test s.iteration == 4
                @test grid_data!(u) isa Metal.MtlArray{Float32}
                expected = exp(-0.5*s.sim_time) .* initial
                @test maximum(abs, Array(grid_data!(u)) .- expected) < 7f-5
                close(dom.dist)
            end
        end
    end
end
