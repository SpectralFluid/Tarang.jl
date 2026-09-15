using Test
using Tarang
using Random

function _stochastic_restart_solver(; architecture=CPU(), separable=false, seed=42,
                                      rate=0.1, forced=true)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=architecture)
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    yb = separable ? ChebyshevT(coords["y"]; size=8, bounds=(-1.0, 1.0)) :
                     RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
    q = ScalarField(Domain(dist, (xb, yb)), "q")
    fill!(grid_data!(q), 0)
    kwargs = (; architecture, energy_injection_rate=rate, k_forcing=2.0,
                dk_forcing=1.0, dt=0.01, rng=MersenneTwister(seed))
    forcing = separable ? SeparableStochasticForcing(; fourier_size=(8,),
        chebyshev_basis=yb, chebyshev_profile=ones(8), kwargs...) :
        StochasticForcing(; field_size=(8, 8), kwargs...)
    problem = InitialValueProblem([q])
    add_equation!(problem, "∂t(q) = 0")
    forced && add_stochastic_forcing!(problem, :q, forcing)
    return InitialValueSolver(problem, RK222(); dt=0.01), forcing
end

function _test_stochastic_restart(architecture)
    @testset "restart preserves noise, separable=$separable cached=$precache" for
        separable in (false, true), precache in (false, true)
        original, forcing = _stochastic_restart_solver(; architecture, separable)
        for _ in 1:3
            step!(original, 0.01)
        end
        # Include a draw already made at the checkpoint time: restoring the RNG
        # alone would advance past this cached realization on the next step.
        precache && generate_forcing!(forcing, original.sim_time)
        saved_timestamp = forcing.last_update_time
        path = save_state(original, joinpath(mktempdir(), "stochastic"))
        resumed, restored = _stochastic_restart_solver(; architecture, separable, seed=999)
        output_buffer = get_cached_forcing(restored)
        load_state!(resumed, path)
        @test Array(output_buffer) == Array(forcing.cached_forcing)
        @test restored.last_update_time == forcing.last_update_time
        @test Array(restored.cached_forcing) == Array(forcing.cached_forcing)
        @test rand(copy(restored.rng), UInt64, 8) == rand(copy(forcing.rng), UInt64, 8)
        for dt in (0.01, 0.02, 0.02)
            step!(original, dt)
            step!(resumed, dt)
            @test Array(restored.cached_forcing) == Array(forcing.cached_forcing)
            @test Array(grid_data!(resumed.state[1])) ≈ Array(grid_data!(original.state[1]))
        end
        # Reload into a solver whose RNG and cache have already advanced.
        load_state!(resumed, path)
        @test restored.last_update_time == saved_timestamp
    end
end

@testset "Stochastic checkpoint restart" begin
    _test_stochastic_restart(CPU())
    @testset "configuration mismatches fail before restoring fields" begin
        original, _ = _stochastic_restart_solver()
        step!(original, 0.01)
        path = save_state(original, joinpath(mktempdir(), "forcing_config"))
        for kwargs in ((; rate=0.2), (; forced=false))
            target, forcing = _stochastic_restart_solver(; kwargs...)
            before = copy(grid_data!(target.state[1]))
            rng_before = rand(copy(forcing.rng), UInt64, 8)
            @test_throws Exception load_state!(target, path)
            @test grid_data!(target.state[1]) == before
            @test target.sim_time == 0.0
            @test rand(copy(forcing.rng), UInt64, 8) == rng_before
        end
    end
    @testset "missing forcing state is refused" begin
        unforced, _ = _stochastic_restart_solver(; forced=false)
        path = save_state(unforced, joinpath(mktempdir(), "legacy"))
        target, _ = _stochastic_restart_solver()
        @test_throws Exception load_state!(target, path)
    end
    @testset "incompatible RNG format is refused before field mutation" begin
        original, _ = _stochastic_restart_solver()
        step!(original, 0.01)
        path = save_state(original, joinpath(mktempdir(), "format"))
        Tarang.ncputatt(path, "global", Dict("stochastic_forcing_julia" => "0.0"))
        target, forcing = _stochastic_restart_solver()
        before = copy(grid_data!(target.state[1]))
        @test_throws Exception load_state!(target, path)
        @test grid_data!(target.state[1]) == before
        @test forcing.last_update_time == -Inf
    end

end

const _STOCHASTIC_RESTART_JL_AVAILABLE = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch
    false
end

@testset "Stochastic checkpoint CPU/device transfer" begin
    if _STOCHASTIC_RESTART_JL_AVAILABLE
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}) = JLArrays.JLArray
        @eval Tarang.array_type(::Tarang.GPU{<:JLArrays.JLBackend}, ::Type{T}) where T = JLArrays.JLArray{T}
        @eval Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, data::Array) = JLArrays.JLArray(data)
        @eval Tarang.is_gpu_array(::JLArrays.JLArray) = true
        @eval Tarang.architecture(::JLArrays.JLArray) = Tarang.GPU(JLArrays.JLBackend())
        GPUArrays.allowscalar(false)
        arch = Tarang.GPU(JLArrays.JLBackend())
        # JLArray supports the Fourier solver's construction, but has no sparse
        # device solver for coupled Chebyshev fields. The CUDA test below covers
        # the separable case with its actual device solver and transforms.
        for separable in (false,)
            cpu, fc = _stochastic_restart_solver(; separable)
            # Grid-current fields need no device FFT for this transfer test.
            Tarang._update_registered_forcings!(cpu, 0.0, 0.01)
            path = save_state(cpu, joinpath(mktempdir(), "to_device"))
            device, fd = _stochastic_restart_solver(; architecture=arch, separable,
                                                      seed=123)
            load_state!(device, path)
            @test fd.cached_forcing isa JLArrays.JLArray
            @test get_grid_data(device.state[1]) isa JLArrays.JLArray
            @test Array(fd.cached_forcing) == fc.cached_forcing
            for t in (0.01, 0.02)
                @test Array(generate_forcing!(fd, t)) ≈ generate_forcing!(fc, t)
            end
            back = save_state(device, joinpath(mktempdir(), "to_cpu"))
            loaded, fl = _stochastic_restart_solver(; separable, seed=789)
            load_state!(loaded, back)
            @test fl.cached_forcing == Array(fd.cached_forcing)
            @test rand(copy(fl.rng), UInt64, 8) == rand(copy(fd.rng), UInt64, 8)
        end
    else
        @test_skip "JLArrays unavailable"
    end
end

if try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
    CUDA.allowscalar(false)
    @testset "CUDA stochastic checkpoint restart" begin
        _test_stochastic_restart(GPU())
    end
else
    @testset "CUDA stochastic checkpoint restart" begin
        @test_skip "CUDA unavailable"
    end
end
