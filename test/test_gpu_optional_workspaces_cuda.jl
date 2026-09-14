using Test
using Tarang
using LinearAlgebra
using SparseArrays
using FFTW

const _OPTIONAL_REQUIRE_CUDA = lowercase(get(ENV, "TARANG_REQUIRE_CUDA", "false")) in ("1", "true", "yes")
const _OPTIONAL_HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if !_OPTIONAL_HAS_CUDA
    _OPTIONAL_REQUIRE_CUDA && error("Optional solver/FFT validation requires a functional CUDA device")
    @testset "Optional GPU workspace reuse" begin
        @test_skip "CUDA not functional on this host"
    end
else
    CUDA.allowscalar(false)
    ext = Base.get_extension(Tarang, :TarangCUDAExt)
    @testset "Batched CUDA transforms reuse task/stream-owned storage" begin
        for shape in ((7,), (8,5)), real_input in (false,true)
            T = real_input ? Float64 : ComplexF64
            plan = ext.get_batched_fft_plan(GPU(), shape, T, 3; real_input)
            @test ext.get_batched_fft_plan(GPU(), shape, T, 3; real_input) === plan
            inputs = [CUDA.CuArray(reshape(T.(sin.((1:prod(shape)).*(i/3))),shape)) for i in 1:3]
            outputs = [similar(plan.packed_output, size(plan.packed_output)[1:end-1]...) for _ in 1:3]
            restored = [similar(a) for a in inputs]
            CUDA.@sync ext.batched_fft!(outputs,inputs,plan)
            saved = Array.(outputs)
            CUDA.@sync ext.batched_ifft!(restored,outputs,plan)
            @test Array.(restored) ≈ Array.(inputs)
            @test Array.(outputs) == saved
            before = CUDA.alloc_stats.alloc_bytes
            CUDA.@sync begin
                ext.batched_fft!(outputs,inputs,plan)
                ext.batched_ifft!(restored,outputs,plan)
            end
            @test CUDA.alloc_stats.alloc_bytes == before
            @test fetch(@async begin
                @test_throws ArgumentError ext.batched_fft!(outputs,inputs,plan)
                ext.get_batched_fft_plan(GPU(),shape,T,3;real_input) !== plan
            end)
            CUDA.stream!(CUDA.CuStream()) do
                @test_throws ArgumentError ext.batched_fft!(outputs,inputs,plan)
                @test ext.get_batched_fft_plan(GPU(),shape,T,3;real_input) !== plan
            end
        end
    end
    @testset "CUDA complex plans accept real inputs" begin
        plan = ext.get_batched_fft_plan(GPU(), (8,), Float64, 2)
        inputs = [CUDA.CuArray(sin.((1:8) .* (i/3))) for i in 1:2]
        outputs = [CUDA.zeros(ComplexF32,8) for _ in 1:2]
        CUDA.@sync ext.batched_fft!(outputs,inputs,plan)
        @test Array.(outputs) ≈ FFTW.fft.(Array.(inputs)) rtol=1e-6
    end
    @testset "CUDA iterative solve! parity and retained workspace" begin
        n = 64
        A = spdiagm(-1 => fill(-0.1,n-1), 0 => fill(2.0,n), 1 => fill(-0.1,n-1))
        rhs = CUDA.CuArray(ComplexF64.(sin.(1:n)))
        for constructor in (Tarang.CuIterativeCG, Tarang.CuIterativeGMRES)
            solver = constructor(A; preconditioner=:none, tol=1e-10)
            dest = similar(rhs)
            Tarang.MatSolvers.solve!(dest,solver,rhs)
            @test norm(A*Array(dest)-Array(rhs)) < 1e-9
            saved = copy(Array(dest))
            host_dest = zeros(ComplexF64,n)
            Tarang.MatSolvers.solve!(host_dest,solver,rhs)
            @test host_dest ≈ saved
            owned = Tarang.MatSolvers.solve(solver,2 .* rhs)
            @test Array(dest) == saved
            @test Array(owned) ≈ 2saved rtol=1e-9
            workspace = solver isa Tarang.CuIterativeCG ? solver.r : solver.workspace
            before = CUDA.alloc_stats.alloc_bytes
            CUDA.@sync Tarang.MatSolvers.solve!(dest,solver,rhs)
            @info "Warmed GPU iterative solve allocation, including CUDA library workspaces" solver=constructor bytes=CUDA.alloc_stats.alloc_bytes-before
            @test (solver isa Tarang.CuIterativeCG ? solver.r : solver.workspace) === workspace
            @test norm(A*Array(dest)-Array(rhs)) < 1e-9
        end
    end
end
