using Test, Tarang, LinearAlgebra

@testset "BatchedDenseLU constructor compatibility" begin
    A = zeros(ComplexF64, 2, 2, 1)
    for factor in (Tarang.BatchedDenseLU(A, nothing, nothing, false),
                   Tarang.BatchedDenseLU{typeof(A)}(A, nothing, nothing, false))
        @test factor.A === A
        @test !factor.factored
        @test factor.backend_workspace === nothing
    end
end

# A device-array stand-in checks the actual CuDenseLU solve! dispatch without
# loading CUDA or replacing global backend hooks. Only these probe types opt in.
mutable struct DenseSolveBufferProbe{T,GPU} <: AbstractVector{T}
    data::Vector{T}
    copies::Base.RefValue{Int}
end
Base.size(x::DenseSolveBufferProbe) = size(x.data)
Base.IndexStyle(::Type{<:DenseSolveBufferProbe}) = IndexLinear()
Base.getindex(x::DenseSolveBufferProbe, i::Int) = x.data[i]
Base.setindex!(x::DenseSolveBufferProbe, value, i::Int) = (x.data[i] = value)
Base.mightalias(x::DenseSolveBufferProbe, y::DenseSolveBufferProbe) =
    Base.mightalias(x.data, y.data)
function Base.copy(x::DenseSolveBufferProbe{T,G}) where {T,G}
    x.copies[] += 1
    return DenseSolveBufferProbe{T,G}(copy(x.data), x.copies)
end
Tarang._is_gpu_array(::DenseSolveBufferProbe{T,G}) where {T,G} = G
function Tarang._cudense_solve!(dest::DenseSolveBufferProbe{ComplexF64,true},
                              A::Matrix{ComplexF64}, pivots::Nothing)
    ldiv!(lu(A), dest.data)
    return dest
end

@testset "CuDenseLU writes into the caller's destination" begin
    A = ComplexF64[4+im 1-2im; -1+im 3-im]
    expected = ComplexF64[1-2im, 0.5+im]
    rhs_values = A * expected
    copies = Ref(0)
    rhs = DenseSolveBufferProbe{ComplexF64,true}(copy(rhs_values), copies)
    dest = DenseSolveBufferProbe{ComplexF64,true}(zeros(ComplexF64, 2), copies)
    factor = Tarang.CuDenseLU{ComplexF64}(A, nothing, nothing, nothing, 2)
    @test Tarang.MatSolvers.solve!(dest, factor, rhs) === dest
    @test dest.data ≈ expected
    @test rhs.data == rhs_values
    @test copies[] == 0
    @test Tarang.MatSolvers.solve!(rhs, factor, rhs) === rhs
    @test rhs.data ≈ expected
    @test copies[] == 0
    @test Tarang.MatSolvers.solve!(dest, factor, rhs_values) === dest
    @test dest.data ≈ expected
    @test copies[] == 0

    # Host/type-converting destinations retain the allocating compatibility path.
    copyto!(rhs, rhs_values)
    host = DenseSolveBufferProbe{ComplexF64,false}(zeros(ComplexF64, 2), copies)
    @test Tarang.MatSolvers.solve!(host, factor, rhs) === host
    @test host.data ≈ expected
    @test rhs.data == rhs_values
    @test copies[] == 1
    smaller = DenseSolveBufferProbe{ComplexF32,true}(zeros(ComplexF32, 2), copies)
    @test Tarang.MatSolvers.solve!(smaller, factor, rhs) === smaller
    @test smaller.data ≈ expected rtol=1e-6
    @test copies[] == 2
    alias = DenseSolveBufferProbe{ComplexF64,true}(rhs.data, copies)
    @test alias !== rhs
    @test Tarang.MatSolvers.solve!(alias, factor, rhs) === alias
    @test alias.data ≈ expected
    @test copies[] == 3
    @test_throws DimensionMismatch Tarang.MatSolvers.solve!(dest, factor, ComplexF64[1])
    @test_throws DimensionMismatch Tarang.MatSolvers.solve!(ComplexF64[0], factor, rhs)
    @test_throws DimensionMismatch Tarang.MatSolvers.solve(factor, ComplexF64[1])
end

const _GPU_SOLVER_REUSE_CUDA = try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

@testset "Native CUDA solver metadata reuse" begin
    if !_GPU_SOLVER_REUSE_CUDA
        @test_skip "Native CUDA device unavailable"
    else
        CUDA.allowscalar(false)
        n, nmodes = 5, 4
        matrices = cat([Matrix{ComplexF64}(I, n, n) * (m+2) .+
                        reshape(ComplexF64.(1:n*n), n, n) ./ 200
                        for m in 1:nmodes]...; dims=3)
        rhs = reshape(ComplexF64.(1:n*nmodes), n, nmodes)
        expected = hcat([matrices[:,:,m] \ rhs[:,m] for m in 1:nmodes]...)
        factor = Tarang.BatchedDenseLU(CUDA.CuArray(matrices))
        Tarang.batched_factor!(factor)
        X, B = CUDA.zeros(ComplexF64, n, nmodes), CUDA.CuArray(rhs)
        @test Tarang.batched_solve!(X, factor, B) === X
        @test Array(X) ≈ expected rtol=2e-12
        @test Array(B) == rhs
        ws = factor.backend_workspace
        a_ptrs, x_ptrs, pivots, info = ws.matrix_ptrs, ws.rhs_ptrs, factor.pivots, factor.info
        for _ in 1:3
            Tarang.batched_solve!(X, factor, B)
            @test ws.matrix_ptrs === a_ptrs
            @test ws.rhs_ptrs === x_ptrs
            @test factor.pivots === pivots
            @test factor.info === info
        end
        @test Array(X) ≈ expected rtol=2e-12
        CUDA.synchronize()
        allocated_before = CUDA.alloc_stats.alloc_bytes
        for _ in 1:5
            Tarang.batched_solve!(X, factor, B)
        end
        CUDA.synchronize()
        @test CUDA.alloc_stats.alloc_bytes == allocated_before

        # Replacing the output refreshes only its pointer table; in-place RHS is valid.
        Y = copy(B)
        Tarang.batched_solve!(Y, factor, Y)
        @test Array(Y) ≈ expected rtol=2e-12
        @test ws.rhs === Y
        @test ws.rhs_ptrs !== x_ptrs
        @test ws.matrix_ptrs === a_ptrs
        @test_throws DimensionMismatch Tarang.batched_solve!(Y, factor, CUDA.zeros(ComplexF64, n+1, nmodes))
        devices = collect(CUDA.devices())
        if length(devices) > 1
            other = first(device for device in devices if device != CUDA.device(factor.A))
            foreign = CUDA.device!(other) do
                CUDA.zeros(ComplexF64, n, nmodes)
            end
            @test_throws "same CUDA context" Tarang.batched_solve!(foreign, factor, B)
        end
        overlapping = CUDA.CuArray(vcat(vec(rhs), 0))
        overlap_rhs = reshape(view(overlapping, 1:length(rhs)), n, nmodes)
        overlap_dest = reshape(view(overlapping, 2:length(rhs)+1), n, nmodes)
        Tarang.batched_solve!(overlap_dest, factor, overlap_rhs)
        @test Array(overlap_dest) ≈ expected rtol=2e-12

        # Fully reassemble before refactoring; LU destroys its matrix on CUDA.
        copyto!(factor.A, 2matrices)
        Tarang.batched_factor!(factor)
        @test factor.pivots === pivots
        @test factor.info === info
        @test factor.backend_workspace === ws
        Tarang.batched_solve!(X, factor, B)
        @test Array(X) ≈ expected ./ 2 rtol=2e-12

        # Raw pointer tables must participate in task-current stream dependencies.
        CUDA.stream!(CUDA.CuStream()) do
            Tarang.batched_solve!(X, factor, B)
        end
        @test Array(X) ≈ expected ./ 2 rtol=2e-12
        @test ws.matrix_ptrs === a_ptrs
        x_ptrs = ws.rhs_ptrs
        CUDA.stream!(CUDA.CuStream()) do
            Tarang.batched_solve!(X, factor, B)
        end
        @test ws.rhs_ptrs === x_ptrs
        @test Array(X) ≈ expected ./ 2 rtol=2e-12

        # New factor storage invalidates metadata; failure invalidates the factor.
        factor.A = CUDA.CuArray(matrices)
        @test_throws "matrix storage changed" Tarang.batched_solve!(X, factor, B)
        Tarang.batched_factor!(factor)
        @test factor.backend_workspace !== ws
        @test factor.backend_workspace.matrix === factor.A
        Tarang.batched_solve!(X, factor, B)
        @test Array(X) ≈ expected rtol=2e-12
        singular = copy(matrices)
        singular[:,:,3] .= 0
        copyto!(factor.A, singular)
        @test_throws "mode(s) [3]" Tarang.batched_factor!(factor)
        @test !factor.factored
        @test_throws "before" Tarang.batched_solve!(X, factor, B)

        @testset "native CuDenseLU aliasing and host destination" begin
            A = matrices[:,:,1]
            dense_factor = Tarang.CuDenseLU(A)
            b = CUDA.CuArray(rhs[:,1])
            x = similar(b)
            @test Tarang.MatSolvers.solve!(x, dense_factor, b) === x
            @test Array(x) ≈ expected[:,1] rtol=2e-12
            @test Array(b) == rhs[:,1]
            @test Tarang.MatSolvers.solve!(b, dense_factor, b) === b
            @test Array(b) ≈ expected[:,1] rtol=2e-12
            copyto!(b, rhs[:,1])
            host = zeros(ComplexF64,n)
            @test Tarang.MatSolvers.solve!(host, dense_factor, b) === host
            @test host ≈ expected[:,1] rtol=2e-12
            real_matrix = Float64[4 1; 1 3]
            real_rhs = CUDA.CuArray(Float64[1, 2])
            real_factor = Tarang.CuDenseLU(real_matrix)
            converted = Tarang.MatSolvers.solve(real_factor, real_rhs)
            @test eltype(converted) === ComplexF64
            @test Array(converted) ≈ real_matrix \ [1.0, 2.0] rtol=2e-12
            @test Array(real_rhs) == [1.0, 2.0]
            overlapping = CUDA.CuArray(vcat(rhs[:,1], 0))
            overlap_rhs = view(overlapping, 1:n)
            overlap_dest = view(overlapping, 2:n+1)
            Tarang.MatSolvers.solve!(overlap_dest, dense_factor, overlap_rhs)
            @test Array(overlap_dest) ≈ expected[:,1] rtol=2e-12
            if length(devices) > 1
                other = first(device for device in devices if device != CUDA.device(dense_factor.A_gpu))
                CUDA.device!(other) do
                    foreign = CUDA.zeros(ComplexF64, n)
                    @test !Tarang._cudense_inplace_compatible(foreign, dense_factor.A_gpu)
                    @test Tarang.MatSolvers.solve!(foreign, dense_factor, b) === foreign
                    @test Array(foreign) ≈ expected[:,1] rtol=2e-12
                    @test CUDA.device() == other
                    # Matching storage still solves on its owning context when
                    # the caller currently has another device selected.
                    Tarang.MatSolvers.solve!(x, dense_factor, b)
                    @test Array(x) ≈ expected[:,1] rtol=2e-12
                    @test CUDA.device() == other
                end
            end
        end
    end
end
