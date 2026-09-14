using Test
using Tarang
using LinearAlgebra
using SparseArrays

function _array_iterative_solver(kind, A; tol=1e-10, maxiter=100, restart=8)
    T = eltype(A)
    n = size(A, 1)
    pc = Tarang.NoPreconditioner()
    if kind === :cg
        return Tarang.CuIterativeCG{T,typeof(pc)}(A, pc, tol, maxiter, n,
            zeros(T,n), zeros(T,n), zeros(T,n), zeros(T,n))
    end
    return Tarang.CuIterativeGMRES{T,typeof(pc)}(A, pc, tol, maxiter, restart, n)
end

@testset "Iterative solve! owns reusable workspace and caller output" begin
    for kind in (:cg, :gmres)
        n = 64
        A = spdiagm(-1 => fill(-0.1+0im,n-1), 0 => fill(2.0+0im,n), 1 => fill(-0.1+0im,n-1))
        rhs = ComplexF64.(sin.(1:n))
        saved_rhs = copy(rhs)
        solver = _array_iterative_solver(kind, A)
        dest = similar(rhs)
        @test Tarang.MatSolvers.solve!(dest, solver, rhs) === dest
        @test norm(A*dest-rhs) < 1e-9
        @test rhs == saved_rhs
        saved = copy(dest)
        owned = Tarang.MatSolvers.solve(solver, 2rhs)
        @test dest == saved
        @test norm(A*owned-2rhs) < 1e-9
        Tarang.MatSolvers.solve!(dest, solver, rhs)
        @test owned ≈ 2saved
        # Preserve RHS before clearing x, including in-place solve!(b,s,b).
        copyto!(dest, rhs)
        Tarang.MatSolvers.solve!(dest, solver, dest)
        @test norm(A*dest-rhs) < 1e-9
        cast_dest = zeros(ComplexF32,n)
        Tarang.MatSolvers.solve!(cast_dest, solver, rhs)
        @test cast_dest ≈ dest rtol=1e-6
        @test_throws DimensionMismatch Tarang.MatSolvers.solve!(zeros(ComplexF64,n-1), solver, rhs)
    end
end

@testset "Warmed iterative solves allocate no full vectors" begin
    for kind in (:cg, :gmres)
        n = 32768
        A = Diagonal(fill(2.0+0im,n))
        rhs = ComplexF64.(sin.(1:n))
        solver = _array_iterative_solver(kind, A)
        dest = similar(rhs)
        Tarang.MatSolvers.solve!(dest, solver, rhs)
        bytes = @allocated Tarang.MatSolvers.solve!(dest, solver, rhs)
        @test bytes < 100_000  # one vector alone would be 512 KiB
        @test norm(A*dest-rhs) < 1e-9
    end
end

@testset "Iterative stopping and restart semantics remain unchanged" begin
    A = Diagonal(fill(2.0+0im, 8))
    rhs = ComplexF64.(1:8)
    dest = similar(rhs)
    short = _array_iterative_solver(:gmres, A; maxiter=3, restart=8)
    # The original GMRES checks convergence at restart entry; exhausting a
    # truncated cycle returns its computed solution with the same warning.
    @test_logs (:warn, r"GMRES did not converge") Tarang.MatSolvers.solve!(dest, short, rhs)
    @test norm(A*dest-rhs) < 1e-10
    exhausted = _array_iterative_solver(:cg, A; maxiter=0)
    @test_throws ErrorException Tarang.MatSolvers.solve!(dest, exhausted, rhs)
end
