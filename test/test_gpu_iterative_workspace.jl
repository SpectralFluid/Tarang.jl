using Test
using Tarang
using LinearAlgebra
using SparseArrays

function _array_iterative_solver(kind, A; tol=1e-10, maxiter=100, restart=8,
                                 pc=Tarang.NoPreconditioner())
    T = eltype(A)
    n = size(A, 1)
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
        @test norm(A*owned-2rhs) < 2e-10 * norm(2rhs)  # tol is relative to ‖b‖
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

@testset "GMRES tests the true residual after every cycle and raises on failure" begin
    A = Diagonal(fill(2.0+0im, 8))
    rhs = ComplexF64.(1:8)
    dest = similar(rhs)
    # Converges inside a truncated final cycle (maxiter < restart). The old loop
    # tested convergence only on entering a cycle, so it warned "did not
    # converge" here although the iterate it returned was exact.
    short = _array_iterative_solver(:gmres, A; maxiter=3, restart=8)
    @test_logs Tarang.MatSolvers.solve!(dest, short, rhs)
    @test norm(A*dest-rhs) <= 1e-10 * norm(rhs)

    # The cyclic shift stagnates completely until the n-th iteration: GMRES(n)
    # converges exactly on its LAST allowed iteration and must report success...
    n = 8
    P = sparse([mod1(i+1, n) for i in 1:n], 1:n, ones(ComplexF64, n), n, n)
    e1 = [1.0+0im; zeros(ComplexF64, n-1)]
    x = zeros(ComplexF64, n)
    @test_logs Tarang.MatSolvers.solve!(x, _array_iterative_solver(:gmres, P; maxiter=n, restart=n), e1)
    @test norm(P*x - e1) <= 1e-10
    # ...while GMRES(4) never leaves the plateau: an error, not a warning plus a
    # last iterate that is not a solution.
    stalled = _array_iterative_solver(:gmres, P; maxiter=40, restart=4)
    @test_throws "did not converge" Tarang.MatSolvers.solve!(x, stalled, e1)
    @test_throws ArgumentError _array_iterative_solver(:gmres, P; restart=0)

    exhausted = _array_iterative_solver(:cg, A; maxiter=0)
    @test_throws ErrorException Tarang.MatSolvers.solve!(dest, exhausted, rhs)
end

# A tau/saddle-point system: a non-symmetric bulk block, two tau columns lifted
# into the highest modes, and two boundary rows evaluating the Chebyshev series at
# z = ±1 (T_k(1) = 1, T_k(-1) = (-1)^k). The boundary rows have a ZERO diagonal.
function _tau_saddle_system(n=24)
    K = spdiagm(-1 => fill(0.6+0.1im, n-1), 0 => ComplexF64.(3 .+ (1:n) ./ n),
                 1 => fill(-0.4+0im, n-1), 2 => fill(0.2+0im, n-2))
    B = sparse([n-1, n], [1, 2], ComplexF64[1, 1], n, 2)
    C = sparse(ComplexF64[ones(n)'; [(-1.0)^k for k in 0:n-1]'])
    return [K B; C spzeros(ComplexF64, 2, 2)]
end

@testset "Jacobi GMRES solves zero-diagonal tau systems to the true tolerance" begin
    A = _tau_saddle_system()
    N = size(A, 1)
    @test findall(iszero, diag(A)) == [N-1, N]
    # Zero diagonals pass through unscaled. Mapping them to 0 made M⁻¹ singular:
    # the boundary rows vanished from the preconditioned system, and GMRES
    # returned tau = 0 with a ~1e-2 true relative residual as "converged".
    jacobi = Tarang.DiagonalPreconditioner(Tarang._jacobi_diag_inv(A, ComplexF64))
    @test jacobi.diag_inv[N-1:N] == ComplexF64[1, 1]
    @test jacobi.diag_inv[1:N-2] ≈ inv.(diag(A)[1:N-2])
    expected = ComplexF64[sin(k) + im*cos(2k) for k in 1:N]
    rhs = A * expected
    for restart in (5, 30)
        x = zeros(ComplexF64, N)
        solver = _array_iterative_solver(:gmres, A; pc=jacobi, maxiter=200, restart)
        Tarang.MatSolvers.solve!(x, solver, rhs)
        @test norm(A*x - rhs) <= 1e-10 * norm(rhs)
        @test x ≈ expected rtol=1e-8
        @test x[N-1:N] ≈ expected[N-1:N] rtol=1e-8  # the tau values themselves
    end
end

@testset "Iterative tolerances are relative to ‖b‖; b = 0 returns x = 0" begin
    n = 16
    A = spdiagm(-1 => fill(-0.1+0im,n-1), 0 => fill(2.0+0im,n), 1 => fill(-0.1+0im,n-1))
    shape = ComplexF64.(sin.(1:n))
    for kind in (:cg, :gmres), scale in (1.0, 1e-9, 1e-14, 1e-200)
        # The old absolute tests returned x = 0 once ‖b‖ < tol (GMRES) or stopped
        # with an O(1) relative error (CG); at ‖b‖ ~ 1e-200 CG's p'Ap underflowed.
        rhs = scale .* shape
        dest = zeros(ComplexF64, n)
        Tarang.MatSolvers.solve!(dest, _array_iterative_solver(kind, A), rhs)
        @test norm(A*dest - rhs) <= 1e-10 * norm(rhs)
    end
    for kind in (:cg, :gmres)
        # maxiter = 0: x = 0 for b = 0 must not take a single iteration (CG used to
        # divide 0/0 into a "breakdown" and then raise "did not converge").
        solver = _array_iterative_solver(kind, A; maxiter=0)
        dest = fill(1.0+0im, n)
        @test Tarang.MatSolvers.solve!(dest, solver, zeros(ComplexF64, n)) === dest
        @test iszero(dest)
        # ‖b‖ = Inf would otherwise pass `‖r‖ ≤ tol·‖b‖` with x = 0.
        for bad in (NaN, Inf)
            @test_throws ArgumentError Tarang.MatSolvers.solve!(
                dest, solver, [shape[1:2]; bad; shape[4:end]])
        end
    end
end

@testset "CG breakdown test is relative to the operator scale" begin
    n = 16
    A = spdiagm(-1 => fill(-0.1+0im,n-1), 0 => fill(2.0+0im,n), 1 => fill(-0.1+0im,n-1))
    rhs = ComplexF64.(sin.(1:n))
    for scale in (1e-20, 1e20)
        # Unpreconditioned: p'Ap/‖p‖² ~ 1e-20 is small in absolute terms but is
        # not a breakdown — |p'Ap| is compared with ‖p‖·‖Ap‖.
        dest = zeros(ComplexF64, n)
        Tarang.MatSolvers.solve!(dest, _array_iterative_solver(:cg, scale .* A), rhs)
        @test norm(scale .* (A*dest) - rhs) <= 1e-10 * norm(rhs)
    end
    # A genuinely indefinite operator: p'Ap = 0 exactly at the first step.
    indefinite = sparse(ComplexF64[1 0; 0 -1])
    @test_throws "broke down" Tarang.MatSolvers.solve!(zeros(ComplexF64, 2),
        _array_iterative_solver(:cg, indefinite), ComplexF64[1, 1])
end
