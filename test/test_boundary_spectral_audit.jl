using Test
using Tarang
using LinearAlgebra

# These oracles are ordinary polynomials in the physical coordinate, independent
# of the package's basis recurrence, normalization, and differentiation matrices.
function boundary_spectral_polynomial(z, order=0)
    coefficients = (1.0, 0.2, 0.3, -0.1, 0.02)
    return sum(coefficients[n + 1] * factorial(n) / factorial(n - order) *
               z^(n - order) for n in order:(length(coefficients) - 1))
end

function boundary_spectral_manufactured(mk, kind; axis_order=("z",), fourth=false)
    coords = CartesianCoordinates(axis_order...)
    dist = Distributor(coords)
    a, b = -1.25, 2.75
    zb = mk(coords["z"], 12, (a, b))
    xb = length(axis_order) > 1 ?
         RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi)) : nothing
    bases = Tuple(c == "z" ? zb : xb for c in axis_order)
    domain = Domain(dist, bases)
    u = ScalarField(domain, "u")
    n_tau = fourth ? 4 : 2
    tau_bases = xb === nothing ? () : (xb,)
    taus = [ScalarField(dist, "tau$i", tau_bases, Float64) for i in 1:n_tau]
    problem = LinearBoundaryValueProblem([u; taus])
    lift_basis = derivative_basis(zb, n_tau)
    lifts = [lift(tau, lift_basis, -i) for (i, tau) in enumerate(taus)]
    mesh = Tarang.create_meshgrid(domain)
    zs = mesh["z"]
    forcing = ScalarField(domain, "forcing")
    forcing["g"] .= boundary_spectral_polynomial.(zs, fourth ? 4 : 2)
    if xb !== nothing
        forcing["g"] .= (boundary_spectral_polynomial.(zs, 2) .-
                         4 .* boundary_spectral_polynomial.(zs)) .* cos.(2 .* mesh["x"])
    end
    add_parameters!(problem; forcing)
    add_parameters!(problem; Dict(Symbol("l$i") => op for (i, op) in enumerate(lifts))...)
    bulk = fourth ? "d(d(d(d(u,z),z),z),z)" : xb === nothing ? "d(d(u,z),z)" : "lap(u)"
    add_equation!(problem, bulk * " + " * join(["l$i" for i in 1:n_tau], " + ") * " = forcing")
    boundary_value(v) = xb === nothing ? v : string(v) * "*cos(2*x)"

    if fourth && kind == :higher_neumann
        add_bc!(problem, dirichlet_bc("u", "z", a, boundary_value(boundary_spectral_polynomial(a))))
        add_bc!(problem, neumann_bc("u", "z", a, boundary_value(boundary_spectral_polynomial(a, 1))))
        for order in (2, 3)
            add_bc!(problem, neumann_bc("u", "z", b,
                boundary_value(boundary_spectral_polynomial(b, order)); derivative_order=order))
        end
    elseif fourth
        for z in (a, b)
            add_bc!(problem, dirichlet_bc("u", "z", z, boundary_value(boundary_spectral_polynomial(z))))
            add_bc!(problem, neumann_bc("u", "z", z, boundary_value(boundary_spectral_polynomial(z, 1))))
        end
    elseif kind == :DD
        for z in (a, b)
            add_bc!(problem, dirichlet_bc("u", "z", z, boundary_value(boundary_spectral_polynomial(z))))
        end
    elseif kind == :DN
        add_bc!(problem, dirichlet_bc("u", "z", a, boundary_value(boundary_spectral_polynomial(a))))
        add_bc!(problem, neumann_bc("u", "z", b, boundary_value(boundary_spectral_polynomial(b, 1))))
    elseif kind == :reverse_RR
        add_bc!(problem, "3*d(u,z)(z=$a) + 2*u(z=$a) = " *
            string(boundary_value(2boundary_spectral_polynomial(a) + 3boundary_spectral_polynomial(a, 1))))
        add_bc!(problem, "0.5*d(u,z)(z=$b) + u(z=$b) = " *
            string(boundary_value(boundary_spectral_polynomial(b) + 0.5boundary_spectral_polynomial(b, 1))))
    else
        add_bc!(problem, robin_bc("u", "z", a, 2.0, 3.0,
                                 boundary_value(2boundary_spectral_polynomial(a) + 3boundary_spectral_polynomial(a, 1))))
        add_bc!(problem, robin_bc("u", "z", b, 1.0, 0.5,
                                 boundary_value(boundary_spectral_polynomial(b) + 0.5boundary_spectral_polynomial(b, 1))))
    end

    solver = BoundaryValueSolver(problem)
    solve!(solver)
    exact = boundary_spectral_polynomial.(zs)
    xb === nothing || (exact = exact .* cos.(2 .* mesh["x"]))
    @test maximum(abs, u["g"] .- exact) < 5e-10
    return solver, u, zb, taus
end

@testset "Boundary spectral basis and normalization audit" begin
    makers = ((z, n, b) -> ChebyshevT(z; size=n, bounds=b),
              (z, n, b) -> Legendre(z; size=n, bounds=b),
              (z, n, b) -> ChebyshevU(z; size=n, bounds=b),
              (z, n, b) -> Jacobi(z; size=n, bounds=b, a=0.2, b=0.7))

    @testset "Manufactured shifted-domain BVPs" begin
        for mk in makers, kind in (:DD, :DN, :RR)
            boundary_spectral_manufactured(mk, kind)
        end
        for mk in makers, kind in (:clamped, :higher_neumann)
            boundary_spectral_manufactured(mk, kind; fourth=true)
        end
        for mk in makers, order in (("x", "z"), ("z", "x"))
            boundary_spectral_manufactured(mk, :RR; axis_order=order)
        end
        # The derivative-first spelling must register its spatial RHS just like
        # the canonical field-first form, rather than silently enforcing zero.
        for order in (("x", "z"), ("z", "x"))
            boundary_spectral_manufactured(first(makers), :reverse_RR; axis_order=order)
        end
    end

    @testset "Point functionals and tau columns" begin
        # Degree nine exercises the high modes without using the implementation's
        # evaluation functions as the oracle. The affine chain rule is explicit.
        coefficients = (0.7, -0.2, 0.12, 0.25, -0.31, 0.11, 0.09, -0.04, 0.08, -0.03)
        a, b = -1.25, 2.75
        half_width = (b - a) / 2
        midpoint = (a + b) / 2
        polynomial(z, order=0) = sum(coefficients[n + 1] * factorial(n) /
            factorial(n - order) * ((z - midpoint) / half_width)^(n - order) /
            half_width^order for n in order:(length(coefficients) - 1))

        for mk in makers
            solver, u, basis, taus = boundary_spectral_manufactured(mk, :DD)
            sp = only(solver.subproblems)
            z = u.dist.coordsys["z"]
            nodes = vec(Tarang.local_grid(basis, u.dist, 1.0))
            u["g"] .= polynomial.(nodes)
            coeffs = copy(u["c"])
            for position in (a, a + 0.17 * (b - a), midpoint, b), order in 0:4
                operand = order == 0 ? u : d(u, z, order)
                op = interpolate(operand, z, position)
                block = expression_matrices(op, sp, [u])[u]
                @test only(block * coeffs) ≈ polynomial(position, order) atol=2e-9 rtol=2e-10
            end

            # Solver lifts intentionally inject into the equation's coefficient
            # rows; docs/tau_method.md documents that changing op.basis does not
            # convert this column. Pin both signed indexing and the assembled
            # placement rather than assuming a particular classical polynomial.
            n = basis.meta.size
            for mode in (0, 3, -2, -1), lift_basis in (basis, derivative_basis(basis, 2))
                op = lift(first(taus), lift_basis, mode)
                block = expression_matrices(op, sp, [first(taus)])[first(taus)]
                expected = zeros(ComplexF64, n, 1)
                expected[mode < 0 ? n + mode + 1 : mode + 1, 1] = 1
                @test Matrix(block) == expected
            end
            # Undo the row/column ordering used by the solver before selecting
            # the equation and variable blocks in their public input order.
            assembled = sp.pre_left_pinv * sp.L_min * sp.pre_right_pinv
            @test Matrix(assembled[1:n, n+1:n+2]) == hcat(
                [i == n ? 1 : 0 for i in 1:n], [i == n - 1 ? 1 : 0 for i in 1:n])
            @test sum(assembled[n+1, 1:n] .* coeffs) ≈ polynomial(a) atol=2e-10
            @test sum(assembled[n+2, 1:n] .* coeffs) ≈ polynomial(b) atol=2e-10
        end
    end
end
