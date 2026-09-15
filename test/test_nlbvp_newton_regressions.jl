using Test
using Tarang

@testset "Newton retains state-dependent coefficients: radial Lane–Emden" begin
    coords = CartesianCoordinates("r")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    rb = ChebyshevT(coords["r"]; size=48, bounds=(0.0, 1.0), dealias=2.0)
    domain = Domain(dist, (rb,))
    f = ScalarField(domain, "f")
    radius = ScalarField(domain, "radius")
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    r, = local_grids(dist, rb)
    ensure_layout!(radius, :g)
    get_grid_data(radius) .= r
    ensure_layout!(f, :g)
    get_grid_data(f) .= 5 .* (1 .- r.^2).^2
    problem = NonlinearBoundaryValueProblem([f, tau1, tau2])
    lb = derivative_basis(rb, 2)
    add_parameters!(problem; radius, fr=Differentiate(f, coords["r"], 1),
                            l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
    add_equation!(problem, "radius*Δ(f) + 2*fr + l1 + l2 = -radius*f*f*f")
    add_bc!(problem, "∂r(f)(r=0) = 0")
    add_bc!(problem, "f(r=1) = 0")
    solve!(BoundaryValueSolver(problem; tolerance=1e-9, max_iterations=15))
    ensure_layout!(f, :g)
    values = get_grid_data(f)
    # n=3 first zero: Boyd (2011), doi:10.4208/NMTMA.2011.42S.2.
    @test values[1] ≈ 6.896848619376960375454528 atol=1e-7
    @test abs(values[end]) < 1e-9
    @test minimum(values) > -1e-9
end

@testset "Newton cross-field and derivative coefficients" begin
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=24, bounds=(0.0, 1.0), dealias=2.0)
    domain = Domain(dist, (zb,))
    u = ScalarField(domain, "u")
    # A valid user name must not collide with a solver-created base-state field.
    v = ScalarField(domain, "__newton_base_1_u")
    taus = [ScalarField(dist, "tau$i", (), Float64) for i in 1:4]
    g1, g2 = ScalarField(domain, "g1"), ScalarField(domain, "g2")
    z, = local_grids(dist, zb)
    exact_u, exact_v = 1 .+ z .* (1 .- z), 2 .+ z .* (1 .- z)
    for (field, data) in ((u, exact_u .+ 0.05 .* sin.(π .* z)),
                          (v, exact_v .- 0.04 .* sin.(π .* z)),
                          (g1, -2 .- exact_u .* (1 .- 2 .* z)),
                          (g2, -2 .- exact_v .* (1 .- 2 .* z)))
        ensure_layout!(field, :g)
        get_grid_data(field) .= data
    end
    problem = NonlinearBoundaryValueProblem([u, v, taus...])
    lb = derivative_basis(zb, 2)
    add_parameters!(problem; g1, g2, uz=Differentiate(u, coords["z"], 1),
                    vz=Differentiate(v, coords["z"], 1),
                    l1=lift(taus[1], lb, -1), l2=lift(taus[2], lb, -2),
                    l3=lift(taus[3], lb, -1), l4=lift(taus[4], lb, -2))
    add_equation!(problem, "Δ(u) + l1 + l2 = u*vz + g1")
    add_equation!(problem, "Δ(__newton_base_1_u) + l3 + l4 = __newton_base_1_u*uz + g2")
    for wall in (0, 1)
        add_bc!(problem, "u(z=$wall) = 1")
        add_bc!(problem, "__newton_base_1_u(z=$wall) = 2")
    end
    solve!(BoundaryValueSolver(problem; tolerance=1e-9, max_iterations=8))
    ensure_layout!(u, :g)
    ensure_layout!(v, :g)
    @test get_grid_data(u) ≈ exact_u atol=1e-9
    @test get_grid_data(v) ≈ exact_v atol=1e-9
end
