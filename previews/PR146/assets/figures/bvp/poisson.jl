using Tarang

"""
    poisson_example(; Nx=256, Ny=128)

Solve Poisson's equation with mixed wall conditions and filtered random forcing.
Run with `julia --project examples/bvp/poisson.jl`.
"""
function poisson_example(; Nx=256, Ny=128)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    yb = ChebyshevT(coords["y"]; size=Ny, bounds=(0.0, Float64(π)))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u")
    f = ScalarField(domain, "f")
    tau1 = ScalarField(dist, "tau1", (xb,), Float64)
    tau2 = ScalarField(dist, "tau2", (xb,), Float64)
    x, y = local_grids(dist, xb, yb)

    fill_random!(f, "g"; seed=40)
    low_pass_filter!(f; shape=(min(64, Nx ÷ 2), min(32, Ny ÷ 2)))
    # Tarang's low_pass_filter! truncates Fourier axes; truncate Chebyshev
    # degrees explicitly to obtain the (64, 32) forcing bandwidth.
    ensure_layout!(f, :c)
    get_coeff_data(f)[:, min(32, Ny ÷ 2)+1:end] .= 0

    problem = LinearBoundaryValueProblem([u, tau1, tau2])
    lift_basis = derivative_basis(yb, 2)
    add_parameters!(problem; f, l1=lift(tau1, lift_basis, -1),
                                 l2=lift(tau2, lift_basis, -2))
    add_equation!(problem, "Δ(u) + l1 + l2 = f")
    add_bc!(problem, "u(y=0) = 0.025*sin(8*x)")
    add_bc!(problem, "∂y(u)(y=π) = 0")
    solver = BoundaryValueSolver(problem)
    solve!(solver)

    ensure_layout!(u, :g)
    solution = copy(get_grid_data(u))
    ensure_layout!(f, :g)
    forcing = copy(get_grid_data(f))
    lap = evaluate(Δ(u))
    ensure_layout!(lap, :g)
    residual = copy(get_grid_data(lap)) .- forcing
    uy = evaluate(Differentiate(u, coords["y"], 1))
    ensure_layout!(uy, :g)
    bottom_error = maximum(abs.(solution[:, 1] .- 0.025 .* sin.(8 .* x)))
    top_error = maximum(abs, get_grid_data(uy)[:, end])
    residual_error = maximum(abs, residual)
    @info "Poisson verification" bottom_error top_error residual_error
    @assert bottom_error < 1e-9
    @assert top_error < 1e-9
    @assert residual_error < 1e-7
    return (; x=collect(x), y=collect(y), solution, forcing, residual,
            bottom_error, top_error, residual_error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    poisson_example()
end
