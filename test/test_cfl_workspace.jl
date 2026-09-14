using Test, Tarang

@testset "CFL reuses its grid-frequency workspace" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=Float64)
    bases = ntuple(d -> RealFourier(coords[d]; size=32, bounds=(0.0, 2π)), 3)
    u = VectorField(Domain(dist, bases), "u")
    for (i, c) in enumerate(u.components)
        fill!(grid_data!(c), i)
    end
    problem = InitialValueProblem([u])
    add_equation!(problem, "dt(u)=0")
    solver = InitialValueSolver(problem, RK111())
    cfl = CFL(solver; initial_dt=1.0, threshold=0.0, min_change=0.0)
    add_velocity!(cfl, u)
    expected = cfl.safety / sum(i / Tarang.grid_spacing(u.domain)[i] for i in 1:3)
    @test Tarang.compute_timestep(cfl) ≈ expected
    Tarang.compute_timestep(cfl)
    @test (@allocated Tarang.compute_timestep(cfl)) < 65536
    fill!(grid_data!(u.components[2]), 4)
    expected = cfl.safety / sum(v / dx for (v, dx) in zip((1, 4, 3), Tarang.grid_spacing(u.domain)))
    @test Tarang.compute_timestep(cfl) ≈ expected
end
