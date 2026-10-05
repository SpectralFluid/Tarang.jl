using Test
using Tarang
using MPI

@testset "Boundary callbacks preserve coordinate sets and declared order" begin
    manager = BoundaryConditionManager()
    for names in (("x", "z"), ("y", "z"), ("z",), ("y",), ("r", "phi"), ("theta", "phi"))
        coords = Dict{String,Any}(name => [Float64(i), Float64(i + 1)] for (i, name) in enumerate(names))
        expected = sum((i .* coords[name] for (i, name) in enumerate(names)))
        spatial = length(names) == 1 ? (x::AbstractArray) -> x :
                  (x::AbstractArray, y::AbstractArray) -> x .+ 2 .* y
        moving = length(names) == 1 ? (t::Real, x::AbstractArray) -> t .+ x :
                 (t::Real, x::AbstractArray, y::AbstractArray) -> t .+ x .+ 2 .* y
        @test Tarang._evaluate_space_function_expression(spatial, coords) == expected
        @test Tarang._evaluate_function_expression(moving, 0.25, coords) == expected .+ 0.25
    end

    coords = Dict{String,Any}("x" => [1.0, 2.0], "z" => [-1.0, 2.0], "s" => [3.0, 4.0])
    # The normal coordinate is replaced by the wall position before callbacks run.
    for value in (SpaceDependentValue("", ["s", "z", "x"], (s,z,x) -> s .+ 10z .+ 100 .* x),
                  TimeSpaceDependentValue("", ["t"], ["s", "z", "x"],
                                          (t,s,z,x) -> t .+ s .+ 10z .+ 100 .* x))
        bc = dirichlet_bc("u", "z", 2.0, value)
        offset = value isa SpaceDependentValue ? 0.0 : 0.25
        @test evaluate_bc_value(manager, bc, 0.25, coords) == [123.0, 224.0] .+ offset
    end
    @test coords["z"] == [-1.0, 2.0]
    subset = SpaceDependentValue("", ["s"], s::AbstractArray -> 3 .* s)
    @test evaluate_bc_value(manager, dirichlet_bc("u", "z", 2.0, subset), 0.0, coords) == [9.0, 12.0]
    inferred = SpaceDependentValue("", String[], (x,z) -> x .+ z)
    @test evaluate_bc_value(manager, dirichlet_bc("u", "z", 2.0, inferred), 0.0, coords) == [3.0, 4.0]

    # Existing x-only callbacks on x/z domains keep their original dispatch fallback.
    @test Tarang._evaluate_space_function_expression(x::AbstractArray -> 2 .* x, coords) == [2.0, 4.0]
    @test Tarang._evaluate_function_expression((t::Real,x::AbstractArray) -> t .+ x, 0.5, coords) == [1.5, 2.5]
    @test Tarang._evaluate_function_expression((t,c::AbstractDict) -> t .+ c["s"], 0.5, coords) == [3.5, 4.5]
    @test_throws ArgumentError Tarang._evaluate_space_function_expression((a,b,c,d) -> 0, coords)
    @test_throws ArgumentError Tarang._evaluate_function_expression((t,a,b,c,d) -> 0, 0.5, coords)

    for timed in (false, true)
        sentinel = ErrorException("positional boundary callback failed")
        callback = timed ? (t::Real,x::AbstractArray,z::AbstractArray) -> throw(sentinel) :
                           (x::AbstractArray,z::AbstractArray) -> throw(sentinel)
        caught = try
            timed ? Tarang._evaluate_function_expression(callback, 0.5, coords) :
                    Tarang._evaluate_space_function_expression(callback, coords)
        catch err
            err
        end
        @test caught === sentinel
    end

    # A cylindrical axial coordinate must not replace the radial argument of
    # existing one-coordinate callbacks, including untyped positional callbacks.
    cylindrical = Dict{String,Any}("r" => [1.0, 2.0], "z" => [0.0, 5.0])
    for callback in (r -> 2 .* r, r::AbstractArray -> 2 .* r)
        @test Tarang._evaluate_space_function_expression(callback, cylindrical) == [2.0, 4.0]
        value = SpaceDependentValue("", String[], callback)
        @test evaluate_bc_value(manager, dirichlet_bc("u", "z", 5.0, value), 0.0, cylindrical) == [2.0, 4.0]
    end
    radial_axial = SpaceDependentValue("", String[], (r,z) -> 2 .* r .+ z)
    @test evaluate_bc_value(manager, dirichlet_bc("u", "z", 5.0, radial_axial),
                            0.0, cylindrical) == [7.0, 9.0]
    moving_cylindrical = (t,r,z) -> t .+ 2 .* r .+ z
    @test evaluate_bc_value(manager, dirichlet_bc("u", "z", 5.0, moving_cylindrical;
                            time_dependent=true), 0.25, cylindrical) == [7.25, 9.25]
    cylindrical["theta"] = [0.1, 0.2]
    angular = SpaceDependentValue("", String[], (r,theta,z) -> r .+ 10 .* theta .+ z)
    @test evaluate_bc_value(manager, dirichlet_bc("u", "z", 5.0, angular),
                            0.0, cylindrical) == [7.0, 9.0]
end

function _boundary_api_problem(kind, bottom, top; names=("x", "z"), nx=6, nz=12)
    xn, zn = names
    coords = CartesianCoordinates(xn, zn)
    dist = Distributor(coords; comm=MPI.COMM_SELF, dtype=Float64, architecture=CPU())
    xb = RealFourier(coords[xn]; size=nx, bounds=(0.0, 2π))
    zb = ChebyshevT(coords[zn]; size=nz, bounds=(-1.0, 2.0))
    u = ScalarField(dist, "u", (xb, zb), Float64)
    a = ScalarField(dist, "a", (xb,), Float64)
    b = ScalarField(dist, "b", (xb,), Float64)
    problem = kind([u, a, b])
    lb = derivative_basis(zb, 2)
    add_parameters!(problem; l1=lift(a, lb, -1), l2=lift(b, lb, -2))
    lhs = kind === InitialValueProblem ? "dt(u)-lap(u)" : "lap(u)"
    add_equation!(problem, "$lhs+l1+l2=0")
    add_bc!(problem, bottom)
    add_bc!(problem, top)
    return problem, u
end

@testset "Public callback boundaries use the wall position and stage time" begin
    xs = (0:5) .* (2π/6)
    for wrapped in (false, true)
        value = wrapped ? TimeSpaceDependentValue("", ["t"], ["x", "z"],
                                                 (t,x,z) -> (1+t) .* (cos.(x) .+ z)) :
                          (t,x,z) -> (1+t) .* (cos.(x) .+ z)
        bottom = dirichlet_bc("u", "z", -1.0, value; time_dependent=true, space_dependent=true)
        top = dirichlet_bc("u", "z", 2.0, value; time_dependent=true, space_dependent=true)
        problem, u = _boundary_api_problem(InitialValueProblem, bottom, top)
        solver = InitialValueSolver(problem, RK222(); dt=0.01)
        for _ in 1:3
            step!(solver)
        end
        values = Array(grid_data!(u))
        @test values[:,1] ≈ (1+solver.sim_time) .* (cos.(xs) .- 1) atol=1e-10
        @test values[:,end] ≈ (1+solver.sim_time) .* (cos.(xs) .+ 2) atol=1e-10
    end

    # Explicit wrapper metadata enables positional callbacks on custom-named domains.
    callback = SpaceDependentValue("", ["wall", "s"], (wall,s) -> wall .+ 0 .* s)
    problem, u = _boundary_api_problem(LinearBoundaryValueProblem,
        dirichlet_bc("u", "wall", -1.0, callback),
        dirichlet_bc("u", "wall", 2.0, callback); names=("s", "wall"))
    solve!(BoundaryValueSolver(problem))
    zs = -1 .+ 1.5 .* (1 .- cos.(π .* (0:11) ./ 11))
    @test Array(grid_data!(u)) ≈ [z for x in xs, z in zs] atol=1e-10
end
