using Test, Tarang, MPI

function _bc_expression_cache_probe(vars)
    value = 0.0
    for _ in 1:100
        value += Tarang._safe_eval_math_expr("(1+t)*cos(x)", vars)
    end
    return value
end

function _bc_geometry_cache_probe(sp)
    for _ in 1:100
        Tarang._bc_fourier_axis_sizes(sp)
        Tarang._subproblem_fourier_group_indices(sp)
    end
    return nothing
end

@testset "Boundary expression plans retain syntax, never values" begin
    expression = "scale*(1+t)*cos(x) + offset"
    coordinates = Dict("x" => [0.0, pi/2, pi])
    parameters = Dict{String,Any}("scale" => 2.0, "offset" => 0.5,
                                  "t" => -100.0, "x" => 100.0)
    plan = Tarang._cached_bc_expression(expression)
    @test Tarang.evaluate_expression(expression, 0.0, coordinates; parameters) ≈
          2 .* cos.(coordinates["x"]) .+ 0.5
    parameters["scale"] = 3.0
    parameters["offset"] = [1.0, 2.0, 3.0]
    coordinates["x"] .= [pi, 0.0, pi/2]
    @test Tarang.evaluate_expression(expression, 2.0, coordinates; parameters) ≈
          9 .* cos.(coordinates["x"]) .+ parameters["offset"]
    @test Tarang._cached_bc_expression(expression) === plan

    # The same cached syntax must work with unrelated scalar or array bindings.
    @test Tarang._safe_eval_math_expr(expression,
        Dict("scale"=>4.0,"t"=>0.5,"x"=>0.0,"offset"=>2.0)) == 8.0
    parameters["offset"] .= -1
    @test Tarang.evaluate_expression(expression, 0.0, coordinates; parameters) ≈
          3 .* cos.(coordinates["x"]) .- 1
    scalar_vars = Dict("t"=>0.5,"x"=>0.0)
    @test _bc_expression_cache_probe(scalar_vars) == 150.0
    # Allow scalar dispatch overhead, but prohibit reparsing or constructing a
    # fresh whitelist/argument-vector tree on every warmed evaluation.
    @test (@allocated _bc_expression_cache_probe(scalar_vars)) <= 8192
    @test Tarang._safe_eval_math_expr("begin x+1; 2*x end", Dict("x"=>3.0)) == 6.0
    @test Tarang._safe_eval_math_expr("-x + x%2 + mod(-x,2)", Dict("x"=>3.0)) == -1.0

    # Cache entries are validated before storage, including nested calls and
    # syntax which could otherwise execute arbitrary Julia code.
    for unsafe in ("run(`false`)", "sin(eval(x))", "x=2", "x[1]",
                   "Base.sin(x)", "begin x+1; x=2 end", "@time sin(x)")
        @test_throws ArgumentError Tarang._safe_eval_math_expr(unsafe, Dict("x"=>1.0))
        @test !haskey(Tarang._BC_EXPRESSION_CACHE, unsafe)
    end
    @test_throws ArgumentError Tarang._safe_eval_math_expr("missing+1", Dict("x"=>1.0))
    @test Tarang._safe_eval_math_expr("missing+1", Dict("missing"=>2.0)) == 3.0

    # Concurrent users can share immutable plans while supplying distinct data.
    results = Vector{Float64}(undef, 32)
    @sync for i in eachindex(results)
        Threads.@spawn results[i] = Tarang._safe_eval_math_expr("(1+t)*cos(x)",
                                                               Dict("t"=>Float64(i),"x"=>0.0))
    end
    @test results == collect(2.0:33.0)
    for i in 1:Tarang._BC_EXPRESSION_CACHE_CAPACITY+1
        @test Tarang._safe_eval_math_expr("t+$i", Dict("t"=>1.0)) == i+1
    end
    @test length(Tarang._BC_EXPRESSION_CACHE) <= Tarang._BC_EXPRESSION_CACHE_CAPACITY
    @test Tarang.evaluate_expression(expression, 0.0, coordinates; parameters) ≈
          3 .* cos.(coordinates["x"]) .- 1
end

@testset "Boundary geometry cache follows compiled subproblem and mode" begin
    coords = CartesianCoordinates("z", "x", "y")
    dist = Distributor(coords; comm=MPI.COMM_SELF)
    z = ChebyshevT(coords["z"]; size=8, bounds=(0.0,1.0))
    x = RealFourier(coords["x"]; size=10, bounds=(0.0,2pi))
    y = RealFourier(coords["y"]; size=6, bounds=(0.0,2pi))
    u = ScalarField(dist, "u", (z,x,y), Float64)
    v = ScalarField(dist, "v", (z,x,y), Float64)
    problem = InitialValueProblem([u,v])
    sp = Tarang.Subproblem((;problem), (), (nothing,0,0))
    @test Tarang._bc_fourier_axis_sizes(sp) == (10,6)
    @test Tarang._subproblem_fourier_group_indices(sp) == (1,1)
    @test Tarang._bc_constant_projection(2.0,sp) == 120
    sp.group = (nothing,1,0)
    @test Tarang._subproblem_fourier_group_indices(sp) == (2,1)
    @test Tarang._bc_constant_projection(2.0,sp) == 0
    values = reshape(cos.(2pi .* (0:9) ./ 10),10,1)
    @test Tarang._bc_array_projection(values,sp) ≈ 30 atol=1e-13
    geometry = Tarang._bc_fourier_axis_sizes(sp)
    values .*= 2
    Tarang.invalidate_bc_array_cache!(problem)
    @test Tarang._bc_array_projection(values,sp) ≈ 60 atol=2e-13
    @test Tarang._bc_fourier_axis_sizes(sp) === geometry
    sp.group = (nothing,0,0)
    @test Tarang._bc_constant_projection(2.0,sp) == 120

    # New compiled geometry cannot inherit a previous problem's cached sizes.
    other = InitialValueProblem([ScalarField(dist,"w",(z,y),Float64)])
    other_sp = Tarang.Subproblem((;problem=other), (), (nothing,0))
    @test Tarang._bc_fourier_axis_sizes(other_sp) == (6,)
    @test Tarang._bc_constant_projection(2.0,other_sp) == 12
    coupled = InitialValueProblem([ScalarField(dist,"q",(z,),Float64)])
    coupled_sp = Tarang.Subproblem((;problem=coupled), (), (nothing,))
    @test Tarang._bc_fourier_axis_sizes(coupled_sp) == ()
    @test Tarang._subproblem_fourier_group_indices(coupled_sp) == ()
    @test Tarang._bc_constant_projection(2.0,coupled_sp) == 2
    _bc_geometry_cache_probe(sp)
    @test (@allocated _bc_geometry_cache_probe(sp)) <= 1024
end
