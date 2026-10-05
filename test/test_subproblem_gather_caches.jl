using Test, Tarang, MPI, LinearAlgebra, SparseArrays

function _gather_cache_case()
    coords = CartesianCoordinates("z", "x", "y")
    dist = Distributor(coords; dtype=Float64, comm=MPI.COMM_SELF)
    z = ChebyshevT(coords["z"]; size=8, bounds=(0.0,1.0))
    x = RealFourier(coords["x"]; size=8, bounds=(0.0,2pi))
    y = RealFourier(coords["y"]; size=6, bounds=(0.0,2pi))
    domain = Domain(dist,(z,x,y))
    u, v = ScalarField(domain,"u"), ScalarField(domain,"v")
    taus = [ScalarField(dist,"tau$i",(),Float64) for i in 1:4]
    wall = ScalarField(dist,"wall",(),Float64)
    Tarang.set_grid_data!(wall,[0.25])
    problem = InitialValueProblem([u,v,taus...])
    lb = derivative_basis(z,2)
    add_parameters!(problem; wall, l1=lift(taus[1],lb,-1), l2=lift(taus[2],lb,-2),
                             l3=lift(taus[3],lb,-1), l4=lift(taus[4],lb,-2))
    # Deliberately mix variable order, PDE order and boundary row positions.
    for equation in ("dt(v)-lap(v)+l3+l4=v", "u(z=0)=wall",
                     "dt(u)-lap(u)+l1+l2=u", "u(z=1)=0", "v(z=0)=0", "v(z=1)=0")
        add_equation!(problem,equation)
    end
    solver = InitialValueSolver(problem,RK222();dt=0.001,threaded_modes=false)
    for (field,multiplier) in ((u,1),(v,3))
        data = coeff_data!(field)
        data .= reshape(multiplier .* collect(1:length(data)),size(data))
    end
    return solver,u,v,wall,z
end

@testset "Gather wrappers preserve compression and buffer ownership" begin
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; comm=MPI.COMM_SELF)
    basis = ChebyshevT(coords["z"]; size=4)
    u = ScalarField(dist,"u",(basis,),Float64)
    f = ScalarField(dist,"f",(basis,),Float64)
    x, y = ComplexF64[1,2,3,4], ComplexF64[5,6,7,8]
    coeff_data!(u) .= x
    coeff_data!(f) .= y
    problem = InitialValueProblem([u])
    # Cover no preconditioner, cached row selection, and general sparse mul!.
    for (P,Q) in ((nothing,nothing),
                  (sparse([0 0 1 0; 1 0 0 0]), sparse([0 1 0 0])),
                  (sparse([2 0 1 0; 0 1 0 -1]), sparse([1 2 0 3])))
        sp = Tarang.Subproblem((;problem),(),(nothing,))
        sp.pre_right_pinv, sp.pre_left = P, Q
        expected_x = P === nothing ? x : P*x
        expected_y = P === nothing ? y : P*y
        input = Tarang.gather_inputs(sp,[u])
        @test input == expected_x
        output = Tarang.gather_outputs(sp,[f])
        @test output === input # Existing nonbang result lifetime is borrowed.
        @test output == expected_y
        @test sp.runtime.gather_inputs_raw == x
        @test sp.runtime.gather_outputs_raw == y
        @test !Base.mightalias(sp.runtime.gather_inputs_raw,sp.runtime.gather_outputs_raw)
        dest = similar(expected_x)
        @test Tarang.gather_inputs!(dest,sp,[u]) === dest
        @test dest == expected_x
        @test output == expected_y # Bang calls must not overwrite that result.
        @test Tarang.gather_outputs!(dest,sp,[f]) === dest
        @test dest == expected_y
        equation = Tarang.compress_equation_space(sp,x)
        @test equation == (Q === nothing ? x : Q*x)
        @test !Base.mightalias(equation,output)
        @test output == expected_y
    end
    close(dist)
end

function _gather_cache_probe(dest,sp,solver,F)
    for _ in 1:100
        Tarang.gather_eqn_F!(dest,sp,solver,F,solver.state)
        Tarang.gather_alg_F!(dest,sp)
    end
    return nothing
end

@testset "Compiled gather geometry preserves live values and equation order" begin
    solver,u,v,wall,basis = _gather_cache_case()
    problem = solver.problem
    F = Union{Nothing,ScalarField}[u,v,nothing,nothing,nothing,nothing]
    sps = Tarang.compiled_subproblems(problem)
    for sp in sps
        sizes = Tarang._subproblem_eqn_sizes(sp)
        @test sizes == [8,1,8,1,1,1]
        @test Tarang._subproblem_cheb_basis_from_sp(sp) === basis
        dest = zeros(ComplexF64,size(sp.M_min,1))
        raw = zeros(ComplexF64,sum(sizes))
        kx,ky = Tarang._subproblem_fourier_group_indices(sp)
        raw[1:8] .= Tarang.get_coeff_data(v)[:,kx,ky]
        raw[10:17] .= Tarang.get_coeff_data(u)[:,kx,ky]
        expected = sp.pre_left === nothing ? raw : sp.pre_left*raw
        @test Tarang.gather_eqn_F!(dest,sp,solver,F,solver.state) ≈ expected
        @test !Tarang.alg_F_is_static(sp) # ScalarField parameters remain live.
        for value in (0.25,0.75)
            Tarang.set_grid_data!(wall,[value])
            fill!(raw,0)
            raw[9] = kx==1 && ky==1 ? 8*6*value : 0
            expected = sp.pre_left === nothing ? raw : sp.pre_left*raw
            @test Tarang.gather_alg_F!(dest,sp) ≈ expected
        end
    end

    # A refresh may replace the whole IR entry, not just mutate its expression.
    # Row geometry can stay cached, but that must not retain the old forcing.
    profile = reshape(cos.(2pi .* (0:7) ./ 8),8,1)
    old = problem.equation_data[2]
    replacement = Tarang.EquationIR(Dict(key=>value for (key,value) in old))
    replacement.forcing_expr = Tarang.ArrayOperator(profile)
    problem.equation_data[2] = replacement
    for scale in (1.0,2.0)
        profile .= scale .* reshape(cos.(2pi .* (0:7) ./ 8),8,1)
        Tarang.invalidate_bc_array_cache!(problem)
        for sp in sps
            dest = zeros(ComplexF64,size(sp.M_min,1))
            raw = zeros(ComplexF64,Tarang._subproblem_raw_eqn_size(sp))
            kx,ky = Tarang._subproblem_fourier_group_indices(sp)
            raw[9] = kx==2 && ky==1 ? scale*8*6/2 : 0
            expected = sp.pre_left === nothing ? raw : sp.pre_left*raw
            @test Tarang.gather_alg_F!(dest,sp) ≈ expected atol=1e-12
        end
    end

    sp = first(sps)
    dest = zeros(ComplexF64,size(sp.M_min,1))
    # Missing RHS fields still advance over their target's block and zero it.
    F[2] = nothing
    raw = zeros(ComplexF64,Tarang._subproblem_raw_eqn_size(sp))
    kx,ky = Tarang._subproblem_fourier_group_indices(sp)
    raw[10:17] .= Tarang.get_coeff_data(u)[:,kx,ky]
    expected = sp.pre_left === nothing ? raw : sp.pre_left*raw
    @test Tarang.gather_eqn_F!(dest,sp,solver,F,solver.state) ≈ expected
    F[2] = v
    _gather_cache_probe(dest,sp,solver,F)
    @test (@allocated _gather_cache_probe(dest,sp,solver,F)) <= 16384
end

@testset "Typed sparse LHS updates preserve matrix identity and values" begin
    mass = sparse([2.0 1.0; 1.0 3.0])
    linear = sparse([4.0 2.0; 2.0 5.0])
    lhs = ComplexF64.(mass)
    for coefficient in (0.25+0im,0.5+0im,0.25+0im)
        @test Tarang._update_subproblem_lhs_values!(lhs,mass,linear,coefficient) === lhs
        @test lhs ≈ mass + coefficient*linear
    end
    @test (@allocated Tarang._update_subproblem_lhs_values!(lhs,mass,linear,0.25+0im)) <= 128
end
