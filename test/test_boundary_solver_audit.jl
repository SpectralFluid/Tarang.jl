using Test, Tarang

@testset "Robin unit coefficients and signs" begin
    for (lhs, alpha, beta) in (("u(z=0)+d(u,z)(z=0)", 1.0, 1.0),
                               ("-u(z=0)+d(u,z)(z=0)", -1.0, 1.0),
                               ("2*u(z=0)+d(u,z)(z=0)", 2.0, 1.0),
                               ("u(z=0)+2*d(u,z)(z=0)", 1.0, 2.0),
                               ("u(z=0)-d(u,z)(z=0)", 1.0, -1.0),
                               ("u(z=0)-2*d(u,z)(z=0)", 1.0, -2.0),
                               ("u(z=0)+(-2)*d(u,z)(z=0)", 1.0, -2.0),
                               ("(2)*u(z=0)+(3)*d(u,z,1)(z=0)", 2.0, 3.0),
                               ("d(u,z)(z=0)+u(z=0)", 1.0, 1.0),
                               ("d(u,z)(z=0)-u(z=0)", -1.0, 1.0),
                               ("-d(u,z)(z=0)+u(z=0)", 1.0, -1.0),
                               ("-d(u,z)(z=0)-u(z=0)", -1.0, -1.0),
                               ("3*d(u,z)(z=0)-2*u(z=0)", -2.0, 3.0),
                               ("(-3)*d(u,z,1)(z=0)+(2)*u(z=0)", 2.0, -3.0))
        @test parse_robin_bc_string(lhs * "=1+t") == ("u", "z", 0.0, alpha, beta, "1+t")
    end
    @test parse_robin_bc_string("alpha*u(z=Lz)+beta*d(u,z)(z=Lz)=forcing") ==
          ("u", "z", "Lz", "alpha", "beta", "forcing")
    @test parse_robin_bc_string("beta*d(u,z)(z=Lz)+alpha*u(z=Lz)=forcing") ==
          ("u", "z", "Lz", "alpha", "beta", "forcing")
end

@testset "Eigenvalue boundary spectra and residuals" begin
    for kind in (:neumann, :mixed, :robin)
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zb = ChebyshevT(coords["z"]; size=32, bounds=(0.0,1.0))
        u = ScalarField(Domain(dist,(zb,)),"u")
        tau1 = ScalarField(dist,"tau1",(),Float64)
        tau2 = ScalarField(dist,"tau2",(),Float64)
        problem = EigenvalueProblem([u,tau1,tau2]; eigenvalue=:sigma)
        lb = derivative_basis(zb,2)
        add_parameters!(problem; l1=lift(tau1,lb,-1),l2=lift(tau2,lb,-2))
        add_equation!(problem,"dt(u)-lap(u)+l1+l2=0")
        if kind == :neumann
            add_bc!(problem,"d(u,z)(z=0)=0")
            add_bc!(problem,"d(u,z)(z=1)=0")
            expected = [-(n*pi)^2 for n in 0:4]
        elseif kind == :mixed
            add_bc!(problem,"u(z=0)=0")
            add_bc!(problem,"d(u,z)(z=1)=0")
            expected = [-((n+0.5)*pi)^2 for n in 0:4]
        else
            # Coordinate derivatives at both walls: exp(z) has growth rate +1;
            # cos(n*pi*z)+sin(n*pi*z)/(n*pi) has growth rate -(n*pi)^2.
            add_bc!(problem,"-u(z=0)+d(u,z)(z=0)=0")
            add_bc!(problem,"-u(z=1)+d(u,z)(z=1)=0")
            expected = [1.0; [-(n*pi)^2 for n in 1:4]]
        end
        solver = EigenvalueSolver(problem;nev=5,which=:SM)
        eigenvalues,vectors = solve!(solver)
        @test sort(real.(eigenvalues);by=abs) ≈ expected atol=2e-8 rtol=2e-8
        @test maximum(abs,imag.(eigenvalues)) < 2e-8
        sp = only(solver.subproblems)
        @test maximum(abs,(sp.L_min*vectors)[sp.bc_rows,:]) < 2e-8
    end
end

# u=(1+t)*phi with lap(phi)=0 is exact for every consistent IMEX scheme,
# including startup and changing timesteps. Check the full field, not only walls.
function boundary_solver_audit_case(ts, kind; fourier=false, batched=false, quadratic_time=false, derivative_first=false)
    coords = fourier ? CartesianCoordinates("x", "z") : CartesianCoordinates("z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=18, bounds=(0.0, 1.0))
    xb = fourier ? RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi)) : nothing
    bases = fourier ? (xb, zb) : (zb,)
    tau_bases = fourier ? (xb,) : ()
    domain = Domain(dist, bases)
    u = ScalarField(domain, "u")
    source = ScalarField(domain, "source")
    tau1 = ScalarField(dist, "tau1", tau_bases, Float64)
    tau2 = ScalarField(dist, "tau2", tau_bases, Float64)
    phi = fourier ? ((x,z) -> cos(2x)*exp(2z)) : ((z,) -> 1+z)
    set!(u, phi); set!(source, phi)
    exact_profile = copy(Array(grid_data!(u)))
    lb = derivative_basis(zb, 2)
    problem = InitialValueProblem([u,tau1,tau2])
    add_parameters!(problem; source, l1=lift(tau1,lb,-1), l2=lift(tau2,lb,-2))
    add_equation!(problem, "dt(u) - 0.2*lap(u) + l1 + l2 = " * (quadratic_time ? "0" : "source"))
    timefactor = quadratic_time ? "(1+t*t)" : "(1+t)"
    if fourier
        spatial = "cos(2*x)"
        if kind == :dirichlet
            bottom = "u(z=0) = $timefactor*$spatial"
            top = "u(z=1) = $(exp(2))*$timefactor*$spatial"
        elseif kind == :neumann
            bottom = "d(u,z)(z=0) = 2*$timefactor*$spatial"
            top = "u(z=1) = $(exp(2))*$timefactor*$spatial"
        else
            bottom_lhs = derivative_first ? "d(u,z)(z=0) + u(z=0)" : "u(z=0) + d(u,z)(z=0)"
            top_lhs = derivative_first ? "2*d(u,z)(z=1) + u(z=1)" : "u(z=1) + 2*d(u,z)(z=1)"
            bottom = "$bottom_lhs = 3*$timefactor*$spatial"
            top = "$top_lhs = $(5exp(2))*$timefactor*$spatial"
        end
    else
        if kind == :dirichlet
            bottom = "u(z=0) = $timefactor"
            top = "u(z=1) = 2*$timefactor"
        elseif kind == :neumann
            bottom = "d(u,z)(z=0) = $timefactor"
            top = "u(z=1) = 2*$timefactor"
        else
            bottom_lhs = derivative_first ? "d(u,z)(z=0) + 2*u(z=0)" : "2*u(z=0) + d(u,z)(z=0)"
            top_lhs = derivative_first ? "2*d(u,z)(z=1) + u(z=1)" : "u(z=1) + 2*d(u,z)(z=1)"
            bottom = "$bottom_lhs = 3*$timefactor"
            top = "$top_lhs = 4*$timefactor"
        end
    end
    add_bc!(problem, bottom); add_bc!(problem, top)
    if quadratic_time
        forcing = fourier ? DeterministicForcing((x,z,t,p)->2t .* cos.(2x).*exp.(2z),(8,18)) :
                            DeterministicForcing((z,t,p)->2t .* (1 .+ z),(18,))
        add_stochastic_forcing!(problem,:u,forcing)
    end
    solver = InitialValueSolver(problem,ts;dt=0.01,batched_modes=batched)
    return solver,u,exact_profile
end

@testset "Boundary solver manufactured solutions" begin
    schemes = (RK111(), RK222(), RK443(), RKSMR(), Tarang.RKGFY(),
               CNAB1(), CNAB2(), SBDF1(), SBDF2(), SBDF3(), SBDF4(), Tarang.MCNAB2(), Tarang.CNLF2())
    @testset "moving $kind / $(nameof(typeof(ts)))" for kind in (:dirichlet,:neumann,:robin), ts in schemes
        solver,u,phi = boundary_solver_audit_case(ts,kind)
        for dt in (0.01,0.025,0.015,0.02,0.03)
            step!(solver,dt)
            @test maximum(abs,Array(grid_data!(u)) .- (1+solver.sim_time).*phi) < 2e-9
        end
    end
    @testset "Fourier moving $kind / $(nameof(typeof(ts))) / batched=$batched" for kind in (:dirichlet,:neumann,:robin), ts in (RK222(),RK443(),SBDF2()), batched in (false,true)
        solver,u,phi = boundary_solver_audit_case(ts,kind;fourier=true,batched)
        for dt in (0.01,0.025,0.015)
            step!(solver,dt)
            @test maximum(abs,Array(grid_data!(u)) .- (1+solver.sim_time).*phi) < 2e-8
        end
    end
    @testset "Robin term order preserves moving boundary values" begin
        for ts in (RK222(),SBDF2()), fourier in (false,true), batched in (false,true)
            reference,reference_u,_ = boundary_solver_audit_case(ts,:robin;fourier,batched)
            reversed,u,phi = boundary_solver_audit_case(ts,:robin;fourier,batched,derivative_first=true)
            for dt in (0.01,0.025,0.015)
                step!(reference,dt)
                step!(reversed,dt)
                @test maximum(abs,Array(grid_data!(u)) .- (1+reversed.sim_time).*phi) < 2e-8
                @test Array(grid_data!(u)) ≈ Array(grid_data!(reference_u)) atol=2e-9
            end
        end
    end
    @testset "quadratic boundary data converges / $(nameof(typeof(ts)))" for ts in (RK222(),RK443(),RKSMR(),Tarang.RKGFY())
        for kind in (:dirichlet,:neumann,:robin), batched in (false,true)
            errors = Float64[]
            for dt in (0.02,0.01)
                solver,u,phi = boundary_solver_audit_case(ts,kind;fourier=true,batched,quadratic_time=true)
                for _ in 1:round(Int,0.1/dt)
                    step!(solver,dt)
                end
                factor = 1+solver.sim_time^2
                push!(errors,maximum(abs,Array(grid_data!(u)) .- factor.*phi))
                z = only(c for c in u.dist.coords if c.name == "z")
                x = vec(Tarang.create_meshgrid(u.domain)["x"][:,1])
                mode = cos.(2x)
                values = [vec(grid_data!(evaluate(interpolate(u,z,p)))) for p in (0.0,1.0)]
                derivatives = [vec(grid_data!(evaluate(interpolate(Differentiate(u,z,1),z,p)))) for p in (0.0,1.0)]
                if kind == :dirichlet
                    @test maximum(abs,values[1] .- factor.*mode) < 2e-9
                    @test maximum(abs,values[2] .- exp(2)*factor.*mode) < 2e-9
                elseif kind == :neumann
                    @test maximum(abs,derivatives[1] .- 2factor.*mode) < 2e-9
                    @test maximum(abs,values[2] .- exp(2)*factor.*mode) < 2e-9
                else
                    @test maximum(abs,values[1].+derivatives[1] .- 3factor.*mode) < 2e-9
                    @test maximum(abs,values[2].+2derivatives[2] .- 5exp(2)*factor.*mode) < 2e-9
                end
            end
            # IMEX stages need not integrate the full quadratic profile exactly:
            # intermediate boundary solves introduce ordinary truncation error.
            # Require convergence in the full field as well as exact wall traces.
            @test errors[2] < 5e-5
            @test errors[2] < max(0.35errors[1],2e-10)
        end
    end
    @testset "ETD refuses boundary DAE rather than dropping constraints" for ts in (ETD_RK222(),ETD_CNAB2(),ETD_SBDF2())
        solver,_,_ = boundary_solver_audit_case(ts,:dirichlet)
        @test_throws "singular mass matrix" step!(solver)
        @test solver.iteration == 0
    end
end
