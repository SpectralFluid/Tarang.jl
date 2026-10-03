using Test, Tarang, MPI, PencilArrays

# Each wall varies in only one tangential coordinate. Broadcasting it to the
# whole plane before looking up the cache used to retain one FFT per mode.
@testset "3D moving boundary planes share Fourier projections" begin
    for ts in (RK222(), SBDF2())
        coords = CartesianCoordinates("z", "x", "y")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
        xb = RealFourier(coords["x"]; size=12, bounds=(0.0, 2pi))
        yb = RealFourier(coords["y"]; size=10, bounds=(0.0, 2pi))
        domain = Domain(dist, (zb, xb, yb))
        u = ScalarField(domain, "u")
        source = ScalarField(domain, "source")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        # lap(phi)=0, so u=(1+t)*phi solves dt(u)-0.2lap(u)=phi.
        phi(z,x,y) = 1 + z + cos(x)*sinh(1-z)/sinh(1) + sin(y)*sinh(z)/sinh(1)
        grids = create_meshgrid(domain; on_device=false)
        initial_values = phi.(grids["z"], grids["x"], grids["y"])
        for field in (u,source)
            data = grid_data!(field)
            if data isa PencilArrays.PencilArray
                parent(data) .= initial_values[PencilArrays.pencil(data).axes_local...]
            else
                data .= initial_values
            end
        end
        initial = copy(parent(grid_data!(u)))
        problem = InitialValueProblem([u,tau1,tau2])
        lb = derivative_basis(zb, 2)
        add_parameters!(problem; source, l1=lift(tau1,lb,-1), l2=lift(tau2,lb,-2))
        add_equation!(problem, "dt(u)-0.2*lap(u)+l1+l2=source")
        add_bc!(problem, "u(z=0)=(1+t)*(1+cos(x))")
        add_bc!(problem, "u(z=1)=(1+t)*(2+sin(y))")
        solver = InitialValueSolver(problem, ts; dt=0.01, threaded_modes=true,
                                    batched_modes=false)
        for dt in (0.01,0.02,0.015)
            step!(solver,dt)
            @test parent(grid_data!(u)) ≈ (1+solver.sim_time).*initial atol=3e-9 rtol=3e-9
            cache = problem.compiled.caches.bc_rfft
            @test length(cache) == 2
            @test Set(size(arr) for arr in keys(cache)) == Set(((12,1),(1,10)))
            @test all(length(by_shape) == 1 for by_shape in values(cache))
        end
        close(dist)
    end
end
