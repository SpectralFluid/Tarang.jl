using Test, Tarang

# Manufactured incompressible Brinkman flow: u - nu*lap(u) + grad(p) = f.
# A streamfunction supplies an independent divergence-free polynomial solution.
# The reaction term removes the constant tangential-velocity nullspace for
# stress-free walls. Compare the interior and pressure as well as wall traces.
@testset "Manufactured incompressible wall conditions" begin
    for wall in (:no_slip, :stress_free), labels in (("x", "z"), ("z", "x"))
        @testset "$wall / $labels" begin
            coords = CartesianCoordinates(labels...)
            dist = Distributor(coords; dtype=Float64)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
            zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 2.0))
            bases = labels[1] == "x" ? (xb, zb) : (zb, xb)
            domain = Domain(dist, bases)
            u = VectorField(domain, "u")
            p = ScalarField(domain, "p")
            f = VectorField(domain, "f")
            taup = ScalarField(dist, "taup", (), Float64)
            tau1 = VectorField(dist, coords, "tau1", (xb,), Float64)
            tau2 = VectorField(dist, coords, "tau2", (xb,), Float64)
            ix, iz = findfirst(==("x"), labels), findfirst(==("z"), labels)
            ez = unit_vector_fields(coords, dist)[iz]
            lb = derivative_basis(zb, 1)
            tlift(A) = lift(A, lb, -1)
            grad_u = grad(u) + ez * tlift(tau1)
            nu, H = 0.3, 3.0
            g(r) = wall == :no_slip ? r^2 - 2r^3 + r^4 : r - 2r^3 + r^4
            dg(r) = wall == :no_slip ? 2r - 6r^2 + 4r^3 : 1 - 6r^2 + 4r^3
            ddg(r) = wall == :no_slip ? 2 - 12r + 12r^2 : -12r + 12r^2
            dddg(r) = -12 + 24r
            fx(x,z) = ((1+nu)*dg((z+1)/H)/H - nu*dddg((z+1)/H)/H^3 - ((z+1)/H-0.5))*sin(x)
            fz(x,z) = (-(1+nu)*g((z+1)/H) + nu*ddg((z+1)/H)/H^2 + 1/H)*cos(x)
            for (j, fun) in ((ix, fx), (iz, fz))
                set!(f.components[j], (a,b) -> labels[1] == "x" ? fun(a,b) : fun(b,a))
            end
            problem = LinearBoundaryValueProblem([p, u, taup, tau1, tau2])
            add_parameters!(problem; nu, f, grad_u, tlift)
            add_equation!(problem, "trace(grad_u) + taup = 0")
            add_equation!(problem, "u - nu*div(grad_u) + grad(p) + tlift(tau2) = f")
            for position in (-1.0, 2.0)
                if wall == :no_slip
                    no_slip!(problem, "u", "z", position)
                else
                    add_bc!(problem, stress_free_bc("u", "z", position; component_coordinates=collect(labels)))
                end
            end
            add_bc!(problem, "integ(p) = 0")
            solver = BoundaryValueSolver(problem)
            solve!(solver)
            xs, zs = vec(local_grid(xb, dist, 1)), vec(local_grid(zb, dist, 1))
            ux = [dg((z+1)/H)/H*sin(x) for x in xs, z in zs]
            uz = [-g((z+1)/H)*cos(x) for x in xs, z in zs]
            pressure = [((z+1)/H-0.5)*cos(x) for x in xs, z in zs]
            orient(A) = labels[1] == "x" ? A : permutedims(A)
            @test maximum(abs, grid_data!(u.components[ix]) .- orient(ux)) < 2e-10
            @test maximum(abs, grid_data!(u.components[iz]) .- orient(uz)) < 2e-10
            @test maximum(abs, grid_data!(p) .- orient(pressure)) < 2e-10
            @test maximum(abs, grid_data!(evaluate(Divergence(u), :g))) < 2e-10
            for position in (-1.0, 2.0)
                @test maximum(abs, grid_data!(evaluate(interpolate(u.components[iz], coords["z"], position)))) < 2e-10
                tangential = wall == :no_slip ? u.components[ix] : Differentiate(u.components[ix], coords["z"], 1)
                @test maximum(abs, grid_data!(evaluate(interpolate(tangential, coords["z"], position)))) < 2e-10
            end
        end
    end
end
