using Random: MersenneTwister
using Tarang: MPI

function check_forcing_grid_registration(arch)
    @testset "Real FFT forcing matches the physical grid on $arch" for n in (8, 9)
        domain = PeriodicDomain(n; device=arch, dtype=Float32)
        try
            u = ScalarField(domain, "u")
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) = 0")
            make(m) = StochasticForcing(; field_size=(m,), architecture=arch, dtype=Float32,
                spectrum_type=:band, k_forcing=2, dk_forcing=0.25, dt=0.125,
                rng=MersenneTwister(42))
            valid = make(n)
            add_stochastic_forcing!(problem, :u, valid)
            wrong = make(17 - n)  # 8 and 9 both store five real-FFT coefficients
            preview = Array(copy(generate_forcing!(wrong, 0.0)))
            rng = copy(wrong.rng)
            @test_throws ArgumentError add_stochastic_forcing!(problem, :u, wrong)
            @test problem.stochastic_forcings[1] === valid
            @test wrong.rng == rng && Array(wrong.cached_forcing) == preview
            step!(InitialValueSolver(problem, RK222(); dt=0.125))
            rate = sum(abs2, Array(grid_data!(u))) / (2 * n * 0.125)
            @test rate ≈ 1 atol=2e-6
        finally
            close(domain.dist)
        end
    end
end

# The same plain local coefficient buffers are used on CPU reference arrays,
# CuArray and ROCArray. Check ownership independently from the forcing matcher.
function check_distributed_forcing_slabs(arch, comm)
    nranks, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    nx, ny, nz = 2 * nranks + 1, 6, 4
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; comm, mesh=(nranks,), device=arch, dtype=Float32, use_pencil_arrays=false)
    try
        @testset "Rank-local stochastic forcing on $arch, mixed=$mixed" for mixed in (false, true)
            z = mixed ? ChebyshevT(coords["z"]; size=nz) : ComplexFourier(coords["z"]; size=nz)
            domain = Domain(dist, (ComplexFourier(coords["x"]; size=nx),
                                   ComplexFourier(coords["y"]; size=ny), z))
            u = ScalarField(domain, "u")
            kwargs = (; architecture=arch, dtype=Float32, dt=0.125, k_forcing=2,
                      rng=MersenneTwister(64 + rank))
            forcing = mixed ? SeparableStochasticForcing(; fourier_size=(nx,ny),
                chebyshev_basis=z, chebyshev_profile=x -> 1 + x, kwargs...) :
                StochasticForcing(; field_size=(nx,ny,nz), kwargs...)

            rank > 0 && generate_forcing!(forcing, 0.0)
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) = 0")
            add_stochastic_forcing!(problem, :u, forcing)
            for time in (0.0, 0.125, 0.25)
                draw = Array(generate_forcing!(forcing, time))
                reference = MPI.Bcast!(copy(draw), comm; root=0)
                @test draw == reference
                expected = reference[Tarang.local_range(nx, nranks, rank), :, :]
                data = get_coeff_data(u)
                fill!(data, 0)
                u.current_layout = :c
                mixed || store_prevsol!(forcing, u)
                Tarang._add_registered_forcings_to_lazy_rhs!([u], problem)
                @test Array(data) == expected
                apply_forcing!(u, forcing, time, 2)
                @test Array(data) == 2 .* expected
                if !mixed
                    power = 2 * sum(abs2, ComplexF64.(reference)) / (nx * ny * nz)^2
                    @test instantaneous_power(forcing, u) ≈ power rtol=4e-6
                    @test work_ito(forcing, u) ≈ forcing.dt * (power + forcing.energy_injection_rate) rtol=4e-6
                    @test work_stratonovich(forcing, u) ≈ forcing.dt * power / 2 rtol=4e-6
                end
            end
        end
    finally
        close(dist)
    end
end
