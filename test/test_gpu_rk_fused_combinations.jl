using Test
using Tarang

@testset "RK fused combinations preserve coefficients and storage" begin
    for T in (Float32, Float64, ComplexF64)
        base = fill(T(0.125), 1024)
        terms = (fill(T(1e8), 1024), fill(T(-2e8), 1024), fill(T(NaN), 1024))
        weights = (1e-16, -2e-16, 0.0)
        expected = copy(base)
        for (w, term) in zip(weights, terms)
            iszero(w) || (expected .+= w .* term)
        end
        dest = similar(base)
        @test Tarang._rk_combine_arrays!(dest, base, terms, weights) === dest
        @test dest ≈ expected
        @test dest[1] != base[1]  # tiny dt still contributes against a large RHS
        @test all(==(T(1e8)), terms[1])
        # Destination may alias the base for a recycled final update.
        Tarang._rk_combine_arrays!(dest, dest, terms, weights)
        @test dest ≈ expected .+ (expected .- base)
        Tarang._rk_combine_arrays!(dest, base, terms, weights)
        @test (@allocated Tarang._rk_combine_arrays!(dest, base, terms, weights)) < 4096
    end
end

@testset "Fused IMEX stage includes all implicit contributions" begin
    base = fill(1.0 + 0.3im, 1024)
    L = collect(range(0.1, 2.0; length=1024))
    terms = (fill(2.0 + 0.1im, 1024), fill(-0.1 + 0.2im, 1024))
    weights = (0.3, -0.7)
    gamma = 0.2
    denominator = Base.Broadcast.broadcasted(+, 1, Base.Broadcast.broadcasted(*, gamma, L))
    multiplied = (terms[1], Base.Broadcast.broadcasted(*, L, terms[2]))
    expected = (base .+ weights[1] .* terms[1] .+ weights[2] .* L .* terms[2]) ./ (1 .+ gamma .* L)
    dest = similar(base)
    @test Tarang._rk_combine_arrays_divide!(dest, base, multiplied, weights, denominator) === dest
    @test dest ≈ expected
end

@testset "Fused IMEX preserves implicit multiplication range" begin
    domain = PeriodicDomain(8)
    base = ScalarField(domain, "base")
    stage = ScalarField(domain, "stage")
    forcing = ScalarField(domain, "forcing")
    dest = ScalarField(domain, "dest")
    for field in (base, stage, forcing, dest)
        ensure_layout!(field, :c)
    end
    fill!(get_coeff_data(base), 1.0)
    fill!(get_coeff_data(stage), 1e200)
    fill!(get_coeff_data(forcing), 0.0)
    L = fill(1e200, size(get_coeff_data(base)))
    expected = copy(get_coeff_data(base))
    Tarang._ddirk_axpy_lhat!(expected, -1e-200, L, get_coeff_data(stage))
    Tarang._serial_rk_imex_combine!(dest, base, (forcing,), (stage,), (0.0,), (-1e-200,), L, 0.0)
    @test all(isfinite, get_coeff_data(dest))
    @test get_coeff_data(dest) ≈ expected
end
