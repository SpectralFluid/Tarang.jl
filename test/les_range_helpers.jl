# Shared analytic cases for CPU, reference device arrays, and real CUDA.
# G = diag(-a, a [, 0]) is incompressible. With widths (2, 1 [, 3]),
# the exact unclipped predictors are nu = 3*C*a/2 and kappa = 4*C*a
# for any nonzero scalar gradient (b, 0 [, 0]), independent of b's units.
function check_amd_gradient_range(make_model, to_array=identity)
    @testset "AMD gradient range T=$T N=$N clip=$clip" for
        T in (Float32, Float64), N in (2, 3), clip in (false, true)
        dims = ntuple(_ -> 2, N)
        widths = N == 2 ? (2, 1) : (2, 1, 3)
        model = make_model(; filter_width=widths, field_size=dims,
                           dtype=T, clip_negative=clip)
        small_u = T == Float32 ? T(1e-16) : T(1e-110)
        small_b = T == Float32 ? T(1e-25) : T(1e-170)
        large = T == Float32 ? T(1e20) : T(1e160)
        cases = ((small_u, one(T)), (large, one(T)),
                 (one(T), small_b), (one(T), large), (small_u, large),
                 (large, small_b))
        for (amplitude, b) in cases, sign in (-one(T), one(T))
            a = sign * amplitude
            gradients = [zeros(T, dims) for _ in 1:N^2]
            gradients[1] .= -a
            gradients[N + 2] .= a
            scalar = [zeros(T, dims) for _ in 1:N]
            scalar[1] .= b
            # Evaluate the reference in wider arithmetic, without squaring the
            # tiny/large inputs that made the original implementation fail.
            expected_nu = T(BigFloat(3) * BigFloat(model.C) * BigFloat(a) / 2)
            expected_kappa = T(4 * BigFloat(model.C) * BigFloat(a))
            if clip
                expected_nu = max(zero(T), expected_nu)
                expected_kappa = max(zero(T), expected_kappa)
            end
            g = to_array.(gradients)
            bg = to_array.(scalar)
            nu = Array(compute_eddy_viscosity!(model, g...))
            kappa = Array(compute_eddy_diffusivity!(model, g..., bg...))
            @test all(x -> isapprox(x, expected_nu; rtol=8eps(T), atol=0), nu)
            @test all(x -> isapprox(x, expected_kappa; rtol=8eps(T), atol=0), kappa)
        end
        zeros_g = [to_array(zeros(T, dims)) for _ in 1:N^2]
        zeros_b = [to_array(zeros(T, dims)) for _ in 1:N]
        @test all(iszero, Array(compute_eddy_viscosity!(model, zeros_g...)))
        @test all(iszero, Array(compute_eddy_diffusivity!(model, zeros_g..., zeros_b...)))
    end
end
