using CairoMakie
using LinearAlgebra
using Printf
import Tarang
using Tarang: local_grids, ensure_layout!, get_grid_data

# Run the documented example itself so the figures cannot drift to another setup.
function documented_example()
    page = normpath(joinpath(@__DIR__, "../src/problems/eigenvalue.md"))
    blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)\n```"s, read(page, String))]
    example = Module(:DiffusionEigenvalueExample)
    Base.include_string(example, only(blocks), page)
    return example
end

function main(example)
    solver, u, dist, zb = example.solver, example.u, example.dist, example.zb
    order = sortperm(real.(example.eigenvalues); rev=true)
    values = example.eigenvalues[order]
    vectors = example.eigenvectors[:, order]
    modes = collect(1:length(values))
    exact_values = -(π .* modes).^2
    relative_errors = abs.(values .- exact_values) ./ abs.(exact_values)
    z = collect(only(local_grids(dist, zb)))
    profiles = zeros(length(z), length(modes))
    profile_errors = zeros(length(modes))
    wall_errors = zeros(length(modes))
    residuals = zeros(length(modes))
    sp = only(solver.subproblems)
    L, M = Matrix(sp.L_min), Matrix(sp.M_min)

    for n in modes
        v = copy(vectors[:, n])
        residuals[n] = norm(L * v + values[n] * M * v) /
                       ((opnorm(L) + abs(values[n]) * opnorm(M)) * norm(v))
        # A real diffusion eigenmode has arbitrary complex phase and amplitude.
        v .*= cis(-angle(v[argmax(abs.(v))]))
        @assert maximum(abs, imag.(v)) < 1e-10
        Tarang.scatter_inputs(sp, v, [u, example.tau1, example.tau2])
        ensure_layout!(u, :g)
        profile = copy(vec(get_grid_data(u)))
        reference = sin.(n * π .* z)
        profile .*= dot(profile, reference) / dot(profile, profile)
        profiles[:, n] .= profile
        profile_errors[n] = maximum(abs.(profile .- reference))
        wall_errors[n] = maximum(abs, profile[[1, end]])
    end

    @assert maximum(relative_errors) < 1e-8
    @assert maximum(profile_errors) < 1e-8
    @assert maximum(wall_errors) < 1e-10
    @assert maximum(residuals) < 1e-12

    output = normpath(joinpath(@__DIR__, "../src/assets/figures/eigenvalue"))
    mkpath(output)
    open(joinpath(output, "diffusion_modes.csv"), "w") do io
        println(io, "mode,N,sigma_real,sigma_imag,exact_sigma,relative_eigenvalue_error,max_profile_error,max_wall_error,scaled_matrix_residual")
        for n in modes
            @printf(io, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    n, length(z), real(values[n]), imag(values[n]), exact_values[n],
                    relative_errors[n], profile_errors[n], wall_errors[n], residuals[n])
        end
    end
    open(joinpath(output, "diffusion_profiles.csv"), "w") do io
        println(io, "z,mode_1,mode_2,mode_3,mode_4")
        for i in eachindex(z)
            @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g\n", z[i], profiles[i, :]...)
        end
    end

    colors = ["#2166AC", "#D17A22", "#178572", "#8660A8"]
    ink, gray = "#172B42", "#536477"
    CairoMakie.activate!()
    set_theme!(Theme(
        fontsize=16, textcolor=ink, backgroundcolor=:white,
        Axis=(; titlefont=:bold, titlesize=18, spinewidth=1,
              topspinevisible=false, rightspinevisible=false,
              xgridvisible=false, ygridcolor=(:black, 0.07),
              xticklabelsize=14, yticklabelsize=14, xlabelsize=16, ylabelsize=16),
    ))

    spectrum = Figure(size=(1040, 480), figure_padding=24)
    Label(spectrum[0, 1:2], "Diffusion eigenvalues", fontsize=25, font=:bold,
          halign=:left)
    ax = Axis(spectrum[1, 1], title="Decay rate grows with mode number",
              xlabel="Mode n", ylabel="Decay rate −Re(σ)", xticks=modes)
    curve_n = range(1, length(modes); length=200)
    lines!(ax, curve_n, (π .* curve_n).^2; color=gray, linestyle=:dash,
           linewidth=2, label="Exact  (nπ)²")
    scatter!(ax, modes, -real.(values); color=colors, markersize=13,
             strokecolor=:white, strokewidth=1, label="Tarang")
    axislegend(ax; position=:lt, framevisible=false, labelsize=14)
    axerror = Axis(spectrum[1, 2], title="Relative eigenvalue error",
                   xlabel="Mode n", ylabel="|σ − σexact| / |σexact|",
                   xticks=modes, yscale=log10)
    # An exactly rounded result is shown at machine epsilon, identified below.
    plotted_errors = max.(relative_errors, eps(Float64))
    scatter!(axerror, modes, plotted_errors; color=colors, markersize=13)
    xlims!(axerror, 0.6, 4.4)
    hlines!(axerror, [eps(Float64)]; color=(gray, 0.6), linestyle=:dot)
    ylims!(axerror, eps(Float64) / 3, max(maximum(plotted_errors) * 8, 1e-14))
    Label(spectrum[2, 1:2],
          "N = $(length(z)) Chebyshev coefficients  ·  Float64 / CPU  ·  Errors below machine epsilon shown at ε",
          fontsize=13, color=gray, halign=:left)

    shapes = Figure(size=(1040, 740), figure_padding=24)
    Label(shapes[0, 1:2], "The first four diffusion eigenmodes", fontsize=25,
          font=:bold, halign=:left)
    zfine = range(0, 1; length=500)
    for n in modes
        row, col = fldmod(n - 1, 2) .+ (1, 1)
        axmode = Axis(shapes[row, col], title=@sprintf("Mode %d   ·   σ = %.5f", n, real(values[n])),
                      xlabel="z", ylabel="Mode amplitude", xticks=0:0.25:1,
                      yticks=[-1, 0, 1])
        lines!(axmode, zfine, sin.(n * π .* zfine); color=gray,
               linewidth=1.8, linestyle=:dash)
        scatter!(axmode, z, profiles[:, n]; color=colors[n], markersize=8,
                 strokecolor=:white, strokewidth=0.6)
        xlims!(axmode, -0.025, 1.025)
        ylims!(axmode, -1.12, 1.12)
    end
    Legend(shapes[3, 1:2],
           [LineElement(color=gray, linestyle=:dash, linewidth=2),
            MarkerElement(color=colors[1], marker=:circle, markersize=10)],
           ["Exact sin(nπz)", "Tarang at Chebyshev nodes"];
           orientation=:horizontal, framevisible=false, labelsize=15)
    Label(shapes[4, 1:2],
          "Homogeneous Dirichlet walls  ·  Eigenmode sign and amplitude aligned to the analytic sine",
          fontsize=13, color=gray, halign=:left)

    for (name, figure) in (("diffusion_spectrum", spectrum), ("diffusion_eigenmodes", shapes))
        save(joinpath(output, name * ".svg"), figure)
        save(joinpath(output, name * ".png"), figure; px_per_unit=2)
    end
    @printf("Maximum relative eigenvalue error: %.3e\n", maximum(relative_errors))
    @printf("Maximum profile error: %.3e\n", maximum(profile_errors))
    @printf("Maximum wall error: %.3e\n", maximum(wall_errors))
    @printf("Maximum scaled matrix residual: %.3e\n", maximum(residuals))
    println("Figures and data saved to ", output)
end

example = documented_example()
main(example)
