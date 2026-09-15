using CairoMakie
using Printf

module BVPExamples
include(joinpath(@__DIR__, "../../examples/bvp/poisson.jl"))
include(joinpath(@__DIR__, "../../examples/bvp/lane_emden.jl"))
end

function main()
    poisson = BVPExamples.poisson_example()
    resolutions = [16, 24, 32, 48, 64]
    radial = [BVPExamples.lane_emden_example(Nr=n, check_accuracy=n >= 48) for n in resolutions]
    lane = last(radial)
    @assert radial[1].radius_error > radial[2].radius_error > radial[3].radius_error
    output = normpath(joinpath(@__DIR__, "../src/assets/figures/bvp"))
    mkpath(output)
    for name in ("poisson", "lane_emden")
        cp(joinpath(@__DIR__, "../../examples/bvp", name * ".jl"),
           joinpath(output, name * ".jl"); force=true)
    end

    open(joinpath(output, "poisson_fields.csv"), "w") do io
        println(io, "x,y,forcing,solution,residual")
        for j in eachindex(poisson.y), i in eachindex(poisson.x)
            @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g\n", poisson.x[i], poisson.y[j],
                    poisson.forcing[i,j], poisson.solution[i,j], poisson.residual[i,j])
        end
    end
    open(joinpath(output, "lane_emden_profiles.csv"), "w") do io
        println(io, "r,initial,solution,weighted_radial_residual")
        for i in eachindex(lane.r)
            @printf(io, "%.17g,%.17g,%.17g,%.17g\n", lane.r[i], lane.initial[i],
                    lane.solution[i], lane.residual[i])
        end
    end
    open(joinpath(output, "lane_emden_refinement.csv"), "w") do io
        println(io, "N,recovered_radius,reference_radius,radius_error,weighted_radial_residual,origin_residual,boundary_error")
        for (n, result) in zip(resolutions, radial)
            @printf(io, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", n,
                    result.recovered_radius, result.reference_radius, result.radius_error,
                    result.residual_error, result.origin_error, result.wall_error)
        end
    end

    ink, gray, blue = "#172B42", "#536477", "#2166AC"
    CairoMakie.activate!()
    set_theme!(Theme(fontsize=16, textcolor=ink, backgroundcolor=:white,
        Axis=(; titlefont=:bold, titlesize=18, topspinevisible=false,
              rightspinevisible=false, xgridvisible=false,
              ygridcolor=(:black, 0.07), xticklabelsize=14, yticklabelsize=14)))
    fig = Figure(size=(1040, 780), figure_padding=24)
    Label(fig[0, 1:2], "Poisson equation with mixed boundary conditions",
          fontsize=24, font=:bold, halign=:left)
    xclosed = vcat(poisson.x, 2π)
    for (row, values, title, label) in
            ((1, poisson.forcing, "Filtered random forcing", "f"),
             (2, poisson.solution, "Computed solution", "u"))
        ax = Axis(fig[row, 1]; title, xlabel="x", ylabel="y", aspect=DataAspect(),
                  xticks=([0, π, 2π], ["0", "π", "2π"]),
                  yticks=([0, π/2, π], ["0", "π/2", "π"]))
        amplitude = maximum(abs, values)
        hm = heatmap!(ax, xclosed, poisson.y, vcat(values, values[1:1, :]);
                      colormap=:balance, colorrange=(-amplitude, amplitude), rasterize=2)
        xlims!(ax, 0, 2π)
        ylims!(ax, 0, π)
        Colorbar(fig[row, 2], hm; label, width=18)
    end
    Label(fig[3, 1:2],
          "256 × 128 Fourier–Chebyshev  ·  u(x, 0) = 0.025 sin(8x)  ·  ∂u/∂y(x, π) = 0",
          fontsize=13, color=gray, halign=:left)
    colsize!(fig.layout, 1, Aspect(1, 2.0))
    resize_to_layout!(fig)

    shapes = Figure(size=(1040, 500), figure_padding=24)
    Label(shapes[0, 1:2], "Lane–Emden equation · n = 3", fontsize=25,
          font=:bold, halign=:left)
    ax = Axis(shapes[1, 1], title="Nonzero radial solution", xlabel="Rescaled radius r",
              ylabel="f(r)", xticks=0:0.25:1)
    lines!(ax, lane.r, lane.initial; color=gray, linestyle=:dash,
           linewidth=2, label="Initial guess")
    lines!(ax, lane.r, lane.solution; color=blue, linewidth=2.5,
           label="Converged solution")
    scatter!(ax, lane.r, lane.solution; color=blue, markersize=5)
    axislegend(ax; position=:rt, framevisible=false, labelsize=14)
    xlims!(ax, 0, 1)
    ylims!(ax, 0, 7.2)
    axerror = Axis(shapes[1, 2], title="Radius agrees under refinement",
                   xlabel="Chebyshev coefficients N", ylabel="|R − Rreference|",
                   xticks=resolutions, yscale=log10, yticks=10.0 .^ (-14:2:-6))
    errors = [max(result.radius_error, eps(result.reference_radius)) for result in radial]
    lines!(axerror, resolutions, errors; color=blue, linewidth=2)
    scatter!(axerror, resolutions, errors; color=blue, markersize=10)
    Label(shapes[2, 1:2], @sprintf("R = f(0) = %.12f  ·  Float64 / CPU  ·  Errors below eps(Rreference) shown at that value",
                                  lane.recovered_radius),
          fontsize=13, color=gray, halign=:left)
    for (name, figure) in (("poisson", fig), ("lane_emden", shapes))
        save(joinpath(output, name * ".svg"), figure)
        save(joinpath(output, name * ".png"), figure; px_per_unit=2)
    end
    println("BVP figures, source scripts, and CSV data saved to ", output)
end

main()
