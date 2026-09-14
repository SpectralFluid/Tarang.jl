# Shared full-file regression for CPU and device-backed fields. The caller
# supplies the distributor (and a transform backend for coefficient output).
function test_output_postprocess_isolation(dist)
    xb = RealFourier(dist.coordsys["x"]; size=8, bounds=(0.0, 2pi))
    u = ScalarField(Domain(dist, (xb,)), "u")
    for layout in (:g, :c)
        @testset "postprocess isolation layout=$layout" begin
            h = Tarang.NetCDFFileHandler(joinpath(mktempdir(), "isolated"), dist, Dict("u" => u))
            Tarang.add_task!(h, u; name="before", layout)
            Tarang.add_task!(h, u; name="doubled", layout,
                postprocess=data -> (data .*= 2; view(data, :)))
            Tarang.add_task!(h, u; name="after", layout)
            for rec in 1:2
                copyto!(grid_data!(u), rec .+ cos.(collect(0:7) .* (2pi/8)))
                ensure_layout!(u, layout)
                storage = layout == :g ? get_grid_data(u) : get_coeff_data(u)
                expected = Array(copy(storage))
                @test Tarang.process!(h; iteration=rec, sim_time=0.25rec)
                @test Array(storage) == expected
                file = Tarang.current_file(h)
                for (task, factor) in (("before", 1), ("doubled", 2), ("after", 1))
                    written = Tarang.group_ncread(file, "vars", task)
                    values = layout == :c ? complex.(written[rec,1,:], written[rec,2,:]) : written[rec,:]
                    @test values ≈ factor .* expected atol=2e-12
                end
            end

            # A callback failure must preserve the source for a retry, even if
            # the callback has already overwritten its entire input buffer.
            retry = Tarang.NetCDFFileHandler(joinpath(mktempdir(), "retry"), dist, Dict("u" => u))
            fail = Ref(true)
            Tarang.add_task!(retry, u; name="changed", layout, postprocess=data -> begin
                data .*= 3
                fail[] && error("postprocess failed after mutation")
                data
            end)
            Tarang.add_task!(retry, u; name="original", layout)
            ensure_layout!(u, layout)
            storage = layout == :g ? get_grid_data(u) : get_coeff_data(u)
            expected = Array(copy(storage))
            @test_throws ErrorException Tarang.process!(retry; iteration=1, sim_time=0.25)
            @test Array(storage) == expected
            @test retry.file_write_num == 0
            fail[] = false
            @test Tarang.process!(retry; iteration=1, sim_time=0.25)
            @test Array(storage) == expected
            for (task, factor) in (("changed", 3), ("original", 1))
                written = Tarang.group_ncread(Tarang.current_file(retry), "vars", task)
                values = layout == :c ? complex.(written[1,1,:], written[1,2,:]) : written[1,:]
                @test values ≈ factor .* expected atol=2e-12
            end
        end
    end
end
