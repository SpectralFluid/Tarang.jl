# The full-complex FFTW reference covers folds on intersections of Nyquist
# planes, including reflections on the other axes of a real half spectrum.
function check_gpu_workspace_cases(arch, ext, alloc_bytes; reusable_real_fft=true)
    @testset "GPU grid resize parity and buffer reuse" begin
        rng = MersenneTwister(84)
        for T in (Float32, ComplexF32), (small, large) in (
                ((1,), (8,)), ((8,), (13,)), ((9,6), (6,11)),
                ((8,6), (12,10)), ((1,5,6), (7,4,8)), ((7,6,5), (4,8,4)))
            for (a,b) in ((small,large), (large,small))
                data = randn(rng, T, a)
                src = on_architecture(arch, data)
                dst = similar(src, b)
                old_fft = Tarang.FFTW.fft(data)
                target = similar(old_fft, b)
                rules = ntuple(d -> (a[d], b[d], :c2c), length(a))
                Tarang._resample_fourier_spectrum!(target, old_fft, rules)
                expected = Tarang.FFTW.ifft(target)
                Tarang.resample_grid_data!(dst, src, a, b)
                @test Array(dst) ≈ expected atol=6f-5
                Tarang.synchronize(arch)
                cache = Tarang._gpu_workspace_cache(ext._GRID_RESAMPLE_CACHE)
                key = (arch.device, Tarang._gpu_workspace_queue(arch), T, length(a))
                work = cache[key]
                before = alloc_bytes()
                Tarang.resample_grid_data!(dst, src, a, b)
                Tarang.synchronize(arch)
                @test cache[key] === work
                if reusable_real_fft || T <: Complex
                    @test alloc_bytes() == before
                end
                @test Array(src) == data
            end
        end
    end

    @testset "Workspace ownership and scale changes" begin
        coords = CartesianCoordinates("x","z")
        bases = (RealFourier(coords["x"]; size=16), ChebyshevT(coords["z"]; size=9))
        dists = ntuple(_ -> Distributor(coords; device=arch, dtype=Float32, comm=Tarang.MPI.COMM_SELF), 2)
        fields = map(d -> ScalarField(Domain(d,bases), "u"), dists)
        function exercise(f)
            coeff_data!(f)
            grid_data!(f)
            return nothing
        end
        try
            foreach(f -> set!(f, (x,z)->sin(x)*z), fields)
            reference = Array(grid_data!(fields[2]))
            foreach(exercise, fields)
            @test all(d -> !isempty(d.gpu_workspaces), dists)
            @test first(values(dists[1].gpu_workspaces)) !== first(values(dists[2].gpu_workspaces))
            worker_queues_id = fetch(@async begin
                exercise(fields[1])
                objectid(dists[1].gpu_workspaces[current_task()])
            end)
            @test worker_queues_id != objectid(dists[1].gpu_workspaces[current_task()])
            periodic = (first(bases), RealFourier(coords["z"]; size=9))
            f = ScalarField(Domain(dists[1], periodic), "periodic")
            set!(f, (x,z) -> cos(x)*sin(z))
            initial = Array(grid_data!(f))
            coeffs = get_coeff_data(f)
            set_scales!(f, 1.5)
            set_scales!(f, 1.0)
            @test get_coeff_data(f) === coeffs
            @test Array(grid_data!(f)) ≈ initial atol=6f-5
            resize_cache = Tarang._with_gpu_workspace(dists[1]) do
                cache = Tarang._gpu_workspace_cache(ext._GRID_RESAMPLE_CACHE)
                @test length(cache) == 1
                cache
            end
            clear_gpu_workspaces!(dists[1])
            @test isempty(dists[1].gpu_workspaces)
            @test isempty(resize_cache)
            exercise(fields[1])
            close(dists[1])
            @test isempty(dists[1].gpu_workspaces)
            @test !isempty(dists[2].gpu_workspaces)
            Tarang.synchronize(arch)
            before = alloc_bytes()
            exercise(fields[2])
            Tarang.synchronize(arch)
            if reusable_real_fft
                @test alloc_bytes() == before
            end
            @test Array(grid_data!(fields[2])) ≈ reference atol=6f-5
        finally
            foreach(close, dists)
        end
    end
end
