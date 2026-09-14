using Test

@testset "CPU scaling benchmark configuration" begin
    config_file = joinpath(@__DIR__, "..", "scripts", "cpu_parallel_config.jl")
    @test isfile(config_file)
    if isfile(config_file)
        include(config_file)
        default = cpu_benchmark_config(Dict{String,String}())
        @test default.shape == (64, 64)
        @test default.ranks == [1, 2, 4]
        @test default.gather
        @test !cpu_benchmark_config(Dict{String,String}(); worker=true).gather
        c = cpu_benchmark_config(Dict("TARANG_BENCH_SHAPE" => "1024x1024x512",
                                     "TARANG_BENCH_RANKS" => "250,500,1000",
                                     "TARANG_BENCH_THREADS" => "2"))
        @test c.shape == (1024, 1024, 512)
        @test c.ranks == [250, 500, 1000]
        @test c.threads == c.fftw_threads == 2
        @test c.blas_threads == 1
        @test c.mesh === nothing
        @test cpu_benchmark_config(Dict("TARANG_BENCH_MESH" => "40x25"); worker=true).mesh == (40, 25)
        for (key, value) in (("TARANG_BENCH_SHAPE", "4x16"),
                             ("TARANG_BENCH_SHAPE", "16x16x16x16"),
                             ("TARANG_BENCH_RANKS", "0,4"),
                             ("TARANG_BENCH_RANKS", "2,2"),
                             ("TARANG_BENCH_THREADS", "0"),
                             ("TARANG_BENCH_STEPS", "-1"),
                             ("TARANG_BENCH_SAMPLES", "0"),
                             ("TARANG_BENCH_GATHER", "yes"))
            @test_throws ArgumentError cpu_benchmark_config(Dict(key => value))
        end
        @test_throws ArgumentError cpu_benchmark_config(Dict("TARANG_BENCH_MESH" => "2x2"))
    end
end
