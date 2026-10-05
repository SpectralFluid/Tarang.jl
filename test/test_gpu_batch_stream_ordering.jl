using Test, Tarang, LinearAlgebra, SparseArrays
try
    using CUDA
catch
end

if !(@isdefined CUDA) || !CUDA.functional()
    @testset "Batched CUDA stream ordering" begin
        @test_skip "A functional CUDA GPU is required"
    end
else
    CUDA.allowscalar(false)
    @testset "Batched CUDA stream ordering" begin
        # Chain all stage kernels with broadcasts and cuBLAS on a nondefault
        # stream. Only the final host copy waits for results. Uneven dimensions
        # and an offset gather also exercise partially occupied workgroups.
        for stream in (CUDA.stream(), CUDA.CuStream())
            CUDA.stream!(stream) do
                n, modes = 7, 5
                host = reshape(ComplexF64.(1:n*modes), modes, n)
                source = CuArray(host)
                starts = CuArray(collect(1:modes))
                raw = CUDA.zeros(ComplexF64, n + 2, modes)
                rhs = CUDA.zeros(ComplexF64, n, modes)
                solution = similar(rhs)
                dest = similar(source)
                rows = CuArray(collect(1:n+1))
                cols = CuArray(collect(1:n))
                values = CUDA.fill(ComplexF64(2), n, modes)
                bcrows = CuArray([1, n])
                alg = CUDA.fill(ComplexF64(3), n, modes)
                dense = CUDA.zeros(ComplexF64, n, n, modes)
                mass = CUDA.fill(ComplexF64(1), n, modes)
                source_rows = CuArray(collect(1:n))
                scaling = CUDA.fill(ComplexF64(2), n)

                Tarang.batched_gather!(raw, source, starts, modes, n, 1)
                gathered = @view raw[2:n+1, :]
                Tarang.batched_spmv!(rhs, rows, cols, values, gathered)
                rhs .+= 1
                Tarang.batched_bc_override!(rhs, alg, bcrows, 2)
                Tarang.batched_mass_apply!(solution, rhs, source_rows, scaling)
                Tarang.batched_scatter!(dest, solution, starts, modes, n, 0)
                expected = permutedims(host) .+ 0.5
                expected[[1, n], :] .= 3
                @test Array(dest) == permutedims(expected)

                # Assembly's zero and place launches must remain ordered with
                # factorization, solve, and the next assembly into reused A.
                solver = Tarang.BatchedDenseLU(dense)
                for coefficient in (0.5, 1.5)
                    Tarang.batched_assemble_lhs!(dense, rows, cols, mass, values, coefficient)
                    Tarang.batched_factor!(solver)
                    Tarang.batched_solve!(solution, solver, rhs)
                    @test Array(solution) ≈ (2 .* expected) ./ (1 + 2coefficient)
                end
            end
        end
    end
end
