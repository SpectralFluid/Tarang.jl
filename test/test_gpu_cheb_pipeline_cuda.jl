module GPUChebPipelineCUDA
using Test, Tarang, CUDA

@testset "Fused complex DCT staging and coefficient-space derivatives on CUDA" begin
    if !CUDA.functional()
        @test_skip "No functional CUDA device; native kernel/allocation checks require GPU CI"
    else
        CUDA.allowscalar(false)
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @testset "Lazy derivative buffers and FFT cache dispatch" begin
            plan = ext.GPUChebyshevDerivPlan{Float64}(5, 3,
                CUDA.zeros(Float64, 8, 3), CUDA.zeros(ComplexF64, 5, 3),
                nothing, nothing, nothing, nothing)
            @test all(getfield(plan, slot) === nothing for
                      slot in (:work_real, :work_deriv, :work_perm))
            buffers = map((Val(:work_real), Val(:work_deriv), Val(:work_perm))) do slot
                buffer = ext._plan_work!(plan, slot)
                @test size(buffer) == (5, 3)
                @test ext._plan_work!(plan, slot) === buffer
                buffer
            end
            @test all(buffers[i] !== buffers[j] for i in 1:3 for j in 1:i-1)
            for real_input in (false, true)
                @test ext.get_gpu_fft_plan(GPU(nothing), (7, 5), Float64; real_input) ===
                      ext.get_gpu_fft_plan(GPU(CUDA.device()), (7, 5), Float64; real_input)
            end
        end
        for T in (Float32, Float64), axis in 2:3
            original = randn(Complex{T}, 5,7,9)
            input, output = CuArray(original), CuArray(zeros(Complex{T},size(original)))
            scratch_count = length(ext._GPU_DCT_SCRATCH_CACHE)
            ext.gpu_dct1_along_dim!(output,input,axis,:forward)
            @test Array(input) == original
            ext.gpu_dct1_along_dim!(output,output,axis,:backward)
            @test Array(output) ≈ original rtol=50eps(T) atol=50eps(T)
            @test length(ext._GPU_DCT_SCRATCH_CACHE) == scratch_count
        end

        shape = (9,10,11)
        scale = 2.0 / 3.7
        for T in (Float32,Float64,ComplexF32,ComplexF64), axis in 1:3
            R = typeof(real(zero(T)))
            n, degree = shape[axis], 8
            nodes = R.(-cos.(pi .* (0:n-1) ./ (n-1)))
            amplitude = [T((1 + sum(0.03*d*I[d] for d in 1:3 if d != axis)) *
                           (T <: Complex ? 1+0.7im : 1)) for I in CartesianIndices(shape)]
            coordinate = reshape(nodes, ntuple(d -> d == axis ? n : 1,3))
            original = amplitude .* coordinate.^degree
            input, output = CuArray(original), CuArray(zeros(T,shape))
            derivative! = T <: Complex ? ext._gpu_cheb_deriv_complex_into! : ext._gpu_cheb_deriv_into!
            orders = R === Float32 ? (0,1,2,3,n) : (0,1,2,3,4,8,n)
            for order in orders
                exact = order > degree ? zeros(T,shape) :
                        amplitude .* R(prod(degree-order+1:degree)*scale^order) .* coordinate.^(degree-order)
                tolerance = (R === Float32 ? 3e-3 : 5e-8) * max(1,maximum(abs,exact))
                @test derivative!(output,input,axis,order,scale) === output
                @test Array(output) ≈ exact atol=tolerance rtol=0
                @test Array(input) == original
                inplace = copy(input)
                derivative!(inplace,inplace,axis,order,scale)
                @test Array(inplace) ≈ exact atol=tolerance rtol=0
            end
            derivative!(output,input,axis,3,scale)
            CUDA.synchronize()
            stats = getfield(CUDA,:alloc_stats)
            bytes_before = stats.alloc_bytes
            derivative!(output,input,axis,3,scale)
            CUDA.synchronize()
            @test stats.alloc_bytes == bytes_before
        end
    end
end
end
