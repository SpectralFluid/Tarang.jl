using Tarang, CUDA, FFTW, Test, InteractiveUtils
versioninfo()

function warmed_allocations(f::F, args...) where F
    f(args...)
    @allocated f(args...)
end

@testset "Allocation diagnostics" begin
    ext = Base.get_extension(Tarang, :TarangCUDAExt)
    for shape in ((7,), (8,5), (256,), (64,64)), real_input in (false, true)
        T = real_input ? Float64 : ComplexF64
        packed = zeros(T, shape..., 3)
        dims = ntuple(identity, length(shape))
        forward = real_input ? plan_rfft(packed,dims) : plan_fft(packed,dims)
        spectrum = forward * packed
        inverse = real_input ? plan_irfft(spectrum, shape[1], dims) : plan_ifft(spectrum,dims)
        inputs = [reshape(T.(sin.((1:prod(shape)) .* (i/3))), shape) for i in 1:3]
        outputs = [similar(spectrum, size(spectrum)[1:end-1]...) for _ in 1:3]
        restored = [similar(a) for a in inputs]
        ext._execute_batched_fft!(outputs, inputs, forward, packed, spectrum)
        expected = [real_input ? rfft(a,dims) : fft(a,dims) for a in inputs]
        @test outputs ≈ expected
        saved = deepcopy(outputs)
        ext._execute_batched_fft!(restored, outputs, inverse, spectrum, packed)
        @test restored ≈ inputs
        @test outputs == saved
        original = @allocated ext._execute_batched_fft!(outputs, inputs, forward, packed, spectrum)
        repeats = [@allocated ext._execute_batched_fft!(outputs, inputs, forward, packed, spectrum) for _ in 1:5]
        helper = [warmed_allocations(ext._execute_batched_fft!, outputs, inputs, forward, packed, spectrum) for _ in 1:5]
        println((; shape, real_input, original, repeats, helper))
    end
end

include("test_batched_fft_workspace.jl")
