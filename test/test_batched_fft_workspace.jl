using Test
using Tarang
using FFTW
using CUDA

# Warm the measurement call as well as the executor, and use the minimum of
# several samples to exclude one-off runtime allocations. Keep the callable as a
# specialized argument so the measurement does not capture the testset's locals.
function batched_fft_allocated(execute!::F, args...) where {F}
    execute!(args...)
    @allocated execute!(args...)
    bytes = typemax(Int)
    for _ in 1:5
        bytes = min(bytes, @allocated execute!(args...))
    end
    return bytes
end

@testset "Packed FFT workspace reuses buffers and preserves inputs" begin
    ext = Base.get_extension(Tarang, :TarangCUDAExt)
    # The larger case makes even one new field buffer exceed the allocation
    # budget, so allowing small runtime overhead cannot hide workspace copies.
    for shape in ((7,), (8,5), (64,64)), real_input in (false, true)
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
        bytes = batched_fft_allocated(ext._execute_batched_fft!, outputs, inputs, forward, packed, spectrum)
        @test bytes < 4096
        inverse_bytes = batched_fft_allocated(ext._execute_batched_fft!, restored, outputs, inverse, spectrum, packed)
        @test inverse_bytes < 4096
        @test outputs == saved
    end
end

@testset "Complex batched plans preserve input/output conversion" begin
    ext = Base.get_extension(Tarang, :TarangCUDAExt)
    packed = zeros(ComplexF64, 8, 3)
    spectrum = similar(packed)
    forward = plan_fft(packed, (1,))
    inputs = [sin.((1:8) .* (i/3)) for i in 1:3]
    outputs = [zeros(ComplexF32,8) for _ in 1:3]
    ext._execute_batched_fft!(outputs, inputs, forward, packed, spectrum)
    @test outputs ≈ fft.(inputs) rtol=1e-6
end
