using Test, Tarang, FFTW, Random

function _sg_fields(shape, ::Type{T}=Float64; device=CPU(), mixed=false) where T
    names = ("x", "y", "z")[1:length(shape)]
    coords = CartesianCoordinates(names...)
    dist = Distributor(coords; dtype=T, device)
    bases = ntuple(length(shape)) do d
        mixed && d == length(shape) ? ChebyshevT(coords[names[d]]; size=shape[d]) :
        (T <: Real ? RealFourier : ComplexFourier)(coords[names[d]];
            size=shape[d], bounds=(0.0, 2π), dealias=1.5)
    end
    domain = Domain(dist, bases)
    return ntuple(i -> ScalarField(domain, "sg_$i"), 3)
end

@testset "Padded operand reuse is scoped and uses existing storage" begin
    for T in (Float64, ComplexF64)
        a, b, out = _sg_fields((12, 10), T)
        grid_data!(a) .= T(2)
        grid_data!(b) .= T(3)
        ev = Tarang._get_evaluator(a.dist)
        ws = Tarang._get_padded_workspace!(ev, a.bases, real(T); real_input=T <: Real)
        token = Ref(0)
        @test hasproperty(ws, :cached_operand)
        if hasproperty(ws, :cached_operand)
            Tarang.evaluate_padded_multiply(a, b, ev, ws; destination=out, operand_key=token)
            @test Array(grid_data!(out)) ≈ fill(T(6), 12, 10)
            cached_grid = copy(ws.padded2)
            grid_data!(a) .= T(4)
            Tarang.evaluate_padded_multiply(a, b, ev, ws; destination=out, operand_key=token)
            @test Array(grid_data!(out)) ≈ fill(T(12), 12, 10)
            @test ws.padded2 == cached_grid
            # A new RHS scope has a new token, so changed operands must refresh.
            grid_data!(b) .= T(5)
            Tarang.evaluate_padded_multiply(a, b, ev, ws; destination=out, operand_key=Ref(1))
            @test Array(grid_data!(out)) ≈ fill(T(20), 12, 10)
        end
    end
end

function _sg_complex_reference(a, b, dims)
    shape = size(a)
    padded_shape = ntuple(d -> d in dims ? 2cld(ceil(Int, 1.5shape[d]), 2) : shape[d], ndims(a))
    pa = zeros(Complex{real(eltype(a))}, padded_shape)
    pb = similar(pa)
    Tarang._pad_spectral!(pa, fft(a, dims), shape, padded_shape, dims)
    Tarang._pad_spectral!(pb, fft(b, dims), shape, padded_shape, dims)
    product = fft(ifft(pa, dims) .* ifft(pb, dims), dims)
    truncated = zeros(eltype(pa), shape)
    Tarang._truncate_spectral!(truncated, product, shape, padded_shape, dims)
    return real.(ifft(truncated, dims)) .* prod(padded_shape[d]/shape[d] for d in dims)
end

@testset "Real padded FFT uses reduced spectra" begin
    for shape in ((9, 10), (10, 9, 8)), mixed in (false, true), T in (Float32, Float64)
        a, b, out = _sg_fields(shape, T; mixed)
        rng = MersenneTwister(200)
        aa, bb = randn(rng, T, shape), randn(rng, T, shape)
        grid_data!(a) .= aa
        grid_data!(b) .= bb
        ev = Tarang._get_evaluator(a.dist)
        Tarang._dealiased_lazy_product!(out, a, b)
        dims = findall(b -> b isa Union{RealFourier,ComplexFourier}, collect(a.bases))
        @test Array(grid_data!(out)) ≈ _sg_complex_reference(aa, bb, dims) rtol=200eps(T) atol=200eps(T)
        configs = values(ev.pencil_transforms.padded_dealiasing)
        # At least one workspace uses real padded grids instead of full complex ones.
        @test any(ws -> any(k -> getfield(ws, k) isa AbstractArray{T}, fieldnames(typeof(ws))), configs)
        # Reuse with identical operands and no assumptions about Nyquist content.
        Tarang._dealiased_lazy_product!(out, a, a)
        @test Array(grid_data!(out)) ≈ _sg_complex_reference(aa, aa, dims) rtol=200eps(T) atol=200eps(T)
    end
end

@testset "Nonlinear scratch memory budget" begin
    for shape in ((16, 12), (12, 10, 8)), T in (Float32, Float64, ComplexF64)
        a, b, out = _sg_fields(shape, T)
        ev = Tarang._get_evaluator(a.dist)
        ws = Tarang._get_padded_workspace!(ev, a.bases, real(T))
        buffers = [getfield(ws, k) for k in fieldnames(typeof(ws))
                   if getfield(ws, k) isa AbstractArray{<:Complex}]
        bytes = sum(sizeof, unique(objectid, buffers))
        budget = sizeof(Complex{real(T)}) *
                 (2prod(ws.padded_shape) + prod(ws.original_shape))
        @test bytes <= budget
    end
end

@testset "Lazy nonlinear product writes its supplied destination" begin
    for shape in ((16, 12), (12, 10, 8)), T in (Float32, Float64, ComplexF64)
        a, b, out = _sg_fields(shape, T)
        x = reshape(T.(sin.(2π .* (0:shape[1]-1) ./ shape[1])),
                    (shape[1], ntuple(_ -> 1, length(shape)-1)...))
        grid_data!(a) .= x
        grid_data!(b) .= T(2)
        expected = Array(grid_data!(a)) .* T(2)
        ev = Tarang._get_evaluator(a.dist)
        @test Tarang._dealiased_lazy_product!(out, a, b) === out
        @test Array(grid_data!(out)) ≈ expected rtol=100eps(real(T)) atol=100eps(real(T))
        @test isempty(ev.nl_result_pool)
        # Inputs may alias each other and the destination, after their final read.
        Tarang._dealiased_lazy_product!(a, a, b)
        @test Array(grid_data!(a)) ≈ expected rtol=100eps(real(T)) atol=100eps(real(T))
    end
end

@testset "Pure Fourier products preserve coefficient-resident inputs" begin
    for T in (Float64, ComplexF64), shape in ((9, 10), (10, 9, 8))
        a, b, out = _sg_fields(shape, T)
        rng = MersenneTwister(123)
        grid_data!(a) .= randn(rng, T, shape)
        grid_data!(b) .= randn(rng, T, shape)
        ev = Tarang._get_evaluator(a.dist)
        ref = copy(Array(grid_data!(evaluate_transform_multiply(a, b, ev))))
        ensure_layout!(a, :c); ensure_layout!(b, :c)
        ac, bc = copy(Array(get_coeff_data(a))), copy(Array(get_coeff_data(b)))
        result = evaluate_transform_multiply(a, b, ev; result_layout=:c)
        @test a.current_layout == :c
        @test b.current_layout == :c
        @test Array(get_coeff_data(a)) == ac
        @test Array(get_coeff_data(b)) == bc
        @test result.current_layout == :c
        @test Array(grid_data!(result)) ≈ ref rtol=1e-12 atol=1e-12
    end
end

@testset "Mixed precision nonlinear inputs keep conversion behavior" begin
    a, _, _ = _sg_fields((8, 8), Float32)
    b = ScalarField(a.dist, "wide", a.bases, Float64)
    grid_data!(a) .= 2f0
    grid_data!(b) .= 3.0
    product = evaluate_transform_multiply(a, b, Tarang._get_evaluator(a.dist))
    @test eltype(grid_data!(product)) === Float32
    @test grid_data!(product) ≈ fill(6f0, 8, 8)
end
