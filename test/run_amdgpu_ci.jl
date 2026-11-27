# Run in an environment containing Tarang and AMDGPU. Pass --require-device
# for a hardware job. mpiexec -n 2 also exercises the distributed pencils.
using Test, Tarang, AMDGPU, LinearAlgebra
using Tarang: MPI, FFTW
using .FFTW: fft, rfft
using Random

const roc = Base.get_extension(Tarang, :TarangAMDGPUExt)
const comm = MPI.COMM_WORLD
include("gpu_workspace_cases.jl")

@testset "AMDGPU extension" begin
    @test roc !== nothing
    @test has_amdgpu() == (AMDGPU.functional() && AMDGPU.has_rocm_gpu())
    @test Tarang.MatSolvers.get_solver(:rocm_dense) === roc.ROCDenseLU
    @test Tarang._matsolver_backend(roc.ROCDenseLU) === :amdgpu
    @test_throws ArgumentError Tarang._select_ivp_matsolver(:rocm_dense, true, true; backend=:cuda)
    @test_throws ArgumentError Tarang._select_ivp_matsolver(roc.ROCDenseLU, true, true; backend=:cuda)
    @test_throws ArgumentError Tarang._select_ivp_matsolver(Tarang.CuSparseLU, true, true; backend=:amdgpu)
    @test which(Tarang._gpu_device, Tuple{Val{:amdgpu},Int}).module === roc
    for f in (Tarang._gpu_forward_transform_backend!, Tarang._gpu_backward_transform_backend!)
        @test which(f, Tuple{GPU{AMDGPU.HIPDevice},ScalarField}).module === roc
    end
    @test which(Tarang.on_architecture, Tuple{GPU{AMDGPU.HIPDevice},Array{Float64,2}}).module === roc
    real_grid = AMDGPU.ROCMatrix{Float32}
    @test which(Tarang.gpu_resample_grid_data!,
                Tuple{real_grid,real_grid,NTuple{2,Int},NTuple{2,Int}}).module === roc
    @test which(Tarang._padded_plan_spectrum,
                Tuple{AMDGPU.rocFFT.rROCFFTPlan,GPU{AMDGPU.HIPDevice},Type{Float32},NTuple{2,Int}}).module === roc
    @test_throws ArgumentError GPU(backend=:unknown)
    if !has_amdgpu()
        @test_throws ErrorException GPU(backend=:amdgpu)
    end
end

# On a driverless runner, execute the actual permutation, planning and inverse
# logic with FFTW so a wrong batch stride or inverse permutation fails CI.
function check_fft_axes(arch)
    for T in (Float32, Float64), shape in ((7,), (1,5), (5,6), (5,6,7)), axis in eachindex(shape),
        real_input in (true, false)
        src = randn(real_input ? T : Complex{T}, shape)
        expected = real_input ? rfft(src, (axis,)) : fft(src, (axis,))
        a = on_architecture(arch, src)
        b = zeros(arch, Complex{T}, size(expected))
        restored = similar(a)
        plan = roc._plan_fft_axis(arch, shape, eltype(a), axis, real_input)
        roc.gpu_fft_dim!(b, a, plan)
        @test Array(b) ≈ expected
        roc.gpu_ifft_dim!(restored, b, plan)
        @test Array(restored) ≈ src
        @test Array(b) ≈ expected
    end
    for dims in ((2,), (3,), (1,3), (2,3), (1,2,3))
        real_grid = randn(5,6,7)
        a = on_architecture(arch, real_grid)
        pf, pb = roc._plan_fft_region(a, dims; real_input=true)
        coeff = on_architecture(arch, rfft(real_grid, dims))
        restored = similar(a)
        actual = similar(coeff)
        mul!(actual, pf, a)
        @test Array(actual) ≈ Array(coeff)
        mul!(restored, pb, actual)
        @test Array(restored) ≈ real_grid
        # FFTW can overwrite its input for a direct multidimensional C2R.
        if pb isa roc.PermutedFFTPlan || is_gpu(arch)
            @test Array(actual) ≈ Array(coeff)
        end

        z = complex.(real_grid, randn(size(real_grid)))
        scratch = on_architecture(arch, copy(z))
        forward, inverse = roc._plan_fft_region(scratch, dims; inplace=true)
        @test forward * scratch === scratch
        @test Array(scratch) ≈ fft(z, dims)
        @test inverse * scratch === scratch
        @test Array(scratch) ≈ z
    end
end
@testset "AMD FFT axis layout on host arrays" begin
    check_fft_axes(CPU())
end

include("gpu_boundary_cases.jl")

# Compare the distributed real half-spectrum layout with a serial field on
# every rank. The CPU version uses the same pencil algorithm and real MPI.
function check_half_pencils(arch; mesh=(MPI.Comm_size(comm),), nx=9)
    coords = CartesianCoordinates("x","y","z")
    bases = (RealFourier(coords["x"]; size=nx), ChebyshevT(coords["y"]; size=7),
             ChebyshevT(coords["z"]; size=6))
    dist = Distributor(coords; device=arch, comm, mesh, use_pencil_arrays=false)
    serial = Distributor(coords; comm=MPI.COMM_SELF)
    try
        src = on_architecture(arch, fill(dist.rank + 1, 3))
        dest = similar(src)
        Tarang.wait_async!(Tarang.async_allreduce!(dest, src, +, dist), dist)
        total = sum(1:dist.size)
        @test Array(dest) == fill(total, 3)
        Tarang.wait_async!(Tarang.async_allreduce!(dest, dest, MPI.SUM, dist), dist)
        @test Array(dest) == fill(total * dist.size, 3)
        f = ScalarField(Domain(dist, bases), "f")
        reference = ScalarField(Domain(serial, bases), "reference")
        input = randn(MersenneTwister(47), nx, 7, 6)
        rx, ry = mesh[1], div(dist.size, mesh[1])
        ix(n) = Tarang.local_range(n, rx, mod(dist.rank, rx))
        iy = Tarang.local_range(size(input,2), ry, div(dist.rank, rx))
        grid_block = input[ix(nx),iy,:]
        copyto!(get_grid_data(f), grid_block)
        copyto!(get_grid_data(reference), input)
        forward_transform!(reference)
        coeff = get_coeff_data(reference)
        coeff_block = coeff[ix(size(coeff,1)),iy,:]
        if arch isa CPU
            g = Tarang._build_transpose_workspace(f, size(input), nothing)
            c = Tarang._build_transpose_workspace(f, size(coeff), nothing)
            try
                real_x = zeros(size(g.buffers.x_local_data))
                p = roc._plan_fft_axis(arch, size(real_x), Float64, 1, true)
                roc._half_pencil_transform!(arch, f, g, c, real_x, p, :forward)
                @test get_coeff_data(f) ≈ coeff_block
                roc._half_pencil_transform!(arch, f, g, c, real_x, p, :backward)
            finally
                close(c)
                close(g)
            end
        else
            forward_transform!(f)
            @test Array(get_coeff_data(f)) ≈ coeff_block
            backward_transform!(f)
        end
        @test Array(get_grid_data(f)) ≈ grid_block
    finally
        close(serial)
        close(dist)
    end
end

if MPI.Comm_size(comm) > 1
    @testset "AMD pencil transforms with host arrays and MPI" begin
        check_half_pencils(CPU())
        if MPI.Comm_size(comm) == 4
            check_half_pencils(CPU(); mesh=(2,2))
            check_half_pencils(CPU(); nx=3)
        end
    end
else
    @testset "Steady boundary reference" begin
        check_steady_boundary(CPU())
    end
end

if has_amdgpu()
    AMDGPU.allowscalar(false)
    node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    arch = try
        # Each process may see one scheduler-assigned GPU, or every GPU on its node.
        count = length(AMDGPU.devices())
        id = count == 1 ? 0 : MPI.Comm_rank(node)
        GPU(backend=:amdgpu, device_id=id)
    finally
        MPI.free(node)
    end
    if MPI.Comm_size(comm) == 1
        @testset "AMD GPU numerical checks" begin
            check_gpu_workspace_cases(arch, roc, () -> AMDGPU.alloc_stats.alloc_bytes)
            sample = AMDGPU.zeros(Float32, 8, 12)
            fwd, invplan, spectrum = roc._plan_real_workspace(arch, sample, (1,2), (5,12))
            @test fwd.buffer === invplan.p.buffer === spectrum
            check_fft_axes(arch)
            check_steady_boundary(arch; expected_solver=roc.ROCDenseLU{AMDGPU.device_id(arch.device)})
            a = on_architecture(arch, reshape(collect(1.0:24), 4,6))
            @test a isa ROCArray{Float64}
            @test Tarang.architecture(view(a, 1:2:4, :)) == arch
            @test on_architecture(CPU(), view(a, 1:2:4, :)) == Array(a)[1:2:4,:]
            @test copy_to_device(a, a) !== a
            @test copy_to_device(a, zeros(1)) isa Matrix{Float64}
            @test copy_to_device(a, zeros(1)) == Array(a)

            coords = CartesianCoordinates("x","z")
            bases = (RealFourier(coords["x"]; size=9), ChebyshevT(coords["z"]; size=8))
            input = randn(MersenneTwister(63), 9, 8)
            fields = map((CPU(), arch)) do backend
                f = ScalarField(Domain(Distributor(coords; device=backend), bases), "u")
                copyto!(get_grid_data(f), input)
                forward_transform!(f)
                f
            end
            cpu, gpu = fields
            @test get_coeff_data(gpu) isa ROCArray
            @test Array(get_coeff_data(gpu)) ≈ get_coeff_data(cpu)
            backward_transform!(gpu)
            @test Array(get_grid_data(gpu)) ≈ input
            for name in ("x", "z")
                derivative(f) = Tarang.evaluate_differentiate(Differentiate(f, coords[name], 1), :g)
                @test Array(get_grid_data(derivative(gpu))) ≈ get_grid_data(derivative(cpu)) atol=2e-9
            end
            products = map(fields) do f
                ev = Tarang.NonlinearEvaluator(f.dist; dealiasing_factor=3/2)
                product = Tarang.evaluate_transform_multiply(f, f, ev; result_layout=:g)
                Array(get_grid_data(product))
            end
            @test products[2] ≈ products[1]
            for f in fields
                preset_scales!(f, (4/3, 3/2))
                ensure_layout!(f, :g)
            end
            scaled = randn(MersenneTwister(75), size(get_grid_data(cpu)))
            for f in fields
                copyto!(get_grid_data(f), scaled)
                forward_transform!(f)
            end
            @test Array(get_coeff_data(gpu)) ≈ get_coeff_data(cpu)
            @test Array(grid_data!(gpu)) ≈ grid_data!(cpu)

            A = ComplexF64[3 1; 1 2]
            b = ComplexF64[1, 4]
            device_A, device_b = ROCArray(A), ROCArray(b)
            solver = Tarang.MatSolvers.solver_instance(:rocm_dense, device_A)
            @test Array(device_A) == A
            result = Tarang.MatSolvers.solve(solver, device_b)
            AMDGPU.synchronize()
            before = AMDGPU.alloc_stats.alloc_bytes
            Tarang.MatSolvers.solve!(result, solver, device_b)
            AMDGPU.synchronize()
            @test AMDGPU.alloc_stats.alloc_bytes == before
            @test Array(result) ≈ A \ b
            @test_throws SingularException Tarang.MatSolvers.solver_instance(:rocm_dense, zeros(2,2))
            matrices = cat(A, 2A; dims=3)
            batched = Tarang.batched_factor!(Tarang.BatchedDenseLU(ROCArray(matrices)))
            rhs = ROCArray(hcat(b,b))
            x = Tarang.batched_solve!(similar(rhs), batched, rhs)
            AMDGPU.synchronize()
            before = AMDGPU.alloc_stats.alloc_bytes
            Tarang.batched_solve!(x, batched, rhs)
            AMDGPU.synchronize()
            @test AMDGPU.alloc_stats.alloc_bytes == before
            @test Array(x) ≈ hcat(A \ b, (2A) \ b)
            copyto!(batched.A, zeros(ComplexF64,2,2,2))
            @test_throws ErrorException Tarang.batched_factor!(batched)
            @test !batched.factored
            include("timestepper_fourier_cases.jl")
            check_fourier_timestepper_cases(arch)
        end
    else
        @testset "ROCm-aware MPI on device arrays" begin
            check_half_pencils(arch)
            MPI.Comm_size(comm) == 4 && check_half_pencils(arch; mesh=(2,2))
        end
    end
elseif "--require-device" in ARGS
    error("The AMD hardware test requires a functional ROCm GPU")
else
    @info "AMD hardware tests skipped because no ROCm device is available"
end
