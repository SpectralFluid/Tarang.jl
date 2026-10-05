using Test, Tarang, MPI
MPI.Initialized() || MPI.Init()

function _copy_scaling_apply!(data, axis, which, factor)
    if which == :first
        Tarang._scale_first_along_axis!(data, axis, factor)
    elseif which == :last
        Tarang._scale_last_along_axis!(data, axis, factor)
    else
        Tarang._scale_slice_along_axis!(data, axis, 3, factor)
    end
    return nothing
end

function _copy_scaling_repeat!(data)
    for _ in 1:10, axis in 1:3
        Tarang._scale_first_along_axis!(data, axis, one(eltype(data)))
        Tarang._scale_last_along_axis!(data, axis, one(eltype(data)))
        Tarang._scale_slice_along_axis!(data, axis, 3, one(eltype(data)))
    end
    return nothing
end

function _copy_scaling_stash_roundtrip!(buffer, sp, field, data, n)
    for _ in 1:10
        Tarang._stash_missing_field!(sp, field, data, 2, n)
        Tarang._restore_missing_field!(buffer, 3, n, sp, field)
    end
    return nothing
end

function _copy_scaling_tau_fixture()
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; comm=MPI.COMM_SELF, mesh=(1,), dtype=Float64)
    basis = ChebyshevT(coords["z"]; size=8, bounds=(0.0,1.0))
    u = ScalarField(dist, "u", (basis,), Float64)
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    problem = InitialValueProblem([u,tau1,tau2])
    add_parameters!(problem; lb=derivative_basis(basis,2))
    add_equation!(problem,"dt(u)-lap(u)+lift(tau1,lb,-1)+lift(tau2,lb,-2)=0")
    add_bc!(problem,"u(z=0)=0")
    add_bc!(problem,"u(z=1)=0")
    solver = InitialValueSolver(problem,RK222();dt=0.001)
    sp = only(Tarang._timestepper_subproblems(solver))
    # Independent runtime state, sharing the same physical tau field.
    other = Tarang.Subproblem(solver, (), (1,))
    return sp, other, tau1, dist
end

@testset "3D Chebyshev slice scaling" begin
    shape = (7,9,11)
    for T in (Float32,Float64,ComplexF32,ComplexF64), axis in 1:3,
        which in (:first,:last,:selected)
        original = reshape(T.(1:prod(shape)),shape)
        factor = T <: Complex ? T(0.5+0.25im) : T(0.5)
        expected = copy(original)
        index = which == :first ? 1 : which == :last ? shape[axis] : 3
        for I in CartesianIndices(expected)
            I[axis] == index && (expected[I] *= factor)
        end
        actual = copy(original)
        _copy_scaling_apply!(actual,axis,which,factor)
        @test actual == expected
        @test eltype(actual) == T
    end
    # Increasing the sliced plane's area must not allocate copied planes.
    small, large = ones(4,6,8), ones(32,40,48)
    _copy_scaling_repeat!(small); _copy_scaling_repeat!(large)
    small_bytes = @allocated _copy_scaling_repeat!(small)
    large_bytes = @allocated _copy_scaling_repeat!(large)
    @test large_bytes <= small_bytes + 1024
end

@testset "Tau stash copies preserve values without element boxing" begin
    sp, other, tau, dist = _copy_scaling_tau_fixture()
    source = ComplexF32.(1:13) .+ ComplexF32(0.25im)
    output = fill(ComplexF64(-9im),15)
    @test !Tarang._restore_missing_field!(output,3,8,sp,tau)
    Tarang._stash_missing_field!(sp,tau,source,2,8)
    original_stash = sp.runtime.zero_dim_stash[objectid(tau)]
    Tarang._stash_missing_field!(other,tau,2 .* source,2,8)
    @test Tarang._restore_missing_field!(output,3,8,sp,tau)
    @test output[4:11] == source[3:10]
    @test all(==(-9im),output[[1,2,3,12,13,14,15]])
    @test !Tarang._restore_missing_field!(output,3,7,sp,tau)
    @test Tarang._restore_missing_field!(output,3,8,other,tau)
    @test output[4:11] == 2 .* source[3:10]
    source .+= 5
    Tarang._stash_missing_field!(sp,tau,view(source,:),2,8)
    @test sp.runtime.zero_dim_stash[objectid(tau)] === original_stash
    @test Tarang._restore_missing_field!(output,3,8,sp,tau)
    @test output[4:11] == source[3:10]

    allocations = Int[]
    for n in (1,1024)
        data = fill(ComplexF64(1+2im),n+4)
        buffer = zeros(ComplexF64,n+6)
        _copy_scaling_stash_roundtrip!(buffer,sp,tau,data,n)
        push!(allocations,@allocated _copy_scaling_stash_roundtrip!(buffer,sp,tau,data,n))
        @test buffer[4:n+3] == data[3:n+2]
    end
    @test allocations[2] <= allocations[1] + 1024
    close(dist)
end

@testset "Kernel-backed slice and tau copies" begin
    jlarrays_available = try
        @eval using JLArrays, GPUArrays
        true
    catch
        false
    end
    if !jlarrays_available
        @test_skip "JLArrays unavailable"
    else
        @eval Tarang.is_gpu_array(::JLArrays.JLArray) = true
        GPUArrays.allowscalar(false)
        for T in (Float32,ComplexF32), axis in 1:3, which in (:first,:last,:selected)
            original = reshape(T.(1:315),5,7,9)
            expected = copy(original)
            factor = T <: Complex ? T(0.5+0.25im) : T(0.5)
            index = which == :first ? 1 : which == :last ? size(original,axis) : 3
            for I in CartesianIndices(expected)
                I[axis] == index && (expected[I] *= factor)
            end
            actual = JLArrays.JLArray(original)
            _copy_scaling_apply!(actual,axis,which,factor)
            @test Array(actual) == expected
        end
        sp, _, tau, dist = _copy_scaling_tau_fixture()
        source = JLArrays.JLArray(ComplexF32.(1:12) .+ ComplexF32(0.5im))
        dest = JLArrays.JLArray(fill(ComplexF64(-9im),16))
        Tarang._stash_missing_field!(sp,tau,source,2,8)
        @test sp.runtime.zero_dim_stash[objectid(tau)] isa JLArrays.JLArray
        @test Tarang._restore_missing_field!(dest,3,8,sp,tau)
        @test Array(dest)[4:11] == Array(source)[3:10]
        @test all(==(-9im),Array(dest)[[1,2,3,12,13,14,15,16]])
        @test !Tarang._restore_missing_field!(zeros(ComplexF64,16),3,8,sp,tau)
        source .+= 3
        Tarang._stash_missing_field!(sp,tau,source,2,8)
        @test Tarang._restore_missing_field!(dest,3,8,sp,tau)
        @test Array(dest)[4:11] == Array(source)[3:10]
        close(dist)
    end
end
