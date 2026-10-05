using Test
using Tarang
using LinearAlgebra
using Random

@noinline function _idle_pool_refs(dist, prewarm)
    pool = FieldPool(dist)
    if prewarm
        prewarm!(pool, (), Float64, 1)
        field = first(first(values(pool.available)))
    else
        field = checkout!(pool, (), Float64)
        return!(pool, field)
    end
    return WeakRef(pool), WeakRef(field)
end
@noinline function _active_pool_refs(dist)
    pool = FieldPool(dist)
    return WeakRef(pool), checkout!(pool, (), Float64)
end
@noinline _pool_ref_alive(ref) = ref.value !== nothing
_collect_pool_refs() = foreach(_ -> GC.gc(true), 1:3)

function _matrix_axis_reference(data, D, axis)
    perm = [axis; [i for i in 1:ndims(data) if i != axis]]
    input = permutedims(data, perm)
    return permutedims(reshape(D * reshape(input, size(input, 1), :), size(input)), invperm(perm))
end

@testset "Memory ownership and derivative workspaces" begin
    @testset "Idle pools are collectible; live checkouts retain their owner" begin
        domain = PeriodicDomain(8)
        for prewarm in (false, true)
            pool_ref, field_ref = _idle_pool_refs(domain.dist, prewarm)
            _collect_pool_refs()
            @test !_pool_ref_alive(pool_ref)
            @test !_pool_ref_alive(field_ref)
        end
        pool_ref, field = _active_pool_refs(domain.dist)
        _collect_pool_refs()
        @test _pool_ref_alive(pool_ref)
        maybe_return!(field)
        _collect_pool_refs()
        @test !_pool_ref_alive(pool_ref)
        @test field._from_pool # retaining a returned handle must not retain the pool
    end

    @testset "Copies own only their retained arrays" begin
        domain = PeriodicDomain(128, 128)
        field = ScalarField(domain, "u")
        set!(field, (x, y) -> sin(x) * cos(y))
        for layout in (:g, :c)
            ensure_layout!(field, layout)
            copy(field) # warm dispatch
            bytes = @allocated copy(field)
            retained = sizeof(Tarang.get_grid_data(field)) + sizeof(Tarang.get_coeff_data(field))
            # Allow allocator size-class rounding, but not a discarded grid or
            # coefficient array (each exceeds 128 KiB here).
            @test bytes <= retained + 32768
            for copier in (copy, deepcopy)
                owned = copier(field)
                @test owned.storage !== field.storage
                @test owned.current_layout === layout
                @test owned[String(layout)] ≈ field[String(layout)]
                @test Tarang.get_grid_data(owned) !== Tarang.get_grid_data(field)
                @test Tarang.get_coeff_data(owned) !== Tarang.get_coeff_data(field)
                @test !owned._from_pool
                ensure_layout!(owned, :g)
                ensure_layout!(field, :g)
                @test owned["g"] ≈ field["g"]
                ensure_layout!(field, layout)
            end
        end
        # Empty-basis live scalar buffers must survive the copy constructor.
        scalar = ScalarField(domain.dist, "constant", ())
        for layout in (:g, :c), copier in (copy, deepcopy)
            if layout === :g
                Tarang.set_grid_data!(scalar, [3.0])
            else
                Tarang.set_coeff_data!(scalar, [3.0 + 2.0im])
            end
            scalar.current_layout = layout
            owned = copier(scalar)
            @test owned[String(layout)] == scalar[String(layout)]
            @test owned[String(layout)] !== scalar[String(layout)]
        end
        # Metadata can refer back to the field; deepcopy must preserve the cycle.
        field.bases[1].transforms[:copy_cycle_test] = field
        try
            owned = deepcopy(field)
            @test owned.bases[1].transforms[:copy_cycle_test] === owned
        finally
            delete!(field.bases[1].transforms, :copy_cycle_test)
        end

        coords = CartesianCoordinates("z")
        dist = Distributor(coords)
        for basis in (RealFourier(coords["z"]; size=12), ChebyshevT(coords["z"]; size=12))
            scaled = ScalarField(dist, "scaled", (basis,))
            fill!(grid_data!(scaled), 2.5)
            set_scales!(scaled, 1.5)
            for layout in (:g, :c), copier in (copy, deepcopy)
                ensure_layout!(scaled, layout)
                owned = copier(scaled)
                @test owned.scales == (1.5,)
                @test owned.current_layout === layout
                @test owned[String(layout)] ≈ scaled[String(layout)]
                ensure_layout!(owned, :c)
                @test size(grid_data!(owned)) == (18,)
                @test all(x -> isapprox(x, 2.5; atol=1e-12), get_grid_data(owned))
            end
        end
        close(dist)
    end

    @testset "Matrix derivatives reuse storage on every axis" begin
        rng = MersenneTwister(412)
        basis = ChebyshevT(CartesianCoordinates("z")["z"]; size=8, bounds=(-1.0, 1.0))
        for T in (Float32, ComplexF64), shape in ((8,), (8, 8), (8, 8, 8), (4, 5, 6, 7))
            original = rand(rng, T, shape)
            for axis in 1:length(shape)
                D = rand(rng, T, shape[axis], shape[axis])
                expected = _matrix_axis_reference(original, D, axis)
                data = copy(original)
                @test Tarang._apply_1d_matrix!(data, D, axis, basis) === data
                @test data ≈ expected
                copyto!(data, original)
                @test (@allocated Tarang._apply_1d_matrix!(data, D, axis, basis)) < 4096
                @test data ≈ expected
            end
        end
        # Sparse polynomial matrix, complex coefficients, substantial 3-D volume.
        D = Tarang.spectral_derivative_matrix(basis, 1)
        original = rand(rng, ComplexF64, 16, 8, 32)
        expected = _matrix_axis_reference(original, D, 2)
        data = copy(original)
        Tarang._apply_1d_matrix!(data, D, 2, basis)
        @test data ≈ expected
        @test (@allocated Tarang._apply_1d_matrix!(data, D, 2, basis)) < 4096
        @test_throws ArgumentError Tarang._apply_1d_matrix!(data, D, 0, basis)
        @test_throws DimensionMismatch Tarang._apply_1d_matrix!(data, D, 1, basis)
        # Children can inherit task-local storage; each must get its own scratch.
        tasks = map(1:8) do _
            Threads.@spawn begin
                local_data = similar(original)
                for _ in 1:10
                    copyto!(local_data, original)
                    yield()
                    Tarang._apply_1d_matrix!(local_data, D, 2, basis)
                    isapprox(local_data, expected) || return false
                end
                true
            end
        end
        @test all(fetch, tasks)
    end
end
