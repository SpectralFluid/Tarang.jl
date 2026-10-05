# Run at 1, 2, and 4 ranks: numeric/type parity plus an actual MPI collective
# budget, including uneven/empty slabs and independent subcommunicators.
module DiagnosticMomentAudit
using Test, MPI, Tarang
using Tarang: PencilArrays
MPI.Initialized() || MPI.Init()

const COLLECTIVES = Ref(0)
# Delegate to MPI.jl's existing Union method after counting. This more-specific
# test-only method observes the real collective boundary, including accidental
# PencilArray reductions, without replacing the MPI implementation.
function MPI.Allreduce!(buf::MPI.RBuffer, op::MPI.Op, comm::MPI.Comm)
    COLLECTIVES[] += 1
    invoke(MPI.Allreduce!, Tuple{MPI.RBuffer,Union{MPI.Op,MPI.MPI_Op},MPI.Comm}, buf, op, comm)
end

function counted(f, expected)
    COLLECTIVES[] = 0
    result = f()
    @test COLLECTIVES[] == expected
    return result
end

function sample(::Type{T}, rank, n) where {T}
    T <: Complex ? T[complex(2rank + i, rank - i) for i in 1:n] : T[2rank + i for i in 1:n]
end

function audit_comm(comm)
    rank, np = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    members = MPI.Allgather(MPI.Comm_rank(MPI.COMM_WORLD), comm)
    field = (dist=(comm=comm, size=np),)
    reducer = GlobalArrayReducer(comm)
    for T in (Float32, Float64, ComplexF32, ComplexF64, Int64), empty_first in (false, true)
        sizes = [empty_first && r == 0 ? 0 : r + 2 for r in 0:np-1]
        parts = [sample(T, members[r+1], sizes[r+1]) for r in 0:np-1]
        data, all_data = parts[rank+1], vcat(parts...)
        s, s2, n = sum(all_data), sum(abs2, all_data), length(all_data)
        references = (s / n, (s2 - abs2(s) / n) / max(1, n-1), sqrt(s2 / n))
        for (fn, reference) in zip((Tarang._global_mean_val, Tarang._global_var_val, Tarang._global_rms_val), references)
            actual = counted(() -> fn(data, field), np > 1 ? 1 : 0)
            @test typeof(actual) == typeof(reference)
            @test isequal(actual, reference) || isapprox(actual, reference; rtol=5e-6, atol=5e-6)
        end
        actual = counted(() -> global_mean(reducer, data), 1)
        reference = n == 0 ? 0.0 : (T <: Complex ? ComplexF64(s) : Float64(s)) / n
        @test typeof(actual) == typeof(reference)
        @test actual ≈ reference
    end

    # A view must retain its logical subset; parent(view) is not a local slab.
    data = [10000.0, members[rank+1] + 1.0, members[rank+1] + 2.0, -20000.0]
    v = @view data[2:3]
    reference = sum(members) / np + 1.5
    @test counted(() -> global_mean(reducer, v), 1) == reference
    @test counted(() -> Tarang._global_mean_val(v, field), np > 1 ? 1 : 0) == reference

    # Counts must stay exact beyond both Float32 and Float64 integer precision.
    n = Int(2)^53 + rank + 1
    moments = counted(() -> Tarang._allreduce_moments((1f0 + 2f0im, 3f0, n), comm), 1)
    @test moments === (ComplexF32(np, 2np), Float32(3np), np * Int(2)^53 + np*(np+1) ÷ 2)
    @test counted(() -> global_mean(reducer, ComplexF32[]), 1) === 0.0
end

const WORLD = MPI.COMM_WORLD
@testset "Fused diagnostic moments (np=$(MPI.Comm_size(WORLD)))" begin
    audit_comm(WORLD)
    sub = MPI.Comm_split(WORLD, mod(MPI.Comm_rank(WORLD), 2), MPI.Comm_rank(WORLD))
    audit_comm(sub)
    MPI.free(sub)

    # Real PencilArray input verifies that local moments don't trigger an extra
    # implicit reduction before the explicit fused collective.
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=12, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=10, bounds=(0.0, 2π))
    field = ScalarField(dist, "moment_field", (xb, yb), Float64)
    ensure_layout!(field, :g)
    data = get_grid_data(field)
    fill!(parent(data), 3.0)
    reducer = GlobalArrayReducer(dist.comm)
    @test counted(() -> global_mean(reducer, data), 1) == 3.0
    @test counted(() -> Tarang._global_mean_val(data, field), dist.size > 1 ? 1 : 0) == 3.0
    @test counted(() -> Tarang._global_var_val(data, field), dist.size > 1 ? 1 : 0) == 0.0
    @test counted(() -> Tarang._global_rms_val(data, field), dist.size > 1 ? 1 : 0) == 3.0
end
end
