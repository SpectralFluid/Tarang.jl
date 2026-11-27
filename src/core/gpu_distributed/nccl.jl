# ============================================================================
# NCCL Support for GPU Collectives
# ============================================================================

"""
    NCCLConfig

Configuration for NCCL GPU collective operations.
NCCL provides optimized GPU-to-GPU communication that can be faster than MPI.
"""
mutable struct NCCLConfig
    initialized::Bool
    comm_handle::Any  # NCCL communicator (when NCCL.jl is loaded)
    rank::Int
    size::Int
    mpi_comm::Union{Nothing, MPI.Comm}  # communicator init_nccl! mirrored (teardown barrier)

    NCCLConfig() = new(false, nothing, 0, 1, nothing)
end

const NCCL_CONFIG = NCCLConfig()

# ----------------------------------------------------------------------------
# Runtime NCCL.jl module resolution
#
# Tarang does NOT depend on NCCL.jl, so `using NCCL` (or any static `NCCL.foo`
# reference) from package code can never work. The user must load NCCL.jl in
# their session (`using NCCL`) before multi-GPU transposes; we then locate the
# already-loaded root module in `Base.loaded_modules` and route every NCCL call
# through the resolved module object. The result is cached in a `Ref{Any}`.
# ----------------------------------------------------------------------------
const _NCCL_MODULE = Ref{Any}(nothing)

"""
    _find_nccl_module() -> Union{Module, Nothing}

Scan `Base.loaded_modules` for a root module named "NCCL" and cache it.
Returns `nothing` when NCCL.jl has not been loaded.
"""
function _find_nccl_module()
    _NCCL_MODULE[] === nothing || return _NCCL_MODULE[]
    for (pkgid, mod) in Base.loaded_modules
        if pkgid.name == "NCCL" && mod isa Module
            _NCCL_MODULE[] = mod
            return mod
        end
    end
    return nothing
end

"""
    _require_nccl_module() -> Module

Like [`_find_nccl_module`](@ref) but throws a descriptive error when NCCL.jl
is not loaded.
"""
function _require_nccl_module()
    mod = _find_nccl_module()
    if mod === nothing
        error("NCCL.jl is not loaded. Load NCCL.jl with `using NCCL` before multi-GPU transposes.")
    end
    return mod
end

"""
    NCCLSubComms

NCCL sub-communicators for pencil decomposition transposes.
These mirror the MPI row_comm and col_comm from PencilDecomposition.

In a 2D pencil decomposition with process grid (P1, P2):
- row_comm: Communicator for ranks in the same row (used for Y↔Z transpose)
- col_comm: Communicator for ranks in the same column (used for X↔Y transpose)

The NCCL sub-communicators enable GPU-to-GPU collective operations
within each row/column group, avoiding CPU staging for transposes.
"""
mutable struct NCCLSubComms
    initialized::Bool
    row_comm::Any  # NCCL communicator for row (Y↔Z transpose)
    col_comm::Any  # NCCL communicator for column (X↔Y transpose)
    row_rank::Int
    row_size::Int
    col_rank::Int
    col_size::Int

    NCCLSubComms() = new(false, nothing, nothing, 0, 1, 0, 1)
end

"""
    init_nccl_subcomms!(row_mpi_comm::MPI.Comm, col_mpi_comm::MPI.Comm)

Initialize NCCL sub-communicators for pencil transpose operations.
Creates NCCL communicators that match the MPI row and column communicators
from PencilDecomposition.

# Arguments
- `row_mpi_comm`: MPI communicator for ranks in the same row
- `col_mpi_comm`: MPI communicator for ranks in the same column

# Returns
NCCLSubComms with initialized=true if NCCL is available and setup succeeds,
otherwise returns uninitialized NCCLSubComms (graceful fallback).

# Example
```julia
# After creating PencilDecomposition
pencil = PencilDecomposition(global_shape, proc_grid, rank, comm)
nccl_subcomms = init_nccl_subcomms!(pencil.row_comm, pencil.col_comm)

if nccl_subcomms.initialized
    # Use NCCL for GPU-to-GPU transposes
else
    # Fall back to MPI-based transposes
end
```
"""
function init_nccl_subcomms!(row_mpi_comm::MPI.Comm, col_mpi_comm::MPI.Comm)
    subcomms = NCCLSubComms()

    # A multi-rank communicator without NCCL cannot be silently ignored: the
    # transposes would degrade into local self-copies and produce wrong answers.
    multi_rank = MPI.Comm_size(row_mpi_comm) > 1 || MPI.Comm_size(col_mpi_comm) > 1

    if !has_cuda()
        if multi_rank
            @warn "NCCL sub-comm initialization skipped - CUDA not available; multi-GPU transposes will fail loudly rather than fall back"
        else
            @debug "NCCL sub-comm initialization skipped - CUDA not available"
        end
        return subcomms
    end

    # Check if NCCL is available (requires the user to have loaded NCCL.jl)
    if !nccl_available() && !_try_load_nccl()
        if multi_rank
            @warn "NCCL sub-comm initialization skipped - NCCL.jl is not loaded. " *
                  "Load NCCL.jl with `using NCCL` before multi-GPU transposes."
        else
            @debug "NCCL sub-comm initialization skipped - NCCL not available"
        end
        return subcomms
    end

    try
        # Get ranks and sizes from MPI communicators
        row_rank = MPI.Comm_rank(row_mpi_comm)
        row_size = MPI.Comm_size(row_mpi_comm)
        col_rank = MPI.Comm_rank(col_mpi_comm)
        col_size = MPI.Comm_size(col_mpi_comm)

        # Initialize row NCCL comm (only if more than one rank)
        if row_size > 1
            row_nccl_comm = create_nccl_comm_from_mpi(row_mpi_comm)
        else
            row_nccl_comm = nothing  # Single rank, no communication needed
        end

        # Initialize col NCCL comm (only if more than one rank)
        if col_size > 1
            col_nccl_comm = create_nccl_comm_from_mpi(col_mpi_comm)
        else
            col_nccl_comm = nothing  # Single rank, no communication needed
        end

        subcomms.initialized = true
        subcomms.row_comm = row_nccl_comm
        subcomms.col_comm = col_nccl_comm
        subcomms.row_rank = row_rank
        subcomms.row_size = row_size
        subcomms.col_rank = col_rank
        subcomms.col_size = col_size

        @debug "NCCL sub-communicators initialized" row_rank=row_rank row_size=row_size col_rank=col_rank col_size=col_size

    catch e
        @warn "NCCL sub-comm initialization failed" exception=(e, catch_backtrace())
        subcomms.initialized = false
    end

    return subcomms
end

"""
    create_nccl_comm_from_mpi(mpi_comm::MPI.Comm)

Create an NCCL communicator that matches an MPI communicator.

Rank 0 in the MPI communicator generates a unique NCCL ID and broadcasts
it to all other ranks. Each rank then creates its NCCL communicator
using this shared unique ID.

# Arguments
- `mpi_comm`: MPI communicator to mirror

# Returns
NCCL.Communicator matching the MPI communicator topology

# Notes
- Requires NCCL.jl to be loaded
- All ranks in mpi_comm must call this function collectively
- The NCCL UniqueId is 128 bytes
"""
function create_nccl_comm_from_mpi(mpi_comm::MPI.Comm)
    rank = MPI.Comm_rank(mpi_comm)
    nprocs = MPI.Comm_size(mpi_comm)

    # Resolve the user-loaded NCCL.jl module (Tarang has no NCCL dependency)
    nccl = _require_nccl_module()
    uid_type = nccl.UniqueID  # NCCL.UniqueID === LibNCCL.ncclUniqueId (128-byte bits type)

    # Generate unique ID on rank 0 and broadcast to all ranks
    if rank == 0
        unique_id = nccl.UniqueID()
        id_bytes = Vector{UInt8}(reinterpret(UInt8, [unique_id]))
    else
        id_bytes = Vector{UInt8}(undef, sizeof(uid_type))
    end

    MPI.Bcast!(id_bytes, mpi_comm; root=0)

    if rank != 0
        unique_id = reinterpret(uid_type, id_bytes)[1]
    end

    # Create NCCL communicator (real API: Communicator(nranks, rank; unique_id))
    nccl_comm = nccl.Communicator(nprocs, rank; unique_id=unique_id)

    return nccl_comm
end

"""
    _try_load_nccl()

Check whether NCCL.jl is already loaded in the session (Tarang cannot load it
itself since NCCL.jl is not a dependency). Returns true if the NCCL module was
found, false otherwise.
"""
_try_load_nccl() = _find_nccl_module() !== nothing

# ----------------------------------------------------------------------------
# Explicit, collective NCCL teardown
#
# `ncclCommDestroy` is an intra-node collective, yet NCCL.jl only registers it
# as a GC finalizer (`finalizer(destroy, comm)`). Merely dropping references let
# every rank destroy whenever its own GC ran: one rank could block in
# ncclCommDestroy while its peer waited in an MPI collective. Teardown is
# therefore explicit, at a point all ranks reach together: finish this rank's
# device work, barrier on the parent MPI communicator, then destroy in a fixed,
# rank-invariant order — row, then column (finalize_nccl_subcomms!), the world
# communicator last (finalize_nccl!). NCCL.jl's `destroy` (≥ 0.1.2) nulls the
# handle, so the finalizer that runs later is a no-op.
# ----------------------------------------------------------------------------

"""Wait for this rank's queued work on `comm`'s device (`NCCL.device(comm)`)."""
function _synchronize_nccl_device(nccl::Module, comm)
    dev = nccl.device(comm)
    synchronize(GPU{typeof(dev)}(dev))
    return nothing
end

_nccl_teardown_barrier(::Nothing) = nothing
function _nccl_teardown_barrier(comm::MPI.Comm)
    MPI.Initialized() && !MPI.Finalized() && MPI.Barrier(comm)
    return nothing
end

"""
    _destroy_nccl_comms!(comms; barrier_comm=nothing)

Collectively destroy the NCCL communicators in `comms` (skipping `nothing`), in
exactly that order on every rank: synchronize their devices, barrier on
`barrier_comm`, then `NCCL.destroy` each one. A failure is rethrown only after
every later destroy has been posted, so peers waiting on them are not stranded.
"""
function _destroy_nccl_comms!(comms; barrier_comm=nothing)
    live = Any[comm for comm in comms if comm !== nothing]
    if isempty(live)
        _nccl_teardown_barrier(barrier_comm)
        return nothing
    end
    nccl = _require_nccl_module()
    for comm in live
        _synchronize_nccl_device(nccl, comm)
    end
    _nccl_teardown_barrier(barrier_comm)
    first_error = nothing
    for comm in live
        try
            nccl.destroy(comm)
        catch err
            first_error === nothing && (first_error = err)
        end
    end
    first_error === nothing || throw(first_error)
    return nothing
end

"""
    finalize_nccl_subcomms!(subcomms::NCCLSubComms; barrier_comm=nothing)

Destroy the NCCL sub-communicators explicitly and reset `subcomms` to its
uninitialized state. Collective: every rank of the sub-communicators must call
it at the same point. Their devices are synchronized, `barrier_comm` (the parent
MPI communicator, when given) is barriered, and the row communicator is
destroyed before the column one on every rank.
"""
function finalize_nccl_subcomms!(subcomms::NCCLSubComms; barrier_comm=nothing)
    subcomms.initialized || return nothing
    comms = (subcomms.row_comm, subcomms.col_comm)   # rank-invariant order: row, column
    subcomms.row_comm = nothing
    subcomms.col_comm = nothing
    subcomms.initialized = false
    _destroy_nccl_comms!(comms; barrier_comm)
    @debug "NCCL sub-communicators destroyed"
    return nothing
end

"""
    nccl_available()

Check if NCCL is available for GPU collectives.
Returns true if NCCL.jl is loaded and functional.
"""
function nccl_available()
    # NCCL support requires NCCL.jl package
    # This is a placeholder - actual check depends on package being loaded
    return NCCL_CONFIG.initialized
end

"""
    init_nccl!(comm::MPI.Comm)

Initialize NCCL for GPU collective operations.

NCCL must be initialized after MPI and after GPU devices are selected.
Each MPI rank should call this on their assigned GPU. Release it collectively
with [`finalize_nccl!`](@ref).

# Example
```julia
MPI.Init()
# Pick the GPU from the NODE-local rank: the global rank is not a device ordinal
# on several nodes or with one visible GPU per rank.
node = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(MPI.COMM_WORLD))
arch = GPU(device_id=MPI.Comm_rank(node) % CUDA.ndevices())
init_nccl!(MPI.COMM_WORLD)
```
"""
function init_nccl!(comm::MPI.Comm)
    if !has_cuda()
        @warn "NCCL initialization skipped - CUDA not available"
        return false
    end

    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    try
        # NCCL.jl must already be loaded by the user (`using NCCL`)
        nccl = _require_nccl_module()
        uid_type = nccl.UniqueID

        # Create NCCL unique ID on rank 0 and broadcast
        if rank == 0
            unique_id = nccl.UniqueID()
            id_bytes = Vector{UInt8}(reinterpret(UInt8, [unique_id]))
        else
            id_bytes = Vector{UInt8}(undef, sizeof(uid_type))
        end

        MPI.Bcast!(id_bytes, comm; root=0)

        if rank != 0
            unique_id = reinterpret(uid_type, id_bytes)[1]
        end

        # Initialize NCCL communicator (real API: Communicator(nranks, rank; unique_id))
        nccl_comm = nccl.Communicator(size, rank; unique_id=unique_id)

        NCCL_CONFIG.initialized = true
        NCCL_CONFIG.comm_handle = nccl_comm
        NCCL_CONFIG.rank = rank
        NCCL_CONFIG.size = size
        NCCL_CONFIG.mpi_comm = comm

        @info "NCCL initialized" rank=rank size=size
        return true

    catch e
        @warn "NCCL initialization failed" exception=(e, catch_backtrace())
        NCCL_CONFIG.initialized = false
        return false
    end
end

"""
    nccl_allreduce!(data, op::Symbol=:sum)

Perform NCCL all-reduce on GPU data.

# Arguments
- `data`: GPU array to reduce (in-place)
- `op`: Reduction operation (:sum, :prod, :max, :min)
"""
function nccl_allreduce!(data, op::Symbol=:sum)
    if !NCCL_CONFIG.initialized
        error("NCCL not initialized. Call init_nccl! first.")
    end

    # Map operation symbol to a reduction function; NCCL.jl converts it via
    # ncclRedOp_t(::typeof(+)) etc. internally.
    nccl_op = _get_nccl_op(op)

    # Perform all-reduce (real API: Allreduce!(sendrecvbuf, op, comm))
    nccl = _require_nccl_module()
    nccl.Allreduce!(data, nccl_op, NCCL_CONFIG.comm_handle)

    return data
end

"""
    nccl_broadcast!(data, root::Int=0)

Broadcast GPU data from root rank to all ranks.
"""
function nccl_broadcast!(data, root::Int=0)
    if !NCCL_CONFIG.initialized
        error("NCCL not initialized. Call init_nccl! first.")
    end

    # Real API: Broadcast!(sendrecvbuf, comm; root)
    nccl = _require_nccl_module()
    nccl.Broadcast!(data, NCCL_CONFIG.comm_handle; root=root)
    return data
end

"""
    nccl_allgather!(recv_data, send_data)

Gather GPU data from all ranks.
"""
function nccl_allgather!(recv_data, send_data)
    if !NCCL_CONFIG.initialized
        error("NCCL not initialized. Call init_nccl! first.")
    end

    # Real API: Allgather!(sendbuf, recvbuf, comm)
    nccl = _require_nccl_module()
    nccl.Allgather!(send_data, recv_data, NCCL_CONFIG.comm_handle)
    return recv_data
end

"""
    _get_nccl_op(op::Symbol)

Convert operation symbol to a reduction function accepted by NCCL.jl's
collectives (which map `+`, `*`, `max`, `min` to ncclSum/ncclProd/ncclMax/ncclMin
via `ncclRedOp_t`). Avoids any static NCCL binding reference.
"""
function _get_nccl_op(op::Symbol)
    if op == :sum
        return +
    elseif op == :prod
        return *
    elseif op == :max
        return max
    elseif op == :min
        return min
    else
        error("Unsupported NCCL operation: $op")
    end
end

"""
    finalize_nccl!()

Destroy the communicator created by [`init_nccl!`](@ref). Collective over the MPI
communicator passed to `init_nccl!`: call it on every rank at the same point,
after closing the Distributors and plans whose NCCL sub-communicators it
outlives (see `finalize_nccl_subcomms!`). The device is synchronized and the MPI
communicator barriered before `NCCL.destroy` runs.
"""
function finalize_nccl!()
    NCCL_CONFIG.initialized || return nothing
    comm, mpi_comm = NCCL_CONFIG.comm_handle, NCCL_CONFIG.mpi_comm
    NCCL_CONFIG.comm_handle = nothing
    NCCL_CONFIG.mpi_comm = nothing
    NCCL_CONFIG.initialized = false
    _destroy_nccl_comms!((comm,); barrier_comm=mpi_comm)
    @info "NCCL finalized"
    return nothing
end

