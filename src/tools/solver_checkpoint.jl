# ============================================================================
# Solver checkpoint / restart.
#
# Writes every evolved field in `solver.state` plus the simulation clock, using
# the slab layer, so a checkpoint written on N ranks can be read on M.
#
# Writes `solver.state`, NOT the problem-variable handles: those are separate
# objects and writing to them does not restore the integrator.
# ============================================================================

"""Steps a multistep scheme spends re-seeding after a restart with no stored history.

Enumerated from the timestepper roster in `src/core/timesteppers/types.jl`, not
from the subset that happened to be tested: a scheme missing from this table
restarts with a silently re-seeded history and NO warning, which is exactly the
quiet-wrong-answer failure this repo keeps finding. `_ONE_STEP_SCHEMES` below
records the deliberate exclusions, and `test_checkpoint_restart.jl` asserts that
every concrete `TimeStepper` appears in one table or the other, so a new scheme
cannot be silently omitted.

The count is the number of steps the scheme runs below its nominal order while
it rebuilds the history the checkpoint does not carry.

NOTE ON THE WARNING DEDUP: `_warn_multistep_restart` tags its `@warn` with
`_id=Symbol(:multistep_restart_, scheme)`, and that dedup is PROCESS-global. The
serial suite runs every test file in one process, so a scheme whose warning has
already been asserted once (`test_checkpoint_restart.jl` spends CNAB2/SBDF2/
SBDF3/SBDF4) will emit nothing the second time. A later test re-asserting an
already-spent scheme's warning would observe silence and fail for a reason that
has nothing to do with the code under test — pick an unspent scheme instead."""
const _MULTISTEP_RESEED_STEPS = Dict(
    "CNAB2"              => 1,   # AB2 explicit extrapolation needs F^{n-1}
    "MCNAB2"             => 1,   # theta-method + AB2 explicit: same F^{n-1}
    "CNLF2"              => 1,   # leapfrog: needs X^{n-1}
    "SBDF2"              => 1,
    "SBDF3"              => 2,
    "SBDF4"              => 3,
    "ETD_CNAB2"          => 1,   # AB2 coefficients under an exponential integrator
    "ETD_SBDF2"          => 1,   # BDF2 coefficients under an exponential integrator
    "DiagonalIMEX_SBDF2" => 1,   # its own docstring: "Multi-step method"
)

"""Schemes deliberately absent from `_MULTISTEP_RESEED_STEPS`: they restart exactly.

Two groups. The one-step schemes (every RK variant, including the diagonal-IMEX
and exponential RK ones) carry no history at all. `CNAB1` and `SBDF1` are the
first-order bootstrap formulas of their families: both depend only on the
current state, so there is nothing to re-seed."""
const _ONE_STEP_SCHEMES = Set([
    "RK111", "RK222", "RK443", "RKSMR", "RKGFY",
    "ETD_RK222", "DiagonalIMEX_RK222", "DiagonalIMEX_RK443",
    "CNAB1", "SBDF1",
])

function _warn_multistep_restart(solver::InitialValueSolver)
    scheme = string(nameof(typeof(solver.timestepper)))
    steps = get(_MULTISTEP_RESEED_STEPS, scheme, 0)
    steps == 0 && return nothing
    # `maxlog` dedups by macro-expansion callsite, not by message content, and
    # there is exactly one @warn callsite here serving every scheme. Without a
    # per-scheme `_id`, the first multistep restart in a process would be the
    # only one that ever warns -- a different scheme, or the same scheme on a
    # different solver, would restart silently for the rest of the session.
    @warn "$scheme restart re-seeds its multistep history: the checkpoint carries the " *
          "state but not the stored time levels, so the first $steps step(s) run at the " *
          "seeding order. The run is correct but not bit-identical to an uninterrupted " *
          "one. One-step schemes (RK111/RK222/RK443, RKSMR, RKGFY, ETD_RK222) " *
          "and the first-order bootstraps CNAB1/SBDF1 restart exactly; SBDF2 does NOT, " *
          "including its internal diagonal implementation -- it is multistep." maxlog=1 _id=Symbol(:multistep_restart_, scheme)
    return nothing
end

# Julia 1.10 and 1.11 may fall back to copying after a failed filesystem rename.
# A checkpoint must appear in one operation, or leave the old generation intact.
function _rename_checkpoint_path(src, dst)
    err = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), src, dst)
    Base.uv_error("renaming checkpoint '$src' to '$dst'", err)
    return nothing
end

# All files live on the target filesystem, so publication uses renames. One
# publisher per stem also serializes concurrent writes with different rank counts.
function _publish_checkpoint!(staged, stem, nranks)
    serial = string(stem, ".nc")
    lockdir = string(stem, ".tarang-publish")
    mkdir(lockdir)
    olddir = joinpath(lockdir, "previous")
    oldfile = joinpath(lockdir, "previous.nc")
    carried = String[]
    try
        if isdir(stem)
            prefix = string(basename(stem), "_p")
            for name in readdir(stem)
                numbered = startswith(name, prefix) && endswith(name, ".nc") &&
                    tryparse(Int, chop(name; head=length(prefix), tail=3)) !== nothing
                if !numbered || !isfile(joinpath(stem, name))
                    nranks > 1 || error("save_state: '$stem' contains non-checkpoint data")
                    push!(carried, name)
                end
            end
            _rename_checkpoint_path(stem, olddir)
            for name in carried
                dst = joinpath(staged, name)
                ispath(dst) && error("save_state: existing entry '$name' conflicts with a staged slab")
                _rename_checkpoint_path(joinpath(olddir, name), dst)
            end
        end
        if nranks > 1 && isfile(serial)
            _rename_checkpoint_path(serial, oldfile)
        end
        target = nranks == 1 ? serial : stem
        ispath(target) && !(nranks == 1 && isfile(target)) &&
            error("save_state: cannot replace '$target' with a checkpoint")
        _rename_checkpoint_path(staged, target)
    catch publish_error
        if isdir(olddir)
            for name in carried
                path = joinpath(staged, name)
                previous = joinpath(olddir, name)
                if ispath(path) && !ispath(previous)
                    _rename_checkpoint_path(path, previous)
                end
            end
            _rename_checkpoint_path(olddir, stem)
        end
        isfile(oldfile) && _rename_checkpoint_path(oldfile, serial)
        rm(lockdir)
        rethrow(publish_error)
    end
    rm(lockdir; recursive=true)
    return nothing
end

"""
    save_state(solver, path) -> String

Write `solver.state` and the simulation clock to NetCDF. Serial runs produce
`<path>.nc`; under MPI each rank writes `<path>/<name>_p<rank>.nc` with no gather.

Registered stochastic forcings also store their RNG, cached realization, and
cache clock. These host-only payloads support CPU/GPU and rank-count changes,
but require the same Julia major/minor version for RNG deserialization.
Previous-solution work-diagnostic scratch is not checkpointed.

Zero-dimensional tau variables are skipped: they carry no spatial data and are
re-solved from the state on the next step.

Every rank writes into a temporary generation on the target filesystem. Rank
zero publishes the generation after all writes and clock attributes succeed.
"""
function save_state(solver::InitialValueSolver, path::AbstractString)
    isempty(solver.state) && error("save_state: solver.state is empty; nothing to write.")
    dist = solver.state[1].dist
    forcing_payload = _collectively(dist, "save_state stochastic forcing") do
        _stochastic_checkpoint_payload(solver)
    end
    stem = endswith(path, ".nc") ? chop(path; tail=3) : String(path)
    serial = dist.size == 1
    filename = serial ? basename(stem) * ".nc" : string(basename(stem), "_p", dist.rank, ".nc")

    # Transpose EVERY field to grid space before the write loop, not inside it.
    # `ensure_layout!` is collective under MPI, and the guard below wraps the whole
    # loop rather than each iteration: with the transposes interleaved, a per-rank
    # write failure at field k (a full filesystem on one node) would unwind that
    # rank to the guard's Allreduce while its peers entered field k+1's transpose —
    # a hang, not an error. Hoisting leaves the loop body free of collectives, so
    # whole-loop granularity is sound. Every rank runs the same transposes in the
    # same order because `solver.state` is replicated metadata.
    for field in solver.state
        isempty(field.bases) && continue
        ensure_layout!(field, :g)
    end

    stage = _collectively(dist, "create checkpoint staging directory") do
        dist.rank == 0 || return ""
        parent_dir = dirname(abspath(stem))
        mkpath(parent_dir)
        mktempdir(parent_dir)
    end
    serial || (stage = MPI.bcast(stage, 0, dist.comm))
    try
        staged_file = joinpath(stage, filename)
        _collectively(dist, "write checkpoint") do
            for field in solver.state
                isempty(field.bases) && continue
                global_shape, _, local_start = _field_slab_geometry(field)
                host = Array(get_local_data(get_grid_data(field)))
                write_local_slab(staged_file, field.name, host, local_start, global_shape)
            end
            _write_stochastic_checkpoint!(staged_file, forcing_payload)
            ncputatt(staged_file, "global", Dict("sim_time" => Float64(solver.sim_time),
                     "iteration" => Int(solver.iteration), "dt" => Float64(solver.dt)))
        end
        _collectively(dist, "publish checkpoint") do
            dist.rank == 0 && _publish_checkpoint!(serial ? staged_file : stage, stem, dist.size)
        end
    finally
        _collectively(dist, "clean checkpoint staging directory") do
            dist.rank == 0 && isdir(stage) && rm(stage; recursive=true)
        end
    end
    return serial ? string(stem, ".nc") : joinpath(stem, filename)
end

"""Clock attributes every checkpoint must carry."""
const _CHECKPOINT_CLOCK_ATTRS = ("sim_time", "iteration", "dt")

"""Validate completion and a common simulation clock in every checkpoint slab."""
function _checkpoint_clock(src::SlabSource, path::AbstractString)
    reference = nothing
    for file in src.files
        attrs = netcdf_file_info(file).gatts
        missing_attrs = filter(k -> !haskey(attrs, k), _CHECKPOINT_CLOCK_ATTRS)
        isempty(missing_attrs) || error(
            "load_state!: '$path' is missing the required clock attribute(s) " *
            "$(join(missing_attrs, ", ")) in '$file'. The checkpoint is incomplete " *
            "(save_state writes clock attributes last in every slab) or is " *
            "save_field output. No state has been restored.")
        clock = (sim_time=Float64(attrs["sim_time"]),
                 iteration=Int(attrs["iteration"]), dt=Float64(attrs["dt"]))
        if reference === nothing
            reference = clock
        elseif clock != reference
            error("load_state!: inconsistent checkpoint clock attributes in '$path': " *
                  "'$file' has $clock, but '$(first(src.files))' has $reference. " *
                  "The slabs belong to different simulation states; no state has been restored.")
        end
    end
    return reference
end

function _stage_checkpoint_grid(field::ScalarField, src::SlabSource, path::AbstractString)
    host = _read_field_slab(field, src, path, field.name)
    grid = get_grid_data(field)
    distributed = field.dist.size > 1
    if distributed || field.current_layout === :g
        _check_slab_grid_shape(get_local_data(grid), host)
    end
    grid isa Array && return host::typeof(grid)

    # Preserve the PencilArray endpoint or device backend without changing the
    # field's active buffer. Serial fields may have released the inactive grid.
    staged = (distributed ? similar(grid) : similar(grid, size(host)))::typeof(grid)
    target = get_local_data(staged)
    copyto!(target, host)
    synchronize_device!(field.dist.architecture)
    return staged
end

"""
    load_state!(solver, path) -> solver

Restore `solver.state` and the simulation clock from a checkpoint written at any
rank count. Discards the timestepper's cached state so the scheme restarts from
the loaded fields.

Registered stochastic forcing must match the saved configuration. A legacy
checkpoint without forcing state cannot reproduce a stochastic trajectory and
is rejected for a stochastically forced solver before any fields are restored.

Every slab must carry all three clock attributes with identical values. The
clock, forcing state, and every field slab are validated and staged on all ranks
before the live solver state changes.
"""
function load_state!(solver::InitialValueSolver, path::AbstractString)
    isempty(solver.state) && error("load_state!: solver.state is empty; nothing to restore.")
    dist = solver.state[1].dist

    clock, forcing_state, staged, unused = _collectively(dist, "load_state! staging") do
        src = open_slab_source(path)
        clock = _checkpoint_clock(src, path)
        forcing_state = _prepare_stochastic_restart(solver, src)
        staged = [(field, _stage_checkpoint_grid(field, src, path))
                  for field in solver.state if !isempty(field.bases)]
        restored = Set(field.name for (field, _) in staged)
        unused = sort([v for v in keys(src.entries) if !(v in restored)])
        return (clock, forcing_state, staged, unused)
    end

    _collectively(dist, "load_state! stochastic forcing") do
        _restore_stochastic_restart!(forcing_state)
    end
    # Install staged grids through the shared storage so problem variable handles
    # continue to alias solver state. This phase performs no file or device I/O.
    for (field, grid) in staged
        set_grid_data!(field, grid)
        field.current_layout = :g
    end
    solver.sim_time = clock.sim_time
    solver.iteration = clock.iteration
    solver.dt = clock.dt

    # Drop the cached timestepper state: its history refers to the pre-restart
    # trajectory. `_ensure_timestepper_state!` rebuilds it from the loaded fields.
    solver.timestepper_state = nothing
    isempty(unused) || @info "load_state!: '$path' also holds variable(s) " *
        "$(join(unused, ", ")) with no matching field in solver.state; they were " *
        "not loaded."
    _warn_multistep_restart(solver)
    return solver
end
