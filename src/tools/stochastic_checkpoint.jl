# Host-only stochastic restart payload. RNG serialization is scoped to a Julia
# major/minor version; device buffers and architecture objects never go on disk.
using Serialization

const _STOCHASTIC_CHECKPOINT_VAR = "__tarang_stochastic_state"
const _STOCHASTIC_CHECKPOINT_VERSION = 1
_stochastic_checkpoint_julia() = "$(VERSION.major).$(VERSION.minor)"

function _checkpoint_forcings(solver)
    result = Dict{String,StochasticForcingType}()
    for (index, forcing) in solver.problem.stochastic_forcings
        forcing isa StochasticForcingType || continue
        name = solver.state[index].name
        haskey(result, name) && error("Stochastic checkpoint requires unique field names: '$name'")
        result[name] = forcing
    end
    return result
end

function _forcing_checkpoint_config(forcing::Union{StochasticForcing,SeparableStochasticForcing})
    profile = forcing isa SeparableStochasticForcing ?
        (bounds=forcing.chebyshev_basis.meta.bounds, values=Array(forcing.chebyshev_profile)) : nothing
    return (kind=string(nameof(typeof(forcing))), field_size=forcing.field_size,
            dtype=string(eltype(forcing.forcing_spectrum)), domain_size=forcing.domain_size,
            energy_injection_rate=forcing.energy_injection_rate,
            injection_metric=forcing.injection_metric, k_forcing=forcing.k_forcing,
            dk_forcing=forcing.dk_forcing, spectrum_type=forcing.spectrum_type,
            enforce_hermitian=forcing.enforce_hermitian,
            spectrum=Array(forcing.forcing_spectrum), profile=profile)
end

function _stochastic_checkpoint_payload(solver)
    forcings = _checkpoint_forcings(solver)
    isempty(forcings) && return nothing
    any(field -> field.name == _STOCHASTIC_CHECKPOINT_VAR, solver.state) &&
        error("Field name '$_STOCHASTIC_CHECKPOINT_VAR' is reserved for stochastic checkpoints")
    snapshots = map(sort!(collect(keys(forcings)))) do name
        forcing = forcings[name]
        # Constructors and registration always derive a private MersenneTwister.
        forcing.rng isa Random.MersenneTwister || error(
            "Stochastic checkpoints require the forcing's private MersenneTwister RNG")
        (name=name, config=_forcing_checkpoint_config(forcing), rng=copy(forcing.rng),
         cached_forcing=Array(forcing.cached_forcing), dt=forcing.dt,
         last_update_time=forcing.last_update_time)
    end
    io = IOBuffer()
    serialize(io, snapshots)
    return collect(reinterpret(Int8, take!(io)))
end

function _write_stochastic_checkpoint!(path, payload)
    payload === nothing && return
    nccreate(path, _STOCHASTIC_CHECKPOINT_VAR, "__tarang_stochastic_bytes", length(payload);
             t=NetCDF.NC_BYTE)
    ncwrite(payload, path, _STOCHASTIC_CHECKPOINT_VAR)
    ncputatt(path, "global", Dict(
        "stochastic_forcing_version" => _STOCHASTIC_CHECKPOINT_VERSION,
        "stochastic_forcing_julia" => _stochastic_checkpoint_julia()))
    return nothing
end

# Run before restoring any field, and inside the collective metadata preflight.
# Realizations and RNGs are replicated on every rank, so a byte-identical payload
# in every slab both detects mixed checkpoints and permits changing rank counts.
function _prepare_stochastic_restart(solver, src::SlabSource)
    forcings = _checkpoint_forcings(solver)
    reference = nothing
    presence = nothing
    for file in src.files
        attrs = netcdf_file_info(file).gatts
        present = haskey(attrs, "stochastic_forcing_version")
        presence === nothing || present == presence || error(
            "load_state!: stochastic forcing metadata is missing from some checkpoint slabs")
        presence = present
        present || continue
        attrs["stochastic_forcing_version"] == _STOCHASTIC_CHECKPOINT_VERSION || error(
            "load_state!: unsupported stochastic forcing checkpoint version in '$file'")
        get(attrs, "stochastic_forcing_julia", nothing) == _stochastic_checkpoint_julia() || error(
            "load_state!: stochastic RNG restart requires the same Julia major/minor version as the writer")
        payload = ncread(file, _STOCHASTIC_CHECKPOINT_VAR)
        payload isa Vector{Int8} || error("load_state!: invalid stochastic forcing payload in '$file'")
        if reference === nothing
            reference = payload
        else
            payload == reference || error("load_state!: inconsistent stochastic forcing state across checkpoint slabs")
        end
    end
    if reference === nothing
        isempty(forcings) || error(
            "load_state!: checkpoint has no stochastic forcing state; its random sequence cannot be resumed")
        return []
    end
    snapshots = deserialize(IOBuffer(collect(reinterpret(UInt8, reference))))
    names = [snapshot.name for snapshot in snapshots]
    length(unique(names)) == length(names) && Set(names) == Set(keys(forcings)) || error(
        "load_state!: registered stochastic forcing fields do not match the checkpoint")
    return map(snapshots) do snapshot
        forcing = forcings[snapshot.name]
        isequal(snapshot.config, _forcing_checkpoint_config(forcing)) || error(
            "load_state!: stochastic forcing configuration differs for field '$(snapshot.name)'")
        snapshot.rng isa Random.MersenneTwister || error("load_state!: unsupported forcing RNG")
        host = snapshot.cached_forcing
        size(host) == size(forcing.cached_forcing) &&
            eltype(host) == eltype(forcing.cached_forcing) || error("load_state!: invalid cached forcing shape or dtype")
        isfinite(snapshot.dt) && snapshot.dt > 0 || error("load_state!: invalid forcing timestep")
        (isfinite(snapshot.last_update_time) || snapshot.last_update_time == -Inf) ||
            error("load_state!: invalid forcing timestamp")
        # Stage on the destination device before mutating any simulation state.
        cached = on_architecture(forcing.architecture, host)
        (forcing=forcing, rng=snapshot.rng, cached=cached,
         dt=snapshot.dt, last_update_time=snapshot.last_update_time)
    end
end

function _restore_stochastic_restart!(prepared)
    for entry in prepared
        forcing = entry.forcing
        forcing.rng = entry.rng
        # Keep views retained by output tasks attached to the live realization.
        copyto!(forcing.cached_forcing, entry.cached)
        forcing.dt = entry.dt
        forcing.last_update_time = entry.last_update_time
        # Previous-solution work diagnostics are rank-local scratch, not restart
        # state. Clear stale history; store_prevsol! initializes the next sample.
        forcing.prevsol = nothing
    end
    return nothing
end
