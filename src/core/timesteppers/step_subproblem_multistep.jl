# ── Per-subproblem IMEX multistep step ────────────────────────────────────────
#
# Generic IMEX multistep stepper that operates on per-Fourier-mode subproblems.
#
# Handles any multistep method parameterized by (a, b, c) coefficient tuples:
#
#   Σ_k a[k] M X^{n+1-k} + Σ_k b[k] L X^{n+1-k} = Σ_k c[k] F^{n+1-k}
#
# where k=0 is the unknown new step and k≥1 are stored history values.
# Rearranging to solve form:
#
#   (a[0]*M + b[0]*L) X^{n+1} = Σ_{k≥1} [ c[k]*F_k − a[k]*M*X_k − b[k]*L*X_k ]
#
# For inhomogeneous algebraic constraints (BC rows where M_row = 0), the RHS
# rows are OVERRIDDEN with `b[0] * F_BC`. This mirrors the dt*a_ii override in
# `step_subproblem_rk!` — without it, the accumulation formula gives wrong
# scaling for F_BC (the previous-step L*X cancellation only works if prior
# stages satisfied the constraint, which fails on the first call with a
# history-free state).
#
# **Per-step allocation**: history is stored in fixed-capacity `SPHistoryRing`
# circular buffers, pre-allocated on first use and reused thereafter. The only
# allocations per step are the small cached vectors managed by
# `_subproblem_cached_vector!`, which are also reused across steps. Expected
# per-step allocation for steady-state runs: **zero**.
# ──────────────────────────────────────────────────────────────────────────────

"""
    SPHistoryRing

Fixed-capacity circular buffer for per-subproblem multistep history. Stores
`capacity` pre-allocated `AbstractVector{ComplexF64}` slots and maintains a
`head` index pointing at the newest entry. Allows O(1) replacement
with one spare slot for the current attempt and O(1) age-indexed access without any
per-step allocation after the initial slot allocation.

Used by `step_subproblem_multistep!` to store `M*X`, `L*X`, and `F` history
per subproblem across timesteps. Replaces the earlier `Vector{Vector}` +
`pushfirst!` / `pop!` implementation which allocated three fresh vectors per
subproblem per step.
"""
mutable struct SPHistoryRing
    buffers::Vector{_SubproblemVector}  # Pre-allocated slots (capacity plus one candidate)
    head::Int                                     # 1-based slot of newest entry, or 0 if empty
    count::Int                                    # Number of valid entries, 0 ≤ count ≤ capacity
    capacity::Int
end

SPHistoryRing(capacity::Int) = SPHistoryRing(
    Vector{_SubproblemVector}(undef, 0),
    0, 0, max(capacity, 1),
)

"""Return the k-th newest entry (k=1 is newest), or `nothing` if `k > count`.

Mapping from age `k` to slot: with `head` pointing at the newest slot, age
`k` lives at `mod1(head - k + 1, length(buffers))`.
"""
@inline function ring_get(ring::SPHistoryRing, k::Int)
    (k < 1 || k > ring.count) && return nothing
    slot = mod1(ring.head - k + 1, length(ring.buffers))
    return ring.buffers[slot]
end

"""Return the spare slot for an attempted step. Committed entries stay readable."""
function ring_stage!(ring::SPHistoryRing, reference::AbstractVector)
    if isempty(ring.buffers)
        ring.buffers = [similar_zeros(reference, eltype(reference), length(reference))
                        for _ in 1:(ring.capacity + 1)]
    end
    return _ring_candidate(ring)
end

@inline _ring_candidate(ring::SPHistoryRing) =
    ring.buffers[mod1(ring.head + 1, length(ring.buffers))]

@inline _ring_step_entry(ring::SPHistoryRing, k::Int) =
    k == 1 ? _ring_candidate(ring) : ring_get(ring, k - 1)

function ring_accept!(ring::SPHistoryRing)
    ring.head = mod1(ring.head + 1, length(ring.buffers))
    ring.count = min(ring.count + 1, ring.capacity)
    return nothing
end

"""Grow a ring's capacity (e.g. when the user switches from SBDF2 to SBDF4
mid-run). The existing entries are preserved, with new empty slots added
at the back. A no-op if the requested capacity is not larger than the
current capacity."""
function ring_resize!(ring::SPHistoryRing, new_capacity::Int)
    new_capacity <= ring.capacity && return ring
    if isempty(ring.buffers)
        ring.capacity = new_capacity
        return ring
    end
    # Copy entries age-sorted (newest first) into a new slot array.
    old_cap = length(ring.buffers)
    old_count = ring.count
    new_bufs = similar(ring.buffers, new_capacity + 1)
    for k in 1:old_count
        old_slot = mod1(ring.head - k + 1, old_cap)
        new_slot = length(new_bufs) - k + 1
        new_bufs[new_slot] = ring.buffers[old_slot]
    end
    # Any remaining new slots point at scratch buffers we'll create on demand;
    # mark them by reusing the first allocated buffer (shape reference).
    ref = ring.buffers[1]
    n = length(ref)
    for i in eachindex(new_bufs)
        if !isassigned(new_bufs, i)
            new_bufs[i] = similar_zeros(ref, eltype(ref), n)
        end
    end
    ring.buffers = new_bufs
    ring.capacity = new_capacity
    ring.head = length(new_bufs)
    return ring
end

"""Return the per-subproblem MX/LX/F rings, growing them as required."""
function _sp_multistep_rings!(state::TimestepperState, n_sp::Int, capacity::Int)
    cap = max(capacity, 1)
    MX_rings = get!(state.timestepper_data, :sp_multistep_MX_rings) do
        [SPHistoryRing(cap) for _ in 1:n_sp]
    end::Vector{SPHistoryRing}
    LX_rings = get!(state.timestepper_data, :sp_multistep_LX_rings) do
        [SPHistoryRing(cap) for _ in 1:n_sp]
    end::Vector{SPHistoryRing}
    F_rings = get!(state.timestepper_data, :sp_multistep_F_rings) do
        [SPHistoryRing(cap) for _ in 1:n_sp]
    end::Vector{SPHistoryRing}

    length(MX_rings) == n_sp && length(LX_rings) == n_sp && length(F_rings) == n_sp ||
        error("subproblem count changed after multistep history initialization")
    @inbounds for i in 1:n_sp
        MX_rings[i].capacity < cap && ring_resize!(MX_rings[i], cap)
        LX_rings[i].capacity < cap && ring_resize!(LX_rings[i], cap)
        F_rings[i].capacity  < cap && ring_resize!(F_rings[i], cap)
    end
    return MX_rings, LX_rings, F_rings
end

"""
    step_subproblem_multistep!(state, solver, subproblems, a, b, c)

Advance the state using an IMEX multistep method with coefficients `(a, b, c)`:

    Σ_k a[k] M X^{n+1-k} + Σ_k b[k] L X^{n+1-k} = Σ_k c[k] F^{n+1-k}

History is stored per subproblem and automatically trimmed to `length(a) - 1`
entries (the minimum needed for the next step). The caller is responsible for
falling back to a lower-order startup scheme when insufficient history exists.
"""
function step_subproblem_multistep!(
    state::TimestepperState, solver::InitialValueSolver,
    subproblems::Tuple, a::Tuple, b::Tuple, c::Tuple,
)
    problem = solver.problem

    # History depth — how many back steps we need to store for the NEXT call.
    # `a`, `b`, `c` contain the current-step value at index 1 (Julia 1-based),
    # and history at indices 2..end. We only need to store up to max(len-1).
    a_hist_depth = length(a) - 1
    b_hist_depth = length(b) - 1
    c_hist_depth = length(c) - 1

    state_fields = _cached_state_fields!(state, problem)
    for f in state_fields
        ensure_layout!(f, :c)
    end

    # Distributor handle for the mixed Fourier–Chebyshev solve-layout transpose
    # (to_solve_layout!/from_solve_layout! are no-ops outside distributed
    # mixed-basis runs). Same bracketing pattern as step_subproblem_rk!.
    dist = first(state_fields).dist

    n_sp = length(subproblems)

    # ── Per-subproblem ring-buffer history (cached in timestepper_data) ──────
    # We pre-allocate `max_capacity` slots per ring on first use. Subsequent
    # steps reuse those slots via `ring_stage!` (zero per-step
    # allocation). If the user switches to a higher-order method mid-run
    # (e.g. SBDF2 → SBDF4), we `ring_resize!` to grow capacity while
    # preserving existing history.
    max_capacity = max(a_hist_depth, b_hist_depth, c_hist_depth, 1)
    MX_rings, LX_rings, F_rings =
        _sp_multistep_rings!(state, n_sp, max_capacity)

    # ── Step 1: compute M*X_current, L*X_current, F_current per subproblem ──
    # Write directly into ring slots — no per-step allocation.
    F_fields = evaluate_rhs_buffered(solver, state_fields, solver.sim_time)
    _group_rhs_coefficients!(F_fields)

    # Solve-layout bracketing: `evaluate_rhs_buffered` above (the only op needing
    # the FFT pencil) has already run, so state enters the solve pencil ONCE here
    # and STAYS there through the gather AND the step-4 solve/scatter, popping back
    # to the FFT pencil only once before the history push. `ms_state_stash` threads
    # the restore. This collapses the old gather→restore→re-transpose round-trip
    # (steps 2-3 in between read only ring counts + equation_data, never FFT-pencil
    # state). One collective transpose per field list, OUTSIDE the subproblem loop.
    # F_fields are read-only here → restored by pointer swap only (no transpose back).
    # `fuse_from_grid=true`: state is in grid space (evaluate_rhs_buffered above pulled
    # it to :g), so fuse the coupled DCT into the single fft→solve transpose. Safe —
    # ms_state_stash is restored by `from_solve_layout!` (real transpose back), not a
    # pointer swap. F keeps the default (non-fused; it IS pointer-swapped).
    ms_state_stash = to_solve_layout!(state_fields, dist; fuse_from_grid=true)
    _ms_g_F        = to_solve_layout!(F_fields, dist)

    for (sp_idx, sp) in enumerate(subproblems)
        sp.M_min === nothing && continue
        # Gather X in variable space — CPU or GPU matching state-field storage.
        x_cur = gather_inputs(sp, state_fields)

        # The previous accepted time levels remain available if this attempt fails.
        mx_cur = ring_stage!(MX_rings[sp_idx], x_cur)
        lx_cur = ring_stage!(LX_rings[sp_idx], x_cur)
        f_cur  = ring_stage!(F_rings[sp_idx],  x_cur)

        # M*X_current and L*X_current in equation space
        M_op = _subproblem_operator(sp, :M, x_cur)
        L_op = _subproblem_operator(sp, :L, x_cur)
        _apply_subproblem_operator!(mx_cur, M_op, x_cur)
        _apply_subproblem_operator!(lx_cur, L_op, x_cur)

        # F_current in equation space (PDE rows only — BC rows handled below)
        gather_eqn_F!(f_cur, sp, solver, F_fields, state_fields)
    end

    # State STAYS in the solve pencil for step 4's solve/scatter (no transpose
    # back — nothing between here and the scatter reads FFT-pencil state).
    _restore_subproblem_rhs!(F_fields, _ms_g_F, solver)

    # ── Step 2: check if we have enough history to advance ──────────────────
    # The candidate supplies the current time level. Older levels can be
    # missing during startup, in which case the accumulator skips them, giving a
    # lower-order first step if the caller's startup dispatch didn't fall
    # back. This is the same behavior as the old `Vector{Vector}` path.
    have_enough = true
    for sp_idx in 1:n_sp
        subproblems[sp_idx].M_min === nothing && continue
        if F_rings[sp_idx].count + 1 < max(c_hist_depth, 1) ||
           MX_rings[sp_idx].count + 1 < max(a_hist_depth, 1) ||
           LX_rings[sp_idx].count + 1 < max(b_hist_depth, 1)
            have_enough = false
            break
        end
    end
    if !have_enough
        @debug "step_subproblem_multistep!: insufficient history, using zero padding"
    end

    # ── Step 3: compute F_alg once per step for BC row override ─────────────
    # Uses cached per-subproblem stage vectors — no per-step allocation.
    ALG_F = _sp_slots!(state, :_sp_ms_ALG_F, n_sp)
    for (sp_idx, sp) in enumerate(subproblems)
        if sp.M_min === nothing
            ALG_F[sp_idx] = ComplexF64[]
            continue
        end
        n = size(sp.M_min, 1)
        ref = _ring_candidate(F_rings[sp_idx])
        alg_f = _sp_stage_vector!(sp, :multistep_alg_f, n, ref)
        ALG_F[sp_idx] = _ensure_alg_F!(alg_f, sp,
                                     has_time_dependent_bcs(sp.problem.bc_manager))
    end

    # ── Step 4: prepare, solve independent modes, then scatter ─────────────
    # Cache/factor construction is serial. Workers use only per-mode buffers;
    # shared field writes and all pencil transposes stay on this task.
    RHS = _sp_slots!(state, :_sp_ms_RHS, n_sp)
    solutions = _sp_slots!(state, :_sp_ms_solutions, n_sp)
    factors = _local_mode_factor_slots!(state, n_sp)
    try
        _collectively(dist, "multistep local solves") do
            for (sp_idx, sp) in enumerate(subproblems)
                sp.M_min === nothing && continue
                n = size(sp.M_min, 1)
                ref = _ring_candidate(F_rings[sp_idx])
                RHS[sp_idx] = _sp_stage_vector!(sp, :multistep_rhs, n, ref)
                factors[sp_idx] = _get_or_build_multistep_lhs!(sp, a[1], b[1])
                solutions[sp_idx] = _sp_stage_vector!(sp, :multistep_sol, size(sp.M_min, 2), ref)
            end
            threaded_modes = _thread_local_modes(solver.base.threaded_modes, subproblems,
                                                 solver.base.matsolver)
            _foreach_local_mode!(n_sp, threaded_modes) do sp_idx
                sp = subproblems[sp_idx]
                sp.M_min === nothing && return
                _solve_multistep_local_mode!(sp, RHS[sp_idx], solutions[sp_idx], factors[sp_idx],
                                             ALG_F[sp_idx], F_rings[sp_idx], MX_rings[sp_idx],
                                             LX_rings[sp_idx], a, b, c)
            end
        end
        for (sp_idx, sp) in enumerate(subproblems)
            sp.M_min === nothing && continue
            scatter_inputs(sp, solutions[sp_idx], state_fields)
        end
    finally
        from_solve_layout!(ms_state_stash, dist)
    end

    for (i, sp) in enumerate(subproblems)
        sp.M_min === nothing && continue
        ring_accept!(MX_rings[i])
        ring_accept!(LX_rings[i])
        ring_accept!(F_rings[i])
    end

    # ── Step 5: push new state to history ───────────────────────────────────
    _push_trim!(state.history, state_fields, 1)
end

function _solve_multistep_local_mode!(sp, rhs, solution, factor, alg_f,
                                      F_ring, MX_ring, LX_ring, a, b, c)
    fill!(rhs, zero(eltype(rhs)))
    for k in 2:length(c)
        ck = c[k]
        iszero(ck) && continue
        entry = _ring_step_entry(F_ring, k - 1)
        entry === nothing || _sp_axpy!(rhs, ck, entry)
    end
    for k in 2:length(a)
        ak = a[k]
        iszero(ak) && continue
        entry = _ring_step_entry(MX_ring, k - 1)
        entry === nothing || _sp_axpy!(rhs, -ak, entry)
    end
    for k in 2:length(b)
        bk = b[k]
        iszero(bk) && continue
        entry = _ring_step_entry(LX_ring, k - 1)
        entry === nothing || _sp_axpy!(rhs, -bk, entry)
    end
    if !iszero(b[1])
        apply_bc_override!(rhs, alg_f, sp, b[1])
    end
    _solve_implicit_stage!(solution, factor, rhs, sp)
    return nothing
end

"""
    _get_or_build_multistep_lhs!(sp, a0, b0)

Return a cached solver for `(a0*M_min + b0*L_min)`, building it lazily and
invalidating when `(a0, b0)` changes (e.g., under variable-dt CNAB2).
"""
function _get_or_build_multistep_lhs!(sp::Subproblem, a0::Float64, b0::Float64)
    M = sp.M_min
    L = sp.L_min
    M === nothing && return nothing

    cache_key = "_sp_multistep_lhs"
    cache_key_k = "_sp_multistep_lhs_key"
    cached = get(sp.matrices, cache_key, nothing)
    cached_k = get(sp.matrices, cache_key_k, nothing)

    current_key = (a0, b0)
    if cached !== nothing && cached_k === current_key
        return cached
    end

    # Build the new LHS matrix. Under variable-dt multistep the sparsity
    # pattern is unchanged — only `nzval` differs — so we can refactor the
    # cached `SparseLUSolver` in place and reuse its symbolic analysis.
    LHS = if L !== nothing && !iszero(b0)
        ComplexF64(a0) * M + ComplexF64(b0) * L
    else
        ComplexF64(a0) * M
    end

    # Symbolic-reuse fast path: if the cached solver supports
    # `refactor!` (CPU SparseLUSolver, GPU CuSparseLU :rf backend, or
    # any future solver that opts in), call it to reuse the symbolic
    # factorization. Detected via `hasmethod` so the GPU path doesn't
    # need a hard dependency on CUDA.jl being loaded.
    if cached !== nothing && cached_k !== nothing && !_gpu_lhs_pinned(sp, cache_key) &&
       hasmethod(MatSolvers.refactor!, Tuple{typeof(cached), typeof(LHS)})
        try
            MatSolvers.refactor!(cached, LHS)
            sp.matrices[cache_key_k] = current_key
            return cached
        catch err
            @debug "multistep refactor! failed, rebuilding from scratch" exception=(err, catch_backtrace())
            # fall through to the full-rebuild path below.
        end
    end

    solver_type = _subproblem_solver_type(sp.solver.base.matsolver)
    solver_kwargs = _subproblem_solver_kwargs(sp.solver.base.matsolver)

    try
        lhs_solver = MatSolvers.solver_instance(solver_type, LHS; solver_kwargs...)
        sp.matrices[cache_key] = lhs_solver
        sp.matrices[cache_key_k] = current_key
        return lhs_solver
    catch err
        if _gpu_subproblem_execution(sp)
            pinned = _gpu_singular_factorization(sp, solver_type, LHS, solver_kwargs,
                                                 "multistep system")
            if pinned !== nothing
                sp.matrices[cache_key] = pinned
                sp.matrices[cache_key_k] = current_key
                _mark_gpu_lhs_pinned!(sp, cache_key)
                return pinned
            end
            error("GPU multistep factorization failed for group=$(sp.group); " *
                  "CPU sparse/dense fallback is disabled. Original error: " *
                  "$(sprint(showerror, err))")
        end
        # A gauge can leave a null space. QR solves that case and checks
        # consistency against this step's RHS before the solution is committed.
        if solver_type != MatSolvers.SPQRSolver
            try
                qr_solver = MatSolvers.solver_instance(MatSolvers.SPQRSolver, LHS)
                sp.matrices[cache_key] = qr_solver
                sp.matrices[cache_key_k] = current_key
                @warn "Multistep system for group=$(sp.group) is singular. " *
                      "Using sparse QR to check consistency and solve for its determined modes." maxlog=1
                return qr_solver
            catch qr_err
                @debug "multistep: sparse QR fallback also failed for group=$(sp.group)" exception=qr_err
            end
        end
        _throw_singular_stage_system(sp, "the multistep system (a₀·M + b₀·L)", err)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Coefficient builders for concrete multistep schemes
# ──────────────────────────────────────────────────────────────────────────────

"""
    _sp_multistep_history_depth(state)

Return the minimum F-history depth across all subproblems stored in
`timestepper_data`. Callers use this to decide whether a higher-order
multistep method has enough history to advance, or whether to fall back to a
lower-order startup scheme.
"""
function _sp_multistep_history_depth(state::TimestepperState)
    # Prefer the ring-buffer storage (new path). Fall back to the old
    # Vector{Vector} storage if a legacy cache is still present (e.g.,
    # mid-session after a code reload).
    F_rings = get(state.timestepper_data, :sp_multistep_F_rings, nothing)
    if F_rings !== nothing
        depth = typemax(Int)
        any_counted = false
        for ring in F_rings
            ring.count == 0 && continue
            depth = min(depth, ring.count)
            any_counted = true
        end
        return any_counted ? depth : 0
    end
    F_hist = get(state.timestepper_data, :sp_multistep_F_hist, nothing)
    F_hist === nothing && return 0
    depth = typemax(Int)
    any_counted = false
    for h in F_hist
        isempty(h) && continue
        depth = min(depth, length(h))
        any_counted = true
    end
    return any_counted ? depth : 0
end

"""CNAB1 coefficients for fixed timestep `dt`."""
@inline _cnab1_coefs(dt::Float64) =
    ((1.0/dt, -1.0/dt),    # a
     (0.5, 0.5),           # b
     (0.0, 1.0))           # c  (c[1] = current, c[2] = F_{n-0} → 1*F_n)

"""CNAB2 coefficients for current/previous timestep (variable step)."""
@inline function _cnab2_coefs(dt::Float64, dt_prev::Float64)
    w1 = dt / dt_prev
    a = (1.0/dt, -1.0/dt)
    b = (0.5, 0.5)
    c = (0.0, 1.0 + w1/2.0, -w1/2.0)
    return a, b, c
end

"""Modified CNAB2 coefficients, including the older implicit level."""
@inline function _mcnab2_coefs(dt::Float64, dt_prev::Float64)
    a, _, c = _cnab2_coefs(dt, dt_prev)
    w = dt / dt_prev
    return a, ((8 + 1/w)/16, (7 - 1/w)/16, 1/16), c
end

"""Integrate from tₙ₋₁ to tₙ₊₁ using trapezoidal L and the linear interpolant of F.

The interval midpoint is displaced from tₙ by (dt - dt_prev)/2. Evaluating the
interpolant there integrates it exactly. The homogeneous recurrence Xₙ₊₁ = Xₙ₋₁
keeps both leapfrog modes bounded as step sizes change.
"""
@inline function _cnlf2_coefs(dt::Float64, dt_prev::Float64)
    inv_span = 1.0 / (dt_prev + dt)
    shift = (dt - dt_prev) / dt_prev / 2
    a = (inv_span, 0.0, -inv_span)
    b = (0.5, 0.0, 0.5)
    c = (0.0, 1.0 + shift, -shift)
    return a, b, c
end

"""SBDF1 coefficients (backward Euler, 1st-order explicit)."""
@inline _sbdf1_coefs(dt::Float64) =
    ((1.0/dt, -1.0/dt),
     (1.0,),
     (0.0, 1.0))

"""SBDF2 coefficients."""
@inline _sbdf2_coefs(dt::Float64) = _sbdf2_coefs(dt, dt)
@inline function _sbdf2_coefs(dt::Float64, dt_prev::Float64)
    w1 = dt / dt_prev
    a = ((1.0 + 2.0*w1) / ((1.0 + w1) * dt),
         -(1.0 + w1) / dt,
         w1^2 / ((1.0 + w1) * dt))
    b = (1.0,)
    c = (0.0, 1.0 + w1, -w1)
    return a, b, c
end

"""SBDF3 coefficients."""
@inline _sbdf3_coefs(dt::Float64) = _sbdf3_coefs(dt, dt, dt)
@inline function _sbdf3_coefs(k2::Float64, k1::Float64, k0::Float64)
    w2 = k2 / k1
    w1 = k1 / k0
    a = ((1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2,
         (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2,
         w2^2 * (w1 + 1/(1 + w2)) / k2,
         -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2)
    b = (1.0, 0.0, 0.0, 0.0)
    c = (0.0,
         (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1),
         -w2*(1 + w1*(1 + w2)),
         w1*w1*w2*(1 + w2) / (1 + w1))
    return a, b, c
end

"""SBDF4 coefficients."""
@inline _sbdf4_coefs(dt::Float64) = _sbdf4_coefs(dt, dt, dt, dt)
@inline function _sbdf4_coefs(k3::Float64, k2::Float64, k1::Float64, k0::Float64)
    w3 = k3 / k2
    w2 = k2 / k1
    w1 = k1 / k0
    A1 = 1 + w1*(1 + w2)
    A2 = 1 + w2*(1 + w3)
    A3 = 1 + w1*A2
    a = ((1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3,
         (-1 - w3*(1 + (w2*(1 + w3)/(1 + w2)) * (1 + w1*A2/A1))) / k3,
         w3 * (w3/(1 + w3) + (w2*w3*(A3 + w1))/(1 + w1)) / k3,
         -(w2^3 * w3^2 * (1 + w3) * A3) / ((1 + w2) * A2 * k3),
         ((1 + w3) * A2 * w1^4 * w2^3 * w3^2) / ((1 + w1) * A1 * A3 * k3))
    b = (1.0, 0.0, 0.0, 0.0, 0.0)
    c = (0.0,
         (w2 * (1 + w3) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2)) / ((1 + w2) * A1),
         -(A2 * A3 * w3) / (1 + w1),
         (w2^2 * w3 * (1 + w3) * A3) / (1 + w2),
         -(w1^3 * w2^2 * w3 * (1 + w3) * A2) / ((1 + w1) * A1))
    return a, b, c
end
