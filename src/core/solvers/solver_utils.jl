# ============================================================================
# Exports
# ============================================================================

# Export solver types
export Solver, SolverPerformanceStats, SolverBaseData
export InitialValueSolver, BoundaryValueSolver, EigenvalueSolver

# Export compiled RHS plan
export CompiledRHSPlan, compile_rhs_plan!, execute_compiled_rhs!

# Export introspection
export diagnose

# ============================================================================
# Solver Introspection
# ============================================================================

"""
    diagnose(solver)

Print a formatted summary of solver state, configuration, and resource usage.
Useful for debugging and understanding solver behavior.

# Example
```julia
solver = InitialValueSolver(problem, RK222; dt=1e-3)
diagnose(solver)
```
"""
function diagnose(solver::InitialValueSolver)
    println("╔══════════════════════════════════════════════════╗")
    println("║              Solver Diagnostics                 ║")
    println("╠══════════════════════════════════════════════════╣")

    # Timestepper info
    ts = solver.timestepper
    ts_name = typeof(ts)
    println("║ Timestepper: $ts_name")
    println("║ dt = $(solver.dt), sim_time = $(round(solver.sim_time; digits=6))")
    println("║ iteration = $(solver.iteration)")
    println("╠──────────────────────────────────────────────────╣")

    # State fields
    println("║ State fields: $(length(solver.state))")
    total_dof = 0
    total_mem = 0
    for (i, field) in enumerate(solver.state)
        gdata = get_grid_data(field)
        sz = gdata !== nothing ? size(gdata) : ()
        dof = prod(sz; init=1)
        mem = dof * sizeof(field.dtype)
        total_dof += dof
        total_mem += 2 * mem  # grid + coeff
        layout = field.current_layout
        println("║   $(i). $(field.name) $(sz) [$(layout)] $(field.dtype)")
    end
    println("║ Total DOF: $(total_dof), Memory: $(round(total_mem / 1024^2; digits=2)) MB")
    println("╠──────────────────────────────────────────────────╣")

    # Architecture
    arch = solver.state[1].dist.architecture
    println("║ Architecture: $(typeof(arch))")
    println("║ MPI ranks: $(solver.state[1].dist.size)")
    mesh = solver.state[1].dist.mesh
    if mesh !== nothing
        println("║ Process mesh: $mesh")
    end
    println("╠──────────────────────────────────────────────────╣")

    # Transforms
    dist = solver.state[1].dist
    n_transforms = length(dist.transforms)
    println("║ Transforms: $n_transforms registered")
    for (i, tr) in enumerate(dist.transforms)
        println("║   $(i). $(typeof(tr))")
    end

    # Bases
    if !isempty(solver.state) && solver.state[1].domain !== nothing
        bases = solver.state[1].bases
        println("║ Bases:")
        for (i, basis) in enumerate(bases)
            btype = nameof(typeof(basis))
            N = basis.meta.size
            bounds = basis.meta.bounds
            println("║   $(i). $btype(N=$N, bounds=$bounds)")
        end
    end
    println("╠──────────────────────────────────────────────────╣")

    # Compiled RHS
    if solver.compiled_rhs !== nothing
        plan = solver.compiled_rhs
        if plan.is_compiled
            println("║ RHS: COMPILED ($(length(plan.instructions)) instructions, $(length(plan.workspace)) workspace fields)")
        else
            println("║ RHS: compilation failed — using interpreted evaluation")
        end
    else
        println("║ RHS: interpreted (not compiled)")
    end

    # Nonlinear evaluator
    if dist.nonlinear_evaluator !== nothing
        eval = dist.nonlinear_evaluator
        println("║ Nonlinear: dealiasing=$(eval.dealiasing_factor)")
        n_cached = length(eval.temp_fields)
        println("║   Cached temp fields: $n_cached")
        stats = eval.performance_stats
        if stats.total_evaluations > 0
            avg = stats.total_time / stats.total_evaluations * 1000
            println("║   Evaluations: $(stats.total_evaluations), avg=$(round(avg; digits=2)) ms")
        end
    end
    println("╠──────────────────────────────────────────────────╣")

    # Boundary conditions
    bc = solver.problem.bc_manager
    n_bcs = length(bc.conditions)
    println("║ Boundary conditions: $n_bcs")
    has_time_dep = has_time_dependent_bcs(bc)
    println("║   Time-dependent: $has_time_dep")

    # Stochastic forcing
    if hasfield(typeof(solver.problem), :stochastic_forcings) && !isempty(solver.problem.stochastic_forcings)
        println("║ Stochastic forcing: $(length(solver.problem.stochastic_forcings)) fields")
    end

    # Performance
    stats = solver.performance_stats
    if stats.total_steps > 0
        avg_step = stats.total_time / stats.total_steps * 1000
        println("╠──────────────────────────────────────────────────╣")
        println("║ Performance:")
        println("║   Total steps: $(stats.total_steps)")
        println("║   Total time: $(round(stats.total_time; digits=2))s")
        println("║   Avg step: $(round(avg_step; digits=2)) ms")
    end

    println("╚══════════════════════════════════════════════════╝")
end

function Base.show(io::IO, plan::CompiledRHSPlan)
    status = plan.is_compiled ? "compiled" : "failed"
    print(io, "CompiledRHSPlan($status, $(length(plan.instructions)) instructions, $(length(plan.workspace)) workspace)")
end

# Export core solver API
export step!, solve!, proceed, run!
export attach_evaluator!, solver_comm

# Export matrix building functions
export build_solver_matrices!, apply_entry_cutoff!

# Export solution vector functions
export fields_to_vector, copy_solution_to_fields!
export compute_field_vector_size, extract_field_data_for_vector, set_field_data_from_vector!
export get_basis_size

# Export nonlinear solver functions
export solve_linear!, solve_nonlinear!
export evaluate_residual_and_jacobian

# Export performance/logging functions
export log_stats, log_solver_performance

# Export architecture query
export get_solver_architecture
