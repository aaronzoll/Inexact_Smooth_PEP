include("../Solvers/BnB_PEP_Inexact_Smooth.jl")

using Optim
using OffsetArrays

function outer_optimize_PEP(
    N, L, R, p, zero_idx;
    # Initial values (optional)
    α_init       = nothing,
    ε_init       = nothing,
    # Optim options
    max_iterations = 500,
    x_tol          = 1e-5,
    f_tol          = 1e-8,
    show_trace     = true,
    # Passed through to inner solver
    show_output          = :off,
    ϵ_tol_feas           = 1e-6,
    objective_type       = :default,
    obj_val_upper_bound  = default_obj_val_upper_bound
)
    M = 1  # fixed per your specification

    # ------------------------------------------------------------------ #
    #  Index bookkeeping                                                   #
    # ------------------------------------------------------------------ #
    # α[i, j] for i in 1:N, j in 0:i-1  →  N*(N+1)/2 entries
    # We store them in a flat vector in row-major (i outer, j inner) order.

    α_indices = [(i, j) for i in 1:N for j in 0:i-1]
    n_α   = length(α_indices)        # N*(N+1)/2
    n_ε   = M                        # = 1
    n_tot = n_ε + n_α

    # ------------------------------------------------------------------ #
    #  Pack / unpack helpers                                               #
    # ------------------------------------------------------------------ #
    # We optimize in *unconstrained* space u, where the physical variables
    # are x = exp(u) ≥ 0.  This replaces Fminbox for pure lower-bound = 0
    # constraints and is compatible with NelderMead.

    function pack(ε_set, α)
        u = Vector{Float64}(undef, n_tot)
        for m in 1:M
            u[m] = log(ε_set[m])
        end
        for (k, (i, j)) in enumerate(α_indices)
            u[n_ε + k] = log(α[i, j])
        end
        return u
    end

    function unpack(u)
        # ε_set: plain Vector indexed 1:M, wrapped as OffsetVector
        ε_vec  = OffsetArray([exp(u[m]) for m in 1:M], 1:M)

        # α: OffsetMatrix with row indices 1:N, col indices 0:N-1
        α_mat  = OffsetArray(zeros(N, N), 1:N, 0:N-1)
        for (k, (i, j)) in enumerate(α_indices)
            α_mat[i, j] = exp(u[n_ε + k])
        end

        return ε_vec, α_mat
    end

    # ------------------------------------------------------------------ #
    #  Build initial point                                                 #
    # ------------------------------------------------------------------ #
    if α_init === nothing
        # Default: constant stepsize 1/N for each entry
        α_init = OffsetArray(zeros(N, N), 1:N, 0:N-1)
        for (i, j) in α_indices
            α_init[i, j] = 1.0 / N
        end
    end

    if ε_init === nothing
        ε_init = OffsetArray(ones(M), 1:M)
    end

    u0 = pack(ε_init, α_init)

    # ------------------------------------------------------------------ #
    #  Objective  (calls inner SDP solver)                                #
    # ------------------------------------------------------------------ #
    call_count = Ref(0)

    function objective(u)
        call_count[] += 1
        ε_set, α = unpack(u)

        obj = try
            val, _, _ = solve_dual_PEP_with_known_stepsizes(
                N, L, α, R, ε_set, p, zero_idx;
                show_output         = show_output,
                ϵ_tol_feas          = ϵ_tol_feas,
                objective_type      = objective_type,
                obj_val_upper_bound = obj_val_upper_bound
            )
            val
        catch err
            @warn "Inner solve failed at call $(call_count[]) — returning Inf" err
            Inf
        end

        if show_trace && mod(call_count[], 10) == 0
            ε_set, α = unpack(u)
            @info "Outer iter $(call_count[])  objective = $obj  ε = $(collect(ε_set))"
        end

        return obj
    end

    # ------------------------------------------------------------------ #
    #  Run Nelder-Mead                                                     #
    # ------------------------------------------------------------------ #
    opts = Optim.Options(
        iterations       = max_iterations,
        x_tol            = x_tol,
        f_tol            = f_tol,
        show_trace       = show_trace,
        store_trace      = true,
        extended_trace   = false
    )

    result = Optim.optimize(objective, u0, NelderMead(), opts)

    # ------------------------------------------------------------------ #
    #  Unpack and return optimal solution                                  #
    # ------------------------------------------------------------------ #
    u_opt       = Optim.minimizer(result)
    ε_set_opt, α_opt = unpack(u_opt)
    obj_opt     = Optim.minimum(result)

    @info """
    ✅ Outer optimization complete
       Converged  : $(Optim.converged(result))
       Iterations : $(Optim.iterations(result))
       Objective  : $obj_opt
       ε_opt      : $(collect(ε_set_opt))
    """

    return obj_opt, ε_set_opt, α_opt, result
end


outer_optimize_PEP(50, 1, 1, 1, [];)