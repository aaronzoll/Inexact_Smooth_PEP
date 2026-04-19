include("../Solvers/BnB_PEP_Inexact_Smooth.jl")

using OffsetArrays
using LinearAlgebra, Random, Printf
using DelimitedFiles
using DataFrames, CSV
using NLopt

# ─────────────────────────────────────────────────────────────────────────────
#  α / h conversion utilities
# ─────────────────────────────────────────────────────────────────────────────

function compute_α_from_h(h, N)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for ℓ in 1:N
        for i in 0:ℓ-1
            if i == ℓ - 1
                α[ℓ, i] = h[ℓ, ℓ-1]
            elseif i <= ℓ - 2
                α[ℓ, i] = α[ℓ-1, i] + h[ℓ, i]
            end
        end
    end
    return α
end

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm: subgradient averaging
# ─────────────────────────────────────────────────────────────────────────────

function make_α_SG_avg_no_scale_by_N(N, β, R)
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    η = R / (β)

    for k in 1:N-1
        H[k, k-1] = η
    end

    for j in 0:N-2
        H[N, j] = -η * (1 + j) / (N + 1)
    end
    H[N, N-1] = η / (N + 1)

    return compute_α_from_h(H, N)
end


function make_α_subgradient_avg(N, β, R)
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    η = R / (β * sqrt(N + 1))

    for k in 1:N-1
        H[k, k-1] = η
    end

    for j in 0:N-2
        H[N, j] = -η * (1 + j) / (N + 1)
    end
    H[N, N-1] = η / (N + 1)

    return compute_α_from_h(H, N)
end

# ─────────────────────────────────────────────────────────────────────────────
#  δ_set helpers
# ─────────────────────────────────────────────────────────────────────────────

function make_δ_set(N, M, value=1.0)
    n = N + 2
    raw = fill(value, n, n, M)
    return OffsetArray(raw, -1:N, -1:N, 1:M)
end

# ─────────────────────────────────────────────────────────────────────────────
#  δ optimizer (α fixed)
# ─────────────────────────────────────────────────────────────────────────────

function optimize_δ_for_fixed_α_simple(
    N, β, α, R, p, zero_idx;
    δ_lb=1e-6,
    δ_ub=1e6,
    max_sdp_calls=2000,
    ftol_rel=1e-6,
    ftol_abs=1e-8,
    xtol_rel=1e-6,
    show_trace=false,
    trace_interval=10,
    algorithm=:LN_BOBYQA,
    show_output=:off,
    ϵ_tol_feas=1e-6,
    objective_type=:default,
    obj_val_upper_bound=1e6,
)
    M = 1
    I_N_star = -1:N
    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star, M)

    idx_set_upper = filter(t -> t.i < t.j, idx_set_λ)
    n_vars = length(idx_set_upper)

    u_lb = log(δ_lb)
    u_ub = log(δ_ub)

    function make_δ_set_from_u(u)
        δ_set = make_δ_set(N, M, 0.0)
        for (k, t) in enumerate(idx_set_upper)
            δ_set[t.i, t.j, t.m] = exp(clamp(u[k], u_lb, u_ub))
        end
        return δ_set
    end

    function eval_sdp(u)
        δ_set = make_δ_set_from_u(clamp.(u, u_lb, u_ub))
        val = try
            model_val, _, _ = solve_dual_PEP_with_known_stepsizes(
                N, β, α, R, δ_set, p, zero_idx;
                show_output=show_output,
                ϵ_tol_feas=ϵ_tol_feas,
                objective_type=objective_type,
                obj_val_upper_bound=obj_val_upper_bound,
                acceptable_termination_statuses=[MOI.OPTIMAL]
            )
            model_val
        catch
            Inf
        end
        return isfinite(val) ? val : Inf
    end

    call_count = Ref(0)
    best_val   = Ref(Inf)
    best_u     = zeros(n_vars)

    function nlopt_obj(u::Vector, grad::Vector)
        call_count[] += 1
        u_c = clamp.(u, u_lb, u_ub)
        val = eval_sdp(u_c)

        if val < best_val[]
            best_val[] = val
            best_u .= u_c
        end

        if show_trace && mod(call_count[], trace_interval) == 0
            @info "[call $(call_count[])] obj=$(round(val;sigdigits=5))  best=$(round(best_val[];sigdigits=5))"
        end

        return val
    end

    opt = NLopt.Opt(algorithm, n_vars)
    opt.lower_bounds  = fill(u_lb, n_vars)
    opt.upper_bounds  = fill(u_ub, n_vars)
    opt.maxeval       = max_sdp_calls
    opt.ftol_rel      = ftol_rel
    opt.ftol_abs      = ftol_abs
    opt.xtol_rel      = xtol_rel
    opt.min_objective = nlopt_obj

    n_restarts       = 10
    global_best_val  = Inf
    global_best_u    = zeros(n_vars)

    for restart in 1:n_restarts
        u0 = restart == 1 ? zeros(n_vars) : randn(n_vars) .* 0.5
        u0 .= clamp.(u0, u_lb, u_ub)

        call_count[] = 0
        best_val[]   = Inf
        best_u      .= 0.0

        (obj_opt, u_opt_nlopt, ret) = NLopt.optimize(opt, u0)

        u_candidate   = best_val[] <= obj_opt ? best_u : clamp.(u_opt_nlopt, u_lb, u_ub)
        obj_candidate = min(best_val[], obj_opt)

        @info "Restart $restart/$n_restarts: obj=$(round(obj_candidate;sigdigits=6)) ret=$ret calls=$(call_count[])"

        if obj_candidate < global_best_val
            global_best_val   = obj_candidate
            global_best_u    .= u_candidate
        end
    end

    δ_set_opt = make_δ_set_from_u(global_best_u)
    @info "✅ done | obj=$(global_best_val)"

    return global_best_val, δ_set_opt
end

# ─────────────────────────────────────────────────────────────────────────────
#  2-D sweep: N ∈ [N_start, N_stop], p ∈ p_vals
# ─────────────────────────────────────────────────────────────────────────────

function run_sweep_N_p_subgradient_avg(;
    R::Real          = 1.0,
    β::Real          = 1.0,
    Lip_scaling::Real = 1.0,
    N_start::Int     = 2,
    N_stop::Int      = 20,
    N_step::Int      = 1,
    p_vals::AbstractVector = range(0.0, 1.0; length=11),
    zero_idx         = [],
    show_trace::Bool = false,
    obj_val_upper_bound::Real = 1e6,
    csv_path::String = "./Data and Plotting/subgradient_avg_sweep_N_p.csv",
)
    L      = β * Lip_scaling
    N_vals = collect(N_start:N_step:N_stop)

    # results[N_index, p_index] = obj_opt
    results = fill(NaN, length(N_vals), length(p_vals))

    for (ni, N) in enumerate(N_vals), (pi, p) in enumerate(p_vals)
        display("Running N=$(N), p=$(round(p; digits=4))")

        α = make_α_SG_avg_no_scale_by_N(N, L, R)


        obj_opt, _ = optimize_δ_for_fixed_α_simple(
                N, β, α, R, p, zero_idx;
                show_trace          = show_trace,
                obj_val_upper_bound = obj_val_upper_bound)


        results[ni, pi] = obj_opt
    end

    # Build wide DataFrame: one column per p value
    p_col_names = [Symbol("p_$(round(p; digits=4))") for p in p_vals]
    df = DataFrame(N = N_vals)
    for (pi, col) in enumerate(p_col_names)
        df[!, col] = results[:, pi]
    end

    CSV.write(csv_path, df)
    @info "Saved results to $csv_path"
    return df
end

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

df = run_sweep_N_p_subgradient_avg(
    R            = 1.0,
    β            = 1.0,
    Lip_scaling  = 1.0,
    N_start      = 1,
    N_stop       = 1,
    N_step       = 1,
    p_vals       = range(0.0, 1.0; length=5),   # 0.0, 0.1, …, 1.0
    show_trace   = false,
    csv_path     = "./Data and Plotting/SGD_sweep_no_N_scaling_full_N_1.csv",
)