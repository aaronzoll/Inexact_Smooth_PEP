include("../Solvers/BnB_PEP_Inexact_Smooth.jl")

using OffsetArrays
using LinearAlgebra, Random, Printf
using DelimitedFiles, Plots, LaTeXStrings
using DataFrames, CSV

# Used by `optimize_δ_for_fixed_α` (keyword default) and sweeps below.

function compute_theta(N)
    θ = zeros(N + 1)
    θ[1] = 1.0  # θ₀
    for i in 2:N
        θ[i] = (1 + sqrt(1 + 4 * θ[i-1]^2)) / 2
    end
    θ[N+1] = (1 + sqrt(1 + 8 * θ[N]^2)) / 2  # θ_N
    return θ
end
function compute_H(N)
    θ = compute_theta(N)
    H = zeros(N, N)
    for i in 1:N
        for k in 1:i-1
            sum_hjk = sum(H[j, k] for j in k:i)
            H[i, k] = (1 / θ[i+1]) * (2 * θ[k] - sum_hjk)
        end
        H[i, i] = 1 + (2 * θ[i] - 1) / θ[i+1]
    end
    return H
end

function OGM_rates(β, R, N)

    theta = 1
    rate = []
    for i in 0:N-1
        if i < N - 1
            theta = (1 + sqrt(1 + 4 * theta^2)) / 2
        else
            theta = (1 + sqrt(1 + 8 * theta^2)) / 2
        end

        push!(rate, β * R^2 / (2 * theta^2))
    end

    return rate
end

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

function compute_h_from_α(α, N)
    h_new = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for l in N:-1:1
        h_new[l, l-1] = α[l, l-1]
        for i in l-2:-1:0
            h_new[l, i] = α[l, i] - α[l-1, i]
        end
    end
    return h_new
end



# ─────────────────────────────────────────────────────────────────────────────
#  α-matrix constructors  (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────

function make_α_gradient_descent(N, β, R)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for i in 1:N, j in 0:i-1
        α[i, j] = h/β
    end
    return α
end


function make_α_nesterov(N, L, R)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)

    t = ones(N + 2)
    for k in 2:N+2
        t[k] = (1 + sqrt(1 + 4 * t[k-1]^2)) / 2
    end
    β(k) = (t[k] - 1) / t[k+1]

    N >= 1 && (α[1, 0] = 1.0)

    for k in 2:N-1
        βk = β(k)
        α[k, k-1] = 1 + βk
        α[k, k-2] = (1 + βk) * α[k-1, k-2] - βk
        for j in 0:k-3
            α[k, j] = (1 + βk) * α[k-1, j] - βk * α[k-2, j]
        end
    end

    for j in 0:N-2
        α[N, j] = α[N-1, j]
    end
    α[N, N-1] = 1.0

    return α ./ L
end


function make_α_ogm(N, L, R)
    θ = compute_theta(N)
    H = OffsetArray(compute_H(N), 1:N, 0:N-1)
    return compute_α_from_h(H, N)
end


function verify_nesterov(N, L; d=3, seed=42)
    Random.seed!(seed)

    x0 = randn(d)
    g = [randn(d) for _ in 0:N-1]

    t = ones(N + 2)
    for k in 2:N+2
        t[k] = (1 + sqrt(1 + 4 * t[k-1]^2)) / 2
    end
    β(k) = (t[k] - 1) / t[k+1]

    nes_query = [zeros(d) for _ in 0:N]
    grad_step = [zeros(d) for _ in 0:N-1]

    nes_query[1] = x0

    for k in 0:N-1
        grad_step[k+1] = nes_query[k+1] - (1 / L) .* g[k+1]

        if k < N - 1
            if k == 0
                nes_query[2] = grad_step[1]
            else
                βk = β(k + 1)
                nes_query[k+2] = grad_step[k+1] .+
                                 βk .* (grad_step[k+1] .- grad_step[k])
            end
        end
    end
    nes_output = grad_step[N]

    α = make_α_nesterov(N, L)

    fsfom_query = [zeros(d) for _ in 0:N]
    fsfom_query[1] = x0

    for i in 1:N
        fsfom_query[i+1] = x0 .-
                           sum(α[i, j] .* g[j+1] for j in 0:i-1)
    end
    fsfom_output = fsfom_query[N+1]

    println("="^60)
    println("Nesterov verification   N=$N   L=$L   d=$d")
    println("="^60)

    all_match = true
    for k in 0:N-1
        err = norm(nes_query[k+1] .- fsfom_query[k+1], Inf)
        ok = err < 1e-10
        all_match &= ok
        match_str = ok ? "✓" : "✗  ← MISMATCH"
        println("  x_$k \t ‖error‖∞ = $(round(err; sigdigits=3)) \t $match_str")
    end

    err_out = norm(nes_output .- fsfom_output, Inf)
    ok_out = err_out < 1e-10
    all_match &= ok_out
    match_str = ok_out ? "✓" : "✗  ← MISMATCH"
    println("  x_$N \t ‖error‖∞ = $(round(err_out; sigdigits=3)) \t $match_str  (output)")

    println("="^60)
    println(all_match ? "  ✅ All points match." : "  ❌ Mismatch detected.")
    println("="^60)

    return all_match, nes_output, fsfom_output
end


function make_α_optimal_SG_step(N, β, R)
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for k in 1:N
        H[k, k-1] = R * (N + 1 - k) / (β * sqrt((N + 1)^3))
    end
    return compute_α_from_h(H, N)
end



function make_α_ssep(N, β, R)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    η = R / (β * sqrt(N + 1))

    for i in 1:N
        # Coefficient for g_{i-1} (newest subgradient): only the d_i term
        α[i, i-1] = η / (i + 1)

        # Coefficients for g_0, ..., g_{i-2}: inherited from x_{i-1} plus d_i term
        for j in 0:(i-2)
            α[i, j] = (i / (i + 1)) * α[i-1, j] + η / (i + 1)
        end
    end

    return α
end


function make_α_subgradient_avg(N, β, R)
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    η = R / (β * sqrt(N + 1))

    # Rows 1..N-1: plain gradient descent steps
    for k in 1:N-1
        H[k, k-1] = η
    end

    # Row N: averaging correction from z_{N-1} to z̄_N
    for j in 0:N-2
        H[N, j] = -η * (1 + j) / (N + 1)
    end
    H[N, N-1] = η / (N + 1)


    return compute_α_from_h(H, N)
end

function make_α_BSOGM(N, β, R)
    η = sqrt(2) * R * sqrt(N + 1) / (β * N)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for k in 1:N          # row  (iterate index)
        for i in 0:k-1    # col  (gradient index, 1-based: i+1)
            α[k, i] = η * (k - i) / (k + 1)
        end
    end
    return α
end


# ─────────────────────────────────────────────────────────────────────────────
#  δ_set helpers
#
#  δ_set is a 3-D OffsetArray:  δ_set[i, j, m]
#    i, j ∈ I_N_star  =  -1:N      (the usual PEP index set)
#    m    ∈ 1:M
#
#  Only entries with i ≠ j are ever read; the diagonal is unused.
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_δ_set(N, M, value = 1.0) -> OffsetArray{Float64, 3}

Allocate an δ_set tensor of shape (-1:N, -1:N, 1:M) filled with `value`.
"""
function make_δ_set(N, M, value=1.0)
    n = N + 2                          # length of -1:N
    raw = fill(value, n, n, M)
    return OffsetArray(raw, -1:N, -1:N, 1:M)
end

"""
    pack_δ(δ_set, idx_set_λ) -> Vector{Float64}

Flatten the entries of δ_set that are actually used (one per (i,j,m) triple
in idx_set_λ) into a plain vector, in the same order as idx_set_λ.
The log transform is applied so the optimisation variable is unconstrained.
"""
function pack_δ(δ_set, idx_set_λ)
    return [log(δ_set[t.i, t.j, t.m]) for t in idx_set_λ]
end

"""
    unpack_δ(u, idx_set_λ, N, M) -> OffsetArray{Float64, 3}

Inverse of pack_δ: reconstruct a full δ_set tensor from a flat vector `u`
(which lives in unconstrained log-space).
"""
function unpack_δ(u, idx_set_λ, N, M)
    δ_set = make_δ_set(N, M, 1.0)
    for (k, t) in enumerate(idx_set_λ)
        δ_set[t.i, t.j, t.m] = exp(u[k])
    end
    return δ_set
end



# ─────────────────────────────────────────────────────────────────────────────
#  δ-only outer optimizer  (α fixed)
# ─────────────────────────────────────────────────────────────────────────────

using NLopt

# function optimize_δ_for_fixed_α(
#     N, L, α, R, p, zero_idx;
#     δ_init          = nothing,
#     δ_lb            = 1e-8,
#     δ_ub            = 1e4,
#     max_sdp_calls   = 4000,
#     ftol_rel        = 1e-6,
#     ftol_abs        = 1e-8,
#     xtol_rel        = 1e-6,
#     show_trace      = true,
#     trace_interval  = 20,
#     algorithm       = :LN_BOBYQA,   # or :LN_SBPLX, :LN_COBYLA, :LN_NELDERMEAD
#     show_output          = :off,
#     ϵ_tol_feas           = 1e-8,
#     objective_type       = :default,
#     obj_val_upper_bound  = 1e6,
# )
#     M         = 1
#     I_N_star  = -1:N
#     idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star, M)

#     # ── active (i,j) pairs ────────────────────────────────────────────────────
#     function is_active(i, j)
#         is_consecutive = (j == i + 1) && (i in -1:N-1)
#         is_star_to_all = (i == -1)    && (j in 0:N)
#         return is_consecutive || is_star_to_all
#     end

#     idx_set_λ_lower          = filter(t -> t.i > t.j, idx_set_λ)
#     idx_set_λ_active         = filter(t -> t.i < t.j && is_active(t.i, t.j), idx_set_λ)
#     idx_set_λ_inactive_upper = filter(t -> t.i < t.j && !is_active(t.i, t.j), idx_set_λ)
#     n_vars = length(idx_set_λ_active)

#     # ── augmented zero_idx ────────────────────────────────────────────────────
#     lower_tri_zero      = [(t.i, t.j, t.m) for t in idx_set_λ_lower]
#     inactive_upper_zero = [(t.i, t.j, t.m) for t in idx_set_λ_inactive_upper]
#     if p == 1.0
#         zero_idx_aug        = vcat(zero_idx, lower_tri_zero, inactive_upper_zero)
#     else 
#         zero_idx_aug        = vcat(zero_idx)
#     end
#     @info "Algorithm: $algorithm  |  Active δ variables: $n_vars"
#     @info "  consecutive (i,i+1) : $(count(t -> t.j == t.i+1, idx_set_λ_active))"
#     @info "  star-to-all (-1,i)  : $(count(t -> t.i == -1,    idx_set_λ_active))"
#     @info "  δ ∈ [$δ_lb, $δ_ub]"

#     # ── δ_set builder ─────────────────────────────────────────────────────────
#     function make_δ_set_from_δ_vec(δ_vec)
#         δ_set = make_δ_set(N, M, δ_lb)
#         for (k, t) in enumerate(idx_set_λ_active)
#             δ_set[t.i, t.j, t.m] = δ_vec[k]
#         end
#         return δ_set
#     end

#     # ── tracking ──────────────────────────────────────────────────────────────
#     call_count = Ref(0)
#     best_val   = Ref(Inf)
#     best_δ_vec = δ_init === nothing ? ones(n_vars) :
#                  clamp.([δ_init[t.i, t.j, t.m] for t in idx_set_λ_active], δ_lb, δ_ub)

#     # ── NLopt objective (must match signature (x, grad) -> f) ─────────────────
#     function nlopt_objective(δ_vec::Vector, grad::Vector)
#         # grad is always empty for derivative-free methods — NLopt still passes it
#         @assert isempty(grad) "Gradient unexpectedly requested — use a derivative-free algorithm"

#         call_count[] += 1
#         δ_set = make_δ_set_from_δ_vec(δ_vec)

#         val = try
#             v, _, _ = solve_dual_PEP_with_known_stepsizes(
#                 N, L, α, R, δ_set, p, zero_idx_aug;
#                 show_output         = :off,
#                 ϵ_tol_feas          = ϵ_tol_feas,
#                 objective_type      = objective_type,
#                 obj_val_upper_bound = obj_val_upper_bound
#             )
#             v
#         catch err
#             @warn "Inner SDP failed at call $(call_count[])" err
#             return Inf   # NLopt treats Inf as a failed evaluation and steps away
#         end

#         if val < best_val[]
#             best_val[]   = val
#             best_δ_vec  .= δ_vec
#         end

#         if show_trace && mod(call_count[], trace_interval) == 0
#             @info "  [call $(call_count[])]" *
#                   "  obj = $(round(val; sigdigits=6))" *
#                   "  best = $(round(best_val[]; sigdigits=6))" *
#                   "  δ ∈ [$(round(minimum(δ_vec); sigdigits=3))," *
#                   " $(round(maximum(δ_vec); sigdigits=3))]"
#         end

#         return val
#     end

#     # ── build NLopt optimizer ─────────────────────────────────────────────────
#     opt = NLopt.Opt(algorithm, n_vars)

#     opt.lower_bounds  = fill(δ_lb, n_vars)
#     opt.upper_bounds  = fill(δ_ub, n_vars)
#     opt.maxeval       = max_sdp_calls
#     opt.ftol_rel      = ftol_rel
#     opt.ftol_abs      = ftol_abs
#     opt.xtol_rel      = xtol_rel
#     opt.min_objective = nlopt_objective

#     # ── initial point ─────────────────────────────────────────────────────────
#     u0 = δ_init === nothing ? ones(n_vars) :
#          clamp.([δ_init[t.i, t.j, t.m] for t in idx_set_λ_active], δ_lb, δ_ub)

#     @info "Initial objective: $(best_val[])"

#     # ── optimize ──────────────────────────────────────────────────────────────
#     (obj_opt, δ_opt_vec, ret) = NLopt.optimize(opt, u0)

#     # use best seen rather than final iterate in case NLopt ends on a bad step
#     if best_val[] < obj_opt
#         δ_opt_vec = copy(best_δ_vec)
#         obj_opt   = best_val[]
#     end

#     δ_set_opt    = make_δ_set_from_δ_vec(δ_opt_vec)
#     active_δ_opt = [(idx_set_λ_active[k].i, idx_set_λ_active[k].j,
#                      round(δ_opt_vec[k]; sigdigits=5))
#                     for k in 1:n_vars]

#     @info """
#     ✅  δ optimisation complete
#         Algorithm      : $algorithm
#         Return code    : $ret
#         Active pairs   : $n_vars
#         SDP calls      : $(call_count[])
#         δ bounds       : [$δ_lb, $δ_ub]
#         Objective      : $obj_opt
#     """
#     @info "Optimal δ (i, j, δ):" active_δ_opt

#     return obj_opt, δ_set_opt, zero_idx_aug
# end


function s_recurrence(a, N)
    s = zeros(N + 1)
    s[1] = a
    for i = 2:N+1
        s[i] = s[i-1] + 1 / s[i-1]
    end

    return s

end



# function run_obj_vs_N_sweep(
#     L::Real,
#     μ::Real,
#     R::Real,
#     p::Real;
#     Lip_scaling = 1.0,
#     data_path::String = "",
#     plot_path::String = "",
#     N_start::Int = 2,
#     N_stop::Int = 10,
#     N_step::Int = 1,
#     zero_idx = [],
#     show_trace::Bool = false,
#     obj_val_upper_bound::Real = 1e6,
#     optimize_kwargs...,
# )
#     N_vals = collect(N_start:N_step:N_stop)
#     isempty(N_vals) && error("Empty N range: N_start=$N_start, N_stop=$N_stop, N_step=$N_step")

#     obj_opts = Float64[]
#     true_BSD = Float64[]
#     theory_SGM = Float64[]
#     true_Lip = Float64[]
#     for N in N_vals
#         display("running N = " * string(N))
#         s_rec = s_recurrence(1, N)
#         s_last = s_rec[N+1]
#         h_star = 1/(s_last*sqrt(s_last^2 - 2*N))

#         h = h_star
#         α = make_α_gradient_descent(N, R/(Lip_scaling*L)* h)

#         obj_opt, _, _ = optimize_δ_for_fixed_α(
#             N, L, α, R, p, zero_idx;
#             show_trace = show_trace,
#             obj_val_upper_bound = obj_val_upper_bound,
#             optimize_kwargs...,
#         )
#         obj_opt_true_BS, _, _ = solve_primal_with_known_stepsizes_bounded_subgrad(N, L, α, R; show_output=:off)
#         obj_opt_true_Lip, _, _ = solve_primal_with_known_stepsizes_Lipschitz(N, Lip_scaling*L, α, R; show_output=:off)

#         push!(true_BSD, obj_opt_true_BS)
#         push!(true_Lip, obj_opt_true_Lip)
#         push!(obj_opts, obj_opt)
#         push!(theory_SGM, Lip_scaling*L*R*((1/2*s_last^2-N)*h + 1/(2(s_last^2*h))))
#     end

#     # exponent = (1 + 3 * p) / 2
#     # ref_line = (obj_opts[1] * N_vals[1]^exponent) ./ (N_vals .^ exponent)

#     # p_round = round(p, sigdigits = 6)
#     # plt = plot(
#     #     N_vals,
#     #     obj_opts;
#     #     label = "obj_opt",
#     #     yaxis = :log,
#     #     marker = :circle,
#     #     xlabel = "N",
#     #     ylabel = "objective",
#     #     title = "PEP optimum vs N (log y), p = $p_round",
#     #     legend = :topright,
#     # )
#     # plot!(
#     #     plt,
#     #     N_vals,
#     #     ref_line;
#     #     label = L"$\propto N^{-(1+3p)/2}$" * " (scaled)",
#     #     linestyle = :dash,
#     # )



#     return N_vals, true_Lip, theory_SGM, true_BSD, obj_opts
# end



function optimize_δ_for_fixed_α_simple(
    N, L, α, R, p, zero_idx;
    δ_lb=1e-6,
    δ_ub=1e6,
    max_sdp_calls=2000,
    ftol_rel=1e-6,
    ftol_abs=1e-8,
    xtol_rel=1e-6,
    show_trace=true,
    trace_interval=10,
    algorithm=:LN_SBPLX,
    show_output=:off,
    ϵ_tol_feas=1e-6,
    objective_type=:default,
    obj_val_upper_bound=1e6,
)
    M = 1
    I_N_star = -1:N
    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star, M)

    # all upper triangular pairs are free — no filtering, no active set
    idx_set_upper = filter(t -> t.i < t.j, idx_set_λ)
    n_vars = length(idx_set_upper)

    u_lb = log(δ_lb)
    u_ub = log(δ_ub)

    @info "Simple δ optimizer | algorithm=$algorithm | n_vars=$n_vars | δ∈[$δ_lb,$δ_ub]"

    # ── ε_set builder ─────────────────────────────────────────────────────────
    function make_δ_set_from_u(u)
        δ_set = make_δ_set(N, M, 0.0)
        for (k, t) in enumerate(idx_set_upper)
            δ_set[t.i, t.j, t.m] = exp(clamp(u[k], u_lb, u_ub))
        end
        return δ_set
    end

    # ── inner SDP — accept SLOW_PROGRESS as usable ────────────────────────────
    function eval_sdp(u)
        δ_set = make_δ_set_from_u(clamp.(u, u_lb, u_ub))

        val = try
            model_val, _, _ = solve_dual_PEP_with_known_stepsizes(
                N, L, α, R, δ_set, p, zero_idx;
                show_output=show_output,
                ϵ_tol_feas=ϵ_tol_feas,
                objective_type=objective_type,
                obj_val_upper_bound=obj_val_upper_bound,
                acceptable_termination_statuses=[MOI.OPTIMAL]
            )
            model_val
        catch err
            Inf
        end

        return isfinite(val) ? val : Inf
    end

    # ── tracking ──────────────────────────────────────────────────────────────
    call_count = Ref(0)
    best_val = Ref(Inf)
    best_u = zeros(n_vars)

    function nlopt_obj(u::Vector, grad::Vector)
        call_count[] += 1
        u_c = clamp.(u, u_lb, u_ub)
        val = eval_sdp(u_c)

        if val < best_val[]
            best_val[] = val
            best_u .= u_c
        end

        if show_trace && mod(call_count[], trace_interval) == 0
            @info "[call $(call_count[])] obj=$(round(val;sigdigits=5))  best=$(round(best_val[];sigdigits=5))" *
                  "  δ∈[$(round(exp(minimum(u_c));sigdigits=3)), $(round(exp(maximum(u_c));sigdigits=3))]"
        end

        return val
    end

    # ── NLopt setup ───────────────────────────────────────────────────────────
    opt = NLopt.Opt(algorithm, n_vars)
    opt.lower_bounds = fill(u_lb, n_vars)
    opt.upper_bounds = fill(u_ub, n_vars)
    opt.maxeval = max_sdp_calls
    opt.ftol_rel = ftol_rel
    opt.ftol_abs = ftol_abs
    opt.xtol_rel = xtol_rel
    opt.min_objective = nlopt_obj

    # ── multi-start optimization ──────────────────────────────────────────────
    n_restarts = 5
    global_best_val = Inf
    global_best_u = zeros(n_vars)

    for restart in 1:n_restarts
        u0 = restart == 1 ? zeros(n_vars) : randn(n_vars) .* 0.5
        u0 .= clamp.(u0, u_lb, u_ub)

        call_count[] = 0
        best_val[] = Inf
        best_u .= 0.0

        (obj_opt, u_opt_nlopt, ret) = NLopt.optimize(opt, u0)

        u_candidate = best_val[] <= obj_opt ? best_u : clamp.(u_opt_nlopt, u_lb, u_ub)
        obj_candidate = min(best_val[], obj_opt)

        @info "Restart $restart/$n_restarts: obj=$(round(obj_candidate;sigdigits=6)) ret=$ret calls=$(call_count[])"

        if obj_candidate < global_best_val
            global_best_val = obj_candidate
            global_best_u .= u_candidate
        end
    end

    u_final = global_best_u
    obj_final = global_best_val
    δ_set_opt = make_δ_set_from_u(u_final)

    @info """
    ✅ done | algorithm=$algorithm | restarts=$n_restarts | obj=$obj_final
    """
    @info "optimal δ values:" [(idx_set_upper[k].i, idx_set_upper[k].j,
        round(exp(u_final[k]); sigdigits=4))
                               for k in 1:n_vars]

    return obj_final, δ_set_opt
end

function run_obj_vs_N_sweep_optimal(
    R::Real,
    β::Real,
    p::Real,
    Lip_scaling::Real,
    make_α::Function;
    csv_path::String,
    N_start::Int=2,
    N_stop::Int=10,
    N_step::Int=1,
    zero_idx=[],
    show_trace::Bool=false,
    obj_val_upper_bound::Real=1e6,
    optimize_kwargs...,
)
    N_vals = collect(N_start:N_step:N_stop)
    isempty(N_vals) && error("Empty N range: N_start=$N_start, N_stop=$N_stop, N_step=$N_step")

    obj_opts = Float64[]
    true_BSD = Float64[]
    minimax_rate_true = Float64[]
    L = β * Lip_scaling
    for N in N_vals
        display("running N = " * string(N))
        α = make_α(N, L, R)

        if N < 1
            obj_opt, _ = optimize_δ_for_fixed_α_simple(
                N, β, α, R, p, zero_idx;
                show_trace=show_trace,
                obj_val_upper_bound=obj_val_upper_bound,
                optimize_kwargs...,
            )
        else
            obj_opt = 0.0
        end

        obj_opt_true_BS, _, _ = solve_primal_with_known_stepsizes_bounded_subgrad(N, β, α, R; show_output=:off)

        push!(obj_opts, obj_opt)
        push!(true_BSD, obj_opt_true_BS)
        push!(minimax_rate_true, L * R / sqrt(2*(N + 1)))
    end

    df = DataFrame(N=N_vals, delta_opt=obj_opts, bounded_subgrad=true_BSD, minimax_rate=minimax_rate_true)
    CSV.write(csv_path, df)

    return N_vals, obj_opts, true_BSD, minimax_rate_true
end






L = 1.0
β = L
μ = 0.0
R = 1.0
p = 0.0 # DO NOT CHANGE
N = 11
Lip_scaling = 1.0

α_type = make_α_optimal_SG_step
csv_path = string(α_type)* "_results_scaling_" * string(Lip_scaling) * ".csv"
N_vals, obj_opts, true_BSD, minimax_rate_true = run_obj_vs_N_sweep_optimal(R, β, p, Lip_scaling, α_type; csv_path)


# α = make_α_ssep(N, β, R)
# α = make_α_subgradient_avg(N, β, R)
# α = make_α_optimal_SG_step(N, β, R)

# obj_opt_true_Lip, _, _ = solve_primal_with_known_stepsizes_Lipschitz(N, Lip_scaling*L, α, R; show_output=:off)
# display(obj_opt_true_Lip-β*R/sqrt(N+1))




# display(minimum(true_Lip ./ true_BSD))
# display(maximum(true_Lip ./ true_BSD))


#display(true_Lip ./ theory_SGM)                

# # Single-N check (fast):
# N = 5
# α = make_α_gradient_descent(N, 2^(1-p)*L)
# obj_opt, δ_set_opt, result = optimize_δ_for_fixed_α(N, L, α, R, p, []; show_trace = false)
# obj, λ_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, δ_set_opt, p, [])