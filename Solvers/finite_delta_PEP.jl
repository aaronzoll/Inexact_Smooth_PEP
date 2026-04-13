include("../Solvers/BnB_PEP_Inexact_Smooth.jl")

using OffsetArrays
using LinearAlgebra, Random, Printf
using DelimitedFiles, Plots, LaTeXStrings

# Used by `optimize_ε_for_fixed_α` (keyword default) and sweeps below.

function compute_theta(N)
    θ = zeros(N+1)
    θ[1] = 1.0  # θ₀
    for i in 2:N
        θ[i] = (1 + sqrt(1 + 4*θ[i-1]^2)) / 2
    end
    θ[N+1] = (1 + sqrt(1 + 8*θ[N]^2)) / 2  # θ_N
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
        H[i, i] = 1+ (2*θ[i]-1) / θ[i+1]
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

function compute_α_from_h(h, N, μ, L)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for ℓ in 1:N
        for i in 0:ℓ-1
            if i==ℓ-1
                α[ℓ,i] = h[ℓ,ℓ-1]
            elseif i <= ℓ-2
                α[ℓ,i] = α[ℓ-1,i] + h[ℓ,i] - (μ/L)*sum(h[ℓ,j]*α[j,i] for j in i+1:ℓ-1)
            end
        end
    end
    return α
end

function compute_h_from_α(α, N, μ, L)
    h_new = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for l in N:-1:1
        h_new[l,l-1] = α[l,l-1]
        for i in l-2:-1:0
            h_new[l,i] = α[l,i] - α[l-1,i] + (μ/L)*sum(h_new[l,j]*α[j,i] for j in i+1:l-1)
        end
    end
    return h_new
end



# ─────────────────────────────────────────────────────────────────────────────
#  α-matrix constructors  (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────

function make_α_gradient_descent(N, L)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for i in 1:N, j in 0:i-1
        α[i, j] = 1.0/L
    end
    return α
end


function make_α_nesterov(N, L)
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


function make_α_ogm(N, L)
    θ = compute_theta(N)
    H = OffsetArray(compute_H(N), 1:N, 0:N-1)
    return compute_α_from_h(H, N, 0.0, L)
end


function verify_nesterov(N, L; d = 3, seed = 42)
    Random.seed!(seed)

    x0 = randn(d)
    g  = [randn(d) for _ in 0:N-1]

    t = ones(N + 2)
    for k in 2:N+2
        t[k] = (1 + sqrt(1 + 4 * t[k-1]^2)) / 2
    end
    β(k) = (t[k] - 1) / t[k+1]

    nes_query = [zeros(d) for _ in 0:N]
    grad_step = [zeros(d) for _ in 0:N-1]

    nes_query[1] = x0

    for k in 0:N-1
        grad_step[k+1] = nes_query[k+1] - (1/L) .* g[k+1]

        if k < N-1
            if k == 0
                nes_query[2] = grad_step[1]
            else
                βk = β(k+1)
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

    println("=" ^ 60)
    println("Nesterov verification   N=$N   L=$L   d=$d")
    println("=" ^ 60)

    all_match = true
    for k in 0:N-1
        err = norm(nes_query[k+1] .- fsfom_query[k+1], Inf)
        ok  = err < 1e-10
        all_match &= ok
        match_str = ok ? "✓" : "✗  ← MISMATCH"
        println("  x_$k \t ‖error‖∞ = $(round(err; sigdigits=3)) \t $match_str")
    end

    err_out = norm(nes_output .- fsfom_output, Inf)
    ok_out  = err_out < 1e-10
    all_match &= ok_out
    match_str = ok_out ? "✓" : "✗  ← MISMATCH"
    println("  x_$N \t ‖error‖∞ = $(round(err_out; sigdigits=3)) \t $match_str  (output)")

    println("=" ^ 60)
    println(all_match ? "  ✅ All points match." : "  ❌ Mismatch detected.")
    println("=" ^ 60)

    return all_match, nes_output, fsfom_output
end


# ─────────────────────────────────────────────────────────────────────────────
#  ε_set helpers
#
#  ε_set is a 3-D OffsetArray:  ε_set[i, j, m]
#    i, j ∈ I_N_star  =  -1:N      (the usual PEP index set)
#    m    ∈ 1:M
#
#  Only entries with i ≠ j are ever read; the diagonal is unused.
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_ε_set(N, M, value = 1.0) -> OffsetArray{Float64, 3}

Allocate an ε_set tensor of shape (-1:N, -1:N, 1:M) filled with `value`.
"""
function make_ε_set(N, M, value = 1.0)
    n   = N + 2                          # length of -1:N
    raw = fill(value, n, n, M)
    return OffsetArray(raw, -1:N, -1:N, 1:M)
end

"""
    pack_ε(ε_set, idx_set_λ) -> Vector{Float64}

Flatten the entries of ε_set that are actually used (one per (i,j,m) triple
in idx_set_λ) into a plain vector, in the same order as idx_set_λ.
The log transform is applied so the optimisation variable is unconstrained.
"""
function pack_ε(ε_set, idx_set_λ)
    return [log(ε_set[t.i, t.j, t.m]) for t in idx_set_λ]
end

"""
    unpack_ε(u, idx_set_λ, N, M) -> OffsetArray{Float64, 3}

Inverse of pack_ε: reconstruct a full ε_set tensor from a flat vector `u`
(which lives in unconstrained log-space).
"""
function unpack_ε(u, idx_set_λ, N, M)
    ε_set = make_ε_set(N, M, 1.0)
    for (k, t) in enumerate(idx_set_λ)
        ε_set[t.i, t.j, t.m] = exp(u[k])
    end
    return ε_set
end



# ─────────────────────────────────────────────────────────────────────────────
#  ε-only outer optimizer  (α fixed)
# ─────────────────────────────────────────────────────────────────────────────

using NLopt

function optimize_ε_for_fixed_α(
    N, L, α, R, p, zero_idx;
    ε_init          = nothing,
    ε_lb            = 1e-8,
    ε_ub            = 1e4,
    max_sdp_calls   = 1000,
    ftol_rel        = 1e-6,
    ftol_abs        = 1e-8,
    xtol_rel        = 1e-6,
    show_trace      = true,
    trace_interval  = 20,
    algorithm       = :LN_BOBYQA,   # or :LN_SBPLX, :LN_COBYLA, :LN_NELDERMEAD
    show_output          = :off,
    ϵ_tol_feas           = 1e-8,
    objective_type       = :default,
    obj_val_upper_bound  = 1e6,
)
    M         = 1
    I_N_star  = -1:N
    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star, M)

    # ── active (i,j) pairs ────────────────────────────────────────────────────
    function is_active(i, j)
        is_consecutive = (j == i + 1) && (i in -1:N-1)
        is_star_to_all = (i == -1)    && (j in 0:N)
        return is_consecutive || is_star_to_all
    end

    idx_set_λ_lower          = filter(t -> t.i > t.j, idx_set_λ)
    idx_set_λ_active         = filter(t -> t.i < t.j && is_active(t.i, t.j), idx_set_λ)
    idx_set_λ_inactive_upper = filter(t -> t.i < t.j && !is_active(t.i, t.j), idx_set_λ)
    n_vars = length(idx_set_λ_active)

    # ── augmented zero_idx ────────────────────────────────────────────────────
    lower_tri_zero      = [(t.i, t.j, t.m) for t in idx_set_λ_lower]
    inactive_upper_zero = [(t.i, t.j, t.m) for t in idx_set_λ_inactive_upper]
    if p == 1.0
        zero_idx_aug        = vcat(zero_idx, lower_tri_zero, inactive_upper_zero)
    else 
        zero_idx_aug        = vcat(zero_idx)
    end
    @info "Algorithm: $algorithm  |  Active ε variables: $n_vars"
    @info "  consecutive (i,i+1) : $(count(t -> t.j == t.i+1, idx_set_λ_active))"
    @info "  star-to-all (-1,i)  : $(count(t -> t.i == -1,    idx_set_λ_active))"
    @info "  ε ∈ [$ε_lb, $ε_ub]"

    # ── ε_set builder ─────────────────────────────────────────────────────────
    function make_ε_set_from_ε_vec(ε_vec)
        ε_set = make_ε_set(N, M, ε_lb)
        for (k, t) in enumerate(idx_set_λ_active)
            ε_set[t.i, t.j, t.m] = ε_vec[k]
        end
        return ε_set
    end

    # ── tracking ──────────────────────────────────────────────────────────────
    call_count = Ref(0)
    best_val   = Ref(Inf)
    best_ε_vec = ε_init === nothing ? ones(n_vars) :
                 clamp.([ε_init[t.i, t.j, t.m] for t in idx_set_λ_active], ε_lb, ε_ub)

    # ── NLopt objective (must match signature (x, grad) -> f) ─────────────────
    function nlopt_objective(ε_vec::Vector, grad::Vector)
        # grad is always empty for derivative-free methods — NLopt still passes it
        @assert isempty(grad) "Gradient unexpectedly requested — use a derivative-free algorithm"

        call_count[] += 1
        ε_set = make_ε_set_from_ε_vec(ε_vec)

        val = try
            v, _, _ = solve_dual_PEP_with_known_stepsizes(
                N, L, α, R, ε_set, p, zero_idx_aug;
                show_output         = :off,
                ϵ_tol_feas          = ϵ_tol_feas,
                objective_type      = objective_type,
                obj_val_upper_bound = obj_val_upper_bound
            )
            v
        catch err
            @warn "Inner SDP failed at call $(call_count[])" err
            return Inf   # NLopt treats Inf as a failed evaluation and steps away
        end

        if val < best_val[]
            best_val[]   = val
            best_ε_vec  .= ε_vec
        end

        if show_trace && mod(call_count[], trace_interval) == 0
            @info "  [call $(call_count[])]" *
                  "  obj = $(round(val; sigdigits=6))" *
                  "  best = $(round(best_val[]; sigdigits=6))" *
                  "  ε ∈ [$(round(minimum(ε_vec); sigdigits=3))," *
                  " $(round(maximum(ε_vec); sigdigits=3))]"
        end

        return val
    end

    # ── build NLopt optimizer ─────────────────────────────────────────────────
    opt = NLopt.Opt(algorithm, n_vars)

    opt.lower_bounds  = fill(ε_lb, n_vars)
    opt.upper_bounds  = fill(ε_ub, n_vars)
    opt.maxeval       = max_sdp_calls
    opt.ftol_rel      = ftol_rel
    opt.ftol_abs      = ftol_abs
    opt.xtol_rel      = xtol_rel
    opt.min_objective = nlopt_objective

    # ── initial point ─────────────────────────────────────────────────────────
    u0 = ε_init === nothing ? ones(n_vars) :
         clamp.([ε_init[t.i, t.j, t.m] for t in idx_set_λ_active], ε_lb, ε_ub)

    @info "Initial objective: $(best_val[])"

    # ── optimize ──────────────────────────────────────────────────────────────
    (obj_opt, ε_opt_vec, ret) = NLopt.optimize(opt, u0)

    # use best seen rather than final iterate in case NLopt ends on a bad step
    if best_val[] < obj_opt
        ε_opt_vec = copy(best_ε_vec)
        obj_opt   = best_val[]
    end

    ε_set_opt    = make_ε_set_from_ε_vec(ε_opt_vec)
    active_ε_opt = [(idx_set_λ_active[k].i, idx_set_λ_active[k].j,
                     round(ε_opt_vec[k]; sigdigits=5))
                    for k in 1:n_vars]

    @info """
    ✅  ε optimisation complete
        Algorithm      : $algorithm
        Return code    : $ret
        Active pairs   : $n_vars
        SDP calls      : $(call_count[])
        ε bounds       : [$ε_lb, $ε_ub]
        Objective      : $obj_opt
    """
    @info "Optimal ε (i, j, ε):" active_ε_opt

    return obj_opt, ε_set_opt, zero_idx_aug
end


"""
    run_obj_vs_N_sweep(L, μ, R, p; data_path, plot_path, N_start=5, N_stop=100, N_step=5, ...)

For each `N` in `N_start:N_step:N_stop`, run `optimize_ε_for_fixed_α` with
`α = make_α_gradient_descent(N, L)`, write `(N, obj_opt)` to `data_path`, and
save a log-y figure comparing `obj_opt` to `∝ N^{-(1+3p)/2}` scaled to match
the first `N` (so both curves coincide at the left endpoint).

Pass `data_path` and `plot_path` as the exact paths you want; include `p` in the
filename with string concatenation or interpolation (e.g. `*` and `string(p)`).
Parent directories are not created automatically.

`μ` is included for a fixed parameter tuple; it is not used by the current
gradient-descent `α` constructor.

Returns `(N_vals, obj_opts)`.
"""
function run_obj_vs_N_sweep(
    L::Real,
    μ::Real,
    R::Real,
    p::Real;
    algo = "GD",
    data_path::String,
    plot_path::String,
    N_start::Int = 5,
    N_stop::Int = 50,
    N_step::Int = 5,
    zero_idx = [],
    show_trace::Bool = false,
    obj_val_upper_bound::Real = 1e6,
    optimize_kwargs...,
)
    N_vals = collect(N_start:N_step:N_stop)
    isempty(N_vals) && error("Empty N range: N_start=$N_start, N_stop=$N_stop, N_step=$N_step")

    obj_opts = Float64[]
    for N in N_vals
        if algo == "GD"
            α = make_α_gradient_descent(N, L)
        elseif algo == "OGM"
            α = make_α_ogm(N, L)
        elseif algo == "NAG"
            α = make_α_nesterov(N, L)
        else
            @error "Use on of constructed algorithms: GD, OGM, NAG"
        end
        obj_opt, _, _ = optimize_ε_for_fixed_α(
            N, L, α, R, p, zero_idx;
            show_trace = show_trace,
            obj_val_upper_bound = obj_val_upper_bound,
            optimize_kwargs...,
        )
        push!(obj_opts, obj_opt)
    end

    open(data_path, "w") do io
        println(io, "N,obj_opt")
        for (n, o) in zip(N_vals, obj_opts)
            println(io, "$n,$o")
        end
    end
    @info "Saved sweep data" data_path

    exponent = (1 + 3 * p) / 2
    ref_line = (obj_opts[1] * N_vals[1]^exponent) ./ (N_vals .^ exponent)

    p_round = round(p, sigdigits = 6)
    plt = plot(
        N_vals,
        obj_opts;
        label = "obj_opt",
        yaxis = :log,
        marker = :circle,
        xlabel = "N",
        ylabel = "objective",
        title = "PEP optimum vs N (log y), p = $p_round",
        legend = :topright,
    )
    plot!(
        plt,
        N_vals,
        ref_line;
        label = L"$\propto N^{-(1+3p)/2}$" * " (scaled)",
        linestyle = :dash,
    )
    savefig(plt, plot_path)
    @info "Saved plot" plot_path

    return N_vals, obj_opts
end


"""
    plot_obj_vs_N_from_csv(csv_path, p, plot_path)

Load `(N, obj_opt)` from a CSV written by `run_obj_vs_N_sweep` and save the
log-y comparison plot to `plot_path` (reference curve scaled to the first row).
"""
function plot_obj_vs_N_from_csv(csv_path::String, p::Real, plot_path::String)
    M = readdlm(csv_path, ','; skipstart = 1)
    N_vals = Int.(M[:, 1])
    obj_opts = Float64.(M[:, 2])

    exponent = (1 + 3 * p) / 2
    ref_line = (obj_opts[1] * N_vals[1]^exponent) ./ (N_vals .^ exponent)

    p_round = round(p, sigdigits = 6)
    plt = plot(
        N_vals,
        obj_opts;
        label = "obj_opt",
        yaxis = :log,
        marker = :circle,
        xlabel = "N",
        ylabel = "objective",
        title = "PEP optimum vs N (log y), p = $p_round",
        legend = :topright,
    )
    plot!(
        plt,
        N_vals,
        ref_line;
        label = L"$\propto N^{-(1+3p)/2}$" * " (scaled)",
        linestyle = :dash,
    )
    savefig(plt, plot_path)
    @info "Saved plot" plot_path
    return plot_path
end


L = 1.0
μ = 0.0
R = 1.0
p = 0.5


run_obj_vs_N_sweep(L, μ, R, p; data_path="finite_delta_obj_vs_N_p_" * string(p) * ".csv",
                               plot_path="finite_delta_obj_vs_N_p_" * string(p) * ".png")

# # Single-N check (fast):
# N = 5
# α = make_α_gradient_descent(N, 2^(1-p)*L)
# obj_opt, ε_set_opt, result = optimize_ε_for_fixed_α(N, L, α, R, p, []; show_trace = false)
# obj, λ_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, ε_set_opt, p, [])