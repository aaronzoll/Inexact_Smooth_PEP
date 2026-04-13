using Convex
using LinearAlgebra
# Pick one solver:
# using SCS          # uncomment this and the optimizer below
using MosekTools  # if you have MOSEK

# --- Utilities / indexing (unchanged) ---
is_star_arc(s) = (s.i == -1)
is_nonstar_pair_to_j(s, j) = (s.j == j && s.i >= 0 && s.i < j)   # arcs (i,j) with 0≤i<j
is_nonstar_pair_from_j(s, j) = (s.i == j && s.j > j)              # arcs (j,i) with j<i≤N
is_step_arc(s, N) = (s.i >= 0) && (s.j == s.i + 1) && (s.j <= N)


csum(xs) = isempty(xs) ? 0 : sum(xs)

struct i_j_idx
    i::Int64
    j::Int64
end

function idx_set_λ_constructor(I_N_star)
    idx_set_λ = i_j_idx[]
    for i in I_N_star, j in I_N_star
        i != j && push!(idx_set_λ, i_j_idx(i, j))
    end
    return idx_set_λ
end

# --- Convex.jl version of your solver ---
function solve_fractional_convex(N, R, β, q;
    t_lower=1e-8, λ_lower=1e-8, s_lower=1e-8,
    show_output=:off, λ_sparsity=:OGM, t_sparsity=:OGM)

    @assert q > 0 "q must be in (0,1]; code supports q>0 (q=1 allowed)."
    I_N_star = -1:N
    idx = idx_set_λ_constructor(I_N_star)
    I = 0:N

    # exponents
    a = 1 - 1 / q      # can be ≤ 0 (convex-friendly via log-vars)
    b = 1 / q          # ≥ 1 if q∈(0,1], still fine in exp-of-affine
    c = β^(1 / q)

    # Variables: log-domain and scaling s
    # λ = exp(X[s]), t = exp(Y[s]), s ≥ s_lower
    X = Dict{i_j_idx, Variable}()
    Y = Dict{i_j_idx, Variable}()
    for s_ in idx
        X[s_] = Variable()    # log λ_{s_}
        Y[s_] = Variable()    # log t_{s_}
    end
    S = Variable()            # scaling variable (same role as your s)

    constraints = []

    # Lower bounds (λ ≥ λ_lower, t ≥ t_lower, s ≥ s_lower)
    for s_ in idx
        push!(constraints, X[s_] >= log(λ_lower))
        push!(constraints, Y[s_] >= log(t_lower))
    end
    push!(constraints, S >= s_lower)

    # Sparsity: for forbidden arcs, pin to the lower bound (like your upper=lower trick)
    allowed_idx = [s_ for s_ in idx if (is_star_arc(s_) || is_step_arc(s_, N))]
    forbidden_idx = setdiff(idx, allowed_idx)
    if λ_sparsity == :OGM
        for s_ in forbidden_idx
            push!(constraints, X[s_] <= log(λ_lower))  # together with ≥ becomes equality
        end
    end
    if t_sparsity == :OGM
        for s_ in forbidden_idx
            push!(constraints, Y[s_] <= log(t_lower))
        end
    end

    # Helper closures for exp’d variables
    λ(s_) = exp(X[s_])   # λ_{s_}
    t_(s_) = exp(Y[s_])  # t_{s_}

#     # (A) Flow-like balance to star, for j ≠ N
#   for j in I
#     j == N && continue
#     to_j   = [ -λ(s_) for s_ in idx if is_nonstar_pair_to_j(s_, j) ]
#     from_j = [  λ(s_) for s_ in idx if is_nonstar_pair_from_j(s_, j) ]
#     star_j = [  λ(s_) for s_ in idx if is_star_arc(s_) && s_.j == j     ]

#     lhs = csum(to_j) + csum(from_j)
#     rhs = csum(star_j)
#     push!(constraints, lhs == rhs)
# end

#     # (B) Terminal balance at N: Σ_{i=0}^{N-1} λ_{i,N} = Σ_{i=0}^{N-1} λ_{-1,i}
#     function safe_sum(v)
#         isempty(v) ? 0.0 : sum(v)
#     end
#     push!(constraints,
#         safe_sum([λ(s) for s in idx if (s.j == N) && (s.i >= 0) && (s.i <= N - 1)]) ==
#         safe_sum([λ(s) for s in idx if is_star_arc(s) && (s.j >= 0) && (s.j <= N - 1)])
#     )

#     # (C) t-inequalities (convex):  ... - t_{-1,j} + (Σ λ_{-1,j})^2 / s ≤ 0
#     # Use quad_over_lin([sumλ], S) to represent (sumλ)^2 / S.
# for j in I
#     to_j_t    = [ -t_(s_) for s_ in idx if is_nonstar_pair_to_j(s_, j) ]
#     from_j_t  = [ -t_(s_) for s_ in idx if is_nonstar_pair_from_j(s_, j) ]
#     star_j_t  = [ -t_(s_) for s_ in idx if is_star_arc(s_) && s_.j == j  ]
#     star_j_λ  = [  λ(s_)  for s_ in idx if is_star_arc(s_) && s_.j == j  ]

#     sum_to    = csum(to_j_t)
#     sum_from  = csum(from_j_t)
#     sum_star  = csum(star_j_t)
#     sumλstar  = csum(star_j_λ)

#     qol = quadoverlin(sumλstar, S)   # (sumλstar)^2 / S
#     push!(constraints, sum_to + sum_from + sum_star + qol <= 0)
# end

    # (D) Denominator normalization: sum_{j=0}^N λ_{-1,j} = 1
    const1 =  sum(λ(s_) for s_ in idx if is_star_arc(s_) && (s_.j in I)) == 1
    push!(constraints, const1)

    # Objective: 0.5*R^2*S + sum_s c * λ^a * t^b
    # Each λ^a * t^b = exp(a*X + b*Y)  (convex: exp of affine)
    terms = [ exp(a*X[s_] + b*Y[s_]) for s_ in idx ]
    obj = 0.5 * R^2 * S + c * sum(terms)

    problem = minimize(obj, constraints)

    # Choose solver
    #solver = SCS.Optimizer
     solver = Mosek.Optimizer  # if using MosekTools

    # Logging
    if show_output == :off
        solve!(problem, solver; silent_solver=true)
    else
        solve!(problem, solver)
    end

    # Extract numeric values
    Xv = Dict(s_ => evaluate(X[s_]) for s_ in idx)
    Yv = Dict(s_ => evaluate(Y[s_]) for s_ in idx)
    Sv = evaluate(S)

    λ_opt = Dict(s_ => exp(Xv[s_]) for s_ in idx)
    t_opt = Dict(s_ => exp(Yv[s_]) for s_ in idx)
    objv  = evaluate(obj)

    return objv, λ_opt, t_opt, Sv, problem
end

# --- Helpers to convert Dicts to your matrices, matching your original API ---
function get_λ_matrices_from_dict(λ_opt::Dict{i_j_idx,Float64}, N, TOL)
    λ_matrices = zeros(N + 2, N + 2)
    for i in -1:N, j in -1:N
        i == j && continue
        key = i_j_idx(i, j)
        if haskey(λ_opt, key) && λ_opt[key] > TOL
            λ_matrices[i+2, j+2] = λ_opt[key]
        end
    end
    return λ_matrices
end

# --- Drop-in test wrapper that mirrors your run_test behavior ---
function run_test_convex(N, R, β, q; sparsity_pattern=:OGM, test_type=:general, plotting=:off)

    L(δ) = β / (δ^q)
    L_inv(u) = (β / u)^(1 / q)

    obj, λ_opt, t_opt, s_opt, problem =
        solve_fractional_convex(N, R, β, q;
            λ_sparsity=sparsity_pattern, t_sparsity=sparsity_pattern, show_output=:off)

    if test_type == :subgrad_diff
        println("Minimax Optimal objective (theory) = ", sqrt(β) * R / sqrt(N + 1))
        println("Numerical objective (convex)       = ", obj)

        λ_mat = get_λ_matrices_from_dict(λ_opt, N, 1e-4)
        t_mat = get_λ_matrices_from_dict(t_opt, N, 1e-4)
        δ_mat = @. L_inv(λ_mat / (t_mat + (t_mat==0.0)*Inf))  # guard zeros

        δ_vals = [δ_mat[i, i+1] for i = 2:N+1]
        δ_vals_theo = zeros(N)
        for i = 1:N
            if isodd(i)
                δ_vals_theo[i] = sqrt(β) * R / sqrt(N + 1) * 1 / i
            end
        end
        println("Optimal δ set:")
        display(δ_vals)
        println("δ set different from theory: ", norm(δ_vals - δ_vals_theo))

        return obj, λ_mat, t_mat, δ_vals, norm(δ_vals - δ_vals_theo)
    end

    # General test branch
    println((q + 1)^((q - 1) / (q + 1)) / q^(q / (q + 1)) * β^(1 / (1 + q)) * R^(2 / (1 + q)) / (N + 1)^((2 - q) / (1 + q)))
    println("Numerical objective (convex) = ", obj)

    λ_mat = get_λ_matrices_from_dict(λ_opt, N, 1e-4)
    t_mat = get_λ_matrices_from_dict(t_opt, N, 1e-4)
    δ_mat = @. L_inv(λ_mat / (t_mat + (t_mat==0.0)*Inf))
    δ_vals = [δ_mat[i, i+1] for i = 2:N+1]
    println("Optimal δ set:")
    display(δ_vals)

    X = 1:N
    Y1 = δ_vals[X]
    Y2 = @. q * β^(q + 1) * R^(2 / (q + 1)) / (2^(q / (q + 1)) * (1 + q)^(2 / (q + 1)) * (N)^(1 / (q + 1))) * X^(-2 / (q + 1))
    println("δ set different from theory: ", norm(Y1 - Y2))

    return obj, λ_mat, t_mat, δ_vals, norm(Y1 - Y2)
end

# ---------------- Example call (mirrors yours) ----------------
N = 17
R = 3.0
β = 2.5
q = 1.0
sparsity_pattern = :OGM
# obj, λ_mat, t_mat, δ_vals, δ_diff =
#     run_test_convex(N, R, β, q; sparsity_pattern=:OGM, test_type=:subgrad_diff)
 solve_fractional_convex(N, R, β, q)