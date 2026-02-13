using JuMP
using LinearAlgebra
using MosekTools
using MathOptInterface

# --- Utilities & indexing (unchanged) ---
is_star_arc(s) = (s.i == -1)
is_nonstar_pair_to_j(s, j) = (s.j == j && s.i >= 0 && s.i < j)
is_nonstar_pair_from_j(s, j) = (s.i == j && s.j > j)
is_step_arc(s, N) = (s.i >= 0) && (s.j == s.i + 1) && (s.j <= N)

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

# --- Conic solver for general q in (0,1] ---
function solve_fractional_conic(N, R, β, q;
    t_lower=1e-12, λ_lower=1e-12, s_lower=1e-12,
    show_output=:off, λ_sparsity=:OGM, t_sparsity=:OGM)

    @assert 0 <= q ≤ 1 "q must be in (0,1]."
    I_N_star = -1:N
    I = 0:N
    idx = idx_set_λ_constructor(I_N_star)

    # Exponents/constants
    βpow = 1 / q              # β = 1/q in earlier notation; here reserve β as parameter
    b = βpow               # exponent on t
    a = 1 - b              # exponent on λ (≤ 0 for q<1; equals 0 for q=1)
    c = β^(1 / q)            # front coefficient

    model = Model(Mosek.Optimizer)
    show_output == :off && set_silent(model)

    # Variables (original domain, positive)
    @variable(model, λ[s in idx] ≥ λ_lower)
    @variable(model, t[s in idx] ≥ t_lower)
    @variable(model, s ≥ s_lower) # each λ, t is scaled by s, hence the change to the objective

    # Epigraph vars for each arc term z_s ≥ c * λ^a * t^b
    @variable(model, z[s in idx] ≥ 0)

    # Sparsity: pin forbidden arcs to lower bounds
    allowed_idx = [s_ for s_ in idx if (is_star_arc(s_) || is_step_arc(s_, N))]
    forbidden_idx = setdiff(idx, allowed_idx)
    if λ_sparsity == :OGM
        for s_ in forbidden_idx
            @constraint(model, λ[s_] == λ_lower)
        end
    end
    if t_sparsity == :OGM
        for s_ in forbidden_idx
            @constraint(model, t[s_] == t_lower)
        end
    end

    # (A) Flow-like balance to star, for j ≠ N
    con_A = Dict{Int,ConstraintRef}()
    for j in I
        j == N && continue
        con_A[j] = @constraint(model,
            sum(-λ[s_] for s_ in idx if is_nonstar_pair_to_j(s_, j)) +
            sum(λ[s_] for s_ in idx if is_nonstar_pair_from_j(s_, j)) ==
            sum(λ[s_] for s_ in idx if is_star_arc(s_) && s_.j == j)
        )
    end

    # (B) Terminal balance at N
    con_B = @constraint(model,
        sum(λ[s] for s in idx if (s.j == N) && (s.i >= 0) && (s.i <= N - 1)) ==
        sum(λ[s] for s in idx if is_star_arc(s) && (s.j >= 0) && (s.j <= N - 1))
    )

    # (C) t-inequalities with (sum λ_star_j)^2 / s via SOC
    con_C = Dict{Int,Vector{ConstraintRef}}()
    @variable(model, r[j in I] >= 0)
    for j in I
        lin = @expression(model,
            sum(-t[s_] for s_ in idx if is_nonstar_pair_to_j(s_, j)) +
            sum(-t[s_] for s_ in idx if is_nonstar_pair_from_j(s_, j)) -
            sum(t[s_] for s_ in idx if is_star_arc(s_) && s_.j == j)
        )
        sumλstar = @expression(model, sum(λ[s_] for s_ in idx if is_star_arc(s_) && s_.j == j))
        # r_j will model (sumλstar)^2 / s
        # Standard SOC epigraph of x^2 / y:  ||[2x, y - r]||₂ ≤ y + r, with y≥0, r≥0


        c1 = @constraint(model, [s + r[j], 2 * sumλstar, s - r[j]] in SecondOrderCone())
        c2 = @constraint(model, lin + r[j] ≤ 0)

        con_C[j] = [c1, c2]
    end

    # (D) Denominator normalization
    con_D = @constraint(model, sum(λ[s_] for s_ in idx if is_star_arc(s_) && (s_.j in I)) == 1)

    # Power-cone links for z_s ≥ c * λ^a * t^b
    # For q ∈ (0,1): set α = 1/b = q ∈ (0,1). Use (x,y,z) ∈ PowerCone(α) meaning |z| ≤ x^α y^(1-α).
    # Put x = z_s, y = λ_s, z = c^(1/b) * t_s ⇒  t_s^b * c ≤ z_s * λ_s^(b-1)  ⇒ z_s ≥ c * λ^a t^b.
    # Furthermore, since a = 1-b, the extra "s" term is pulled out and this is the necessary epigraph
    # of the scaled z_s ≥ s * ̂λ^a ̂t^b
    if q < 1
        α = q  # α = 1/b
        K = MathOptInterface.PowerCone(α)
        for s_ in idx
            @constraint(model, [z[s_], λ[s_], c^(1 / b) * t[s_]] in K)
        end
    else
        # q = 1: then a = 0, b = 1, z_s ≥ c * t_s = β * t_s (since c = β^(1/1) = β)
        for s_ in idx
            @constraint(model, z[s_] ≥ β * t[s_])
        end
    end

    # Objective
    @objective(model, Min, 0.5 * R^2 * s + sum(z[s_] for s_ in idx))

    optimize!(model)

    # Report
    term = termination_status(model)
    λ_opt = value.(λ)
    t_opt = value.(t)
    s_opt = value(s)
    obj = objective_value(model)

    # Collect some duals (examples)
    duals_A = Dict(j => dual(con_A[j]) for j in keys(con_A))
    dual_B = dual(con_B)
    duals_C = Dict(j => (dual(con_C[j][1]), dual(con_C[j][2])) for j in keys(con_C))
    dual_D = dual(con_D)

    return (; status=term, obj, λ_opt, t_opt, s_opt, duals_A, dual_B, duals_C, dual_D)
end

# --- Your matrix helper (unchanged) ---
function get_λ_matrices(λ_opt::Dict{i_j_idx,Float64}, N, TOL)
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


function standard_basis(i, N)

    e = zeros(N)
    e[i] = 1
    return e

end

# ------------- Example run -------------
N = 7;
R = 1;
β = 1;
q = 0.001;
L(δ) = β / ((δ)^q)
L_inv(u) = (β / u)^(1 / q)

res = solve_fractional_conic(N, R, β, q; show_output=:off, λ_sparsity=:off, t_sparsity=:off)
println("------------------------------------------------")
println(("status" => res.status, "obj" => res.obj, "s" => res.s_opt))
theoretical_obj = (q + 1)^((q - 1) / (q + 1)) / q^(q / (q + 1)) * β^(1 / (1 + q)) * R^(2 / (1 + q)) / (N + 1)^((2 - q) / (1 + q))
display(theoretical_obj)
println("\n ------- f_i duals -------\n")
for i = 0:N-1
    println("f_", string(i), " = ", -res.duals_A[i] + res.obj)
end
println("f_", string(N), " = ", -res.dual_B + res.obj)
println("\n ------- g_i duals -------\n")
for i = 0:N
    println("||g_", string(i), "||^2 = ", -2 * res.duals_C[i][2])
end

using OffsetArrays

f = OffsetArray(zeros(N + 1), 0:N)
for i = 0:N-1
    f[i] = -res.duals_A[i] + res.obj
end

f[N] = -res.dual_B + res.obj

G = OffsetArray(zeros(N + 1), 0:N)
for i = 0:N
    G[i] = -2 * res.duals_C[i][2]
end

g = OffsetArray(zeros(N + 2, N + 1), 1:N+2, 0:N)
for i = 0:N
    g[:, i] = sqrt(G[i]) * standard_basis(i + 1, N + 2)
end

s = res.s_opt

λ_opt = Dict{i_j_idx,Float64}(k => 1/s*res.λ_opt[k] for k in axes(res.λ_opt, 1))
t_opt = Dict{i_j_idx,Float64}(k => 1/s*res.t_opt[k] for k in axes(res.t_opt, 1))

λ_mat = OffsetArray(get_λ_matrices(λ_opt, N, 1e-4), -1:N, -1:N)
t_mat = OffsetArray(get_λ_matrices(t_opt, N, 1e-4), -1:N, -1:N)
δ_mat = @. L_inv(λ_mat / t_mat)
δ_mat[.!isfinite.(δ_mat)] .= 0.0


X = OffsetArray(zeros(N + 2, N + 1), 1:N+2, 0:N)
X[:, 0] = zeros(N + 2)
z = OffsetArray(zeros(N + 2, N + 1), 1:N+2, 0:N)
z[:, 1] = X[:, 0] - λ_mat[-1, 0] * g[:, 0]

for j = 1:N
    num = sum(λ_mat[i, j] * (X[:, i] .- (1 / L(δ_mat[i, j])) .* g[:, i]) for i = 0:j-1) + λ_mat[-1, j] * z[:, j]
    denom = sum(λ_mat[i, j] for i = 0:j-1) + λ_mat[-1, j]

    X[:, j] = num / denom             
    if j < N
        z[:, j+1] = z[:, j] .- λ_mat[-1, j] .* g[:, j] 
    end
end

# for i = 0:N 
#     for  j = 0 : N 
#         if i != j
#         display(minimum([f[i]-f[j] - g[:,j]' * (X[:,i]-X[:,j]) - 1/(2*L(δ))*(G[i]+G[j]) + δ for δ = LinRange(1e-6,4,300)]))
#         end
#     end
# end


# CHECK!!!
function drori_iterates_from_fG(f, G)
    N = length(f)-1
    zeta = OffsetArray(zeros(N+3), 0:N+2)
    # reconstruct zeta from each pair (consistent across i)
 
    for i in 0:N
        zeta[i]   = f[i] + 0.5*G[i]
        zeta[i+1]   = f[i] - 0.5*G[i]
    end

    # build x_i coefficients along u_0..u_{i-1}
    X = OffsetArray(zeros(N+1,N+1), 1:N+1, 0:N)  # X[1] is x_0 = []
    for i in 1:N+1   
       X[:, i] = sum(-((zeta[j]-zeta[i+1])/sqrt(zeta[j]-zeta[j+1]))*standard_basis(j+1,N+1) for j = 0 : i-1)
    end
    return zeta, X
end


zeta, X_driori = drori_iterates_from_fG(f, G)
display(X_driori)