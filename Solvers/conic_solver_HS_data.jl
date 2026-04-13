using JuMP
using LinearAlgebra
using MosekTools
using MathOptInterface
using Plots
using OffsetArrays

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
function solve_fractional_conic(N, R, κ, q;
    t_lower=1e-12, λ_lower=1e-12, s_lower=1e-12,
    show_output=:off, λ_sparsity=:off, t_sparsity=:off)
    @assert 0 <= q ≤ 1 "q must be in (0,1]."
    I_N_star = -1:N
    I = 0:N
    idx = idx_set_λ_constructor(I_N_star)

    # Exponents/constants
    κpow = 1 / q              # κ = 1/q in earlier notation; here reserve κ as parameter
    b = κpow               # exponent on t
    a = 1 - b              # exponent on λ (≤ 0 for q<1; equals 0 for q=1)
    c = κ^(1 / q)            # front coefficient

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
        # q = 1: then a = 0, b = 1, z_s ≥ c * t_s = κ * t_s (since c = κ^(1/1) = κ)
        for s_ in idx
            @constraint(model, z[s_] ≥ κ * t[s_])
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

function coeff_vs_N(q;
    N_min::Int=5,
    N_max::Int=200,
    step::Int=5,
    R::Float64=1.0,
    κ::Float64=1.0)
    @assert 0 < q ≤ 1 "q must be in (0,1]."
    # ------------- Example run -------------
   

    display(string("Determining coefficients for q = ", q))
    display("---------------------------------")
    Ns = collect(N_min:step:N_max)
    coeffs = zeros(Float64, length(Ns))
    κ = 1
    R = 1
    L(δ) = κ / ((δ)^q)
    L_inv(u) = (κ / u)^(1 / q)

    for (k, N) in enumerate(Ns)
        res = solve_fractional_conic(N, R, κ, q; show_output=:off,
            λ_sparsity=:off, t_sparsity=:off)
        theoretical_obj = κ^(1 / (1 + q)) * R^(2 / (1 + q)) /
                          (N + 1)^((2 - q) / (1 + q))
        coeffs[k] = res.obj / theoretical_obj
        if N % 20 == 0
            display(string("Running for N = ", N, " out of ", N_max))
        end
    end

    display("done")

    plt = plot(Ns, coeffs;
        xlabel="N",
        ylabel="coeff",
        title="coeff vs N for q = $(q)",
        legend=false)

    return Ns, coeffs, plt
end

"""
    coeff_vs_q(N, R, κ; q_min=0.01, q_max=1.0, n_q=50)

For fixed N, R, κ, solve the fractional conic for a range of q from q_min to q_max
and plot the ratio (PEP objective / theoretical objective) vs q.
Returns (qs, coeffs, plt).
"""
function coeff_vs_q(N::Int, R::Float64, κ::Float64;
    q_min::Float64=0.01,
    q_max::Float64=1.0,
    n_q::Int=50)
    @assert 0 < q_min < q_max ≤ 1 "q_min and q_max must satisfy 0 < q_min < q_max ≤ 1."

    qs = range(q_min, q_max; length=n_q)
    coeffs = zeros(Float64, length(qs))

    display(string("Running for N = ", N))
    display(string("------------------"))
    for (k, q) in enumerate(qs)
        res = solve_fractional_conic(N, R, κ, q; show_output=:off,
            λ_sparsity=:off, t_sparsity=:off)
        theoretical_obj = κ^(1 / (1 + q)) * R^(2 / (1 + q)) /
                          (N + 1)^((2 - q) / (1 + q))
        coeffs[k] = res.obj / theoretical_obj
        if k % 2 == 0
            display(string("computing for q = ", q))
        end
    end

    plt = plot(collect(qs), coeffs;
        xlabel="q",
        ylabel="coeff",
        title="coefficients vs q (N = $N)",
        label="constructive approach",
        linewidth=2.5,
        legend=:bottomright)

    sparsity = 1
    theo_coeffs = [(q + 1)^((q - 1) / (q + 1)) / (q^(q / (q + 1))) for (k, q) in enumerate(qs) if k % sparsity == 0]
    qs_sparse = [q for (k, q) in enumerate(qs) if k % sparsity == 0]
    plot!(plt, qs_sparse, theo_coeffs; label="theoretical asymptotics", linewidth=2.5)
    #$\ \left(q+1\right)^{\frac{q-1}{q+1}}q^{\frac{-q}{q+1}}$
    return collect(qs), coeffs, plt
end

function obj_vs_N(q;
    N_min::Int=5,
    N_max::Int=200,
    step::Int=5,
    R::Float64=1.0,
    κ::Float64=1.0,
    save_csv::Bool=false,
    csv_path::AbstractString="obj_vs_N_q$(q).csv")

    @assert 0 < q ≤ 1 "q must be in (0,1]."

    display(string("Computing objective values for q = ", q))
    display("----------------------------------------")

    Ns = collect(N_min:step:N_max)
    objs = zeros(Float64, length(Ns))

    for (k, N) in enumerate(Ns)
        res = solve_fractional_conic(N, R, κ, q;
            show_output=:off,
            λ_sparsity=:off,
            t_sparsity=:off)
        objs[k] = res.obj
        if N % 20 == 0
            display(string("Running for N = ", N, " out of ", N_max))
        end
    end

    display("done")

    plt = plot(Ns, objs;
        xlabel="N",
        ylabel="obj",
        title="objective value vs N for q = $(q)",
        legend=false)

    if save_csv
        open(csv_path, "w") do io
            println(io, "N,obj")
            for (N, obj) in zip(Ns, objs)
                println(io, "$(N),$(obj)")
            end
        end
    end

    return Ns, objs, plt
end


N = 5
R = 1
κ = 1
q = 0.5




res = solve_fractional_conic(N, R, κ, q; show_output=:off, λ_sparsity=:off, t_sparsity=:off)
println("------------------------------------------------")
println(("status" => res.status, "obj" => res.obj, "s" => res.s_opt))
theo_obj = ((q + 1)^((q - 1) / (q + 1)) / (q^(q / (q + 1))) * κ^(1 / (1 + q)) * R^(2 / (1 + q)) /
(N + 1)^((2 - q) / (1 + q)))


# f = OffsetArray(zeros(N + 1), 0:N)
# for i = 0:N-1
#     f[i] = -res.duals_A[i] + res.obj
# end

# f[N] = -res.dual_B + res.obj

# G = OffsetArray(zeros(N + 1), 0:N)
# for i = 0:N
#     G[i] = -2 * res.duals_C[i][2]
# end

# g = OffsetArray(zeros(N + 1, N + 1), 1:N+1, 0:N)
# for i = 0:N
#     g[:, i] = sqrt(G[i]) * standard_basis(i + 1, N + 1)
# end

s = res.s_opt

λ_opt = Dict{i_j_idx,Float64}(k => 1 / s * res.λ_opt[k] for k in axes(res.λ_opt, 1))
t_opt = Dict{i_j_idx,Float64}(k => 1 / s * res.t_opt[k] for k in axes(res.t_opt, 1))

λ_mat = OffsetArray(get_λ_matrices(λ_opt, N, 1e-6), -1:N, -1:N)
t_mat = OffsetArray(get_λ_matrices(t_opt, N, 1e-6), -1:N, -1:N)


display(t_mat)
display(λ_mat)

display(res.obj)
display(theo_obj)

# qs, coeffs, plt = coeff_vs_q(100, 1.0, 1.0, q_min=1e-8,
#     q_max=1.0,
#     n_q=51)
# display(plt)


# csv_path = "constructive_coeffs_N_100.csv"
# open(csv_path, "w") do io
#     println(io, "q, coeff")
#     for (q, coeff) in zip(qs, coeffs)
#         println(io, "$(q),$(coeff)")
#     end
# end

# Ns, objs, plt = obj_vs_N(1.0,
#                          N_min = 10,
#                          N_max = 200,
#                          step = 5,
#                          save_csv = true,
#                          csv_path = "obj_vs_N_q1p0.csv")
# display(plt)




# X = OffsetArray(zeros(N + 1, N + 1), 1:N+1, 0:N)
# X[:, 0] = zeros(N + 1)
# z = OffsetArray(zeros(N + 1, N + 1), 1:N+1, 0:N)
# z[:, 1] = X[:, 0] - λ_mat[-1, 0] * g[:, 0]

# for j = 1:N
#     num = sum(λ_mat[i, j] * (X[:, i] .- (1 / L(δ_mat[i, j])) .* g[:, i]) for i = 0:j-1) + λ_mat[-1, j] * z[:, j]
#     denom = sum(λ_mat[i, j] for i = 0:j-1) + λ_mat[-1, j]

#     X[:, j] = num / denom             
#     if j < N
#         z[:, j+1] = z[:, j] .- λ_mat[-1, j] .* g[:, j] 
#     end
# end

# for i = 0:N 
#     for  j = 0 : N 
#         if i != j
#         display(minimum([f[i]-f[j] - g[:,j]' * (X[:,i]-X[:,j]) - 1/(2*L(δ))*(G[i]+G[j]) + δ for δ = LinRange(1e-6,4,300)]))
#         end
#     end
# end


# # CHECK!!!
# function drori_iterates_from_fG(f, G)
#     N = length(f)-1
#     zeta = OffsetArray(zeros(N+3), 0:N+2)
#     # reconstruct zeta from each pair (consistent across i)

#     for i in 0:N
#         zeta[i]   = f[i] + 0.5*G[i]
#         zeta[i+1]   = f[i] - 0.5*G[i]
#     end

#     # build x_i coefficients along u_0..u_{i-1}
#     X = OffsetArray(zeros(N+1,N+1), 1:N+1, 0:N)  # X[1] is x_0 = []
#     for i in 1:N  
#        X[:, i] = sum(-((zeta[j]-zeta[i+1])/sqrt(zeta[j]-zeta[j+1]))*standard_basis(j+1,N+1) for j = 0 : i-1)
#     end
#     return zeta, X
# end

# # Function to create Drori's hard instance from scratch using theta sequence from OGM
# function drori_hard_instance(N; β=1.0, R=1.0)
#     theta = OffsetArray(compute_theta(N), 0:N)

#     zeta = OffsetArray(zeros(N+3), 0:N+2)
#     zeta[N+2] = 0.0  # zeta[0] is typically 0 (x_0 at origin)
#     zeta[N+1] = (theta[N]-1)/((theta[N]^2)*(2*theta[N]-1))*R^2
#     zeta[N] = theta[N]/(theta[N]-1)*zeta[N+1]
#     for i in 1:N
#        zeta[N-i] = 2*theta[N-i]/(2*theta[N-i]-1)*zeta[N-i+1]
#     end

#     f = OffsetArray(zeros(N+1), 0:N)
#     G = OffsetArray(zeros(N+1), 0:N)
#     for i in 0:N
#         f[i] = (zeta[i] + zeta[i+1]) / 2
#         G[i] = zeta[i] - zeta[i+1]
#     end

#     X = OffsetArray(zeros(N+1, N+1), 1:N+1, 0:N)  # X[:, 0] is x_0 = 0
#     X_taylor = OffsetArray(zeros(N+1, N+1), 1:N+1, 0:N)
#     for i in 1:N  
#         X_taylor[:,i] = sum(-((zeta[j])/sqrt(zeta[j]-zeta[j+1]))*standard_basis(j+1,N+1) for j = 0 : i-1)
#         X[:, i] = sum(-((zeta[j]-zeta[i+1])/sqrt(zeta[j]-zeta[j+1]))*standard_basis(j+1,N+1) for j = 0 : i-1)
#     end

#     return X, X_taylor, f, G, zeta
# end

# # Helper function to compute theta sequence (same as in OGM.jl)
# function compute_theta(N)
#     θ = zeros(N+1)
#     θ[1] = 1.0  # θ₀
#     for i in 2:N
#         θ[i] = (1 + sqrt(1 + 4*θ[i-1]^2)) / 2
#     end
#     θ[N+1] = (1 + sqrt(1 + 8*θ[N]^2)) / 2  # θ_N
#     return θ
# end


# X_hard, X_hard_taylor, f_hard, G_hard, zeta_hard = drori_hard_instance(N)
# display(X_hard_taylor)
# display(f_hard)
# display(G_hard)


# zeta, X_driori = drori_iterates_from_fG(f, G)
# # display(X_driori)

# we expect coeff = \frac{\left(q+1\right)^{\frac{q-1}{q+1}}}{q^{\frac{q}{q+1}}}
# Ns, coeffs, plt = coeff_vs_N(0.5, N_min = 10, N_max = 200, step = 10)
# display(plt) 


