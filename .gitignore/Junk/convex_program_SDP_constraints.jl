using JuMP
using MosekTools
using LinearAlgebra, OffsetArrays

function e_i(n, i)
    e_i_vec = zeros(n, 1)
    e_i_vec[i] = 1
    return e_i_vec
end

# this symmetric outer product is used when a is constant, b is a JuMP variable
function ⊙(a,b)
    return ((a*b') .+ transpose(a*b')) ./ 2
end

# this symmetric outer product is for computing ⊙(a,a) where a is a JuMP variable
function ⊙(a)
    return a*transpose(a)
end



function data_generator_function(N, α, μ; input_type = :stepsize_constant)

    dim_𝐱 = N+2
    dim_𝐠 = N+2
    dim_𝐟 = N+1
    N_pts = N+2 # number of points corresponding to [x_⋆=x_{-1} x_0 ... x_N]

    𝐱_0 = e_i(dim_𝐱, 1)

    𝐱_star = zeros(dim_𝐱, 1)

    # initialize 𝐠 and 𝐟 vectors

    # 𝐠 = [𝐠_{-1}=𝐠_⋆ ∣ 𝐠_0 ∣ 𝐠_1 ∣... ∣ 𝐠_N]

    𝐠 =  OffsetArray(zeros(dim_𝐠, N_pts), 1:dim_𝐠, -1:N)

    # 𝐟  = [𝐟_{-1}=𝐟_⋆ ∣ 𝐟_0 ∣ 𝐟_1 ∣... ∣ 𝐟_N]

    𝐟 = OffsetArray(zeros(dim_𝐟, N_pts), 1:dim_𝐟, -1:N)

    # construct 𝐠 vectors, note that 𝐠_⋆  is already constructed zero

    for k in 0:N
        𝐠[:,k] = e_i(dim_𝐠, k+2)
    end

    # construct 𝐟 vectors, note that 𝐟_⋆ is already constructed zero

    for k in 0:N
        𝐟[:,k] = e_i(dim_𝐟, k+1)
    end

    # time to define the 𝐱 vectors, which requires more care

    if input_type == :stepsize_constant

        # 𝐱 = [𝐱_{⋆} = 𝐱{-1} ∣ 𝐱_0 ∣ 𝐱_1 ∣ ... ∣ 𝐱_N]

        𝐱 = OffsetArray(zeros(dim_𝐱, N_pts), 1:dim_𝐱, -1:N)

        # define 𝐱_0 which corresponds to x_0

        𝐱[:,0] = 𝐱_0

        # construct part of 𝐱 corresponding to the x iterates: x_1, ..., x_N

        for i in 1:N
            𝐱[:,i] = ( ( 1 - ( (μ)*(sum(α[i,j] for j in 0:i-1)) ) ) * 𝐱_0 ) - ( (1)*sum( α[i,j] * 𝐠[:,j] for j in 0:i-1) )
        end

    elseif input_type == :stepsize_variable

        # caution 💀: keep in mind that this matrix 𝐱 is not 0 indexed yet, so while constructing its elements, ensure to use the full formula for 𝐱_i

        𝐱 = [𝐱_star 𝐱_0]

        # construct part of 𝐱 corresponding to the x iterates: x_1, ..., x_N

        for i in 1:N
            𝐱_i = ( ( 1 - ( (μ)*(sum(α[i,j] for j in 0:i-1)) ) ) * 𝐱_0 ) - ( (1)*sum( α[i,j] * 𝐠[:,j] for j in 0:i-1) )
            𝐱 = [𝐱 𝐱_i]
        end

        # make 𝐱 an offset array to make our life comfortable

        𝐱 = OffsetArray(𝐱, 1:dim_𝐱, -1:N)
    end

    # time to return

    return 𝐱, 𝐠, 𝐟

end

function index_set_constructor_for_dual_vars_full(I_N_star)

    # construct the index set for λ
    idx_set_λ = i_j_idx[]
    for i in I_N_star
        for j in I_N_star
            if i!=j
                push!(idx_set_λ, i_j_idx(i,j))
            end
        end
    end

    return idx_set_λ

end

struct i_j_idx # correspond to (i,j) pair, where i,j ∈ I_N_⋆
    i::Int64 # corresponds to index i
    j::Int64 # corresponds to index j
end

A_mat(i,j,𝐠,𝐱) = ⊙(𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
B_mat(i,j,𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])
C_mat(i,j,𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
a_vec(i,j,𝐟) = 𝐟[:, j] - 𝐟[:, i]
"""
solve_convex_program_like_PEP(
    N, μ, L, α, R;
    show_output = :off,
    t_lower = 1e-8,                 # keep perspective well-posed
    ugrid = range(1e-6, stop=10.0, length=41),  # tangent grid for ψ
    ψ::Function,                    # ψ(u) = u * L^{-1}(u), convex & nondecreasing on u≥0
    dψ::Function,                   # subgradient/derivative of ψ on u≥0
    mosek_params = Dict{String,Any}(),
)

Assumes the following helpers are already defined in your environment:
- data_generator_function(N, α, μ, L; input_type=:stepsize_constant) -> (𝐱, 𝐠, 𝐟)
- idx_set_λ(I_N_star) -> index set (e.g., vector of structs with fields .i, .j)
- a_vec(i, j, 𝐟) -> vector (same length as a_target)
- A_mat(i, j, α, 𝐠, 𝐱) -> square matrix
- B_mat(i, j, α, 𝐱) -> square matrix
- C_mat(i, j, 𝐠) -> square matrix

Returns:
  obj_val, λ_opt (container), v_opt::Float64, t_opt (container), Z_opt::Matrix
"""
function solve_convex_program_like_PEP(
    N, μ, α, R;
    show_output = :off,
    t_lower::Float64 = 1e-8,
    ugrid = range(1e-6, stop=10.0, length=41),
    ψ::Function,
    dψ::Function,
    mosek_params = Dict{String,Any}(),
)

    # --- Data (you already have these helpers) ---
    𝐱, 𝐠, 𝐟 = data_generator_function(N, α, μ; input_type = :stepsize_constant)

    I_N_star = -1:N
    dim_Z    = N + 2               # matches your example style for matrix size

    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)

    # Precompute tangent cuts for ψ(u) = u*L^{-1}(u):  ψ(u) ≥ α u + β
    tangents = [(dψ(u), ψ(u) - dψ(u)*u) for u in ugrid]

    # --- Model ---
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    if show_output == :off
        set_silent(model)
    end
    for (k,v) in mosek_params
        set_optimizer_attribute(model, k, v)
    end

    # Variables
    @variable(model, λ[idx_set_λ] >= 0)
    @variable(model, t[idx_set_λ] >= t_lower)
    @variable(model, v >= 0)
    @variable(model, Z[1:dim_Z, 1:dim_Z], PSD)         # PSD block like your PEP code
    @variable(model, z[idx_set_λ] >= 0)                      # epigraph variables for ψ-perspective

    # Linear constraint:  Σ λ[i,j] a_vec(i,j,𝐟)  = a_target
    # We build it componentwise to match broadcast equality.
    # Let a_vec return a Vector{Float64} of length m; this creates m scalar constraints.

    @constraint(model,  sum(λ[s] * a_vec(s.i, s.j, 𝐟) for s in idx_set_λ) - a_vec(-1, N, 𝐟) .== 0
    )

    # LMI: v*B_mat(0,-1,α,𝐱) + Σ λ[i,j] A_mat(i,j,α,𝐠,𝐱) + (1/2) Σ t[i,j] C_mat(i,j,𝐠)  ==  Z
    # (Equality into the PSD variable Z, like your PEP pattern.)
    @constraint(model,
        v * B_mat(0, -1, 𝐱)
        + sum( λ[s] * A_mat(s.i, s.j, 𝐠, 𝐱) for s in idx_set_λ )
        + 0.5 * sum( t[s] * C_mat(s.i, s.j, 𝐠) for s in idx_set_λ )
        .== Z
    )

    # Perspective epigraph via cuts on each (i,j):
    # z[s] ≥ α_tan*λ[s] + β_tan*t[s]   for each tangent (α_tan, β_tan)
    for (α_tan, β_tan) in tangents
        @constraint(model, [s in idx_set_λ], z[s] >= α_tan * λ[s] + β_tan * t[s])
    end

    # Objective:  v*R^2 + Σ z[s]
    @objective(model, Min, v * R^2 + sum(z[s] for s in idx_set_λ))

    optimize!(model)

    term = termination_status(model)
    if term != MOI.OPTIMAL && term != MOI.LOCALLY_SOLVED
        @warn "Solver finished with status $term"
    end

    obj_val = objective_value(model)
    λ_opt   = value.(λ)
    v_opt   = value(v)
    t_opt   = value.(t)
    Z_opt   = value.(Z)

    return obj_val, λ_opt, v_opt, t_opt, Z_opt
end

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


function compute_α_from_h(h, N, μ, β)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for ℓ in 1:N
        for i in 0:ℓ-1
            if i==ℓ-1
                α[ℓ,i] = h[ℓ,ℓ-1]
            elseif i <= ℓ-2
                α[ℓ,i] = α[ℓ-1,i] + h[ℓ,i] - (μ/β)*sum(h[ℓ,j]*α[j,i] for j in i+1:ℓ-1)
            end
        end
    end
    return α
end

function compute_h_from_α(α, N, μ, β)
    h_new = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for l in N:-1:1
        h_new[l,l-1] = α[l,l-1]
        for i in l-2:-1:0
            h_new[l,i] = α[l,i] - α[l-1,i] + (μ/β)*sum(h_new[l,j]*α[j,i] for j in i+1:l-1)
        end
    end
    return h_new
end

function get_λ_matrices(λ_opt, N, TOL)
    λ_matrices = zeros(N + 2, N + 2)
        for i in -1:N
            for j in -1:N
                if i == j
                    continue
                end
                if λ_opt[i_j_idx(i,j)] > TOL
                λ_matrices[i+2,j+2] = λ_opt[i_j_idx(i,j)]
                end
            end
        end
    return λ_matrices
end

β = 1
μ = 0
N = 5
h = OffsetArray(compute_H(N), 1:N, 0:N-1)
α = compute_α_from_h(h, N, μ, β)
R = 1

using ForwardDiff

# Generic monotone inverse by bracketing + bisection
# Assumes L is (strictly) monotone on [δ_lo, δ_hi_max].
# Returns δ ≈ L^{-1}(u) for u in (L(δ_hi), L(δ_lo)).
function Linv_bisect(u; L::Function, δ_lo::Float64=0.0, δ_hi_max::Float64=1e6,
                     max_expand::Int=60, tol::Float64=1e-10, maxit::Int=200)
    # Determine whether L decreases or increases (typical: decreasing)
    L_lo = L(δ_lo)
    δ_hi = 1.0
    L_hi = L(δ_hi)

    # Expand δ_hi until u is bracketed between L(δ_lo) and L(δ_hi)
    # Works for either increasing or decreasing L.
    expanding = 0
    while expanding <= max_expand
        if (L_lo - u) * (L_hi - u) <= 0
            break
        end
        δ_hi = min(δ_hi * 2, δ_hi_max)
        L_hi = L(δ_hi)
        expanding += 1
    end
    if expanding > max_expand
        error("Failed to bracket u=$(u). Consider adjusting δ_hi_max or domain; got L(0)=$(L_lo), L($(δ_hi))=$(L_hi).")
    end

    # Bisection
    a, b = δ_lo, δ_hi
    La, Lb = L_lo, L_hi
    # Decide monotonicity once
    decreasing = Lb < La

    for _ in 1:maxit
        m = 0.5*(a+b)
        Lm = L(m)
        if abs(Lm - u) <= tol
            return m
        end
        if decreasing
            # L decreasing: u in [La, Lb] with La >= u >= Lb
            if Lm > u
                a, La = m, Lm
            else
                b, Lb = m, Lm
            end
        else
            # L increasing
            if Lm < u
                a, La = m, Lm
            else
                b, Lb = m, Lm
            end
        end
        if abs(b-a) <= tol
            return 0.5*(a+b)
        end
    end
    return 0.5*(a+b)
end

# ψ(u) = u * L^{-1}(u)
function psi_from_L(u; L::Function, δ_lo::Float64=0.0, δ_hi_max::Float64=1e6)
    if u <= 0
        return 0.0   # typical safe convention when lim_{u→0+} u*L^{-1}(u) = 0
    end
    δ = Linv_bisect(u; L=L, δ_lo=δ_lo, δ_hi_max=δ_hi_max)
    return u * δ
end

# ψ'(u) using inverse-function derivative and AD for L'(δ)
function dpsi_from_L(u; L::Function, δ_lo::Float64=0.0, δ_hi_max::Float64=1e6, eps_reg::Float64=1e-12)
    if u <= 0
        # right-derivative at 0 (if needed): use limiting slope ~ δ as u→0+; often 0
        return 0.0
    end
    δ = Linv_bisect(u; L=L, δ_lo=δ_lo, δ_hi_max=δ_hi_max)
    Lp = ForwardDiff.derivative(L, δ)
    # Guard if derivative is near zero (flat region); use a regularized denominator
    denom = abs(Lp) > eps_reg ? Lp : (Lp ≥ 0 ? eps_reg : -eps_reg)
    return δ + u/denom
end

# Build tangents α u + β for a grid of u's (for cutting-plane epigraph)
function psi_tangents_from_L(ugrid; L::Function, δ_lo::Float64=0.0, δ_hi_max::Float64=1e6)
    [(dpsi_from_L(u; L=L, δ_lo=δ_lo, δ_hi_max=δ_hi_max),
      psi_from_L(u; L=L, δ_lo=δ_lo, δ_hi_max=δ_hi_max) - dpsi_from_L(u; L=L, δ_lo=δ_lo, δ_hi_max=δ_hi_max)*u)
     for u in ugrid]
end

L(δ) = 1# your L(δ), δ ≥ 0

ψ(u)  = 1        # scalar function
dψ(u) = 0


obj_val, λ_opt, v_opt, t_opt, Z_opt = solve_convex_program_like_PEP(
    N, μ, h, R;
    show_output = :off,
    t_lower = 1e-8,
    ugrid = range(1e-6, stop=10.0, length=41),
    ψ,
    dψ,
    mosek_params = Dict{String,Any}(),
)

get_λ_matrices(λ_opt, N, 1, 0.0001)