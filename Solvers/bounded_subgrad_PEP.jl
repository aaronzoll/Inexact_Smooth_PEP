using OffsetArrays
using LinearAlgebra
include("BnB_PEP_Inexact_Smooth.jl")
using OffsetArrays

"""
    compute_lambda_t(N, R, β; free_λ=nothing) -> (λ, t)

Construct the λ and t matrices from the dual-generated algorithm constraints.

Free variables are λ[i,j] for 0 ≤ i ≤ j-2 (i.e., j ≥ i+2), defaulting to 0.

Arguments:
- `N`:       number of iterations
- `R`, `β`:  problem parameters
- `free_λ`:  optional OffsetArray with free entries λ[i,j], j ≥ i+2.
             Must support indexing on (0:N-2, 2:N). Defaults to all zeros.

Returns:
- `λ`: OffsetArray, row indices -1:N-1, col indices 1:N
         row -1 = ★ entries; λ[i,j]=0 for i≥j (lower triangle)
- `t`: OffsetArray, row indices 0:N-1, col indices 1:N
         t[i,j] = 2R²/(β²N(N+1)) for i < j, else 0
"""
function compute_lambda_t(N, R, β; free_λ=nothing)
    c     = R * sqrt(2) / (β * sqrt(N + 1))        # λ_{★,k} constant
    t_val = 2R^2 / (β^2 * N * (N + 1))             # t_{i,j} constant

    # --- λ: rows -1:N-1, cols 1:N (initialised to zero) ---
    λ = OffsetArray(zeros(N + 2, N+2), -1:N, -1:N)

    # Constraint 1: λ_{★,k} = c for all k
    for k in 0:N
        λ[-1, k] = c
    end

    # Free variables: λ[i,j] for j ≥ i+2
    if free_λ !== nothing
        for j in 2:N, i in 0:j-2
            λ[i, j] = free_λ[i, j]
        end
    end

    # Constraint 2: λ[0,1] = c - Σ_{k=2}^N λ[0,k]   (free vars only)
    λ[0, 1] = c - sum(λ[0, k] for k in 2:N; init=0.0)

    # Constraint 3: λ[j-1,j] = c + λ[j-2,j-1]
    #                           + Σ_{ℓ=0}^{j-2}  λ[ℓ,j]    (free: col j, above sub-diag)
    #                           - Σ_{ℓ=j+2}^{N}  λ[j,ℓ]    (free: row j, ≥2 past diag)
    for j in 1:N-1
        sum_col = sum(λ[ℓ, j] for ℓ in 0:j-2;  init=0.0)   # free entries in col j
        sum_row = sum(λ[j, ℓ] for ℓ in j+2:N;  init=0.0)   # free entries in row j
        λ[j, j+1] = c + λ[j-1, j] + sum_col - sum_row
    end

    # --- t: rows 0:N-1, cols 1:N ---
    t = OffsetArray(zeros(N+2, N+2), -1:N, -1:N)
    for j in 1:N, i in 0:j-1
        t[i, j] = t_val
    end

    return λ, t
end

function test_constraints(λ_mat, t_mat, N)

    const_1 = zeros(N)
    const_2 = 0
    const_3 = zeros(N+1)

    for j = 0:N-1
        const_1[j+1] = sum([-λ_mat[i, j] for i = 0:j-1]) + sum([λ_mat[j, i] for i = j+1:N]) - λ_mat[-1, j]
    end

    const_2 = -sum([λ_mat[-1,i] for i = 0:N-1])+sum([λ_mat[i,N] for i = 0:N-1])

    for j = 0:N
        const_3[j+1] = sum([-t_mat[i, j] for i = 0:j-1]) + sum([-t_mat[j, i] for i = j+1:N]) - t_mat[-1, j] + λ_mat[-1,j]^2
    end

    return const_1, const_2, const_3
end

function build_H(N::Int, R::Real, β::Real)
    α = sqrt(2) * R * sqrt(N + 1) / (β * N)
    H = zeros(N, N)
    for k in 1:N          # row  (iterate index)
        for i in 0:k-1    # col  (gradient index, 1-based: i+1)
            H[k, i+1] = α * (k - i) / (k + 1)
        end
    end
    return H
end


"""
    compute_H(λ, t, N) -> H

Compute the H matrix such that xₖ = x₀ - Σᵢ H[k,i] gᵢ, equivalent to the
dual-generated algorithm defined by λ and t.

Arguments:
- `λ`: OffsetArray with λ[i,k] for i ∈ -1:k-1, k ∈ 1:N  (index -1 = ★)
- `t`: OffsetArray with t[j,k] for j ∈ 0:k-1, k ∈ 1:N
- `N`: number of iterations

Returns:
- `H`: OffsetArray with H[k,j] for k ∈ 1:N, j ∈ 0:k-1
"""
function compute_H(λ, t, N)
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)

    for k in 1:N
        σ_k = λ[-1, k] + sum(λ[i, k] for i in 0:k-1; init = 0.0)

        for j in 0:k-1
            # Contribution from previous iterates x_i (i = j+1, ..., k-1)
            inner = sum(λ[i, k] * H[i, j] for i in j+1:k-1; init = 0.0)

            # Direct step-size term + z-sequence term
            H[k, j] = (t[j, k] + inner + λ[-1, k] * λ[-1, j]) / σ_k
        end
    end

    return H
end

# Run the original recursion
function run_recursion(N, R, β, x0, λ_guess, t_guess)
    d = N + 1 
    G = OffsetArray(Diagonal(ones(N + 1)), 1:d, 0:N)

    X3 = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    X3[:, 0] .= x0
    z = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    z[:, 1] = X3[:, 0] - λ_guess[-1, 0] * G[:, 0]

    for j = 1:N
        num = sum(λ_guess[i, j] .* X3[:, i] .- t_guess[i, j] .* G[:, i] for i = 0:j-1) + λ_guess[-1, j] .* z[:, j]
        denom = sum(λ_guess[i, j] for i = 0:j-1) + λ_guess[-1, j]

        X3[:, j] = num / denom
        if j < N
            z[:, j+1] = z[:, j] .- λ_guess[-1, j] .* G[:, j]
        end
    end

    X4 = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    y = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    X4[:, 0] .= x0

    d_vec = zeros(d)  # d_k = sum_{i=0}^{k-1} g_i

    for k = 1:N
        d_vec .+= G[:, k-1]  # accumulate g_{k-1}
        X4[:, k] = (k .* X4[:, k-1] .+ x0) ./ (k + 1) .- (sqrt(2) * R * sqrt(N + 1)) / (β * N * (k + 1)) .* d_vec
    end

    return X3, X4
end

function run_fom1(x0::AbstractVector, g::AbstractMatrix, H, N::Int)
    d = length(x0)
    X = OffsetArray(zeros(eltype(x0), d, N + 1), 1:d, 0:N)
    X[:, 0] .= x0
    for j in 1:N
        X[:, j] .= x0
        for i in 0:j-1
            X[:, j] .-= H[j, i] .* g[:, i]
        end
    end
    return X
end


N = 9
R = 2 
β = 3
x0 = zeros(N+1)
# Equivalently via H (scalar / 1-D case):
H1 = OffsetArray(build_H(N, R, β), 1:N, 0:N-1)
d = N + 1
G = OffsetArray(Diagonal(ones(N + 1)), 1:d, 0:N)


Js = -1:N
t_guess = OffsetArray(zeros(length(Js), length(Js)), Js, Js)
λ_guess = OffsetArray(zeros(length(Js), length(Js)), Js, Js)
for j in 0:N
    λ_guess[-1, j] = sqrt(2) * R / (β * sqrt((N + 1)))
end
# Off-diagonal: arcs (i, i+1), i = 0..N-1
for i in 0:(N-1)
    λ_guess[i, i+1] = sqrt(2) * R * (i + 1) / (β * sqrt((N + 1)))
end
for i in 0:(N-1)
    for j in i+1:N
        t_guess[i, j] = 2 * R^2 / (β^2 * (N) * (N + 1))
    end
end
for j in 0:N
    t_guess[-1, j] =  0
end

# constructable λ and therefore H matrix with free variables
λ_free = OffsetArray(zeros(N + 2, N+2), -1:N, -1:N)
λ_free[1,3] = 0.2
λ_free[1,4] = 0.3
λ_free[2,5] = 0.1
λ_free[2,7] = 0.4
λ_free[5,7] = 0.3
λ_const, t_const = compute_lambda_t(N, R, β, free_λ = λ_free)
H2 = OffsetArray(compute_H(λ_const, t_const, N), 1:N, 0:N-1)


X1 = run_fom1(x0, G, H1, N)
X2, X3 = run_recursion(N, R, β, x0, λ_guess, t_guess)

c1, c2, c3 = test_constraints(λ_const, t_const, N)


 p_star, G_star, Ft_star=solve_primal_with_known_stepsizes_bounded_subgrad(N, β, H2, R)
 display(p_star)
 display(β*R/(sqrt(2*(N+1))))