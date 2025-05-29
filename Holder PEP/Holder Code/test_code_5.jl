using Revise, OffsetArrays
include("BnB_PEP_Inexact_Smooth.jl")

struct OptimizationProblem
    N::Int
    L::Float64
    R::Float64
    p::Float64
    k::Int
end


function get_H(N, M, args)
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    cnt = 0
    for i in 1:N
        for j in 0:i-1
            H[i, j] = args[M+1+cnt]
            cnt += 1
        end
    end
    return H
end


function compute_α_from_h(prob::OptimizationProblem, h, μ)
    N, L = prob.N, prob.L
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for ℓ in 1:N
        for i in 0:ℓ-1
            if i == ℓ - 1
                α[ℓ, i] = h[ℓ, ℓ-1]
            elseif i <= ℓ - 2
                α[ℓ, i] = α[ℓ-1, i] + h[ℓ, i] - (μ / L) * sum(h[ℓ, j] * α[j, i] for j in i+1:ℓ-1)
            end
        end
    end
    return α
end


function get_λ_matrices(λ_opt, N, M, TOL)
    λ_matrices = zeros(N + 2, N + 2, M)
    for m = 1:M
        for i in -1:N
            for j in -1:N
                if i == j
                    continue
                end
                if λ_opt[i_j_m_idx(i,j,m)] > TOL
                λ_matrices[i+2,j+2,m] = λ_opt[i_j_m_idx(i,j,m)]
                end
            end
        end
    end
    return λ_matrices
end



N, L, R, = 2, 1.0, 1.0
p = 0.905263
M = 3
μ = 0
args = [0.00165039, 0.0033389, 0.00688822, 1.46046, 0.122474, 1.56142]
prob = OptimizationProblem(N, L, R, p, M)
h = get_H(N, M, args)
α = compute_α_from_h(prob::OptimizationProblem, h, μ)
ε_set = args[1:M]
default_obj_val_upper_bound = 1e6
zero_idx = [] # in the form (i_idx, j_idx,m) where -1 -> ⋆

F_opt, λ_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, ε_set, p, zero_idx;show_output = :off,
    obj_val_upper_bound=default_obj_val_upper_bound)

λ_matrices = get_λ_matrices(λ_opt, N, M, 1e-4)
display(F_opt)
display(λ_matrices)