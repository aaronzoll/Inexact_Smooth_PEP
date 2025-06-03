using Revise, Optim, JLD2

include("BnB_PEP_Inexact_Smooth.jl")




L, R = 7.4, 0.2
p = 0.8

N = 1
M = 3 # 2N+1?
trials = 5

μ = 0 # strong convexity parameter?
 sparsity_pattern = "OGM"
 min_F, H, ε_set = run_batch_trials(N, L, R, p, M, trials, sparsity_pattern)

 

### Build α 
α = compute_α_from_h(OptimizationProblem(N, L, R, p, M), H, μ)

## Dual Optimization 
default_obj_val_upper_bound = 1e6
zero_idx = [] # in the form (i_idx, j_idx,m) where -1 -> ⋆

display(zero_idx)

F_opt, λ_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, ε_set, p, zero_idx;show_output = :off,
    obj_val_upper_bound=default_obj_val_upper_bound)

λ_matrices = get_λ_matrices(λ_opt, N, M, 1e-4)  # sets values below TOL to zero
display(F_opt)
display(λ_matrices)
display((F_opt-min_F)/min_F)
display(sum(λ_matrices[:,:,m] for m in 1:M))

Z_chol = compute_pivoted_cholesky_L_mat(Z_opt;ϵ_tol =  1e-3)
Z_chol_scaled = Z_chol./Z_chol[1,1]
v = Z_chol[:,1]
v_scaled = Z_chol_scaled[:,1]
display(v_scaled)

function L_eps_p(ε,p)

    return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))

end


function find_eps_idx(TOL)
    ε_0_1, ε_star_0, ε_star_1 = 0, 0, 0
    for m = 1:M
        if λ_opt[i_j_m_idx(0,1,m)] > TOL
            ε_0_1 = ε_set[m]
        end
        if λ_opt[i_j_m_idx(-1,0,m)] > TOL
            ε_star_0 = ε_set[m]
        end
        if λ_opt[i_j_m_idx(-1,1,m)] > TOL
            ε_star_1 = ε_set[m]
        end
    end

    return  ε_0_1, ε_star_0, ε_star_1 
end
# ε_star_1 = ε_set[2]
# ε_0_1 = ε_set[1]
# ε_star_0 = ε_set[3]

ε_0_1, ε_star_0, ε_star_1 = find_eps_idx(1e-5)

α_0 = 1/L_eps_p(ε_0_1,p)+ 1/L_eps_p(ε_star_0,p)
λ_star_0 = α_0
λ_0_1 = α_0

λ_star_1 = 1/2*(1/(L_eps_p(ε_star_1,p)) + sqrt( (1/(L_eps_p(ε_star_1,p))^2 + 4*λ_0_1/L_eps_p(ε_0_1,p)))) 
α_1 = λ_star_1

τ = λ_star_1 + λ_0_1 

H_1_1 = (λ_0_1 + L_eps_p(ε_0_1,p)*α_0*α_1)/(L_eps_p(ε_0_1,p) * (λ_0_1 + λ_star_1)) * L
σ = (ε_0_1*λ_0_1 + ε_star_0*λ_star_0 + ε_star_1*λ_star_1)/2
### test values ###

Scaled_λ = 2 * v[1]^2 * [λ_star_0, λ_star_1, λ_0_1] # should look like values in support of λ_opt
Scaled_τ = 2 * v[1]^2 * τ
Scaled_α = v[1] * [1, -α_0, -α_1] # should look like v, with v*v' = Z
Scaled_H = H_1_1 # note H shouldn't scale, as it is calcluated from the proof where α_{-1} =: v[1] = 1
Scaled_σ = σ * 2 * v[1]^2

rate = (1/2*R^2 + σ)/τ