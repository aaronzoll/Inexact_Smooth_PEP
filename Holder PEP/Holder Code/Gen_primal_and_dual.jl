using Revise, Optim, JLD2

include("BnB_PEP_Inexact_Smooth.jl")



N, L, R, p, M, trials = 4, 1.0, 1.0, 0.92, 5, 3
 μ = 0 # strong convexity parameter?

 min_F, H, ε_set = run_batch_trials(N, L, R, p, M, trials)

 

### Build α 
α = compute_α_from_h(OptimizationProblem(N, L, R, p, M), H, μ)

## Dual Optimization 
default_obj_val_upper_bound = 1e6
zero_idx = [] # in the form (i_idx, j_idx,m) where -1 -> ⋆

F_opt, λ_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, ε_set, p, zero_idx;show_output = :off,
    obj_val_upper_bound=default_obj_val_upper_bound)

λ_matrices = get_λ_matrices(λ_opt, N, M, 1e-4)  # sets values below TOL to zero
display(F_opt)
display(λ_matrices)
display((F_opt-min_F)/min_F)
display(sum(λ_matrices[:,:,m] for m in 1:M))