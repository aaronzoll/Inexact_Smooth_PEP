using Revise, OffsetArrays
include("BnB_PEP_Inexact_Smooth.jl")

#### Set WLOG L=R=1 ####
L, R, = 1.0, 1.0
p = 0.905263
μ = 0 # strong convexity parameter? 

#### number of N number of steps, M number of ε to be used #####
N = 2
M = 3

### Build α 
args = [0.00165039, 0.0033389, 0.00688822, 1.46046, 0.122474, 1.56142] 
α = compute_α_from_h(OptimizationProblem(N, L, R, p, M), get_H(N, M, args), μ)
ε_set = args[1:M]

## Dual Optimization 
default_obj_val_upper_bound = 1e6
zero_idx = [] # in the form (i_idx, j_idx,m) where -1 -> ⋆

F_opt, λ_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, ε_set, p, zero_idx;show_output = :off,
    obj_val_upper_bound=default_obj_val_upper_bound)

λ_matrices = get_λ_matrices(λ_opt, N, M, 1e-4)  # sets values below TOL to zero
display(F_opt)
display(λ_matrices)