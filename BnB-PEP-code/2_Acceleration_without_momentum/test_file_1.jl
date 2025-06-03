
using OffsetArrays 
include("BnB_PEP_reducing_function_value_AWM.jl")
default_obj_val_upper_bound = 1e6
N = 1
L, R = 1.0, 1.0
h = OffsetArray([1.5], 0:0)


original_performance_measure, ℓ_1_norm_λ, tr_Z, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h, idx_set_λ_effective = solve_dual_PEP_with_known_stepsizes(N, L, h, R;
    show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = default_obj_val_upper_bound)