using Revise, Optim, JLD2
include("BnB_PEP_Inexact_Smooth.jl")

# set parameters, breaking for large L, R and small p
L, R, p = 1.7, 0.8, 0.87

N = 2
M = 2*N+1 # 2N+1?
trials = 3 # number of outer optimizations to run

default_obj_val_upper_bound = 1e6
μ = 0 # strong convexity parameter?
 

##### Batch Run Primal ##### 
# nonconvex heuristic to get optimal H, ε_set for given N, L, R, p, M 

sparsity_pattern = "single step"
min_F, H, ε_set = run_batch_trials(N, L, R, p, M, trials, sparsity_pattern)
 


##### Dual Optimization #####

# Build α matrix from optimal H 
α = compute_α_from_h(OptimizationProblem(N, L, R, p, M), H, μ)

zero_idx = [] # in the form (i_idx, j_idx,m) where -1 -> ⋆, forces λ_{i,j,m} = 0

F_opt, λ_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, ε_set, p, zero_idx; show_output = :off,
    obj_val_upper_bound=default_obj_val_upper_bound)

λ_matrices = get_λ_matrices(λ_opt, N, M, 1e-4)  # sets values below TOL to zero
display(sum(λ_matrices[:,:,m] for m in 1:M)) # displays support of dual multipliers λ matrix


# computes vector v s.t. Z = vv', i.e. rank 1 cholesky
Z_chol = compute_pivoted_cholesky_L_mat(Z_opt;ϵ_tol =  1e-3)
Z_chol_scaled = Z_chol./Z_chol[1,1]
v = Z_chol[:,1]
v_scaled = Z_chol_scaled[:,1]


##### Verify Scalar Values #####

# sets appropriate values of ε from optimal set, used in next calculations
function find_eps_idx(TOL)
    ε_0_1, ε_1_2, ε_star_0, ε_star_1, ε_star_2 = 0, 0, 0, 0, 0
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
        if λ_opt[i_j_m_idx(1,2,m)] > TOL
            ε_1_2 = ε_set[m]
        end
        if λ_opt[i_j_m_idx(-1,2,m)] > TOL
            ε_star_2 = ε_set[m]        
        end
    end

    return   ε_0_1, ε_1_2, ε_star_0, ε_star_1, ε_star_2
end

# set scalar_type = "scaled" to compare to SDP results like Z, λ_opt, etc, 
# regardless we should expect rate = min_F = F_opt
function get_scalars_N_1(scalar_type)
    ε_0_1, ε_star_0, ε_star_1 = find_eps_idx(1e-5)

    α_0 = 1/L_eps_p(ε_0_1,p)+ 1/L_eps_p(ε_star_0,p)
    λ_star_0 = α_0
    λ_0_1 = α_0

    λ_star_1 = 1/2*(1/(L_eps_p(ε_star_1,p)) + sqrt( (1/(L_eps_p(ε_star_1,p))^2 + 4*λ_0_1/L_eps_p(ε_0_1,p)))) 
    α_1 = λ_star_1

    τ = λ_star_1 + λ_0_1 

    H_certificate = (λ_0_1 + L_eps_p(ε_0_1,p)*α_0*α_1)/(L_eps_p(ε_0_1,p) * (λ_0_1 + λ_star_1)) * L
    σ = (ε_0_1*λ_0_1 + ε_star_0*λ_star_0 + ε_star_1*λ_star_1)/2
    ### test values ###

    

    ε_certificate = [ε_0_1, ε_star_0, ε_star_1]
    λ_certificate = [λ_star_0, λ_star_1, λ_0_1]   
    z_vec = [1, -α_0, -α_1]
    rate = (1/2*R^2 + σ)/τ

    if scalar_type == "scaled" 
        Scaled_λ = 2 * v[1]^2 * [λ_star_0, λ_star_1, λ_0_1] # should look like values in support of λ_opt
        Scaled_τ = 2 * v[1]^2 * τ
        Scaled_z = v[1] * [1, -α_0, -α_1] # should look like v, with v*v' = Z
        Scaled_σ = σ * 2 * v[1]^2
        return ε_certificate, Scaled_λ, Scaled_z, H_certificate, Scaled_σ, Scaled_τ, rate
    end


    return ε_certificate, λ_certificate, z_vec, H_certificate, σ, τ, rate 
end


# ε_certificate, λ_certificate, z_vec, H_certificate, σ, τ, rate =  get_scalars_N_1("scaled")


function get_scalars_N_2(scalar_type)
    ε_0_1, ε_1_2, ε_star_0, ε_star_1, ε_star_2 = find_eps_idx(1e-4)
    α_0 = 1/L_eps_p(ε_0_1,p)+ 1/L_eps_p(ε_star_0,p)
    λ_0_1 = α_0
    λ_star_0 = α_0

    B1 = -(1/L_eps_p(ε_star_1,p)+ 1/L_eps_p(ε_1_2,p))
    C1 = -λ_0_1 * (1/L_eps_p(ε_0_1,p)+ 1/L_eps_p(ε_1_2,p))
    λ_star_1 = 1/2 * (-B1 + sqrt(B1^2-4*C1))

    λ_1_2 = λ_0_1 + λ_star_1

    α_1 = λ_star_1

    B2 = -1/L_eps_p(ε_star_2,p)
    C2 = -λ_1_2*1/L_eps_p(ε_1_2,p)
    λ_star_2 = 1/2 * (-B2 + sqrt(B2^2-4*C2))

    α_2 = λ_star_2



    # NOTE: In H_2_1 calc there is an extra 1/L because algo built that in x_k updates
    # This makes sense, but make sure to implement in future recurrsion where applicable
    H_1_1 = (λ_0_1 + L_eps_p(ε_0_1,p)*α_0*α_1)/(L_eps_p(ε_0_1,p) * (λ_0_1 + λ_star_1)) * L
    H_2_1 = (α_0*α_2-1/L*λ_star_2*H_1_1)/(λ_1_2 + λ_star_2) * L  
    H_2_2 = (λ_1_2 + L_eps_p(ε_1_2,p)*α_1*α_2)/(L_eps_p(ε_1_2,p) * (λ_1_2 + λ_star_2)) * L


    τ = λ_star_2 + λ_1_2 
    σ = (ε_0_1*λ_0_1 + ε_1_2*λ_1_2 + ε_star_0*λ_star_0 + ε_star_1*λ_star_1 + ε_star_2*λ_star_2)/2
    ε_certificate = [ε_0_1, ε_1_2, ε_star_0, ε_star_1, ε_star_2]
    λ_certificate = [λ_star_0, λ_1_2, λ_star_0, λ_star_1, λ_star_2]   
    z_vec = [1, -α_0, -α_1, -α_2]
    H_certificate = zeros(2,2)
    H_certificate[1,1] = H_1_1
    H_certificate[2,1] = H_2_1
    H_certificate[2,2] = H_2_2

    rate = (1/2*R^2 + σ)/τ

    if scalar_type == "scaled" 
        Scaled_λ = 2 * v[1]^2 * λ_certificate # should look like values in support of λ_opt
        Scaled_τ = 2 * v[1]^2 * τ
        Scaled_z = v[1] * [1, -α_0, -α_1, -α_2] # should look like v, with v*v' = Z
        Scaled_σ = σ * 2 * v[1]^2
        return ε_certificate, Scaled_λ, Scaled_z, H_certificate, Scaled_σ, Scaled_τ, rate
    end

    return ε_certificate, λ_certificate, z_vec, H_certificate, σ, τ, rate 

end

ε_certificate, λ_certificate, z_vec, H_certificate, σ, τ, rate =  get_scalars_N_2("scaled")
