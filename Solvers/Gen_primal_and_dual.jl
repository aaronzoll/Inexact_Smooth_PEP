using Revise, Optim, JLD2
include("BnB_PEP_Inexact_Smooth.jl")

# set parameters. Poor convergence for large L, R and small p
L, R, p = 1, 1, 1

N = 4
M = 2*N+1 # 2N+1?
trials = 1 # number of outer optimizations to run

default_obj_val_upper_bound = 1e6
μ = 0 # strong convexity parameter?
 

##### Batch Run Primal ##### 
# nonconvex heuristic to get optimal H, δ_set for given N, L, R, p, M 
max_iter, max_time = 1000, 30
sparsity_pattern = "OGM"
min_F, H, δ_set = run_batch_trials(N, L, R, p, M, trials, sparsity_pattern, max_iter, max_time)
 


##### Dual Optimization #####

# Build α matrix from optimal H 
α = compute_α_from_h(OptimizationProblem(N, L, R, p, M), H, μ)

zero_idx = [] # in the form (i_idx, j_idx,m) where -1 -> ⋆, forces λ_{i,j,m} = 0

F_opt, λ_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, L, α, R, δ_set, p, zero_idx; show_output = :off,
    obj_val_upper_bound=default_obj_val_upper_bound)

λ_matrices = get_λ_matrices(λ_opt, N, M, 1e-4)  # sets values below TOL to zero
display(sum(λ_matrices[:,:,m] for m in 1:M)) # displays support of dual multipliers λ matrix


# computes vector v s.t. Z = vv', i.e. rank 1 cholesky
Z_chol = compute_pivoted_cholesky_L_mat(Z_opt;δ_tol =  1e-3)
Z_chol_scaled = Z_chol./Z_chol[1,1]
v = Z_chol[:,1]
v_scaled = Z_chol_scaled[:,1]


##### Verify Scalar Values #####

# sets appropriate values of δ from optimal set, used in next calculations
function find_eps_idx(N, TOL)
    δ_i_j = zeros(N)
    δ_star_i = zeros(N+1)
    for m = 1:M
        for n = 1:N
            if λ_opt[i_j_m_idx(n-1,n,m)] > TOL
                δ_i_j[n] = δ_set[m]
            end

            if λ_opt[i_j_m_idx(-1,n-1,m)] > TOL
                δ_star_i[n] = δ_set[m]
            end        
        end

        if λ_opt[i_j_m_idx(-1,N,m)] > TOL
            δ_star_i[N+1] = δ_set[m]
        end     
    end

    return  δ_i_j, δ_star_i
end

# set scalar_type = "scaled" to compare to SDP results like Z, λ_opt, etc, 
# regardless we should expect rate = min_F = F_opt

function get_scalars(N, scalar_type)
    δ_i_j = zeros(N) # [δ_0_1, δ_1_2, ..., δ_{N-1}_N]
    λ_i_j = zeros(N)
    λ_star_i = zeros(N+1)  # [δ_star_0, δ_star_1, ..., δ_star_N]
    δ_star_i = zeros(N+1)
    δ_i_j, δ_star_i = find_eps_idx(N, 1e-4)

    α_set = zeros(N+1)
    α_set[1] = 1/L_eps_p(δ_i_j[1],p)+ 1/L_eps_p(δ_star_i[1],p)
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
    B = -(1/L_eps_p(δ_star_i[k],p)+ 1/L_eps_p(δ_i_j[k],p))
    C = -λ_i_j[k-1] * (1/L_eps_p(δ_i_j[k-1],p)+ 1/L_eps_p(δ_i_j[k],p))
    λ_star_i[k] = 1/2 * (-B + sqrt(B^2-4*C))

    λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
    α_set[k] = λ_star_i[k]
    end



    B = -1/L_eps_p(δ_star_i[N+1],p)
    C = -λ_i_j[N]*1/L_eps_p(δ_i_j[N],p)
    λ_star_i[N+1] = 1/2 * (-B + sqrt(B^2-4*C))

    α_set[N+1] = λ_star_i[N+1]


    H_certificate = zeros(N,N)

    for i in 1:N
        for j in 1:i
            if i == j
                H_certificate[i,j] = (λ_i_j[i] + L_eps_p(δ_i_j[i],p)*α_set[i]*α_set[i+1])/(L_eps_p(δ_i_j[i],p) * (λ_i_j[i] + λ_star_i[i+1])) * L
            else
                H_certificate[i,j] = (α_set[i+1]*α_set[j] - 1/L*λ_star_i[i+1]*sum([H_certificate[k,j] for k in j:i-1]))/(λ_i_j[i]+λ_star_i[i+1]) * L
            end
        end
    end
   
    τ = λ_star_i[N+1] + λ_i_j[N] 
    δ_certificate = [δ_i_j; δ_star_i]
    λ_certificate = [λ_i_j; λ_star_i]
    σ = 1/2 * δ_certificate' * λ_certificate
    z_vec = [1; -α_set]
  
    rate = (1/2*R^2 + σ)/τ

    if scalar_type == "scaled" 
        Scaled_λ = 2 * v[1]^2 * λ_certificate # should look like values in support of λ_opt
        Scaled_τ = 2 * v[1]^2 * τ
        Scaled_z = v[1] * z_vec # should look like v, with v*v' = Z
        Scaled_σ = σ * 2 * v[1]^2
        return δ_certificate, Scaled_λ, Scaled_z, H_certificate, Scaled_σ, Scaled_τ, rate
    end

    return δ_certificate, λ_certificate, z_vec, H_certificate, σ, τ, rate 

end

δ_certificate, λ_certificate, z_vec, H_certificate, σ, τ, rate =  get_scalars(N, "scaled")
