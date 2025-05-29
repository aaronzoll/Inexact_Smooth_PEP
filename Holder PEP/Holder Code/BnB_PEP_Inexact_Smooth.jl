## Load the packages:
# -------------------
using JuMP, MosekTools, Mosek, LinearAlgebra, OffsetArrays, Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools

## Load the pivoted Cholesky finder
# ---------------------------------
include("code_to_compute_pivoted_cholesky.jl")


## Some helper functions

# construct e_i in R^n
function e_i(n, i)
    e_i_vec = zeros(n, 1)
    e_i_vec[i] = 1
    return e_i_vec
end

# this symmetric outer product is used when a is constant, b is a JuMP variable
function ⊙(a, b)
    return ((a * b') .+ transpose(a * b')) ./ 2
end

# this symmetric outer product is for computing ⊙(a,a) where a is a JuMP variable
function ⊙(a)
    return a * transpose(a)
end

# function to compute cardinality of a vector
function compute_cardinality(x, ϵ_sparsity)
    n = length(x)
    card_x = 0
    for i in 1:n
        if abs(x[i]) >= ϵ_sparsity
            card_x = card_x + 1
        end
    end
    return card_x
end

# function to compute rank of a matrix
function compute_rank(X, ϵ_sparsity)
    eigval_array_X = eigvals(X)
    rnk_X = 0
    n = length(eigval_array_X)
    for i in 1:n
        if abs(eigval_array_X[i]) >= ϵ_sparsity
            rnk_X = rnk_X + 1
        end
    end
    return rnk_X
end


##########################
##  Generators of Data  ##
##########################

# Options for these function are
# step_size_type = :Default => will create a last step of 1/(L) rest will be zero
# step_size_type = :Random => will create a random stepsize

function feasible_h_generator(N, L; step_size_type=:Default)

    # construct h
    # -----------
    h = OffsetArray(zeros(N), 0:N-1)

    if step_size_type == :Default
        for i in 0:N-1
            h[i] = 1 # because we have defined h[i]/L in the FSFOM, h[i]=1 will correspond to a stepsize of 1/L
        end
    elseif step_size_type == :Random
        for i in 0:N-1
            h[i] = Uniform(0, 1) # same
        end
    end

    return h

end


function compute_α_from_h(h, N, μ, L)
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

function feasible_H_generator(N, L; step_size_type=:Default)

    # construct H
    # -----------
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    if step_size_type == :Default
        for i in 1:N
            H[i, i-1] = 1 # because we have defined h[i,i-1]/L in the algorithm, so declaring 1 will make the stepsizes equal to 1/L
        end
    elseif step_size_type == :Random
        for i in 1:N
            H[i, i-1] = Uniform(0, 1)
        end
    end

    # find α from h
    # -------------


    return H
end


function data_generator_function(N, L, α; input_type=:stepsize_constant)

    # define all the bold vectors
    # --------------------------

    # define 𝐱_0 and 𝐱_star

    𝐱_0 = e_i(N + 2, 1)

    𝐱_star = zeros(N + 2, 1)

    # define 𝐠_0, 𝐠_1, …, 𝐠_N

    # first we define the 𝐠 vectors,
    # index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

    # 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]
    𝐠 = OffsetArray(zeros(N + 2, N + 2), 1:N+2, -1:N)
    for i in 0:N
        𝐠[:, i] = e_i(N + 2, i + 2)
    end

    # time to define the 𝐟 vectors

    # 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

    𝐟 = OffsetArray(zeros(N + 1, N + 2), 1:N+1, -1:N)

    for i in 0:N
        𝐟[:, i] = e_i(N + 1, i + 1)
    end

    if input_type == :stepsize_constant

        # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}] ∈ 𝐑^(N+2 × N+2)
        𝐱 = OffsetArray(zeros(N + 2, N + 2), 1:N+2, -1:N)

        # assign values next using our formula for 𝐱_k
        𝐱[:, 0] = 𝐱_0

        for i in 1:N
            𝐱[:, i] = 𝐱[:, 0] - (1 / L) .* (sum(α[i, j] * 𝐠[:, j] for j in 0:i-1))
        end

    elseif input_type == :stepsize_variable

        𝐱 = [𝐱_star 𝐱_0]

        # assign values next using our formula for 𝐱_k
        for k in 1:N
            𝐱_k = 𝐱_0 - (1 / L) * (sum(h[i] .* 𝐠[:, i] for i in 0:k-1))
            𝐱 = [𝐱 𝐱_k]
        end

        # make 𝐱 an offset array to make our life comfortable
        𝐱 = OffsetArray(𝐱, 1:N+2, -1:N)



    end

    return 𝐱, 𝐠, 𝐟

end


# Index set creator function for the dual variables λ

struct i_j_m_idx # correspond to (i,j) tuple, where i,j ∈ I_N_⋆ 
    i::Int64 # corresponds to index i
    j::Int64 # corresponds to index j
    m::Int64
end

# We have dual variable λ={λ_ij}_{i,j} where i,j ∈ I_N_star
# The following function creates the maximal index set for λ

function index_set_constructor_for_dual_vars_full(I_N_star, M)

    # construct the index set for λ
    idx_set_λ = i_j_m_idx[] 
    for m in 1:M
        for i in I_N_star
            for j in I_N_star
                if i != j
                    push!(idx_set_λ, i_j_m_idx(i, j, m))
                end
            end
        end
    end
    return idx_set_λ

end

# The following function will return the effective index set of a known λ i.e., those indices of  that are  λ  that are non-zero.

function effective_index_set_finder(λ; ϵ_tol=0.0005)

    # the variables λ are of the type DenseAxisArray whose index set can be accessed using _.axes and data via _.data syntax

    idx_set_λ_current = (λ.axes)[1]

    idx_set_λ_effective = i_j_idx[]

    # construct idx_set_λ_effective

    for i_j_λ in idx_set_λ_current
        if abs(λ[i_j_λ]) >= ϵ_tol # if λ[i,j] >= ϵ, where ϵ is our cut off for accepting nonzero
            push!(idx_set_λ_effective, i_j_λ)
        end
    end

    return idx_set_λ_effective

end

# The following function will return the zero index set of a known L_cholesky i.e., those indices of  that are  L  that are zero. 💀 Note that for λ we are doing the opposite.

function zero_index_set_finder_L_cholesky(L_cholesky; ϵ_tol=1e-4)
    n_L_cholesky, _ = size(L_cholesky)
    zero_idx_set_L_cholesky = []
    for i in 1:n_L_cholesky
        for j in 1:n_L_cholesky
            if i >= j # because i<j has L_cholesky[i,j] == 0 for lower-triangual structure
                if abs(L_cholesky[i, j]) <= ϵ_tol
                    push!(zero_idx_set_L_cholesky, (i, j))
                end
            end
        end
    end
    return zero_idx_set_L_cholesky
end


A_mat(i, j, h, 𝐠, 𝐱) = ⊙(𝐠[:, j], 𝐱[:, i] - 𝐱[:, j])
B_mat(i, j, h, 𝐱) = ⊙(𝐱[:, i] - 𝐱[:, j], 𝐱[:, i] - 𝐱[:, j])
C_mat(i, j, 𝐠) = ⊙(𝐠[:, i] - 𝐠[:, j], 𝐠[:, i] - 𝐠[:, j])
a_vec(i, j, 𝐟) = 𝐟[:, j] - 𝐟[:, i]




##################
##  Optimizers  ##
##################

# step_size options, scaled sets algorithm to use L_epsilon
# otherwise, stepsizes are 1/L

function solve_primal_with_known_stepsizes(N, L, h, R, ε, p; show_output=:off, step_size=:scaled)
    # data generator
    # --------------
    L_eps = ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))
    if step_size == :scaled
        𝐱, 𝐠, 𝐟 = data_generator_function(N, L_eps, h; input_type=:stepsize_constant)
    else
        𝐱, 𝐠, 𝐟 = data_generator_function(N, L, h; input_type=:stepsize_constant)
    end
    # number of points etc
    # --------------------

    I_N_star = -1:N
    dim_G = N + 2
    dim_Ft = N + 1


    # define the model
    # ----------------

    model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model_primal_PEP_with_known_stepsizes, G[1:dim_G, 1:dim_G], PSD)

    # construct Ft (this is just transpose of F)
    @variable(model_primal_PEP_with_known_stepsizes, Ft[1:dim_Ft])

    # define objective
    # ----------------

    @objective(model_primal_PEP_with_known_stepsizes, Max,
        Ft' * a_vec(-1, N, 𝐟)
    )

    # interpolation constraint
    # ------------------------

    for i in I_N_star, j in I_N_star
        if i != j
            @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, h, 𝐠, 𝐱)) + ((1 / (2 * (L_eps))) * tr(G * C_mat(i, j, 𝐠))) - ε <= 0
            )
        end
    end


    # initial condition
    # -----------------

    @constraint(model_primal_PEP_with_known_stepsizes, tr(G * B_mat(0, -1, h, 𝐱)) <= R^2)

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_primal_PEP_with_known_stepsizes)
    end

    optimize!(model_primal_PEP_with_known_stepsizes)

    # store and return the solution
    # -----------------------------

    if termination_status(model_primal_PEP_with_known_stepsizes) != MOI.OPTIMAL
        # @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    p_star = objective_value(model_primal_PEP_with_known_stepsizes)

    G_star = value.(G)

    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star

end


# Following function uses batch of ε to set inexact constraints for set
function solve_primal_with_known_stepsizes_batch(N, L, α, R, ε_set, p, μ; show_output=:off)
    # number of points etc
    # --------------------

    I_N_star = -1:N
    dim_G = N + 2
    dim_Ft = N + 1


    # define the model
    # ----------------

    model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model_primal_PEP_with_known_stepsizes, G[1:dim_G, 1:dim_G], PSD)

    # construct Ft (this is just transpose of F)
    @variable(model_primal_PEP_with_known_stepsizes, Ft[1:dim_Ft])

    # pick stepsize
    ε_opt = maximum(ε_set)
    L_step = L

    # define objective
    # ----------------
    #  α = compute_α_from_h(H, N, μ, L)
    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, α; input_type=:stepsize_constant)
    @objective(model_primal_PEP_with_known_stepsizes, Max,
        Ft' * a_vec(-1, N, 𝐟)
    )
    # data generator
    # --------------
    for ε in ε_set
        if p < 1
            L_eps = ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))
        else
            L_eps = L
        end

        # interpolation constraint
        # ------------------------
        for i in I_N_star, j in I_N_star
            if i != j


                @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, α, 𝐠, 𝐱)) + ((1 / (2 * (L_eps))) * tr(G * C_mat(i, j, 𝐠))) - ε / 2 <= 0)
            end
        end
    end


    # initial condition
    # -----------------

    @constraint(model_primal_PEP_with_known_stepsizes, tr(G * B_mat(0, -1, α, 𝐱)) <= R^2)

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_primal_PEP_with_known_stepsizes)
    end

    optimize!(model_primal_PEP_with_known_stepsizes)

    # store and return the solution
    # -----------------------------

    if termination_status(model_primal_PEP_with_known_stepsizes) != MOI.OPTIMAL
        #   @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    p_star = objective_value(model_primal_PEP_with_known_stepsizes)

    G_star = value.(G)

    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star

end


function solve_dual_PEP_with_known_stepsizes(N, L, α, R, ε_set, p, zero_idx;
    show_output=:off,
    ϵ_tol_feas=1e-6,
    objective_type=:default,
    obj_val_upper_bound=default_obj_val_upper_bound)
    M = length(ε_set)
    # data generator
    # --------------

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, α; input_type=:stepsize_constant)

    # Number of points etc
    # --------------------

    I_N_star = -1:N
    dim_Z = N + 2

    # define the model
    # ----------------

    model_dual_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    # define the variables
    # --------------------

    # define the index set of λ
    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star, M)

    @variable(model_dual_PEP_with_known_stepsizes, λ[idx_set_λ] >= 0)

    # define ν

    @variable(model_dual_PEP_with_known_stepsizes, ν >= 0)

    # define Z ⪰ 0
    @variable(model_dual_PEP_with_known_stepsizes, Z[1:dim_Z, 1:dim_Z], PSD)




    if objective_type == :default

        @info "[🐒 ] Minimizing the usual performance measure"

        @objective(model_dual_PEP_with_known_stepsizes, Min, ν * R^2 + sum(λ[i_j_m_λ]*ε_set[i_j_m_λ.m]/2 for i_j_m_λ in idx_set_λ))

    elseif objective_type == :find_sparse_sol #### WARNING, not of these take into account ε_set

        @info "[🐮 ] Finding a sparse dual solution given the objective value upper bound"

        @objective(model_dual_PEP_with_known_stepsizes, Min, sum(λ[i_j_m_λ] for i_j_m_λ in idx_set_λ))

        @constraint(model_dual_PEP_with_known_stepsizes, ν * R^2 <= obj_val_upper_bound)

    elseif objective_type == :find_M_λ

        @info "[🐷 ] Finding upper bound on the entries of λ for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(λ[i_j_m_λ] for i_j_m_λ in idx_set_λ))

        @constraint(model_dual_PEP_with_known_stepsizes, ν * R^2 <= obj_val_upper_bound)

    elseif objective_type == :find_M_Z

        @info "[🐯 ] Finding upper bound on the entries of Z for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, tr(Z))

        @constraint(model_dual_PEP_with_known_stepsizes, ν * R^2 <= obj_val_upper_bound)

    end
  
    # add the linear constraint
    # -------------------------

    @constraint(model_dual_PEP_with_known_stepsizes, sum(λ[i_j_m_λ] * a_vec(i_j_m_λ.i, i_j_m_λ.j, 𝐟) for i_j_m_λ in idx_set_λ) - a_vec(-1, N, 𝐟) .== 0)


    # add the LMI constraint
    # ----------------------

    function L_ε(ε, p)
        return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))
    end






        @constraint(model_dual_PEP_with_known_stepsizes,
            ν * B_mat(0, -1, α, 𝐱) + sum(λ[i_j_m_λ] * (A_mat(i_j_m_λ.i, i_j_m_λ.j, α, 𝐠, 𝐱)) for i_j_m_λ in idx_set_λ) +
             sum((1 / (2 * (L_ε(ε_set[i_j_m_λ.m], p)))) * λ[i_j_m_λ] * C_mat(i_j_m_λ.i, i_j_m_λ.j, 𝐠) for i_j_m_λ in idx_set_λ) 
          .==
            Z
        )


  

    for idx in zero_idx

        @constraint(model_dual_PEP_with_known_stepsizes, λ[i_j_m_idx(idx...)] == 0)
    end
    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_dual_PEP_with_known_stepsizes)
    end

    optimize!(model_dual_PEP_with_known_stepsizes)

    if termination_status(model_dual_PEP_with_known_stepsizes) != MOI.OPTIMAL
        @info "💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀"
        @error "model_dual_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_dual_PEP_with_known_stepsizes)
    end

    # store the solutions and return
    # ------------------------------

    # store λ_opt

    λ_opt = value.(λ)

    # store ν_opt

    ν_opt = value.(ν)

    # store Z_opt

    #  Z_opt1 = value.(Z_1)
    #  Z_opt2 = value.(Z_2)

    # compute cholesky

    #   L_cholesky_opt =  compute_pivoted_cholesky_L_mat(Z_opt1)

    #   if norm(Z_opt1 - L_cholesky_opt*L_cholesky_opt', Inf) > 1e-6
    #      @info "checking the norm bound"
    #       @warn "||Z - L*L^T|| = $(norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf))"
    #  end

    # effective index sets for the dual variables λ, μ, ν

    # idx_set_λ_effective = effective_index_set_finder(λ_opt ; ϵ_tol = 0.0005)

    # store objective

    #  ℓ_1_norm_λ = sum(λ_opt)
    #   tr_Z = tr(Z_opt1)
    original_performance_measure = ν_opt * R^2 + sum(λ_opt[i_j_m_λ]*ε_set[i_j_m_λ.m]/2 for i_j_m_λ in idx_set_λ)
    # return all the stored values

    return original_performance_measure, λ_opt

end


# We also provide a function to check if in a particular feasible solution, these bounds are violated

function bound_violation_checker_BnB_PEP(
    # input point
    # -----------
    d_star_sol, λ_sol, ν_sol, Z_sol, L_cholesky_sol, h_sol,
    # input bounds
    # ------------
    λ_lb, λ_ub, ν_lb, ν_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, h_lb, h_ub;
    # options
    # -------
    show_output=:on,
    computing_global_lower_bound=:off
)

    if show_output == :on
        @show [minimum(λ_sol) maximum(λ_sol) λ_ub]
        @show [ν_lb ν_sol ν_ub]
        @show [Z_lb minimum(Z_sol) maximum(Z_sol) Z_ub]
        @show [L_cholesky_lb minimum(L_cholesky_sol) maximum(L_cholesky_sol) L_cholesky_ub]
        @show [h_lb minimum(h_sol) maximum(h_sol) h_ub]
    end

    # bound satisfaction flag

    bound_satisfaction_flag = 1

    # verify bound for λ
    if !(maximum(λ_sol) < λ_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found λ is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for ν: this is not necessary because this will be ensured due to our objective function being ν R^2
    if !(maximum(ν_sol) <= ν_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found ν is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for Z
    if !(Z_lb - 1e-8 < minimum(Z_sol) && maximum(Z_sol) < Z_ub + 1e-8)
        @error "found Z is violating the input bound"
        bound_satisfaction_flag = 0
    end

    if computing_global_lower_bound == :off
        # verify bound for L_cholesky
        if !(L_cholesky_lb - 1e-8 < minimum(L_cholesky_sol) && maximum(L_cholesky_sol) < L_cholesky_ub + 1e-8)
            @error "found L_cholesky is violating the input bound"
            bound_satisfaction_flag = 0
        end
    elseif computing_global_lower_bound == :on
        @info "no need to check bound on L_cholesky"
    end

    # # verify bound for objective value
    # if abs(obj_val_sol-BnB_PEP_cost_lb) <= ϵ_tol_sol
    #     @error "found objective value is violating the input bound"
    #     bound_satisfaction_flag = 0
    # end

    if bound_satisfaction_flag == 0
        @error "[💀 ] some bound is violated, increase the bound intervals "
    elseif bound_satisfaction_flag == 1
        @info "[😅 ] all bounds are satisfied by the input point, rejoice"
    end

    return bound_satisfaction_flag

end



##### Aaron's Code ######

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



