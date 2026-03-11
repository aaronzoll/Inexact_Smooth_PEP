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



function data_generator_function(N, L, h; input_type=:stepsize_constant)

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



    # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}] ∈ 𝐑^(N+2 × N+2)
    𝐱 = OffsetArray(zeros(N + 2, N + 2), 1:N+2, -1:N)

    # assign values next using our formula for 𝐱_k
    𝐱[:, 0] = 𝐱_0

    for i in 1:N
        𝐱[:, i] = 𝐱[:, i-1] - (h / L) .* 𝐠[:, i-1]
    end



    return 𝐱, 𝐠, 𝐟

end


struct i_j_idx # correspond to (i,j) pair, where i,j ∈ I_N_⋆
    i::Int64 # corresponds to index i
    j::Int64 # corresponds to index j
end

# We have dual variable λ={λ_ij}_{i,j} where i,j ∈ I_N_star
# The following function creates the maximal index set for λ

function index_set_constructor_for_dual_vars_full(I_N_star)

    # construct the index set for λ
    idx_set_λ = i_j_idx[]
    for i in I_N_star
        for j in I_N_star
            if i != j
                push!(idx_set_λ, i_j_idx(i, j))
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


A_mat(i, j, 𝐠, 𝐱) = ⊙(𝐠[:, j], 𝐱[:, i] - 𝐱[:, j])
B_mat(i, j, 𝐱) = ⊙(𝐱[:, i] - 𝐱[:, j], 𝐱[:, i] - 𝐱[:, j])
C_mat(i, j, 𝐠) = ⊙(𝐠[:, i] - 𝐠[:, j], 𝐠[:, i] - 𝐠[:, j])
a_vec(i, j, 𝐟) = 𝐟[:, j] - 𝐟[:, i]




##################
##  Optimizers  ##
##################

# step_size options, scaled sets algorithm to use L_epsilon
# otherwise, stepsizes are 1/L

function solve_primal_with_known_stepsizes(N, L, h, R; show_output=:off)

    # data generator
    # --------------

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, h; input_type=:stepsize_constant)

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

    # interpolation constraints
    # ------------------------
    for i in 0:N-1

        j = i + 1
        @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, 𝐠, 𝐱)) + ((1 / (2 * (L))) * tr(G * C_mat(i, j, 𝐠))) <= 0)

    end

    for j in 0:N

        i = -1
        @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, 𝐠, 𝐱)) + ((1 / (2 * (L))) * tr(G * C_mat(i, j, 𝐠))) <= 0)

    end


    # initial condition
    # -----------------

    @constraint(model_primal_PEP_with_known_stepsizes, tr(G * B_mat(0, -1,
        𝐱)) <= R^2)

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_primal_PEP_with_known_stepsizes)
    end

    optimize!(model_primal_PEP_with_known_stepsizes)

    # store and return the solution
    # -----------------------------

    if termination_status(model_primal_PEP_with_known_stepsizes) != MOI.OPTIMAL
        @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    p_star = objective_value(model_primal_PEP_with_known_stepsizes)

    G_star = value.(G)

    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star

end


# In this function, the most important option is objective type:
# 0) :default will minimize ν*R^2 (this is the dual of the primal pep for a given stepsize)
# other options are
# 1) :find_sparse_sol, this will find a sparse solution given a particular stepsize and objective value upper bound
# 2) :find_M_λ , find the upper bound for the λ variables by maximizing ||λ||_1 for a given stepsize and particular objective value upper bound
# 3) :find_M_Z, find the upper bound for the entries of the slack matrix Z, by maximizing tr(Z) for for a given stepsize and particular objective value upper bound

function solve_dual_PEP_with_known_stepsizes(N, L, h, R;
    show_output=:off,
    ϵ_tol_feas=1e-6,
    objective_type=:default,
    obj_val_upper_bound=default_obj_val_upper_bound)

    # data generator
    # --------------

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, h; input_type=:stepsize_constant)

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
    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)

    # define λ
    @variable(model_dual_PEP_with_known_stepsizes, λ[idx_set_λ] >= 0)

    # define ν

    @variable(model_dual_PEP_with_known_stepsizes, ν >= 0)

    # define Z ⪰ 0
    @variable(model_dual_PEP_with_known_stepsizes, Z[1:dim_Z, 1:dim_Z], PSD)




    @info "[🐒 ] Minimizing the usual performance measure"

    @objective(model_dual_PEP_with_known_stepsizes, Min, ν * R^2)

 

    # add the linear constraint
    # -------------------------

    # note that in the code i_j_λ = (i,j), i_j_λ.i = i, i_j_λ.j = j
    @constraint(model_dual_PEP_with_known_stepsizes, sum(λ[i_j_λ] * a_vec(i_j_λ.i, i_j_λ.j, 𝐟) for i_j_λ in idx_set_λ) - a_vec(-1, N, 𝐟) .== 0)

    # add the LMI constraint
    # ----------------------

    @constraint(model_dual_PEP_with_known_stepsizes,
        ν * B_mat(0, -1, 𝐱) + sum(λ[i_j_λ] * A_mat(i_j_λ.i, i_j_λ.j, 𝐠, 𝐱) for i_j_λ in idx_set_λ) +
        (1 / (2 * (L))) * sum(λ[i_j_λ] * C_mat(i_j_λ.i, i_j_λ.j, 𝐠) for i_j_λ in idx_set_λ)
        .==
        Z
    )

    # sparsity 
    for i in I_N_star, j in I_N_star
        if j != i+1 && i != -1 && i != j
            @constraint(model_dual_PEP_with_known_stepsizes, λ[i_j_idx(i,j)] == 0)
        end
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

    Z_opt = value.(Z)

    # compute cholesky

    L_cholesky_opt = compute_pivoted_cholesky_L_mat(Z_opt)

    if norm(Z_opt - L_cholesky_opt * L_cholesky_opt', Inf) > 1e-6
        @info "checking the norm bound"
        @warn "||Z - L*L^T|| = $(norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf))"
    end

    # effective index sets for the dual variables λ, μ, ν

    idx_set_λ_effective = effective_index_set_finder(λ_opt; ϵ_tol=0.0005)

    # store objective

    ℓ_1_norm_λ = sum(λ_opt)
    tr_Z = tr(Z_opt)
    original_performance_measure = ν_opt * R^2

    # return all the stored values

    return original_performance_measure, λ_opt, ν_opt, Z_opt

end


N = 6
L = 3.4
R = 3
h = 0
p_star, G_star, Ft_star = solve_primal_with_known_stepsizes(N, L, h, R; show_output=:off)


function get_λ_matrices(λ_opt, N, TOL)
    λ_matrices = zeros(N + 2, N + 2)

    for i in -1:N
        for j in -1:N
            if i == j
                continue
            end
            if λ_opt[i_j_idx(i, j)] > TOL
                λ_matrices[i+2, j+2] = λ_opt[i_j_idx(i, j)]
            end
        end
    end

    return λ_matrices
end


d_star, λ_opt, ν_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, L, h, R;
    show_output=:off,
    ϵ_tol_feas=1e-6,
    objective_type=:default,
    obj_val_upper_bound=1e-6)

λ_matrices = get_λ_matrices(λ_opt, N, 1e-5)

Z_chol = compute_pivoted_cholesky_L_mat(Z_opt; ϵ_tol=1e-3)
Z_chol_scaled = Z_chol ./ Z_chol[1, 1]
v = Z_chol[:, 1]
v_scaled = Z_chol_scaled[:, 1]

λ = zeros(N)

for i = 1:N
    λ[i] = i/(2*N+1-i)
end

S_0 = zeros(N+1, N+1)
S_1 = zeros(N+1, N+1)

for i = 1:N+1
    for j = 1:N+1
        if i == j && i <= N
            S_0[i,j] = 2*λ[i]

        elseif abs(i-j) == 1
            S_0[i,j] = -λ[max(i,j)-1]
        end
    end
end
S_0[N+1, N+1] = 1

for i = 1:N
    for j = 1:N
        if i == j
            S_1[i,j] = 2*λ[i]

        else
            S_1[i,j] = λ[max(i,j)]-λ[max(i,j)-1]
        end
    end

    S_1[N+1,i] = 1-λ[N]
    S_1[i, N+1] = 1-λ[N]
end

S_1[N+1, N+1] = 1
q = zeros(N+1)
q[1] = λ[1]
for i in 2:N
    q[i] = λ[i]-λ[i-1]
end
q[N+1] = 1-λ[N]
t = p_star
S = [t/R^2 -1/2*q'; -1/2*q  1/(2*L)*((1-h)*S_0 + h*S_1)]

display(λ_matrices)
display(Z_opt)
display(p_star)
display(d_star)
display(R^2*Z_opt[1,1])



