## Load the packages:
# -------------------
using JuMP, MosekTools, Mosek, LinearAlgebra, OffsetArrays, Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, Optim

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
             𝐱[:, i] = 𝐱[:, 0] -  (sum(α[i, j] * 𝐠[:, j] for j in 0:i-1))
          #  𝐱[:, i] = 𝐱[:, 0] - (1) .* (sum(α[i, j] * 𝐠[:, j] for j in 0:i-1)) # no longer divide by L in construction
        end

    elseif input_type == :stepsize_variable

        𝐱 = [𝐱_star 𝐱_0]

        # assign values next using our formula for 𝐱_k
        for k in 1:N
            𝐱_k = 𝐱_0 -  (sum(h[i] .* 𝐠[:, i] for i in 0:k-1))
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

function solve_primal_PEP_with_known_stepsizes(N, L, α, R, ε_set, p, zero_idx;
    show_output = :off,
    ϵ_tol_feas  = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = 1e6,
    acceptable_termination_statuses = [MOI.OPTIMAL, MOI.SLOW_PROGRESS]
)
    M = size(ε_set, 3)
    I_N_star = -1:N
    dim_G  = N + 2
    dim_Ft = N + 1

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, α; input_type = :stepsize_constant)

    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star, M)

    # remove zero'd-out pairs (same as dual fixes λ == 0)
    zero_set = Set([i_j_m_idx(z...) for z in zero_idx])
    active_idx = filter(t -> t ∉ zero_set, idx_set_λ)

    function L_ε(ε, p)
        return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))
    end

    # define the model
    model = Model(optimizer_with_attributes(Mosek.Optimizer))

    @variable(model, G[1:dim_G, 1:dim_G], PSD)
    @variable(model, Ft[1:dim_Ft])

    # objective: maximize f_* - f_N  (same as dual minimizes)
    @objective(model, Max, Ft' * a_vec(-1, N, 𝐟))

    # interpolation constraints — one per active (i,j,m) triple
    for t in active_idx
        i = t.i
        j = t.j
        ε = ε_set[i, j, t.m]
        L_eps = (p < 1) ? L_ε(ε, p) : L

        @constraint(model,
            Ft' * a_vec(i, j, 𝐟) +
            tr(G * A_mat(i, j, α, 𝐠, 𝐱)) +
            (1 / (2 * L_eps)) * tr(G * C_mat(i, j, 𝐠)) -
            ε / 2 <= 0
        )
    end

    # initial condition
    @constraint(model, tr(G * B_mat(0, -1, α, 𝐱)) <= R^2)

    # solve
    show_output == :off && set_silent(model)
    JuMP.optimize!(model)

    if termination_status(model) ∉ acceptable_termination_statuses
        @error "solve_primal_PEP: not optimal — status = " termination_status(model)
        return Inf, nothing, nothing
    end

    p_star  = objective_value(model)
    G_star  = value.(G)
    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star
end

function solve_primal_with_known_stepsizes_bounded_subgrad(N, β, h, R; show_output=:off)
    # data generator
    # --------------

    𝐱, 𝐠, 𝐟 = data_generator_function(N, 1, h; input_type=:stepsize_constant)
    # number of points etc
    # --------------------

    I_N_star = -1:N
    dim_G = N + 2
    dim_Ft = N + 1


    # define the model
    # ----------------

    model_primal_PEP_bounded_subgrad = Model(optimizer_with_attributes(Mosek.Optimizer))

    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model_primal_PEP_bounded_subgrad, G[1:dim_G, 1:dim_G], PSD)

    # construct Ft (this is just transpose of F)
    @variable(model_primal_PEP_bounded_subgrad, Ft[1:dim_Ft])

    # define objective
    # ----------------

    @objective(model_primal_PEP_bounded_subgrad, Max,
        Ft' * a_vec(-1, N, 𝐟)
    )

    # interpolation constraint
    # ------------------------

    for i in I_N_star, j in I_N_star
        if i != j
            @constraint(model_primal_PEP_bounded_subgrad, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, h, 𝐠, 𝐱)) <= 0
            )
            @constraint(model_primal_PEP_bounded_subgrad,  tr(G * C_mat(i, j, 𝐠)) <= β^2)
        end
    end


    # initial condition
    # -----------------

    @constraint(model_primal_PEP_bounded_subgrad, tr(G * B_mat(0, -1, h, 𝐱)) <= R^2)

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_primal_PEP_bounded_subgrad)
    end

    JuMP.optimize!(model_primal_PEP_bounded_subgrad)

    # store and return the solution
    # -----------------------------

    if termination_status(model_primal_PEP_bounded_subgrad) != MOI.OPTIMAL
        # @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    p_star = objective_value(model_primal_PEP_bounded_subgrad)

    G_star = value.(G)

    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star

end

function solve_primal_with_known_stepsizes_Lipschitz(N, β, h, R; show_output=:off)
    # data generator
    # --------------

    𝐱, 𝐠, 𝐟 = data_generator_function(N, 1, h; input_type=:stepsize_constant)
    # number of points etc
    # --------------------

    I_N_star = -1:N
    dim_G = N + 2
    dim_Ft = N + 1


    # define the model
    # ----------------

    model_primal_PEP_bounded_subgrad = Model(optimizer_with_attributes(Mosek.Optimizer))

    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model_primal_PEP_bounded_subgrad, G[1:dim_G, 1:dim_G], PSD)

    # construct Ft (this is just transpose of F)
    @variable(model_primal_PEP_bounded_subgrad, Ft[1:dim_Ft])

    # define objective
    # ----------------

    @objective(model_primal_PEP_bounded_subgrad, Max,
        Ft' * a_vec(-1, N, 𝐟)
    )

    # interpolation constraint
    # ------------------------

    for i in I_N_star, j in I_N_star
        if i != j
            @constraint(model_primal_PEP_bounded_subgrad, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, h, 𝐠, 𝐱)) <= 0
            )
            @constraint(model_primal_PEP_bounded_subgrad,  tr(G * C_mat(i, -1, 𝐠)) <= β^2)
        end
    end


    # initial condition
    # -----------------

    @constraint(model_primal_PEP_bounded_subgrad, tr(G * B_mat(0, -1, h, 𝐱)) <= R^2)

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_primal_PEP_bounded_subgrad)
    end

    JuMP.optimize!(model_primal_PEP_bounded_subgrad)

    # store and return the solution
    # -----------------------------

    if termination_status(model_primal_PEP_bounded_subgrad) != MOI.OPTIMAL
        # @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    p_star = objective_value(model_primal_PEP_bounded_subgrad)

    G_star = value.(G)

    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star

end


# Following function uses batch of ε to set inexact constraints for set
function solve_primal_with_known_stepsizes_batch(N, L, α, R, ε_set, p, μ, sparsity_pattern; show_output=:off)
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

    # TODO rename patterns later
    if sparsity_pattern == "none"
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

    elseif sparsity_pattern == "OGM" # here there are M(2N+1) constraints, for each m, we have i,i+1 and ⋆,i 
        for m in 1:length(ε_set)
            if p < 1
                L_eps = ((1 - p) / (1 + p) * 1 / ε_set[m])^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))
            else
                L_eps = L
            end

            # interpolation constraint
            # ------------------------
            for i in 0:N-1
                
                j = i+1 

                @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, α, 𝐠, 𝐱)) + ((1 / (2 * (L_eps))) * tr(G * C_mat(i, j, 𝐠))) - ε_set[m] / 2 <= 0)
                                

            end

            for j in 0:N

                i = -1

                @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, α, 𝐠, 𝐱)) + ((1 / (2 * (L_eps))) * tr(G * C_mat(i, j, 𝐠))) - ε_set[m] / 2 <= 0)

            end
        end

    elseif sparsity_pattern == "single step" # here each ε handles a different constraint
        m = 1

        # interpolation constraint
        # ------------------------


        for i in 0:N-1
            L_eps = ((1 - p) / (1 + p) * 1 / ε_set[m])^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))

            j = i+1 

            @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, α, 𝐠, 𝐱)) + ((1 / (2 * (L_eps))) * tr(G * C_mat(i, j, 𝐠))) - ε_set[m] / 2 <= 0)
            
            m = m+1

        end

        for j in 0:N
            L_eps = ((1 - p) / (1 + p) * 1 / ε_set[m])^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))

            i = -1

            @constraint(model_primal_PEP_with_known_stepsizes, Ft' * a_vec(i, j, 𝐟) + tr(G * A_mat(i, j, α, 𝐠, 𝐱)) + ((1 / (2 * (L_eps))) * tr(G * C_mat(i, j, 𝐠))) - ε_set[m] / 2 <= 0)

            m = m+1
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

    JuMP.optimize!(model_primal_PEP_with_known_stepsizes)

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
    show_output          = :off,
    ϵ_tol_feas           = 1e-6,
    objective_type       = :default,
    obj_val_upper_bound  = default_obj_val_upper_bound,
    acceptable_termination_statuses = [MOI.OPTIMAL, MOI.SLOW_PROGRESS]  # ← add this
    )

    # ... all existing code unchanged ...

    M = size(ε_set, 3)          # last axis is m ∈ 1:M

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, α; input_type = :stepsize_constant)

    I_N_star = -1:N
    dim_Z    = N + 2

    model = Model(optimizer_with_attributes(Mosek.Optimizer))

    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star, M)

    @variable(model, λ[idx_set_λ] >= 0)
    @variable(model, ν >= 0)
    @variable(model, Z[1:dim_Z, 1:dim_Z], PSD)

    function L_ε(ε, p)
        return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))
    end

    if objective_type == :default
      #  @info "[🐒] Minimizing the usual performance measure"
        # ε_set now indexed by (i, j, m) — each pair has its own ε
        @objective(model, Min,
            ν * R^2 +
            sum(λ[t] * ε_set[t.i, t.j, t.m] / 2 for t in idx_set_λ))
    end

    # linear constraint — unchanged
    @constraint(model,
        sum(λ[t] * a_vec(t.i, t.j, 𝐟) for t in idx_set_λ) - a_vec(-1, N, 𝐟) .== 0)

    # LMI constraint — ε_set[i,j,m] replaces ε_set[m]
    @constraint(model,
        ν * B_mat(0, -1, α, 𝐱) +
        sum(λ[t] * A_mat(t.i, t.j, α, 𝐠, 𝐱) for t in idx_set_λ) +
        sum((1 / (2 * L_ε(ε_set[t.i, t.j, t.m], p))) *
            λ[t] * C_mat(t.i, t.j, 𝐠)
            for t in idx_set_λ)
        .== Z)

    for idx in zero_idx
        @constraint(model, λ[i_j_m_idx(idx...)] == 0)
    end

    show_output == :off && set_silent(model)

    JuMP.optimize!(model)

    # replace the old status check with:
    if termination_status(model) ∉ acceptable_termination_statuses
       # @error "solve_dual_PEP: not optimal — status = " termination_status(model)
        return Inf, nothing, nothing   # ← return Inf cleanly instead of erroring
    end



    λ_opt   = value.(λ)
    ν_opt   = value.(ν)
    Z_opt   = value.(Z)

    obj = ν_opt * R^2 +
          sum(λ_opt[t] * ε_set[t.i, t.j, t.m] / 2 for t in idx_set_λ)

    return obj, λ_opt, Z_opt
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
    #### issues may arise if μ ≠ 0, since we took away scaling by L

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


# function get_λ_matrices(λ_opt, N, M, TOL)
#     λ_matrices = zeros(N + 2, N + 2, M)
#     for m = 1:M
#         for i in -1:N
#             for j in -1:N
#                 if i == j
#                     continue
#                 end
#                 if λ_opt[i_j_m_idx(i,j,m)] > TOL
#                 λ_matrices[i+2,j+2,m] = λ_opt[i_j_m_idx(i,j,m)]
#                 end
#             end
#         end
#     end
#     return λ_matrices
# end

# function run_batch(prob::OptimizationProblem, args, sparsity_pattern)
#     N, L, R, p, k = prob.N, prob.L, prob.R, prob.p, prob.k
    
#     H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
#     cnt = 0
#     for i in 1:N
#         for j in 0:i-1

#             H[i, j] = args[k+1+cnt]
#             cnt += 1
#         end
#     end
#     ε_set = args[1:k]
#     μ = 0
#     α = compute_α_from_h(prob, H, μ)
#     Y, _, _ = solve_primal_with_known_stepsizes_batch(N, L, α, R, ε_set, p, μ, sparsity_pattern; show_output=:off)
#     return Y
# end


# function optimize_ε_h(N, L, R, p, k, max_iter, max_time, lower, upper, initial_vals, sparsity_pattern)

#     prob = OptimizationProblem(N, L, R, p, k)
   
#     terminated_early = false

#     run_batch(prob, initial_vals, sparsity_pattern)

#     iter_print_freq = 100

#     function my_callback(state)

#         if state.iteration % iter_print_freq == 0
#             println("Iter $(state.iteration): f = $(state.value)")
#         end
#         if state.value < 0
#             println("Terminating early: negative objective detected (f = $(state.value))")
#             terminated_early = true

#             return terminated_early
#         end
#         return false
#     end

#     options = Optim.Options(
#         iterations=max_iter,
#         f_tol= 1e-12,
#         x_tol= 1e-12,
#         time_limit=max_time,
#         show_trace=false,
#         callback=my_callback
#     )

#     result = Optim.optimize(args -> run_batch(prob, args, sparsity_pattern), lower, upper, initial_vals, Fminbox(NelderMead()), options)
#     return result, terminated_early
# end



# function Optimize(N, L, R, p, M, initial_vals, sparsity_pattern, max_iter, max_time)
#     T = 0
#     H_size = div(N * (N + 1), 2)
#     lower = 0.00000001 * [ones(M); ones(H_size)]
#     upper = 2 * [ones(M); ones(H_size)]
#    # prob = OptimizationProblem(N, L, R, p, M)
#     result, terminated_early = optimize_ε_h(N, L, R, p, M, max_iter, max_time, lower, upper, initial_vals, sparsity_pattern)
#     minimizer = Optim.minimizer(result)
#     H = get_H(N, M, minimizer)

#     ε_set = minimizer[1:M]
#     return ε_set, H, result, terminated_early

# end

 
# function run_batch_trials(N, L, R, p, M, trials, sparsity_pattern, max_iter, max_time)

#     min_F = Inf
#     min_H = nothing
#     min_ε = nothing
#     H_size = div(N * (N + 1), 2)
#     OGM = get_OGM_vector(N)
#     for i in 1:trials
#         initial_vals = [0.03 * rand(M) .+ 0.002; OGM ./ (1.5 .+ 0.1 * rand(H_size))]
#         ε_set, H, result, terminated_early = Optimize(N, L, R, p, M, initial_vals, sparsity_pattern, max_iter, max_time)
#         T = Optim.f_calls(result)
#         println("Trial $i done")
#         if !terminated_early
#             println(ε_set)
#             display(H)
#             if Optim.minimum(result) < min_F
#                 min_F = Optim.minimum(result)
#                 min_H = H
#                 min_ε = ε_set
#             end
#         end
#     end

#     return min_F, min_H, min_ε
# end



# function compute_theta(N)  ## Ben's Code
#     θ = zeros(N + 1)
#     θ[1] = 1.0  # θ₀
#     for i in 2:N
#         θ[i] = (1 + sqrt(1 + 4 * θ[i-1]^2)) / 2
#     end
#     θ[N+1] = (1 + sqrt(1 + 8 * θ[N]^2)) / 2  # θ_N
#     return θ
# end
# function compute_H(N)
#     θ = compute_theta(N)
#     H = zeros(N, N)
#     for i in 1:N
#         for k in 1:i-1
#             sum_hjk = sum(H[j, k] for j in k:i)
#             H[i, k] = (1 / θ[i+1]) * (2 * θ[k] - sum_hjk)
#         end
#         H[i, i] = 1 + (2 * θ[i] - 1) / θ[i+1]
#     end
#     return H
# end



# function get_OGM_vector(N)
#     OGM = compute_H(N)
#     vec_lower = zeros(div(N * (N + 1), 2))
#     cnt = 0
#     for i in 1:N
#         for j in 1:i
#             cnt = cnt + 1
#             vec_lower[cnt] = OGM[i, j]
#         end
#     end
#     return vec_lower
# end


# function get_vector(N,H)
#     vec_lower = zeros(div(N * (N + 1), 2))
#     cnt = 0
#     for i in 1:N
#         for j in 1:i
#             cnt = cnt + 1
#             vec_lower[cnt] = H[i, j]
#         end
#     end
#     return vec_lower
# end


# function gen_data(L, R, trials, p_cnt, N_cnt, M_cnt)

#     results = Dict{Int, Dict{Int, Dict{String, Any}}}()

#     for N = 1:N_cnt
#         results[N] = Dict{Int, Dict{String, Any}}()

#         for M = 1:N+M_cnt  # change back to just 1:M_cnt
#             ε_p_data = Dict{Float64, Any}()
#             F_p_data = Dict{Float64, Any}()
#             H_p_data = Dict{Float64, Any}()

#             for p in LinRange(0.9, 1, p_cnt)
#                 min_F, min_H, min_ε = run_batch_trials(N, L, R, p, M, trials)

#                 ε_p_data[p] = min_ε
#                 F_p_data[p] = min_F
#                 H_p_data[p] = min_H
#             end

#             results[N][M] = Dict(
#                 "F_values" => F_p_data,
#                 "ε_sets" => ε_p_data,
#                 "H_matrices" => H_p_data
#             )
#         end
#     end
#     return results
# end

# function L_eps_p(ε,p)

#     return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * (L)^(2 / (1 + p))

# end


# N = 1
# L = 1
# R = 1
# p = 1
# M = 2*N + 1
# initial_vals = 0.1*ones(2*M+1)
# sparsity_pattern = "OGM"
# ε_set, H, result, terminated_early = Optimize(N, L, R, p, M, initial_vals, sparsity_pattern, 1000, 60)