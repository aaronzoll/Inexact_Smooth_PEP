using JuMP, MosekTools, Mosek, Plots

function solve_convex_program(N, ε_set)
    #currently working for p = 0

    M = 1 # Lipschitz constant
    R = 1 # distance to optimal
    ε_i = ε_set[1:N] # [ε_0_1, ε_1_2, ..., ε_{N-1}_N]
    ε_star = ε_set[N+1:2*N+1]

    # define the model
    # ----------------    
    model_rate = Model(Mosek.Optimizer)

    # add the variables
    # -----------------

    # construct t ≥ 0
    @variable(model_rate, t >= 0)

    # construct λ_{i-1,i}
    @variable(model_rate, λ[1:N] >= 0)

    # construct α_{i-1} 
    @variable(model_rate, α[1:N+1])

    # construct ε_{i-1,i}, ε_{⋆,i-1}
   # @variable(model_rate, ε_i[1:N] >= 0)
   # @variable(model_rate, ε_star[1:N+1] >= 0)

    # define objective
    # ----------------
    @objective(model_rate, Min, 0.5*R^2*t + λ'*ε_i + α'*ε_star)    

    # define constraints
    # ------------------
    
    # function value constraints
    @constraint(model_rate, λ[1]-α[1] == 0)

    for i = 2:N
        @constraint(model_rate, λ[i-1]-λ[i]-α[i] == 0)
    end

    @constraint(model_rate, λ[N]+α[N+1] == t)


    # gradient value constraints

    @constraint(model_rate, 4*M^2*α[1]^2 - t^2*(ε_i[1]+ε_star[1]) <= 0)

    for i = 2:N
       @constraint(model_rate, 4*M^2*α[i]^2 - t^2*(ε_i[i-1]+ε_i[i]+ε_star[i]) <= 0)
    end
    
    @constraint(model_rate, 4*M^2*α[N+1]^2 - t^2*(ε_i[N]+ε_star[N+1]) <= 0)




    optimize!(model_rate)

    # store and return the solution
    # -----------------------------

    if termination_status(model_rate) != MOI.OPTIMAL
        #   @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    r_star = objective_value(model_rate)

    t_star = value.(t)
    λ_star = value.(λ)
    α_star = value.(α)
    ε_i_output = value.(ε_i)    
    ε_star_output = value.(ε_star)    


    return r_star, t_star, λ_star, α_star, ε_i_output, ε_star_output

end

ε_set = [ 0.7003238270279833, 0.010274771494946114, 0.2300842232836793, 0, 0, 0, 0]
t_star, λ_star, α_star, ε_i_output, ε_star_output = solve_convex_program(3, ε_set)