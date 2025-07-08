using JuMP, MosekTools, Mosek, Plots, Random, Revise, Ipopt, Optim

function make_L_eps(β, p)
    return ε -> 1/ε + 1 + ((1 - p)/(1 + p) * 1/ε)^((1 - p)/(1 + p)) * β^(2/(1 + p))  #nonsmooth + smooth + HS
end

function solve_convex_program(ε, ε_star, L, R)
    N = length(ε_star) - 1
    model = Model(Mosek.Optimizer)

    set_silent(model)
   
    @variable(model, t >= 1e-10)
    @variable(model, λ[0:N, 0:N] >= 0)
    @variable(model, α[0:N] >= 0)
    @variable(model, s[0:N] >= 0)

    @objective(model, Min,
        0.5 * R^2 * t +
        sum(λ[i,j] * ε[i+1,j+1]/2 for i in 0:N, j in 0:N) +
        sum(α[i] * ε_star[i+1]/2 for i in 0:N)
    )

    for i in 0:N
        @constraint(model, λ[i,i] == 0)


        for j in 0:i
            @constraint(model, λ[i,j] == 0)
        end
    end



    for j in 0:N-1
        @constraint(model, sum(-λ[i,j] + λ[j,i] for i in 0:N) == α[j])
    end
    @constraint(model, sum(λ[i,N] for i in 0:N-1) == sum(α[0:N-1]))

    for j in 0:N
        sum_term = sum(-λ[i,j]/L(ε[i+1,j+1]) - λ[j,i]/L(ε[j+1,i+1]) for i in 0:N)
        linear_term = -α[j]/L(ε_star[j+1])
        @constraint(model, sum_term + linear_term + 1/2*s[j] <= 0)

        @constraint(model, [t; s[j]; 2*α[j]] in MOI.RotatedSecondOrderCone(3))
    end

    @constraint(model, sum(α[i] for i in 0:N) == 1)

    optimize!(model)

    return objective_value(model), value(t), value.(λ), value.(α)
end

function get_rate(N, M, L, p, ε_set)
    T = eltype(ε_set)  # determines if Float64 or Dual


        ε_i_j = ε_set[1:N]
        ε_star_i = ε_set[N+1:2*N+1]


    if p == 1
        ε_i_j = zero(T) * ε_set[1:N]
        ε_star_i = zero(T) * ε_set[N+1:2*N+1]
    end

    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)
    α_set = zeros(T, N + 1)

    α_set[1] = 1 / L(ε_i_j[1]) + 1 / L(ε_star_i[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(ε_star_i[k]) + 1 / L(ε_i_j[k],))
        C = -λ_i_j[k-1] * (1 / L(ε_i_j[k-1]) + 1 / L(ε_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(ε_star_i[N+1])
    C = -λ_i_j[N] * 1 / L(ε_i_j[N])
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]



    τ = λ_star_i[N+1] + λ_i_j[N]
    ε_certificate = [ε_i_j; ε_star_i]
    λ_certificate = [λ_i_j; λ_star_i]
    σ = 1 / 2 * ε_certificate' * λ_certificate

    rate = (1 / 2 * R^2 + σ) / τ


    return rate


end

function run_N_opti(N, p)
    M = 2 * N + 1
    lower = 0.00000000001 * ones(M)
    upper = 3 * ones(M)
    initial_vals = 0.3 * ones(M)
    iter_print_freq = 100

    function my_callback(state)

        if state.iteration % iter_print_freq == 0 && state.iteration > 1
            println("Iter $(state.iteration): f = $(state.value)")
        end
        if state.value < 0
            println("Terminating early: negative objective detected (f = $(state.value))")
            terminated_early = true

            return terminated_early
        end
        return false
    end

    options = Optim.Options(
        iterations=2500,
        f_tol=1e-8,
        x_tol=1e-8,
        time_limit=60,
        show_trace=false,
        callback = my_callback
    )

    f = ε_set -> get_rate(N, M, L_eps, p, ε_set)
    result = Optim.optimize(f, lower, upper, initial_vals, Fminbox(NelderMead()), options)

    min_ε = Optim.minimizer(result)


    rate = Optim.minimum(result)


    return min_ε, rate
end

function solve_convex_program_OGM()

    model = Model(Ipopt.Optimizer)
    
    @variable(model, λ[0:N, 0:N] >= 0)
    @variable(model, α[0:N] >= 0)

    @objective(model, Max,
        sum(α[i] for i in 0:N)
    )

    for i in 0:N
        @constraint(model, λ[i,i] == 0)


        for j in 0:i
            @constraint(model, λ[i,j] == 0)
        end
    end

    for j in 0:N-1
        @constraint(model, sum(-λ[i,j] + λ[j,i] for i in 0:N) == α[j])
    end
    @constraint(model, sum(λ[i,N] for i in 0:N-1) == sum(α[0:N-1]))

    for j in 0:N
        sum_term = sum(-λ[i,j]/β - λ[j,i]/β for i in 0:N)
        quad_term = -α[j]/β + α[j]^2
        @constraint(model, sum_term + quad_term  <= 0)

    end


    optimize!(model)

    return objective_value(model), value.(α), value.(λ)
end

function run_wizard_opti(N, p, L, R)
    M_eps = (N + 1)^2             # number of elements in ε matrix
    M_star = N + 1                # number of elements in ε_star
    M = M_eps + M_star            # total number of optimization variables

    lower = 1e-10 * ones(M)
    upper = 3.0 * ones(M)
    initial_vals = 0.031 * ones(M)

    iter_print_freq = 100

    function my_callback(state)

        if state.iteration % iter_print_freq == 0 && state.iteration > 1
            println("Iter $(state.iteration): f = $(state.value)")
        end
        if state.value < 0
            println("Terminating early: negative objective detected (f = $(state.value))")
            terminated_early = true

            return terminated_early
        end
        return false
    end

    options = Optim.Options(
        iterations=25000,
        f_tol=1e-8,
        x_tol=1e-8,
        time_limit=60,
        show_trace=false,
        callback = my_callback
    )

    f = function(x)
        ε_flat = x[1:M_eps]
        ε_star = x[M_eps+1:end]

        ε = reshape(ε_flat, N+1, N+1)
        obj_val, _, _, _ = solve_convex_program(ε, ε_star, L, R)
        return obj_val
    end

    result = Optim.optimize(f, lower, upper, initial_vals, Fminbox(NelderMead()), options)

    x_opt = Optim.minimizer(result)
    ε_opt = reshape(x_opt[1:M_eps], N+1, N+1)
    ε_star_opt = x_opt[M_eps+1:end]
    rate = Optim.minimum(result)

    return ε_opt, ε_star_opt, rate
end

# === Run Example ===
N = 4
ε = zeros(N+1, N+1)
ε_star = zeros(N+1)
β = 1
p = 0.5
L_eps = make_L_eps(β, p)
R = 1

# compute OGM sparsity optimal epsilons
min_ε_sparse, rate_sparse = run_N_opti(N, p)

# set those for Wizard run (just for sanity check)
for i = 1:N
    ε[i,i+1] = min_ε_sparse[i]
    ε_star[i] = min_ε_sparse[N+i]
end
ε_star[N+1] = min_ε_sparse[2*N+1]

rate, t, λ, α = solve_convex_program(ε, ε_star, L_eps, R)

# optimize wizard run
ε_opt, ε_star_opt, rate2 = run_wizard_opti(N, p, L_eps, R)
rate_wizard, t1, λ1, α1 = solve_convex_program(ε_opt, ε_star_opt, L_eps, R)

display(rate_sparse)
display(rate)
display(rate_wizard)


# OGM check 
#-----------------

# τ, α, λ_1 = solve_convex_program_OGM()
# display(0.5 * R^2/τ)

# function OGM_rates(β, R, N)

#     theta = 1
#     rate = []
#     for i in 0:N-1
#         if i < N - 1
#             theta = (1 + sqrt(1 + 4 * theta^2)) / 2
#         else
#             theta = (1 + sqrt(1 + 8 * theta^2)) / 2
#         end

#         push!(rate, β * R^2 / (2 * theta^2))
#     end

#     return rate
# end

# rate = OGM_rates(β, R, N)
# display(rate[N])