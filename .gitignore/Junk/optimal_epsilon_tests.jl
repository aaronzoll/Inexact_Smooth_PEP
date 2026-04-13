using Random, Revise

include("wizardry_solver.jl")

function make_L_eps(β, p)
    return δ ->  ((1 - p)/(1 + p) * 1/δ)^((1 - p)/(1 + p)) * β^(2/(1 + p))  #nonsmooth + smooth + HS
end

# === Run Example ===
N = 4
δ = zeros(N+1, N+1)
δ_star = zeros(N+1)
β = 1
p = 0.00000001
L_eps = make_L_eps(β, p)
R = 1

# compute OGM sparsity optimal epsilons
min_δ_sparse, rate_sparse = run_N_opti(N, p)
display(min_δ_sparse)
display(rate_sparse)
# # set those for Wizard run (just for sanity check)
# for i = 1:N
#     δ[i,i+1] = min_δ_sparse[i]
#     δ_star[i] = min_δ_sparse[N+i]
# end
# δ_star[N+1] = min_δ_sparse[2*N+1]

# rate, t, λ, α = solve_convex_program(δ, δ_star, L_eps, R)

# # optimize wizard run
# δ_opt, δ_star_opt, rate2 = run_wizard_opti(N, p, L_eps, R)
# rate_wizard, t1, λ1, α1 = solve_convex_program(δ_opt, δ_star_opt, L_eps, R)

# display(rate_sparse)
# display(rate)
# display(rate_wizard)


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