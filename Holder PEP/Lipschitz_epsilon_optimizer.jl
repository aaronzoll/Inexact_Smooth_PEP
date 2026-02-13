using JuMP, MosekTools, Mosek, Plots, Random, Revise, Ipopt, Optim, CurveFit, JLD2


function get_rate_Lip(N, β, δ_set)
    T = eltype(δ_set)  # determines if Float64 or Dual
    δ_i_j = δ_set[1:N]

    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)
    α_set = zeros(T, N + 1)

    α_set[1] = δ_i_j[1] / β^2
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]

    for k = 2:N
        B = -δ_i_j[k]/β^2
        C = -λ_i_j[k-1] * (δ_i_j[k-1]/β^2 + δ_i_j[k]/β^2)
        λ_star_i[k] = (-B + sqrt(B^2 - 4 * C)) / (2)

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end

    λ_star_i[N+1] = sqrt(λ_i_j[N] * δ_i_j[N]/β^2) 
    α_set[N+1] = λ_star_i[N+1]


    τ = λ_star_i[N+1] + λ_i_j[N]
    δ_certificate = [δ_i_j; zeros(N + 1)]
    λ_certificate = [λ_i_j; λ_star_i]
    σ = 1 / 2 * δ_certificate' * λ_certificate

    rate = (1 / 2 * R^2 + σ) / τ
    return rate
end



function run_N_opti(N, β)
    lower = 0.00000000001 * ones(N)
    upper = 3 * ones(N)
    initial_vals = 0.3 * ones(N)
    iter_print_freq = 500

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
        iterations=5000,
        f_tol=1e-10,
        x_tol=1e-10,
        time_limit=60,
        show_trace=false,
        callback=my_callback
    )

    f = δ_set -> get_rate_Lip(N, β, δ_set)
    result = Optim.optimize(f, lower, upper, initial_vals, Fminbox(NelderMead()), options)

    min_δ = Optim.minimizer(result)


    rate = Optim.minimum(result)


    return min_δ, rate
end


function get_H_val(N, β, δ_set)
    δ_i_j = δ_set[1:N]

    λ_i_j = zeros(N)
    λ_star_i = zeros(N + 1)
    α_set = zeros(N + 1)

    α_set[1] = δ_i_j[1] / β^2
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]

    for k = 2:N
        B = -δ_i_j[k] / β^2
        C = -λ_i_j[k-1] * (δ_i_j[k-1] / β^2 + δ_i_j[k] / β^2)
        λ_star_i[k] = (-B + sqrt(B^2 - 4 * C)) / (2)

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end

    λ_star_i[N+1] = sqrt(λ_i_j[N] * δ_i_j[N] / β^2)
    α_set[N+1] = λ_star_i[N+1]

    H_certificate = zeros(N, N)

    for i in 1:N
        for j in 1:i
            if i == j
                H_certificate[i, j] = (δ_i_j[i] * λ_i_j[i] / β^2 + α_set[i] * α_set[i+1]) / (λ_i_j[i] + λ_star_i[i+1]) * β
            else
                H_certificate[i, j] = (α_set[i+1] * α_set[j] - 1 / β * λ_star_i[i+1] * sum([H_certificate[k, j] for k in j:i-1])) / (λ_i_j[i] + λ_star_i[i+1]) * β
            end
        end
    end

    return H_certificate
end


function make_L_eps(β)
    return δ -> (β^2 / δ)
end



# === Run Example ===

# for given key "N" saves (a_odd, b_odd, a_even, b_even)
# where for "odd epsilons" i.e. δ_{2i,2i+1} has relationship
# a_odd * N^b_odd and evens have a_even + b_even * N
# results = Dict{Int, Any}() 
function compute_δ_set(N)
    δ_set = zeros(N)

    for i = 1:N
        if i % 2 == 1
            δ_set[i] = β * R * sqrt(2) * (N + 1)^-0.5 / (i)

        else
            #  δ_set[i] = max(0,(-1.33*10^-5 - 8.2*10^-7*N)*i + (0.00005*N-0.0002))
            δ_set[i] = 0
           #  δ_set[i] = β * R * sqrt(2) * (N + 1)^-0.5 / (i)
        end


    end

    return δ_set
end


N = 3
β = 1
R = 1

function sweep_rates(N, β, bounds, counts, TOL)
    axes = ntuple(d -> LinRange(bounds[1], bounds[2], counts), N)
    g = Iterators.product(axes...)
    results = Dict{NTuple{N, Float64}, Float64}()

    for δ_set in g
        rate = get_rate_Lip(N, β, collect(δ_set))
        if abs(rate - sqrt(2)/2*β*R/sqrt(N+1)) < TOL
            results[δ_set] = rate 
        end
    end

    return results
end


results = sweep_rates(N, β, [0.0,1], 100, 1e-4)
for δ_set in keys(results)
    H_val = get_H_val(N, β, collect(δ_set))
    display(H_val)
end

# # compute OGM sparsity optimal epsilons
# min_δ_sparse, rate_sparse = run_N_opti(N, β)
# display(min_δ_sparse)
# display(rate_sparse)

# odds = min_δ_sparse[1:2:N]
# evens = min_δ_sparse[2:2:N]

# odds_x = 1:2:N
# evens_x = 2:2:N
# scatter(odds_x, odds)
# scatter!(evens_x, evens)

# a_odd, b_odd = power_fit(odds_x, odds)
# a_even, b_even = linear_fit(evens_x, evens)

# x_range = LinRange(1, N, 1000)
# plot!(x_range, @. a_odd * x_range^b_odd)
# plot!(x_range, @. a_even + b_even * x_range)

# #   results[N] = (a_odd, b_odd, a_even, b_even)


