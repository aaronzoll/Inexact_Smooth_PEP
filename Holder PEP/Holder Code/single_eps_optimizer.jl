using Optim, Plots, ForwardDiff

function L_eps(ε, p)
    return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(N, L, p, ε)
    T = typeof(ε)
    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)

    M_ε = 1 ./ L.(ε, p)

    λ_i_j[1] = 2 * M_ε
    λ_star_i[1] = 2 * M_ε

    for k = 2:N
        B = -2 * M_ε
        C = -λ_i_j[k-1] * 2 * M_ε
        λ_star_i[k] = 0.5 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
    end

    B = -M_ε
    C = -λ_i_j[N] * M_ε
    λ_star_i[N+1] = 0.5 * (-B + sqrt(B^2 - 4 * C))

    τ = λ_star_i[N+1] + λ_i_j[N]
    σ = 0.5 * (sum(λ_i_j) + sum(λ_star_i)) * ε
    rate = (0.5 * R^2 + σ) / τ

    println(λ_i_j)
    println(λ_star_i)
    return rate
end

function run_N_opti(N, R, β, p)
    M = 2 * N + 1
    initial_vals = [1e-1]
    lower = [1e-16]
    upper = [5.0]

    options = Optim.Options(
        iterations=25000,
        f_tol=1e-14,
        x_tol=1e-14,
        time_limit=60,
        show_trace=false,
    )
    f = x -> get_rate(N, M, L_eps, p, x[1])
    g! = (G, x) -> (G[1] = ForwardDiff.derivative(z -> get_rate(N, L_eps, p, z), x[1]))
    result = Optim.optimize(f, g!, lower, upper, initial_vals, Fminbox(LBFGS()), options)
    min_ε = Optim.minimizer(result)
    rate = Optim.minimum(result)

    return min_ε, rate
end

function plot_N_rates(R, β, p, k)
    X = 1:k
    Y1 = zeros(k)

    for (cnt, N) in enumerate(X)
        min_ε1, Y1[cnt] = run_N_opti(N, R, β, p) # PRINTING OFF
        if mod(N, 50) == 0
            println("trial: $N")
        end
    end
    coeff = (Y1 ./ (β * R^(1 + p) .* X .^ (-(1 + 3 * p) / 2)))
    plot(title="β = $β, R = $R, p = $p, plotting: $plotting_type")
    plot!(X, Y1, labels="same ε", xaxis=:log, yaxis=:log)

    return coeff
end

function plot_p_coeff(R, β, k, N=100)
    p_range = LinRange(1e-8, 1, k)
    coeffs = zeros(k)
    min_ε_sets = zeros(k)

    for (cnt, p) in enumerate(p_range)
        min_ε, rate = run_N_opti(N, R, β, p) # PRINTING OFF
        min_ε_sets[cnt] = min_ε[1]
        coeffs[cnt] = (rate ./ (β * R^(1 + p) .* N .^ (-(1 + 3 * p) / 2)))
        if mod(cnt, 5) == 0
            println("trial: $cnt")
        end

    end

    return p_range, coeffs, min_ε_sets
end

R = 3
β = 0.3
k = 100 # number of points to test
p = 0.6
N = 1000

coeff = plot_N_rates(R, β, p, k)
X_range = LinRange(1, k, k)
Y_range = coeff[end] * β * R^(1 + p) ./ ((X_range) .^ ((1 + 3 * p) / 2))
K = round(coeff[end], digits=4)

plot!(X_range, Y_range, labels="$K * βR^(1+p)/N^(-(1+3p)/2)", xaxis=:log, yaxis=:log)

# p_range, coeffs, min_ε_sets = plot_p_coeff(R, β, k, N)
# plot(title="β = $β, R = $R, N = $N, plotting: Coefficients")
# plot!(p_range,  coeffs)