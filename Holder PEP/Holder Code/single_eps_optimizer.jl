using Optim, Plots, ForwardDiff

function L_eps(ε, p)
    return ((1 - p) / (1 + p) * 1 / (ε))^((1 - p) / (1 + p)) * β^(2 / (1 + p))
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
    # note, σ = 0.5 * (sum(λ_i_j) + τ) * ε
    rate = (0.5 * R^2 + σ) / τ

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
    f = x -> get_rate(N, L_eps, p, x[1])
    g! = (G, x) -> (G[1] = ForwardDiff.derivative(z -> get_rate(N, L_eps, p, z), x[1]))
    result = Optim.optimize(f, g!, lower, upper, initial_vals, Fminbox(LBFGS()), options)
    min_ε = Optim.minimizer(result)[1]
    rate = Optim.minimum(result)

    return min_ε, rate
end

function get_sequences(N)
    a = zeros(N+1)
    b = zeros(N)

   
    a[1] = 2
    b[1] = 2    
    for i in 2:N 
        a[i] = 1 + sqrt(1+2*b[i-1])    
        b[i] = a[i] + b[i-1]
    end

    a[N+1] = 0.5 * (1 + sqrt(1 + 4*b[N]))
    return a, b
end

function plot_N_rates(R, β, p, k)
    X = 1:k
    Y1 = zeros(k)
    ε_set = zeros(k)
    s = zeros(k)
    ε_calc = zeros(k)
    for (cnt, N) in enumerate(X)
        ε_set[cnt], Y1[cnt] = run_N_opti(N, R, β, p) # PRINTING OFF
        if mod(N, 50) == 0
            println("trial: $N")
        end
        a, b = get_sequences(N)
        s[cnt] = (b[N]+a[N+1]+sum(b))/R^2
        ε_calc[cnt] = (1-p)/(1+p)*β*s[cnt]^(-(1+p)/2)
    end
    coeff = (Y1 ./ (β * R^(1 + p) .* X .^ (-(1 + 3 * p) / 2)))
    plot(title="β = $β, R = $R, p = $p")
    plot(X, Y1, labels="same ε")
    return ε_set, coeff, ε_calc, Y1
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
k = 20 # number of points to test
p = 0.5
N = 1000

# coeff = plot_N_rates(R, β, p, k)
plot_N_rates(R, β, p, k)
# plot!(X_range, Y_range, labels="$K * βR^(1+p)/N^(-(1+3p)/2)", ylims = (0,1))

p_range, coeffs, min_ε_sets = plot_p_coeff(R, β, k, N)
plot(p_range, coeffs)
plot!(p_range, @. 2/(1+p_range)*6^((p_range-1)/2))
# display(coeffs)
# display(min_ε_sets)
# plot(title="β = $β, R = $R, N = $N, plotting: Coefficients")
# plot!(p_range,  coeffs)
