using Optim, Plots, ForwardDiff, CurveFit

function L_smooth(ε, β, p)
    return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(N, L, ε_set)
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
        B = -(1 / L(ε_star_i[k]) + 1 / L(ε_i_j[k]))
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

function get_H_val(N, L, ε_set)
    ε_i_j = ε_set[1:N] # [ε_0_1, ε_1_2, ..., ε_{N-1}_N]
    ε_star_i = ε_set[N+1:2*N+1]


    λ_i_j = zeros(N)
    λ_star_i = zeros(N + 1)

    α_set = zeros(N + 1)
    α_set[1] = 1 / L(ε_i_j[1]) + 1 / L(ε_star_i[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(ε_star_i[k]) + 1 / L(ε_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(ε_i_j[k-1]) + 1 / L(ε_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(ε_star_i[N+1])
    C = -λ_i_j[N] * 1 / L(ε_i_j[N])
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]


    H_certificate = zeros(N, N)

    for i in 1:N
        for j in 1:i
            if i == j
                H_certificate[i, j] = (λ_i_j[i] + L(ε_i_j[i]) * α_set[i] * α_set[i+1]) / (L(ε_i_j[i]) * (λ_i_j[i] + λ_star_i[i+1]))
            else
                H_certificate[i, j] = (α_set[i+1] * α_set[j] - 1 * λ_star_i[i+1] * sum([H_certificate[k, j] for k in j:i-1])) / (λ_i_j[i] + λ_star_i[i+1])
            end
        end
    end




    return H_certificate
end

function get_lambda(N, L, ε_set)
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
        B = -(1 / L(ε_star_i[k]) + 1 / L(ε_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(ε_i_j[k-1]) + 1 / L(ε_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(ε_star_i[N+1])
    C = -λ_i_j[N] * 1 / L(ε_i_j[N])
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]



    λ_certificate = [λ_i_j; λ_star_i]


    return λ_certificate


end

function get_H_guess(N, L, ε_set)
    λ_set = get_lambda(N, L, ε_set)
    λ_i = λ_set[1:N]
    λ_star = λ_set[N+1:2*N+1]
    H_guess = zeros(N, N)


    for i = 1:N
        for j = 1:i

            if j == i
                H_guess[i, j] = (λ_i[i] + L(ε_set[i]) * λ_star[i] * λ_star[i+1]) / ((λ_i[i] + λ_star[i+1]) * L(ε_set[i]))
            end

            if j == i - 1
                H_guess[i, j] = (λ_star[i+1] * λ_i[i-1]) / (λ_star[i] * (λ_i[i] + λ_star[i+1])) * (H_guess[i-1, i-1] - 1 / (L(ε_set[i-1])))

            end

            if j < i - 1
                H_guess[i, j] = λ_star[i+1] / (λ_i[i] + λ_star[i+1]) * (λ_i[i-1]) / λ_star[i] * H_guess[i-1, j]
            end
        end
    end
    
    # for i = 1:N-1  # uses the fact that for i < N, then λ_i[i] + λ_star[i+1] = λ_i[i+1], can combine all into one method
    #     for j = 1:i 

    #         if j == i 
    #             H_guess[i,j] = (λ_i[i] + L_eps(min_ε[i],p)*λ_star[i]*λ_star[i+1])/(λ_i[i+1] * L_eps(min_ε[i],p))
    #         end

    #         if j == i-1
    #             H_guess[i,j] = (λ_star[i+1] * λ_i[i-1])/(λ_star[i] * λ_i[i+1]) * (H_guess[i-1,i-1]-1/(L_eps(min_ε[i-1],p) )) 

    #         end

    #         if j < i-1
    #             H_guess[i,j] = λ_star[i+1]/λ_i[i+1] * (λ_i[i-1])/λ_star[i] * H_guess[i-1,j]
    #         end
    #     end
    # end

    return H_guess
end

function run_N_opti(N, L)
    M = 2 * N + 1
    lower = 0.00000000001 * ones(M)
    upper = 3 * ones(M)
    initial_vals = 0.03 * ones(M)

    options = Optim.Options(
        iterations=2500,
        f_tol=1e-8,
        x_tol=1e-8,
        time_limit=60,
        show_trace=false,
    )
    f = ε_set -> get_rate(N, L, ε_set)
    g! = (G, x) -> (G[:] = ForwardDiff.gradient(f, x))
    result = Optim.optimize(f, g!, lower, upper, initial_vals, Fminbox(BFGS()), options)

    rate = Optim.minimum(result)
    min_ε = Optim.minimizer(result)
    H_val = get_H_val(N, L_eps, min_ε)


    return min_ε, rate, H_val
end

function OGM_rates(β, R, N)
    theta = 1
    rate = []
    for i in 0:N-1
        if i < N - 1
            theta = (1 + sqrt(1 + 4 * theta^2)) / 2
        else
            theta = (1 + sqrt(1 + 8 * theta^2)) / 2
        end

        push!(rate, β * R^2 / (2 * theta^2))
    end

    return rate
end

N = 8
β = 0.2
p = 1
R = 3
L_eps = ε -> L_smooth(ε, β, p)

min_ε, rate, H_val = run_N_opti(N, L_eps)
H_guess = get_H_guess(N, L_eps, min_ε)

