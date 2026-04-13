using Optim, Plots, ForwardDiff, CurveFit

include("../Solvers/BnB_PEP_Inexact_Smooth.jl")

function L_smooth(δ, β, p)
    return ((1 - p) / (1 + p) * 1 / δ)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(N, L, δ_set)
    T = eltype(δ_set)  # determines if Float64 or Dual
    δ_i_j = δ_set[1:N]
    δ_star_i = δ_set[N+1:2*N+1]



    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)
    α_set = zeros(T, N + 1)

    α_set[1] = 1 / L(δ_i_j[1]) + 1 / L(δ_star_i[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(δ_star_i[k]) + 1 / L(δ_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(δ_i_j[k-1]) + 1 / L(δ_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(δ_star_i[N+1])
    C = -λ_i_j[N] * 1 / L(δ_i_j[N])
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]



    τ = λ_star_i[N+1] + λ_i_j[N]
    δ_certificate = [δ_i_j; δ_star_i]
    λ_certificate = [λ_i_j; λ_star_i]
    σ = 1 / 2 * δ_certificate' * λ_certificate

    rate = (1 / 2 * R^2 + σ) / τ


    return rate


end

function get_H_val(N, L, δ_set)
    δ_i_j = δ_set[1:N] # [δ_0_1, δ_1_2, ..., δ_{N-1}_N]
    δ_star_i = δ_set[N+1:2*N+1]


    λ_i_j = zeros(N)
    λ_star_i = zeros(N + 1)

    α_set = zeros(N + 1)
    α_set[1] = 1 / L(δ_i_j[1]) + 1 / L(δ_star_i[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(δ_star_i[k]) + 1 / L(δ_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(δ_i_j[k-1]) + 1 / L(δ_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(δ_star_i[N+1])
    C = -λ_i_j[N] * 1 / L(δ_i_j[N])
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]


    H_certificate = zeros(N, N)

    for i in 1:N
        for j in 1:i
            if i == j
                H_certificate[i, j] = (λ_i_j[i] + L(δ_i_j[i]) * α_set[i] * α_set[i+1]) / (L(δ_i_j[i]) * (λ_i_j[i] + λ_star_i[i+1]))
            else
                H_certificate[i, j] = (α_set[i+1] * α_set[j] - 1 * λ_star_i[i+1] * sum([H_certificate[k, j] for k in j:i-1])) / (λ_i_j[i] + λ_star_i[i+1])
            end
        end
    end




    return H_certificate
end

function get_lambda(N, L, δ_set)
    T = eltype(δ_set)  # determines if Float64 or Dual


    δ_i_j = δ_set[1:N]
    δ_star_i = δ_set[N+1:2*N+1]


    if p == 1
        δ_i_j = zero(T) * δ_set[1:N]
        δ_star_i = zero(T) * δ_set[N+1:2*N+1]
    end

    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)
    α_set = zeros(T, N + 1)

    α_set[1] = 1 / L(δ_i_j[1]) + 1 / L(δ_star_i[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(δ_star_i[k]) + 1 / L(δ_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(δ_i_j[k-1]) + 1 / L(δ_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(δ_star_i[N+1])
    C = -λ_i_j[N] * 1 / L(δ_i_j[N])
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]



    λ_certificate = [λ_i_j; λ_star_i]


    return λ_certificate


end

function get_H_guess(N, L, δ_set)
    λ_set = get_lambda(N, L, δ_set)
    λ_i = λ_set[1:N]
    λ_star = λ_set[N+1:2*N+1]
    H_guess = zeros(N, N)


    for i = 1:N
        for j = 1:i

            if j == i
                H_guess[i, j] = (λ_i[i] + L(δ_set[i]) * λ_star[i] * λ_star[i+1]) / ((λ_i[i] + λ_star[i+1]) * L(δ_set[i]))
            end

            if j == i - 1
                H_guess[i, j] = (λ_star[i+1] * λ_i[i-1]) / (λ_star[i] * (λ_i[i] + λ_star[i+1])) * (H_guess[i-1, i-1] - 1 / (L(δ_set[i-1])))

            end

            if j < i - 1
                H_guess[i, j] = λ_star[i+1] / (λ_i[i] + λ_star[i+1]) * (λ_i[i-1]) / λ_star[i] * H_guess[i-1, j]
            end
        end
    end
    
    # for i = 1:N-1  # uses the fact that for i < N, then λ_i[i] + λ_star[i+1] = λ_i[i+1], can combine all into one method
    #     for j = 1:i 

    #         if j == i 
    #             H_guess[i,j] = (λ_i[i] + L_delta(min_δ[i],p)*λ_star[i]*λ_star[i+1])/(λ_i[i+1] * L_delta(min_δ[i],p))
    #         end

    #         if j == i-1
    #             H_guess[i,j] = (λ_star[i+1] * λ_i[i-1])/(λ_star[i] * λ_i[i+1]) * (H_guess[i-1,i-1]-1/(L_delta(min_δ[i-1],p) )) 

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
    f = δ_set -> get_rate(N, L, δ_set)
    g! = (G, x) -> (G[:] = ForwardDiff.gradient(f, x))
    result = Optim.optimize(f, g!, lower, upper, initial_vals, Fminbox(BFGS()), options)

    rate = Optim.minimum(result)
    min_δ = Optim.minimizer(result)
    H_val = get_H_val(N, L_delta, min_δ)


    return min_δ, rate, H_val
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

function compute_α_from_h(h, N, μ, L)
    α = zeros(N, N)
    for ℓ in 1:N
        for i in 1:ℓ
            if i==ℓ
                α[ℓ,i] = h[ℓ,ℓ]
            elseif i <= ℓ-1
                α[ℓ,i] = α[ℓ-1,i] + h[ℓ,i] 
            end
        end
    end
    return α
end

function make_α_gradient_descent(N, L)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for i in 1:N, j in 0:i-1
        α[i, j] = 1.0/L
    end
    return α
end




N = 31
β = 1
p = 1.0
R = 1
L_delta = δ -> L_smooth(δ, β, p)

min_δ, rate, H_val = run_N_opti(N, L_delta)
H_guess = get_H_guess(N, L_delta, min_δ)

δ_i = min_δ[1:N]
δ_star = min_δ[N+1:2*N+1]

display(H_val)
display(H_guess)
display(δ_i)
display(δ_star)

#α = OffsetArray(compute_α_from_h(H_val, N, 0, 1), 1:N, 0:N-1)
α = make_α_gradient_descent(N, β)
raw = fill(0.0, N+2, N+2, 1)
ε_set_opt = OffsetArray(raw, -1:N, -1:N, 1:1)
ε_set_opt[-1,0:N] = δ_star
for j = 0:N-1 
    ε_set_opt[j,j+1] = δ_i[j+1]
end

display(ε_set_opt)

obj, λ_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, β, α, R, ε_set_opt, p, [];
                show_output         = :off,
                ϵ_tol_feas          = 1e-8,
                objective_type      = :default,
                obj_val_upper_bound = 1e6)