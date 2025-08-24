using Plots, LinearAlgebra
function get_rate_Lip(N, β, δ_set)
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


    τ = λ_star_i[N+1] + λ_i_j[N]
    δ_certificate = [δ_i_j; zeros(N + 1)]
    λ_certificate = [λ_i_j; λ_star_i]
    σ = 1 / 2 * δ_certificate' * λ_certificate

    rate = (1 / 2 * R^2 + σ) / τ
    return rate, λ_certificate, α_set
end


function get_H_val(N, β, δ_set, λ_set, α_set)
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

function compute_H_guess(N)
    H_guess = zeros(N, N)

    for i = 1:N
        for j = 1:i
            if j == i
                if j % 2 == 0
                    coeff = sqrt(2)
                else
                    coeff = 2 * sqrt(2)
                end
                H_guess[j, j] = coeff * R / ((j + 1) * sqrt(N + 1))
            else
                if j % 2 == 0
                    coeff = j
                else
                    coeff = j - 1
                end
                H_guess[i, j] = coeff * sqrt(2) * R / (i * (i + 1) * sqrt(N + 1))
            end
        end
    end

    # for i = 1:N
    #     for j = 1:i
    #         if j == i
    #             H_guess[j, j] = (mod(j,2) + 1) * sqrt(2) * R / ((j + 1) * sqrt(N + 1))

    #         else
    #             H_guess[i, j] = (j - mod(j, 2)) * sqrt(2) * R / (i * (i + 1) * sqrt(N + 1))

    #         end
    #     end
    # end

    return H_guess
end


function max_with_zero(v::AbstractVector{<:Real})
    return max(0., maximum(v))
end

function subgrad_max_with_zero(v::AbstractVector{<:Real})
    max_val = max_with_zero(v)

    if max_val == 0
        return zeros(length(v))
    else
        g = zeros(length(v))
        for (i, x) in enumerate(v)
            if x == max_val
                g[i] = β/sum(v .== max_val)
            end
        end
        return g
    end
end

function compute_Q_ij(x_i, x_j, δ_i_j)
    f_i = max_with_zero(x_i)
    f_j = max_with_zero(x_j)

    g_i = subgrad_max_with_zero(x_i)
    g_j = subgrad_max_with_zero(x_j)

    return f_i - f_j - g_j' * (x_i - x_j) - δ_i_j / (2 * β^2) * norm(g_i - g_j)^2 + δ_i_j / (2)
end

function compute_δ_set(N)
    δ_set = zeros(N)

    for i = 1:N
        if i % 2 == 1
            δ_set[i] = β * R * sqrt(2) * (N + 1)^-0.5 / (i)

        else
            #  δ_set[i] = max(0,(-1.33*10^-5 - 8.2*10^-7*N)*i + (0.00005*N-0.0002))
            δ_set[i] = 0
        end


    end

    return δ_set
end

# X = 7:3:K
# rates = zeros(K)
# for N = X
#     global β = 50
#     global R = 1
#     local δ_set = zeros(N)
#     for i = 1:N
#         if i % 2 == 1
#             δ_set[i] = β * R* sqrt(2)*(N+1)^-0.5 / (i)

#         else
#           #  δ_set[i] = max(0,(-1.33*10^-5 - 8.2*10^-7*N)*i + (0.00005*N-0.0002))
#             δ_set[i] = 0
#         end

#     end


#     rates[N]  =  get_rate_Lip(N, β, δ_set)

# end


# max_err = maximum(abs.((rates[X].-1 ./sqrt(2).*β.*R./sqrt.(X.+1))))
# scatter(X, rates[X])

# Y = @. 1/sqrt(2)*β*R/sqrt(X+1)
# display( rates[X]./Y)

# scatter!(X, Y)
# #plot(X, (rates[X].-1 ./sqrt(2).*β.*R./sqrt.(X.+1)), ylims = (-10*max_err,10*max_err), labels = "absolute error")
# plot(X, (rates[X].-1 ./sqrt(2).*β.*R./sqrt.(X.+1))./(1/sqrt(2).*β.*R./sqrt.(X.+1)), labels = "relative error")

N = 19
global β = 2
global R = 1

δ_set = compute_δ_set(N)

rate, λ_certificate, α_set = get_rate_Lip(N, β, δ_set)
display(rate)
display(sqrt(2)/2*β*R/sqrt(N+1))
# function run_method(N, δ)
#     M = Int(ceil(β * R/(2 * δ^2) - 1))
#     display(M)
#     x_0 = R/sqrt(N) * ones(N)

#     H_guess = compute_H_guess(M)
#     x_set = zeros(N,M+1)
#     x_set[:,1] = x_0
#     grads = zeros(N,M+1)
#     grads[:,1] = subgrad_max_with_zero(x_set[:,1])
#     for i = 1:M
#         x_set[:,i+1] = x_set[:,i] - sum(H_guess[i,k]*grads[:,k] for k = 1:i)
#         grads[:,i+1] = subgrad_max_with_zero(x_set[:,i+1])

#     end

#     return x_0, x_set, grads
# end

# x_0, x_set, grads = run_method(N, H_guess, 0.1)