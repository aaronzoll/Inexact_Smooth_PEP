using Plots, JLD2, Optim, ForwardDiff, CurveFit


 

function L_smooth(ε, β, p)
    return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(N, L, ε_set)
    T = eltype(ε_set)  # determines if Float64 or Dual
    ε_i_j = ε_set[1:N]
    ε_star_i = ε_set[N+1:2*N+1]



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

function compute_ε_set(N, k, p)
    ε_set = zeros(2 * N + 1)
    k1 = k[1] # should only be function of p?
    k2 = k[2]
    k3 = k[3]
    for i = 1:N
        if i % 2 == 1
            ε_set[i] = k1 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))

        else
            #  ε_set[i] = max(0,(-1.33*10^-5 - 8.2*10^-7*N)*i + (0.00005*N-0.0002))
            ε_set[i] = k2 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))
        end


    end

    for i = 1:N+1
        ε_set[N+i] = k3 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))
    end

    return ε_set
end


@load "coeffs_epsilons_guess.jld2"
global β = 1
global R = 1
N = 10001

coeffs = zeros(length(X))
for (cnt, p) in enumerate(X)
    L_eps = ε -> L_smooth(ε, β, p)
    k = [k1[cnt], k2[cnt], k3[cnt]]
    ε_set = compute_ε_set(N, k, p)
    rate = get_rate(N, L_eps, ε_set)
    coeffs[cnt] = rate/(β * R^(1+p) * N^(-(1+3*p)/2))
end
X1 = X

X2 = LinRange(0.0000001, 1, 200)
coeffs2 = zeros(length(X2))
for (cnt, p) in enumerate(X2)
    L_eps = ε -> L_smooth(ε, β, p)
    k = [exp(p)*(-sqrt(2)*p^0.25+sqrt(2)), max(0,(5-p^(-1/4))*(p^(1/4)-p^(1/2))*p^p), max(0, 1/7*(p^(1/(25*p))-p^2))]
    ε_set = compute_ε_set(N, k, p)
    rate = get_rate(N, L_eps, ε_set)
    coeffs2[cnt] = rate/(β * R^(1+p) * N^(-(1+3*p)/2))
end


plot(ylims = ())
scatter!(X,coeffs)
scatter!(X2,coeffs2)
X = collect(X2)
scatter!(X, @. 2^((X+1)/2)*3^((X-1)/(2))/(1+X))
#scatter!(X3, coeffs3)
# @load "coeffs_epsilons_guess_zoom.jld2"

# coeffs2 = zeros(length(X))

# for (cnt, p) in enumerate(X)
#     L_eps = ε -> L_smooth(ε, β, p)
#     k = [k1[cnt], k2[cnt], k3[cnt]]
#     ε_set = compute_ε_set(N, k, p)
#     rate = get_rate(N, L_eps, ε_set)
#     coeffs2[cnt] = rate/(β * R^(1+p) * N^(-(1+3*p)/2))
# end

# X2 = X

# @load "coeffs_epsilons_guess_zoom_2.jld2"

# coeffs3 = zeros(length(X))

# for (cnt, p) in enumerate(X)
#     L_eps = ε -> L_smooth(ε, β, p)
#     k = [k1[cnt], k2[cnt], k3[cnt]]
#     ε_set = compute_ε_set(N, k, p)
#     rate = get_rate(N, L_eps, ε_set)
#     coeffs3[cnt] = rate/(β * R^(1+p) * N^(-(1+3*p)/2))
# end

# X3 = X

# @load "coeffs_epsilons_guess_zoom_3.jld2"

# coeffs4 = zeros(length(X))

# for (cnt, p) in enumerate(X)
#     L_eps = ε -> L_smooth(ε, β, p)
#     k = [k1[cnt], k2[cnt], k3[cnt]]
#     ε_set = compute_ε_set(N, k, p)
#     rate = get_rate(N, L_eps, ε_set)
#     coeffs4[cnt] = rate/(β * R^(1+p) * N^(-(1+3*p)/2))
# end

# X4 = X
# plot()
# scatter!(X1, coeffs)
# scatter!(X2, coeffs2)
# scatter!(X3, coeffs3)
# scatter!(X4[1:100], coeffs4[1:100])