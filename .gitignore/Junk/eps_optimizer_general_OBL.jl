using Optim, Plots, ForwardDiff, CurveFit

function L_smooth(δ, β, p)
    return ((1 - p) / (1 + p) * 1 / δ)^((1 - p) / (1 + p)) * β^(2 / (1 + p)) 
end



function get_rate(N, L, δ_set)
    T = eltype(δ_set)  # determines if Float64 or Dual
    δ_i_j = δ_set[1:N]



    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)
    α_set = zeros(T, N + 1)

    α_set[1] = 1 / L(δ_i_j[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(δ_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(δ_i_j[k-1]) + 1 / L(δ_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = 0
    C = -λ_i_j[N] * 1 / L(δ_i_j[N])
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]



    τ = λ_star_i[N+1] + λ_i_j[N]
    δ_certificate = δ_i_j
    λ_certificate = [λ_i_j; λ_star_i]
    σ = 1 / 2 * δ_certificate' * λ_certificate[1:N]

    rate = (1 / 2 * R^2 + σ) / τ


    return rate


end

function get_H_val(N, L, δ_set)
    δ_i_j = δ_set[1:N] # [δ_0_1, δ_1_2, ..., δ_{N-1}_N]


    λ_i_j = zeros(N)
    λ_star_i = zeros(N + 1)

    α_set = zeros(N + 1)
    α_set[1] = 1 / L(δ_i_j[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(δ_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(δ_i_j[k-1]) + 1 / L(δ_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = 0
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


    if p == 1
        δ_i_j = zero(T) * δ_set[1:N]
        δ_star_i = zero(T) * δ_set[N+1:2*N+1]
    end

    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)
    α_set = zeros(T, N + 1)

    α_set[1] = 1 / L(δ_i_j[1])
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(δ_i_j[k]))
        C = -λ_i_j[k-1] * (1 / L(δ_i_j[k-1]) + 1 / L(δ_i_j[k]))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = 0
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
    #             H_guess[i,j] = (λ_i[i] + L_eps(min_δ[i],p)*λ_star[i]*λ_star[i+1])/(λ_i[i+1] * L_eps(min_δ[i],p))
    #         end

    #         if j == i-1
    #             H_guess[i,j] = (λ_star[i+1] * λ_i[i-1])/(λ_star[i] * λ_i[i+1]) * (H_guess[i-1,i-1]-1/(L_eps(min_δ[i-1],p) )) 

    #         end

    #         if j < i-1
    #             H_guess[i,j] = λ_star[i+1]/λ_i[i+1] * (λ_i[i-1])/λ_star[i] * H_guess[i-1,j]
    #         end
    #     end
    # end

    return H_guess
end

function obj(x)

    return β / 4 * norm(x)^2 + β / 2 * sqrt(1 + norm(x)^2)

end

function grad(f, x)
    return ForwardDiff.gradient(f, x)
end

function FOM(N, H, x_0, f)
    d = length(x_0)
    gradients = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    x_save = OffsetArray(zeros(d, N + 1), 1:d, 0:N)

    g_0 = grad(f, x_0)
    gradients[:, 0] = g_0
    fvals = []
    x_save[:, 0] = x_0
    x_k = x_0
    push!(fvals, f(x_k))

    for k = 1:N
        x_k = x_k - sum(H[k, i] * gradients[:, i] for i = 0:k-1)
        x_save[:, k] = x_k
        gradients[:, k] = grad(f, x_k)
        push!(fvals, f(x_k))

    end

    return x_k, x_save, fvals, gradients
end

function IOGM_BL(N, M_i, x_0, f)
    d = length(x_0)
    gradients = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    x_save = OffsetArray(zeros(d, N + 1), 1:d, 0:N)

    g_0 = grad(f, x_0)
    gradients[:, 0] = g_0
    fvals = []
    x_save[:, 0] = x_0
    x_k = x_0
    push!(fvals, f(x_k))


    τ = M_i[1] 
    ψ = τ
    z = x_0
    x_k = x_0

    for k = 1:N
        z = z - ψ * gradients[:, k-1]
        φ = τ
        if k <= N - 1
            ψ = 1 / 2 * (M_i[k+1] + sqrt(( M_i[k+1])^2 + 4 * φ * (M_i[k] + M_i[k+1])))

        else
            ψ = sqrt(φ * M_i[k])

        end

        τ = ψ + φ
        x_k = φ / τ * (x_k - M_i[k] * gradients[:, k-1]) + ψ / τ * z
        x_save[:, k] = x_k
        gradients[:, k] = grad(f, x_k)
        push!(fvals, f(x_k))
    end

    return x_k, x_save, fvals, gradients

end





N = 13
β = 1
p = 0.999999
R = 1
L_eps = δ -> L_smooth(δ, β, p)

δ_set, rate, H_val = run_N_opti(N, L_eps)
H = OffsetArray(get_H(N, L_eps, δ_set),1:N, 0:N-1)
d = 8
x_0 = R*collect(1:d)

δ_i = δ_set[1:N]
M_i = [1 / L_eps(δ_i[i]) for i = 1:N]


x_k, x_save, fvals, gradients = FOM(N, H, x_0, obj)
x_kOGM, x_saveOGM, fvalsOGM, gradientsOGM = IOGM_BL(N, M_i, x_0, obj)
norm(x_save-x_saveOGM)
