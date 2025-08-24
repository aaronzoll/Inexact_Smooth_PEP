using Optim, ForwardDiff, CurveFit, LinearAlgebra, OffsetArrays

function L_smooth(δ, β, p)
    return ((1 - p) / (1 + p) * 1 / (2*δ))^((1 - p) / (1 + p)) * β^(2 / (1 + p))
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

function get_H(N, L, δ_set)
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


    return H_guess
end

function compute_δ_set(N, k, p)
    δ_set = zeros(2 * N + 1)
    k1 = k[1] # should only be function of p?
    k2 = k[2]
    k3 = k[3]
    for i = 1:N
        if i % 2 == 1
            δ_set[i] = k1 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))

        else
            #  δ_set[i] = max(0,(-1.33*10^-5 - 8.2*10^-7*N)*i + (0.00005*N-0.0002))
            δ_set[i] = k2 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))
        end


    end

    for i = 1:N+1
        δ_set[N+i] = k3 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))
    end

    return δ_set
end


function obj(x)

    return β / (p + 1) * 2^(p-1) * norm(x)^(p + 1)

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

function IOGM(N, M_i, M_star, x_0, f)
    d = length(x_0)
    gradients = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    x_save = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    z_save = OffsetArray(zeros(d, N + 2), 1:d, 0:N+1)

    g_0 = grad(f, x_0)
    gradients[:, 0] = g_0
    fvals = []
    x_save[:, 0] = x_0
    x_k = x_0
    push!(fvals, f(x_k))


    τ = M_i[1] + M_star[0]
    ψ = τ
    z = x_0
    z_save[:, 0] = z

    x_k = x_0

    ψ_save = []
    φ_save = []
    τ_save = []

    push!(ψ_save, ψ)
    push!(τ_save, τ)
    for k = 1:N
        z = z - ψ * gradients[:, k-1]
        φ = τ
        if k <= N - 1
            ψ = 1 / 2 * (M_star[k] + M_i[k+1] + sqrt((M_star[k] + M_i[k+1])^2 + 4 * φ * (M_i[k] + M_i[k+1])))

        else
            ψ = 1 / 2 * (M_star[k] + sqrt(M_star[k]^2 + 4 * φ * M_i[k]))

        end

        τ = ψ + φ
        x_k = φ / τ * (x_k - M_i[k] * gradients[:, k-1]) + ψ / τ * z
        x_save[:, k] = x_k
        z_save[:, k] = z
        gradients[:, k] = grad(f, x_k)
        push!(fvals, f(x_k))

        push!(φ_save, φ)
        push!(ψ_save, ψ)
        push!(τ_save, τ)

     
    end
    z_save[:, N+1] = z - ψ * gradients[:, N]
    return x_k, x_save, z_save, fvals, gradients, φ_save, ψ_save, τ_save

end



N = 7
β = 2
global p = 0.004999999
R = 3

L_eps = δ -> L_smooth(δ, β, p)
k = [exp(p) * (-sqrt(2) * p^0.25 + sqrt(2)), max(0, (7 - p^(-0.25)) * (p^(1 / 3) - p^(1 / 2)) * p^p), max(0, 1 / 7 * (p^(0.04 * p) - p^2))]
δ_set = compute_δ_set(N, k, p)

# δ_set = 0.8*ones(2*N + 1)

H = OffsetArray(get_H(N, L_eps, δ_set), 1:N, 0:N-1)
d = 3
x_0 = collect(1:d)
x_0 = R * x_0 / norm(x_0)

x_k, x_save, fvals, gradients, = FOM(N, H, x_0, obj)


δ_i = δ_set[1:N]
δ_star_i = δ_set[N+1:2*N+1]
M_i = [1 / L_eps(δ_i[i]) for i = 1:N]
M_star = OffsetArray([1 / L_eps(δ_star_i[i]) for i = 1:N+1], 0:N)

x_kOGM, x_saveOGM, z_set, fvalsOGM, gradientsOGM, φ, ψ, τ = IOGM(N, M_i, M_star, x_0, obj)


φ = OffsetArray(φ, 1:N)
ψ = OffsetArray(ψ, 0:N)
τ = OffsetArray(τ, 0:N)
x_set = OffsetArray(x_saveOGM, 1:d, 0:N)
f_set = OffsetArray(fvalsOGM, 0:N)
g_set = OffsetArray(gradientsOGM, 1:d, 0:N)

f_star = 0
x_star = zeros(d)
H = OffsetArray(zeros(N+1), 0:N)
Q_i = zeros(N)
Q_star = OffsetArray(zeros(N+1), 0:N)
δ_star_i = OffsetArray(δ_star_i, 0:N)

for k = 1:N
    H[k-1] = τ[k-1] * (f_star - f_set[k-1] + 1/(2*L_eps(δ_i[k]))*norm(g_set[1:d, k-1])^2 + δ_star_i[k-1]) + 1/2*norm(x_set[:, 0]-x_star)^2 - 1/2*norm(z_set[:, k]-x_star)^2 

    Q_star[k-1] = f_star - f_set[k-1] - g_set[1:d, k-1]' * (x_star - x_set[1:d, k-1]) - 1/(2*L_eps(δ_star_i[k-1])) * norm(g_set[1:d, k-1])^2 + δ_star_i[k-1]
    Q_i[k] = f_set[k-1] - f_set[k] - g_set[1:d, k]' * (x_set[1:d, k-1] - x_set[1:d, k]) - 1/(2*L_eps(δ_i[k])) * norm(g_set[1:d, k]- g_set[1:d, k-1])^2 + δ_i[k]

end
Q_star[N] = f_star - f_set[N] - g_set[1:d, N]' * (x_star - x_set[1:d, N]) - 1/(2*L_eps(δ_star_i[N])) * norm(g_set[1:d, N])^2 + δ_star_i[N]

H[N] = τ[N] * (f_star - f_set[N] + δ_star_i[N]) + 1/2*norm(x_set[:, 0]-x_star)^2 - 1/2*norm(z_set[:, N+1]-x_star)^2


for k = 1:N
 local   a = H[k] + τ[k-1]*δ_star_i[k-1] + φ[k]*(δ_i[k]-δ_star_i[k])
  local  b = H[k-1] + φ[k]*Q_i[k] + ψ[k]*Q_star[k]
    #display(H[k] + τ[k-1]*δ_star_i[k-1] + φ[k]*(δ_i[k]-δ_star_i[k]))
    #display(H[k-1] + φ[k]*Q_i[k] + ψ[k]*Q_star[k])
    display(a-b)
end

