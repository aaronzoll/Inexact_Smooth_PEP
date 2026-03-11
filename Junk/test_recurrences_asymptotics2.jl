using Optim, ForwardDiff, CurveFit, LinearAlgebra, OffsetArrays, Plots
using CSV, DataFrames

function L_smooth(δ, β, p)
    return ((1 - p) / (1 + p) * 1 / (2 * δ))^((1 - p) / (1 + p)) * β^(2 / (1 + p))
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



    τ_N = λ_star_i[N+1] + λ_i_j[N]
    δ_certificate = [δ_i_j; δ_star_i]
    λ_certificate = [λ_i_j; λ_star_i]
    σ =  δ_certificate' * λ_certificate

    rate = (1 / 2 * R^2 + σ) / τ_N
    φ = λ_i_j
    ψ = λ_star_i

    return rate, φ, ψ, τ_N


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

    for i = 0:N
        δ_set[N+i+1] = k3 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i + 1)^(-(1 + p))
    end

    return δ_set
end



N = 23
β = 1
global p = 0.4
R = 1

L_eps = δ -> L_smooth(δ, β, p)
k = [exp(p) * (-sqrt(2) * p^0.25 + sqrt(2)), max(0, (7 - p^(-0.25)) * (p^(1 / 3) - p^(1 / 2)) * p^p), max(0, 1 / 7 * (p^(0.04 * p) - p^2))]
δ_set = compute_δ_set(N, k, p)
rate, φ, ψ, τ_N = get_rate(N, L_eps, δ_set)



γ = β * R^(1 + p) / ((N + 1)^((1 + p) / 2))
ϕ = 1 / L_eps(γ)


function simplified_recurrance(N, k1, k2, k3, p, r)

    # Storage
    tau = OffsetArray(zeros(N + 1), 0:N)
    psi = OffsetArray(zeros(N + 1), 0:N)
    phi = zeros(N)

    # τ₀ = ψ₀ = k1^r + k2^r
    tau[0] = (k1^r + k3^r)
    psi[0] = (k1^r + k3^r)



    # Main loop: i = 1..N-1
    for i in 1:N-1
        # (a) φ_i = τ_{i-1}
        phi[i] = tau[i-1]

        # (b) ψ_i = ...
        if isodd(i)
            psi[i] = 0.5 * ((i + 1)^(p - 1) * (k2^r + k3^r) + sqrt(((i + 1)^(p - 1) * (k2^r + k3^r))^2 + 4.0 * phi[i] * (k1^r * i^(p - 1) + k2^r * (i + 1)^(p - 1))))
        else
            psi[i] = 0.5 * ((i + 1)^(p - 1) * (k1^r + k3^r) + sqrt(((i + 1)^(p - 1) * (k1^r + k3^r))^2 + 4.0 * phi[i] * (k2^r * i^(p - 1) + k1^r * (i + 1)^(p - 1))))
        end


        # (c) τ_i = ψ_i + φ_i
        tau[i] = psi[i] + phi[i]
    end

    # Final step: i = N

    # φ_N = τ_{N-1}
    phi[N] = tau[N-1]

    # ψ_N with the special (k3^r-only) A term and B = k1 * N^(p-1)
    A = (N + 1)^(p - 1) * (k3^r)
    B = k1^r * N^(p - 1)
    psi[N] = 0.5 * (A + sqrt(A * A + 4.0 * phi[N] * B))

    # τ_N = ψ_N + φ_N
    tau[N] = psi[N] + phi[N]

    φ = [phi[i] for i = 1:N]
    ψ = [psi[i] for i = 0:N]
    τ = [tau[i] for i = 0:N]
    return φ, ψ, τ
end

k1 = k[1]
k2 = k[2]
k3 = k[3]



phi, psi, tau = simplified_recurrance(N, k1, k2, k3, p, (1 - p) / (1 + p))
φ_2 = ϕ * phi
ψ_2 = OffsetArray(ϕ * psi, 0:N)
τ_2 = OffsetArray(ϕ * tau, 0:N)





function gen_asymptotic_const(N, k1, k2, k3, p; constant=:tau)

    L_eps = δ -> L_smooth(δ, β, p)
    k = [k1, k2, k3]
    δ_set = compute_δ_set(N, k, p)
    r = (1-p) / (1+p)

    if constant == :tau
        _, _, _, τ1 = get_rate(N, L_eps, δ_set)
        τ_scaled_1 = τ1 * (β * R^(p - 1) / (N^((3 * p + 1) / 2)))

        δ_set = compute_δ_set(2 * N, k, p)

      _, _, _, τ2 = get_rate(2*N, L_eps, δ_set)
        τ_scaled_2 = τ2 * (β * R^(p - 1) / ((2*N)^((3 * p + 1) / 2)))

        δ_set = compute_δ_set(3 * N, k, p)
      _, _, _, τ3 = get_rate(3*N, L_eps, δ_set)
        τ_scaled_3 = τ3 * (β * R^(p - 1) / ((3*N)^((3 * p + 1) / 2)))

        return 0.5 * τ_scaled_1 - 4 * τ_scaled_2 + 9 / 2 * τ_scaled_3

    elseif constant == :rate
        rate1, _, _, _ = get_rate(N, L_eps, δ_set)
        rate_scaled_1 = rate1 / (β * R^(p + 1) / (N^((3 * p + 1) / 2)))

        δ_set = compute_δ_set(2 * N, k, p)

        rate2, _, _, _ = get_rate(2 * N, L_eps, δ_set)
        rate_scaled_2 = rate2 / (β * R^(p + 1) / ((2 * N)^((3 * p + 1) / 2)))

        δ_set = compute_δ_set(3 * N, k, p)

        rate3, _, _, _ = get_rate(3 * N, L_eps, δ_set)
        rate_scaled_3 = rate3 / (β * R^(p + 1) / ((3 * N)^((3 * p + 1) / 2)))

        return 0.5 * rate_scaled_1 - 4 * rate_scaled_2 + 9 / 2 * rate_scaled_3

    elseif constant == :sigma


        rate1, _, _, τ1 = get_rate(N, L_eps, δ_set)
        rate_scaled_1 = rate1 * τ1 - 1/2*R^2

        δ_set = compute_δ_set(2 * N, k, p)

        rate2, _, _, τ2 = get_rate(2 * N, L_eps, δ_set)
        rate_scaled_2 = rate2 * τ2 - 1/2*R^2

        δ_set = compute_δ_set(3 * N, k, p)

        rate3, _, _, τ3 = get_rate(3 * N, L_eps, δ_set)
        rate_scaled_3 = rate3 * τ3 - 1/2*R^2

        return 0.5 * rate_scaled_1 - 4 * rate_scaled_2 + 9 / 2 * rate_scaled_3
    else 

        return nothing
    end
    

end





Big_N = 2000

gen_asymptotic_const(Big_N, k1, k2, k3, p; constant = :tau)



# Assume your gen_asymptotic_const(N,k1,k2,k3,p) is already defined.

function generate_dataset(N;
    k_range=0.0:0.5:5.0,
    p_range=range(0.01, 0.99; length=20),
    outfile="asymptotic_data_sigma.csv")

    results = DataFrame(k1=Float64[], k2=Float64[], k3=Float64[], p=Float64[], r=Float64[], value=Float64[])
    cnt = 0
    for k1 in k_range, k2 in k_range, k3 in k_range, p in p_range
        try
            val = gen_asymptotic_const(N, k1, k2, k3, p; constant = :sigma)
            push!(results, (k1, k2, k3, p, (1-p)/(1+p), val))
        catch e
            @warn "Failed at (k1=$k1, k2=$k2, k3=$k3, p=$p): $e"
        end

        cnt = cnt + 1

        if cnt % 100 == 0
            display(cnt)
        end
    end

    CSV.write(outfile, results)
    return results
end

# Example usage:
N = 2001   # pick large N
#df = generate_dataset(N; k_range=range(0.01,0.99; length=15), p_range=range(0.01,0.99; length=30))


gen_asymptotic_const(Big_N, k1, k2, k3, p; constant = :tau)




function gen_tau_N(m, M, k1, k2, k3, p)
    X = m:10:M
    Y = zeros(length(X))
    r = (1-p)/(1+p)
    k = [k1, k2, k3]
    for (cnt, n) = enumerate(X)
        L_eps = δ -> L_smooth(δ, β, p)
        δ_set = compute_δ_set(n, k, p)


        rate, _, _, τ = get_rate(n, L_eps, δ_set)
        Y[cnt] = τ * (β * R^(p-1)/((n+1)^((3*p+1)/2)))* (r/2)^r *(p+1)^2
    end

    return X, Y 

end

function gen_rate_N(m, M, k1, k2, k3, p)
    X = m:10:M
    Y = zeros(length(X))
    r = (1-p)/(1+p)
    k = [k1, k2, k3]
    for (cnt, n) = enumerate(X)
        L_eps = δ -> L_smooth(δ, β, p)
        δ_set = compute_δ_set(n, k, p)


        rate, _, _, τ = get_rate(n, L_eps, δ_set)
        Y[cnt] = rate * τ - 1/2*R^2
    end

    return X, Y 

end

m = 50
M = 9000


p = rand()
r = (1-p)/(1+p)

k1 = 10*rand()
k2 = 10*rand()
k3 = 10*rand()
X, Y =  gen_rate_N(m, M, k1, k2, k3, p)
min_Y = minimum(Y)
max_Y = maximum(Y)
display([k1,k2,k3,p])

Big_N = 100001
display(gen_asymptotic_const(Big_N, k1, k2, k3, p; constant = :tau) )

s1 = (k1^r + k2^r)/( (r/2)^r *(p+1)^2) * R^2 *k1/2
s2 = (k1^r + k2^r)/( (r/2)^r *(p+1)^2) * R^2 *k2/2

plot(X,Y)
plot!(X,(s1+s2)*ones(length(X)))



