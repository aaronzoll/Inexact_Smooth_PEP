using Optim, Plots, ForwardDiff, CurveFit

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
    H_val = get_H_val(N, L_eps, min_δ)


    return min_δ, rate, H_val
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


function run_δ_opti(N, L, p)

    lower = 0 * ones(3)
    upper = 10 * ones(3)
    initial_vals = 0.1 * ones(3)

    options = Optim.Options(
        iterations=1000,
        f_tol=1e-8,
        x_tol=1e-8,
        time_limit=45,
        show_trace=false,
    )
    f = k -> get_rate(N, L, compute_δ_set(N, k, p))

    result = Optim.optimize(f, lower, upper, initial_vals, Fminbox(NelderMead()), options)
    k_star = Optim.minimizer(result)
    δ_set_optimal = compute_δ_set(N, k_star, p)

    rate = Optim.minimum(result)

    return δ_set_optimal, k_star, rate


end


function get_coeffs(N)
    δ = 1e-8
    k = 20
    X = 0.5

    coeffs = []
    k1 = []
    k2 = []
    k3 = []
    for p in X
        L_eps = δ -> L_smooth(δ, β, p)
        _, k_star, rate = run_δ_opti(N, L_eps, p) 
        coeff = rate/(β*R^(1+p) * N^(-(1+3*p)/2))     
        
        push!(coeffs, coeff)
        push!(k1, k_star[1])
        push!(k2, k_star[2])
        push!(k3, k_star[3])
    end

    return X, coeffs, k1, k2, k3
end


global β = 1
global R = 1

k1_coeff = []
k2_coeff = []
k3_coeff = []
X = 3:2:137
for N = X
    X, coeffs, k1, k2, k3 = get_coeffs(N)
    push!(k1_coeff, k1)
    push!(k2_coeff, k2)
    push!(k3_coeff, k3)

end

plot()
scatter!(X, k1_coeff)
scatter!(X, k2_coeff)
scatter!(X, k3_coeff)
