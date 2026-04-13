using Optim, Plots, ForwardDiff, CurveFit

include("../Solvers/BnB_PEP_Inexact_Smooth.jl")


function compute_theta(N)
    θ = zeros(N+1)
    θ[1] = 1.0  # θ₀
    for i in 2:N
        θ[i] = (1 + sqrt(1 + 4*θ[i-1]^2)) / 2
    end
    θ[N+1] = (1 + sqrt(1 + 8*θ[N]^2)) / 2  # θ_N
    return θ
end
function compute_H(N)
    θ = compute_theta(N)
    H = zeros(N, N)
    for i in 1:N
        for k in 1:i-1
            sum_hjk = sum(H[j, k] for j in k:i)
            H[i, k] = (1 / θ[i+1]) * (2 * θ[k] - sum_hjk)
        end
        H[i, i] = 1+ (2*θ[i]-1) / θ[i+1]
    end
    return H
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

function compute_α_from_h(h, N)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for ℓ in 1:N
        for i in 0:ℓ-1
            if i==ℓ-1
                α[ℓ,i] = h[ℓ,ℓ-1]
            elseif i <= ℓ-2
                α[ℓ,i] = α[ℓ-1,i] + h[ℓ,i] 
            end
        end
    end
    return α
end

function compute_h_from_α(α, N)
    h_new = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for l in N:-1:1
        h_new[l,l-1] = α[l,l-1]
        for i in l-2:-1:0
            h_new[l,i] = α[l,i] - α[l-1,i] 
        end
    end
    return h_new
end


# ─────────────────────────────────────────────────────────────────────────────
#  α-matrix constructors  (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────

function make_α_gradient_descent(N, L)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for i in 1:N, j in 0:i-1
        α[i, j] = 1.0/L
    end
    return α
end


function make_α_nesterov(N, L)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)

    t = ones(N + 2)
    for k in 2:N+2
        t[k] = (1 + sqrt(1 + 4 * t[k-1]^2)) / 2
    end
    β(k) = (t[k] - 1) / t[k+1]

    N >= 1 && (α[1, 0] = 1.0)

    for k in 2:N-1
        βk = β(k)
        α[k, k-1] = 1 + βk
        α[k, k-2] = (1 + βk) * α[k-1, k-2] - βk
        for j in 0:k-3
            α[k, j] = (1 + βk) * α[k-1, j] - βk * α[k-2, j]
        end
    end

    for j in 0:N-2
        α[N, j] = α[N-1, j]
    end
    α[N, N-1] = 1.0

    return α ./ L
end


function make_α_ogm(N, L)
    θ = compute_theta(N)
    H = OffsetArray(compute_H(N), 1:N, 0:N-1)
    return compute_α_from_h(H, N, 0.0, L)
end


# Optimization over δ_set and H matrix for fixed N, assuming OGM structure

function L_smooth(δ, β, p)
    return ((1 - p) / (1 + p) * 1 / δ)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(N, L, δ_set)
    T = eltype(δ_set)
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
        f_abstol=1e-8,
        x_abstol=1e-8,
        time_limit=60,
        show_trace=false,
    )
    f = δ_set -> get_rate(N, L, δ_set)
    g! = (G, x) -> (G[:] = ForwardDiff.gradient(f, x))
    result = Optim.optimize(f, g!, lower, upper, initial_vals, Fminbox(BFGS()), options)

    rate = Optim.minimum(result)
    min_δ = Optim.minimizer(result)
    return min_δ, rate
end

# Parameters
function sweep_p(N, β, R; p_values = 0.05:0.05:0.95)
    scaled_rates = Float64[]

    for p in p_values
        L_delta = δ -> L_smooth(δ, β, p)
        _, rate = run_N_opti(N, L_delta)
        scaling = (β * R^(1 + p)) / (N^((1 + 3p) / 2))
        push!(scaled_rates, rate / scaling)
        println("p = $p, rate = $rate, scaled = $(rate / scaling)")
    end
    p_vals = collect(p_values)

    p_plot = plot(p_vals, scaled_rates,
        xlabel = "p",
        ylabel = "coefficient",
        title = "Scaled rate vs p (N = $N)",
        marker = :circle,
        legend = true)


    plot!(p_vals, @. ((p_vals+1)^p_vals)/(2^((p_vals+1)/2)))
    savefig(p_plot, "scaled_rates_vs_p.png")

    return p_vals, scaled_rates, p_plot
end

N = 25
β = 1
R = 1
p = 0.99



#p_values, coeffs, p_plot = sweep_p(N, β, R)
#display(p_plot)

L_delta = δ -> L_smooth(δ, β, p)
min_δ, rate = run_N_opti(N, L_delta)
δ_i = min_δ[1:N]
δ_star = min_δ[N+1:2*N+1]

H_val = OffsetArray(get_H_val(N, L_delta, min_δ), 1:N, 0:N-1)
α = compute_α_from_h(H_val, N)
raw = fill(0.0, N+2, N+2, 1)
δ_set_opt = OffsetArray(raw, -1:N, -1:N, 1:1)
δ_set_opt[-1,0:N] = δ_star
for j = 0:N-1 
    δ_set_opt[j,j+1] = δ_i[j+1]
end

α_GD = make_α_gradient_descent(N, 1)

obj, λ_opt, Z_opt = solve_dual_PEP_with_known_stepsizes(N, β, α_GD, R, δ_set_opt, p, [];
                show_output         = :on,
                ϵ_tol_feas          = 1e-8,
                objective_type      = :default,
                obj_val_upper_bound = 1e6)

                display(obj)