using Optim, Plots

function L_eps(ε, p)
    return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(N, M, L, p, ε_set, type)
    if type == "same"
        ε_i_j = ε_set[1]*ones(N) # [ε_0_1, ε_1_2, ..., ε_{N-1}_N]
        ε_star_i = ε_set[1]*ones(N+1)
    elseif type == "diff"
        ε_i_j = ε_set[1:N] # [ε_0_1, ε_1_2, ..., ε_{N-1}_N]
        ε_star_i = ε_set[N+1:2*N+1]
    end
    if p == 1
        ε_i_j = 0*ε_set[1:N] # [ε_0_1, ε_1_2, ..., ε_{N-1}_N]
        ε_star_i = 0*ε_set[N+1:2*N+1]
    end

    λ_i_j = zeros(N)
    λ_star_i = zeros(N + 1)

    α_set = zeros(N + 1)
    α_set[1] = 1 / L(ε_i_j[1], p) + 1 / L(ε_star_i[1], p)
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(ε_star_i[k], p) + 1 / L(ε_i_j[k], p))
        C = -λ_i_j[k-1] * (1 / L(ε_i_j[k-1], p) + 1 / L(ε_i_j[k], p))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(ε_star_i[N+1], p)
    C = -λ_i_j[N] * 1 / L(ε_i_j[N], p)
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]



    τ = λ_star_i[N+1] + λ_i_j[N]
    ε_certificate = [ε_i_j; ε_star_i]
    λ_certificate = [λ_i_j; λ_star_i]
    σ = 1 / 2 * ε_certificate' * λ_certificate

    rate = (1 / 2 * R^2 + σ) / τ


    return rate


end

function get_H_val(N, M, L, p, ε_set)
    ε_i_j = ε_set[1:N] # [ε_0_1, ε_1_2, ..., ε_{N-1}_N]
    ε_star_i = ε_set[N+1:2*N+1]


    λ_i_j = zeros(N)
    λ_star_i = zeros(N + 1)

    α_set = zeros(N + 1)
    α_set[1] = 1 / L(ε_i_j[1], p) + 1 / L(ε_star_i[1], p)
    λ_i_j[1] = α_set[1]
    λ_star_i[1] = α_set[1]


    for k = 2:N
        B = -(1 / L(ε_star_i[k], p) + 1 / L(ε_i_j[k], p))
        C = -λ_i_j[k-1] * (1 / L(ε_i_j[k-1], p) + 1 / L(ε_i_j[k], p))
        λ_star_i[k] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
        α_set[k] = λ_star_i[k]
    end



    B = -1 / L(ε_star_i[N+1], p)
    C = -λ_i_j[N] * 1 / L(ε_i_j[N], p)
    λ_star_i[N+1] = 1 / 2 * (-B + sqrt(B^2 - 4 * C))

    α_set[N+1] = λ_star_i[N+1]


    H_certificate = zeros(N, N)

    for i in 1:N
        for j in 1:i
            if i == j
                H_certificate[i, j] = (λ_i_j[i] + L(ε_i_j[i], p) * α_set[i] * α_set[i+1]) / (L(ε_i_j[i], p) * (λ_i_j[i] + λ_star_i[i+1])) * β
            else
                H_certificate[i, j] = (α_set[i+1] * α_set[j] - 1 / β * λ_star_i[i+1] * sum([H_certificate[k, j] for k in j:i-1])) / (λ_i_j[i] + λ_star_i[i+1]) * β
            end
        end
    end




    return H_certificate
end

function run_N_opti(N, R, β, p, printing, type)
    M = 2*N + 1
    lower = 0.0000000001 * ones(M)
    upper = 3 * ones(M)
    initial_vals = 0.3 * ones(M)

    options = Optim.Options(
        iterations=2500,
        f_tol=1e-14,
        x_tol=1e-14,
        time_limit=60,
        show_trace=false,
    )

    result = Optim.optimize(ε_set -> get_rate(N, M, L_eps, p, ε_set, type), lower, upper, initial_vals, Fminbox(NelderMead()), options)
    if type == "same"
        min_ε = Optim.minimizer(result)[1] .* (ones(M))
    elseif type == "diff"
        min_ε = Optim.minimizer(result)
    end

    rate = Optim.minimum(result)
    H_val = get_H_val(N, M, L_eps, p, min_ε)

    if printing == "1"
        println("Minimizer:")
        display(min_ε)

        println()
        println("with minimum rate:")
        display(rate)

        println()
        println("and optimal step size H = ")
        display(H_val)
    end

    return min_ε, rate, H_val
end

function plot_p_rates(R, β, k, plotting_type)
    plot(title="β = $β, R = $R, plotting: $plotting_type")
    X = LinRange(0.001, 1, k)
    Y1 = zeros(k)
    Y2 = zeros(k)
    ε_sets1 = zeros(k)
    ε_sets2 = zeros(k, M)

    for (cnt, p) in enumerate(X)
        min_ε1, Y1[cnt], H_val = run_N_opti(N, R, β, p, 0, "same") # PRINTING OFF
        min_ε2, Y2[cnt], H_val = run_N_opti(N, R, β, p, 0, "diff") # PRINTING OFF

        ε_sets1[cnt] = min_ε1[1]
        ε_sets2[cnt, :] = min_ε2

        if mod(cnt, k / 5) == 0
            display(cnt)
        end
    end

    if plotting_type == "epsilons"
        plot!(X, ε_sets2, labels=["ε_0_1" "ε_1_2" "ε_star_0" "ε_star_1" "ε_star_2"])
        plot!(X, ε_sets1[:, 1], linestyle=:dash, labels="ε_same")


    elseif plotting_type == "Rates"
        plot!(X, Y1, labels="same ε")
        plot!(X, Y2, labels="different ε")

    end



end


function plot_N_rates(R, β, p, k, plotting_type)
    plot(title="β = $β, R = $R, p = $p, plotting: $plotting_type")
    X = 1:k
    Y1 = zeros(k)
    Y2 = zeros(k)


    for (cnt, N) in enumerate(X)
     
        min_ε1, Y1[cnt], H_val = run_N_opti(N, R, β, p, 0, "same") # PRINTING OFF
        min_ε2, Y2[cnt], H_val = run_N_opti(N, R, β, p, 0, "diff") # PRINTING OFF

        println("trial: $N")
        println(maximum(min_ε1))
        println(maximum(min_ε2))


    end

    if plotting_type == "epsilons"
        plot!(X, ε_sets2, labels=["ε_0_1" "ε_1_2" "ε_star_0" "ε_star_1" "ε_star_2"])
        plot!(X, ε_sets1[:, 1], linestyle=:dash, labels="ε_same")


    elseif plotting_type == "Rates"
        plot!(X, Y1, labels="same ε",  xaxis=:log, yaxis=:log)
        plot!(X, Y2, labels="different ε",  xaxis=:log, yaxis=:log)

    end



end


function OGM_rates(β, R, N)

    theta = 1
    rate = []
    for i in 0:N-1
        if i < N-1
            theta = (1 + sqrt(1+4*theta^2))/2
        else
            theta = (1 + sqrt(1+8*theta^2))/2
        end

        push!(rate, β*R^2/(2*theta^2))
    end
   
    return rate
end



R = 1
β = 1
k = 10 # number of points to test
N = 2
M = 2*N + 1
plotting_type = "Rates" # choose "epsilons" or "Rates"
# plot_p_rates(R, β, k, plotting_type)

p = 0.0000001
if p == 1
    X_range = 1:k
    
    Y_range = []
    for i = 1:k
        push!(Y_range, OGM_rates(β, R, i)[i]) # last iterate different for OGM
    end
else
    X_range = LinRange(1,k,20)

    Y_range = β*R^(1+p) ./ ((X_range .+ 1).^((1+3*p)/2)) # add +1 to match subgrad method
    # is the rate of βR/sqrt(T+1) not optimal up to coeffs?
end
plot_N_rates(R, β, p, k, plotting_type)
plot!(X_range, Y_range,  xaxis=:log, yaxis=:log)