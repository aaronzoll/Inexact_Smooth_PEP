using Optim, Plots

function L_eps(ε, p)
    return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(L, p, ε_set, type)
    if type == "same"
        ε_1, ε_2, ε_3, ε_4, ε_5 = ε_set[1], ε_set[1], ε_set[1], ε_set[1], ε_set[1]
    elseif type == "diff"
        ε_1, ε_2, ε_3, ε_4, ε_5 = ε_set[1], ε_set[2], ε_set[3], ε_set[4], ε_set[5]
    end

    α_0 = 1/L(ε_1,p)+ 1/L(ε_3,p)
    λ_0_1 = α_0
    λ_star_0 = α_0

    B1 = -(1/L(ε_4,p)+ 1/L(ε_2,p))
    C1 = -λ_0_1 * (1/L(ε_1,p)+ 1/L(ε_2,p))
    λ_star_1 = 1/2 * (-B1 + sqrt(B1^2-4*C1))

    λ_1_2 = λ_0_1 + λ_star_1

    B2 = -1/L(ε_5,p)
    C2 = -λ_1_2*1/L(ε_2,p)
    λ_star_2 = 1/2 * (-B2 + sqrt(B2^2-4*C2))


    τ = λ_star_2 + λ_1_2 
    σ = (ε_1*λ_0_1 + ε_2*λ_1_2 + ε_3*λ_star_0 + ε_4*λ_star_1 + ε_5*λ_star_2)/2


    rate = (1/2*R^2 + σ)/τ

    return rate 

end

function get_H_val(L, p, ε_set)
    ε_1, ε_2, ε_3, ε_4, ε_5 = ε_set[1], ε_set[2], ε_set[3], ε_set[4], ε_set[5]

    α_0 = 1/L(ε_1,p)+ 1/L(ε_3,p)
    λ_0_1 = α_0

    B1 = -(1/L(ε_4,p)+ 1/L(ε_2,p))
    C1 = -λ_0_1 * (1/L(ε_1,p)+ 1/L(ε_2,p))
    λ_star_1 = 1/2 * (-B1 + sqrt(B1^2-4*C1))

    λ_1_2 = λ_0_1 + λ_star_1

    α_1 = λ_star_1

    B2 = -1/L(ε_5,p)
    C2 = -λ_1_2*1/L(ε_2,p)
    λ_star_2 = 1/2 * (-B2 + sqrt(B2^2-4*C2))

    α_2 = λ_star_2

    H_1_1 = (λ_0_1 + L(ε_1,p)*α_0*α_1)/(L(ε_1,p) * (λ_0_1 + λ_star_1)) * β 
    H_2_1 = (α_0*α_2-1/β*λ_star_2*H_1_1)/(λ_1_2 + λ_star_2) * β
    H_2_2 = (λ_1_2 + L(ε_2,p)*α_1*α_2)/(L(ε_2,p) * (λ_1_2 + λ_star_2)) * β

    H_certificate = zeros(2,2)
    H_certificate[1,1] = H_1_1
    H_certificate[2,1] = H_2_1
    H_certificate[2,2] = H_2_2



    return H_certificate
end

function run_N_2_opti(R, β, p, printing, type)
    lower = 0.00001 * ones(5)
    upper = 8 * ones(5)
    initial_vals = 3 * ones(5)

    options = Optim.Options(
        iterations=1500,
        f_tol=1e-14,
        x_tol=1e-14,
        time_limit=30,
        show_trace=false,
    )

    result = Optim.optimize(ε_set -> get_rate(L_eps, p, ε_set, type), lower, upper, initial_vals, Fminbox(NelderMead()), options)
    if type == "same"
        min_ε = Optim.minimizer(result)[1] .* (ones(5))
    elseif type == "diff"
        min_ε = Optim.minimizer(result)
    end

    rate = Optim.minimum(result)
    H_val = get_H_val(L_eps, p, min_ε)

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
    plot(title = "β = $β, R = $R, plotting: $plotting_type")
    X = LinRange(0.001,1, k)
    Y1 = zeros(k)
    Y2 = zeros(k)
    ε_sets1 = zeros(k)
    ε_sets2 = zeros(k,5)

    for (cnt,p) in enumerate(X)
        min_ε1, Y1[cnt], H_val = run_N_2_opti(R, β, p, 0, "same") # PRINTING OFF
        min_ε2, Y2[cnt], H_val = run_N_2_opti(R, β, p, 0, "diff") # PRINTING OFF

        ε_sets1[cnt] = min_ε1[1]
        ε_sets2[cnt,:] = min_ε2

        if mod(cnt,k/5) == 0
            display(cnt)
        end
    end

    if plotting_type == "epsilons"
        plot!(X,ε_sets2, labels = ["ε_0_1" "ε_1_2" "ε_star_0" "ε_star_1" "ε_star_2"])
        plot!(X,ε_sets1[:,1],linestyle=:dash, labels = "ε_same")
        scatter!([0],[β*R/sqrt(2)], labels = "βR/sqrt(N)")

    elseif plotting_type == "Rates"
        plot!(X,Y1,labels = "same ε")
        plot!(X,Y2, labels = "different ε")
        scatter!([0], [β*R/(sqrt(2)+1)], labels = "βR/(sqrt(N)+1)")
        K = sqrt((34+8*sqrt(5)+2*sqrt(13+4*sqrt(5)))/(82+32*sqrt(5)+2*sqrt(1457+640*sqrt(5))))
        scatter!([0], [K*β * R], labels = "K*βR")
    end
    


end

R = 1
β = 1
k = 10 # number of points to test
plotting_type = "epsilons" # choose "epsilons" or "Rates"
plot_p_rates(R, β, k, plotting_type)

