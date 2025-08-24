using Optim, Plots, ForwardDiff

# Exact only for single HS. DO NOT CHANGE L_eps
function L_eps(δ, p)
    return ((1 - p) / (1 + p) * 1 / (δ))^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_sequences(N)
    a = zeros(N+1)
    b = zeros(N)

    a[1] = 2
    b[1] = 2    
    for i in 2:N 
        a[i] = 1 + sqrt(1+2*b[i-1])    
        b[i] = a[i] + b[i-1]
    end

    a[N+1] = 0.5 * (1 + sqrt(1 + 4*b[N]))
    return a, b
end

function plot_N_rates(L, R, β, p, k)
    X = 1:k
    s = zeros(k)
    δ_calc = zeros(k)
    r = zeros(k)
    
    comp1 = zeros(k)
    comp2 = zeros(k)
    for (cnt, N) in enumerate(X)
        a, b = get_sequences(N)
        s[cnt] = (b[N]+a[N+1]+sum(b))/R^2
        δ_calc[cnt] = (1-p)/(1+p)*β*s[cnt]^(-(1+p)/2)
        δ = δ_calc[cnt]
        r[cnt] = (L(δ,p)*R^2+δ*sum(b))/(2*(a[N+1]+b[N])) + δ/2

        comp1[cnt] = (L(δ,p)*R^2)/(2*(a[N+1]+b[N])) 
        comp2[cnt] = δ*sum(b)/(2*(a[N+1]+b[N])) + δ/2
    end

    return δ_calc, r, comp1, comp2, comp3
end


R = 2.2
β = 1.3
k = 1000 # number of points to test
p = 0.6

δ_calc, rates, comp1, comp2, comp3 = plot_N_rates(L_eps, R, β, p, k)
plotting_type = "loglog"

if plotting_type == "loglog"
    plot(1:k, [comp1, comp2, rates], linewidth = 2, 
        labels = ["L(δ)R^2/(2τ)" "δ-term" "total rate"], 
        xaxis = :log, yaxis = :log)

elseif plotting_type == "normal"
    plot(1:k, [comp1, comp2, rates], linewidth = 2, 
        labels = ["L(δ)R^2/(2τ)" "δ-term"  "total rate"])
end

X_range = LinRange(1, k, 2000)
Y_range = β * R^(1 + p) ./ ((X_range) .^ ((1 + 3 * p) / 2))
plot!(X_range, Y_range, linestyle = :dash, labels = "reference")