using Plots, ForwardDiff

function f(x, β)
    # β smooth function
    return  β * x^2
end

function g(x, M)
    # M Lipschitz function
    return M * abs(x - 3) + M * abs(x+1)
end

function F(x, β, M)
    # composite functions
    return f(x, β) + g(x, M)
end

function F_smooth(x, β, M)
    k = 20000
    return 1 / k * sum([F(x + β * (rand() - 0.5), β, M) for _ in 1:k])
end

function F_lip(x, β, M)
    return F(x, β, M) - F_smooth(x, β, M)
end

function Lip_cont(func, β, M)
    g = x -> ForwardDiff.derivative(z -> F_lip(z, β, M), x)
    println(maximum([g(x) for x in LinRange(-10,10,1000)]))
end

β = 15
M = 0
X = LinRange(-10, 10, 3000)
F_data = F.(X, β, M)
F_smooth_data = F_smooth.(X, β, M)
F_lip_data = F_lip.(X, β, M)

Lip_cont(F_lip, β, M)
plot()
plot!(X, F_data, linewidth =  5, labels = "F(x)")
plot!(X, F_smooth_data, linewidth =  2, linestyle = :dash, labels = "Smoothed")
plot!(X, F_lip_data, linestyle = :dash, linewidth =  2, labels = "Lipschitz Component")
plot!(X, F_smooth_data + F_lip_data, color = :black, linealpha = 0.5,
    linewidth = 3, linestyle = :dot, labels = "Sum Recovery")

