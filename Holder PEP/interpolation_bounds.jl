using ForwardDiff
using Optim
using LinearAlgebra
#using GLMakie
# Example definition of L(δ); replace with your own

# θ(x) = inf_{δ≥0} (L(δ)/2 * ||x||^2 + δ)
function θ(x; δmax=100)
    norm2 = isa(x, Number) ? x^2 : dot(x, x)
    obj(δ) = L(δ) / 2 * norm2 + δ
    res = optimize(obj, 0.0, δmax)   # bounded 1D minimization
    return Optim.minimum(res)
end

function θ_star(x; δmax=100)
    norm2 = isa(x, Number) ? x^2 : dot(x, x)
    obj(δ) = -1 / (2 * L(δ)) * norm2 + δ
    res = optimize(obj, 0.0, δmax)   # bounded 1D minimization
    return -Optim.minimum(res)
end

# Gradient or derivative of θ at x
function ∇θ(x)
    if isa(x, Number)
        return ForwardDiff.derivative(θ, x)
    else
        return ForwardDiff.gradient(θ, x)
    end
end

function ∇θ_star(x)
    if isa(x, Number)
        return ForwardDiff.derivative(θ_star, x)
    else
        return ForwardDiff.gradient(θ_star, x)
    end
end



function compute_kappa()
    obj(r) = -(2 * abs(∇θ(r[1])) / abs(∇θ(2r[1])))   # negate for maximization
    res = optimize(obj, [0], [20], [0.3], Fminbox(BFGS()))            # search radius domain
    return -Optim.minimum(res)
end


function compute_cL(options)
    function obj_upper(vars)
        r, s = vars

        θr, θs = θ(r), θ(s)
        gr = ∇θ(r)
        num = θs - θr - dot(gr, (s - r))
        denom = θ(s - r)

        return num / denom
    end
    # Maximize via multivariate optimization
    obj_neg(v) = -obj_upper(v)


    lower = [-100.0, -100.0]
    upper = [100.0, 100.0]
    initial_vals = [-1.0, 1.0]


    res = optimize(obj_neg, lower, upper, initial_vals, Fminbox(BFGS()), options)
    return -Optim.minimum(res)
end






function compute_kappa_norm(d, options)
    function obj_multi(vals)
        x = vals[1:d]
        y = vals[d+1:2d]
        return -(norm(∇θ(x) - ∇θ(y)) / norm(∇θ(x - y)))
    end


    lower = -10 * ones(2 * d)
    upper = 10 * ones(2 * d)
    initial_vals = collect(rand(2d))




    res = optimize(obj_multi, lower, upper, initial_vals, Fminbox(BFGS()), options)         # search radius domain
    return -Optim.minimum(res)
end


function compute_cL_norm(d, options)
    function obj_upper(vars)
        x = vars[1:d]
        y = vars[d+1 : 2*d] 
        
        θx, θy = θ(x), θ(y)
        gx = ∇θ(x)
        num = θy - θx - dot(gx, (y - x))
        denom = θ(y - x)

        return num / denom
    end
    # Maximize via multivariate optimization
    obj_neg(v) = -obj_upper(v)

    lower = -10 * ones(2 * d)
    upper = 10 * ones(2 * d)
    initial_vals = collect(rand(2d))


    res = optimize(obj_neg, lower, upper, initial_vals, Fminbox(BFGS()), options)
    return -Optim.minimum(res)
end




iter_print_freq = 3

function my_callback(state)

    if state.iteration % iter_print_freq == 0 && state.iteration > 1
        println("Iter $(state.iteration): f = $(state.value)")
    end

    return false
end
options = Optim.Options(
    iterations=500,
    f_abstol=1e-6,
    x_reltol=1e-6,
    time_limit=30,
    show_trace=false,
    callback=my_callback
)




p = [0.5]
q = @. (1 - p) / (1 + p)
β = [2,]

k = length(p)
L(δ) = sum([((1 - p[i]) / (1 + p[i]) * 1 / (2 * δ))^((1 - p[i]) / (1 + p[i])) * β[i]^(2 / (1 + p[i])) for i = 1:k]) # must be ≥ 0 for δ ≥ 0

q = 1.05
L(δ) = 1/δ^q

κ = compute_kappa_norm(1, options)
c_L = compute_cL(options)
d = 5
c_L_5 = compute_cL_norm(d, options)

display([κ, c_L, c_L_5])

# d = 1
# κ_1 = compute_kappa_norm(d, options)

# d = 7
# κ_7 = compute_kappa_norm(d, options)
# display([c_L, C_L_exact, C_L_bound])
# display([κ, κ_1, κ_7, c_L])

# N = 100

# X = LinRange(-2, 2, N)
# Y = LinRange(-2, 2, N)

# #Z = [abs(∇θ(x) - ∇θ(y)) / abs(∇θ(x - y)) for x in X, y in Y]
# #Z = [(θ(s)-θ(t)-∇θ(t)*(s-t))/θ(abs(s-t)) for s in X, t in Y]
# Z = @. [min(5.0, (θ_star(u)-θ_star(v)-∇θ_star(v)*(u-v))/θ_star(abs(u-v))) for u in X, v in Y]
# # q_min = @. 4/((1-q)*2^(2/(1-q)))

# fig = Figure()
# ax = Axis3(fig[1,1], xlabel="x", ylabel="y", zlabel="value")
# surface!(ax, X, Y, Z)
# # surface!(ax, X, Y, q_min.*ones(N, N))
# fig