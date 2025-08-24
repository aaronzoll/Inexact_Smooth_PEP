using LinearAlgebra, ForwardDiff, GeometryBasics, QHull, Plots

function L(δ)
    return ((1-p)/(1+p)*1/(2*δ))^((1-p)/(1+p)) * β^(2/(1+p))
end

function f(x)


    x = β_1 * norm(x)^(p_1+1)

end

function g(x)
    if length(x) == 1
        return ForwardDiff.derivative(f, x)
    else
        return ForwardDiff.gradient(f, x)
    end

end

function get_obs()
    x_data = [-1, 0, 1]
    f_data = [f(x) for x in x_data]
    g_data = [g(x) for x in x_data]

    return x_data, f_data, g_data

end

function Q_ijδ(x_data, f_data, g_data, δ)
    N = length(x_data)
    Q = zeros(N, N)
    for i = 1:N, j = 1:N 
        Q[i, j] = f_data[i] - f_data[j] - g_data[j]' * (x_data[i]-x_data[j]) - 1/(2*L(δ))*norm(g_data[i]-g_data[j])^2 + δ
    end

    return Q
end

function check_interp(x_data, f_data, g_data, min_δ, max_δ, T)
    mach_tol = 1e-10
    Δ = LinRange(min_δ, max_δ, T)
    for δ = Δ
        if minimum(Q_ijδ(x_data, f_data, g_data, δ)) < -mach_tol
            return false
        end

    end
    return true
end

function moreau_envelope(f, λ; grid=range(-3.0, 3.0, length=100))
    quad = y -> (x -> f(y) + λ/2 * norm(x - y)^2 )
    return x -> minimum(quad(y).(Ref(x)) for y in grid)
end


function fenchel_conj(f, x_grid=range(-5, 5, length=20))
    f_conj = s -> maximum([s' * x - f(x) for x in x_grid])
    return f_conj

end

function construct_interp(x_data, f_data, g_data, min_δ = 1e-5, max_δ = 0.3, T = 20)
    N = length(x_data)

    Δ = LinRange(min_δ, max_δ, T)
    f_deltas = []

    for δ = Δ
        f_plus = [f_data[i] - 1/(2*L(δ)) * norm(g_data[i])^2 + δ for i in 1:N]
        x_plus = [x_data[i] - 1/L(δ) * g_data[i] for i in 1:N]

        f_max_delta = x -> maximum([f_plus[i] + g_data[i]' * (x - x_plus[i]) for i in 1:N])
        f_delta = moreau_envelope(f_max_delta, L(δ))
        push!(f_deltas, f_delta)

    end

   fstars = [fenchel_conj(f) for f in f_deltas]

    return fstars
end

global β_1 = 2
global p_1 = 0.999
global p = p_1
global β = β_1*(p+1) * 2^(1-p) 

x_data, f_data, g_data = get_obs()

min_δ = 1e-8
max_δ = 10
T = 1000
check_interp(x_data, f_data, g_data, min_δ, max_δ, T)

f_conv_conj = construct_interp(x_data, f_data, g_data)
X = LinRange(-3,3,100)

f_conj = g -> maximum([fstar(g) for fstar in f_conv_conj])
f_interp = fenchel_conj(f_conj)

