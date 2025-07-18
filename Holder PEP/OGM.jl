using LinearAlgebra, ForwardDiff, Plots

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
        if i < N-1
            theta = (1 + sqrt(1+4*theta^2))/2
        else
            theta = (1 + sqrt(1+8*theta^2))/2
        end

        push!(rate, β*R^2/(2*theta^2))
    end
   
    return rate
end

function obj(x)
    return β * sqrt(1 + norm(x)^2)

end

function grad(x)
    return ForwardDiff.gradient(x -> obj(x), x)

end

function FO(N, H, x_0)
    x_set = zeros(d, N+1)
    grad_set = zeros(d, N+1)
    x_set[:, 1] = x_0
    grad_set[:, 1] = grad(x_0)
    f_set = []
    push!(f_set, obj(x_0))
    for i = 1:N
        x_set[:, i+1] = x_set[:, i] - 1/β * sum(H[i,k]*grad_set[:, k] for k = 1:i)
        grad_set[:,i+1] = grad(x_set[:, i+1])
        push!(f_set, obj(x_set[:, i+1]))
    end

    return x_set, f_set, grad_set
end

function OGM(N, x_0)
    x_set = zeros(d, N+1)
    y_set = zeros(d, N+1)
    grad_set = zeros(d, N+1)   

    y_0 = x_0
    x_set[:, 1] = x_0
    y_set[:, 1] = y_0
    grad_set[:, 1] = grad(x_0)
    theta = 1

    f_set = []
    for i = 1:N
        y_set[:, i+1] = x_set[:, i] - 1/β * grad(x_set[:, i])

        theta_past = theta
        if i < N
            theta = (1+sqrt(1+4*theta^2))/2
        else
            theta = (1+sqrt(1+8*theta^2))/2
        end

        x_set[:, i+1] = y_set[:, i+1] + (theta_past - 1)/theta * (y_set[:, i+1]-y_set[:, i]) + theta_past/theta * (y_set[:, i+1]- x_set[:, i])

        push!(f_set, obj(x_set[:, i+1]))
    end

    return x_set, y_set, f_set, grad_set
end

global β = 1
global d = 1
N = 5
H = compute_H(N)
x_0 =  ones(d)
R = norm(x_0)

x_set1, f_set1, grad_set1 = FO(N, H, x_0)
x_set, y_set, f_set, grad_set = OGM(N, x_0)

ogm_rate = []
for i = 1:N
    push!(ogm_rate, OGM_rates(β, R, i)[i])
end


