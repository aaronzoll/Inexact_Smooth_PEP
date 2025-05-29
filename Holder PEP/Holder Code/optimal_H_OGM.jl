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

function get_vector(N)
    vec_lower = zeros(div(N*(N+1),2))
    cnt = 0
    for i in 1:N
        for j in 1:i
            cnt = cnt + 1
            vec_lower[cnt] = OGM[i,j]
        end
    end
    return vec_lower
end

N = 2
OGM = compute_H(N)
vec_lower = get_vector(N)
