using Plots

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

function L_eps(ε, p)
    return ((1 - p) / (1 + p) * 1 / ε)^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end

function get_rate(N, L, p, ε)
    T = typeof(ε)
    λ_i_j = zeros(T, N)
    λ_star_i = zeros(T, N + 1)

    M_ε = 1 ./ L.(ε, p)

    λ_i_j[1] = 2 * M_ε
    λ_star_i[1] = 2 * M_ε

    for k = 2:N
        B = -2 * M_ε
        C = -λ_i_j[k-1] * 2 * M_ε
        λ_star_i[k] = 0.5 * (-B + sqrt(B^2 - 4 * C))

        λ_i_j[k] = λ_i_j[k-1] + λ_star_i[k]
    end

    B = -M_ε
    C = -λ_i_j[N] * M_ε
    λ_star_i[N+1] = 0.5 * (-B + sqrt(B^2 - 4 * C))

    τ = λ_star_i[N+1] + λ_i_j[N]
    σ = 0.5 * (sum(λ_i_j) + sum(λ_star_i)) * ε 
    # note, σ = 0.5 * (sum(λ_i_j) + τ) * ε
    rate = (0.5 * R^2 + σ) / τ

    # here λ_i_j = λ_i_{i+1} = a[i] above, with τ = a[N] + b[N+1]


    return λ_i_j, λ_star_i, rate, τ
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

    return rate, theta
end


R = 0.5
β = 1.3
N = 7
a, b = get_sequences(N)
σ = sum(a) + sum(b)
σ1 = sum(b) + b[N] + a[N+1]

p = 1
ε = 1e-10
M = 1/L_eps(ε,p)
a_seq = M*a
b_seq = M*b
r1 = (R^2 + M*ε*sum(b)) / (2*M*(b[N] + a[N+1])) + ε/2


λ_i_j, λ_star_i, rate1, τ = get_rate(N, L_eps, p, ε)
rate2, theta =  OGM_rates(β, R, N)
display(r1)
display(rate1)
display((L_eps(ε,p)*R^2+ε*sum(b))/(2*(b[N]+a[N+1]))+ε/2)
# display(λ_i_j)

# display(b_seq)
# display(λ_star_i)
# display(a_seq)

# k = 10
# p = 1
# ε = 1e-10
# OGM_X_range = 1:k
# OGM_Y_range = []
# test_Y_range = []
# for i = 1:k
#     push!(OGM_Y_range, OGM_rates(β, R, i)[i]) # last iterate different for OGM
#     _, _, rate1 = get_rate(i, L_eps, p, ε)
#     push!(test_Y_range, rate1)
# end

