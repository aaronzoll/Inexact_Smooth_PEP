using Revise

include("HolderPEPOptimization.jl")
using Optim
using .HolderPEPOptimization
import .HolderPEPOptimization

function Optimize(N, L, R, p, M, initial_vals)
    T = 0
    max_iter = 1500
    max_time = 40
    H_size = div(N * (N + 1), 2)
    lower = 0.00000001 * [ones(M); ones(H_size)]
    upper = 2 * [ones(M); ones(H_size)]
   # prob = OptimizationProblem(N, L, R, p, M)
    result, terminated_early = optimize_ε_h(N, L, R, p, M, max_iter, max_time, lower, upper, initial_vals)
    minimizer = Optim.minimizer(result)
    H = get_H(N, M, minimizer)

    ε_set = minimizer[1:M]
    return ε_set, H, result, terminated_early

end

function run_batch_trials(N, L, R, p, M, trials)

    min_F = Inf
    min_H = nothing
    min_ε = nothing
    H_size = div(N * (N + 1), 2)
    OGM = get_vector(N)
    for i in 1:trials
        initial_vals = [0.00002 * rand(M) .+ 0.0002; OGM ./ (1.5 .+ 0.1 * rand(H_size))]
        ε_set, H, result, terminated_early = Optimize(N, L, R, p, M, initial_vals)
        T = Optim.f_calls(result)
        println("Trial $i done")
        if !terminated_early
            println(ε_set)
            display(H)
            if Optim.minimum(result) < min_F
                min_F = Optim.minimum(result)
                min_H = H
                min_ε = ε_set
            end
        end
    end

    return min_F, min_H, min_ε
end

function compute_theta(N)
    θ = zeros(N + 1)
    θ[1] = 1.0  # θ₀
    for i in 2:N
        θ[i] = (1 + sqrt(1 + 4 * θ[i-1]^2)) / 2
    end
    θ[N+1] = (1 + sqrt(1 + 8 * θ[N]^2)) / 2  # θ_N
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
        H[i, i] = 1 + (2 * θ[i] - 1) / θ[i+1]
    end
    return H
end



function get_vector(N)
    OGM = compute_H(N)
    vec_lower = zeros(div(N * (N + 1), 2))
    cnt = 0
    for i in 1:N
        for j in 1:i
            cnt = cnt + 1
            vec_lower[cnt] = OGM[i, j]
        end
    end
    return vec_lower
end



function gen_data(L, R, trials, p_cnt, N_cnt, M_cnt)
    ## TODO ##
    ## Note, M is taking the role of "k", so need to change that everywhere 
    results = Dict{Int, Dict{Int, Dict{String, Any}}}()

    for N = 1:N_cnt
        results[N] = Dict{Int, Dict{String, Any}}()
        for M = 1:M_cnt
            ε_p_data = Dict{Float64, Any}()
            F_p_data = Dict{Float64, Any}()
            H_p_data = Dict{Float64, Any}()

            for p in LinRange(0.9, 1, p_cnt)
                min_F, min_H, min_ε = run_batch_trials(N, L, R, p, M, trials)

                ε_p_data[p] = min_ε
                F_p_data[p] = min_F
                H_p_data[p] = min_H
            end


            results[N][M] = Dict(
                "F_values" => F_p_data,
                "ε_sets" => ε_p_data,
                "H_matrices" => H_p_data
            )
        end
    end
    return results
end

L, R = 1.0, 1.0
trials = 5
p_cnt = 20
N_cnt = 2
M_cnt = 5

results_3 = gen_data(L, R, trials, p_cnt, N_cnt, M_cnt)


