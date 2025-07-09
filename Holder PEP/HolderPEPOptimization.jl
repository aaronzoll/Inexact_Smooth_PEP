module HolderPEPOptimization
include("BnB_PEP_Inexact_Smooth.jl")

using Optim, OffsetArrays

export OptimizationProblem, optimize_ε_h, get_H

struct OptimizationProblem
    N::Int
    L::Float64
    R::Float64
    p::Float64
    k::Int
end


function get_H(N, k, args)
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    cnt = 0
    for i in 1:N
        for j in 0:i-1
            H[i, j] = args[k+1+cnt]
            cnt += 1
        end
    end
    return H
end

function compute_α_from_h(prob::OptimizationProblem, h, μ)
    N, L = prob.N, prob.L
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for ℓ in 1:N
        for i in 0:ℓ-1
            if i == ℓ - 1
                α[ℓ, i] = h[ℓ, ℓ-1]
            elseif i <= ℓ - 2
                α[ℓ, i] = α[ℓ-1, i] + h[ℓ, i] - (μ / L) * sum(h[ℓ, j] * α[j, i] for j in i+1:ℓ-1)
            end
        end
    end
    return α
end

function compute_h_from_α(prob::OptimizationProblem, α, μ)
    N, L = prob.N, prob.L
    h_new = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for l in N:-1:1
        h_new[l, l-1] = α[l, l-1]
        for i in l-2:-1:0
            h_new[l, i] = α[l, i] - α[l-1, i] + (μ / L) * sum(h_new[l, j] * α[j, i] for j in i+1:l-1)
        end
    end
    return h_new
end

function run_batch(prob::OptimizationProblem, args)
    N, L, R, p, k = prob.N, prob.L, prob.R, prob.p, prob.k
    
    H = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    cnt = 0
    for i in 1:N
        for j in 0:i-1

            H[i, j] = args[k+1+cnt]
            cnt += 1
        end
    end
    ε_set = args[1:k]
    μ = 0
    α = compute_α_from_h(prob, H, μ)
    Y, _, _ = solve_primal_with_known_stepsizes_batch(N, L, α, R, ε_set, p, μ; show_output=:off)
    return Y
end

function optimize_ε_h(N, L, R, p, k, max_iter, max_time, lower, upper, initial_vals)

    prob = OptimizationProblem(N, L, R, p, k)
   
    terminated_early = false

    run_batch(prob, initial_vals)

    iter_print_freq = 100

    function my_callback(state)

        if state.iteration % iter_print_freq == 0
            println("Iter $(state.iteration): f = $(state.value)")
        end
        if state.value < 0
            println("Terminating early: negative objective detected (f = $(state.value))")
            terminated_early = true

            return terminated_early
        end
        return false
    end

    options = Optim.Options(
        iterations=max_iter,
        f_tol= 1e-8,
        x_tol= 1e-8,
        time_limit=max_time,
        show_trace=false,
        callback=my_callback
    )

    result = Optim.optimize(args -> run_batch(prob, args), lower, upper, initial_vals, Fminbox(NelderMead()), options)
    return result, terminated_early
end


end # module
