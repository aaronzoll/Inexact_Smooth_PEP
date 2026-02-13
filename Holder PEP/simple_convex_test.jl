using Convex
using MosekTools

"""
Solve  min_{λ,t ≥ 0} t^(1/q) / λ^(1/q - 1)
via variables x = log(λ), y = log(t):  minimize exp((1/q)*y - ((1-q)/q)*x).
Returns solver status and the recovered (λ,t) if bounded; otherwise expect unbounded.
"""
function solve_min_ratio(q; optimizer=Mosek.Optimizer)
    @assert 0 <= q <= 1 "q must be in (0,1)"

    t = Variable() 
    λ = Variable()  

    # Objective: exp((1/q)*y - ((1-q)/q)*x)
    obj = t^(1/q) / λ^(1/q - 1)
    constraints = []
    prob = minimize(obj, constraints)  # no extra constraints → unbounded problem

    solve!(prob, optimizer)

    # Recover λ, t from x, y (may be huge/small if solver returns a finite iterate)
    λ_val = evaluate(λ)
    t_val = evaluate(t)

    return (status = prob.status,
            optval = prob.optval,
            lambda = λ_val, t = t_val)
end

# example call
q = 0.5
res = solve_min_ratio(0.5)
println(res)