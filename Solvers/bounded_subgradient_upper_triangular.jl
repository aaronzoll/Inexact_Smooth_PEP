using JuMP
using LinearAlgebra
using MosekTools # or Ipopt, etc.
import MathOptInterface as MOI
using Ipopt
using Revise
using OffsetArrays

"""
    solve_fractional_model(N, R, beta)

Build and solve the fractional-linear convex program with indices
i,j ∈ {-1,0,...,N}, where λ[-1,j] = λ_{⋆,j}, t[-1,j] = t_{⋆,j}.

Objective:
    (1/2 R^2 + (β^2/2) * sum_{i,j} t[i,j]) / sum_{j=0}^N λ[-1,j]

Subject to:
    ∑_{i=0}^{j-1} -λ_{i,j} + ∑_{i=j+1}^N λ_{j,i} = λ_{⋆,j} = λ[-1,j],   j = 0,...,N-1
    -∑_{i=0}^{N-1} λ_{i,N} = -∑_{i=0}^{N-1} λ_{⋆,i} = -∑_{i=0}^{N-1} λ[-1,i]
    ∑_{i=0}^{j-1} -t_{i,j} + ∑_{i=j+1}^N -t_{j,i} - t_{⋆,j} + λ_{⋆,j}^2 ≤ 0, j = 0,...,N
"""
function solve_fractional_model(N::Int, R::Float64, beta::Float64)
    Js = -1:N
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    TOL = 1e-8
    # Domain: to keep λ_{i,j}/t_{i,j} > 0, we enforce nonnegativity.
    @variable(model, λ[i in Js, j in Js] >= 0)
    @variable(model, t[i in Js, j in Js] >= 0)

    # Denominator: sum_{j=0}^N λ[-1,j] > 0
    # (Ipopt prefers strict > 0, we enforce a small lower bound.)
    @constraint(model, sum(λ[-1, j] for j in 0:N) >= TOL)



    for i in -1:N, j in -1:(i-1)
        @constraint(model, λ[i, j] == TOL)
        @constraint(model, t[i, j] == TOL)
    end
    # Fractional-linear objective:
    # (1/2 R^2 + (β^2/2) * sum t[i,j]) / sum λ[-1,j]
    @NLobjective(
        model,
        Min,
        (0.5 * R^2 + 0.5 * beta^2 * sum(t[i, j] for i in Js, j in Js)) /
        (sum(λ[-1, j] for j in 0:N))
    )

    # Constraints:
    # 1) For j = 0,...,N-1:
    #    ∑_{i=0}^{j-1} -λ_{i,j} + ∑_{i=j+1}^N λ_{j,i} = λ[-1,j]
    for j in 0:(N-1)
        @constraint(
            model,
            sum(-λ[i, j] for i in 0:(j-1)) +
            sum(λ[j, i] for i in (j+1):N) ==
            λ[-1, j]
        )
    end

    # 2) For j = N:
    #    -∑_{i=0}^{N-1} λ_{i,N} = -∑_{i=0}^{N-1} λ[-1,i]
    @constraint(
        model,
        sum(-λ[i, N] for i in 0:(N-1)) ==
        -sum(λ[-1, i] for i in 0:(N-1))
    )

    # 3) For j = 0,...,N:
    #    ∑_{i=0}^{j-1} -t_{i,j} + ∑_{i=j+1}^N -t_{j,i} - t[-1,j] + λ[-1,j]^2 ≤ 0
    for j in 0:N
        @NLconstraint(
            model,
            sum(-t[i, j] for i in 0:(j-1)) +
            sum(-t[j, i] for i in (j+1):N) -
            t[-1, j] +
            λ[-1, j]^2
            <=
            0.0
        )
    end

    optimize!(model)
    return objective_value(model), value.(λ), value.(t)
end



function solve_fractional_model_mosek(N::Int, R::Float64, beta::Float64; TOL::Float64=1e-8)
    Js = -1:N

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Charnes–Cooper scaling variable
    @variable(model, s >= 1e-12)

    # Scaled variables: λ̃ = s*λ, t̃ = s*t
    @variable(model, λs[i in Js, j in Js] >= 0)
    @variable(model, ts[i in Js, j in Js] >= 0)

    # Aux vars for q[j] >= λ̃[-1,j]^2 / s  via rotated SOC
    @variable(model, q[j in 0:N] >= 0)

    # IMPORTANT: MOSEK disallows the SAME variable from appearing in multiple cones.
    # So we create one copy s_half[j] per cone, and tie it to s with linear equalities.
    @variable(model, s_half[j in 0:N] >= 1e-12)
    @constraint(model, [j in 0:N], s_half[j] == 0.5 * s)

    # Normalization: sum λ̃[-1,j] = 1
    @constraint(model, sum(λs[-1, j] for j in 0:N) == 1.0)

    # Optional: denom = 1/s >= TOL  <=>  s <= 1/TOL
    @constraint(model, s <= 1.0 / TOL)

    # Lower triangle fixed to TOL, but scaled: λ̃ = s*TOL, t̃ = s*TOL
    for i in -1:N, j in -1:i
        @constraint(model, λs[i, j] == TOL * s)
        @constraint(model, ts[i, j] == TOL * s)
    end

    for j in 0:N
        #  @constraint(model, ts[-1,j] >= 2*R^2/(beta^2 * N *(N+1)))
    end

    # Linear constraints with λ̃
    for j in 0:(N-1)
        @constraint(
            model,
            sum(-λs[i, j] for i in 0:(j-1)) +
            sum(λs[j, i] for i in (j+1):N) ==
            λs[-1, j]
        )
    end

    @constraint(
        model,
        sum(-λs[i, N] for i in 0:(N-1)) ==
        -sum(λs[-1, i] for i in 0:(N-1))
    )

    # Conic + linear constraints for each j
    for j in 0:N
        # -Σ t̃[...] - t̃[-1,j] + q[j] <= 0
        @constraint(
            model,
            sum(-ts[i, j] for i in 0:(j-1)) +
            sum(-ts[j, i] for i in (j+1):N) -
            ts[-1, j] +
            q[j] <= 0.0
        )

        # Rotated SOC: 2*q[j]*s_half[j] >= (λs[-1,j])^2
        @constraint(model, [q[j], s_half[j], λs[-1, j]] in MOI.RotatedSecondOrderCone(3))
    end

    # Linear objective after Charnes–Cooper
    @objective(
        model,
        Min,
        0.5 * R^2 * s + 0.5 * beta^2 * sum(ts[i, j] for i in Js, j in Js)
    )

    optimize!(model)

    s_val = value(s)
    obj_val = objective_value(model)

    # Recover original variables: λ = λ̃/s, t = t̃/s
    λ_val = value.(λs) ./ s_val
    t_val = value.(ts) ./ s_val
    q_val = value.(q)

    return obj_val, λ_val, t_val, s_val, q_val
end


function build_FSFOM_matrices(N, λ_matrix, t_matrix)
    # ---------------------------------------------------------------
    # H matrices are (N+1) × (N+1), rows index k = 0:N (output step),
    # cols index i = 0:N (gradient index).
    # x_k = x0 - Σ_i H[k,i] * g_i
    # ---------------------------------------------------------------

    # --- Absolute formulation: x_k = x0 - Σ H_abs[k,i] g_i ----------
    H_abs = OffsetArray(zeros(N + 1, N + 1), 0:N, 0:N)

    # z[:,j] = x0 - Σ_{i=0}^{j-1} λ[-1,i] * g[:,i]
    # So z has its own "H" coefficients:
    z_H = OffsetArray(zeros(N + 1, N + 1), 0:N, 0:N)  # z_H[j,i] = coeff of g_i in z[:,j]
    for j = 1:N
        for i = 0:j-1
            z_H[j, i] = λ_matrix[-1, i]
        end
    end

    # X3[:,0] = x0, so H_abs[0,:] = 0 (no gradient terms)
    # X3[:,j] = [Σ_{i=0}^{j-1} λ[i,j]*X[:,i] - t[i,j]*g[:,i]  +  λ[-1,j]*z[:,j]]
    #           / [Σ_{i=0}^{j-1} λ[i,j] + λ[-1,j]]
    # We expand X[:,i] recursively using H_abs[i,:] already computed.

    for j = 1:N
        denom = sum(λ_matrix[i, j] for i = 0:j-1) + λ_matrix[-1, j]

        # Contribution from Σ λ[i,j] * X[:,i]: expands via H_abs[i,:]
        for i = 0:j-1
            for col = 0:N
                H_abs[j, col] += λ_matrix[i, j] * H_abs[i, col]
            end
        end

        # Contribution from - Σ t[i,j] * g[:,i]
        for i = 0:j-1
            H_abs[j, i] += t_matrix[i, j]   # note: H stores positive coeff (x0 - H*g)
        end

        # Contribution from λ[-1,j] * z[:,j]: expands via z_H[j,:]
        for col = 0:N
            H_abs[j, col] += λ_matrix[-1, j] * z_H[j, col]
        end

        H_abs[j, :] ./= denom
    end

    # --- Incremental formulation: x_k = x_{k-1} - Σ H_inc[k,i] g_i --
    # H_inc[k,i] = H_abs[k,i] - H_abs[k-1,i]
    H_inc = OffsetArray(zeros(N + 1, N + 1), 0:N, 0:N)
    for k = 1:N
        for i = 0:N
            H_inc[k, i] = H_abs[k, i] - H_abs[k-1, i]
        end
    end

    return H_abs, H_inc
end


function run_fom1(x0::AbstractVector, g::AbstractMatrix, H, N::Int)
    d = length(x0)
    X = OffsetArray(zeros(eltype(x0), d, N + 1), 1:d, 0:N)
    X[:, 0] .= x0
    for j in 1:N
        X[:, j] .= x0
        for i in 0:j-1
            X[:, j] .-= H[j, i] .* g[:, i]
        end
    end
    return X
end

function run_fom2(x0::AbstractVector, g::AbstractMatrix, H, N::Int)
    d = length(x0)
    X = OffsetArray(zeros(eltype(x0), d, N + 1), 1:d, 0:N)
    X[:, 0] .= x0
    for j in 1:N
        X[:, j] .= X[:, j-1]
        for i in 0:j-1
            X[:, j] .-= H[j, i] .* g[:, i]
        end
    end
    return X
end

function check_foms(t_guess, λ_guess)
    H_abs, H_inc = build_FSFOM_matrices(N, λ_guess, t_guess)

    #H_inc = absolute_to_incremental(H, N)
    d = N + 1
    G = OffsetArray(Diagonal(ones(N + 1)), 1:d, 0:N)
    x0 = zeros(d)
    X1 = run_fom1(x0, G, H_abs, N)
    X2 = run_fom2(x0, G, H_inc, N)


    X3 = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    X3[:, 0] .= x0
    z = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    z[:, 1] = X3[:, 0] - λ_guess[-1, 0] * G[:, 0]

    for j = 1:N
        num = sum(λ_guess[i, j] .* X3[:, i] .- t_guess[i, j] .* G[:, i] for i = 0:j-1) + λ_guess[-1, j] .* z[:, j]
        denom = sum(λ_guess[i, j] for i = 0:j-1) + λ_guess[-1, j]

        X3[:, j] = num / denom
        if j < N
            z[:, j+1] = z[:, j] .- λ_guess[-1, j] .* G[:, j]
        end
    end

    X4 = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    y = OffsetArray(zeros(d, N + 1), 1:d, 0:N)
    X4[:, 0] .= x0
    d_vec = zeros(d)  # d_k = sum_{i=0}^{k-1} g_i

    for k = 1:N
        d_vec .+= G[:, k-1]  # accumulate g_{k-1}
        X4[:, k] = (k .* X4[:, k-1] .+ x0) ./ (k + 1) .- (sqrt(2) * R * sqrt(N + 1)) / (β * N * (k + 1)) .* d_vec
    end

    return X1, X2, X3, X4


end

function test_constraints(λ_mat, t_mat, N)

    const_1 = zeros(N)
    const_2 = 0
    const_3 = zeros(N+1)

    for j = 0:N-1
        const_1[j+1] = sum([-λ_mat[i, j] for i = 0:j-1]) + sum([λ_mat[j, i] for i = j+1:N]) - λ_mat[-1, j]
    end

    const_2 = -sum([λ_mat[-1,i] for i = 0:N-1])+sum([λ_mat[i,N] for i = 0:N-1])

    for j = 0:N
        const_3[j+1] = sum([-t_mat[i, j] for i = 0:j-1]) + sum([-t_mat[j, i] for i = j+1:N]) - t_mat[-1, j] + λ_mat[-1,j]^2
    end

    return const_1, const_2, const_3
end

function count_nonzeros(mat, TOL)

    cnt = 0
    for i = mat 
        if i > TOL
            cnt += 1
        end
    end

    return cnt

end

N = 8
R = 1.0
β = 1.0


val, λ_opt, t_opt, s_val, q_val = solve_fractional_model_mosek(N, R, β)
display(val)
display(β * R / (sqrt(2 * (N + 1))))


Js = axes(t_opt, 1)       
t_guess = JuMP.Containers.DenseAxisArray(zeros(length(Js), length(Js)), Js, Js)
λ_guess = JuMP.Containers.DenseAxisArray(zeros(length(Js), length(Js)), Js, Js)


for j in 0:N
    λ_guess[-1, j] = sqrt(2) * R / (β * sqrt((N + 1)))
end

# Off-diagonal: arcs (i, i+1), i = 0..N-1
for i in 0:(N-1)
    λ_guess[i, i+1] = sqrt(2) * R * (i + 1) / (β * sqrt((N + 1)))
end

for i in 0:(N-1)
    for j in i+1:N
        t_guess[i, j] = 2 * R^2 / (β^2 * (N) * (N + 1))
    end
end

for j in 0:N
    t_guess[-1, j] =  2 * R^2 / (β^2 * (N) * (N + 1))
end


display((0.5 * R^2 + 0.5 * β^2 * sum(t_guess)) / sum(λ_guess[-1, j] for j in 0:N))



X1, X2, X3, X4 = check_foms(t_guess, λ_guess)

const_1, const_2, const_3 = test_constraints(λ_guess, t_guess, N)



