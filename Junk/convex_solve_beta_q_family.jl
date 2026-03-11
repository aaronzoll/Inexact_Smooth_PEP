using JuMP
using LinearAlgebra
using Ipopt            # single-shot smooth NLP solver

# --- Utilities to detect "star" arcs (i = -1) and index sets ---
is_star_arc(s) = (s.i == -1)
is_nonstar_pair_to_j(s, j) = (s.j == j && s.i >= 0 && s.i < j)   # arcs (i,j) with 0≤i<j
is_nonstar_pair_from_j(s, j) = (s.i == j && s.j > j)               # arcs (j,i) with j<i≤N
is_step_arc(s, N) = (s.i >= 0) && (s.j == s.i + 1) && (s.j <= N)

struct i_j_idx
    i::Int64
    j::Int64
end

function idx_set_λ_constructor(I_N_star)
    idx_set_λ = i_j_idx[]
    for i in I_N_star, j in I_N_star
        i != j && push!(idx_set_λ, i_j_idx(i, j))
    end
    return idx_set_λ
end


function solve_fractional(N, R, β, q;
    t_lower=1e-8, λ_lower=1e-8, s_lower=1e-8, show_output=:off, λ_sparsity=:OGM, t_sparsity=:OGM)
    I_N_star = -1:N
    idx = idx_set_λ_constructor(I_N_star)
    I = 0:N

    a = 1 - 1 / q      # exponent on λ
    b = 1 / q          # exponent on t
    c = β^(1 / q)

    model = Model(Ipopt.Optimizer)
    if show_output == :off
        set_silent(model)
    else
        set_optimizer_attribute(model, "print_level", 5)
    end
    # Mild regularization for numeric stability
    set_optimizer_attribute(model, "tol", 1e-9)

    # Variables (these are the SCALED ones; originals are (λ/s, t/s))
    @variable(model, λ[s in idx] >= λ_lower)
    @variable(model, t[s in idx] >= t_lower)
    @variable(model, s >= s_lower)   # scaling variable
    set_start_value(s, 1.0)

    for s_ in idx
        set_start_value(λ[s_], 1e-3)     # small positive start
        set_start_value(t[s_], 1.0)
    end

    # Sparsity: only star (-1->j) or immediate successor (i->i+1) arcs can be nonzero
    allowed_idx = [s_ for s_ in idx if (is_star_arc(s_) || is_step_arc(s_, N))]
    forbidden_idx = setdiff(idx, allowed_idx)
    for s_ in forbidden_idx
        if λ_sparsity == :OGM
            set_upper_bound(λ[s_], λ_lower)
        end
        if t_sparsity == :OGM
            set_upper_bound(t[s_], t_lower)
        end
    end

    # (A) Flow-like balance to star, for j ≠ N  (scaled form is identical)

    for j in I
        j == N && continue
         @constraint(model,
            sum(-λ[s_] for s_ in idx if is_nonstar_pair_to_j(s_, j)) +
            sum(λ[s_] for s_ in idx if is_nonstar_pair_from_j(s_, j)) ==
            sum(λ[s_] for s_ in idx if is_star_arc(s_) && s_.j == j)
        )
    end

    # (B) Terminal balance at N:  Σ_{i=0}^{N-1} λ_{i,N} = Σ_{i=0}^{N-1} λ_{-1,i}
    safe_sum(exprs) = isempty(exprs) ? 0.0 : sum(exprs)

    @constraint(model,
        safe_sum([λ[s] for s in idx if (s.j == N) && (s.i >= 0) && (s.i <= N - 1)]) ==
        safe_sum([λ[s] for s in idx if is_star_arc(s) && (s.j >= 0) && (s.j <= N - 1)])
    )


    # (C) t-inequalities (scaled):  ... - t_{-1,j} + λ_{-1,j}^2 / s ≤ 0,  for each j
    for j in I
        @NLconstraint(model,
            sum(-t[s_] for s_ in idx if is_nonstar_pair_to_j(s_, j)) +
            sum(-t[s_] for s_ in idx if is_nonstar_pair_from_j(s_, j)) -
            sum(t[s_] for s_ in idx if is_star_arc(s_) && s_.j == j) +
            ((sum(λ[s_] for s_ in idx if is_star_arc(s_) && s_.j == j))^2) / s
            <=
            0.0
        )
    end

    # (D) Denominator normalization: sum_{j=0}^N λ_{-1,j} = 1
     @constraint(model, sum(λ[s_] for s_ in idx if is_star_arc(s_) && (s_.j in I)) == 1.0)
    # Objective: 0.5*R^2*s + sum_s c * λ^a * t^b
    @NLobjective(model, Min, 0.5 * R^2 * s +
                             sum(c * (λ[s_]^a) * (t[s_]^b) for s_ in idx)
    )

    optimize!(model)



    term = termination_status(model)
    display(term)
    # if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
    #     error("Single-solve NLP did not reach optimality (status = $term).")
    # end

    λ_opt = value.(λ)
    t_opt = value.(t)
    s_opt = value(s)
    obj = objective_value(model)
    return obj, λ_opt, t_opt, s_opt
end

function get_λ_matrices(λ_opt, N, TOL)
    λ_matrices = zeros(N + 2, N + 2)
    for i in -1:N
        for j in -1:N
            if i == j
                continue
            end
            if λ_opt[i_j_idx(i, j)] > TOL
                λ_matrices[i+2, j+2] = λ_opt[i_j_idx(i, j)]
            end
        end
    end
    return λ_matrices
end



function run_test(N, R, β, q; sparsity_pattern=:OGM, test_type=:general, plotting=:off)

    L(δ) = β / (δ^q)
    L_inv(u) = (β / u)^(1 / q)

    if test_type == :subgrad_diff
        obj, λ_opt, t_opt, s_opt = solve_fractional(N, R, β, q; show_output=:off, λ_sparsity=sparsity_pattern, t_sparsity=sparsity_pattern)
        println("Minimax Optimal objective = ", sqrt(β) * R / sqrt(N + 1))
        println("Numerical objective = ", obj)



        λ_mat = get_λ_matrices(λ_opt, N, 0.0001)
        t_mat = get_λ_matrices(t_opt, N, 0.0001)
        δ_mat = @. L_inv(λ_mat / t_mat)
        δ_vals = [δ_mat[i, i+1] for i = 2:N+1]
        δ_vals_theo = zeros(N)
        for i = 1:N
            if mod(i, 2) == 1
                δ_vals_theo[i] = sqrt(β) * R / sqrt(N + 1) * 1 / i
            end
        end
        println("Optimal δ set:")
        display(δ_vals)
        println("δ set different from theory: ", norm(δ_vals - δ_vals_theo))
        # display(t_mat)
        # display(λ_mat)

        if plotting == :on
            X = 1:N
            Y1 = δ_vals[X]
            display(t_mat)
            display(λ_mat)
            plot()

            scatter!(X, Y1)
        end

        return obj, λ_mat, t_mat, δ_vals, norm(δ_vals - δ_vals_theo)

    end

    obj, λ_opt, t_opt, s_opt = solve_fractional(N, R, β, q; show_output=:off, λ_sparsity=sparsity_pattern, t_sparsity=sparsity_pattern)
    display((q + 1)^((q - 1) / (q + 1)) / q^(q / (q + 1)) * β^(1 / (1 + q)) * R^(2 / (1 + q)) / (N + 1)^((2 - q) / (1 + q)))
    println("Theoretical objective bound = ", obj)


    λ_mat = get_λ_matrices(λ_opt, N, 0.0001)
    t_mat = get_λ_matrices(t_opt, N, 0.0001)
    δ_mat = @. L_inv(λ_mat / t_mat)
    δ_vals = [δ_mat[i, i+1] for i = 2:N+1]
    println("Optimal δ set:")

    display(δ_vals)


    X = 1:N
    Y1 = δ_vals[X]
    Y2 = @. q * β^(q + 1) * R^(2 / (q + 1)) / (2^(q / (q + 1)) * (1 + q)^(2 / (q + 1)) * (N)^(1 / (q + 1))) * X^(-2 / (q + 1))
    println("δ set different from theory: ", norm(Y1 - Y2))

    if plotting == :on
        display(t_mat)
        display(λ_mat)
        plot()

        scatter!(X, Y1)
        plot!(X, Y2)
    end

    return obj, λ_mat, t_mat, δ_vals, norm(Y1 - Y2)

end



# -------- Bounded subgrad diff case, q = 0 -------------
N = 17
R = 1
β = 1
q = 0.7

obj, λ_mat, t_mat, δ_vals, δ_diff = run_test(N, R, β, q; sparsity_pattern=:OGM, test_type=:subgrad_diff, plotting=:off)



