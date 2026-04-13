using JuMP
using LinearAlgebra
using MosekTools          

# --- Utilities to detect "star" arcs (i = -1) and index sets ---
is_star_arc(s) = (s.i == -1)
is_nonstar_pair_to_j(s, j) = (s.j == j && s.i >= 0 && s.i < j)  # arcs (i,j) with 0≤i<j
is_nonstar_pair_from_j(s, j) = (s.i == j && s.j > j)             # arcs (j,i) with j<i≤N
is_step_arc(s, N) = (s.i >= 0) && (s.j == s.i + 1) && (s.j <= N) 

# Build supporting tangents for ψ(u) = u * L^{-1}(u):  ψ(u) ≥ α u + β
function make_tangents(ugrid; ψ::Function, dψ::Function)
    [(dψ(u), ψ(u) - dψ(u) * u) for u in ugrid]
end




function idx_set_λ_constructor(I_N_star)

    # construct the index set for λ
    idx_set_λ = i_j_idx[]
    for i in I_N_star
        for j in I_N_star
            if i != j
                push!(idx_set_λ, i_j_idx(i, j))
            end
        end
    end

    return idx_set_λ

end

struct i_j_idx # correspond to (i,j) pair, where i,j ∈ I_N_⋆
    i::Int64 # corresponds to index i
    j::Int64 # corresponds to index j
end

"""
    solve_fractional_star_as_variables(
        N, R;
        ψ, dψ,                           # ψ(u)=u*L^{-1}(u) and its (sub)derivative
        idx_set_λ_constructor,           # your `index_set_constructor_for_dual_vars_full`
        ugrid = range(1e-6, stop=10.0, length=41),
        t_lower = 1e-8,
        solver = :mosek,                 # :mosek or :gurobi
        mosek_params = Dict{String,Any}(),
        gurobi_params = Dict{String,Any}(),
        show_output = :off,
        max_iter = 50,
        tol = 1e-8
    )

Solves:
    min  ((1/2)R^2 + ∑_{s} λ_s L^{-1}(λ_s / t_s)) / ( ∑_{j=0}^N λ_{-1,j} )
subject to
    ∑_{i=0}^{j-1} (-λ_{i,j}) + ∑_{i=j+1}^N λ_{j,i} = λ_{-1,j},          ∀ j ≠ N
    ∑_{i=0}^{N-1} λ_{i,N} = ∑_{i=0}^{N-1} λ_{-1,i},
    ∑_{i=0}^{j-1} (-t_{i,j}) + ∑_{i=j+1}^N (-t_{j,i}) - t_{-1,j} + λ_{-1,j}^2 ≤ 0,  ∀ j=0..N
with variables λ_s≥0, t_s≥t_lower, and star arcs s=(i=-1,j).

Returns:
    (ratio_opt, λ_opt, t_opt, τ, denom_star)
where τ is the last Dinkelbach parameter and denom_star = ∑_j λ_{-1,j}.
"""
function solve_fractional(
    N, R, tangents;
    t_lower=1e-8,
    solver=:mosek,
    mosek_params=Dict{String,Any}(),
    gurobi_params=Dict{String,Any}(),
    show_output=:off,
    max_iter=50,
    tol=1e-8)


    I_N_star = -1:N
    idx = idx_set_λ_constructor(I_N_star)   # objects with fields .i, .j
    I = 0:N

    # Precompute tangents for ψ

    # small helper to pick optimizer
    function _optimizer()
        if solver == :gurobi
            return optimizer_with_attributes(Gurobi.Optimizer)
        else
            return optimizer_with_attributes(Mosek.Optimizer)
        end
    end

    τ = 0.0      # Dinkelbach parameter
    ratio = Inf
    λ_sol = nothing
    t_sol = nothing
    denom_star = NaN

    for it in 1:max_iter
        model = Model(_optimizer())
        if show_output == :off
            set_silent(model)
        end
        if solver == :mosek
            for (k, v) in mosek_params
                set_optimizer_attribute(model, k, v)
            end
        else
            for (k, v) in gurobi_params
                set_optimizer_attribute(model, k, v)
            end
        end

        # Variables
        @variable(model, λ[idx] >= 0)
        @variable(model, t[idx] >= t_lower)
        @variable(model, z[idx] >= 0)     # epigraph variables

        # --- Constraints ---
        # helper: immediate-successor arc (i -> i+1), i ≥ 0, j ≤ N

        # allowed = star arcs (-1 -> j) OR immediate successor arcs (i -> i+1)
        allowed_idx = [s for s in idx if (is_star_arc(s) || is_step_arc(s, N))]
        forbidden_idx = setdiff(idx, allowed_idx)

        # Pin forbidden λ to zero by bounds (no extra rows in the model)
        for s in forbidden_idx
       #    set_upper_bound(λ[s], 0.0)
           #set_upper_bound(t[s], 0.0)

        end



        # (1) Flow-like balance to star, for j ≠ N
        for j in I
            j == N && continue
            @constraint(model,
                sum(-λ[s] for s in idx if is_nonstar_pair_to_j(s, j)) +
                sum(λ[s] for s in idx if is_nonstar_pair_from_j(s, j)) ==
                sum(λ[s] for s in idx if is_star_arc(s) && s.j == j)
            )
        end

        # (2) Terminal balance at N:  Σ_{i=0}^{N-1} λ_{i,N} = Σ_{i=0}^{N-1} λ_{-1,i}
        safe_sum(exprs) = isempty(exprs) ? 0.0 : sum(exprs)

        @constraint(model,
            safe_sum([λ[s] for s in idx if (s.j == N) && (s.i >= 0) && (s.i <= N - 1)]) ==
            safe_sum([λ[s] for s in idx if is_star_arc(s) && (s.j >= 0) && (s.j <= N - 1)])
        )

        # (3) t-inequalities for each j:  ... - t_{-1,j} + λ_{-1,j}^2 ≤ 0
        for j in I
            # left sums over nonstar arcs into and out of j
            @constraint(model,
                sum(-t[s] for s in idx if is_nonstar_pair_to_j(s, j)) +
                sum(-t[s] for s in idx if is_nonstar_pair_from_j(s, j)) -
                sum(t[s] for s in idx if is_star_arc(s) && s.j == j) +
                (sum(λ[s] for s in idx if is_star_arc(s) && s.j == j))^2 <= 0
            )
        end


        # (4) sparsity pattern 



        # Perspective epigraph cuts:  z[s] ≥ α*λ[s] + β*t[s]
        for (αt, βt) in tangents
            @constraint(model, [s in idx], z[s] >= αt * λ[s] + βt * t[s])
        end

        # Denominator G(λ) = Σ_j λ_{-1,j}
        star_idx = [s for s in idx if is_star_arc(s) && (s.j in I)]

        @expression(model, G, sum(λ[s] for s in star_idx; init=0.0))
        denom_lb = 1e-9  # tweak as you like

        @constraint(model, G >= denom_lb)


        # Numerator surrogate F(λ,t) ≈ (1/2)R^2 + Σ z[s]
        @expression(model, F, 0.5 * R^2 + sum(z[s] for s in idx))

        # Dinkelbach subproblem:  minimize  F - τ*G
        @objective(model, Min, F - τ * G)

        optimize!(model)

        term = termination_status(model)
        display(term)

        # Extract and update
        λ_val = value.(λ)
        t_val = value.(t)
        z_val = value.(z)

        F_val = 0.5 * R^2 + sum(z_val[s] for s in idx)
        G_val = sum(λ_val[s] for s in star_idx) + 0.001



        ratio_new = F_val / G_val
        gap = abs(F_val - τ * G_val)   # Dinkelbach stopping metric

        # save incumbent
        ratio = ratio_new
        λ_sol = λ_val
        t_sol = t_val
        denom_star = G_val

        # check convergence
        if gap <= tol
            break
        end
        τ = ratio_new
    end

    return ratio, λ_sol, t_sol, τ, denom_star
end


# You supply ψ and dψ (built from your L):
β = 1
N = 7
R = 1
q = 1


L(δ) = β^2/(2*δ)
L_inv(δ) = β^2/(2*δ)
ψ(u) = β^2/2                 # placeholder; replace with u * Linv(u)
dψ(u) = 0                 # placeholder derivative


ugrid = range(1e-6, stop=7.0, length=41)
tangents = make_tangents(ugrid; ψ=ψ, dψ=dψ)


ratio_opt, λ_opt, t_opt, τ_final, denom_star = solve_fractional(N, R, tangents)
display(ratio_opt)
display(β * R / (sqrt(2 * (N + 1))))

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

λ_mat = get_λ_matrices(λ_opt, N, 0.0001)
t_mat = get_λ_matrices(t_opt, N, 0.0001)
δ_mat = @. L_inv(λ_mat / t_mat)
display([δ_mat[i,i+1] for i = 2:N+1])
#display(t_mat)
display(λ_mat)

display(τ_final)
display(β * R/(sqrt(2*(N+1))))