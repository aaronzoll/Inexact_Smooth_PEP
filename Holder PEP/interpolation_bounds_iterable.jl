using ForwardDiff
using Optim
using LinearAlgebra
using Plots



# --- Helpers that *bind* θ and ∇θ to a given L(δ) -----------------------------

# θ and θ_star, parameterized by the chosen L
make_θ(L; δmax=100.0, δmin=1e-12) = x -> begin
    norm2 = isa(x, Number) ? x^2 : dot(x, x)
    obj(δ) = L(δ) / 2 * norm2 + δ
    res = optimize(obj, δmin, δmax)      # 1D bounded minimization (avoid δ=0)
    Optim.minimum(res)
end

make_θ_star(L; δmax=100.0, δmin=1e-12) = x -> begin
    norm2 = isa(x, Number) ? x^2 : dot(x, x)
    obj(δ) = -1 / (2 * L(δ)) * norm2 + δ
    res = optimize(obj, δmin, δmax)      # 1D bounded minimization
    -Optim.minimum(res)
end

# Automatic derivatives bound to the θ / θ_star we pass in
make_∇θ(θ) = x -> isa(x, Number) ? ForwardDiff.derivative(θ, x) : ForwardDiff.gradient(θ, x)
make_∇θ_star(θ_star) = x -> isa(x, Number) ? ForwardDiff.derivative(θ_star, x) : ForwardDiff.gradient(θ_star, x)

# --- Your objective wrappers, but parameterized by θ and ∇θ -------------------

# κ = max_r 2|∇θ(r)| / |∇θ(2r)|
compute_kappa(θ, ∇θ) = begin
    obj(r) = -(2 * abs(∇θ(r[1])) / abs(∇θ(2r[1])))   # negate for maximization
    res = optimize(obj, [0.0], [20.0], [0.3], Fminbox(BFGS()))
    -Optim.minimum(res)
end

# c_L = sup_{r,s} (θ(s) - θ(r) - ∇θ(r)*(s-r)) / θ(s-r)
compute_cL(θ, ∇θ, options) = begin
    function obj_upper(vars)
        r, s = vars
        θr, θs = θ(r), θ(s)
        gr = ∇θ(r)                  # scalar in ℝ
        num   = θs - θr - gr * (s - r)
        denom = θ(s - r)

        ϵ = 1e-12                   # guard near denom≈0
        if abs(denom) ≤ ϵ
            return (num > 0) ? 1e12 : -1e12
        end
        return num / denom
    end
    obj_neg(v) = -obj_upper(v)

    lower = [-10.0, -10.0]
    upper = [ 10.0,  10.0]
    initial_vals = [-0.3, 1.0]

    res = optimize(obj_neg, lower, upper, initial_vals, Fminbox(BFGS()), options)
    -Optim.minimum(res)
end

# --- Sweep over p, recomputing L, θ, ∇θ each time -----------------------------

"""
    sweep_p!(p_list; β=1.0, options)

Given a vector (or any iterable) of scalar `p` values, compute κ and c_L for each and
store results into global arrays `kappa_save` and `c_L_save`.

- `β` can be a scalar or a vector matching `p` if you want mixtures.
- Uses your L(δ) structure: for vector p, L is the sum over components.
"""
function sweep_p!(p_list; β=1.0, options=Optim.Options(iterations=500))
    # ensure β aligns
    β_vec(p) = (isa(β, Number) ? [β] : β)

    # prepare outputs
    global kappa_save = similar(collect(p_list), Float64)
    global c_L_save   = similar(collect(p_list), Float64)

    idx = 1
    for p_val in p_list
        # Allow either scalar p or vector p per iteration.
        p_vec = isa(p_val, Number) ? [p_val] : collect(p_val)
        βv = (length(p_vec) == 1 && isa(β, Number)) ? [β] : β_vec(p_vec)

        @assert length(p_vec) == length(βv) "Length of p and β must match in this iteration."

        # Build L(δ) for this iteration (your original formula, generalized to vector p)
        L = δ -> begin
            # Note: expects δ > 0
            s = 0.0
            @inbounds for i in eachindex(p_vec)
                pi = p_vec[i]
                bi = βv[i]
                coeff = ((1 - pi) / (1 + pi) * 1 / (2 * δ))^((1 - pi) / (1 + pi))
                s += coeff * bi^(2 / (1 + pi))
            end
            s
        end

        # Bind θ, θ_star and their gradients to this L
        θ      = make_θ(L)
        θ_star = make_θ_star(L)      # not directly used here, but ready if needed
        ∇θ     = make_∇θ(θ)
        # ∇θ_star = make_∇θ_star(θ_star)

        # Compute metrics
        κ   = compute_kappa(θ, ∇θ)
        c_L = compute_cL(θ, ∇θ, options)

        # Save
        kappa_save[idx] = κ
        c_L_save[idx]   = c_L

        idx += 1
    end

    return kappa_save, c_L_save
end


iter_print_freq = 10
function my_callback(state)
    if state.iteration % iter_print_freq == 0 && state.iteration > 1
        println("Iter $(state.iteration): f = $(state.value)")
    end
    return false
end

options = Optim.Options(
    iterations=5000,
    f_abstol=1e-8,
    x_reltol=1e-8,
    time_limit=60,
    show_trace=false,
    callback=my_callback
)

# Define a sweep of p values (example)
p_list = collect(range(0.001, 0.999; length=30))   # or any vector of p you like

# Optional: set β (scalar or vector per iteration). Default is 1.0
β = 1.0

# Run the sweep; results land in global arrays kappa_save and c_L_save
kappa_save, c_L_save = sweep_p!(p_list; β=β, options=options)

println("kappa_save = ", kappa_save)
println("c_L_save   = ", c_L_save)
