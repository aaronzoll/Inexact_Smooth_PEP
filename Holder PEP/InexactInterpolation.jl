#module InexactInterpolation

using CairoMakie # swap to GLMakie if you want interactive plots
using Printf
using Random, Optim
# -----------------------------
# θ-construction (generic + power-law special case)
# -----------------------------

"""
    theta_factory(; L=nothing, q=nothing, δmin=1e-12, δmax=1e6)

Return a callable θ(s) = inf_{δ≥0} ( L(δ)/2 * s^2 + δ ).

- If `q` is provided, assumes L(δ) = 1/δ^q with q ∈ [0,1] and uses the closed form.
- Otherwise, expects `L` to be a function δ -> L(δ) that is convex, nonincreasing, and positive;
  θ is computed by scalar minimization over δ ∈ [δmin, δmax] via Brent.

Notes:
- For q=0 (i.e., L(δ)=1), θ(s)= (1/2) s^2.
- For q>0, closed form is θ(s) = C(q) * |s|^{2/(q+1)} with
      t = (q/2)^(1/(q+1)),   C(q) = (1/2)*t^(-q) + t.
"""
function theta_factory(; L=nothing, q::Union{Nothing,Real}=nothing, δmin=1e-12, δmax=1e6)
    if q !== nothing
        @assert 0 ≤ q ≤ 1 "q must be in [0,1]"
        if q == 0
            return s -> 0.5 * (abs(s)^2)
        else
            t = (q/2)^(1/(q+1))
            C = 0.5 * t^(-q) + t
            expo = 2/(q+1)
            return s -> C * abs(s)^expo
        end
    else
        @assert L !== nothing "Provide either q or a function L(δ)."
        # Brent's method (dependency-free) on [δmin, δmax]
        # Unimodality is natural for typical L in this setting.
        function θ(s)
            s2 = (abs(s))^2
            φ(δ) = 0.5 * L(δ) * s2 + δ
            # Simple Brent implementation using Optim-like logic is long; instead,
            # do a coarse log search + local golden-section (robust & short).
            # 1) coarse search over log grid to find a bracket
            lo, hi = δmin, δmax
            ngrid = 80
            bestδ, bestv = lo, φ(lo)
            for t in range(log10(lo), log10(hi), length=ngrid)
                δ = 10.0^t
                v = φ(δ)
                if v < bestv
                    bestv = v
                    bestδ = δ
                end
            end
            # 2) Golden-section refine around bestδ
            width = 10.0
            a = max(δmin, bestδ/width)
            b = min(δmax, bestδ*width)
            ϕ = (sqrt(5)-1)/2
            c = b - ϕ*(b-a)
            d = a + ϕ*(b-a)
            fc = φ(c); fd = φ(d)
            iters = 80
            for _ in 1:iters
                if fc > fd
                    a = c; c = d; fc = fd; d = a + ϕ*(b-a); fd = φ(d)
                else
                    b = d; d = c; fd = fc; c = b - ϕ*(b-a); fc = φ(c)
                end
                if (b-a) ≤ 1e-10*max(1.0, bestδ)
                    break
                end
            end
            return min(fc, fd)
        end
        return θ
    end
end

# -----------------------------
# θ* construction (generic + power-law special case)
# -----------------------------

"""
    theta_star_factory(; L=nothing, q=nothing, δmin=1e-12, δmax=1e6)

Return a callable θ*(u) = sup_{δ≥0} ( u^2 / (2L(δ)) - δ ).

- If `q` is provided, assumes L(δ)=δ^{-q} with q ∈ [0,1] and uses closed forms:
    q = 0   : θ*(u) = 0.5 * u^2
    0<q<1   : θ*(u) = K(q) * |u|^(2/(1 - q)),  K(q) = ((1-q)/q) * (q/2)^(1/(1-q))
    q = 1   : θ*(u) = 0 if |u| ≤ √2, else +Inf
- Otherwise, expects `L` to be a positive, nonincreasing (typically convex) function.
  θ* is computed by maximizing over δ ∈ [δmin, δmax] via coarse log search + golden-section refine.

Notes:
- For `q` near 1, the 0/∞ indicator behavior is used for stability.
- For generic L, widen [δmin, δmax] if you suspect the maximizer sits at a boundary.
"""
function theta_star_factory(; L=nothing, q::Union{Nothing,Real}=nothing, δmin=1e-12, δmax=1e6)
    if q !== nothing
        @assert 0 ≤ q ≤ 1 "q must be in [0,1]"
        if q == 0
            return u -> 0.5 * (abs(u)^2)
        elseif q == 1
            return u -> (abs(u) ≤ sqrt(2.0) + 1e-12 ? 0.0 : Inf)
        else
            p = 2 / (1 - q)                                  # exponent on |u|
            K = ((1 - q) / q) * (q / 2)^(1 / (1 - q))        # coefficient
            return u -> K * abs(u)^p
        end
    else
        @assert L !== nothing "Provide either q or a function L(δ)."
        function θstar(u)
            u2 = abs(u)^2
            ψ(δ) = u2 / (2 * L(δ)) - δ  # objective to maximize

            # 1) Coarse log-grid scan to find a promising δ
            lo, hi = δmin, δmax
            ngrid = 80
            bestδ, bestv = lo, ψ(lo)
            @inbounds for t in range(log10(lo), log10(hi), length=ngrid)
                δ = 10.0^t
                v = ψ(δ)
                if v > bestv
                    bestv = v
                    bestδ = δ
                end
            end

            # 2) Golden-section refine around bestδ (maximize ψ == minimize -ψ)
            a = max(δmin, bestδ / 10)
            b = min(δmax, bestδ * 10)
            ϕ = (sqrt(5) - 1) / 2
            c = b - ϕ * (b - a)
            d = a + ϕ * (b - a)
            fc = -ψ(c); fd = -ψ(d)
            for _ in 1:80
                if fc < fd
                    b = d; d = c; fd = fc; c = b - ϕ * (b - a); fc = -ψ(c)
                else
                    a = c; c = d; fc = fd; d = a + ϕ * (b - a); fd = -ψ(d)
                end
                if (b - a) ≤ 1e-10 * max(1.0, bestδ)
                    break
                end
            end
            return max(ψ(c), ψ(d), bestv)
        end
        return θstar
    end
end


# -----------------------------
# Interpolability check
# -----------------------------

"""
    check_interpolable(x, f, g, θ; tol=1e-10)

Return (ok::Bool, violations::Vector{NamedTuple}) where ok is true iff
f_i - f_j - g_j*(x_i - x_j) - θ(g_i - g_j) >= -tol for all i,j.

`violations` contains items with (i, j, value) when the margin is < -tol.
"""
function check_interpolable(x::AbstractVector, f::AbstractVector, g::AbstractVector, θ_star; tol=1e-10)
    n = length(x)
    @assert length(f)==n && length(g)==n "x, f, g must have same length"
    violations = NamedTuple{(:i,:j,:margin)}[]
    ok = true
    for i in 1:n, j in 1:n
        Δ = f[i] - f[j] - g[j]*(x[i]-x[j]) - θ_star(g[i]-g[j])
        if Δ < -tol
            ok = false
            push!(violations, (i=i, j=j, margin=Δ))
        end
    end
    return ok, violations
end

# -----------------------------
# Plotting
# -----------------------------

"""
    plot_dataset(x, f, g, θ; w_frac=0.08, samples=200, curve_halfwidth=:auto, title="Interpolation Preview")

Create a CairoMakie figure showing:
  • data points (x_i, f_i)
  • short gradient ticks centered at each x_i with slope g_i
  • local θ-upper-bound curves u_i(x) = f_i + g_i*(x-x_i) + θ(x-x_i) on a small window around each x_i

Arguments:
- w_frac: controls the half-length of gradient ticks as a fraction of data x-range
- samples: number of samples per local θ-curve
- curve_halfwidth: if :auto, use 2 * tick half-length; otherwise, pass a numeric half-width
"""
function plot_dataset(x::AbstractVector, f::AbstractVector, g::AbstractVector, θ;
                      w_frac=0.08, samples=200, curve_halfwidth=:auto, title="Interpolation Preview")

    n = length(x)
    @assert length(f)==n && length(g)==n "x, f, g must have same length"

    # Axis ranges
    xmin, xmax = minimum(x), maximum(x)
    xr = xmax - xmin
    xr = xr == 0 ? 1.0 : xr
    tick_halfw = w_frac * xr
    local_halfw = curve_halfwidth === :auto ? 2tick_halfw : float(curve_halfwidth)

    # y-range heuristic (include points and short ticks)
    ymin = minimum(f .- abs.(g).*tick_halfw) - 0.1
    ymax = maximum(f .+ abs.(g).*tick_halfw) + 0.1

    fig = Figure(size=(900, 520))
    ax = Axis(fig[1,1]; title, xlabel="x", ylabel="value",
              limits = (xmin - 0.15xr, xmax + 0.15xr, ymin, ymax))

    # Points
    scatter!(ax, x, f; markersize=9)

    # Gradient ticks (short line with slope g_i centered at each x_i)
    for i in 1:n
        xi, fi, gi = x[i], f[i], g[i]
        xlo, xhi = xi - tick_halfw, xi + tick_halfw
        ylo = fi - gi*tick_halfw
        yhi = fi + gi*tick_halfw
        lines!(ax, [xlo, xhi], [ylo, yhi], linewidth=5)
    end

    # θ-upper-bound small curves near each point
    for i in 1:n
        xi, fi, gi = x[i], f[i], g[i]
        xs = range(xi - local_halfw, xi + local_halfw, length=samples)
        ys = similar(xs)
        @inbounds for (k, xv) in enumerate(xs)
            s = xv - xi
            ys[k] = fi + gi*s + θ(s)
        end
        lines!(ax, xs, ys, linewidth=3, linestyle=:dot)
    end

    fig
end

# -----------------------------
# Convenience runner
# -----------------------------

"""
    demo_powerlaw(; q=0.5, seed=1)

Quick demo:
- builds θ for L(δ)=1/δ^q,
- generates a small synthetic dataset,
- checks interpolability,
- makes the plot.
"""

function demo_powerlaw(; q=0.5, seed=1)
    srand = Random.MersenneTwister(seed)
    x = sort!(rand(srand, 4) .* 4 .- 2)           # five points in [-2,2]
    f = @. 0.2x^2 + 0.5x + 0.3 + 0.05*randn(srand) # a convex-ish signal + noise
    g = @. 0.4x + 0.5 + 0.02*randn(srand)          # derivative-ish signal + noise

    L(δ) = 1/δ^q
    θ = theta_factory(q=q)
    θ_star = theta_star_factory(q=q)
    ok, viol = check_interpolable(x, f, g, θ_star; tol=1e-10)
    @info "Interpolable?" ok
    if !ok
        @info "Violations:" length(viol)
        for v in viol
            @info @sprintf("  i=%d  j=%d  margin=%.3e", v.i, v.j, v.margin)
        end
    end

    fig = plot_dataset(x, f, g, θ; title=@sprintf("θ from L(δ)=1/δ^q, q=%.3f", q))
    return fig, (ok, viol), (x,f,g,θ)
end

# end # module
include("FenchelConjugate.jl")
using .FenchelConjugate

function interpolation(x, f, g; L=nothing, q = nothing)
    n = length(x)
    if q !== nothing 
        θ_star = theta_star_factory(q = q)
    else
        @assert L !== nothing
        θ_star = theta_star_factory(L = L)

    end
    r_star(u) = maximum([-f[i]+u*x[i]+θ_star(u-g[i]) for i = 1:n])
    return fenchel_conjugate(r_star)
end



q=0.3
fig, (ok, viol), (x,f,g,θ)  = demo_powerlaw(q=q, seed = 1)

r1 = interpolation(x, f, g; q= q)
X = LinRange(-2,2, 50)

lines!(X, @. r1(X))
display(fig)
