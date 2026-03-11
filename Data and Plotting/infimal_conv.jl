using LinearAlgebra
using Optim
using CairoMakie
using Random


to_vec(x::Number) = [x]
to_vec(x::AbstractVector) = x

innerlike(g::Number, z::Number) = g * z
innerlike(g::AbstractVector, z::AbstractVector) = dot(g, z)

# -------------------------
# Piecewise-linear convex max
# p(x) = max_i (F[i] + <G[i], x - X[i]>)
# -------------------------
function max_affine(x, X::AbstractVector, F::AbstractVector, G::AbstractVector)
    xv = to_vec(x)
    @assert length(X) == length(F) == length(G)
    best = -Inf
    for i in eachindex(F)
        Xi = to_vec(X[i])
        Gi = to_vec(G[i])
        @assert length(Xi) == length(xv) == length(Gi)
        val = F[i] + innerlike(Gi, xv .- Xi)
        best = max(best, val)
    end
    return best
end

# -------------------------
# Inner minimization: min_y f(y) + g(x - y)
# -------------------------

"""
derivative_free_minimize(ϕ, y0; restarts, radius, seed)
Uses NelderMead in the general setting
Adds a few random restarts around y0 to reduce variance.

Returns: (ystar, ϕ(ystar))
"""
function derivative_free_minimize(
    ϕ::Function,
    y0::AbstractVector;
    restarts::Int=3,
    radius::Real=1.0,
    seed::Int=0,
    nm_iters::Int=2500
)
    rng = MersenneTwister(seed)
    best_y = copy(y0)
    best_val = ϕ(best_y)

    # local runner
    function run_from(yinit)
        opts = Optim.Options(iterations=nm_iters, show_trace=false)
        # NOTE: options is positional, NOT a keyword
        res = optimize(ϕ, yinit, NelderMead(), opts)
        yhat = Optim.minimizer(res)
        vhat = Optim.minimum(res)
        return yhat, vhat
    end

    # run from y0
    yhat, vhat = run_from(y0)
    if vhat < best_val
        best_y, best_val = yhat, vhat
    end

    # random restarts
    for k in 1:restarts
        yinit = y0 .+ radius .* randn(rng, length(y0))
        yhat, vhat = run_from(yinit)
        if vhat < best_val
            best_y, best_val = yhat, vhat
        end
    end

    return best_y, best_val
end

"""
subgradient_minimize(ϕ, subg, y0; iters, α0, decay, tol)

Unconstrained subgradient method with diminishing steps α_k = α0 / sqrt(k).
If subgradient oracle provided, use this method as it is probably faster/more reliable

Returns: (ystar, ϕ(ystar))
"""
function subgradient_minimize(
    ϕ::Function,
    subg::Function,
    y0::AbstractVector;
    iters::Int=5000,
    α0::Real=1.0,
    decay::Real=0.0,  # optional extra decay: α_k = α0 / (sqrt(k) * (1 + decay*k))
    tol::Real=1e-8
)
    y = copy(y0)
    best_y = copy(y0)
    best_val = ϕ(best_y)

    for k in 1:iters
        gk = subg(y)
        ng = norm(gk)
        if ng ≤ tol
            # already near stationary in subgradient sense
            break
        end
        αk = α0 / (sqrt(k) * (1 + decay * k))
        y .-= (αk / max(ng, tol)) .* gk  # normalized step to avoid huge jumps
        vk = ϕ(y)
        if vk < best_val
            best_val = vk
            best_y .= y
        end
    end

    return best_y, best_val
end

"""
solve_infconv_argmin(f, g, x; y0, subgrad_f, subgrad_g, solver=:nelder_mead)

Solves: min_y f(y) + g(x - y)

- If solver=:subgradient, you should provide subgrad_f and subgrad_g.
  Then subgradient of ϕ(y) is: s(y) ∈ ∂f(y) - ∂g(x-y)
  (take s_f - s_g where s_f∈∂f(y), s_g∈∂g(x-y)).
- If solver=:nelder_mead, no oracles required.
"""
function solve_infconv_argmin(
    f::Function,
    g::Function,
    x;
    y0=nothing,
    subgrad_f::Union{Nothing,Function}=nothing,
    subgrad_g::Union{Nothing,Function}=nothing,
    solver::Symbol=:nelder_mead,
    # Nelder-Mead settings
    nm_restarts::Int=3,
    nm_radius::Real=1.0,
    nm_seed::Int=0,
    nm_iters::Int=2500,
    # Subgradient settings
    sg_iters::Int=5000,
    sg_α0::Real=1.0,
    sg_decay::Real=0.0
)
    xv = to_vec(x)
    d = length(xv)

    y_init = if y0 === nothing
        copy(xv)
    else
        yi = to_vec(y0)
        @assert length(yi) == d
        copy(yi)
    end

    ϕ(y) = f(y) + g(xv .- y)

    if solver == :subgradient
        @assert subgrad_f !== nothing "solver=:subgradient requires subgrad_f"
        @assert subgrad_g !== nothing "solver=:subgradient requires subgrad_g"

        subϕ(y) = begin
            sf = subgrad_f(y)          # ∈ ∂f(y)
            sg = subgrad_g(xv .- y)    # ∈ ∂g(x-y)
            return sf .- sg
        end

        return subgradient_minimize(ϕ, subϕ, y_init; iters=sg_iters, α0=sg_α0, decay=sg_decay)
    elseif solver == :nelder_mead
        return derivative_free_minimize(ϕ, y_init; restarts=nm_restarts, radius=nm_radius, seed=nm_seed, nm_iters=nm_iters)
    else
        error("Unknown solver=$(solver). Use :nelder_mead or :subgradient.")
    end
end

# -------------------------
# Infimal convolution wrapper h(x)
# -------------------------
"""
infimal_convolution(f, g; kwargs...) -> h

Returns callable h(x) evaluating (f ⧆ g)(x) by solving inner min.
"""
function infimal_convolution(
    f::Function,
    g::Function;
    subgrad_f::Union{Nothing,Function}=nothing,
    subgrad_g::Union{Nothing,Function}=nothing,
    solver::Symbol=:nelder_mead,
    use_cache::Bool=true,
    # pass-through settings:
    nm_restarts::Int=3,
    nm_radius::Real=1.0,
    nm_seed::Int=0,
    nm_iters::Int=2500,
    sg_iters::Int=5000,
    sg_α0::Real=1.0,
    sg_decay::Real=0.0
)
    last_x = Ref{Any}(nothing)
    last_y = Ref{Any}(nothing)

    function h(x)
        y0 = (use_cache && last_y[] !== nothing) ? last_y[] : nothing
        ystar, val = solve_infconv_argmin(
            f, g, x;
            y0=y0,
            subgrad_f=subgrad_f,
            subgrad_g=subgrad_g,
            solver=solver,
            nm_restarts=nm_restarts,
            nm_radius=nm_radius,
            nm_seed=nm_seed,
            nm_iters=nm_iters,
            sg_iters=sg_iters,
            sg_α0=sg_α0,
            sg_decay=sg_decay
        )
        if use_cache
            last_x[] = to_vec(x)
            last_y[] = ystar
        end
        return val
    end

    return h
end

# -------------------------
# Plotting helpers (supports d=1 or d=2)
# -------------------------

# -------------------------
# Plotting helper: plot (X,F) points + tiny tangent/subgradient segments from G
# Works for:
#   - 1D: X[i] scalar or 1-vector, G[i] scalar or 1-vector -> draws short line with slope G[i]
#   - 2D: X[i] 2-vector, G[i] 2-vector -> draws short segment along direction G[i]
#
# Use `overlay=true` by passing an existing Axis; otherwise creates its own fig/axis.
# -------------------------

"""
plot_points_and_derivatives!(
    ax, X, F, G;
    dims=1,
    pointsize=10,
    seglen=0.3,
    normalize=true,
    label_points="(X,F)",
    label_segs="derivatives"
)

Overlays:
- scatter of points (X, F) when dims=1
- and small line segments representing G.

For dims=1:
  point is at (x_i, F_i)
  segment is: y = F_i + g_i*(x - x_i) for x in [x_i - seglen, x_i + seglen]

For dims=2:
  point is at (x1_i, x2_i) in the *input plane*; there is no single scalar F to plot in 3D
  so we overlay the points and segments on a 2D axis (x1,x2) slice.
  segment is centered at X_i, direction along G_i (optionally normalized), length 2*seglen.
"""
function plot_points_and_derivatives!(
    ax,
    X::AbstractVector,
    F::AbstractVector,
    G::AbstractVector;
    dims::Int=1,
    pointsize::Real=12,
    seglen::Real=0.3,
    normalize::Bool=true,
    label_points::AbstractString="(X,F)",
    label_segs::AbstractString="G segments"
)
    @assert length(X) == length(F) == length(G)

    if dims == 1
        xs = Float64[]
        ys = Float64[]
        for i in eachindex(X)
            xi = to_vec(X[i])[1]
            fi = F[i]
            push!(xs, xi)
            push!(ys, fi)
        end

        scatter!(ax, xs, ys; markersize=pointsize, label=label_points, color = :red)

        # draw small tangent lines
        segcolor = RGBf(0.3,0.5,0.7)
        radius = seglen
        for i in eachindex(X)
            xi = to_vec(X[i])[1]
            fi = F[i]
            gi = to_vec(G[i])[1]

            # direction in (x,y) for the affine line y = fi + gi*(x-xi)
            # unit direction = [1, gi] / sqrt(1 + gi^2)
            denom = sqrt(1 + gi^2)
            dx = radius / denom
            dy = gi * dx

            xL, yL = xi - dx, fi - dy
            xR, yR = xi + dx, fi + dy

            lines!(
                ax,
                [xL, xR],
                [yL, yR];
                linewidth=6,
                color=segcolor,
                label=(i == firstindex(X) ? label_segs : nothing),
            )
        end

        return ax

    elseif dims == 2
        # Overlay in the x-plane (Axis), not Axis3.
        x1s = Float64[]
        x2s = Float64[]
        for i in eachindex(X)
            Xi = to_vec(X[i])
            @assert length(Xi) == 2
            push!(x1s, Xi[1])
            push!(x2s, Xi[2])
        end

        scatter!(ax, x1s, x2s; markersize=pointsize, label="X points")

        for i in eachindex(X)
            Xi = to_vec(X[i])
            Gi = to_vec(G[i])
            @assert length(Xi) == 2 && length(Gi) == 2
            dir = copy(Gi)
            if normalize
                nrm = norm(dir)
                if nrm > 0
                    dir ./= nrm
                end
            end
            p1 = Xi .- seglen .* dir
            p2 = Xi .+ seglen .* dir
            lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]];
                linewidth=2, label=(i == firstindex(X) ? label_segs : nothing))
        end

        return ax
    else
        error("plot_points_and_derivatives! supports dims=1 or dims=2.")
    end
end

"""
Convenience wrapper to toggle overlay inside your 1D infconv plot.

Usage inside plot_infconv after you create `ax`:
    if overlay_derivs
        plot_points_and_derivatives!(ax, X, F, G; dims=1, seglen=0.25)
    end
"""
function maybe_overlay_points_and_derivatives!(
    ax, overlay::Bool,
    X::AbstractVector, F::AbstractVector, G::AbstractVector;
    dims::Int=1,
    pointsize::Real=10,
    seglen::Real=0.3,
    normalize::Bool=true
)
    if overlay
        plot_points_and_derivatives!(ax, X, F, G; dims=dims, pointsize=pointsize, seglen=seglen, normalize=normalize)
    end
    return ax
end


function plot_infconv(
    f::Function,
    g::Function;
    dims::Int=1,
    plot_f::Bool=false,
    plot_g::Bool=false,
    # 1D domain
    xgrid=range(-3.0, 3.0, length=200),
    # 2D domain
    xgrid2=range(-3.0, 3.0, length=80),
    ygrid2=range(-3.0, 3.0, length=80),
    # infconv settings
    subgrad_f::Union{Nothing,Function}=nothing,
    subgrad_g::Union{Nothing,Function}=nothing,
    solver::Symbol=:nelder_mead,
    use_cache::Bool=true,
    nm_restarts::Int=2,
    nm_radius::Real=0.5,
    nm_seed::Int=0,
    nm_iters::Int=1500,
    sg_iters::Int=5000,
    sg_α0::Real=1.0,
    sg_decay::Real=0.0,
    overlay_derivs::Bool=false,
    X::Union{Nothing,AbstractVector}=nothing,
    F::Union{Nothing,AbstractVector}=nothing,
    G::Union{Nothing,AbstractVector}=nothing,
)
    h = infimal_convolution(
        f, g;
        subgrad_f=subgrad_f,
        subgrad_g=subgrad_g,
        solver=solver,
        use_cache=use_cache,
        nm_restarts=nm_restarts,
        nm_radius=nm_radius,
        nm_seed=nm_seed,
        nm_iters=nm_iters,
        sg_iters=sg_iters,
        sg_α0=sg_α0,
        sg_decay=sg_decay
    )

    if dims == 1
        xs = collect(xgrid)
        hv = [h([x]) for x in xs]

        fig = Figure(size=(1000, 650))

        ax = Axis(fig[1, 1], xlabel="x", ylabel="value", title="Infimal convolution h = f ⊞ g")
        lines!(ax, xs, hv, linewidth=3, label="h(x)", color = :orange)

        # Example: overlay sampled supporting hyperplanes / subgrad info
        # X :: Vector of points (each scalar or 1-vector)
        # F :: Vector of function values at X
        # G :: Vector of subgradients at X (each scalar or 1-vector)
        if overlay_derivs
            @assert X !== nothing && F !== nothing && G !== nothing
            maybe_overlay_points_and_derivatives!(ax, true, X, F, G; dims=1, seglen=0.25, pointsize=16)
        end
        if plot_f
            fv = [f([x]) for x in xs]
            lines!(ax, xs, fv, linestyle=:dash, label="f(x)")
        end
        if plot_g
            gv = [g([x]) for x in xs]
            lines!(ax, xs, gv, linestyle=:dot, label="g(x)")
        end

        axislegend(ax)
        display(fig)
        return fig

    elseif dims == 2
        X = collect(xgrid2)
        Y = collect(ygrid2)
        Z = [h([x, y]) for x in X, y in Y]

        fig = Figure(size=(1100, 800))
        ax = Axis3(fig[1, 1], xlabel="x₁", ylabel="x₂", zlabel="value", title="h = f ⊞ g (surface)")
        surface!(ax, X, Y, Z)

        ax2 = Axis(fig[2, 1], xlabel="x₁", ylabel="x₂", title="h = f ⊞ g (contours)")
        contour!(ax2, X, Y, Z)

        display(fig)
        return fig
    else
        error("plot_infconv supports dims=1 or d=2. For d>2, plot a 1D line or 2D slice.")
    end
end

# -------------------------
# Example usage
# -------------------------

f(y) = 1 * norm(y, 1)


# set X, F, G if want max of affine
X = [[-1.0], [0.0], [1.0]]
F = [1.0, 0.3, 0.2]
G = [[-1.0], [-0.5], [0.2]]
g(z) = max_affine(z, X, F, G)
#g(z) = 1/2*dot(z,z)

# Derivative-free is the safe default for black-box convex, nonsmooth f,g:
# Use Nelder Mead and turn overlay_derivs = false if not using max_affine
plot_infconv(f, g; dims=1, plot_f=true, plot_g=true,
    solver=:nelder_mead,
    xgrid=range(-4, 4, length=250),
    nm_restarts=3, nm_radius=0.8, nm_iters=2000,
    overlay_derivs=true, X, F, G)



