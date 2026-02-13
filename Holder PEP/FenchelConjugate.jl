module FenchelConjugate

export fenchel_conjugate
"""
    fenchel_conjugate(f; grad=nothing, dom=(-Inf, Inf), x0=0.0, tol=1e-9,
                      use_forwarddiff::Bool=true, max_expand=60)

Return a callable `ϕ` such that `ϕ(u)` evaluates the Fenchel conjugate `f*(u)`.

Arguments
- `f::Function`: convex, proper, lsc scalar function `R -> R∪{+Inf}`.
- `grad::Union{Nothing, Function}`: optional gradient `f'`. If omitted and
  `use_forwarddiff=true`, ForwardDiff is used (when possible). If neither is
  available, a safe derivative-free path is used.
- `dom=(a,b)`: domain (may include `±Inf`). Constrains the x-search.
- `x0`: starting point for bracketing / root search.
- `tol`: absolute tolerance on x-location (and objective consistency checks).
- `use_forwarddiff`: try `ForwardDiff.derivative(f, x)` if `grad==nothing`.
- `max_expand`: max geometric interval expansions for bracketing.

Returns
- `ϕ(u::Real) -> Real` (or `Inf`), works elementwise on arrays too.

Notes
- If the (constrained) minimum of `f(x)-u*x` occurs at a boundary and objective
  keeps decreasing beyond (and the domain is unbounded that direction),
  the routine returns `Inf` for `f*(u)`.
"""
function fenchel_conjugate(f; grad=nothing, dom=(-Inf, Inf), x0=0.0, tol=1e-9,
                           use_forwarddiff::Bool=true, max_expand::Int=60)

    a, b = dom
    @assert a < b "Require dom[1] < dom[2]."

    # --- Derivative oracle (if available) ---
    _has_grad = grad !== nothing
    if !_has_grad && use_forwarddiff
        # Try to build a ForwardDiff gradient; if it fails at runtime,
        # we’ll fall back to derivative-free path.
        grad = x -> ForwardDiff.derivative(f, x)
        _has_grad = true
    end

    # --- Utilities: safe objective and linesearch helpers ---
    g(u,x) = f(x) - u*x

    function bracket_minimum(u; xstart=x0)
        # Geometric expansion to find [L,R] bracketing the (unique) minimizer of g(u,·).
        # Works for convex 1D; if domain bounded, we clamp and detect boundary minima.
        φ(x) = g(u, x)
        # Start inside domain
        x = clamp(xstart, a == -Inf ? xstart : max(a + 1e-6, xstart),
                            b ==  Inf ? xstart : min(b - 1e-6, xstart))
        step = (b - a) / 64
        if !isfinite(step) || step == 0
            step = 1.0
        end

        L = x - step; R = x + step
        L = max(L, a); R = min(R, b)
        fL, fX, fR = φ(L), φ(x), φ(R)

        # If already unimodal with x in the dip, return
        if fL ≥ fX ≤ fR
            return (L, R, false)  # false => not boundary-driven
        end

        # Decide direction to expand
        dir = fL < fR ? -1.0 : +1.0
        left_open = (a == -Inf); right_open = (b == Inf)

        t = 1.0
        for k in 1:max_expand
            t *= 1.6
            if dir > 0
                # expand right
                L = x; fL = fX
                x = R; fX = fR
                R_try = x + t*step
                R = min(R_try, b)
                fR = φ(R)
            else
                # expand left
                R = x; fR = fX
                x = L; fX = fL
                L_try = x - t*step
                L = max(L_try, a)
                fL = φ(L)
            end
            if fL ≥ fX ≤ fR
                return (L, R, false)
            end
            # If we hit boundary and function still decreases toward it, signal potential ∞
            if (L == a && fX > fL && left_open) || (R == b && fX > fR && right_open)
                return (L, R, true)  # boundary-driven, might be unbounded below
            end
        end
        return (L, R, false)
    end

    function golden_section(u, L, R)
        # Derivative-free convex 1D minimization on [L,R]
        φ(x) = g(u, x)
        ϕ = (sqrt(5)-1)/2  # golden ratio factor
        C = R - ϕ*(R - L)
        D = L + ϕ*(R - L)
        fC = φ(C); fD = φ(D)
        while (R - L) > tol
            if fC < fD
                R = D; D = C; fD = fC
                C = R - ϕ*(R - L); fC = φ(C)
            else
                L = C; C = D; fC = fD
                D = L + ϕ*(R - L); fD = φ(D)
            end
        end
        x̂ = (L + R)/2
        return x̂, φ(x̂)
    end

    function bisection_on_grad(u; xstart=x0)
        # Solve f'(x) = u using monotonicity of f' (convex f ⇒ f' is monotone).
        # Bracket by expansion until sign change of (f'(x)-u), then bisect.
        ϕ(x) = grad(x) - u
        # Start with a small bracket around xstart
        L = max(a, xstart - 1.0)
        R = min(b, xstart + 1.0)
        # Expand until sign change or boundary
        left_open = (a == -Inf); right_open = (b == Inf)
        t = 1.0
        sL = ϕ(L); sR = ϕ(R)

        for _ in 1:max_expand
            if signbit(sL) != signbit(sR) && !(sL == 0 || sR == 0)
                break
            end
            t *= 1.6
            L_new = max(a, xstart - t)
            R_new = min(b, xstart + t)
            # If no expansion possible, stop
            if L_new == L && R_new == R
                break
            end
            L, R = L_new, R_new
            sL, sR = ϕ(L), ϕ(R)
        end

        # If no sign change and boundary is unbounded and signs suggest descent to ±∞ ⇒ f*(u)=Inf
        if !(signbit(sL) != signbit(sR) || sL == 0 || sR == 0)
            # If sL > 0 across, then f'(x) - u > 0 ⇒ f'(x) > u everywhere in [a,b]
            # If left is open, minimizer drifts to -∞ (if allowed) depending on slope; treat as ∞.
            if left_open && sL > 0
                return nothing, -Inf  # implies f*(u)=Inf
            end
            if right_open && sR < 0
                return nothing, -Inf
            end
            # Fall back to derivative-free within [L,R]
            x̂, m = golden_section(u, L, R)
            return x̂, m
        end

        # Bisection
        if sL == 0; return L, g(u,L) end
        if sR == 0; return R, g(u,R) end
        for _ in 1:200
            M = (L + R)/2
            sM = ϕ(M)
            if abs(sM) <= tol
                return M, g(u,M)
            end
            if signbit(sL) != signbit(sM)
                R = M; sR = sM
            else
                L = M; sL = sM
            end
            if abs(R - L) <= tol
                x̂ = (L + R)/2
                return x̂, g(u,x̂)
            end
        end
        x̂ = (L + R)/2
        return x̂, g(u,x̂)
    end

    # --- Main evaluator for a single u ---
    function eval_one(u::Real)
        # If we have a gradient (user or ForwardDiff), prefer first-order method.
        if _has_grad
            try
                x̂, m = bisection_on_grad(u; xstart=x0)
                if x̂ === nothing && m == -Inf
                    return Inf
                end
                return -m
            catch
                # ForwardDiff or grad might fail (nondifferentiable): fall back.
            end
        end

        # Derivative-free path
        L, R, boundary_flag = bracket_minimum(u; xstart=x0)
        if boundary_flag
            # The objective kept decreasing towards an open boundary: f*(u)=Inf
            return Inf
        end
        _, m = golden_section(u, L, R)
        return -m
    end

    ϕ(u::Real) = eval_one(u)
    ϕ(U::AbstractArray) = map(eval_one, U)

    return ϕ
end

end # module