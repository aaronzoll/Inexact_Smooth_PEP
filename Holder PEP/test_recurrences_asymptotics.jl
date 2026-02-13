# ===============================
# Recurrence + asymptotic fitting
# ===============================
using LinearAlgebra, Plots, OffsetArrays

struct RecParams{T<:Real}
    k1::T; k2::T; k3::T; r::T; p::T
end
RecParams(k1,k2,k3,r,p) = RecParams(promote(k1,k2,k3,r,p)...)

@inline ipow(i::Integer, pminus1) = (float(i))^(pminus1)

@inline function onestep_from_alpha_beta(tau_prev, αi, βi)
    root = sqrt(αi*αi + 4*βi*tau_prev)
    ψi = 0.5*(αi + root)
    τi = tau_prev + ψi
    return ψi, τi
end

"""
Exact original recurrence, including parity in the main loop and the special final step at i=N.
Returns (tau, psi, phi) arrays indexed 1..N+1 standing for i=0..N.
"""
function recurrence_original(N::Integer, P::RecParams)
    T = promote_type(typeof(P.k1),typeof(P.k2),typeof(P.k3),typeof(P.r),typeof(P.p))
    τ  = zeros(T, N+1)
    ψ  = zeros(T, N+1)
    ϕ  = zeros(T, N+1)
    p1 = P.p - one(T)

    # init: i=0
    τ[1] = P.k1^P.r + P.k3^P.r
    ψ[1] = τ[1]

    # main loop: i = 1..N-1
    for i in 1:max(N-1,0)
        ϕ[i+1] = τ[i]
        if isodd(i)
            αi = ipow(i+1, p1) * (P.k2^P.r + P.k3^P.r)
            βi = P.k1^P.r * ipow(i, p1) + P.k2^P.r * ipow(i+1, p1)
        else
            αi = ipow(i+1, p1) * (P.k1^P.r + P.k3^P.r)
            βi = P.k2^P.r * ipow(i, p1) + P.k1^P.r * ipow(i+1, p1)
        end
        ψ[i+1], τ[i+1] = onestep_from_alpha_beta(τ[i], αi, βi)
    end

    # special final step: i = N
    if N >= 1
        ϕ[N+1] = τ[N]
        A = ipow(N+1, p1) * (P.k3^P.r)       # α_N special
        B = P.k1^P.r * ipow(N, p1)           # β_N special
        ψ[N+1], τ[N+1] = onestep_from_alpha_beta(τ[N], A, B)
    end

    # backfill ψ,ϕ for i=2..N
    for i in 2:N
        ϕ[i] = τ[i-1]
        ψ[i] = τ[i] - τ[i-1]
    end
    return τ, ψ, ϕ
end

"""
Fit τ_i ≈ C * i^(p+1) + D * i^p on a tail i ∈ [i_min, N], using least squares.
- Choose i_min as floor(frac * N), default frac = 0.6
- Returns (Chat, Dhat, Cstar, Dstar, relerrC, relerrD, slope)
  where slope is the log-log slope ~ p+1 (computed on the tail).
"""
function fit_asymptotics(N::Integer, P::RecParams; frac=0.6)
    τ, _, _ = recurrence_original(N, P)

    # build tail index set (exclude i=0 since model is in i^powers)
    i_min = max(1, floor(Int, frac*N))
    I = collect(i_min:N)                # these are i-values (not array indices)
    T = promote_type(typeof(P.k1),typeof(P.k2),typeof(P.k3),typeof(P.r),typeof(P.p))
    A = zeros(T, length(I), 2)
    b = zeros(T, length(I))

    for (row, i) in enumerate(I)
        A[row,1] = (T(i))^(P.p + one(T))  # i^(p+1)
        A[row,2] = (T(i))^(P.p)           # i^p
        b[row]   = τ[i+1]                 # τ_i lives at index i+1
    end

    xhat = A \ b
    Chat, Dhat = xhat[1], xhat[2]

    # theory
    bsum = P.k1^P.r + P.k2^P.r
    Cstar = bsum / (P.p + one(T))^2
    Dstar = estimate_D(N,P)

    relerrC = abs(Chat - Cstar) / max(abs(Cstar), eps(T))
    relerrD = abs(Dhat - Dstar) / max(abs(Dstar), eps(T))

    # crude log-log slope check over the tail: slope ≈ d log τ / d log i ~ p+1
    slopes = Float64[]
    for k in 2:length(I)
        i1, i2 = I[k-1], I[k]
        t1, t2 = float(τ[i1+1]), float(τ[i2+1])
        push!(slopes, (log(t2)-log(t1)) / (log(i2)-log(i1)))
    end
    slope = sum(slopes)/length(slopes)

    return Chat, Dhat, Cstar, Dstar, relerrC, relerrD, slope
end

function estimate_D(N::Integer, P::RecParams; i0::Int=div(3N,4))
    τ, ψ, ϕ = recurrence_original(N, P)
    b = P.k1^P.r + P.k2^P.r
    C = b / (P.p + 1)^2
    s = zero(eltype(τ)); c = 0
    for i in i0:N
        s += (τ[i] - C*(float(i))^(P.p+1)) / (float(i))^(P.p)
        c += 1
    end
    return s / c
end


# --------------------------
# Example / quick experiment
# --------------------------
# Choose numbers (BigFloat works too; just pass BigFloat inputs)
function L_smooth(δ, β, p)
    return ((1 - p) / (1 + p) * 1 / (2 * δ))^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end


N = 20
β = 1
global p = 0.5
R = 1


L_eps = δ -> L_smooth(δ, β, p)
k = [exp(p) * (-sqrt(2) * p^0.25 + sqrt(2)), max(0, (7 - p^(-0.25)) * (p^(1 / 3) - p^(1 / 2)) * p^p), max(0, 1 / 7 * (p^(0.04 * p) - p^2))]
k = [0.00000003, 0.00000003, 0.00000001]

δ_set = compute_δ_set(N, k, p)

γ = β*R^(1+p)/((N+1)^((1+p)/2))
ϕ = 1/L_eps(γ)
k1 = k[1]
k2 = k[2]
k3 = k[3]


P = RecParams(k1, k2, k3, (p-1)/(p+1), p)  # k1,k2,k3,r,p
b = P.k1^P.r + P.k2^P.r

τ, ψ, φ = recurrence_original(N, P)
ψ = OffsetArray(ϕ * ψ, 0:N)
τ = ϕ * τ
φ = ϕ * φ


# --- Tauberian estimator for D ---


Chat, Dhat, Cstar, Dstar, eC, eD, slope = fit_asymptotics(N, P; frac=0.6)
println("Fitted C ≈ ", Chat, "   | Theory C* = ", Cstar, "   | rel.err = ", eC)
println("Fitted D ≈ ", Dhat, "   | Theory D* = ", Dstar, "   | rel.err = ", eD)
println("Log-log tail slope ≈ ", slope, "   (should be ~ p+1 = ", P.p + 1, ")")


 i = N
 
D_est = (τ[i] - b/(P.p+1)^2 * (i)^(P.p+1))/(i)^P.p
println("D_est = ", D_est)


rate = (1/2*R^2 + sum(φ[i]*δ_set[i] for i = 1:N)+ sum(ψ[i]*δ_set[N+1+i] for i = 0:N))/τ[end]
display(rate)
plot()
plot!(1:N+1, τ, linewidth = 4)
X = Int(3*N/4):N/50:N+1
scatter!(X, @. ϕ * ( Chat*X^(p+1)+Dhat*X^p))



