using Optim, ForwardDiff, CurveFit, LinearAlgebra, OffsetArrays

function L_smooth(δ, β, p)
    return ((1 - p) / (1 + p) * 1 / (2 * δ))^((1 - p) / (1 + p)) * β^(2 / (1 + p))
end



function compute_δ_set(N, k, p)
    δ_set = zeros(2 * N + 1)
    k1 = k[1] # should only be function of p?
    k2 = k[2]
    k3 = k[3]
    for i = 1:N
        if i % 2 == 1
            δ_set[i] = k1 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))

        else
            #  δ_set[i] = max(0,(-1.33*10^-5 - 8.2*10^-7*N)*i + (0.00005*N-0.0002))
            δ_set[i] = k2 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i)^(-(1 + p))
        end


    end

    for i = 0:N
        δ_set[N+i+1] = k3 * β * R^((1 + p)) * (N + 1)^(-(1 + p) / 2) * (i + 1)^(-(1 + p))
    end

    return δ_set
end







##############
# Parameters #
##############

"""
Container for recurrence parameters.
All fields are promoted to a common floating-point type T for stability.
"""
struct RecParams{T<:Real}
    k1::T
    k2::T
    k3::T
    r::T
    p::T
end

RecParams(k1,k2,k3,r,p) = RecParams(promote(k1,k2,k3,r,p)...)

###########################
# Low-level helper pieces #
###########################

"""
(i)^(p-1) as a value of type T without integer overflow concerns.
"""
@inline ipow(i::Integer, p_minus_1) = (float(i))^(p_minus_1)

"""
One-step update for general (parity) rule at index i:
Given τ_{i-1} (= tau_prev) and current coefficients (α_i, β_i), returns (ψ_i, τ_i).
We pass α_i, β_i explicitly to keep this function branch-free.
"""
@inline function onestep_from_alpha_beta(tau_prev, αi, βi)
    # ψ_i = 1/2( α_i + sqrt(α_i^2 + 4 β_i τ_{i-1}) )
    root = sqrt(αi*αi + 4*βi*tau_prev)
    ψi = 0.5*(αi + root)
    τi = tau_prev + ψi
    return ψi, τi
end

#########################################
# Original (parity-based) reference impl #
#########################################

"""
Compute (τ, ψ, φ) for i = 0..N via the original rules with parity and the special N-step.
Returns (tau, psi, phi) as length N+1 vectors.
"""
function recurrence_original(N::Integer, P::RecParams)
    T = promote_type(typeof(P.k1),typeof(P.k2),typeof(P.k3),typeof(P.r),typeof(P.p))
    τ  = zeros(T, N+1)
    ψ  = zeros(T, N+1)
    ϕ  = zeros(T, N+1)

    p1 = P.p - one(T)          # p-1, reused

    # init
    τ[1] = (P.k1^P.r + P.k3^P.r)     # τ_0 (1-based array: index 1 is i=0)
    ψ[1] = τ[1]                      # ψ_0 = τ_0
    # φ_0 is unused; φ_1 = τ_0 when we first need it

    # main loop: i = 1..N-1
    for i in 1:max(N-1,0)
        ϕ[i+1] = τ[i]  # φ_i = τ_{i-1}
        if isodd(i)
            αi = ipow(i+1, p1) * (P.k2^P.r + P.k3^P.r)
            βi = P.k1^P.r * ipow(i, p1) + P.k2^P.r * ipow(i+1, p1)
        else
            αi = ipow(i+1, p1) * (P.k1^P.r + P.k3^P.r)
            βi = P.k2^P.r * ipow(i, p1) + P.k1^P.r * ipow(i+1, p1)
        end
        ψ[i+1], τ[i+1] = onestep_from_alpha_beta(τ[i], αi, βi)
    end

    # final step: i = N (special)
    if N >= 1
        ϕ[N+1] = τ[N]                                      # φ_N = τ_{N-1}
        A = ipow(N+1, p1) * (P.k3^P.r)                     # α_N special = (N+1)^(p-1) k3^r
        B = P.k1^P.r * ipow(N, p1)                         # β_N special = k1^r N^(p-1)
        ψ[N+1], τ[N+1] = onestep_from_alpha_beta(τ[N], A, B)
    end

    return ϕ, ψ, τ
end

##########################################################
# Parity-free two-step scheme via explicit S_m and U_m   #
##########################################################

# α,β for the odd step i = 2m-1
@inline function coeffs_odd(m, P::RecParams)
    p1 = P.p - one(P.p)
    α = ipow(2m,   p1) * (P.k2^P.r + P.k3^P.r)
    β = P.k1^P.r * ipow(2m-1, p1) + P.k2^P.r * ipow(2m, p1)
    return α, β
end

# α,β for the even step i = 2m
@inline function coeffs_even(m, P::RecParams)
    p1 = P.p - one(P.p)
    α = ipow(2m+1, p1) * (P.k1^P.r + P.k3^P.r)
    β = P.k2^P.r * ipow(2m,   p1) + P.k1^P.r * ipow(2m+1, p1)
    return α, β
end

"""
S_m(x) = T_{2m}( T_{2m-1}(x) ), the even-subsequence 2-step map.
Given x = τ_{2m-2}, returns τ_{2m}.
No branches (parity-free): uses the explicit α,β formulas for i=2m-1 and i=2m.
"""
function S_m(x, m::Integer, P::RecParams)
    αo, βo = coeffs_odd(m, P)                # i = 2m-1
    ψo, τo = onestep_from_alpha_beta(x, αo, βo)

    αe, βe = coeffs_even(m, P)               # i = 2m
    _,  τe = onestep_from_alpha_beta(τo, αe, βe)

    return τe
end

"""
U_m(y) = T_{2m+1}( T_{2m}(y) ), the odd-subsequence 2-step map.
Given y = τ_{2m-1}, returns τ_{2m+1}.
Also parity-free internally.
"""
function U_m(y, m::Integer, P::RecParams)
    αe, βe = coeffs_even(m, P)               # i = 2m
    ψe, τe = onestep_from_alpha_beta(y, αe, βe)

    # i = 2m+1 is odd; reuse coeffs_odd with (m+1) because 2(m+1)-1 = 2m+1
    αo, βo = coeffs_odd(m+1, P)              # i = 2m+1
    _,  τo = onestep_from_alpha_beta(τe, αo, βo)

    return τo
end

"""
Compute τ via the parity-free two-step compositions:
- Base cases: τ_0, τ_1 (with i=1 odd one-step)
- Then fill even indices via S_m and odd indices via U_m
- Apply the special final step at i=N exactly as in the original
Returns (tau, psi, phi) matching the original dynamics.
"""
function recurrence_twostep(N::Integer, P::RecParams)
    T = promote_type(typeof(P.k1),typeof(P.k2),typeof(P.k3),typeof(P.r),typeof(P.p))
    τ  = zeros(T, N+1)
    ψ  = zeros(T, N+1)
    ϕ  = zeros(T, N+1)

    # Common p-1
    p1 = P.p - one(T)

    # τ_0, ψ_0
    τ[1] = (P.k1^P.r + P.k3^P.r)
    ψ[1] = τ[1]

    # τ_1 via the i=1 (odd) one-step (no parity branch inside the onestep itself)
    if N >= 1
        ϕ[2] = τ[1]
        α1 = ipow(2, p1) * (P.k2^P.r + P.k3^P.r)               # i=1 -> (i+1)=2, odd case
        β1 = P.k1^P.r * ipow(1, p1) + P.k2^P.r * ipow(2, p1)
        ψ[2], τ[2] = onestep_from_alpha_beta(τ[1], α1, β1)     # stores ψ_1, τ_1
    end

    # Fill indices up to N-1 using S_m (even) and U_m (odd)
    # We already have τ_0 (index 1) and τ_1 (index 2).
    # For m=1, τ_2 = S_1(τ_0); τ_3 = U_1(τ_1); etc.
    if N >= 2
        for m in 1:div(N-1, 2, RoundDown) + 1
            even_idx = 2m       # i = 2m
            if even_idx <= N-1
                τ[even_idx+1] = S_m(τ[even_idx-1], m, P)  # τ_{2m} depends on τ_{2m-2}
                ϕ[even_idx+1] = τ[even_idx]               # fill φ_{2m} = τ_{2m-1} if you later need ψ explicitly
            end
            odd_idx = 2m + 1    # i = 2m+1
            if odd_idx <= N-1
                τ[odd_idx+1] = U_m(τ[odd_idx-1], m, P)    # τ_{2m+1} depends on τ_{2m-1}
                ϕ[odd_idx+1] = τ[odd_idx]                 # φ_{2m+1} = τ_{2m}
            end
        end
    end

    # Special final step at i = N
    if N >= 1
        ϕ[N+1] = τ[N]
        A = ipow(N+1, p1) * (P.k3^P.r)
        B = P.k1^P.r * ipow(N,   p1)
        ψ[N+1], τ[N+1] = onestep_from_alpha_beta(τ[N], A, B)
    end

    # If you need ψ,ϕ at intermediate i<N explicitly from τ, you can reconstruct via:
    # ψ_i = τ_i - τ_{i-1} and ϕ_i = τ_{i-1}.
    for i in 2:N
        ϕ[i] = τ[i-1]
        ψ[i] = τ[i] - τ[i-1]
    end

    return ϕ, ψ, τ
end




function simplified_recurrance(N, k1, k2, k3, p, r)

    # Storage
    tau = OffsetArray(zeros(N+1), 0:N)  
    psi = OffsetArray(zeros(N+1), 0:N)    
    phi = zeros(N)  

    # τ₀ = ψ₀ = k1^r + k2^r
    tau[0] = (k1^r + k3^r)
    psi[0] =  (k1^r + k3^r)



    # Main loop: i = 1..N-1
    for i in 1:N-1
        # (a) φ_i = τ_{i-1}
        phi[i] = tau[i-1]

        # (b) ψ_i = ...
        if isodd(i)
            psi[i] = 0.5 * ((i+1)^(p-1) * (k2^r + k3^r) + sqrt(((i+1)^(p-1) * (k2^r + k3^r))^2 + 4.0 * phi[i] * (k1^r * i^(p-1) + k2^r * (i+1)^(p-1))))
        else
            psi[i] = 0.5 * ((i+1)^(p-1) * (k1^r + k3^r) + sqrt(((i+1)^(p-1) * (k1^r + k3^r))^2 + 4.0 * phi[i] * (k2^r * i^(p-1) + k1^r * (i+1)^(p-1))))
        end
        

        # (c) τ_i = ψ_i + φ_i
        tau[i] = psi[i] + phi[i]
    end

    # Final step: i = N

        # φ_N = τ_{N-1}
        phi[N] = tau[N-1]

        # ψ_N with the special (k3^r-only) A term and B = k1 * N^(p-1)
        A = (N+1)^(p-1) * (k3^r)
        B =  k1^r * N^(p-1)
        psi[N] = 0.5 * (A + sqrt(A*A + 4.0 * phi[N] * B))

        # τ_N = ψ_N + φ_N
        tau[N] = psi[N] + phi[N]
  
    φ = [phi[i] for i = 1:N]
    ψ = [psi[i] for i = 0:N]
    τ = [tau[i] for i = 0:N]
    return φ, ψ, τ
end


###################
# Quick self-test #
###################

"""
Utility to compare the two methods numerically.
Returns the maximum absolute difference in τ.
"""
function compare_schemes(N::Integer, P::RecParams; atol=1e-10)
    τ_ref, ψ_ref, ϕ_ref = recurrence_original(N, P)
    τ_2st, ψ_2st, ϕ_2st = recurrence_twostep(N, P)
    maxerr = maximum(abs.(τ_ref .- τ_2st))
    return maxerr
end

###########
# Example #
###########

#Example usage:

N = 23
β = 1
global p = 0.4
R = 1


L_eps = δ -> L_smooth(δ, β, p)
k = [exp(p) * (-sqrt(2) * p^0.25 + sqrt(2)), max(0, (7 - p^(-0.25)) * (p^(1 / 3) - p^(1 / 2)) * p^p), max(0, 1 / 7 * (p^(0.04 * p) - p^2))]
δ_set = compute_δ_set(N, k, p)

γ = β*R^(1+p)/((N+1)^((1+p)/2))
ϕ = 1/L_eps(γ)
k1 = k[1]
k2 = k[2]
k3 = k[3]

phi, psi, tau = simplified_recurrance(N, k1, k2, k3, p, (1-p)/(1+p))
φ_1 = ϕ*phi
ψ_1 = OffsetArray(ϕ*psi, 0:N)
τ_1 = OffsetArray(ϕ*tau, 0:N)


P = RecParams(k1, k2, k3, (1-p)/(1+p), p)  # k1,k2,k3,r,p
phi_ref, psi_ref, tau_ref = recurrence_original(N, P)
φ_2 = ϕ*phi_ref[2:end]
ψ_2 = OffsetArray(ϕ*psi_ref, 0:N)
τ_2 = OffsetArray(ϕ*tau_ref, 0:N)

phi_2st, psi_2st, tau_2st = recurrence_twostep(N, P)
φ_3 = ϕ*phi_2st[2:end]
ψ_3 = OffsetArray(ϕ*psi_2st, 0:N)
τ_3 = OffsetArray(ϕ*tau_2st, 0:N)


##############################################
# Expanded-only two-step (no helper functions)
##############################################

"""
Compute (τ, ψ, φ) using ONLY the explicit expanded forms of S_m and U_m.
No calls to coeffs_* or onestep_* helpers; everything is inlined.
Matches the same special final step at i = N.
"""
function recurrence_twostep_expanded(N::Integer, P::RecParams)
    T = promote_type(typeof(P.k1),typeof(P.k2),typeof(P.k3),typeof(P.r),typeof(P.p))
    τ  = zeros(T, N+1)
    ψ  = zeros(T, N+1)
    ϕ  = zeros(T, N+1)

    p1 = P.p - one(T)

    # Base: τ_0 = ψ_0 = k1^r + k3^r
    τ[1] = (P.k1^P.r + P.k3^P.r)
    ψ[1] = τ[1]

    # τ_1 from i=1 (odd) — explicit α,β
    if N >= 1
        ϕ[2] = τ[1]
        α1 = (2.0)^(p1) * (P.k2^P.r + P.k3^P.r)
        β1 = P.k1^P.r * (1.0)^(p1) + P.k2^P.r * (2.0)^(p1)
        root1 = sqrt(α1*α1 + 4*β1*τ[1])
        ψ[2] = 0.5*(α1 + root1)
        τ[2] = τ[1] + ψ[2]
    end

    # Fill via explicit S_m and U_m
    # For m = 1,2,... we compute:
    #   τ_{2m}   = S_m(τ_{2m-2}) = apply odd i=2m-1 then even i=2m
    #   τ_{2m+1} = U_m(τ_{2m-1}) = apply even i=2m then odd  i=2m+1
    if N >= 2
        m = 1
        while true
            # even index i = 2m (compute τ_{2m} from τ_{2m-2})
            even_i = 2m
            if even_i <= N-1
                # Step i=2m-1 (odd): α_o, β_o
                αo = (float(2m))^p1 * (P.k2^P.r + P.k3^P.r)
                βo = P.k1^P.r * (float(2m-1))^p1 + P.k2^P.r * (float(2m))^p1
                # τ_{2m-1}
                root_o = sqrt(αo*αo + 4*βo*τ[even_i-1])         # τ_{2m-2} is τ[even_i-1]
                τ_2m_minus_1 = τ[even_i-1] + 0.5*(αo + root_o)

                # Step i=2m (even): α_e, β_e
                αe = (float(2m+1))^p1 * (P.k1^P.r + P.k3^P.r)
                βe = P.k2^P.r * (float(2m))^p1 + P.k1^P.r * (float(2m+1))^p1
                # τ_{2m}
                root_e = sqrt(αe*αe + 4*βe*τ_2m_minus_1)
                τ[even_i+1] = τ_2m_minus_1 + 0.5*(αe + root_e)
                ϕ[even_i+1] = τ[even_i]   # φ_{2m} = τ_{2m-1} (optional bookkeeping)
            end

            # odd index i = 2m+1 (compute τ_{2m+1} from τ_{2m-1})
            odd_i = 2m + 1
            if odd_i <= N-1
                # Step i=2m (even): α_e, β_e
                αe = (float(2m+1))^p1 * (P.k1^P.r + P.k3^P.r)
                βe = P.k2^P.r * (float(2m))^p1 + P.k1^P.r * (float(2m+1))^p1
                # τ_{2m}
                root_e = sqrt(αe*αe + 4*βe*τ[odd_i-1])          # τ_{2m-1} is τ[odd_i-1]
                τ_2m = τ[odd_i-1] + 0.5*(αe + root_e)

                # Step i=2m+1 (odd): α_o', β_o'
                αo′ = (float(2m+2))^p1 * (P.k2^P.r + P.k3^P.r)
                βo′ = P.k1^P.r * (float(2m+1))^p1 + P.k2^P.r * (float(2m+2))^p1
                # τ_{2m+1}
                root_o′ = sqrt(αo′*αo′ + 4*βo′*τ_2m)
                τ[odd_i+1] = τ_2m + 0.5*(αo′ + root_o′)
                ϕ[odd_i+1] = τ[odd_i]    # φ_{2m+1} = τ_{2m}
            end

            m += 1
            if (2m > N-1) && (2m+1 > N-1)
                break
            end
        end
    end

    # Special final step i = N (exactly as spec)
    if N >= 1
        ϕ[N+1] = τ[N]
        A = (float(N+1))^p1 * (P.k3^P.r)      # α_N special
        B = P.k1^P.r * (float(N))^p1          # β_N special
        rootN = sqrt(A*A + 4*B*τ[N])
        ψ[N+1] = 0.5*(A + rootN)
        τ[N+1] = τ[N] + ψ[N+1]
    end

    # Reconstruct ψ,ϕ for intermediate steps if desired
    for i in 2:N
        ϕ[i] = τ[i-1]
        ψ[i] = τ[i] - τ[i-1]
    end

    return ϕ, ψ, τ
end



phi_exp, psi_exp, tau_exp = recurrence_twostep_expanded(N, P)
φ_4 = ϕ*phi_exp[2:end]
ψ_4 = OffsetArray(ϕ*psi_exp, 0:N)
τ_4 = OffsetArray(ϕ*tau_exp, 0:N)
