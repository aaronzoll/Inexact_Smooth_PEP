using OffsetArrays, LinearAlgebra

function verify_FSFOM(N, λ_matrix, t_matrix, d=5)
    H_abs, H_inc = build_FSFOM_matrices(N, λ_matrix, t_matrix)

    # --- Simulate the original loop to get reference X3 ---
    x0 = randn(d)
    G  = OffsetArray(randn(d, N+1), 1:d, 0:N)
    X  = OffsetArray(zeros(d, N+1), 1:d, 0:N)
    X[:, 0] .= x0

    X3 = OffsetArray(zeros(d, N+1), 1:d, 0:N)
    X3[:, 0] .= x0
    z  = OffsetArray(zeros(d, N+1), 1:d, 0:N)
    z[:, 1] = X[:, 0] - λ_matrix[-1, 0] * G[:, 0]

    for j = 1:N
        num   = sum(λ_matrix[i,j] .* X[:,i] .- t_matrix[i,j] .* G[:,i] for i = 0:j-1) +
                λ_matrix[-1,j] .* z[:,j]
        denom = sum(λ_matrix[i, j] for i = 0:j-1) + λ_matrix[-1, j]
        X3[:, j] = num / denom
        X[:, j]  = X3[:, j]          # feed forward for next iteration
        if j < N
            z[:, j+1] = z[:, j] .- λ_matrix[-1, j] .* G[:, j]
        end
    end

    # --- Reconstruct using H_abs: x_k = x0 - Σ_i H_abs[k,i]*g_i ---
    X_abs = OffsetArray(zeros(d, N+1), 1:d, 0:N)
    for k = 0:N
        X_abs[:, k] = x0 - sum(H_abs[k, i] .* G[:, i] for i = 0:N)
    end

    # --- Reconstruct using H_inc: x_k = x_{k-1} - Σ_i H_inc[k,i]*g_i ---
    X_inc = OffsetArray(zeros(d, N+1), 1:d, 0:N)
    X_inc[:, 0] .= x0
    for k = 1:N
        X_inc[:, k] = X_inc[:, k-1] - sum(H_inc[k, i] .* G[:, i] for i = 0:N)
    end

    err_abs = maximum(norm(X_abs[:, k] - X3[:, k]) for k = 0:N)
    err_inc = maximum(norm(X_inc[:, k] - X3[:, k]) for k = 0:N)

    println("Max error (absolute formulation):    $err_abs")
    println("Max error (incremental formulation): $err_inc")
    return H_abs, H_inc
end

# --- Example: random positive λ and t matrices ---
N = 4
λ_matrix = OffsetArray(rand(N+2, N+1) .+ 0.1, -1:N, 0:N)
t_matrix  = OffsetArray(rand(N+1, N+1) .+ 0.1, 0:N,  0:N)

H_abs, H_inc = verify_FSFOM(N, λ_matrix, t_matrix)