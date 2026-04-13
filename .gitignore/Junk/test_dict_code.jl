results = Dict{Int, Dict{Int, Dict{String, Any}}}()



for n in 1:10
    results[n] = Dict{Int, Dict{String, Any}}()
    for m in 1:5
        # Example data generated for (n, m)
        min_F_p = rand(n)
        ε_set_p = [rand(m) for _ in 1:50]
        H_p = [rand(n, n) for _ in 1:50]

        results[n][m] = Dict(
            "F_values" => min_F_p,
            "ε_sets" => ε_set_p,
            "H_matrices" => H_p
        )
    end
end


