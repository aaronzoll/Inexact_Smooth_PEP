using DelimitedFiles, Plots, LaTeXStrings

"""
    graph_obj_vs_N(csv_path, q; save_path=nothing)

Load (N, obj) data from `csv_path`, then plot on shared N x-axis:
- obj from the CSV (PEP/computed values),
- theoretical bound: ((q+1)^((q-1)/(q+1)) / q^(q/(q+1))) * 1/(N+1)^((2-q)/(1+q)),
- theoretical bound: 2^(1+2*(1-q)/(1+q)) * 1/N^((2-q)/(1+q)).

Optional: set `save_path` (e.g. "obj_vs_N_plot.png") to save the figure.
"""
function graph_obj_vs_N(csv_path::String, q::Real; save_path::Union{String,Nothing}=nothing)
    # Load CSV: expect header row "N,obj" then numeric rows
    M = readdlm(csv_path, ',')
    N = Int.(M[2:end, 1])
    obj = Float64.(M[2:end, 2])

    # Theoretical curves (same N on x-axis)
    exponent = (2 - q) / (1 + q)
    coeff1 = (q + 1)^((q - 1) / (q + 1)) / (q^(q / (q + 1)))
    curve1 = coeff1 * 1.0 ./ (N .+ 1) .^ exponent
    coeff2 = 2^(3/(1+q))*q^(-q/(1+q))
    curve2 = coeff2 * 1.0 ./ (N .^ exponent)

    q = round(q,digits = 2)
    plt = plot(
        N,
        [obj curve1 curve2];
        label=["Constructive Method" "Asymptotic Theory" "Nesterov UFGM"],
        xlabel="N",
        ylabel=L"$f_N - f_\star$",
        legend=:topright,
        linewidth=2.5,
        yaxis=:log,
        title = L"Performance for $L(\delta) = 1/\delta^{%$(q)}$",
        dpi = 600)


    if q < 1e-6
        ogm_curve = [OGM_rates(1, 1, n)[end] for n in N]
        plot!(plt, N, ogm_curve; label="OGM", linewidth=2.5)
    end
    isnothing(save_path) || savefig(plt, save_path)
    return plt
end


function graph_constructive_coeffs_vs_q(; 
    path_N50::AbstractString = "Data and Plotting/constructive_coeffs_N_50.csv",
    path_N150::AbstractString = "Data and Plotting/constructive_coeffs_N_150.csv",
    path_N250::AbstractString = "Data and Plotting/constructive_coeffs_N_250.csv",
    save_path::Union{String,Nothing} = nothing)

# Load CSVs: header "q, coeff"
M50   = readdlm(path_N50,  ',')
M150  = readdlm(path_N150, ',')
M250  = readdlm(path_N250, ',')

q   = Float64.(M50[2:end, 1])
c50 = Float64.(M50[2:end, 2])
q150 = Float64.(M150[2:end, 1])
c150 = Float64.(M150[2:end, 2])
q250 = Float64.(M250[2:end, 1])
c250 = Float64.(M250[2:end, 2])

# Theoretical coefficient for each q > 0
coeff_theory = ((q .+ 1) .^ ((q .- 1) ./ (q .+ 1))) ./ (q .^ (q ./ (q .+ 1)))

plt = plot(
    q,
    c50;
    label = "N = 50",
    xlabel = L"$q$",
    ylabel = "coefficient",
    linewidth = 2.5,
    color =:dodgerblue,
    legend = :best,
    title = L"Asymptotic Coefficients for $L(\delta) = 1/{\delta^q}$",
    dpi = 600)

plot!(plt, q, c150; label = "N = 150", linewidth = 2.5, color =:blue)
plot!(plt, q, c250; label = "N = 250", linewidth = 2.5,
color =:navy)
plot!(plt, q, coeff_theory; label = "Asymptotic Theory", linestyle = :dot, linewidth = 2.5,
color =:red)

isnothing(save_path) || savefig(plt, save_path)
return plt
end

function OGM_rates(β, R, N)

    theta = 1
    rate = []
    for i in 0:N-1
        if i < N-1
            theta = (1 + sqrt(1+4*theta^2))/2
        else
            theta = (1 + sqrt(1+8*theta^2))/2
        end

        push!(rate, β*R^2/(2*theta^2))
    end
   
    return rate
end

# plt = graph_obj_vs_N("Data and Plotting/obj_vs_N_q0p25.csv", 0.25, save_path = "obj_q_0p25.png")
# plt = graph_obj_vs_N("Data and Plotting/obj_vs_N_q0p5.csv", 0.5, save_path = "obj_q_0p5.png")
# plt = graph_obj_vs_N("Data and Plotting/obj_vs_N_q1e-8.csv", 1e-8, save_path = "obj_q_1e-8.png")
# plt = graph_obj_vs_N("Data and Plotting/obj_vs_N_q1p0.csv", 1.0, save_path = "obj_q_1p0.png")

# plt = graph_constructive_coeffs_vs_q(save_path = "coeff_vs_q_highres.png")



plt = graph_obj_vs_N("Data and Plotting/obj_vs_N_q0p25.csv", 0.25)
