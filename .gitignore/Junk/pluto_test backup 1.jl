### A Pluto.jl notebook ###
# To run: ] add Pluto PlutoUI CSV DataFrames Plots
# then: using Pluto; Pluto.run()

using CSV, DataFrames, Plots, PlutoUI

# Load the data
df = CSV.read("asymptotic_data.csv", DataFrame)

# Widgets for parameter centers
@bind k2_fixed Slider(0:0.1:10, default=2.0, show_value=true)
@bind k3_fixed Slider(0:0.1:10, default=3.0, show_value=true)
@bind p_fixed  Slider(0.0:0.01:1.0, default=0.5, show_value=true)

# Widgets for tolerances
@bind tol_k2 Slider(0.0:0.05:1.0, default=0.1, show_value=true)
@bind tol_k3 Slider(0.0:0.05:1.0, default=0.1, show_value=true)
@bind tol_p  Slider(0.0:0.01:0.2, default=0.05, show_value=true)

# Filtered subset
subset = filter(row -> abs(row.k2 - k2_fixed) ≤ tol_k2 &&
                        abs(row.k3 - k3_fixed) ≤ tol_k3 &&
                        abs(row.p  - p_fixed)  ≤ tol_p,
                df)

# Plot
subset_sorted = sort(subset, :k1)
plt = plot(subset_sorted.k1, subset_sorted.value,
           marker=:o,
           xlabel="k1", ylabel="value",
           title="k2≈$k2_fixed±$tol_k2, k3≈$k3_fixed±$tol_k3, p≈$p_fixed±$tol_p")
plt