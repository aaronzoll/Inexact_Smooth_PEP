using CSV, DataFrames, GLMakie

# --- Load data ---
df = CSV.read("asymptotic_data_bigger.csv", DataFrame)  # expects columns: k1,k2,k3,p,value

# --- Figure & axes ---
fig = Figure(size = (900, 600))
ax  = Axis(fig[1, 1], xlabel="k1", ylabel="value", title="Slice: value vs k1")

# --- Controls (sliders) ---
# Centers
k2_slider = Slider(fig[2, 1], range=0.0:0.1:1.0, startvalue=0.5)
k3_slider = Slider(fig[3, 1], range=0.0:0.1:1.0, startvalue=0.3)
p_slider  = Slider(fig[4, 1], range=0.00:0.01:1.00, startvalue=0.50)

# Tolerances
tol_k2_slider = Slider(fig[2, 2], range=0.0:0.05:1.0, startvalue=0.10)
tol_k3_slider = Slider(fig[3, 2], range=0.0:0.05:1.0, startvalue=0.10)
tol_p_slider  = Slider(fig[4, 2], range=0.0:0.01:0.2, startvalue=0.05)

# Labels for readability
Label(fig[2, 1, Top()], "k2 center");        Label(fig[2, 2, Top()], "k2 tol")
Label(fig[3, 1, Top()], "k3 center");        Label(fig[3, 2, Top()], "k3 tol")
Label(fig[4, 1, Top()], "p center");         Label(fig[4, 2, Top()], "p tol")

# --- Plot objects backed by Observables ---
xobs = Observable(Float64[])
yobs = Observable(Float64[])
lines!(ax, xobs, yobs)

# --- Update logic: filter by ranges around (k2,k3,p) with given tolerances ---
function update!(args...)
    k2    = k2_slider.value[]
    k3    = k3_slider.value[]
    p     = p_slider.value[]
    tol2  = tol_k2_slider.value[]
    tol3  = tol_k3_slider.value[]
    tolp  = tol_p_slider.value[]

    mask = (abs.(df.k2 .- k2) .<= tol2) .&
           (abs.(df.k3 .- k3) .<= tol3) .&
           (abs.(df.p  .- p ) .<= tolp)

    sub = df[mask, :]
    if nrow(sub) == 0
        xobs[] = Float64[]
        yobs[] = Float64[]
        return
    end

    sort!(sub, :k1)
    xobs[] = collect(sub.k1)
    yobs[] = collect(sub.value)
end
# Trigger updates on any slider movement
onany(update!, k2_slider.value, k3_slider.value, p_slider.value,
                 tol_k2_slider.value, tol_k3_slider.value, tol_p_slider.value)

# Initial draw
update!()

GLMakie.activate!()
display(fig)
# Prevent Julia from immediately exiting if run as a script:
