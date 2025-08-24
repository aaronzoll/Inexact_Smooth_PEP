using JLD2
using DelimitedFiles


open("coeffs.csv", "w") do io
    X, k1, k2, k3 = jldopen("Holder PEP/coeffs_epsilons_guess_zoom.jld2", "r") do file
        X = file["X"]
        k1 = file["k1"]
        k2 = file["k2"]
        k3 = file["k3"]
        return (X, k1, k2, k3)
    end

    Y = hcat(k1, k2, k3)
    data = hcat(X, Y)

    writedlm(io, data, ',')

    X, k1, k2, k3 = jldopen("Holder PEP/coeffs_epsilons_guess_zoom_2.jld2", "r") do file
        X = file["X"]
        k1 = file["k1"]
        k2 = file["k2"]
        k3 = file["k3"]
        return (X, k1, k2, k3)
    end

    Y = hcat(k1, k2, k3)
    data = hcat(X, Y)

    writedlm(io, data, ',')

    X, k1, k2, k3 = jldopen("Holder PEP/coeffs_epsilons_guess_zoom_3.jld2", "r") do file
        X = file["X"]
        k1 = file["k1"]
        k2 = file["k2"]
        k3 = file["k3"]
        return (X, k1, k2, k3)
    end

    Y = hcat(k1, k2, k3)
    data = hcat(X, Y)

    writedlm(io, data, ',')

end
