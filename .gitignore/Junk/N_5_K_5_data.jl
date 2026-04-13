using JLD2, Plots, OffsetArrays

@load "N_2_M_5_data.jld2" results_3

N = length(collect(keys(results_3)))
M = length(collect(keys(results_3[1])))
p_range = sort(collect(keys(results_3[1][1]["ε_sets"])))
P_cnt = length(p_range)


function get_vector(N, H)
    vec_lower = zeros(div(N * (N + 1), 2))
    cnt = 0
    for i in 1:N
        for j in 1:i
            cnt = cnt + 1
            vec_lower[cnt] = H[i, j]
        end
    end
    return vec_lower
end




function plot_graphs(n, m, display_type, data_type)


    if data_type == "F_values"
        data = zeros(P_cnt,1)
        for i in 1:P_cnt
            data[i] = results[n][m]["F_values"][p_range[i]]
        end

    elseif data_type == "ε_sets"
        data = zeros(m, P_cnt)
        for i in 1:P_cnt
            data[:, i] = sort(collect(results[n][m]["ε_sets"][p_range[i]]))

        end

    elseif data_type == "H_matrices"
        data = zeros(div(n * (n + 1), 2), P_cnt)
        for i in 1:P_cnt
            H = collect(results[n][m]["H_matrices"][p_range[i]])
            data[:, i] = get_vector(n, H)

        end

    end
    if display_type == "same"
        plot()
        if data_type == "F_values"
            plot!(p_range, data, ylims = (0,2))

        elseif data_type == "ε_sets"
            for i = 1:m
                plot!(p_range, data[i, :], label="$i")
            end
            display(current())
        elseif data_type == "H_matrices"
            for k = 1:div(n * (n + 1), 2)
                # TODO fix indexing of H is label
                plot!(p_range, data[k, :], label="H_$k")
            end
            display(current())

        end




    elseif display_type == "different"

        if data_type == "F_values"
            plot(p_range, data, ylims = (0,0.2))

        elseif data_type == "ε_sets"
            plots = []
            for i = 1:m
                push!(plots, plot(p_range, data[i, :], title="$i", ylims = (0,1)))
            end
            plot(plots..., layout=(m, 1),  size=(1000, 1200))

        elseif data_type == "H_matrices"
            plots = []
            for k = 1:div(n * (n + 1), 2)
                # TODO fix indexing of H is label
                push!(plots, plot(p_range, data[k, :], title="H_$k"))
            end
            if mod(n, 2) == 0
                plot(plots..., layout=(div(n, 2), n + 1))
            else
                plot(plots..., layout=(div(n + 1, 2), n))
            end
        end




    end
end


#plot_graphs(1,5,"different","ε_sets")
data1 = zeros(P_cnt,1)
data2 = zeros(P_cnt,1)

for i in 1:P_cnt
    data1[i] = results_3[2][4]["F_values"][p_range[i]]
    data2[i] = results_3[2][5]["F_values"][p_range[i]]
end

plot(p_range, data1-data2, ylims = (-0.001,.001))
