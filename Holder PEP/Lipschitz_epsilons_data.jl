using Plots, JLD2, CurveFit, Optim

function objective(p, x, y)
    a, b, c = p
    if any(x .+ b .< 0)
        return Inf  # Penalize invalid region
    end
    model_y = @. a * (x+b)^(-c)
    return sum((y .- model_y) .^ 2)
end

function run_lst_sqr(x_data, y_data)
    ε = 1e-6
    lower = [0, -minimum(x_data) + ε, 0]
    upper = [Inf, Inf, 3]
    p0 = [1.0, 1.0, 0.5]  # Initial guess


    iter_print_freq = 500

    function my_callback(state)

        if state.iteration % iter_print_freq == 0 && state.iteration > 1
            println("Iter $(state.iteration): f = $(state.value)")
        end

        return false
    end

    options = Optim.Options(
        iterations=5000,
        f_tol=1e-10,
        x_tol=1e-10,
        time_limit=60,
        show_trace=false,
        callback=my_callback
    )
    result = optimize(p -> objective(p, x_data, y_data),
        lower, upper,
        p0,
        Fminbox(), options)

    a, b, c = Optim.minimizer(result)

    return a, b, c

end


@load "Lipschitz_epsilons_datasave.jld2" results

odd_a_vec = zeros(101)
odd_b_vec = zeros(101)
even_a_vec = zeros(101)
even_b_vec = zeros(101)
N_range = []

for N in keys(results)
    push!(N_range, N)
    odd_a_vec[N] = results[N][1]
    odd_b_vec[N] = results[N][2]
    even_a_vec[N] = results[N][3]
    even_b_vec[N] = results[N][4]
end



odds = 3:2:55
evens = 5:2:55


a, b, c = run_lst_sqr(odds, odd_a_vec[odds])

a1, b1 = linear_fit(evens, even_b_vec[evens])
#scatter(odds, odd_a_vec[odds], labels="coeff for odd ε")
#plot!(odds, @. a * (odds + b)^-c)
#scatter!(odds, odd_b_vec[odds], labels = "exponent for odd ε")
scatter(evens, even_a_vec[evens], labels = "intercept for even ε")
#scatter(evens, even_b_vec[evens], labels = "slope for even ε")