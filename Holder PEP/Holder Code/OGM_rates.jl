function OGM_rates(L, R, N)

    theta = 1
    rate = []
    for i in 0:N-1
        if i < N-1
            theta = (1 + sqrt(1+4*theta^2))/2
        else
            theta = (1 + sqrt(1+8*theta^2))/2
        end

        push!(rate, L*R^2/(2*theta^2))
    end
   
    return rate
end

L = 1
R = 1
N = 4

rate = OGM_rates(L,R,N)