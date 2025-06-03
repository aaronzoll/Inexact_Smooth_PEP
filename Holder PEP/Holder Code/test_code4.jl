using Revise, Optim, JLD2

include("BnB_PEP_Inexact_Smooth.jl")


L, R = 1.0, 1.0

trials = 3
p_cnt = 11
N_cnt = 6

results = gen_data(L, R, trials, p_cnt, N_cnt, 3)

@save "data_N_6" results