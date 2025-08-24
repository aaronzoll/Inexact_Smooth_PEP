import numpy as np
from pysr import PySRRegressor
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("coeffs.csv", header=None)
X = df.iloc[:, 0].values  
k1 = df.iloc[:, 1].values  
k2 = df.iloc[:, 2].values  
k3 = df.iloc[:, 3].values 

X = X.reshape(-1, 1) 
k1 = k1.reshape(-1, 1)
k2 = k2.reshape(-1, 1)
k3 = k3.reshape(-1, 1)
#plt.plot(X, k1)
#plt.show()

model = PySRRegressor(
    maxsize=20,
    niterations=1000, 
    binary_operators=["+", "*", "-", "/",  "pow"],
    unary_operators=[
        "exp"
    ],
    constraints={"pow": (-1, 5)},
    nested_constraints={"exp": {"exp": 0, "pow": 0}, "pow": {"exp": 0, "pow" : 0}},
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    model_selection="best",  # Ensures final model is stored in memory
  
)
model.fit(X, k2)

