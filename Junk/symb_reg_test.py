import numpy as np
import pandas as pd
from pysr import PySRRegressor
from matplotlib import pyplot as plt

# --- Load data ---
# If your Julia script saved headers (k1,k2,k3,p,value):
df = pd.read_csv("asymptotic_data_sigma.csv")   # <- use header row
# If you truly saved with no header, use header=None and rename:
# df = pd.read_csv("asymptotic_data.csv", header=None)
# df.columns = ["k1","k2","k3","p","value"]

X = df[["k1", "k2", "k3", "p", "r"]].to_numpy()
y = df["value"].to_numpy()

# --- Configure & fit PySR ---
model = PySRRegressor(
    # search budget
    niterations=500,
   
    # operators
    binary_operators=["+", "-", "*", "/", "^"],   # use "^" (PySR’s power), not "pow"
  #  unary_operators=["exp"],
  #  unary_operators=[ "exp" ], 
    constraints={"^": (-1, 11)}, 
    nested_constraints={"^": {"^" : 1}},
    # names
   
    # selection
    model_selection="best",   # keep the best-of-run model
    # optional complexity control
    maxsize=25,
    batching=True,
    batch_size=250,
    # loss (default MSE is fine; you can omit this)
    # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
)

model.fit(X, y,   variable_names=["k1", "k2", "k3", "p", "r"])



##### ---------------------------
'''

possible scaling by ((1-p)/(1+p) * 1/2)^((1-p)/(1+p)), and that shows up in \phi * N^(p+1)

'''


# --- Inspect results ---
print("\nBest model (PySR printed form):\n", model)
try:
    print("\nBest model (SymPy):\n", model.sympy())
except Exception:
    # Older/newer PySR versions sometimes differ; this is a fallback
    best = model.get_best()
    try:
        print("\nBest model (SymPy via get_best):\n", best.sympy())
    except Exception:
        print("\nRaw best entry:\n", best)

# --- Parity plot ---
# yhat = model.predict(X)
# plt.figure()
# plt.scatter(y, yhat, s=8, alpha=0.6)
# low, high = np.min([y.min(), yhat.min()]), np.max([y.max(), yhat.max()])
# plt.plot([low, high], [low, high], linewidth=1)
# plt.xlabel("True rate")
# plt.ylabel("Predicted rate")
# plt.title("PySR parity plot")
# plt.tight_layout()
# plt.show()
