import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn")

with open("results/bo.json") as f:
    file = json.load(f)

frame = pd.DataFrame({i: np.maximum.accumulate(j, axis=1).mean(axis=0) for i, j in file.items()})
frame.plot()
plt.show()
print()
