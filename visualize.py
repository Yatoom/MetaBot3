import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn")

with open("results/bo-250-with-fixes.json") as f:
    file = json.load(f)

# Remove datasets with less than 250 runs
indices = [index for index, values in enumerate(file["1"]) if len(values) == 250]
file = {i: np.array(j)[indices].tolist() for i, j in file.items()}

# Create frame
accumulated_max = {i: np.maximum.accumulate(j, axis=1).mean(axis=0) for i, j in file.items()}
accumulated_std = {i: np.maximum.accumulate(j, axis=1).std(axis=0) for i, j in file.items()}
frame = pd.DataFrame(accumulated_max)
# frame2 = pd.DataFrame(accumulated_std)
# frame = frame.drop(["0"], axis=1)
frame.plot()
# frame2.plot()
plt.ylim(0.99, 1.001)
plt.show()
print()
