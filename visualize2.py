import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn")

with open("results/bo-6969.json") as f:
    file = json.load(f)

# Remove datasets with less than 250 runs
indices = [index for index, values in enumerate(file["1"]) if len(values) >= 250]
file = {i: np.array(j)[indices].tolist() for i, j in file.items()}
selection = ["seeded-1", "seeded-3", "0", "1"]
# Create frame
accumulated_max = {i: np.maximum.accumulate(j, axis=1).mean(axis=0) for i, j in file.items() if np.max(j) == 1}
accumulated_std = {i: np.maximum.accumulate(j, axis=1).std(axis=0) for i, j in file.items()}
means = pd.DataFrame(accumulated_max)[selection].rolling(3).mean()
errors = pd.DataFrame(accumulated_std)[selection]

fig, ax = plt.subplots()
means.plot(ax=ax)
# ax.set_yscale('log')
# ax.set_yscale("logit")
# ax.plot(means)
for column in means.columns:
    upper = np.array(means[column] - errors[column]).reshape(-1)
    lower = np.array(means[column] + errors[column]).reshape(-1)
    ax.fill_between(np.arange(0, 250, 1), upper, lower, alpha=0.2)

# frame = frame.drop(["0"], axis=1)
# frame.plot()
# frame2.plot()
plt.ylim(0.98, 1.005)
plt.show()
print()
