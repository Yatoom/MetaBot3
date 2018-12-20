import json

import matplotlib.cm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn")
order = ['GBQR', 'GBQR bad-started', 'GBQR warm-started', 'Meta-learner', 'RFR', 'RFR bad-started', 'RFR warm-started', "GBQR + Delta"]
linestyle = ["--", "--", "--", "--", "--", "--", "--", "-"]

with open("results/6969-rfr.json") as f:
    file = json.load(f)

def to_frame(filio):
    filio = {i: np.maximum.accumulate(j, axis=1) for i, j in filio.items()}
    concatenated = pd.concat([pd.DataFrame(filio[i]) for i in file], axis=0)
    estimators = np.array([[i] * len(j) for i, j in file.items()]).reshape(-1)

    ranked = pd.DataFrame([concatenated[i].groupby(level=0).rank().tolist() for i in sorted(concatenated.columns)])
    ranked = ranked.T
    ranked["estimator"] = estimators
    # fig, ax = plt.subplots()
    mean = ranked.groupby("estimator").mean().T.rolling(10).mean()
    # std = ranked.groupby("estimator").var().T.rolling(10).mean()
    mean = mean.reindex(order, axis=1)
    # mean.reindex(order, axis=1).plot(ax=ax, cmap="tab10", linestyle="--")
    # linewidths = [2, 1, 4]
    cmap = matplotlib.cm.get_cmap('tab10')
    colors = list(cmap.colors)
    del colors[3]
    del colors[5]
    fig, ax = plt.subplots()
    for col, style, color in zip(mean.columns, linestyle, colors):
        mean[col].plot(style=style, ax=ax, color=color)
    l = ax.legend()
    l.set_title('')
    # for c in mean.columns:
    #     ax.fill_between(np.arange(0, 250, 1), mean[c]-std[c], mean[c]+std[c], alpha=0.1)
    plt.xlabel("Iterations")
    plt.ylabel("Average rank per dataset")
    plt.show()


    # molten = pd.DataFrame(filio).melt()
    # frame = pd.DataFrame(np.array(molten["value"].tolist()))
    # frame["estimator"] = molten["variable"]
    return ranked


# Create frame
frame = to_frame(file)
exit()
accumulated_max = {i: (1 - np.maximum.accumulate(j, axis=1)).mean(axis=0) for i, j in file.items()}
accumulated_std = {i: (1 - np.maximum.accumulate(j, axis=1)).var(axis=0) for i, j in file.items()}
means = pd.DataFrame(accumulated_max).rolling(10).mean() # [["surrogate", "meta-learner", "5-seeded"]]
errors = pd.DataFrame(accumulated_std).rolling(10).mean()

fig, ax = plt.subplots()
means.reindex(order, axis=1).plot(ax=ax, cmap="tab10")
ax.set_yscale('log')
# ax.set_yscale("logit")
# ax.plot(means)
for column in means.columns:
    upper = np.array(means[column] - errors[column]).reshape(-1)
    lower = np.array(means[column] + errors[column]).reshape(-1)
    ax.fill_between(np.arange(0, 250, 1), upper, lower, alpha=0.1)

# frame = frame.drop(["0"], axis=1)
# frame.plot()
# frame2.plot()
# plt.ylim(0.990, 1.005)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
print()
