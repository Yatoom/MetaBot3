import json
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor, plot_importance
from scipy.stats import rankdata, kendalltau
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use("seaborn")
warnings.filterwarnings("ignore")

# Functions
def store_json(data, filename):
    with open(os.path.join(results_dir, filename), "w+") as f:
        json.dump(data, f, indent=4, sort_keys=True)

# Settings
flow_id = 6969
cache_dir = "cache"
results_dir = "results"
print(os.path.realpath(__file__))

# Get metafeatures
metafeatures = pd.read_csv(os.path.join(cache_dir, "metafeatures2.csv"), index_col=0)

# Load files
groups = pd.read_json(os.path.join(cache_dir, f"{flow_id}_groups.json"))[0].sort_index()
params = pd.read_json(os.path.join(cache_dir, f"{flow_id}_params.json")).sort_index()
metrics = pd.read_json(os.path.join(cache_dir, f"{flow_id}_metrics.json")).sort_index()
metas = metafeatures.loc[groups].reset_index(drop=True)

# Sorting
indices = groups.argsort()
groups = groups.iloc[indices].reset_index(drop=True)
params = params.iloc[indices].reset_index(drop=True)
metrics = metrics.iloc[indices].reset_index(drop=True)
metas = metas.iloc[indices].reset_index(drop=True)
unique_groups = np.unique(groups)

# Rescale kappa
metrics = metrics.astype(float)
metrics["kappa"] = metrics["kappa"] / 2 + 0.5

# Converting
params = pd.get_dummies(params)

# Get data
meta_X = pd.concat([params, metas], axis=1, sort=False).drop(["Dimensionality"], axis=1)
surr_X = params
y = np.array(metrics["predictive_accuracy"])

# Estimators
meta_estimator = LGBMRegressor(n_estimators=500, num_leaves=16, learning_rate=0.05, min_child_samples=1, verbose=-1)
surr_estimator = LGBMRegressor(n_estimators=100, num_leaves=8, objective="quantile", alpha=0.9, min_child_samples=1, min_data_in_bin=1, verbose=-1)

logo = LeaveOneGroupOut()

result = {
    "randomized 1": [],
    "randomized 2": [],
    "randomized 3": [],
}

for train_index, test_index in tqdm(logo.split(surr_X, y, groups)):

    if len(test_index) < 250 * 4:
        continue

    optimum = np.max(y[test_index])

    # Scale y
    y_converted = np.zeros_like(y)
    for g in unique_groups:
        indices = groups == g
        selection = y[indices]
        y_converted[indices] = StandardScaler().fit_transform(X=selection.reshape(-1, 1)).reshape(-1)

    # Completely random
    observed_ys = []
    for _ in range(10000):
        observed_ys.append(np.random.choice(y[test_index], 250, replace=False))
    observed_y = np.mean(np.maximum.accumulate(observed_ys, axis=1), axis=0)
    result["randomized 1"].append((np.array(observed_y) / optimum).tolist())

    observed_ys = []
    for _ in range(10000):
        chosen = np.random.choice(y[test_index], 500, replace=False)
        chosen = np.max(np.split(chosen, 2), axis=0)
        observed_ys.append(chosen)
    observed_y = np.mean(np.maximum.accumulate(observed_ys, axis=1), axis=0)
    result["randomized 2"].append((np.array(observed_y) / optimum).tolist())

    observed_ys = []
    for _ in range(10000):
        chosen = np.random.choice(y[test_index], 750, replace=False)
        chosen = np.max(np.split(chosen, 3), axis=0)
        observed_ys.append(chosen)
    observed_y = np.mean(np.maximum.accumulate(observed_ys, axis=1), axis=0)
    result["randomized 3"].append((np.array(observed_y) / optimum).tolist())

    # Train estimator
    # meta_estimator.fit(meta_X.iloc[train_index], y_converted[train_index])
    # meta_predictions = meta_estimator.predict(meta_X.iloc[test_index])



    # # Select randomly from top 75%, 50% and 25%
    # for fraction in fractions:
    #     number = int(len(meta_predictions) * 1/4)
    #     best_indices = np.argsort(meta_predictions)[-number:][::-1]
    #
    #     observed_ys = []
    #     for _ in range(100):
    #         np.random.shuffle(best_indices)
    #         observed_y = y[test_index][best_indices[:250]]
    #         observed_ys.append(observed_y)
    #     observed_y = np.mean(observed_ys, axis=0)
    #     result[f"select_from_top {fraction}"].append((np.array(observed_y) / optimum).tolist())
    #
    # # Select with weights
    # for power in powers:
    #     scaled = (meta_predictions - np.min(meta_predictions))/(np.max(meta_predictions) - np.min(meta_predictions))
    #     weight = scaled ** power
    #     weight /= weight.sum()
    #     observed_ys = []
    #     for _ in range(100):
    #         observed_y = np.random.choice(y[test_index], size=250, replace=False, p=weight)
    #         observed_ys.append(observed_y)
    #     observed_y = np.mean(observed_ys, axis=0)
    #     result[f"select_with_weights {power}"].append((np.array(observed_y) / optimum).tolist())

    store_json(result, "randomized-6969.json")