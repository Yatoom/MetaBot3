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
flow_id = 8315
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

result = {}
for alpha in [0.5]:
    result[alpha] = []
    for train_index, test_index in tqdm(logo.split(surr_X, y, groups)):

        # Scale y
        y_converted = np.zeros_like(y)
        for g in unique_groups:
            indices = groups == g
            selection = y[indices]
            y_converted[indices] = StandardScaler().fit_transform(X=selection.reshape(-1, 1)).reshape(-1)

        # Train meta-estimator
        if alpha > 0:
            meta_estimator.fit(meta_X.iloc[train_index], y_converted[train_index])
            meta_predictions = meta_estimator.predict(meta_X.iloc[test_index])
        else:
            meta_predictions = np.zeros_like(test_index)
        mean = np.mean(y[train_index])
        std = np.std(y[train_index])
        meta_predictions = meta_predictions * std + mean

        # Do Blended BO for 250 iterations
        observed_X = []
        observed_y = []
        observed_i = []
        optimum = np.max(y[test_index])
        for iteration in range(1, 151):

            # We need to have observed at least 3 items for the model to be able to predict
            surr_predictions = np.zeros_like(test_index)
            if iteration > 3 and alpha < 1:
                surr_estimator.fit(np.array(observed_X), np.array(observed_y))
                surr_predictions = surr_estimator.predict(meta_X.iloc[test_index])

            # alpha == 0: Only surrogate predictions
            # alpha == 1: Only meta-model predictions
            corrected_iteration = np.maximum(1, iteration - 3)
            scores = alpha**corrected_iteration * meta_predictions + (1 - alpha**corrected_iteration) * surr_predictions
            scores[observed_i] = -10

            index = np.argmax(scores)
            observed_X.append(meta_X.iloc[index])
            observed_y.append(y[test_index][index])
            observed_i.append(index)
        result[alpha].append((np.array(observed_y) / optimum).tolist())
        store_json(result, "bo-0.5-double-corrected.json")