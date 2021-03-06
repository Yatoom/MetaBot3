import json
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor, plot_importance
from scipy.stats import rankdata, kendalltau, norm
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from smac.epm.rf_with_instances import RandomForestWithInstances
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
# metafeatures = metafeatures.dropna()
metafeatures = metafeatures[~metafeatures["DecisionStumpAUC"].isna()]
keep_groups = metafeatures.index.tolist()

# Load files
groups = pd.read_json(os.path.join(cache_dir, f"{flow_id}_groups.json"))[0].sort_index()
params = pd.read_json(os.path.join(cache_dir, f"{flow_id}_params.json")).sort_index()
metrics = pd.read_json(os.path.join(cache_dir, f"{flow_id}_metrics.json")).sort_index()

# Selection
indices = np.any([i == groups for i in metafeatures.index], axis=0)
groups = groups.iloc[indices].reset_index(drop=True)
params = params.iloc[indices].reset_index(drop=True)
metrics = metrics.iloc[indices].reset_index(drop=True)

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
# surr_estimator = LGBMRegressor(n_estimators=100, num_leaves=8, objective="quantile", alpha=0.9, min_child_samples=1, min_data_in_bin=1, verbose=-1)
bounds = [(np.min(surr_X[i]), np.max(surr_X[i])) for i in surr_X.columns]
bounds = np.array(bounds).astype(float)
types = np.zeros_like(bounds).astype(int)
surr_estimator = RandomForestWithInstances(bounds=bounds, types=types)


logo = LeaveOneGroupOut()

result = {}
for alpha in [0, 0.5, 1]:
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
            # mean = np.mean(y[train_index])
            # std = np.std(y[train_index])
            # meta_predictions = meta_predictions * std + mean
        else:
            meta_predictions = np.zeros_like(test_index)

        optimum = np.max(y[test_index])
        if alpha == 1:
            best_indices = np.argsort(meta_predictions)[-250:][::-1]
            observed_y = y[test_index][best_indices]
        else:
            # Do Blended BO for 250 iterations
            observed_X = []
            observed_y = []
            observed_i = []

            surpassed = None
            for iteration in range(0, 250):

                # We need to have observed at least 3 items for the model to be able to predict
                surr_predictions = np.zeros_like(test_index)
                if iteration > 2 and alpha < 1:
                    surr_estimator.train(np.array(observed_X).astype(float), np.array(observed_y))
                    mu, var = surr_estimator.predict(np.array(surr_X.iloc[test_index]).astype(float))
                    mu = mu.reshape(-1)
                    var = var.reshape(-1)
                    sigma = np.sqrt(var)
                    diff = mu - np.max(observed_y)
                    Z = diff / sigma
                    ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)
                    surr_predictions = ei

                    # surr_predictions = surr_estimator.predict(np.array(surr_X.iloc[test_index]).astype(float))
                # print(iteration, "\t", np.std(surr_predictions), "\t", np.std(meta_predictions))

                m_corr, m_pvalue = kendalltau(meta_predictions, y[test_index])
                s_corr, s_pvalue = kendalltau(surr_predictions, y[test_index])

                if s_corr > m_corr:
                    surpassed = iteration if surpassed is None else surpassed

                # alpha == 0: Only surrogate predictions
                # alpha == 1: Only meta-model predictions
                corrected_iteration = np.maximum(1, iteration - 2)
                # scaled_meta_predictions = MinMaxScaler().fit_transform(meta_predictions.reshape(-1, 1)).reshape(-1)
                # scaled_surr_predictions = MinMaxScaler().fit_transform(surr_predictions.reshape(-1, 1)).reshape(-1)
                scaled_meta_predictions = meta_predictions
                scaled_surr_predictions = StandardScaler().fit_transform(surr_predictions.reshape(-1, 1)).reshape(-1)
                scores = alpha**corrected_iteration * scaled_meta_predictions + (1 - alpha**corrected_iteration) * scaled_surr_predictions
                scores[observed_i] = -100

                index = np.argmax(scores)
                observed_X.append(surr_X.iloc[test_index].iloc[index])
                observed_y.append(y[test_index][index])
                observed_i.append(index)
            print(surpassed)
        result[alpha].append((np.array(observed_y) / optimum).tolist())
        store_json(result, "6969-lgbm.json")