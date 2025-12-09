# Copyright contributors to the geobench project
# modified from geobench (https://github.com/ServiceNow/geo-bench/blob/main/geobench/plot_tools.py)

import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.constants import NORMALIZER_DIR, BIOMASSTERS_STD
import json
from scipy.stats import trim_mean, sem
from scipy.stats.mstats import trim

np.random.seed(100)

def biqm(scores):
    """Return a bootstram sample of iqm."""
    b_scores = np.random.choice(scores, size=len(scores), replace=True)
    return trim_mean(b_scores, proportiontocut=0.25, axis=None)


def trimmed_sem(scores):
    """Interquantile mean."""
    scores = trim(scores, limits=(0.25,0.25), relative=True)
    scores = scores.data[np.where(~scores.mask)] 
    return sem(scores)


def iqm(scores):
    """Interquantile mean."""
    return trim_mean(scores, proportiontocut=0.25, axis=None)


def bootstrap_iqm(
    df, group_keys=("model", "dataset"), metric="test_metric", repeat=100
):
    """Boostram of seeds for all model and all datasets to comput iqm score distribution."""
    df_list = []
    for i in range(repeat):
        series = df.groupby(list(group_keys))[metric].apply(biqm)
        df_list.append(series.to_frame().reset_index())

    return pd.concat(df_list)


def bootstrap_iqm_aggregate(df, metric="test_metric", repeat=100):
    """Stratified bootstrap (by dataset) of all seeds to compute iqm score distribution for each backbone."""
    group = df.groupby(["backbone", "dataset"])

    df_list = []
    for i in range(repeat):
        new_df = group.sample(frac=1, replace=True, random_state=100+i)
        series = new_df.groupby(["backbone"])[metric].apply(iqm)
        df_list.append(series.to_frame().reset_index())

    new_df = pd.concat(df_list)
    new_df.loc[:, "dataset"] = "aggregated"
    return new_df


def bootstrap_mean_aggregate(df, metric="test_metric", repeat=100):
    """Stratified bootstrap (by dataset) of all seeds to compute mean score distribution for each backbone."""
    group = df.groupby(["backbone", "dataset"])

    df_list = []
    for i in range(repeat):
        new_df = group.sample(frac=1, replace=True, random_state=100+i)
        series = new_df.groupby(["backbone"])[metric].apply(np.mean)
        df_list.append(series.to_frame().reset_index())

    new_df = pd.concat(df_list)
    new_df.loc[:, "dataset"] = "aggregated"
    return new_df



def scale_rmse(data: pd.DataFrame):
    def scale(row):
        if (row["dataset"] == "biomassters") and ("rmse" in row["Metric"].lower()):
            return 1 - (row["test metric"]*BIOMASSTERS_STD)  
        else:
            return row["test metric"]
    data["test metric"] = data.apply(lambda row: scale(row), axis=1)
    return data


def average_seeds(df, group_keys=("model", "dataset"), metric="test metric"):
    """Average seeds for all model and all datasets."""
    df_avg = df.groupby(list(group_keys))[metric].mean()
    df_avg = df_avg.unstack(level="dataset")

    df_avg = df_avg.round(3)
    return df_avg


class Normalizer:
    """Class used to normalize results beween min and max for each dataset."""

    def __init__(self, range_dict):
        """Initialize a new instance of Normalizer class."""
        self.range_dict = range_dict

    def __call__(self, ds_name, values, scale_only=False):
        """Call the Normalizer class."""
        mn, mx = self.range_dict[ds_name]
        range = mx - mn
        if scale_only:
            return values / range
        else:
            return (values - mn) / range

    def from_row(self, row, scale_only=False):
        """Normalize from row."""
        return [self(ds_name, val, scale_only=scale_only) for ds_name, val in row.items()]

    def normalize_data_frame(self, df, metric):
        """Normalize the entire dataframe."""
        new_metric = f"normalized {metric}"
        df[new_metric] = df.apply(lambda row: self.__call__(row["dataset"], row[metric]), axis=1)
        return new_metric

    def save(self, benchmark_name):
        """Save normalizer to json file."""

        if not os.path.exists(f"{NORMALIZER_DIR}/{benchmark_name}/"):
            print("making directory")
            os.makedirs(f"{NORMALIZER_DIR}/{benchmark_name}/")
        with open(f"{NORMALIZER_DIR}/{benchmark_name}/normalizer.json", "w") as f:
            json.dump(self.range_dict, f, indent=2)


def load_normalizer(benchmark_name):
    """Load normalizer from json file."""
    with open(f"{NORMALIZER_DIR}/{benchmark_name}/normalizer.json", "r") as f:
        range_dict = json.load(f)
    return Normalizer(range_dict)


def make_normalizer(data_frame, metrics=("test metric",), benchmark_name="leaderboard_combined"):
    """Extract min and max from data_frame to build Normalizer object for all datasets."""
    datasets = data_frame["dataset"].unique()
    range_dict = {}

    for dataset in datasets:
        sub_df = data_frame[data_frame["dataset"] == dataset]
        data = []
        for metric in metrics:
            data.append(sub_df[metric].to_numpy())
        range_dict[dataset] = (np.min(data), np.max(data))

    normalizer = Normalizer(range_dict)

    if benchmark_name:
        normalizer.save(benchmark_name)

    return normalizer


