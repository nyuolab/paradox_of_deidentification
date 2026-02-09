"""
subsample huggingface dataset
"""

import logging
from collections import Counter

import numpy as np

from src.constants import COL_TO_TASK, MIN_YEAR


def map_logging_level(level: str) -> int:
    if level == "debug":
        return logging.DEBUG
    elif level == "info":
        return logging.INFO
    elif level == "warning":
        return logging.WARNING
    elif level == "error":
        return logging.ERROR


def setup_logging(level: str):
    logging.basicConfig(level=map_logging_level(level))


class Subsampler:
    def __init__(self, seed, data) -> None:
        self.seed = seed
        self.data = data
        self.total = len(data)
        self.total_indices = np.arange(self.total)
        self.rng = np.random.default_rng(seed)

    def subsample(self, n_samples):
        print(f"subsampling {n_samples} data from {self.total} samples")
        # reference: https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
        indices = self.rng.choice(
            self.total_indices, size=n_samples, replace=True
        )  # use bootstrap for consistency
        samples = self.data.select(indices)
        return samples


PRED_COLS = [
    "sex",
    "dyear",
    "postal_code_borough",
    "dmonth",
    "income_token",
    "payorfinancialclass",
]


def get_model_path_dict(finetune_size: int) -> dict:
    model_path_dict = {}
    for col in PRED_COLS:
        task_name = COL_TO_TASK[col]
        model_path_dict[col] = f"./models/{task_name}_deid_{finetune_size}"
    return model_path_dict


def map_gender(x):
    # merge unspecified with female because this is how we preprocessed the data
    if x["sex"] == "Male":
        res = "Male"
    else:
        res = "Female"
    x["sex"] = res
    return x


def map_payor_class(x):
    fc = x["payorfinancialclass"].strip().lower()
    if "medicaid" in fc or "medicare" in fc:
        res = "gov"
    else:
        res = "non_gov"
    x["payorfinancialclass"] = res
    return x


def postprocess(pred, col):
    # post process prediction for each column
    if col == "sex":
        pred = "Female" if pred == 1 else "Male"
    if col == "dyear":
        pred += MIN_YEAR
    if col == "dmonth":
        pred += 1  # normal month is not zero-indexed. It is one-indexed!
    if col == "income_token":
        b_dict = {0: "[POOR]", 1: "[RICH]"}
        pred = b_dict[pred]
    if col == "payorfinancialclass":
        b_dict = {0: "non_gov", 1: "gov"}
        pred = b_dict[pred]
    elif col == "postal_code_borough":
        b_dict = {
            0: "Manhattan",
            1: "Brooklyn",
            2: "Bronx",
            3: "Queens",
            4: "Staten Island",
            5: "Others",
        }
        pred = b_dict[pred]
    return pred


def get_majority_dict(train_data, check_cols, top_k_map):
    if "payorfinancialclass" in check_cols:
        train_data = train_data.map(map_payor_class)
    majority_dict = {}
    for col in check_cols:
        top_k = top_k_map[col]
        cnt = Counter(train_data[col])
        res = cnt.most_common(top_k)
        majority_values = [top_k_value for top_k_value, _ in res]
        majority_dict[col] = majority_values
    return majority_dict
