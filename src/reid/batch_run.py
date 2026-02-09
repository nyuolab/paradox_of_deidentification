"""
run all combinations of top k for each column in N_CLASSES
and log the results in wandb

Usage Example:
python -m src.reid.batch_run --start 0 --end 5759 --log_level debug
"""

import argparse
import logging
import subprocess
from functools import reduce

import pandas as pd
from pandas.io.json._normalize import nested_to_record
from tqdm import tqdm

import wandb
from src.constants import N_CLASSES
from src.util import setup_logging


def get_all_desired_results(n_classes):
    product_elements = []
    for key, val in n_classes.items():
        product_elements.append(pd.DataFrame({key: [x for x in range(val)]}))
    cartesian_product = reduce(
        lambda left, right: pd.merge(left, right, how="cross"), product_elements
    )
    return cartesian_product


def get_remaining_experiments(wandb_df, desired_df, n_classes):
    if len(wandb_df) == 0:
        return desired_df
    rename_dict = {}
    keep_cols = []
    for col in n_classes.keys():
        rename_dict[f"top_k_map/{col}"] = col
        keep_cols.append(col)
    consistent_df = wandb_df.rename(columns=rename_dict)[keep_cols]
    merged = pd.merge(desired_df, consistent_df, how="left", indicator=True)
    result = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return result


def get_all_run_result(run_name, api, baseline=True):
    runs = api.runs(run_name)
    summary_list = []
    logging.debug(runs)
    config_list = []
    try:
        for wandb_run in runs:
            conf = nested_to_record(wandb_run.config, sep="/")
            config = {k: v for k, v in conf.items() if not k.startswith("_")}
            if "baseline" in config and config["baseline"] == baseline:
                config_list.append(config)
                summary_list.append(wandb_run.summary._json_dict)
        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        all_df = pd.concat([config_df, summary_df], axis=1)
        return all_df
    except ValueError:  # if project does not exists, return empty dataframe
        return pd.DataFrame({})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reid experiments")
    parser.add_argument(
        "--baseline", action="store_true", help="Run baseline experiments"
    )
    parser.add_argument("--log_level", type=str, help="Log level")
    parser.add_argument("--start", type=int, help="Start index for experiments")
    parser.add_argument("--end", type=int, help="End index for experiments")
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Wandb project name",
        default="dummy_reid",
    )
    parser.add_argument(
        "--execute_program", action="store_true", help="Execute the program"
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    logging.info(f"Running experiments from index {args.start} to {args.end}")
    # total 5760 experiments to run
    want_df = get_all_desired_results(N_CLASSES)[args.start : args.end]
    api = wandb.Api()
    wandb_df = get_all_run_result(args.wandb_project, api, baseline=args.baseline)
    remaining = get_remaining_experiments(wandb_df, want_df, N_CLASSES)
    logging.info(f"there are {len(remaining)} experiments to run: {remaining}")
    for idx, row in tqdm(remaining.iterrows()):
        check_cols = [col for col in N_CLASSES.keys() if row[col] > 0]
        prefix = f'python -m src.reid.topk_match +project_name={args.wandb_project} +baseline={args.baseline} "check_cols={check_cols}" '
        # only include nonzero cols to avoid sabotaging the accuracy
        top_k_str = [
            f"top_k_map.{col}={row[col]}" for col in N_CLASSES.keys() if row[col] > 0
        ]
        if len(top_k_str) == 0:
            continue  # do not run if every k is zero, that means 0 accuracy.
        run_str = prefix + " ".join(top_k_str)
        logging.info(run_str)
        if args.execute_program:
            result = subprocess.run(
                run_str, shell=True, check=True, text=True, capture_output=True
            )
