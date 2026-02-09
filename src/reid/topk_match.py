# individual re-id experiment
# usage example:
# python -m src.reid.topk_match +project_name=dummy_reid +baseline=True top_k_map.sex=1 top_k_map.dyear=1 "check_cols=['sex','dyear']"
# python -m src.reid.topk_match +project_name=dummy_reid +baseline=True top_k_map.payorfinancialclass=1  "check_cols=['payorfinancialclass']"
import logging
import os
from datetime import datetime

import hydra
import numpy as np
import pandas as pd
from datasets import disable_caching, load_from_disk
from omegaconf import OmegaConf

import wandb
from src.constants import TOP_K_MATCH_PROJECT_NAME
from src.util import get_majority_dict, postprocess, setup_logging


def map_topk_pred(example_batch, check_cols, top_k_map):
    res = {}
    for col in check_cols:
        prob = np.array(example_batch[f"{col}_prob"])
        top_k = top_k_map[col]
        # size: N * top_k
        preds = list(np.argsort(-prob)[:, :top_k])
        res[f"{col}_top_{top_k}"] = preds
    return res


def map_topk_index_to_label(example, check_cols, top_k_map):
    res = {}
    for col in check_cols:
        top_k = top_k_map[col]
        pred_idxs = example[f"{col}_top_{top_k}"]
        pred_labels = [postprocess(pred_idx, col) for pred_idx in pred_idxs]
        res[f"{col}_top_{top_k}_labels"] = pred_labels
    return res


def map_baseline(example_batch, check_cols, top_k_map, majority_dict):
    res = {}
    batch_size = len(example_batch["patientkey"])
    for col in check_cols:
        top_k = top_k_map[col]
        res[f"{col}_top_{top_k}_labels"] = [
            majority_dict[col] for _ in range(batch_size)
        ]
    return res


def map_match(example, check_cols, top_k_map, reference_database):
    match_idx = []
    patient_id = example["patientkey"]
    for col in check_cols:
        top_k = top_k_map[col]
        preds = example[f"{col}_top_{top_k}_labels"]
        match_idx.append(reference_database[col].isin(preds))
    combined_idx = [all(group) for group in zip(*match_idx)]
    subset = reference_database[combined_idx]
    n_hit = (subset.patientkey == patient_id).sum()
    subset_size = len(np.unique(subset.patientkey))
    return {"n_hit": {n_hit}, "subset_size": {subset_size}}


def map_match_baseline(example, subset):
    patient_id = example["patientkey"]
    # in the matched subset, is the gt patient in it?
    n_hit = (subset.patientkey == patient_id).sum()
    # how large is the matched subset?
    subset_size = len(np.unique(subset.patientkey))  # keep unique patients
    return {"n_hit": {n_hit}, "subset_size": {subset_size}}


def map_payor_class(x):
    fc = x["payorfinancialclass"].strip().lower()
    if "medicaid" in fc or "medicare" in fc:
        res = "gov"
    else:
        res = "non_gov"
    x["payorfinancialclass"] = res
    return x


@hydra.main(
    version_base=None, config_path="../../data/eval_configs", config_name="top_k"
)
def main(cfg):
    setup_logging(cfg.logging_level)
    inferenced_data = load_from_disk(cfg.cached_inference_path)
    baseline = cfg.get("baseline", False)  # predicting majority?
    # load data
    test_data = load_from_disk(cfg.test_data_path)
    check_cols = cfg.check_cols
    if "payorfinancialclass" in check_cols:
        test_data = test_data.map(map_payor_class)
    save = cfg.get("save", False)
    test_data_df = test_data.to_pandas()
    assert len(check_cols) > 0

    top_k_map = cfg.top_k_map
    num_worker = cfg.num_worker
    disable_caching()
    if not baseline:
        logging.debug(f"getting topk index....")
        topk_idx_data = inferenced_data.map(
            lambda x: map_topk_pred(x, check_cols, top_k_map),
            batched=True,
            num_proc=num_worker,
        )
        logging.debug(f"converting indices to labels....")
        topk_label_data = topk_idx_data.map(
            lambda x: map_topk_index_to_label(x, check_cols, top_k_map),
            num_proc=num_worker,
        )
    else:
        majority_data = load_from_disk(cfg.majority_data_path)
        majority_dict = get_majority_dict(majority_data, check_cols, top_k_map)
        topk_label_data = inferenced_data.map(
            lambda x: map_baseline(x, check_cols, top_k_map, majority_dict),
            batched=True,
            num_proc=num_worker,
        )
    logging.debug("finding matched patients...")
    if not baseline:
        hits_and_subsets = topk_label_data.map(
            lambda x: map_match(x, check_cols, top_k_map, test_data_df),
            num_proc=num_worker,
        )
    else:
        # baseline: prediction and match is always constant, hits is different
        match_idx = []
        for col in check_cols:
            top_k = top_k_map[col]
            preds = topk_label_data[0][f"{col}_top_{top_k}_labels"]
            match_idx.append(test_data_df[col].isin(preds))
        # in the reference database, how many patient's col match the joint predictions?
        combined_idx = [all(group) for group in zip(*match_idx)]
        subset = test_data_df[combined_idx]
        hits_and_subsets = topk_label_data.map(lambda x: map_match_baseline(x, subset))
    # prepare saving
    today = datetime.now()
    time_id = today.strftime("%d_%m_%Y_%H_%M_%S")
    save_dir = os.path.join("reid_out", time_id)
    if save:
        hits_and_subsets.save_to_disk(save_dir)
    # calculate statistics
    n_hits = hits_and_subsets["n_hit"]
    subset_sizes = hits_and_subsets["subset_size"]
    correct_hits = np.sum(np.array(n_hits) > 0)
    accuracy = correct_hits / len(hits_and_subsets) * 100
    if correct_hits == 0:  # prevent divide-by-zero error
        avg_size = 0
    else:
        avg_size = np.mean(np.array(subset_sizes)[np.array(n_hits) > 0])
    # for incorrectly predicted patient, probability is 0
    # for correctly predicted patient, probability is 1/subset_size
    # so based on our existing metric, the estimation should be (1/subset_size).sum() / len(n_hits)
    unique_id_probability = (
        1 / (np.array(subset_sizes)[np.array(n_hits) > 0])
    ).sum() / len(n_hits)
    res_str = f"in total, {accuracy}% patients were found in the subset with average size of {avg_size}, total size {len(hits_and_subsets)}; estimated chance of being uniquely identified is {unique_id_probability}"
    logging.info(res_str)
    if save:
        with open(os.path.join(save_dir, "res.txt"), "w") as text_file:
            text_file.write(res_str)
        logging.info(f"output saved to {save_dir}")
    project_name = cfg.get("project_name", TOP_K_MATCH_PROJECT_NAME)
    wandb.init(
        project=project_name,
        name=f"{top_k_map}-{time_id}-{cfg.cached_inference_path}",
        config=OmegaConf.to_container(cfg),
    )
    hits_and_subsets_df = pd.DataFrame(
        {
            "n_hits": np.array(n_hits).squeeze(),
            "subset_sizes": np.array(subset_sizes).squeeze(),
        }
    )
    wandb_table = wandb.Table(dataframe=hits_and_subsets_df)
    wandb.log({"hits_and_subsets": wandb_table})
    wandb.log(
        {
            "accuracy": accuracy,
            "avg_size": avg_size,
            "unique_id_prob": unique_id_probability,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    # important: otherwise hf will create multiple copies!!!
    main()
