# usage: python -m src.reid.cache_probs
import argparse
import logging
import os

import torch
import torch.nn as nn
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from src.constants import ALL_COLS, N_CLASSES
from src.util import get_model_path_dict, setup_logging


def get_model_probs(example_batch, model_dict, missing_cols, device):
    res = {}
    for col in tqdm(missing_cols, desc=f"Getting model probs..."):
        logging.debug(f"col: {col}")
        out = model_dict[col](
            input_ids=torch.tensor(example_batch["input_ids"]).to(device),
            token_type_ids=torch.tensor(example_batch["token_type_ids"]).to(device),
            attention_mask=torch.tensor(example_batch["attention_mask"]).to(device),
        )
        prob = nn.Softmax(dim=-1)(out["logits"]).detach()
        res[f"{col}_prob"] = prob
    return res


def main(args):
    # Note: this script uses cpu only. no need to request GPUs.
    # Load data
    test_data = load_from_disk(args.data_dir)
    model_path_dict = get_model_path_dict(args.n_train_samples)

    # Initialize models and prediction results
    model_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for col_name in ALL_COLS:
        model_path = model_path_dict[col_name]
        model_dict[col_name] = (
            AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=N_CLASSES[col_name]
            )
            .eval()
            .to(device)
        )

    inferenced_data = test_data.map(
        lambda x: get_model_probs(x, model_dict, ALL_COLS, device),
        batched=True,
        batch_size=10,
    )
    save_path = os.path.join(args.save_dir, "inference_data")
    inferenced_data.save_to_disk(save_path)
    logging.info(f"inferenced_data saved to {save_path}")
    logging.info("Examples: {inference_data[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-identification script")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/dummy_data_tokenized_deid_text/test",
        help="test data directory",
    )
    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=100,
        help="Number of training samples used to train the model",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
    parser.add_argument(
        "--save_dir", type=str, default="./models", help="Save directory"
    )
    args = parser.parse_args()
    setup_logging(args.log_level)
    main(args)
