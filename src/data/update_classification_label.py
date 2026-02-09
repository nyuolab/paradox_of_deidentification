# update the label column to proper value
# Usage examle:
# python -m src.data.update_classification_label --data_path ./data/dummy_data_tokenized_condition --colname dyear --output_dir ./data/dummy_data_tokenized_condition_dyear
import argparse

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.constants import MAX_YEAR, MIN_YEAR


def parse_args():
    parser = argparse.ArgumentParser(description="Update label column in dataset")
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="Model name"
    )
    parser.add_argument(
        "--colname",
        choices=[
            "payorfinancialclass",
            "postal_code_borough",
            "dyear",
            "dmonth",
            "sex",
            "income_token",
        ],
        default="payorfinancialclass",
        help="Column name for label",
    )
    parser.add_argument(
        "--max_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    return parser.parse_args()


def update_label(example):
    if args.colname == "dyear":
        year = int(example[args.colname])
        if not (year >= MIN_YEAR and year <= MAX_YEAR):
            raise ValueError(f"year {year} out of range!!!!")
        res = year % MIN_YEAR

    elif args.colname == "dmonth":
        month = int(example[args.colname])
        assert month >= 1 and month <= 12
        res = month - 1  # we zero index the label

    elif args.colname == "sex":  # discard the nonspecified classes
        sex = example[args.colname].strip().lower()
        if sex == "female":
            res = 1
        elif sex == "male":
            res = 0
        else:
            res = -1

    elif args.colname == "postal_code_borough":
        b_dict = {
            "Manhattan": 0,
            "Brooklyn": 1,
            "Bronx": 2,
            "Queens": 3,
            "Staten Island": 4,
            "Others": 5,
        }
        res = b_dict[example[args.colname]]

    elif args.colname == "income_token":
        income = example[args.colname].strip()
        b_dict = {"[POOR]": 0, "[RICH]": 1}
        res = b_dict[income]

    elif args.colname == "payorfinancialclass":
        payor_class = example[args.colname].lower()
        if "medicaid" in payor_class or "medicare" in payor_class:
            res = 1  # label for whether or not the patient was paid by the government
        else:
            res = 0

    else:
        raise ValueError(f"update_label not defined {args.colname}!")
    return {"label": res}


def main(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data = load_from_disk(args.data_path)
    print(f"loaded data from {args.data_path}")
    print(data)

    n_proc = 1 if args.debug else 8
    updated_dataset = data.map(update_label, batched=False, num_proc=n_proc)

    print(updated_dataset)

    updated_dataset.save_to_disk(args.output_dir)
    print(f"saved output to {args.output_dir}")

    # sanity check unique values in label
    unique_labels = np.unique(updated_dataset["train"]["label"])
    print(f"unique labels: {unique_labels}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
