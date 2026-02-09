# visually examine the dataset
# usage: python -m tests.check_dataset --path ./data/dummy_data_tokenized_condition
import argparse

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Visually examine the dataset")
    parser.add_argument(
        "--path",
        type=str,
        default="./data/dummy_data_tokenized_id_text",
        help="Path to the dataset",
    )
    return parser.parse_args()


args = parse_args()
dataset = load_from_disk(args.path)

split = "test"
tokenizer_cols = [
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "special_tokens_mask",
]


def get_cols(example):
    check_cols = [x for x in example.keys() if not (x in tokenizer_cols)]
    return check_cols


for idx in range(min(3, len(dataset[split]))):
    example = dataset[split][idx]
    for col in tokenizer_cols:
        assert col in example.keys()
    for col in get_cols(example):
        print(col, ":", example[col])
    print("===========")
