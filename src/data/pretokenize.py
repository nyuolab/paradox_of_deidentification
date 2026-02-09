# Tokenize and preprocess data for de-identification tasks
#
# This script provides functionality to tokenize data, and optionally augment and transform it.
#
# Usage:
# 1. id_text
# python -m src.data.pretokenize  --model_name bert-base-uncased --input_data_path ./data/dummy_data --basepath ./data --tokenize_col id_text
# 2. deid_text
# python -m src.pretokenize  --model_name bert-base-uncased --input_data_path ./data/dummy_data --basepath ./data --tokenize_col deid_text
# 3. condition
# python -m src.data.pretokenize  --model_name bert-base-uncased --input_data_path ./data/dummy_data --basepath ./data --tokenize_col condition
import argparse
import os

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize and process data for de-identification"
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        help="Path to input data",
        default="./data/dummy_data",
    )
    parser.add_argument("--model_name", default="bert-base-uncased", help="Model name")
    parser.add_argument(
        "--basepath",
        default="./data",
        help="Base path for data",
    )
    parser.add_argument(
        "--tokenize_col",
        choices=["deid_text", "id_text", "condition"],
        default="deid_text",
        help="Column to tokenize",
    )
    return parser.parse_args()


def augment_condition(batch, tok_col):
    input_text = [
        "What medical conditions do this patient have? " + text
        for text in batch[tok_col]
    ]
    tokenized = extract_tokenizer(
        input_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_special_tokens_mask=True,
    )
    return tokenized


def tokenize(x, tokenizer, tok_col):
    print(f"tokenizing with {tokenizer} on {tok_col}")
    try:
        res = tokenizer(
            x[tok_col],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_special_tokens_mask=True,
        )
    except:
        text = x[tok_col]
        print(f"x has type {type(text)}, length {len(text)}, value {text}")
        raise ValueError("Error in tokenization")
    return res


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=512)

    if args.tokenize_col == "condition":
        global extract_tokenizer
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        extract_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-base", max_length=512
        )

    data = load_from_disk(args.input_data_path)

    n_proc = 8

    if args.tokenize_col == "condition" and "condition" not in data.column_names:
        tokenize_col = "id_text"
        tokenized_question = data.map(
            lambda x: augment_condition(x, tokenize_col),
            batched=True,
            num_proc=n_proc,
        )

        seq2seq_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            per_gpu_eval_batch_size=8,
            output_dir="gen",
            do_predict=True,
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=model, tokenizer=extract_tokenizer, args=seq2seq_args
        )
        for split in tokenized_question.keys():
            out = trainer.predict(tokenized_question[split])
            out_text = extract_tokenizer.batch_decode(
                out.predictions, skip_special_tokens=True
            )
            tokenized_question[split] = tokenized_question[split].add_column(
                "condition", out_text
            )
        data = tokenized_question

    tokenized_datasets = data.map(
        lambda x: tokenize(x, tokenizer, args.tokenize_col),
        batched=True,
        num_proc=n_proc,
    )

    ckpt_dir = os.path.join(args.basepath, f"dummy_data_tokenized_{args.tokenize_col}")
    tokenized_datasets.save_to_disk(ckpt_dir)
    print(f"saved tokenized dataset to {ckpt_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
