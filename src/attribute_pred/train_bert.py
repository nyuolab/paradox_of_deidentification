# train the attribute predictor
# usage: python -m src.attribute_pred.train_bert --config-name dmonth_classification_deid trainer.num_train_epochs=0.1
import math
import os
import random
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk, load_metric
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import wandb
from src.util import Subsampler

softmax = nn.Softmax()


# reference: https://huggingface.co/docs/transformers/main_classes/trainer
def preprocess_logits_for_metrics(logits, labels, multiclassification=False):
    # reference: https://github.com/huggingface/transformers/issues/30229
    probs = softmax(logits)
    if multiclassification:
        return probs
    else:
        pos_probs = probs[:, 1]
        return pos_probs


def compute_metrics(
    eval_preds, metric_modules, multiclassification=False, multi_class="ovo"
):
    res = {}
    preds, labels = eval_preds
    for metric_name, metric in metric_modules.items():
        if metric_name == "roc_auc":
            if multiclassification:
                for average in ["macro", "weighted"]:
                    print(
                        f"\n\n\n+++++++++++++{len(preds), len(labels)}++++++++++++++++"
                    )
                    metric_res = metric.compute(
                        references=labels,
                        prediction_scores=preds,
                        average=average,
                        multi_class=multi_class,
                    )
                    res[f"{metric_name}_{average}"] = metric_res[metric_name]
                res[metric_name] = metric_res[metric_name]
            else:
                metric_res = metric.compute(references=labels, prediction_scores=preds)
                res[metric_name] = metric_res[metric_name]
        elif metric_name == "accuracy":
            if len(preds.shape) < 2:
                threshold = 0.5
                pred_labels = (preds >= threshold).astype(int)
            else:
                pred_labels = np.argmax(preds, axis=1)
            metric_res = metric.compute(references=labels, predictions=pred_labels)
            res[metric_name] = metric_res[metric_name]
    return res


def train(model, data, eval_data, test_data, tokenizer, conf):
    args = TrainingArguments(
        output_dir=conf.logger.save_dir,
        save_strategy=conf.trainer.save_strategy,
        save_steps=conf.trainer.save_steps,
        learning_rate=conf.trainer.lr,
        num_train_epochs=conf.trainer.num_train_epochs,
        weight_decay=conf.trainer.weight_decay,
        logging_strategy=conf.trainer.logging_strategy,
        logging_steps=conf.trainer.logging_steps,
        eval_steps=conf.trainer.eval_steps,
        evaluation_strategy=conf.trainer.evaluation_strategy,
        per_device_train_batch_size=conf.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=conf.trainer.per_device_eval_batch_size,
        load_best_model_at_end=True,
        save_total_limit=conf.trainer.save_total_limit,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        gradient_accumulation_steps=conf.trainer.gradient_accumulation_steps,
        report_to=conf.logger.report_to,
    )
    conf.logger.run_id = wandb.util.generate_id()

    callbacks = []
    if conf.trainer.early_stop:
        print("using early stopping callbacks")
        early_stopper = EarlyStoppingCallback(early_stopping_patience=5)
        callbacks.append(early_stopper)

    metrics = conf.trainer.metric
    metric_modules = {}
    multiclassification = conf.data.num_label > 2
    for metric in metrics:
        print(f"loading metric {metric}")
        if metric == "roc_auc" and multiclassification:
            metric_modules[metric] = load_metric(
                metric, "multiclass", experiment_id=conf.logger.run_id
            )
        else:
            metric_modules[metric] = load_metric(
                metric, experiment_id=conf.logger.run_id
            )
    wandb.login()
    trainer = Trainer(
        model,
        args,
        train_dataset=data,
        eval_dataset=eval_data,
        compute_metrics=lambda x: compute_metrics(
            x, metric_modules, multiclassification
        ),
        preprocess_logits_for_metrics=lambda p, y: preprocess_logits_for_metrics(
            p, y, multiclassification
        ),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    conf_save_path = os.path.join(conf.logger.save_dir, "config.yaml")
    OmegaConf.save(config=conf, f=conf_save_path)
    print(f"save configs to {conf_save_path}!")

    if conf.logger.report_to == "wandb":
        print("initializing wandb....")
        wandb.init(
            project=conf.logger.project,
            entity=os.environ.get("WANDB_ENTITY"),
            name=conf.logger.run_name,
            id=conf.logger.run_id,
        )
        wandb.config.update(OmegaConf.to_container(conf))
        print("done init wandb!")

    trainer.train()

    res = trainer.evaluate(eval_dataset=test_data)
    print(f"test result: {res}")
    report = {
        "test/roc_auc": res["eval_roc_auc"],
        "test/loss": res["eval_loss"],
        "test/acc": res["eval_accuracy"],
    }
    if conf.logger.report_to == "wandb":
        wandb.log(report)
    trainer.save_model(
        output_dir=f"./models/{conf.data.task}_{conf.data.num_train_samples}"
    )
    return trainer


@hydra.main(
    version_base=None,
    config_path="../../data/configs",
    config_name="bourough_classification_deid",
)
def finetune(conf: DictConfig) -> None:
    torch.manual_seed(conf.run.seed)
    np.random.seed(conf.run.seed)
    random.seed(conf.run.seed)
    set_seed(conf.run.seed)
    print(conf)
    model = AutoModelForSequenceClassification.from_pretrained(
        conf.model.path, num_labels=conf.data.num_label
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        conf.data.tokenizer.path, max_length=512
    )
    print(f"loaded tokenizer from {conf.data.tokenizer.path}")
    print(f"tokenizer is {tokenizer}")
    full_data = load_from_disk(conf.data.tokenized_data_path)
    print(f"loaded data {full_data} from {conf.data.tokenized_data_path}")
    if conf.data.num_train_samples is not None:
        subsampler = Subsampler(seed=conf.run.seed, data=full_data["train"])
        data = subsampler.subsample(conf.data.num_train_samples)
    else:
        data = full_data["train"]
    if conf.data.num_eval_samples is not None:
        eval_subsampler = Subsampler(seed=conf.run.seed, data=full_data["val"])
        eval_data = eval_subsampler.subsample(conf.data.num_eval_samples)
    else:
        eval_data = full_data["val"]
    if conf.run.debug:
        test_subsampler = Subsampler(seed=conf.run.seed, data=full_data["test"])
        test_data = test_subsampler.subsample(100)
    else:
        test_data = full_data["test"]

    n_gpus = torch.cuda.device_count()
    n_examples_per_step = (
        conf.trainer.per_device_train_batch_size
        * n_gpus
        * conf.trainer.gradient_accumulation_steps
    )
    if n_gpus == 0:  # using cpu for debugging
        eval_steps = 100
    else:
        n_steps_per_epoch = math.ceil(len(data) / n_examples_per_step)
        eval_steps = math.ceil(n_steps_per_epoch * conf.trainer.p_eval)
    if eval_steps < 1:
        eval_steps = 1
    conf.trainer.eval_steps = eval_steps
    conf.trainer.save_steps = conf.trainer.eval_steps
    print(f"setting eval_steps and save_steps to {eval_steps}")

    # configure wandb log
    today = datetime.now()
    time_id = today.strftime("%d_%m_%Y_%H_%M_%S")
    conf.logger.run_name = f"{conf.data.task}-{conf.model.pretrained}-{conf.data.num_train_samples}samples-seed{conf.run.seed}_{time_id}"
    if conf.logger.report_to == "wandb":
        conf.logger.run_id = wandb.util.generate_id()
        print(f"wandb run is is {conf.logger.run_id}")
    save_dir = f"{conf.logger.output_dir}/{conf.logger.run_name}"
    conf.logger.save_dir = save_dir
    print(f"result will save to {save_dir}")

    train(
        model,
        data=data,
        eval_data=eval_data,
        test_data=test_data,
        tokenizer=tokenizer,
        conf=conf,
    )


if __name__ == "__main__":
    finetune()
