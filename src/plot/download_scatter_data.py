"""Download re-identification experiment results from wandb.

Usage: WANDB_ENTITY=myname python -m src.plot.download_scatter_data
"""

import os

import pandas as pd
import wandb
from tqdm import tqdm


def _get_wandb_project():
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise RuntimeError(
            "WANDB_ENTITY environment variable is not set. "
            "Set it to your wandb username or team: export WANDB_ENTITY=myname"
        )
    return f"{entity}/dummy_reid"


WANDB_PROJECT = _get_wandb_project()
CACHE_DIR = "src/plot/figures/data"
CACHE_FILE = os.path.join(CACHE_DIR, "cached_reid_log.csv")


def download_reid_data(wandb_project=WANDB_PROJECT, output_path=CACHE_FILE):
    print(f"Fetching runs from {wandb_project}...")
    api = wandb.Api()
    runs = api.runs(wandb_project)

    data_list = []
    for run in tqdm(iter(runs), desc="Fetching runs"):
        config = run.config
        summary = run.summary

        data_list.append(
            {
                "run_name": run.name,
                "baseline": config.get("baseline", "Unknown"),
                "avg_size": summary.get("avg_size"),
                "accuracy": summary.get("accuracy"),
                "unique_id_prob": summary.get("unique_id_prob"),
                "top_k_map": str(config.get("top_k_map")),
            }
        )

    df = pd.DataFrame(data_list)
    df = df.dropna(subset=["avg_size", "accuracy", "unique_id_prob"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} runs to {output_path}")
    return df


if __name__ == "__main__":
    download_reid_data()
