import wandb
import pandas as pd
import os


def _default_entity_prefix():
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise RuntimeError(
            "WANDB_ENTITY environment variable is not set. "
            "Set it to your wandb username or team: export WANDB_ENTITY=myname"
        )
    return f"{entity}/sdh-"


def get_column_stats(column_name, entity_prefix=None, use_cache=False):
    if entity_prefix is None:
        entity_prefix = _default_entity_prefix()
    cache_file = f"{column_name}.csv"
    if use_cache and os.path.exists(cache_file):
        print(f"Loading stats for {column_name} from cache: {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        api = wandb.Api()
        runs = api.runs(f"{entity_prefix}{column_name}")

        print(f"Found {len(runs)} runs")

        data = []
        for run in runs:
            summary = run.summary
            config = run.config

            # Some runs might not have created_at properly, but usually they do.
            # run.created_at is a string timestamp from wandb API.
            entry = {
                "id": run.id,
                "name": run.name,
                "created_at": run.created_at,
                "num_train_samples": config.get("data", {}).get("num_train_samples"),
                "seed": config.get("run", {}).get("seed"),
                "test_acc": summary.get("test/acc"),
                "test_roc_auc": summary.get("test/roc_auc"),
            }
            data.append(entry)

        df = pd.DataFrame(data)

        if use_cache and not df.empty:
            print(f"Saving stats for {column_name} to cache: {cache_file}")
            df.to_csv(cache_file, index=False)

    if not df.empty:
        if "seed" in df.columns:
            df["seed"] = df["seed"].fillna(0)

        # Ensure we have datetime for sorting
        df["created_at"] = pd.to_datetime(df["created_at"])

        # Sort by created_at descending to keep the latest
        df = df.sort_values("created_at", ascending=False)

        # Drop duplicates, keeping the first (most recent) for each (num_train_samples, seed) combo
        # We need both columns to be present to deduplicate
        if "num_train_samples" in df.columns and "seed" in df.columns:
            df = df.drop_duplicates(subset=["num_train_samples", "seed"], keep="first")

        # Validate 5 runs per combo
        if "num_train_samples" in df.columns and "seed" in df.columns:
            seed_counts = df.groupby("num_train_samples")["seed"].nunique()
            for n_samples, count in seed_counts.items():
                if count != 5:
                    print(
                        f"WARNING: Sample size {n_samples} has {count} runs (expected 5)."
                    )

    result = pd.DataFrame()
    if not df.empty:
        # Group by number of train samples and calculate mean and std of test_acc and test_roc_auc
        result = df.groupby("num_train_samples")[["test_acc", "test_roc_auc"]].agg(
            ["mean", "std", "count"]
        )

    return result, df


if __name__ == "__main__":
    expected_seeds = {0, 13, 24, 36, 42}

    for col in [
        "borough-new",
        "gender-new",
        "income-new",
        "dmonth-new",
        "dyear-new",
        "government-pay-new",
    ]:
        print(col)
        stats, df = get_column_stats(col)

        print(df.head())
        if not stats.empty:
            print("-" * 20)
            print(stats)

        print("-" * 20)
        print(
            "Unique train samples:",
            (
                df["num_train_samples"].unique()
                if "num_train_samples" in df and not df.empty
                else "No train samples"
            ),
        )

        # Check distribution
        if not df.empty:
            df = df.dropna(subset=["test_acc", "test_roc_auc"])
            print(df.groupby(["num_train_samples"]).size())

            # Check for missing seeds
            print("Missing seeds per sample size:")
            groups = df.groupby("num_train_samples")
            for sample_size, group in groups:
                present_seeds = set(group["seed"].unique())
                missing_seeds = expected_seeds - present_seeds
                if missing_seeds:
                    print(f"  Sample size {sample_size}: Missing {missing_seeds}")
                else:
                    print(f"  Sample size {sample_size}: All seeds present")

        print("=" * 10)
        print("\n")
