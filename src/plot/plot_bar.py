# %%
# import packages
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.plot.wandb_helpers import get_column_stats

# %%
# Configuration matching scripts/train_predict_attributes.sh
figure_folder = "src/plot/figures"
os.makedirs(figure_folder, exist_ok=True)

col_names = ["gender", "borough", "year", "month", "income", "insurance"]

# Map display col name -> wandb project suffix (appended to "$WANDB_ENTITY/sdh-")
wandb_name_map = {
    "gender": "gender-dummy",
    "borough": "borough-dummy-deid",
    "year": "dyear-dummy",
    "month": "dmonth-dummy",
    "income": "income-dummy",
    "insurance": "government-pay-dummy",
}

n_classes_map = {
    "gender": 2,
    "borough": 6,
    "year": 10,
    "month": 12,
    "income": 2,
    "insurance": 2,
}

sample_sizes = [100, 500]
USE_CACHE = True


# %%
def generate_col_df(col_name, metric_name, metric_vals, samples_list, n_classes=2):
    if metric_name == "accuracy":
        random_baseline = 1 / n_classes * 100
    elif metric_name == "auc":
        random_baseline = 0.5 * 100
    else:
        raise ValueError(f"metric name {metric_name} not implemented!")
    above_random = [metric_val - random_baseline for metric_val in metric_vals]
    all_vals = [random_baseline] * len(samples_list) + above_random
    two_n_samples = samples_list + samples_list
    types = ["random baseline"] * len(samples_list) + ["above random"] * len(
        samples_list
    )
    df = pd.DataFrame(
        {"metric_val": all_vals, "n_samples": two_n_samples, "perf_type": types}
    )
    df["col"] = col_name
    df["metric"] = metric_name
    return df


# %%
# Fetch data from wandb
df_aucs = []
df_accs = []

for col in col_names:
    print(f"Fetching stats for {col}...")
    try:
        wandb_col = wandb_name_map[col]
        stats, _ = get_column_stats(wandb_col, use_cache=USE_CACHE)

        current_accs = []
        current_aucs = []

        for n in sample_sizes:
            if n in stats.index:
                acc = stats.loc[n, ("test_acc", "mean")]
                auc = stats.loc[n, ("test_roc_auc", "mean")]

                if acc <= 1.0:
                    acc *= 100
                if auc <= 1.0:
                    auc *= 100

                current_accs.append(acc)
                current_aucs.append(auc)
            else:
                print(f"Warning: Missing data for {col} at {n} samples")
                current_accs.append(0)
                current_aucs.append(0)

        n_cls = n_classes_map.get(col, 2)
        df_accs.append(
            generate_col_df(
                col, "accuracy", current_accs, sample_sizes, n_classes=n_cls
            )
        )
        df_aucs.append(
            generate_col_df(col, "auc", current_aucs, sample_sizes, n_classes=n_cls)
        )

    except Exception as e:
        print(f"Error fetching data for {col}: {e}")

combined_acc = pd.concat(df_accs).round({"metric_val": 2})
combined_auc = pd.concat(df_aucs).round({"metric_val": 2})

# %%
# Plotting
mode = "save"  # "interactive"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def format_n(n):
    if n >= 1000000:
        return f"{n/1000000:.0f}M"
    if n >= 1000:
        return f"{n/1000:.0f}k"
    return str(n)


col_map = {
    "gender": (0, 0),
    "borough": (0, 1),
    "year": (0, 2),
    "month": (1, 0),
    "income": (1, 1),
    "insurance": (1, 2),
}

display_name_map = {
    "gender": "Biological Sex",
    "borough": "Neighborhood",
    "year": "Year",
    "month": "Month",
    "income": "Income",
    "insurance": "Insurance",
}

colors = ["#e0e0e0", "#ed2b2b"]

for metric_idx in [0, 1]:
    combined = combined_acc if metric_idx == 0 else combined_auc
    ylabel = "Accuracy (%)" if metric_idx == 0 else "AUC (%)"

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for col in col_names:
        coord = col_map[col]
        ax = axs[coord[0], coord[1]]

        sub_df = combined[combined.col == col]
        plot_df = sub_df.pivot(
            index="n_samples", columns="perf_type", values="metric_val"
        )
        plot_df = plot_df[["random baseline", "above random"]]
        plot_df.index = [format_n(x) for x in plot_df.index]

        plot_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=colors,
            rot=0,
            legend=False,
            width=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        display = display_name_map.get(col, col.capitalize())
        ax.set_title(f"Target: {display}", fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Training Samples", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylim(0, 105)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        ["Signal from 'De-identified' Notes", "Prediction based on Random Guess"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if mode == "interactive":
        plt.show()
    else:
        fname = f"per_col_{'accuracy' if metric_idx == 0 else 'auc'}_bar.pdf"
        plt.savefig(
            os.path.join(figure_folder, fname), bbox_inches="tight", dpi=300
        )
        print(f"Saved {fname}")
    plt.close()
