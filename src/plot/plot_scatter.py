# %%
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import ast
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors

# 1. Check for cache or fetch runs from WandB
cache_file = "src/plot/figures/data/cached_reid_log.csv"
repull = True
match_all_only = False
suffix = "_match_all_only" if match_all_only else ""

if os.path.exists(cache_file) and not repull:
    print(f"Loading data from {cache_file}...")
    df = pd.read_csv(cache_file)
else:
    print("Fetching runs from WandB...")
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise RuntimeError(
            "WANDB_ENTITY environment variable is not set. "
            "Set it to your wandb username or team: export WANDB_ENTITY=myname"
        )
    runs = api.runs(f"{entity}/dummy_reid")

    # 2. Extract data into a list of dictionaries
    data_list = []
    for run in tqdm(iter(runs)):
        # Safely get keys, defaulting to None if missing
        config = run.config
        summary = run.summary

        data_list.append(
            {
                "run_name": run.name,
                "baseline": config.get("baseline", "Unknown"),  # Grouping attribute
                "avg_size": summary.get("avg_size"),  # x-axis (smaller is better)
                "accuracy": summary.get("accuracy"),  # y-axis (bigger is better)
                "unique_id_prob": summary.get(
                    "unique_id_prob"
                ),  # Color (higher is better)
                "top_k_map": config.get("top_k_map"),
            }
        )
        # # DEBUG: test with 10 runs
        # if len(data_list) == 10:
        #     break

    # 3. Convert to DataFrame and clean missing data
    df = pd.DataFrame(data_list)
    print(df["top_k_map"])
    df = df.dropna(subset=["avg_size", "accuracy", "unique_id_prob"])

    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving data to {cache_file}...")
    df.to_csv(cache_file, index=False)

# 4. Create the Plot
sns.set_theme(style="whitegrid")  # Cleaner background


def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x if isinstance(x, dict) else {}


df["top_k_map"] = df["top_k_map"].apply(safe_eval)

if match_all_only:
    # filter for top k > 0
    df = df[df["top_k_map"].apply(lambda d: all(v > 0 for v in d.values()))]

# === CONFIG FOR ACADEMIC PAPER ===
plt.rcParams.update(
    {
        # "font.family": "serif",  # Matches LaTeX/Paper font
        "font.size": 12,  # Readable text
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


# Filter out rows with avg_size=0 (no matches found) to avoid division by zero
df = df[df["avg_size"] > 0]
df["draw_chance"] = 1 / df["avg_size"]

# Separate the groups
# Assuming 'baseline' column is boolean or string. Adjust 'True'/'Unknown' as needed.
# If baseline runs have 'baseline' == True:
df_baseline = df[df["baseline"] == True]  # Or whatever identifies the random guess runs
df_ours = df[df["baseline"] == False]  # The De-identified runs

# 2. Setup Canvas
fig, ax = plt.subplots(figsize=(12, 5))  # Standard column width ratio


## Normalize the color scale
# Calculate the global min/max for the color dimension
all_probs = pd.concat([df_baseline["unique_id_prob"], df_ours["unique_id_prob"]]).dropna()
v_min = all_probs.min() * 100
v_max = all_probs.max() * 100

# Create a normalization object
norm = mcolors.Normalize(vmin=v_min, vmax=v_max)

# 3. Layer 1: The Baseline (Background/Context)
# We make these grey and distinct to show they are the "floor"
ax.scatter(
    df_baseline["accuracy"],
    df_baseline["draw_chance"] * 100,
    # c="gray",  # Neutral color
    c=df_baseline["unique_id_prob"] * 100,
    cmap="viridis",  # 'viridis' or 'plasma' print well in B&W too
    norm=norm,
    marker="X",  # Distinct "Null" marker
    s=80,  # Size
    alpha=0.6,  # Slight transparency
    label="Random Guess",
)

# 4. Layer 2: Your Method (The Hero)
# We map color to 'unique_id_prob' to show the signal density
sc = ax.scatter(
    df_ours["accuracy"],
    df_ours["draw_chance"] * 100,
    c=df_ours["unique_id_prob"] * 100,
    cmap="viridis",  # 'viridis' or 'plasma' print well in B&W too
    norm=norm,
    marker="o",
    s=120,
    edgecolor="black",  # Sharp edges define the points clearly
    linewidth=0.8,
    alpha=0.8,
    label="LM Guess",
    zorder=10,  # Force these on top
)

# 5. Add Colorbar for the Signal
cbar = plt.colorbar(sc, format="%.2f%%")
cbar.set_label("Probability of individual re-identification (%)", labelpad=10)

# 6. Annotate the "Gap" (The visual proof)
# Find means to draw a representative arrow
if not df_baseline.empty and not df_ours.empty:
    base_y = df_baseline["accuracy"].mean()
    ours_y = df_ours["accuracy"].mean()
    mid_x = df_ours["avg_size"].mean()

    # Draw a double-headed arrow to show the lift
    ax.annotate(
        "",
        xy=(mid_x, ours_y),
        xytext=(mid_x, base_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    )
    # Add text label for the arrow
    ax.text(
        mid_x + (mid_x * 0.02),
        (base_y + ours_y) / 2,
        "Performance Gap",
        verticalalignment="center",
        fontweight="bold",
    )

# 7. Polish
ax.set_ylabel("Probability of random draw\nfrom the identified group")
ax.set_xlabel("Probability that patient is in the identified group")
ax.set_title(
    "Risk of $\mathrm{\it{Individual}}$ Re-identification from De-identified Data"
)
ax.grid(True, linestyle="--", alpha=0.5)

ax.xaxis.set_major_formatter(PercentFormatter(100))
ax.yaxis.set_major_formatter(PercentFormatter(100))

# Custom Legend
ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")

plt.tight_layout()
save_name = "src/plot/figures/reid_paper_plot.pdf"
plt.savefig(save_name, dpi=300)  # 300 DPI is standard for papers
print(f"Saved publication-quality plot to {save_name}")
