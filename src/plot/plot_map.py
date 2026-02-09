"""Generate NYC zip code heatmaps for social determinants of health outcomes.

Cached aggregates are protected with differential privacy (Laplace mechanism)
and small-group suppression. Row-level patient data is never written to disk.

References:
    - https://stackoverflow.com/questions/58043978/display-data-on-real-map-based-on-postal-code
    - https://geopandas.org/en/stable/docs/user_guide/mapping.html
"""

import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_BASE = Path("/gpfs/data/oermannlab/users/lavender/NYUTron_hf")

LABEL_PATHS = {
    "in_hospital_mortality": "data/raw/in_hospital_mortality/1.download_train_val_test_2011_to_2021_may.sql.csv",
    "readmitted_in_30_days": "data/raw/readmission/fixed_label/6.download_train_val_test_2011_to_2021_may_new.sql.csv",
    "los": "data/raw/los/4.download_train_val_test_2011_to_2021_may.sql.csv",
    "charles_comorbidity": "data/raw/charles_comorbidity_new/1.download_train_val_test_2011_to_2021_may.sql.csv",
}

# Value bounds per column for clipping and DP sensitivity calculation.
COLUMN_BOUNDS = {
    "in_hospital_mortality": (0, 1),
    "readmitted_in_30_days": (0, 1),
    "los": (0, 30),
    "charles_comorbidity": (0, 20),
    "acspercapitaincomeest": (0, 100_000),
    "sex": (0, 1),
    "gov_pay": (0, 1),
    "age": (0, 120),
}

AVG_COLS = list(LABEL_PATHS.keys()) + ["acspercapitaincomeest", "sex", "gov_pay", "age"]
VIZ_COLS = AVG_COLS  # count is intentionally excluded

# Display metadata for publication-quality figures.
COLUMN_DISPLAY = {
    "in_hospital_mortality": {
        "title": "In-Hospital Mortality Rate",
        "legend": "Mortality Rate",
        "cmap": "YlOrRd",
    },
    "readmitted_in_30_days": {
        "title": "30-Day Readmission Rate",
        "legend": "Readmission Rate",
        "cmap": "YlOrRd",
    },
    "los": {
        "title": "Mean Length of Stay",
        "legend": "Days",
        "cmap": "YlGnBu",
    },
    "charles_comorbidity": {
        "title": "Mean Charlson Comorbidity Index",
        "legend": "CCI Score",
        "cmap": "OrRd",
    },
    "acspercapitaincomeest": {
        "title": "Per Capita Income",
        "legend": "USD",
        "cmap": "GnBu",
    },
    "sex": {
        "title": "Proportion Female",
        "legend": "Proportion",
        "cmap": "PuBuGn",
    },
    "gov_pay": {
        "title": "Government Payer Rate",
        "legend": "Rate",
        "cmap": "BuPu",
    },
    "age": {
        "title": "Mean Patient Age",
        "legend": "Years",
        "cmap": "YlGnBu",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Reprocess raw data instead of loading from cache",
    )
    parser.add_argument(
        "--scheme",
        choices=["quantiles", "equalinterval", "naturalbreaks"],
        default="quantiles",
        help="Map classification scheme (default: quantiles)",
    )
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE, help="Base project directory"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Directory for cached CSVs"
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=None, help="Directory to save plots"
    )
    parser.add_argument(
        "--shapefile", type=Path, default=None, help="Path to NYC zip code shapefile"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Plot resolution (default: 300)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Total DP privacy budget (default: 1.0)",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=10,
        help="Suppress zip codes with fewer than this many records (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for DP noise (for reproducibility)",
    )
    return parser.parse_args()


def load_and_merge_labels(base_dir: Path) -> pd.DataFrame:
    """Load label CSVs and merge them on encounterkey."""
    merged = None
    for label_name, rel_path in LABEL_PATHS.items():
        path = base_dir / rel_path
        df = pd.read_csv(path, usecols=["encounterkey", label_name]).drop_duplicates()
        print(f"Loaded {label_name}: {len(df)} rows from {path}")
        merged = (
            df
            if merged is None
            else pd.merge(merged, df, how="inner", on="encounterkey")
        )
    return merged


def load_income_data(base_dir: Path) -> pd.DataFrame:
    """Load income/demographic data and encode categorical columns."""
    income_path = (
        base_dir
        / "data/raw/social_determinants_of_health/2.download_w_age_n_insurance.sql.csv"
    )
    income = pd.read_csv(
        income_path,
        usecols=[
            "encounterkey",
            "postalcode",
            "acspercapitaincomeest",
            "age",
            "sex",
            "payorfinancialclass",
        ],
    ).drop_duplicates()
    income["sex"] = (income["sex"] == "Female").astype(int)
    income["gov_pay"] = (
        income["payorfinancialclass"].str.lower().str.contains("medic").astype(int)
    )
    return income


def apply_dp(
    group_stats: pd.DataFrame,
    group_counts: pd.Series,
    epsilon: float,
    min_group_size: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply differential privacy to per-zipcode mean statistics.

    Uses the Laplace mechanism with:
      - Parallel composition across zip codes (disjoint groups, no extra cost).
      - Sequential composition across columns (epsilon split evenly).
      - Small-group suppression as a pre-processing step.

    For each column with bounds [a, b] in a group of size n:
      sensitivity = (b - a) / n
      noise ~ Laplace(0, sensitivity / epsilon_per_col)
    """
    rng = np.random.default_rng(seed)

    # Suppress zip codes with too few records
    valid_mask = group_counts >= min_group_size
    dp_stats = group_stats.loc[valid_mask].copy()
    counts = group_counts.loc[valid_mask]

    n_suppressed = (~valid_mask).sum()
    if n_suppressed > 0:
        print(
            f"DP: suppressed {n_suppressed} zip codes with < {min_group_size} records"
        )

    # Split epsilon across columns (sequential composition)
    eps_per_col = epsilon / len(AVG_COLS)

    for col in AVG_COLS:
        lo, hi = COLUMN_BOUNDS[col]
        dp_stats[col] = dp_stats[col].clip(lo, hi)
        # Laplace scale = sensitivity / epsilon = ((hi - lo) / n) / eps_per_col
        scale = (hi - lo) / (counts * eps_per_col)
        noise = rng.laplace(0, scale)
        dp_stats[col] = (dp_stats[col] + noise).clip(lo, hi)

    print(
        f"DP: epsilon={epsilon} ({eps_per_col:.4f} per column, {len(AVG_COLS)} columns)"
    )
    return dp_stats


def process_raw_data(
    base_dir: Path,
    cache_dir: Path,
    epsilon: float,
    min_group_size: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Load raw data, merge, aggregate by zip code, apply DP, and cache.

    Only the DP-protected aggregates are saved to disk.
    Row-level patient data is never written to the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    labels = load_and_merge_labels(base_dir)
    income = load_income_data(base_dir)
    df = pd.merge(income, labels, how="inner", on="encounterkey")
    print(f"Rows after join with zip code: {len(df)} (from {len(labels)} label rows)")

    # Normalize postal codes to first 5 digits (strip -XXXX suffix)
    df["postalcode"] = df["postalcode"].astype(str).str.split("-").str[0]

    # Clip raw values before aggregation so outliers don't inflate sensitivity
    for col in AVG_COLS:
        lo, hi = COLUMN_BOUNDS[col]
        df[col] = df[col].clip(lo, hi)

    group_stats = df.groupby("postalcode")[AVG_COLS].mean()
    group_counts = df.groupby("postalcode")["postalcode"].count()

    dp_stats = apply_dp(group_stats, group_counts, epsilon, min_group_size, seed)
    dp_stats.to_csv(cache_dir / "grouped_merged_labels.csv")
    return dp_stats


def load_cached_data(cache_dir: Path) -> pd.DataFrame:
    """Load pre-computed DP-protected group statistics from cache."""
    return pd.read_csv(cache_dir / "grouped_merged_labels.csv")


def plot_heatmaps(
    group_stats: pd.DataFrame,
    shapefile: Path,
    plot_dir: Path,
    scheme: str = "quantiles",
    dpi: int = 300,
):
    """Generate and save NYC zip code heatmaps for each visualization column."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
        }
    )

    gdf = gpd.read_file(shapefile)
    gdf = gdf.rename(columns={"ZIPCODE": "postalcode"})
    plot_df = pd.merge(gdf, group_stats, how="left", on="postalcode")

    for col in VIZ_COLS:
        display = COLUMN_DISPLAY[col]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        plot_df.plot(
            column=col,
            cmap=display["cmap"],
            legend=True,
            ax=ax,
            scheme=scheme,
            edgecolor="#999999",
            linewidth=0.3,
            missing_kwds={
                "color": "#f0f0f0",
                "edgecolor": "#999999",
                "linewidth": 0.3,
                "label": "No data",
            },
            legend_kwds={
                "fontsize": 9,
                "title": display["legend"],
                "title_fontsize": 11,
                "loc": "lower left",
                "frameon": True,
                "framealpha": 0.9,
                "edgecolor": "#cccccc",
            },
        )

        ax.set_title(display["title"], pad=12)
        ax.set_axis_off()

        for fmt in ("png", "pdf"):
            fig.savefig(
                plot_dir / f"{col}.{fmt}",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
        plt.close(fig)
        print(f"Saved {col}.png and {col}.pdf")


def main():
    args = parse_args()

    sdh_subdir = Path("src/data/social_determinant_of_health")
    cache_dir = args.cache_dir or (args.base_dir / sdh_subdir / "cache")
    plot_dir = args.plot_dir or (args.base_dir / sdh_subdir / "new_plots")
    shapefile = args.shapefile or (
        args.base_dir / sdh_subdir / "nyc_zipcode/ZIP_CODE_040114.shp"
    )

    if args.no_cache:
        group_stats = process_raw_data(
            args.base_dir, cache_dir, args.epsilon, args.min_group_size, args.seed
        )
    else:
        group_stats = load_cached_data(cache_dir)

    plot_heatmaps(group_stats, shapefile, plot_dir, scheme=args.scheme, dpi=args.dpi)


if __name__ == "__main__":
    main()
