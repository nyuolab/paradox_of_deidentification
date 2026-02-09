"""Run a sampled set of reid top-k configs in parallel.

Usage: python scripts/run_reid_sample.py
"""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce

import pandas as pd

# Same as src.constants.N_CLASSES — inlined to allow running as a script
N_CLASSES = {
    "sex": 2,
    "dyear": 10,
    "postal_code_borough": 6,
    "dmonth": 12,
    "income_token": 2,
    "payorfinancialclass": 2,
}

WANDB_PROJECT = "dummy_reid"
N_PARALLEL = 6
N_SAMPLE = 100  # configs to sample (each run as both baseline=True and baseline=False)


def get_all_configs():
    product_elements = []
    for key, val in N_CLASSES.items():
        product_elements.append(pd.DataFrame({key: list(range(val))}))
    cart = reduce(lambda l, r: pd.merge(l, r, how="cross"), product_elements)
    # drop the all-zero row
    cart = cart[cart.sum(axis=1) > 0]
    return cart


def run_one(row_dict, baseline):
    check_cols = [col for col, k in row_dict.items() if k > 0]
    if not check_cols:
        return None
    top_k_parts = [f"top_k_map.{col}={k}" for col, k in row_dict.items() if k > 0]
    cmd = (
        f"python -m src.reid.topk_match "
        f"+project_name={WANDB_PROJECT} "
        f"+baseline={baseline} "
        f"\"check_cols={check_cols}\" "
        + " ".join(top_k_parts)
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tag = "baseline" if baseline else "model"
    k_str = {c: row_dict[c] for c in check_cols}
    if result.returncode == 0:
        return f"[OK] {tag} {k_str}"
    else:
        err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
        return f"[FAIL] {tag} {k_str}: {err}"


def main():
    all_configs = get_all_configs()
    sampled = all_configs.sample(n=min(N_SAMPLE, len(all_configs)), random_state=42)
    rows = [row.to_dict() for _, row in sampled.iterrows()]

    # Build work items: each config × both baseline modes
    work = []
    for row in rows:
        work.append((row, True))
        work.append((row, False))

    total = len(work)
    print(f"Running {total} experiments ({len(rows)} configs × 2) with {N_PARALLEL} workers...")

    done = 0
    with ProcessPoolExecutor(max_workers=N_PARALLEL) as pool:
        futures = {pool.submit(run_one, r, b): (r, b) for r, b in work}
        for future in as_completed(futures):
            done += 1
            msg = future.result()
            if msg:
                print(f"[{done}/{total}] {msg}")
            if done % 20 == 0:
                print(f"--- Progress: {done}/{total} ({done/total*100:.0f}%) ---")

    print(f"\nDone! Completed {done} runs.")


if __name__ == "__main__":
    main()
