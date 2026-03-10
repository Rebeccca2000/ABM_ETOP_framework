import argparse
import glob
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd


ALLOCATION_KEYS = [
    "low_bike",
    "low_car",
    "low_MaaS_Bundle",
    "low_public",
    "middle_bike",
    "middle_car",
    "middle_MaaS_Bundle",
    "middle_public",
    "high_bike",
    "high_car",
    "high_MaaS_Bundle",
    "high_public",
]


def t_critical_975(df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:
        if df <= 1:
            return 12.706
        if df == 2:
            return 4.303
        if df == 3:
            return 3.182
        if df == 4:
            return 2.776
        if df == 5:
            return 2.571
        if df <= 10:
            return 2.228
        if df <= 20:
            return 2.086
        return 1.96


def summarize_with_ci(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci95_halfwidth": np.nan}

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n >= 2 else 0.0
    if n >= 2:
        tcrit = t_critical_975(n - 1)
        ci = float(tcrit * std / np.sqrt(n))
    else:
        ci = 0.0
    return {"n": n, "mean": mean, "std": std, "ci95_halfwidth": ci}


def parse_seed_from_path(path: str) -> int:
    m = re.search(r"/seed_(\d+)(?:/|$)", path.replace("\\", "/"))
    if m:
        return int(m.group(1))
    return -1


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def extract_result_rows(seed: int, all_results: dict, source_path: str) -> list[dict]:
    rows: list[dict] = []
    for fps, result in all_results.items():
        fps_value = safe_float(fps)
        if not isinstance(result, dict):
            continue

        full = result.get("full_results", {}) if isinstance(result.get("full_results"), dict) else {}
        equity_scores = result.get("equity_scores", {}) if isinstance(result.get("equity_scores"), dict) else {}
        subsidy_usage = result.get("subsidy_usage", {}) if isinstance(result.get("subsidy_usage"), dict) else {}
        allocations = result.get("optimal_allocations")
        if allocations is None:
            allocations = full.get("fixed_allocations", {})
        if not isinstance(allocations, dict):
            allocations = {}

        row = {
            "seed": int(seed),
            "fps_value": fps_value,
            "source_path": source_path,
            "avg_equity": safe_float(result.get("avg_equity", np.nan)),
            "equity_low": safe_float(
                equity_scores.get("low", full.get("low", {}).get("equity_indicator", np.nan))
            ),
            "equity_middle": safe_float(
                equity_scores.get("middle", full.get("middle", {}).get("equity_indicator", np.nan))
            ),
            "equity_high": safe_float(
                equity_scores.get("high", full.get("high", {}).get("equity_indicator", np.nan))
            ),
            "budget_used_pct": safe_float(subsidy_usage.get("percentage_used", np.nan)),
            "total_subsidy_used": safe_float(subsidy_usage.get("total_subsidy_used", np.nan)),
            "terminated_early": bool(result.get("terminated_early", False)),
            "termination_reason": result.get("termination_reason", ""),
        }
        for key in ALLOCATION_KEYS:
            row[key] = safe_float(allocations.get(key, np.nan))
        rows.append(row)
    return rows


def build_performance_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "avg_equity": "total",
        "equity_low": "low",
        "equity_middle": "middle",
        "equity_high": "high",
        "budget_used_pct": "budget_used",
    }
    rows = []
    for fps_value, g in raw_df.groupby("fps_value"):
        row = {"fps_value": fps_value}
        for col, prefix in metrics.items():
            summary = summarize_with_ci(g[col].values)
            row[f"{prefix}_n"] = summary["n"]
            row[f"{prefix}_mean"] = summary["mean"]
            row[f"{prefix}_std"] = summary["std"]
            row[f"{prefix}_ci95_halfwidth"] = summary["ci95_halfwidth"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fps_value")


def build_allocation_summary(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    wide_rows = []
    for fps_value, g in raw_df.groupby("fps_value"):
        wide_row = {"fps_value": fps_value}
        for key in ALLOCATION_KEYS:
            summary = summarize_with_ci(g[key].values)
            long_rows.append(
                {
                    "fps_value": fps_value,
                    "allocation_var": key,
                    "n": summary["n"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "ci95_halfwidth": summary["ci95_halfwidth"],
                }
            )
            wide_row[f"{key}_n"] = summary["n"]
            wide_row[f"{key}_mean"] = summary["mean"]
            wide_row[f"{key}_std"] = summary["std"]
            wide_row[f"{key}_ci95_halfwidth"] = summary["ci95_halfwidth"]
        wide_rows.append(wide_row)
    long_df = pd.DataFrame(long_rows).sort_values(["fps_value", "allocation_var"])
    wide_df = pd.DataFrame(wide_rows).sort_values("fps_value")
    return long_df, wide_df


def build_stability_summary(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    summary_rows = []
    for fps_value, g in raw_df.groupby("fps_value"):
        alloc_df = g[["seed"] + ALLOCATION_KEYS].dropna()
        if alloc_df.empty:
            continue

        x = alloc_df[ALLOCATION_KEYS].to_numpy(dtype=float)
        mean_alloc = np.mean(x, axis=0)
        l1_distances = np.sum(np.abs(x - mean_alloc), axis=1)

        for seed, l1 in zip(alloc_df["seed"].tolist(), l1_distances.tolist()):
            per_seed_rows.append(
                {
                    "fps_value": fps_value,
                    "seed": int(seed),
                    "l1_to_mean_allocation": float(l1),
                    "l1_to_mean_allocation_per_dim": float(l1 / len(ALLOCATION_KEYS)),
                }
            )

        summary = summarize_with_ci(l1_distances)
        summary_rows.append(
            {
                "fps_value": fps_value,
                "n": summary["n"],
                "l1_mean": summary["mean"],
                "l1_std": summary["std"],
                "l1_ci95_halfwidth": summary["ci95_halfwidth"],
                "l1_per_dim_mean": summary["mean"] / len(ALLOCATION_KEYS),
                "l1_per_dim_std": summary["std"] / len(ALLOCATION_KEYS)
                if np.isfinite(summary["std"])
                else np.nan,
                "l1_per_dim_ci95_halfwidth": summary["ci95_halfwidth"] / len(ALLOCATION_KEYS)
                if np.isfinite(summary["ci95_halfwidth"])
                else np.nan,
            }
        )

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["fps_value", "seed"])
    summary_df = pd.DataFrame(summary_rows).sort_values("fps_value")
    return per_seed_df, summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate K seed-wise optimization outputs (all_results.pkl) and compute robustness/stability summaries."
    )
    parser.add_argument(
        "--seed_opt_root",
        required=True,
        help="Root folder containing seed_*/all_results.pkl outputs",
    )
    parser.add_argument(
        "--pattern",
        default="**/seed_*/all_results.pkl",
        help="Glob pattern (relative to seed_opt_root) for optimization result files",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder (default: seed_opt_root/aggregated_<timestamp>)",
    )
    args = parser.parse_args()

    search_pattern = os.path.join(args.seed_opt_root, args.pattern)
    files = sorted(glob.glob(search_pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {search_pattern}")

    raw_rows = []
    for path in files:
        seed = parse_seed_from_path(path)
        with open(path, "rb") as f:
            all_results = pickle.load(f)
        if not isinstance(all_results, dict):
            continue
        raw_rows.extend(extract_result_rows(seed=seed, all_results=all_results, source_path=path))

    if not raw_rows:
        raise RuntimeError("No valid optimization rows were extracted from all_results.pkl files.")

    raw_df = pd.DataFrame(raw_rows)
    raw_df = raw_df.sort_values(["fps_value", "seed"])

    out_dir = args.out or os.path.join(
        args.seed_opt_root,
        f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    raw_df.to_csv(os.path.join(out_dir, "raw_seed_optimizer_results.csv"), index=False)

    perf_df = build_performance_summary(raw_df)
    perf_df.to_csv(os.path.join(out_dir, "performance_robustness_by_fps.csv"), index=False)

    alloc_long_df, alloc_wide_df = build_allocation_summary(raw_df)
    alloc_long_df.to_csv(
        os.path.join(out_dir, "allocation_robustness_by_fps_long.csv"), index=False
    )
    alloc_wide_df.to_csv(
        os.path.join(out_dir, "allocation_robustness_by_fps_wide.csv"), index=False
    )

    l1_per_seed_df, l1_summary_df = build_stability_summary(raw_df)
    l1_per_seed_df.to_csv(
        os.path.join(out_dir, "allocation_l1_distance_per_seed.csv"), index=False
    )
    l1_summary_df.to_csv(os.path.join(out_dir, "allocation_stability_by_fps.csv"), index=False)

    print(f"Aggregated {len(files)} seed result files.")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
