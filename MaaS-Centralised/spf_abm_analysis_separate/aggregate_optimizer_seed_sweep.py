#!/usr/bin/env python3
"""
Aggregate Bayesian-optimizer outputs across stochastic seeds.

Expected directory structure (flexible):
  seed_opt_runs/**/seed_*/all_results.pkl

Each all_results.pkl is a dict:
  FPS(float) -> result dict with fields:
    - avg_equity (float)
    - equity_scores (dict: low/middle/high)
    - optimal_allocations (dict: allocation vars, e.g., 16 keys)
    - subsidy_usage (dict, ideally includes percentage_used)
    - terminated_early (bool)

Outputs:
  merged_summary_<timestamp>/
    summary_mean_std_ci95.csv
    allocations_mean_std_ci95.csv
    per_seed_fps_table.csv
    allocation_stability_by_fps.csv
    plots/*.png
    aggregation_log.txt
"""

import argparse
import glob
import math
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Stats helpers
# -----------------------------
def _t_critical_95(df: int) -> float:
    """
    Two-sided 95% CI => alpha=0.05, use t_{0.975, df}.
    Prefer SciPy if available; otherwise use a small lookup / fallback.
    """
    if df <= 0:
        return float("nan")
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, df))
    except Exception:
        # Conservative-ish lookup for df 1..30, else approximate ~1.96
        t_table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
            21: 2.080,
            22: 2.074,
            23: 2.069,
            24: 2.064,
            25: 2.060,
            26: 2.056,
            27: 2.052,
            28: 2.048,
            29: 2.045,
            30: 2.042,
        }
        if df in t_table:
            return t_table[df]
        # df>30: normal approx is fine
        return 1.96


def mean_std_ci95_halfwidth(x: List[float]) -> Tuple[float, float, float, int]:
    """Return mean, std (sample), CI95 halfwidth, n (after dropping NaN)."""
    arr = np.array(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    if n >= 2:
        tcrit = _t_critical_95(n - 1)
        half = float(tcrit * std / math.sqrt(n))
    else:
        half = 0.0
    return mean, std, half, n


# -----------------------------
# Parsing helpers
# -----------------------------
SEED_RE = re.compile(r"(?:^|/)(seed_(\d+))(?:/|$)")


def infer_seed_from_path(path: str) -> Optional[int]:
    m = SEED_RE.search(path.replace("\\", "/"))
    if not m:
        return None
    return int(m.group(2))


def safe_get_budget_used_pct(result: dict) -> float:
    """
    Best effort:
      - result['subsidy_usage']['percentage_used']
      - result['full_results']['subsidy_usage']['percentage_used']
      - or NaN
    """
    try:
        su = result.get("subsidy_usage", None)
        if isinstance(su, dict) and "percentage_used" in su:
            return float(su["percentage_used"])
    except Exception:
        pass

    try:
        fr = result.get("full_results", None)
        if isinstance(fr, dict):
            su2 = fr.get("subsidy_usage", None)
            if isinstance(su2, dict) and "percentage_used" in su2:
                return float(su2["percentage_used"])
    except Exception:
        pass

    return float("nan")


# -----------------------------
# Main aggregation
# -----------------------------
@dataclass
class Record:
    seed: int
    fps: float
    avg_equity: float
    low: float
    middle: float
    high: float
    budget_used_pct: float
    terminated_early: bool
    alloc: Dict[str, float]
    source_path: str


def load_all_records(seed_opt_base: str, log_lines: List[str]) -> List[Record]:
    pattern = os.path.join(seed_opt_base, "**", "seed_*", "all_results.pkl")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise FileNotFoundError(f"No all_results.pkl found under: {pattern}")

    records: List[Record] = []
    bad = 0

    for p in paths:
        seed = infer_seed_from_path(p)
        if seed is None:
            log_lines.append(f"[WARN] Could not infer seed from path: {p}")
            bad += 1
            continue

        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            log_lines.append(f"[ERROR] Failed to load pickle {p}: {e}")
            bad += 1
            continue

        if not isinstance(data, dict):
            log_lines.append(f"[ERROR] all_results.pkl not dict at {p} (type={type(data)})")
            bad += 1
            continue

        for fps, r in data.items():
            try:
                fps_f = float(fps)
            except Exception:
                log_lines.append(f"[WARN] Non-float FPS key in {p}: {fps}")
                continue

            if not isinstance(r, dict):
                log_lines.append(f"[WARN] Non-dict result for FPS={fps_f} in {p}")
                continue

            avg_equity = float(r.get("avg_equity", np.nan))

            eq = r.get("equity_scores", {})
            low = float(eq.get("low", np.nan)) if isinstance(eq, dict) else float("nan")
            middle = float(eq.get("middle", np.nan)) if isinstance(eq, dict) else float("nan")
            high = float(eq.get("high", np.nan)) if isinstance(eq, dict) else float("nan")

            budget_used = safe_get_budget_used_pct(r)

            terminated = bool(r.get("terminated_early", False))

            alloc = r.get("optimal_allocations", {})
            if not isinstance(alloc, dict):
                alloc = {}

            # Ensure floats
            alloc_clean = {}
            for k, v in alloc.items():
                try:
                    alloc_clean[str(k)] = float(v)
                except Exception:
                    # keep NaN if weird
                    alloc_clean[str(k)] = float("nan")

            records.append(
                Record(
                    seed=seed,
                    fps=fps_f,
                    avg_equity=avg_equity,
                    low=low,
                    middle=middle,
                    high=high,
                    budget_used_pct=budget_used,
                    terminated_early=terminated,
                    alloc=alloc_clean,
                    source_path=p,
                )
            )

    log_lines.append(f"[INFO] Found {len(paths)} all_results.pkl files")
    log_lines.append(f"[INFO] Parsed {len(records)} (seed,fps) records")
    if bad:
        log_lines.append(f"[WARN] Skipped {bad} files due to errors/seed inference issues")
    return records


def aggregate_by_fps(records: List[Record]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_df: per FPS mean/std/ci for total+components+budget
      alloc_df: per FPS per allocation-key mean/std/ci
      per_seed_df: one row per (seed,fps) with raw values + allocation vector
    """
    # Per-seed-fps raw table
    # Collect all allocation keys across records
    all_alloc_keys = sorted({k for rec in records for k in rec.alloc.keys()})

    rows = []
    for rec in records:
        row = {
            "seed": rec.seed,
            "fps": rec.fps,
            "avg_equity": rec.avg_equity,
            "low": rec.low,
            "middle": rec.middle,
            "high": rec.high,
            "budget_used_pct": rec.budget_used_pct,
            "terminated_early": rec.terminated_early,
            "source_path": rec.source_path,
        }
        for k in all_alloc_keys:
            row[f"alloc__{k}"] = rec.alloc.get(k, np.nan)
        rows.append(row)

    per_seed_df = pd.DataFrame(rows)
    per_seed_df.sort_values(["fps", "seed"], inplace=True)

    # Summary aggregation
    summary_rows = []
    for fps, g in per_seed_df.groupby("fps", sort=True):
        total_m, total_s, total_ci, n = mean_std_ci95_halfwidth(g["avg_equity"].tolist())
        low_m, low_s, low_ci, _ = mean_std_ci95_halfwidth(g["low"].tolist())
        mid_m, mid_s, mid_ci, _ = mean_std_ci95_halfwidth(g["middle"].tolist())
        high_m, high_s, high_ci, _ = mean_std_ci95_halfwidth(g["high"].tolist())
        bud_m, bud_s, bud_ci, _ = mean_std_ci95_halfwidth(g["budget_used_pct"].tolist())

        summary_rows.append(
            {
                "fps_value": float(fps),
                "n": n,
                "total_mean": total_m,
                "total_std": total_s,
                "total_ci95_halfwidth": total_ci,
                "low_mean": low_m,
                "low_std": low_s,
                "low_ci95_halfwidth": low_ci,
                "middle_mean": mid_m,
                "middle_std": mid_s,
                "middle_ci95_halfwidth": mid_ci,
                "high_mean": high_m,
                "high_std": high_s,
                "high_ci95_halfwidth": high_ci,
                "budget_used_mean": bud_m,
                "budget_used_std": bud_s,
                "budget_used_ci95_halfwidth": bud_ci,
                "terminated_early_count": int(g["terminated_early"].sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("fps_value")

    # Allocation aggregation (mean/std/ci per key per FPS)
    alloc_rows = []
    for fps, g in per_seed_df.groupby("fps", sort=True):
        for k in [c for c in per_seed_df.columns if c.startswith("alloc__")]:
            vals = g[k].tolist()
            m, s, ci, n = mean_std_ci95_halfwidth(vals)
            alloc_rows.append(
                {
                    "fps_value": float(fps),
                    "allocation_key": k.replace("alloc__", ""),
                    "n": n,
                    "mean": m,
                    "std": s,
                    "ci95_halfwidth": ci,
                }
            )

    alloc_df = pd.DataFrame(alloc_rows).sort_values(["allocation_key", "fps_value"])
    return summary_df, alloc_df, per_seed_df


# -----------------------------
# Plotting
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_line_ci(df: pd.DataFrame, x: str, y_mean: str, y_ci: str, title: str, ylabel: str, outpath: str):
    xs = df[x].values.astype(float)
    ym = df[y_mean].values.astype(float)
    ci = df[y_ci].values.astype(float)

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ym, marker="o")
    plt.errorbar(xs, ym, yerr=ci, fmt="none", capsize=4)
    plt.xlabel("FPS")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # Use log-x if values span a lot; your FPS seems moderately spaced but safe:
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_alloc_heatmaps(alloc_df: pd.DataFrame, out_dir: str):
    """
    Heatmap of mean allocations by FPS, and std allocations by FPS.
    If many keys, plot will be tall (still useful for paper appendix / diagnostics).
    """
    mean_pivot = alloc_df.pivot(index="allocation_key", columns="fps_value", values="mean")
    std_pivot = alloc_df.pivot(index="allocation_key", columns="fps_value", values="std")

    # Mean heatmap (matplotlib only, no seaborn)
    plt.figure(figsize=(12, max(6, 0.35 * len(mean_pivot.index))))
    plt.imshow(mean_pivot.values, aspect="auto")
    plt.colorbar(label="Mean allocation")
    plt.yticks(range(len(mean_pivot.index)), mean_pivot.index)
    plt.xticks(range(len(mean_pivot.columns)), [f"{v:.0f}" for v in mean_pivot.columns], rotation=45, ha="right")
    plt.title("Mean optimal allocations by FPS (across seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "allocations_mean_heatmap.png"), dpi=300)
    plt.close()

    # Std heatmap
    plt.figure(figsize=(12, max(6, 0.35 * len(std_pivot.index))))
    plt.imshow(std_pivot.values, aspect="auto")
    plt.colorbar(label="Std allocation")
    plt.yticks(range(len(std_pivot.index)), std_pivot.index)
    plt.xticks(range(len(std_pivot.columns)), [f"{v:.0f}" for v in std_pivot.columns], rotation=45, ha="right")
    plt.title("Std of optimal allocations by FPS (across seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "allocations_std_heatmap.png"), dpi=300)
    plt.close()


def allocation_stability(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each FPS:
      - compute mean allocation vector across seeds
      - compute L2 distance per seed from mean
      - return per-FPS summary mean/std/ci for that distance
    """
    alloc_cols = [c for c in per_seed_df.columns if c.startswith("alloc__")]
    if not alloc_cols:
        return pd.DataFrame()

    rows = []
    for fps, g in per_seed_df.groupby("fps", sort=True):
        mat = g[alloc_cols].to_numpy(dtype=float)
        # drop rows with NaN in any allocation col (rare but safe)
        mask = ~np.isnan(mat).any(axis=1)
        mat = mat[mask]
        if mat.shape[0] == 0:
            continue
        mean_vec = np.mean(mat, axis=0)
        dists = np.linalg.norm(mat - mean_vec, axis=1)
        m, s, ci, n = mean_std_ci95_halfwidth(dists.tolist())
        rows.append(
            {
                "fps_value": float(fps),
                "n": n,
                "l2dist_mean": m,
                "l2dist_std": s,
                "l2dist_ci95_halfwidth": ci,
            }
        )

    return pd.DataFrame(rows).sort_values("fps_value")


def plot_stability(stab_df: pd.DataFrame, outpath: str):
    if stab_df.empty:
        return
    plot_line_ci(
        stab_df,
        x="fps_value",
        y_mean="l2dist_mean",
        y_ci="l2dist_ci95_halfwidth",
        title="Optimizer allocation stability vs FPS (L2 distance to mean, 95% CI)",
        ylabel="L2 distance",
        outpath=outpath,
    )


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed_opt_base",
        default="seed_opt_runs",
        help="Base directory containing seed optimization runs (default: seed_opt_runs)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: seed_opt_base/merged_summary_<timestamp>",
    )
    # --- NEW: reviewer-clean labelling / naming ---
    ap.add_argument(
        "--metric_slug",
        default="metric",
        help="Short identifier for filenames (e.g., mode_share_mae, travel_time_equity, total_system_travel_time)",
    )
    ap.add_argument(
        "--objective_label",
        default="Objective",
        help="Y-axis label for the primary objective (e.g., 'Travel time equity index', 'Total system travel time (min)')",
    )
    ap.add_argument(
        "--objective_title",
        default=None,
        help="Plot title for objective vs FPS. If not set, a default will be constructed.",
    )
    ap.add_argument(
        "--component_label_prefix",
        default="Component",
        help="Prefix for component plot y-labels (e.g., 'MAE', 'Deviation', 'Component')",
    )
    ap.add_argument(
        "--write_metric_prefixed_csv",
        action="store_true",
        help="If set, also write metric-prefixed CSV copies (e.g., travel_time_equity__summary_mean_std_ci95.csv).",
    )
    args = ap.parse_args()

    tstamp = time.strftime("%Y%m%d_%H%M%S")
    seed_opt_base = args.seed_opt_base

    if not os.path.isabs(seed_opt_base):
        # assume relative to CWD
        seed_opt_base = os.path.join(os.getcwd(), seed_opt_base)

    if not os.path.isdir(seed_opt_base):
        raise FileNotFoundError(f"seed_opt_base does not exist: {seed_opt_base}")

    out_dir = args.out
    if out_dir is None:
        out_dir = os.path.join(seed_opt_base, f"merged_summary_{tstamp}")
    _ensure_dir(out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plot_dir)

    log_lines: List[str] = []
    log_lines.append(f"[INFO] seed_opt_base: {seed_opt_base}")
    log_lines.append(f"[INFO] out_dir: {out_dir}")

    # Load
    records = load_all_records(seed_opt_base, log_lines)

    # Aggregate
    summary_df, alloc_df, per_seed_df = aggregate_by_fps(records)

    # Save CSVs
    summary_csv = os.path.join(out_dir, "summary_mean_std_ci95.csv")
    alloc_csv = os.path.join(out_dir, "allocations_mean_std_ci95.csv")
    per_seed_csv = os.path.join(out_dir, "per_seed_fps_table.csv")
    summary_df.to_csv(summary_csv, index=False)
    alloc_df.to_csv(alloc_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    metric_slug = args.metric_slug.strip()

    if args.write_metric_prefixed_csv:
        summary_df.to_csv(os.path.join(out_dir, f"{metric_slug}__summary_mean_std_ci95.csv"), index=False)
        alloc_df.to_csv(os.path.join(out_dir, f"{metric_slug}__allocations_mean_std_ci95.csv"), index=False)
        per_seed_df.to_csv(os.path.join(out_dir, f"{metric_slug}__per_seed_fps_table.csv"), index=False)
        log_lines.append("[INFO] Wrote metric-prefixed CSV copies.")

    log_lines.append(f"[INFO] Wrote: {summary_csv}")
    log_lines.append(f"[INFO] Wrote: {alloc_csv}")
    log_lines.append(f"[INFO] Wrote: {per_seed_csv}")

    # -----------------
    # Plots: summary (reviewer-clean)
    # -----------------
    objective_label = args.objective_label.strip()

    if args.objective_title is None:
        objective_title = f"{objective_label} vs FPS (95% CI)"
    else:
        objective_title = args.objective_title.strip()

    # Primary objective plot (always)
    plot_line_ci(
        summary_df,
        "fps_value",
        "total_mean",
        "total_ci95_halfwidth",
        objective_title,
        objective_label,
        os.path.join(plot_dir, f"{metric_slug}__objective_ci95.png"),
    )

    # Budget usage plot (always)
    plot_line_ci(
        summary_df,
        "fps_value",
        "budget_used_mean",
        "budget_used_ci95_halfwidth",
        "Budget used (%) vs FPS (95% CI)",
        "Budget used (%)",
        os.path.join(plot_dir, f"{metric_slug}__budget_used_ci95.png"),
    )

    # Component plots: only if components are not all-NaN (e.g., skip for total travel time)
    def _all_nan(colname: str) -> bool:
        s = summary_df[colname]
        return s.isna().all()

    component_prefix = args.component_label_prefix.strip()

    if (not _all_nan("low_mean")) or (not _all_nan("middle_mean")) or (not _all_nan("high_mean")):
        plot_line_ci(
            summary_df,
            "fps_value",
            "low_mean",
            "low_ci95_halfwidth",
            "Low-income component vs FPS (95% CI)",
            f"Low-income {component_prefix}",
            os.path.join(plot_dir, f"{metric_slug}__low_component_ci95.png"),
        )

        plot_line_ci(
            summary_df,
            "fps_value",
            "middle_mean",
            "middle_ci95_halfwidth",
            "Middle-income component vs FPS (95% CI)",
            f"Middle-income {component_prefix}",
            os.path.join(plot_dir, f"{metric_slug}__middle_component_ci95.png"),
        )

        plot_line_ci(
            summary_df,
            "fps_value",
            "high_mean",
            "high_ci95_halfwidth",
            "High-income component vs FPS (95% CI)",
            f"High-income {component_prefix}",
            os.path.join(plot_dir, f"{metric_slug}__high_component_ci95.png"),
        )
    else:
        log_lines.append("[INFO] Skipped component plots (low/middle/high means are all NaN).")

    # Plots: allocations
    try:
        plot_alloc_heatmaps(alloc_df, plot_dir)
        log_lines.append("[INFO] Allocation heatmaps created")
    except Exception as e:
        log_lines.append(f"[WARN] Allocation heatmaps failed: {e}")

    # Stability metric
    stab_df = allocation_stability(per_seed_df)
    stab_csv = os.path.join(out_dir, "allocation_stability_by_fps.csv")
    stab_df.to_csv(stab_csv, index=False)
    plot_stability(stab_df, os.path.join(plot_dir, "allocation_stability_ci95.png"))
    log_lines.append(f"[INFO] Wrote: {stab_csv}")

    # Write log
    log_path = os.path.join(out_dir, "aggregation_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"Done. Outputs in: {out_dir}")
    print(f"Summary: {summary_csv}")
    print(f"Allocations: {alloc_csv}")
    print(f"Stability: {stab_csv}")
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()
