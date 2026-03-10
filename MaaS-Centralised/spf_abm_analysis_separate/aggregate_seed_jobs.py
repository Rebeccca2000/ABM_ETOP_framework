import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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


def summarize(values) -> tuple[float, float, float, int]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n >= 2 else 0.0
    if n >= 2:
        tcrit = t_critical_975(n - 1)
        ci = float(tcrit * std / np.sqrt(n))
    else:
        ci = 0.0
    return mean, std, ci, n


def errorbar_plot(df, x, y, yerr, title, ylabel, outpath):
    if plt is None:
        return
    plt.figure(figsize=(10, 6))
    plt.errorbar(df[x].values, df[y].values, yerr=df[yerr].values, fmt="o-", capsize=4)
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("FPS")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-seed seed-evaluation job outputs and rebuild CI summaries."
    )
    parser.add_argument(
        "--seed_eval_dir",
        required=True,
        help="Parent folder containing per-seed raw_seed_results_fps_*.csv files",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder for merged summary (default under seed_eval_dir)",
    )
    args = parser.parse_args()

    files = sorted(
        glob.glob(
            os.path.join(args.seed_eval_dir, "**", "raw_seed_results_fps_*.csv"),
            recursive=True,
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No raw_seed_results_fps_*.csv found under {args.seed_eval_dir}"
        )

    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    raw.to_csv(
        os.path.join(args.seed_eval_dir, "raw_seed_results_all_fps_merged.csv"),
        index=False,
    )

    rows = []
    for fps, group in raw.groupby("fps_value"):
        total_m, total_s, total_ci, n = summarize(group["total_equity_indicator"])
        low_m, low_s, low_ci, _ = summarize(group["equity_low"])
        mid_m, mid_s, mid_ci, _ = summarize(group["equity_middle"])
        high_m, high_s, high_ci, _ = summarize(group["equity_high"])
        used_m, used_s, used_ci, _ = summarize(group["budget_used_pct"])
        rows.append(
            {
                "fps_value": fps,
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
                "budget_used_mean": used_m,
                "budget_used_std": used_s,
                "budget_used_ci95_halfwidth": used_ci,
            }
        )

    summary = pd.DataFrame(rows).sort_values("fps_value")
    out_dir = args.out or os.path.join(
        args.seed_eval_dir,
        f"merged_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    summary.to_csv(os.path.join(out_dir, "summary_mean_std_ci95.csv"), index=False)

    if plt is None:
        print("matplotlib not installed: skipping plot generation.")
    else:
        errorbar_plot(
            summary,
            "fps_value",
            "total_mean",
            "total_ci95_halfwidth",
            "Total MAE Equity vs FPS (95% CI)",
            "Total MAE Equity",
            os.path.join(out_dir, "equity_total_ci95.png"),
        )
        errorbar_plot(
            summary,
            "fps_value",
            "low_mean",
            "low_ci95_halfwidth",
            "Low-income Equity Component vs FPS (95% CI)",
            "Low-income MAE",
            os.path.join(out_dir, "equity_low_ci95.png"),
        )
        errorbar_plot(
            summary,
            "fps_value",
            "middle_mean",
            "middle_ci95_halfwidth",
            "Middle-income Equity Component vs FPS (95% CI)",
            "Middle-income MAE",
            os.path.join(out_dir, "equity_middle_ci95.png"),
        )
        errorbar_plot(
            summary,
            "fps_value",
            "high_mean",
            "high_ci95_halfwidth",
            "High-income Equity Component vs FPS (95% CI)",
            "High-income MAE",
            os.path.join(out_dir, "equity_high_ci95.png"),
        )
        errorbar_plot(
            summary,
            "fps_value",
            "budget_used_mean",
            "budget_used_ci95_halfwidth",
            "Budget Used (%) vs FPS (95% CI)",
            "Budget Used (%)",
            os.path.join(out_dir, "budget_used_ci95.png"),
        )

    print(f"Done. Merged outputs in: {out_dir}")


if __name__ == "__main__":
    main()
