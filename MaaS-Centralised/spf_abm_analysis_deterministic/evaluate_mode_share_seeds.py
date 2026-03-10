import argparse
import os
import pickle
import sys
import traceback
from copy import deepcopy
from datetime import datetime
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Support both:
# 1) python ./spf_abm_analysis_separate/evaluate_mode_share_seeds.py  (from MaaS-Centralised)
# 2) python -m spf_abm_analysis_separate.evaluate_mode_share_seeds
try:
    from spf_abm_analysis_separate.mode_share_optimization import run_single_simulation
except ModuleNotFoundError:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from mode_share_optimization import run_single_simulation


def t_critical_975(df: int) -> float:
    """
    97.5% two-sided critical value for Student t.
    Uses scipy if available; otherwise falls back to a compact table.
    """
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:
        table = {
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
            25: 2.060,
            30: 2.042,
        }
        if df <= 20:
            return table.get(df, 2.086)
        if df <= 30:
            return 2.042
        return 1.96


def summarize_with_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci95_halfwidth": np.nan}

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n >= 2 else 0.0
    if n >= 2:
        tcrit = t_critical_975(n - 1)
        halfwidth = float(tcrit * std / np.sqrt(n))
    else:
        halfwidth = 0.0
    return {"n": n, "mean": mean, "std": std, "ci95_halfwidth": halfwidth}


def extract_best_allocations(all_results: dict) -> dict:
    """
    all_results: {fps_value: result_dict} from optimization.
    returns: {fps_value: allocations_dict}
    """
    allocations_by_fps = {}
    for fps, result in all_results.items():
        fps_f = float(fps)
        alloc = result.get("optimal_allocations")
        if alloc is None:
            full = result.get("full_results", {})
            alloc = full.get("fixed_allocations")
        if alloc is None:
            raise ValueError(f"Missing optimal_allocations for FPS={fps}")
        allocations_by_fps[fps_f] = alloc
    return allocations_by_fps


def run_seed_evaluation_for_fps(
    base_parameters: dict,
    fps_value: float,
    allocations: dict,
    seeds: list[int],
    simulation_steps: int,
    num_cpus_within_job: int = 1,
    deterministic_mode_override=None,
    deterministic_scenario_path=None,
) -> pd.DataFrame:
    """
    Runs simulations for one FPS across multiple seeds while keeping policy fixed.
    """
    param_sets = []
    for s in seeds:
        sim_params = deepcopy(base_parameters)
        sim_params["fps_value"] = float(fps_value)
        sim_params["fixed_allocations"] = allocations
        sim_params["simulation_steps"] = int(simulation_steps)
        sim_params["seed"] = int(s)
        if deterministic_mode_override is not None:
            sim_params["deterministic_mode"] = bool(deterministic_mode_override)
        if deterministic_scenario_path:
            sim_params["deterministic_scenario_path"] = deterministic_scenario_path

        # Disable pilot early-stop during seed robustness evaluation.
        sim_params.pop("pilot_steps", None)
        sim_params.pop("early_stop_best_equity", None)
        sim_params.pop("early_stop_margin", None)
        sim_params.pop("early_stop_exhaust_frac", None)
        param_sets.append(sim_params)

    if num_cpus_within_job <= 1:
        results = [run_single_simulation(p) for p in param_sets]
    else:
        with mp.Pool(processes=min(num_cpus_within_job, len(param_sets))) as pool:
            results = list(pool.map(run_single_simulation, param_sets))

    rows = []
    for r in results:
        if not isinstance(r, dict):
            continue
        trip_diag = r.get("trip_diagnostics", {})
        trip_by_income = trip_diag.get("by_income", {}) if isinstance(trip_diag, dict) else {}
        rows.append(
            {
                "fps_value": float(r.get("fps_value", fps_value)),
                "seed": int(r.get("seed", -1)),
                "final_seed": int(r.get("final_seed", -1)),
                "schema_name": r.get("schema_name", ""),
                "deterministic_mode": bool(r.get("deterministic_mode", base_parameters.get("deterministic_mode", False))),
                "total_equity_indicator": float(r.get("total_equity_indicator", np.nan)),
                "equity_low": float(r.get("low", {}).get("equity_indicator", np.nan)),
                "equity_middle": float(r.get("middle", {}).get("equity_indicator", np.nan)),
                "equity_high": float(r.get("high", {}).get("equity_indicator", np.nan)),
                "total_trips_recorded": float(r.get("total_trips", np.nan)),
                "trips_low": float(r.get("low", {}).get("trips", np.nan)),
                "trips_middle": float(r.get("middle", {}).get("trips", np.nan)),
                "trips_high": float(r.get("high", {}).get("trips", np.nan)),
                "budget_used_pct": float(
                    r.get("subsidy_usage", {}).get("percentage_used", np.nan)
                ),
                "total_subsidy_used": float(
                    r.get("subsidy_usage", {}).get("total_subsidy_used", np.nan)
                ),
                "trip_records_total": float(trip_diag.get("total_records", np.nan)),
                "trip_records_finished": float(trip_diag.get("finished_records", np.nan)),
                "trip_records_expired": float(trip_diag.get("expired_records", np.nan)),
                "trip_records_service_selected": float(
                    trip_diag.get("service_selected_records", np.nan)
                ),
                "trip_records_low": float(trip_by_income.get("low", np.nan)),
                "trip_records_middle": float(trip_by_income.get("middle", np.nan)),
                "trip_records_high": float(trip_by_income.get("high", np.nan)),
                "terminated_early": bool(r.get("terminated_early", False)),
                "termination_reason": r.get("termination_reason", ""),
            }
        )

    return pd.DataFrame(rows)


def make_errorbar_plot(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    yerr_col: str,
    title: str,
    ylabel: str,
    outpath: str,
):
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        summary_df[x_col].values,
        summary_df[y_col].values,
        yerr=summary_df[yerr_col].values,
        fmt="o-",
        capsize=4,
    )
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
        description="Evaluate fixed optimized allocations under multiple random seeds."
    )
    parser.add_argument(
        "--opt_dir",
        required=True,
        help="Folder containing all_results.pkl from optimization",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of seeds")
    parser.add_argument("--seed0", type=int, default=0, help="Starting seed id")
    parser.add_argument(
        "--steps", type=int, default=120, help="Simulation steps per seed run"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder for seed evaluation results",
    )
    parser.add_argument(
        "--within_job_cpus",
        type=int,
        default=1,
        help="Parallel simulations to run within a single job",
    )
    parser.add_argument(
        "--fps_subset",
        default=None,
        help="Comma-separated FPS subset, e.g. '2000,4000,8000'",
    )
    parser.add_argument(
        "--deterministic_mode",
        action="store_true",
        help="Force deterministic scenario mode for this evaluation run",
    )
    parser.add_argument(
        "--deterministic_scenario",
        default=None,
        help="Override deterministic scenario JSON path",
    )
    args = parser.parse_args()

    all_results_path = os.path.join(args.opt_dir, "all_results.pkl")
    if not os.path.exists(all_results_path):
        raise FileNotFoundError(f"Cannot find {all_results_path}")

    base_params_path = os.path.join(args.opt_dir, "base_parameters.pkl")
    if not os.path.exists(base_params_path):
        raise FileNotFoundError(
            f"Missing {base_params_path}. "
            "Re-run optimization after saving base_parameters.pkl in main()."
        )

    with open(all_results_path, "rb") as f:
        all_results = pickle.load(f)
    with open(base_params_path, "rb") as f:
        base_parameters = pickle.load(f)
    if args.deterministic_mode:
        base_parameters["deterministic_mode"] = True
    if args.deterministic_scenario:
        scenario_path = os.path.abspath(args.deterministic_scenario)
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Deterministic scenario not found: {scenario_path}")
        base_parameters["deterministic_scenario_path"] = scenario_path

    allocations_by_fps = extract_best_allocations(all_results)
    if args.fps_subset:
        wanted = {float(x.strip()) for x in args.fps_subset.split(",") if x.strip()}
        allocations_by_fps = {
            fps: alloc for fps, alloc in allocations_by_fps.items() if fps in wanted
        }
    if not allocations_by_fps:
        raise ValueError("No FPS values selected for seed evaluation.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or f"seed_eval_mode_share_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    seeds = list(range(args.seed0, args.seed0 + args.k))
    all_seed_frames = []

    for fps_value in sorted(allocations_by_fps.keys()):
        print(f"Evaluating FPS={fps_value} with seeds={seeds}")
        df_fps = run_seed_evaluation_for_fps(
            base_parameters=base_parameters,
            fps_value=fps_value,
            allocations=allocations_by_fps[fps_value],
            seeds=seeds,
            simulation_steps=args.steps,
            num_cpus_within_job=args.within_job_cpus,
            deterministic_mode_override=True if args.deterministic_mode else None,
            deterministic_scenario_path=os.path.abspath(args.deterministic_scenario) if args.deterministic_scenario else None,
        )
        df_fps.to_csv(
            os.path.join(out_dir, f"raw_seed_results_fps_{int(fps_value)}.csv"),
            index=False,
        )
        all_seed_frames.append(df_fps)

    raw_df = pd.concat(all_seed_frames, ignore_index=True)
    raw_df.to_csv(os.path.join(out_dir, "raw_seed_results_all_fps.csv"), index=False)

    summary_rows = []
    for fps_value, g in raw_df.groupby("fps_value"):
        total = summarize_with_ci(g["total_equity_indicator"].values)
        low = summarize_with_ci(g["equity_low"].values)
        middle = summarize_with_ci(g["equity_middle"].values)
        high = summarize_with_ci(g["equity_high"].values)
        used = summarize_with_ci(g["budget_used_pct"].values)
        total_trips = summarize_with_ci(g["total_trips_recorded"].values)
        finished_trips = summarize_with_ci(g["trip_records_finished"].values)
        expired_trips = summarize_with_ci(g["trip_records_expired"].values)

        summary_rows.append(
            {
                "fps_value": float(fps_value),
                "n": total["n"],
                "total_mean": total["mean"],
                "total_std": total["std"],
                "total_ci95_halfwidth": total["ci95_halfwidth"],
                "low_mean": low["mean"],
                "low_std": low["std"],
                "low_ci95_halfwidth": low["ci95_halfwidth"],
                "middle_mean": middle["mean"],
                "middle_std": middle["std"],
                "middle_ci95_halfwidth": middle["ci95_halfwidth"],
                "high_mean": high["mean"],
                "high_std": high["std"],
                "high_ci95_halfwidth": high["ci95_halfwidth"],
                "budget_used_mean": used["mean"],
                "budget_used_std": used["std"],
                "budget_used_ci95_halfwidth": used["ci95_halfwidth"],
                "total_trips_mean": total_trips["mean"],
                "total_trips_std": total_trips["std"],
                "total_trips_ci95_halfwidth": total_trips["ci95_halfwidth"],
                "finished_trips_mean": finished_trips["mean"],
                "finished_trips_std": finished_trips["std"],
                "finished_trips_ci95_halfwidth": finished_trips["ci95_halfwidth"],
                "expired_trips_mean": expired_trips["mean"],
                "expired_trips_std": expired_trips["std"],
                "expired_trips_ci95_halfwidth": expired_trips["ci95_halfwidth"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("fps_value")
    summary_df.to_csv(os.path.join(out_dir, "summary_mean_std_ci95.csv"), index=False)

    make_errorbar_plot(
        summary_df,
        "fps_value",
        "total_mean",
        "total_ci95_halfwidth",
        "Total MAE Equity vs FPS (95% CI)",
        "Total MAE Equity",
        os.path.join(out_dir, "equity_total_ci95.png"),
    )
    make_errorbar_plot(
        summary_df,
        "fps_value",
        "low_mean",
        "low_ci95_halfwidth",
        "Low-income Equity Component vs FPS (95% CI)",
        "Low-income MAE",
        os.path.join(out_dir, "equity_low_ci95.png"),
    )
    make_errorbar_plot(
        summary_df,
        "fps_value",
        "middle_mean",
        "middle_ci95_halfwidth",
        "Middle-income Equity Component vs FPS (95% CI)",
        "Middle-income MAE",
        os.path.join(out_dir, "equity_middle_ci95.png"),
    )
    make_errorbar_plot(
        summary_df,
        "fps_value",
        "high_mean",
        "high_ci95_halfwidth",
        "High-income Equity Component vs FPS (95% CI)",
        "High-income MAE",
        os.path.join(out_dir, "equity_high_ci95.png"),
    )
    make_errorbar_plot(
        summary_df,
        "fps_value",
        "budget_used_mean",
        "budget_used_ci95_halfwidth",
        "Budget Used (%) vs FPS (95% CI)",
        "Budget Used (%)",
        os.path.join(out_dir, "budget_used_ci95.png"),
    )

    with open(os.path.join(out_dir, "seed_robustness_report.txt"), "w") as f:
        f.write("Seed Robustness Evaluation (Fixed allocations per FPS)\n")
        f.write("=====================================================\n\n")
        f.write(f"Optimization source folder: {args.opt_dir}\n")
        f.write(f"K seeds: {args.k} (seed0={args.seed0})\n")
        f.write(f"Simulation steps per run: {args.steps}\n\n")
        f.write(f"Deterministic mode override: {args.deterministic_mode}\n")
        f.write(f"Deterministic scenario override: {args.deterministic_scenario}\n\n")
        f.write("Summary table: summary_mean_std_ci95.csv\n")
        f.write("Raw per-seed table: raw_seed_results_all_fps.csv\n")

    print(f"Done. Seed evaluation outputs in: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()
        raise
