#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from spf_abm_analysis_deterministic.total_system_travel_time_optimization import run_single_simulation
except ModuleNotFoundError:
    from total_system_travel_time_optimization import run_single_simulation


def parse_fps_values(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No FPS values parsed from --fps_values.")
    return sorted(set(values))


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_policy_from_opt_dir(opt_dir: str, policy_source_fps: float | None):
    all_results_path = os.path.join(opt_dir, "all_results.pkl")
    if not os.path.exists(all_results_path):
        raise FileNotFoundError(f"Missing optimization results: {all_results_path}")
    all_results = load_pickle(all_results_path)
    if not isinstance(all_results, dict) or not all_results:
        raise RuntimeError(f"Invalid or empty all_results.pkl at: {all_results_path}")

    normalized = {}
    for fps_key, result in all_results.items():
        try:
            fps = float(fps_key)
        except Exception:
            continue
        normalized[fps] = result

    if not normalized:
        raise RuntimeError(f"No numeric FPS keys found in: {all_results_path}")

    if policy_source_fps is not None:
        ref_fps = float(policy_source_fps)
        if ref_fps not in normalized:
            nearest = min(normalized.keys(), key=lambda x: abs(x - ref_fps))
            print(
                f"[WARN] Requested policy_source_fps={ref_fps} not found. "
                f"Using nearest={nearest}."
            )
            ref_fps = nearest
    else:
        ref_fps = min(
            normalized.keys(),
            key=lambda f: float(normalized[f].get("total_system_travel_time", float("inf"))),
        )

    ref_result = normalized[ref_fps]
    allocations = ref_result.get("optimal_allocations")
    if allocations is None and isinstance(ref_result.get("full_results"), dict):
        allocations = ref_result["full_results"].get("fixed_allocations")
    if not isinstance(allocations, dict) or not allocations:
        raise RuntimeError(f"Could not resolve fixed policy allocations at FPS={ref_fps}")

    return ref_fps, allocations


def resolve_base_parameters(opt_dir: str):
    base_params_path = os.path.join(opt_dir, "base_parameters.pkl")
    if not os.path.exists(base_params_path):
        raise FileNotFoundError(f"Missing base_parameters.pkl: {base_params_path}")
    base_parameters = load_pickle(base_params_path)
    if not isinstance(base_parameters, dict):
        raise RuntimeError(f"base_parameters.pkl is not a dict: {base_params_path}")
    return base_parameters


def resolve_scenario_path(
    opt_dir: str,
    base_parameters: dict,
    deterministic_scenario_override: str | None,
):
    if deterministic_scenario_override:
        scenario_path = os.path.abspath(deterministic_scenario_override)
    else:
        scenario_path = base_parameters.get("deterministic_scenario_path")
        if scenario_path:
            scenario_path = os.path.abspath(str(scenario_path))
        else:
            fallback = os.path.join(opt_dir, "deterministic_scenario.json")
            scenario_path = os.path.abspath(fallback)

    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Deterministic scenario JSON not found: {scenario_path}")
    return scenario_path


def build_row(fps: float, result: dict):
    subsidy = result.get("subsidy_usage", {}) if isinstance(result, dict) else {}
    return {
        "fps_value": float(fps),
        "total_system_travel_time": float(result.get("total_system_travel_time", np.nan)),
        "avg_trip_time": float(result.get("avg_trip_time", np.nan)),
        "total_trips": float(result.get("total_trips", np.nan)),
        "low_total_travel_time": float(result.get("low", {}).get("total_travel_time", np.nan)),
        "low_trip_count": float(result.get("low", {}).get("trip_count", np.nan)),
        "low_avg_trip_time": float(result.get("low", {}).get("avg_trip_time", np.nan)),
        "middle_total_travel_time": float(result.get("middle", {}).get("total_travel_time", np.nan)),
        "middle_trip_count": float(result.get("middle", {}).get("trip_count", np.nan)),
        "middle_avg_trip_time": float(result.get("middle", {}).get("avg_trip_time", np.nan)),
        "high_total_travel_time": float(result.get("high", {}).get("total_travel_time", np.nan)),
        "high_trip_count": float(result.get("high", {}).get("trip_count", np.nan)),
        "high_avg_trip_time": float(result.get("high", {}).get("avg_trip_time", np.nan)),
        "total_subsidy_used": float(subsidy.get("total_subsidy_used", np.nan)),
        "budget_used_pct": float(subsidy.get("percentage_used", np.nan)),
        "terminated_early": bool(result.get("terminated_early", False)),
        "termination_reason": str(result.get("termination_reason", "")),
    }


def save_visualizations(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(11, 6))
    plt.plot(
        df["fps_value"],
        df["total_system_travel_time"],
        "o-",
        linewidth=2,
        label="Total System Travel Time",
    )
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("FPS")
    plt.ylabel("Total System Travel Time")
    plt.title("Fixed Policy Deterministic Sweep: Total System Travel Time vs FPS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fixed_policy_total_system_travel_time_vs_fps.png"), dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(
        df["fps_value"],
        df["total_trips"],
        "o-",
        color="#1f77b4",
        label="Total Trips",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("FPS")
    ax1.set_ylabel("Total Trips", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        df["fps_value"],
        df["avg_trip_time"],
        "s--",
        color="#d62728",
        label="Average Trip Time",
    )
    ax2.set_ylabel("Average Trip Time", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("Fixed Policy Deterministic Sweep: Trips and Average Time vs FPS")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "fixed_policy_trips_and_avg_time_vs_fps.png"), dpi=300)
    plt.close(fig)

    # Income-specific total travel time trajectories.
    plt.figure(figsize=(11, 6))
    plt.plot(
        df["fps_value"],
        df["total_system_travel_time"],
        "k--",
        linewidth=2.2,
        label="Total System",
    )
    plt.plot(
        df["fps_value"],
        df["low_total_travel_time"],
        "o-",
        linewidth=1.8,
        label="Low Income",
    )
    plt.plot(
        df["fps_value"],
        df["middle_total_travel_time"],
        "s-",
        linewidth=1.8,
        label="Middle Income",
    )
    plt.plot(
        df["fps_value"],
        df["high_total_travel_time"],
        "^-",
        linewidth=1.8,
        label="High Income",
    )
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("FPS")
    plt.ylabel("Total Travel Time by Income")
    plt.title("Fixed Policy Deterministic Sweep: Total Travel Time by Income vs FPS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "fixed_policy_total_system_travel_time_by_income_vs_fps.png"),
        dpi=300,
    )
    plt.close()


def write_summary(
    out_dir: str,
    ref_fps: float,
    scenario_path: str,
    fps_values: list[float],
    rows_df: pd.DataFrame,
):
    trip_cols = ["total_trips", "low_trip_count", "middle_trip_count", "high_trip_count"]
    unique_trip_signatures = rows_df[trip_cols].drop_duplicates().shape[0]

    with open(os.path.join(out_dir, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"policy_source_fps={ref_fps}\n")
        f.write(f"scenario_path={scenario_path}\n")
        f.write(f"fps_values={','.join(str(v) for v in fps_values)}\n")
        f.write(f"num_fps={len(fps_values)}\n")
        f.write(f"trip_signature_count={unique_trip_signatures}\n")
        f.write("\nKey observations\n")
        f.write(
            f"- total_system_travel_time_min={rows_df['total_system_travel_time'].min():.6f}, "
            f"total_system_travel_time_max={rows_df['total_system_travel_time'].max():.6f}\n"
        )
        f.write(
            f"- total_trips_min={rows_df['total_trips'].min():.0f}, "
            f"total_trips_max={rows_df['total_trips'].max():.0f}\n"
        )
        f.write(
            f"- avg_trip_time_min={rows_df['avg_trip_time'].min():.6f}, "
            f"avg_trip_time_max={rows_df['avg_trip_time'].max():.6f}\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic fixed-policy FPS sweep for total system travel time. "
            "A single subsidy allocation policy is held fixed while FPS changes."
        )
    )
    parser.add_argument(
        "--opt_dir",
        required=True,
        help="Directory containing all_results.pkl and base_parameters.pkl",
    )
    parser.add_argument(
        "--fps_values",
        required=True,
        help="Comma-separated FPS values to evaluate, e.g. '300,600,900,1200,1500'",
    )
    parser.add_argument(
        "--policy_source_fps",
        type=float,
        default=None,
        help=(
            "Use allocations optimized at this FPS as fixed policy. "
            "If omitted, use the best-FPS policy from all_results.pkl."
        ),
    )
    parser.add_argument(
        "--deterministic_scenario",
        default=None,
        help="Override scenario JSON path. If omitted, use base_parameters or opt_dir fallback.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed index for run_single_simulation.")
    parser.add_argument("--steps", type=int, default=144, help="Simulation steps per FPS evaluation.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: <opt_dir>/fixed_policy_fps_<timestamp>",
    )
    args = parser.parse_args()

    fps_values = parse_fps_values(args.fps_values)
    base_parameters = resolve_base_parameters(args.opt_dir)
    ref_fps, fixed_allocations = resolve_policy_from_opt_dir(
        args.opt_dir, args.policy_source_fps
    )
    scenario_path = resolve_scenario_path(
        args.opt_dir, base_parameters, args.deterministic_scenario
    )

    if args.out_dir:
        out_dir = args.out_dir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(args.opt_dir, f"fixed_policy_fps_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    results_by_fps = {}

    for idx, fps_value in enumerate(fps_values, start=1):
        print(f"[{idx}/{len(fps_values)}] Running fixed-policy simulation at FPS={fps_value}")
        sim_params = deepcopy(base_parameters)
        sim_params["fps_value"] = float(fps_value)
        sim_params["fixed_allocations"] = deepcopy(fixed_allocations)
        sim_params["simulation_steps"] = int(args.steps)
        sim_params["seed"] = int(args.seed)
        sim_params["deterministic_mode"] = True
        sim_params["deterministic_scenario_path"] = scenario_path
        sim_params.pop("pilot_steps", None)
        sim_params.pop("early_stop_best_equity", None)
        sim_params.pop("early_stop_margin", None)
        sim_params.pop("early_stop_exhaust_frac", None)

        result = run_single_simulation(sim_params)
        if not isinstance(result, dict):
            raise RuntimeError(f"Simulation failed at FPS={fps_value}: non-dict result")

        results_by_fps[float(fps_value)] = result
        rows.append(build_row(float(fps_value), result))

    df = pd.DataFrame(rows).sort_values("fps_value")
    df.to_csv(os.path.join(out_dir, "fixed_policy_fps_results.csv"), index=False)

    with open(os.path.join(out_dir, "fixed_policy_results.pkl"), "wb") as f:
        pickle.dump(results_by_fps, f)

    with open(os.path.join(out_dir, "fixed_policy_allocations.pkl"), "wb") as f:
        pickle.dump(
            {
                "policy_source_fps": float(ref_fps),
                "fixed_allocations": fixed_allocations,
                "scenario_path": scenario_path,
                "seed": int(args.seed),
                "steps": int(args.steps),
                "fps_values": fps_values,
            },
            f,
        )

    save_visualizations(df, out_dir)
    write_summary(out_dir, ref_fps, scenario_path, fps_values, df)
    print(f"Saved fixed-policy sweep outputs to: {out_dir}")


if __name__ == "__main__":
    main()
