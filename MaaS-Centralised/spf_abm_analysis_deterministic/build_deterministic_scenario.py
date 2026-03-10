import argparse
import os
import pickle
import sys

try:
    from spf_abm_analysis_deterministic.deterministic_scenarios import (
        build_scenario,
        save_scenario,
    )
except ModuleNotFoundError:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from deterministic_scenarios import build_scenario, save_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic scenario (fixed commuters, trips, background traffic)."
    )
    parser.add_argument(
        "--base_params_pkl",
        required=True,
        help="Path to base_parameters.pkl from deterministic optimization folder",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=144,
        help="Simulation steps represented in this scenario",
    )
    parser.add_argument(
        "--scenario_seed",
        type=int,
        default=20260224,
        help="Seed used to generate the fixed scenario",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output scenario JSON path",
    )
    args = parser.parse_args()

    with open(args.base_params_pkl, "rb") as f:
        base_parameters = pickle.load(f)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.out
    if output_path is None:
        output_path = os.path.join(
            script_dir,
            "deterministic_assets",
            f"scenario_seed_{int(args.scenario_seed)}_steps_{int(args.steps)}.json",
        )

    scenario = build_scenario(
        base_parameters=base_parameters,
        scenario_seed=int(args.scenario_seed),
        simulation_steps=int(args.steps),
    )
    save_scenario(scenario, output_path)

    total_trips = sum(len(v) for v in scenario["trip_plan"].values())
    total_bg = sum(len(v) for v in scenario["background_traffic"].values())
    print(f"Wrote deterministic scenario: {output_path}")
    print(f"Commuters: {len(scenario['commuters'])}")
    print(f"Preplanned trips: {total_trips}")
    print(f"Background traffic events: {total_bg}")


if __name__ == "__main__":
    main()
