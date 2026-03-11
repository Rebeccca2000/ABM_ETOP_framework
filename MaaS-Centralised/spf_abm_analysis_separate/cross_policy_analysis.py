import csv
import shutil
import subprocess
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cross_policy_analysis_results"

POLICY_SPECS = {
    "Mode Share Equity": {
        "slug": "mode_share_equity",
        "summary": ROOT / "MaaS-Centralised" / "mae_seed_opt_runs" / "merged_summary_20260222_122159" / "summary_mean_std_ci95.csv",
        "alloc": ROOT / "MaaS-Centralised" / "mae_seed_opt_runs" / "merged_summary_20260222_122159" / "allocations_mean_std_ci95.csv",
        "objective_label": "Mean objective value",
        "metric_label": "Mode-share equity score",
        "range_mode": "relative",
        "best_for": "Reducing mode-share disparity across income groups",
        "recommended_when": "The primary concern is unequal access to modal options",
        "tradeoff_note": "Best sampled result occurs at higher FPS than the legacy single-run optimum, and the curve remains irregular across seeds.",
    },
    "Travel Time Equity": {
        "slug": "travel_time_equity",
        "summary": ROOT / "MaaS-Centralised" / "Travel_time_equity_seed_opt_runs" / "merged_summary_20260225_161313" / "summary_mean_std_ci95.csv",
        "alloc": ROOT / "MaaS-Centralised" / "Travel_time_equity_seed_opt_runs" / "merged_summary_20260225_161313" / "allocations_mean_std_ci95.csv",
        "objective_label": "Mean objective value",
        "metric_label": "Travel-time equity index",
        "range_mode": "relative",
        "best_for": "Reducing differences in average travel times across income groups",
        "recommended_when": "Temporal burden across income groups is the main policy concern",
        "tradeoff_note": "The seed-based curve remains noisy, so the revised manuscript should emphasize uncertainty bands rather than a single sharp threshold.",
    },
    "Total System Travel Time": {
        "slug": "total_system_travel_time",
        "summary": ROOT / "MaaS-Centralised" / "Total_system_travel_time_seed_opt_runs" / "merged_summary_20260228_133748" / "total_system_travel_time__summary_mean_std_ci95.csv",
        "alloc": ROOT / "MaaS-Centralised" / "Total_system_travel_time_seed_opt_runs" / "merged_summary_20260228_133748" / "total_system_travel_time__allocations_mean_std_ci95.csv",
        "trip": ROOT / "MaaS-Centralised" / "Total_system_travel_time_seed_opt_runs" / "merged_summary_20260228_133748" / "trip_summary_mean_std_ci95.csv",
        "objective_label": "Mean total system travel time (minutes)",
        "metric_label": "Total system travel time (minutes)",
        "range_mode": "absolute",
        "best_for": "Lowering aggregate system travel time",
        "recommended_when": "System-wide efficiency is the priority and a broad higher-FPS plateau is acceptable",
        "tradeoff_note": "The minimum occurs at 7,500 FPS, but much of the higher-FPS range remains on a shallow plateau.",
    },
}

DETERMINISTIC_CONTROL = ROOT / "MaaS-Centralised" / "det_fixed_policy_runs" / "fixed_policy_3500_seed0" / "fixed_policy_fps_results.csv"


def require_tool(name):
    tool = shutil.which(name)
    if tool is None:
        raise RuntimeError(f"Required tool not found in PATH: {name}")
    return tool


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value):
    return float(value.strip())


def best_row(rows):
    return min(rows, key=lambda row: parse_float(row["total_mean"]))


def near_best_range(rows, mode):
    best = parse_float(best_row(rows)["total_mean"])
    if mode == "absolute":
        threshold = best + max(best * 0.03, 250.0)
        selected = [parse_float(row["fps_value"]) for row in rows if parse_float(row["total_mean"]) <= threshold]
    else:
        threshold = best * 1.05
        selected = [parse_float(row["fps_value"]) for row in rows if parse_float(row["total_mean"]) <= threshold]

    selected = sorted(selected)
    if not selected:
        selected = [parse_float(best_row(rows)["fps_value"])]

    start = int(selected[0])
    end = int(selected[-1])
    label = f"{start}-{end}"
    if end - start >= 4000:
        label += " (broad plateau)"
    return label


def summarise_allocations(rows, fps_value):
    allocations = {}
    ci95 = {}
    income_totals = defaultdict(float)

    for row in rows:
        if parse_float(row["fps_value"]) != fps_value:
            continue
        key = row["allocation_key"]
        mean = parse_float(row["mean"])
        allocations[key] = mean
        ci95[key] = parse_float(row["ci95_halfwidth"])
        income_totals[key.split("_", 1)[0]] += mean

    total = sum(income_totals.values())
    shares = {
        income: (income_totals.get(income, 0.0) / total * 100.0 if total else 0.0)
        for income in ("low", "middle", "high")
    }
    return allocations, shares, ci95


def format_metric(policy, row):
    n = int(parse_float(row["n"]))
    mean = parse_float(row["total_mean"])
    ci95 = parse_float(row["total_ci95_halfwidth"])
    if policy == "Total System Travel Time":
        return f"Mean {mean:.2f} min +/- {ci95:.2f} (95% CI, n={n})"
    return f"Mean {mean:.4f} +/- {ci95:.4f} (95% CI, n={n})"


def describe_pattern(allocations, shares):
    top_keys = sorted(allocations.items(), key=lambda item: item[1], reverse=True)[:3]
    top_desc = ", ".join(key for key, _ in top_keys)
    lead_income = max(shares.items(), key=lambda item: item[1])[0]
    return f"Largest combined allocation share goes to {lead_income} income; highest mean allocations: {top_desc}"


def normalize_series(values):
    min_val = min(values.values())
    max_val = max(values.values())
    if max_val == min_val:
        return {fps: 1.0 for fps in values}
    return {fps: 1.0 - ((value - min_val) / (max_val - min_val)) for fps, value in values.items()}


def marginal_improvements(series):
    fps_values = sorted(series)
    result = {}
    prev_fps = None
    prev_value = None
    for fps in fps_values:
        value = series[fps]
        if prev_fps is None:
            result[fps] = 0.0
        else:
            result[fps] = ((value - prev_value) / (fps - prev_fps)) * 1000.0
        prev_fps = fps
        prev_value = value
    return result


def run_gnuplot(script):
    gnuplot = require_tool("gnuplot")
    with tempfile.NamedTemporaryFile("w", suffix=".gp", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        script_path = handle.name
    try:
        subprocess.run([gnuplot, script_path], check=True)
    finally:
        script_file = Path(script_path)
        if script_file.exists():
            script_file.unlink()


def quoted(path):
    return str(path).replace("\\", "/")


def plot_optimal_fps(csv_path, out_path):
    script = f"""
set terminal pngcairo size 1200,720 enhanced font "DejaVu Sans,14"
set output "{quoted(out_path)}"
set datafile separator comma
set title "Optimal FPS by Policy Objective (Seed-Based Summaries)"
set ylabel "Optimal FPS"
set key off
set style fill solid 0.9 border rgb "#444444"
set boxwidth 0.7
set xtics rotate by -18 right
set grid ytics lc rgb "#dddddd"
plot "{quoted(csv_path)}" using 2:xtic(1) with boxes lc rgb "#4C78A8", \
     "" using 0:2:(sprintf("%.0f",$2)) with labels offset 0,1 notitle
"""
    run_gnuplot(script)


def plot_grouped_income_shares(csv_path, out_path):
    script = f"""
set terminal pngcairo size 1200,720 enhanced font "DejaVu Sans,14"
set output "{quoted(out_path)}"
set datafile separator comma
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.9 border -1
set boxwidth 0.9
set title "Mean Allocation Share by Income Group at Each Policy Optimum"
set ylabel "Share of total mean allocation (%)"
set yrange [0:100]
set key outside bottom center horizontal
set xtics rotate by -18 right
set grid ytics lc rgb "#dddddd"
plot "{quoted(csv_path)}" using 2:xtic(1) title "Low income" lc rgb "#1B9E77", \
     "" using 3 title "Middle income" lc rgb "#D95F02", \
     "" using 4 title "High income" lc rgb "#7570B3"
"""
    run_gnuplot(script)


def plot_lines(csv_path, out_path, title, ylabel, columns):
    plots = []
    for index, (column, label, color) in enumerate(columns):
        prefix = "" if index == 0 else ", \\\n     "
        plots.append(
            f'{prefix}"{quoted(csv_path)}" using 1:{column} with linespoints '
            f'linewidth 2 pointtype 7 pointsize 1.1 linecolor rgb "{color}" title "{label}"'
        )
    script = f"""
set terminal pngcairo size 1200,720 enhanced font "DejaVu Sans,14"
set output "{quoted(out_path)}"
set datafile separator comma
set title "{title}"
set xlabel "FPS"
set ylabel "{ylabel}"
set key outside bottom center horizontal
set xtics rotate by -45 right
set grid xtics ytics lc rgb "#dddddd"
plot {"".join(plots)}
"""
    run_gnuplot(script)


def plot_composite_bars(csv_path, out_path):
    script = f"""
set terminal pngcairo size 1200,720 enhanced font "DejaVu Sans,14"
set output "{quoted(out_path)}"
set datafile separator comma
set title "Composite Cross-Objective Score by FPS"
set ylabel "Mean normalized score (0-1)"
set yrange [0:1.05]
set key off
set style fill solid 0.9 border rgb "#444444"
set boxwidth 0.6
set xtics rotate by -45 right
set grid ytics lc rgb "#dddddd"
plot "{quoted(csv_path)}" using 2:xtic(1) with boxes lc rgb "#4C78A8", \
     "" using 0:2:(sprintf("%.3f",$2)) with labels offset 0,1 notitle
"""
    run_gnuplot(script)


def render_decision_support(text, out_path):
    convert = require_tool("convert")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(text)
        text_path = handle.name
    try:
        subprocess.run(
            [
                convert,
                "-background",
                "white",
                "-fill",
                "#202020",
                "-size",
                "1600x1100",
                f"caption:@{text_path}",
                str(out_path),
            ],
            check=True,
        )
    finally:
        text_file = Path(text_path)
        if text_file.exists():
            text_file.unlink()


def main():
    OUT_DIR.mkdir(exist_ok=True)

    policy_data = {}
    for policy_name, spec in POLICY_SPECS.items():
        summary_rows = read_csv_rows(spec["summary"])
        alloc_rows = read_csv_rows(spec["alloc"])
        best = best_row(summary_rows)
        fps_value = parse_float(best["fps_value"])
        allocations, income_shares, ci95 = summarise_allocations(alloc_rows, fps_value)

        trip_rows = None
        if "trip" in spec:
            trip_rows = read_csv_rows(spec["trip"])

        policy_data[policy_name] = {
            "spec": spec,
            "summary_rows": summary_rows,
            "best": best,
            "best_fps": fps_value,
            "allocations": allocations,
            "allocation_ci95": ci95,
            "income_shares": income_shares,
            "trip_rows": trip_rows,
            "near_best_range": near_best_range(summary_rows, spec["range_mode"]),
            "pattern_text": describe_pattern(allocations, income_shares),
        }

    optimal_rows = []
    allocation_rows = []
    income_share_rows = []
    normalized_series_rows = []
    usage_rows = []

    common_fps_sets = [
        {parse_float(row["fps_value"]) for row in data["summary_rows"]}
        for data in policy_data.values()
    ]
    common_fps = sorted(set.intersection(*common_fps_sets))

    normalized_series = {}
    usage_series = {}
    composite_rows = []

    for policy_name, data in policy_data.items():
        best = data["best"]
        best_fps = data["best_fps"]
        income_shares = data["income_shares"]
        spec = data["spec"]

        optimal_rows.append(
            {
                "Policy Objective": policy_name,
                "Optimal FPS": int(best_fps),
                "Performance Metric": format_metric(policy_name, best),
                "Objective Mean": f"{parse_float(best['total_mean']):.6f}",
                "Objective CI95 Halfwidth": f"{parse_float(best['total_ci95_halfwidth']):.6f}",
                "Budget Used Mean (%)": f"{parse_float(best['budget_used_mean']):.6f}",
                "Sample Size": int(parse_float(best["n"])),
            }
        )

        income_share_rows.append(
            {
                "Policy Objective": policy_name,
                "Low income share (%)": f"{income_shares['low']:.6f}",
                "Middle income share (%)": f"{income_shares['middle']:.6f}",
                "High income share (%)": f"{income_shares['high']:.6f}",
            }
        )

        for allocation_key, mean_value in sorted(data["allocations"].items()):
            allocation_rows.append(
            {
                "Policy Objective": policy_name,
                "Optimal FPS": int(best_fps),
                "allocation_key": allocation_key,
                "mean": f"{mean_value:.6f}",
                "ci95_halfwidth": f"{data['allocation_ci95'][allocation_key]:.6f}",
            }
            )

        values = {
            parse_float(row["fps_value"]): parse_float(row["total_mean"])
            for row in data["summary_rows"]
            if parse_float(row["fps_value"]) in common_fps
        }
        normalized = normalize_series(values)
        normalized_series[policy_name] = normalized
        usage_series[policy_name] = {
            parse_float(row["fps_value"]): parse_float(row["budget_used_mean"])
            for row in data["summary_rows"]
            if parse_float(row["fps_value"]) in common_fps
        }

    for fps_value in common_fps:
        row = {"FPS": int(fps_value)}
        composite = 0.0
        for policy_name in POLICY_SPECS:
            score = normalized_series[policy_name][fps_value]
            row[policy_name] = f"{score:.6f}"
            composite += score
        row["Composite score"] = f"{(composite / len(POLICY_SPECS)):.6f}"
        normalized_series_rows.append(row)

    top_performers = sorted(
        normalized_series_rows,
        key=lambda row: parse_float(row["Composite score"]),
        reverse=True,
    )
    for index, row in enumerate(top_performers, start=1):
        composite_rows.append(
            {
                "FPS": row["FPS"],
                "Composite score": row["Composite score"],
                "Rank": index,
                "Mode Share Equity normalized": row["Mode Share Equity"],
                "Travel Time Equity normalized": row["Travel Time Equity"],
                "Total System Travel Time normalized": row["Total System Travel Time"],
            }
        )

    marginal_rows = []
    marginal_series = dict(
        (policy_name, marginal_improvements(series))
        for policy_name, series in normalized_series.items()
    )
    for fps_value in common_fps:
        marginal_rows.append(
            {
                "FPS": int(fps_value),
                "Mode Share Equity": f"{marginal_series['Mode Share Equity'][fps_value]:.6f}",
                "Travel Time Equity": f"{marginal_series['Travel Time Equity'][fps_value]:.6f}",
                "Total System Travel Time": f"{marginal_series['Total System Travel Time'][fps_value]:.6f}",
            }
        )
        usage_rows.append(
            {
                "FPS": int(fps_value),
                "Mode Share Equity": f"{usage_series['Mode Share Equity'][fps_value]:.6f}",
                "Travel Time Equity": f"{usage_series['Travel Time Equity'][fps_value]:.6f}",
                "Total System Travel Time": f"{usage_series['Total System Travel Time'][fps_value]:.6f}",
            }
        )

    recommendation_rows = []
    for policy_name, data in policy_data.items():
        spec = data["spec"]
        recommendation_rows.append(
            {
                "Policy Objective": policy_name,
                "Optimal FPS Range": data["near_best_range"],
                "Best For": spec["best_for"],
                "Key Trade-offs": spec["tradeoff_note"],
                "Recommended When": spec["recommended_when"],
                "Subsidy Allocation Pattern": data["pattern_text"],
            }
        )

    write_csv(
        OUT_DIR / "optimal_fps_comparison.csv",
        optimal_rows,
        [
            "Policy Objective",
            "Optimal FPS",
            "Performance Metric",
            "Objective Mean",
            "Objective CI95 Halfwidth",
            "Budget Used Mean (%)",
            "Sample Size",
        ],
    )
    write_csv(
        OUT_DIR / "optimal_allocations_comparison.csv",
        allocation_rows,
        ["Policy Objective", "Optimal FPS", "allocation_key", "mean", "ci95_halfwidth"],
    )
    write_csv(
        OUT_DIR / "policy_recommendations.csv",
        recommendation_rows,
        [
            "Policy Objective",
            "Optimal FPS Range",
            "Best For",
            "Key Trade-offs",
            "Recommended When",
            "Subsidy Allocation Pattern",
        ],
    )
    write_csv(
        OUT_DIR / "top_performing_policies.csv",
        composite_rows,
        [
            "FPS",
            "Composite score",
            "Rank",
            "Mode Share Equity normalized",
            "Travel Time Equity normalized",
            "Total System Travel Time normalized",
        ],
    )

    normalized_csv = OUT_DIR / "normalized_performance_curves.csv"
    write_csv(
        normalized_csv,
        normalized_series_rows,
        ["FPS", "Mode Share Equity", "Travel Time Equity", "Total System Travel Time", "Composite score"],
    )
    income_share_csv = OUT_DIR / "income_share_at_optima.csv"
    write_csv(
        income_share_csv,
        income_share_rows,
        ["Policy Objective", "Low income share (%)", "Middle income share (%)", "High income share (%)"],
    )
    marginal_csv = OUT_DIR / "marginal_returns_curves.csv"
    write_csv(
        marginal_csv,
        marginal_rows,
        ["FPS", "Mode Share Equity", "Travel Time Equity", "Total System Travel Time"],
    )
    usage_csv = OUT_DIR / "budget_usage_curves.csv"
    write_csv(
        usage_csv,
        usage_rows,
        ["FPS", "Mode Share Equity", "Travel Time Equity", "Total System Travel Time"],
    )

    plot_optimal_fps(OUT_DIR / "optimal_fps_comparison.csv", OUT_DIR / "optimal_fps_comparison.png")
    plot_grouped_income_shares(income_share_csv, OUT_DIR / "cross_policy_income_distribution_subsidies.png")
    plot_composite_bars(normalized_csv, OUT_DIR / "performance_tradeoffs.png")
    plot_composite_bars(OUT_DIR / "top_performing_policies.csv", OUT_DIR / "cross_objective_performers.png")
    plot_lines(
        normalized_csv,
        OUT_DIR / "normalized_performance.png",
        "Normalized Performance by FPS",
        "Normalized score (0-1, higher is better)",
        [
            (2, "Mode Share Equity", "#4C78A8"),
            (3, "Travel Time Equity", "#F58518"),
            (4, "Total System Travel Time", "#54A24B"),
        ],
    )
    plot_lines(
        marginal_csv,
        OUT_DIR / "marginal_returns_combined.png",
        "Marginal Improvement per 1000 FPS",
        "Change in normalized score per 1000 FPS",
        [
            (2, "Mode Share Equity", "#4C78A8"),
            (3, "Travel Time Equity", "#F58518"),
            (4, "Total System Travel Time", "#54A24B"),
        ],
    )
    plot_lines(
        usage_csv,
        OUT_DIR / "subsidy_usage_standalone.png",
        "Mean Budget Usage Across Objectives",
        "Mean budget used (%)",
        [
            (2, "Mode Share Equity", "#4C78A8"),
            (3, "Travel Time Equity", "#F58518"),
            (4, "Total System Travel Time", "#54A24B"),
        ],
    )

    deterministic_rows = read_csv_rows(DETERMINISTIC_CONTROL)
    deterministic_first = deterministic_rows[0]
    deterministic_last = deterministic_rows[-1]
    decision_support_text = textwrap.dedent(
        f"""
        ABM-ETOP decision support summary (seed-based revision)

        Mode Share Equity:
        - lowest sampled mean objective at FPS {int(policy_data['Mode Share Equity']['best_fps'])}
        - near-best tested range: {policy_data['Mode Share Equity']['near_best_range']}

        Travel Time Equity:
        - lowest sampled mean objective at FPS {int(policy_data['Travel Time Equity']['best_fps'])}
        - near-best tested range: {policy_data['Travel Time Equity']['near_best_range']}

        Total System Travel Time:
        - lowest sampled mean objective at FPS {int(policy_data['Total System Travel Time']['best_fps'])}
        - near-best tested range: {policy_data['Total System Travel Time']['near_best_range']}

        Cross-objective synthesis:
        - the best composite normalized score occurs at FPS {top_performers[0]['FPS']}
        - higher-FPS travel-time results form a broad plateau rather than a single sharp optimum
        - the largest combined allocation share at the evidence-backed optima always goes to the low-income group, but the unconstrained bounds allow non-progressive solutions

        Deterministic control interpretation:
        - fixed-policy control starts at objective {parse_float(deterministic_first['avg_equity']):.3f} and falls to {parse_float(deterministic_last['avg_equity']):.3f}
        - under fixed commuters, trip schedules, and background traffic, the response is monotonic or near-monotonic until subsidy usage saturates
        - this supports treating stochastic non-monotonicity as a consequence of simulated randomness, not invalid optimization logic

        Reporting guidance:
        - describe the workflow as three separate single-objective Bayesian optimization sweeps followed by comparative synthesis
        - report native metric units and uncertainty, not a single common numeric scale
        - avoid claiming that progressive allocations are imposed by hard-coded progressive bounds in the current pipeline
        """
    ).strip()
    render_decision_support(decision_support_text, OUT_DIR / "decision_support_framework_fixed.png")


if __name__ == "__main__":
    main()
