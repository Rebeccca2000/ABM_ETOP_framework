import csv
import shutil
import subprocess
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

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
DETERMINISTIC_TIME_CONTROL = ROOT / "MaaS-Centralised" / "det_fixed_policy_travel_time_equity" / "fixed_policy_3500_seed0" / "fixed_policy_fps_results.csv"
DETERMINISTIC_TSTT_CONTROL = ROOT / "MaaS-Centralised" / "det_fixed_policy_total_system_travel_time" / "fixed_policy_3500_seed0" / "fixed_policy_fps_results.csv"


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


def thousands_formatter(x, pos=None):
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)


def configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "grid.color": "#D7D7D7",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "legend.frameon": False,
        }
    )


SERIES_STYLE = {
    "Mode Share Equity": {"color": "#1F4E79", "marker": "o", "linestyle": "-"},
    "Travel Time Equity": {"color": "#4E7D68", "marker": "s", "linestyle": "--"},
    "Total System Travel Time": {"color": "#A65E2E", "marker": "D", "linestyle": "-."},
}


def apply_fps_axis(ax, xs):
    preferred_ticks = [3500, 4500, 5500, 6500, 7500, 8500, 9500, 15000]
    tick_values = [tick for tick in preferred_ticks if min(xs) <= tick <= max(xs)]
    ax.set_xticks(tick_values if tick_values else xs)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis="x", rotation=0)


def plot_optimal_fps(csv_path, out_path):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    labels = [row["Policy Objective"] for row in rows]
    xs = [parse_float(row["Optimal FPS"]) for row in rows]
    ys = list(range(len(rows)))
    markers = ["o", "s", "D"]
    colors = ["#1F4E79", "#5B8E7D", "#A65E2E"]

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    min_x = min(xs) - 250
    for idx, (x, y, label) in enumerate(zip(xs, ys, labels)):
        ax.hlines(y, min_x, x, color="#BFC7D5", linewidth=2.0, zorder=1)
        ax.scatter(
            x,
            y,
            s=95,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            f"{int(x):,}",
            xy=(x, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=11,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Optimal FPS")
    ax.set_title("Optimal sampled FPS by policy objective")
    ax.grid(True, axis="x")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_xlim(min_x - 150, max(xs) + 700)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_income_shares(csv_path, out_path):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    labels = [row["Policy Objective"] for row in rows]
    low = [parse_float(row["Low income share (%)"]) for row in rows]
    middle = [parse_float(row["Middle income share (%)"]) for row in rows]
    high = [parse_float(row["High income share (%)"]) for row in rows]
    ys = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    ax.barh(ys, low, color="#1F4E79", label="Low income")
    ax.barh(ys, middle, left=low, color="#7A9E7E", label="Middle income")
    ax.barh(
        ys,
        high,
        left=[a + b for a, b in zip(low, middle)],
        color="#C57B57",
        label="High income",
    )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of mean allocation at objective-specific optimum (%)")
    ax.set_title("Income-group composition at each objective-specific optimum")
    ax.grid(True, axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_performance(csv_path, out_path):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    xs = [parse_float(row["FPS"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.7, 5.4))

    ax.axvspan(6500, 9500, color="#E8EDF3", alpha=0.95, zorder=0)
    ax.axvline(7500, color="#6E7782", linestyle="--", linewidth=1.1, zorder=1)
    ax.text(
        8000,
        1.03,
        "Compromise region",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#485766",
        fontweight="bold",
    )

    for label in ("Mode Share Equity", "Travel Time Equity", "Total System Travel Time"):
        style = SERIES_STYLE[label]
        ys = [parse_float(row[label]) for row in rows]
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=2.4,
            marker=style["marker"],
            markersize=5.6,
            linestyle=style["linestyle"],
            label=label,
            zorder=3,
        )

    ax.annotate(
        "Best common sampled FPS",
        xy=(7500, 0.99),
        xytext=(18, -20),
        textcoords="offset points",
        fontsize=9.6,
        color="#374650",
        ha="left",
    )

    ax.set_title("Normalized cross-objective performance across sampled FPS values")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Normalized score (0-1, higher is better)")
    ax.set_ylim(0, 1.06)
    apply_fps_axis(ax, xs)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_marginal_returns(csv_path, out_path):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    xs = [parse_float(row["FPS"]) for row in rows]
    offsets = {
        "Mode Share Equity": -110,
        "Travel Time Equity": 0,
        "Total System Travel Time": 110,
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axhline(0, color="#626D78", linewidth=1.1, linestyle="--", zorder=1)

    for label in ("Mode Share Equity", "Travel Time Equity", "Total System Travel Time"):
        style = SERIES_STYLE[label]
        ys = [parse_float(row[label]) for row in rows]
        shifted_xs = [x + offsets[label] for x in xs]
        ax.vlines(
            shifted_xs,
            [0.0] * len(ys),
            ys,
            color=style["color"],
            linewidth=1.6,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            shifted_xs,
            ys,
            s=34,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=label,
        )

    ax.set_title("Marginal change in normalized score per 1,000 FPS step")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Marginal change in normalized score")
    apply_fps_axis(ax, xs)
    ylim = max(abs(parse_float(row[label])) for row in rows for label in SERIES_STYLE)
    ax.set_ylim(-(ylim + 0.1), ylim + 0.1)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_budget_usage(csv_path, out_path):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    xs = [parse_float(row["FPS"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.7, 5.4))
    ax.axhline(100, color="#7D5A2A", linewidth=1.1, linestyle="--", zorder=1)
    ax.text(
        xs[0],
        100.8,
        "Full nominal budget",
        ha="left",
        va="bottom",
        fontsize=9.4,
        color="#7D5A2A",
    )
    ax.axvspan(9500, 15000, color="#F4EFE7", alpha=0.65, zorder=0)
    ax.text(
        12250,
        18,
        "Unused headroom\nwidens",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#6E5840",
    )

    for label in ("Mode Share Equity", "Travel Time Equity", "Total System Travel Time"):
        style = SERIES_STYLE[label]
        ys = [parse_float(row[label]) for row in rows]
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=2.4,
            marker=style["marker"],
            markersize=5.6,
            linestyle=style["linestyle"],
            label=label,
            zorder=3,
        )

    ax.set_title("Mean realized budget use across sampled FPS values")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Budget used (% of FPS)")
    ax.set_ylim(0, 105)
    apply_fps_axis(ax, xs)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_lines(csv_path, out_path, title, ylabel, columns):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    headers = list(rows[0].keys())
    x_key = headers[0]
    xs = [parse_float(row[x_key]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "--", "-.", ":"]
    for idx, (column, label, color) in enumerate(columns):
        col_key = headers[column - 1] if isinstance(column, int) else column
        ys = [parse_float(row[col_key]) for row in rows]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.2,
            marker=markers[idx % len(markers)],
            markersize=5.0,
            linestyle=linestyles[idx % len(linestyles)],
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("FPS")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=len(columns))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_composite_bars(csv_path, out_path):
    raise NotImplementedError("Use plot_composite_curve or plot_composite_ranking instead.")


def plot_composite_curve(csv_path, out_path):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    xs = [parse_float(row["FPS"]) for row in rows]
    ys = [parse_float(row["Composite score"]) for row in rows]
    best_idx = max(range(len(ys)), key=lambda i: ys[i])

    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    ax.axvspan(6500, 9500, color="#E8ECF2", alpha=0.8, zorder=0)
    ax.plot(xs, ys, color="#1F4E79", linewidth=2.4, marker="o", markersize=5.5, zorder=3)
    ax.scatter(
        xs[best_idx],
        ys[best_idx],
        s=95,
        color="#1F4E79",
        edgecolor="white",
        linewidth=1.0,
        marker="D",
        zorder=4,
    )
    ax.axvline(xs[best_idx], color="#6E6E6E", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_title("Composite cross-objective score across sampled FPS values")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Composite normalized score (0-1)")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="both")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_composite_ranking(csv_path, out_path):
    configure_matplotlib()
    rows = read_csv_rows(csv_path)
    labels = [f"{int(parse_float(row['FPS'])):,}" for row in rows]
    xs = [parse_float(row["Composite score"]) for row in rows]
    ys = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    min_x = 0.0
    max_x = max(xs) + 0.05
    for idx, (x, y, label) in enumerate(zip(xs, ys, labels)):
        color = "#1F4E79" if idx == 0 else "#6F879D"
        ax.hlines(y, min_x, x, color="#C9D2DC", linewidth=2.0, zorder=1)
        ax.scatter(
            x,
            y,
            s=90,
            color=color,
            marker="o" if idx < 3 else "s",
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            f"{x:.3f}",
            xy=(x, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10.5,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Composite normalized score (0-1)")
    ax.set_ylabel("FPS")
    ax.set_title("Composite-score ranking across common sampled FPS values")
    ax.grid(True, axis="x")
    ax.set_xlim(min_x, max_x)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_decision_support(text, out_path):
    configure_matplotlib()
    fig = plt.figure(figsize=(10.4, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(
        0.03,
        0.97,
        text,
        va="top",
        ha="left",
        fontsize=12,
        color="#202020",
        family="DejaVu Sans",
        linespacing=1.35,
        wrap=True,
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def find_saturation_point(rows, metric_key):
    terminal_value = parse_float(rows[-1][metric_key])
    for idx, row in enumerate(rows):
        value = parse_float(row[metric_key])
        if abs(value - terminal_value) > 1e-9:
            continue
        if all(abs(parse_float(later[metric_key]) - terminal_value) <= 1e-9 for later in rows[idx:]):
            return int(parse_float(row["fps_value"]))
    return int(parse_float(rows[-1]["fps_value"]))


def parse_range_bounds(label):
    base = label.split(" ", 1)[0]
    start_str, end_str = base.split("-", 1)
    return int(start_str), int(end_str)


def render_decision_support_panel(policy_data, top_performers, income_share_csv, out_path):
    configure_matplotlib()

    income_rows = read_csv_rows(income_share_csv)
    low_share_values = [parse_float(row["Low income share (%)"]) for row in income_rows]
    income_by_policy = {row["Policy Objective"]: row for row in income_rows}

    det_mode_rows = read_csv_rows(DETERMINISTIC_CONTROL)
    det_time_rows = read_csv_rows(DETERMINISTIC_TIME_CONTROL)
    det_tstt_rows = read_csv_rows(DETERMINISTIC_TSTT_CONTROL)

    mode_sat = find_saturation_point(det_mode_rows, "avg_equity")
    time_sat = find_saturation_point(det_time_rows, "travel_time_equity_index")
    tstt_sat = find_saturation_point(det_tstt_rows, "total_system_travel_time")
    compromise_candidates = sorted(int(row["FPS"]) for row in top_performers[:4])
    compromise_start = compromise_candidates[0]
    compromise_end = compromise_candidates[-1]
    common_best = int(top_performers[0]["FPS"])

    policy_specs = [
        ("Mode Share Equity", "#1F4E79", "o"),
        ("Travel Time Equity", "#5B8E7D", "s"),
        ("Total System Travel Time", "#A65E2E", "D"),
    ]
    y_map = {
        "Mode Share Equity": 2.2,
        "Travel Time Equity": 1.4,
        "Total System Travel Time": 0.6,
    }
    x_min, x_max = 3200, 15300

    fig = plt.figure(figsize=(11.8, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.55], hspace=0.38, wspace=0.24)
    ax_map = fig.add_subplot(gs[0, :])
    ax_alloc = fig.add_subplot(gs[1, 0])
    ax_det = fig.add_subplot(gs[1, 1])
    fig.patch.set_facecolor("white")
    fig.suptitle("Cross-policy decision-support board", fontsize=17, y=0.98)

    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(0.1, 2.75)
    ax_map.axvspan(compromise_start, compromise_end, color="#E9EEF4", alpha=0.9, zorder=0)
    ax_map.text(
        (compromise_start + compromise_end) / 2.0,
        2.63,
        "Compromise region",
        ha="center",
        va="bottom",
        fontsize=10.8,
        color="#415466",
        fontweight="bold",
    )
    ax_map.axvline(common_best, color="#6D7782", linestyle="--", linewidth=1.1, zorder=1)
    ax_map.text(
        common_best,
        0.2,
        "best common\nsampled FPS",
        ha="center",
        va="bottom",
        fontsize=9.4,
        color="#4D5966",
    )

    for policy, color, marker in policy_specs:
        y = y_map[policy]
        start, end = parse_range_bounds(policy_data[policy]["near_best_range"])
        best = int(policy_data[policy]["best_fps"])
        is_plateau = "broad plateau" in policy_data[policy]["near_best_range"]

        ax_map.hlines(y, x_min + 150, x_max - 150, color="#D3D9E1", linewidth=2.0, zorder=0)
        ax_map.hlines(
            y,
            start,
            end,
            color=color,
            linewidth=8.0 if not is_plateau else 9.5,
            alpha=0.82 if not is_plateau else 0.55,
            zorder=2,
        )
        if is_plateau:
            ax_map.hlines(y, start, end, color=color, linewidth=2.0, linestyle="--", zorder=3)
        ax_map.scatter(best, y, s=92, color=color, marker=marker, edgecolor="white", linewidth=1.0, zorder=4)
        ax_map.text(best + 170, y, f"{best:,}", va="center", ha="left", fontsize=10.4, color=color)
        range_label = policy_data[policy]["near_best_range"].replace("-", "–")
        ax_map.text(
            min((start + end) / 2.0, x_max - 1200),
            y + 0.17,
            range_label,
            ha="center",
            va="bottom",
            fontsize=9.1,
            color="#415466" if not is_plateau else "#6D5A48",
        )

    ax_map.set_yticks([y_map[p] for p, _, _ in policy_specs])
    ax_map.set_yticklabels([p for p, _, _ in policy_specs])
    ax_map.set_xticks([3500, 4500, 6500, 7500, 9500, 15000])
    ax_map.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax_map.set_xlabel("FPS")
    ax_map.set_title("A. Objective-specific best points, near-best ranges, and compromise region", fontsize=13.2, pad=10)
    ax_map.grid(True, axis="x")

    ax_alloc.set_title("B. Allocation-share evidence (low-income largest)", fontsize=11.3, pad=8)
    share_order = ["Low income share (%)", "Middle income share (%)", "High income share (%)"]
    share_labels = ["Low", "Middle", "High"]
    share_colors = ["#2E6E65", "#7DA6A1", "#C9D8D5"]
    y_positions = [2, 1, 0]
    lefts = [0.0, 0.0, 0.0]
    objectives_short = {
        "Mode Share Equity": "Mode",
        "Travel Time Equity": "Time",
        "Total System Travel Time": "TSTT",
    }
    objective_names = ["Mode Share Equity", "Travel Time Equity", "Total System Travel Time"]
    for idx, share_key in enumerate(share_order):
        widths = [parse_float(income_by_policy[obj][share_key]) for obj in objective_names]
        ax_alloc.barh(
            y_positions,
            widths,
            left=lefts,
            color=share_colors[idx],
            edgecolor="white",
            height=0.42,
            label=share_labels[idx],
        )
        lefts = [l + w for l, w in zip(lefts, widths)]
    ax_alloc.set_xlim(0, 100)
    ax_alloc.set_yticks(y_positions)
    ax_alloc.set_yticklabels([objectives_short[obj] for obj in objective_names])
    ax_alloc.set_xlabel("Allocation share (%)")
    ax_alloc.set_xticks([0, 25, 50, 75, 100])
    ax_alloc.grid(True, axis="x", alpha=0.5)
    ax_alloc.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=8.6, handlelength=1.2, columnspacing=1.2)
    def normalized_improvement(rows, metric_key):
        xs = [parse_float(row["fps_value"]) for row in rows]
        ys_raw = [parse_float(row[metric_key]) for row in rows]
        start = ys_raw[0]
        end = ys_raw[-1]
        denom = start - end
        if abs(denom) < 1e-12:
            ys = [0.0 for _ in ys_raw]
        else:
            ys = [(start - y) / denom for y in ys_raw]
        return xs, ys

    det_specs = [
        (det_mode_rows, "avg_equity", "Mode", "#1F4E79"),
        (det_time_rows, "travel_time_equity_index", "Time", "#5B8E7D"),
        (det_tstt_rows, "total_system_travel_time", "TSTT", "#A65E2E"),
    ]
    plateau_start = max(time_sat, tstt_sat)
    ax_det.set_title("C. Deterministic-control evidence", fontsize=11.8, pad=8)
    ax_det.axvspan(plateau_start, parse_float(det_tstt_rows[-1]["fps_value"]), color="#F2F2F2", alpha=0.9, zorder=0)
    for rows, metric_key, label, color in det_specs:
        xs, ys = normalized_improvement(rows, metric_key)
        ax_det.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=4.2, label=label)
    ax_det.set_xlim(250, 6650)
    ax_det.set_ylim(-0.02, 1.08)
    ax_det.set_xlabel("FPS")
    ax_det.set_ylabel("Normalized improvement")
    ax_det.set_xticks([300, 2000, 3000, 6500])
    ax_det.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax_det.set_yticks([0.0, 0.5, 1.0])
    ax_det.grid(True, axis="both", alpha=0.45)
    ax_det.legend(loc="upper left", fontsize=8.5, ncol=3, frameon=False, handlelength=1.4, columnspacing=1.0)
    ax_det.text(
        0.04,
        0.11,
        "Near-monotonic until saturation",
        transform=ax_det.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#374650",
    )
    ax_det.text(
        0.74,
        0.18,
        "Plateau / unused\nheadroom",
        transform=ax_det.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        color="#5C6168",
    )
    fig.subplots_adjust(top=0.90, left=0.10, right=0.98, bottom=0.10)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    plot_composite_curve(normalized_csv, OUT_DIR / "performance_tradeoffs.png")
    plot_composite_ranking(OUT_DIR / "top_performing_policies.csv", OUT_DIR / "cross_objective_performers.png")
    plot_normalized_performance(normalized_csv, OUT_DIR / "normalized_performance.png")
    plot_marginal_returns(marginal_csv, OUT_DIR / "marginal_returns_combined.png")
    plot_budget_usage(usage_csv, OUT_DIR / "subsidy_usage_standalone.png")

    render_decision_support_panel(
        policy_data,
        top_performers,
        income_share_csv,
        OUT_DIR / "decision_support_framework_fixed.png",
    )


if __name__ == "__main__":
    main()
