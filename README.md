# ABM_ETOP_framework

Agent-based simulation and policy-analysis code for ABM-ETOP, a framework for evaluating multimodal subsidy allocation policies under equity and efficiency objectives.

## Repository scope

This repository contains the simulation and analysis code used to run:

- mode-share equity optimization
- travel-time equity optimization
- total system travel time optimization
- cross-policy comparison and result aggregation
- deterministic scenario checks for fixed-policy evaluation

The manuscript and reviewer-response folders exist in this workspace for the paper revision process, but they are not the primary code entry points for this repository.

## Main code folders

- `MaaS-Centralised/spf_abm_analysis_separate/`
  Main stochastic ABM analysis and optimization scripts.
- `MaaS-Centralised/spf_abm_analysis_deterministic/`
  Deterministic sandbox for fixed-scenario comparisons.
- `MaaS-Centralised/*results*`, `*seed_opt_runs*`, `det_*`
  Stored outputs from optimization, evaluation, and aggregation runs.
- `cross_policy_analysis_results/`
  Exported cross-policy figures and comparison outputs.
- `Kantana-Job-Script/`
  Job scripts used for larger runs.

## Installation

Create a Python environment and install the declared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Several analysis scripts also import `scikit-learn` and `seaborn`, so install them if they are not already available in your environment:

```bash
pip install scikit-learn seaborn
```

## Core workflows

Run from:

```bash
cd MaaS-Centralised/spf_abm_analysis_separate
```

Main optimization scripts:

- `mode_share_optimization.py`
- `travel_time_equity_optimization.py`
- `total_system_travel_time_optimization.py`

These scripts run the ABM, evaluate candidate subsidy policies, and write timestamped output folders for later aggregation.

Example:

```bash
cd /path/to/ABM_ETOP_framework/MaaS-Centralised/spf_abm_analysis_separate
python mode_share_optimization.py
python travel_time_equity_optimization.py
python total_system_travel_time_optimization.py
```

Post-processing and aggregation scripts in the same folder include:

- `aggregate_optimization_seed_runs.py`
- `aggregate_optimizer_seed_sweep.py`
- `aggregate_results_mae.py`
- `aggregate_travel_time_equity.py`
- `aggregate_results_total_travel_time.py`
- `cross_policy_analysis.py`

## Deterministic sandbox

The deterministic sandbox mirrors the main analysis code while keeping stochastic production runs separate:

```bash
cd /path/to/ABM_ETOP_framework/MaaS-Centralised
python -m spf_abm_analysis_deterministic.mode_share_optimization \
  --deterministic_mode \
  --seed0 0 \
  --steps 144 \
  --num_cpus 10 \
  --out_dir det_mode_share_opt_seed0
```

See [README_DETERMINISTIC.md](/Users/drzry/Desktop/ABM_ETOP_framework/MaaS-Centralised/spf_abm_analysis_deterministic/README_DETERMINISTIC.md) for the deterministic workflow details.

## Important scripts

- `agent_commuter_03.py`
  Commuter agent behavior.
- `agent_service_provider_03.py`
  Service provider behavior.
- `agent_subsidy_pool.py`
  Subsidy pool configuration and allocation logic.
- `run_visualisation_03.py`
  Model execution entry used by optimization scripts.
- `database_01.py`
  Database configuration and logging support.
- `ABM_ETOP_Conceptual_Framework.py`
  Framework figure / conceptual analysis helper.

## Outputs

Typical generated outputs include:

- optimization result folders with timestamped names
- aggregated CSV summaries
- policy comparison figures
- deterministic scenario assets
- visualization exports

Large generated artifacts are already present in this repository in several result directories. If you add new runs, prefer keeping code changes separate from bulky output commits.

## Notes

- The GitHub repository has moved from the old `MaaS-Simulation` location to `ABM_ETOP_framework`.
- The current workspace also contains paper-revision material under `paper/`, `review/`, `Marked_up_revised_manuscript/`, and `Revised_paper_submission/`.
- If you are using this repository only for code and reproducibility, focus on `MaaS-Centralised/`, `cross_policy_analysis_results/`, and `Kantana-Job-Script/`.
