# Deterministic Sandbox

This folder is an isolated copy of `spf_abm_analysis_separate` for controlled FPS testing.
Production stochastic ABM code stays untouched in the original folder.

## Purpose

Use fixed scenario inputs to compare FPS values under the same:

- commuter cohort
- trip schedule
- background traffic pattern

## Workflow

1. Run optimization in deterministic mode (writes `base_parameters.pkl` and scenario path):

```bash
cd /home/z5247491/MaaS-Simulation/MaaS-Centralised
python -m spf_abm_analysis_deterministic.mode_share_optimization \
  --deterministic_mode \
  --seed0 0 \
  --steps 144 \
  --num_cpus 10 \
  --out_dir det_mode_share_opt_seed0
```

2. (Optional) Build a deterministic scenario explicitly from saved base parameters:

```bash
python -m spf_abm_analysis_deterministic.build_deterministic_scenario \
  --base_params_pkl det_mode_share_opt_seed0/base_parameters.pkl \
  --steps 144 \
  --scenario_seed 20260224 \
  --out spf_abm_analysis_deterministic/deterministic_assets/scenario_seed_20260224_steps_144.json
```

3. Evaluate fixed optimized allocations across seeds with deterministic scenario:

```bash
python -m spf_abm_analysis_deterministic.evaluate_mode_share_seeds \
  --opt_dir det_mode_share_opt_seed0 \
  --k 10 \
  --seed0 0 \
  --steps 144
```

Outputs include trip comparability columns in `raw_seed_results_all_fps.csv`.
