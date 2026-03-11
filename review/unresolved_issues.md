# Unresolved Issues

Record any reviewer point that cannot be fully resolved with the current model, data, or time available.

## Status Vocabulary
- `open`
- `partial`
- `resolved`

## Entries

### R1-HR5
- `status`: partial
- `reviewer`: Reviewer 1
- `comment_id`: R1-HR5
- `issue_summary`: Code-availability wording is now in the manuscript, but public visibility and archival-release details were not independently verified from the local environment.
- `why_not_fully_resolved`: The local repository exposes the GitHub remote and supports an accurate repository URL, but not the final release/archival status.
- `current_evidence`: `paper/tex/main.tex`; git remote `origin -> git@github.com:Rebeccca2000/ABM_ETOP_framework.git`
- `planned_rebuttal_position`: Keep the repository URL in the manuscript and finalize any release/archival wording only when that status is confirmed.

### R1-WX1
- `status`: partial
- `reviewer`: Reviewer 1
- `comment_id`: R1-WX1
- `issue_summary`: Model description and internal-check language were improved, but a full external empirical validation is still unavailable.
- `why_not_fully_resolved`: The current revision can support an implementation audit, repeated-seed robustness, and deterministic-control interpretation, but not city-scale external validation.
- `current_evidence`: `paper/tex/main.tex`; `paper/tex/appendix.tex`; merged seed summaries; deterministic fixed-policy results
- `planned_rebuttal_position`: State clearly that the framework is presented as a stylised scenario-analysis tool and that empirical validation is future work.

### R1-WX4
- `status`: partial
- `reviewer`: Reviewer 1
- `comment_id`: R1-WX4
- `issue_summary`: Parameter sourcing is clearer, and the appendix now distinguishes script-defined values from externally calibrated values, but not every behavioural parameter has an external literature citation.
- `why_not_fully_resolved`: The revision prioritised removing unsupported benefit assumptions, documenting audited script values, and recording preserved PBS job-resource requests. A full literature-by-parameter audit was not completed in this run.
- `current_evidence`: `paper/tex/appendix.tex`; audited optimisation scripts; preserved PBS job scripts
- `planned_rebuttal_position`: Emphasize that the revised appendix now distinguishes script-defined implementation parameters from externally calibrated values, and that additional empirical calibration remains future work.

### R1-Q4
- `status`: partial
- `reviewer`: Reviewer 1
- `comment_id`: R1-Q4
- `issue_summary`: The manuscript now reports exact evaluation counts, preserved compute requests, and the measured walltimes recoverable from preserved scheduler outputs, but not every optimisation objective has a complete measured runtime record.
- `why_not_fully_resolved`: Measured walltimes are recoverable for the preserved deterministic jobs and representative total-system seed-array jobs, but not for every objective and every stage of the revised package.
- `current_evidence`: optimisation scripts; `paper/tex/main.tex`; `paper/tex/appendix.tex`; `Kantana-Job-Script/mode_share_optimization.pbs`; `Kantana-Job-Script/mae_sim_array.pbs`; `MaaS-Centralised/tte_det_1fps.o7541950`; `MaaS-Centralised/tte_det_fixed_policy.o7542655`; `MaaS-Centralised/tstt_det_1fps.o7541951`; `MaaS-Centralised/tstt_det_fixed_policy.o7542654`; `MaaS-Centralised/tstt_opt_seed.o7514748.3`; `MaaS-Centralised/tstt_opt_seed.o7514748.7`
- `planned_rebuttal_position`: Report the exact evaluation counts, preserved compute requests, and measured walltimes that are directly recoverable, while avoiding unsupported full-package runtime claims.
