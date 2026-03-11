# ABM-ETOP Revision Workspace

## Mission
Revise the ABM-ETOP manuscript with reproducible evidence from the local codebase and result files, and keep the manuscript, figures, appendices, rebuttal, and review tracker synchronized.

## Workspace Map
- `paper/` holds the LaTeX manuscript project and figure assets for the paper.
- `review/` holds reviewer comments, the response tracker, the response letter draft, the change log, and unresolved issues.
- `codex/skills-src/` holds the source-of-truth for custom Codex skills.
- `MaaS-Centralised/`, `Kantana-Job-Script/`, and top-level result folders are the evidence workspace.

## Hard Rules
- Never invent results, citations, parameter sources, manuscript numbers, or code behavior.
- Verify every numeric or methodological claim against local code, tables, or result files before editing prose.
- Prefer seed-based summaries with sample size, mean, standard deviation, and 95% confidence intervals when they exist.
- Never describe a behavior as emergent if it was imposed by optimization bounds, allocation constraints, or scenario design.
- Treat the manuscript, figures, tables, appendices, tracker, and rebuttal letter as one coupled deliverable.
- Regenerate captions, labels, and cross-references whenever a figure or table changes.
- Update `review/reviewer_response_tracker.csv` and `review/CHANGES_LOG.md` after every substantive revision.
- Mark unresolved comments explicitly in `review/unresolved_issues.md` instead of writing around missing evidence.

## Default Workflow
1. Parse reviewer comments into atomic issues with stable comment IDs.
2. Audit whether the current manuscript claims are still supported by the latest code and outputs.
3. Map each issue to manuscript sections, result files, figures, tables, and required actions.
4. Regenerate evidence where needed before editing the manuscript.
5. Edit LaTeX conservatively and preserve labels where possible.
6. Update the reviewer tracker, response letter draft, and change log.
7. Run a consistency pass across the abstract, methods, results, discussion, appendices, figures, tables, and rebuttal.

## Evidence Hierarchy
- Prefer merged seed summaries, fixed-policy seed checks, deterministic comparisons, and sensitivity outputs over single-run artifacts.
- When a claim depends on optimization behavior, inspect both the optimization script and the aggregated outputs.
- When a reviewer questions metric definition, units, or comparability, trace the definition back to the code before revising the manuscript.

## Paper Preconditions
- Do not edit the manuscript until the full LaTeX project is present under `paper/`.
- The paper project must include the main `.tex` file, bibliography files, figures, class/style files, and appendix assets.
- If `paper/` is incomplete, stop and request the missing manuscript files before performing manuscript edits.
