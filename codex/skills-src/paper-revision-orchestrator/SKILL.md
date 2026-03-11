---
name: paper-revision-orchestrator
description: Coordinate manuscript revision work driven by reviewer comments. Use when Codex needs to turn review feedback into atomic issues, map each issue to manuscript sections and evidence files, coordinate figure/table/manuscript updates, and keep the review tracker, change log, and rebuttal synchronized.
---

# Paper Revision Orchestrator

Drive the end-to-end revision workflow for the paper. Start from reviewer feedback, not from isolated manuscript edits, and keep the tracker, change log, rebuttal, and manuscript aligned.

## Workflow
1. Read `review/reviewer_comments.md`, `review/reviewer_response_tracker.csv`, `review/CHANGES_LOG.md`, and `review/unresolved_issues.md`.
2. Split reviewer feedback into atomic issues with stable comment IDs.
3. Classify each issue with one primary type:
   - `methods`
   - `metrics`
   - `reproducibility`
   - `figures`
   - `overstatement`
   - `references`
   - `writing`
4. Map each issue to:
   - manuscript locations
   - evidence files
   - figure or table assets
   - required actions
5. Route work to the narrower skill when needed:
   - use `$results-audit` before changing claims
   - use `$figure-regeneration` before changing figure assets
   - use `$latex-editor` for manuscript edits
   - use `$reviewer-response-tracker` to update the tracker and rebuttal
6. Mark each issue as `addressed`, `partial`, or `unresolved`. Never imply completion when evidence is missing.

## Required Outputs
- Updated `review/reviewer_response_tracker.csv`
- Updated `review/CHANGES_LOG.md`
- Updated `review/response_to_reviewers.md`
- Updated `review/unresolved_issues.md` when any issue cannot be fully resolved

## Guardrails
- Treat the paper, appendix, figures, tables, tracker, and rebuttal as one coupled system.
- Do not rewrite claims before verifying the underlying evidence.
- Prefer merged seed summaries, deterministic comparisons, and fixed-policy seed checks over legacy single-run outputs.
- If `paper/` does not contain the full LaTeX project, stop manuscript editing and request the missing files.
