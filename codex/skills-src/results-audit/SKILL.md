---
name: results-audit
description: Audit whether manuscript claims are supported by the current code and result files. Use when a reviewer challenges validity, robustness, optimization design, metric interpretation, stochasticity, allocation constraints, or any claim that may have changed after new experiments and aggregations were added.
---

# Results Audit

Verify claims before prose changes. Produce a claim-status verdict and evidence trail, not just a narrative opinion.

## Audit Checklist
- Identify whether the optimization space is constrained or unconstrained.
- Identify whether the claim relies on single-objective runs or a true multi-objective method.
- Identify whether the evidence comes from one run, repeated seeds, deterministic comparisons, or fixed-policy seed checks.
- Verify metric definitions, scales, units, and comparability.
- Verify whether the reported allocation pattern is observed from results or imposed by bounds.
- Verify whether the current manuscript wording matches the latest code and outputs.

## Procedure
1. Locate the manuscript claim or reviewer concern.
2. Trace it to the relevant code path and result files.
3. Classify the claim as:
   - `supported`
   - `needs revision`
   - `unsupported`
4. Record the evidence files needed for the tracker.
5. Propose exact wording replacements when the claim needs revision.

## Guardrails
- Prefer aggregated seed summaries, deterministic scenario outputs, and confidence intervals over legacy plots.
- Do not let a strong narrative survive if the code or bounds no longer support it.
- When units are missing, trace them through the output tables and code before revising text.
