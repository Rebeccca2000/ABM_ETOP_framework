---
name: figure-regeneration
description: Rebuild publication-ready figures from verified result files. Use when plots need to be regenerated for reviewer comments, readability fixes, updated evidence, uncertainty reporting, revised captions, corrected units, or replacement of outdated manuscript figures.
---

# Figure Regeneration

Rebuild figures only from verified source tables or outputs. Treat the figure, caption, axis labels, and manuscript reference as one change set.

## Procedure
1. Identify the manuscript figure or reviewer concern.
2. Locate the verified source CSV, summary table, or result file.
3. Confirm metric definition, unit, and sample basis before plotting.
4. Prefer merged seed summaries and confidence intervals or error bars when available.
5. Regenerate a clean publication-ready asset with readable labels and legends.
6. Update the linked figure path and caption only after the new asset exists.

## Plotting Requirements
- Use readable axis titles and include units where applicable.
- Use color choices and line styles that survive print and grayscale reasonably well.
- Avoid dense legends, cropped labels, and tiny inset text.
- Prefer vector output when practical; otherwise use a high-resolution raster export.
- Make confidence intervals explicit when the data supports them.

## Guardrails
- Never regenerate a figure from stale or ambiguous source data.
- Do not reuse a figure caption if the plotted statistic, aggregation level, or uncertainty treatment changed.
- If the paper directory is missing, prepare the asset and caption text but do not update LaTeX references yet.
