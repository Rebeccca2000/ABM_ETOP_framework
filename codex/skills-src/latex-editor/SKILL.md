---
name: latex-editor
description: Edit the LaTeX manuscript safely and conservatively. Use when Codex needs to revise `.tex`, bibliography references, figure and table captions, appendix text, labels, or cross-references while keeping verified numbers and reviewer responses aligned.
---

# Latex Editor

Revise the manuscript without breaking labels, references, or evidence integrity.

## Procedure
1. Confirm the full LaTeX project exists under `paper/`.
2. Locate the exact section, figure, table, appendix, or bibliography entry that must change.
3. Verify the underlying evidence before touching any numbers or claims.
4. Edit conservatively:
   - preserve existing labels when possible
   - update `\ref`, `\label`, and appendix references when structure changes
   - keep tables and captions consistent with the latest assets
5. Check that rebuttal language and manuscript language do not contradict each other.

## Guardrails
- Never change a numeric statement unless the source result was checked in the local workspace.
- Never invent a citation to fill a reviewer request.
- When a citation or source is still missing, mark it in the tracker and unresolved list instead of fabricating one.
- If figure numbering or ordering changes, update every cross-reference affected by the change.
