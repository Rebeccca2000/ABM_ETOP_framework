---
name: reviewer-response-tracker
description: Maintain the reviewer response matrix, rebuttal draft, and unresolved issue log. Use when Codex needs to record how each reviewer comment was addressed, tie changes to evidence and manuscript locations, and draft exportable responses for the resubmission package.
---

# Reviewer Response Tracker

Keep the formal response package synchronized with the manuscript work.

## Required Records
- `review/reviewer_response_tracker.csv`
- `review/response_to_reviewers.md`
- `review/CHANGES_LOG.md`
- `review/unresolved_issues.md`

## Procedure
1. Use one stable comment ID per atomic reviewer issue.
2. Keep every tracker row complete:
   - reviewer
   - comment ID
   - issue type
   - manuscript locations
   - evidence files
   - figure or table assets
   - action taken
   - status
   - rebuttal draft
   - unresolved reason when needed
3. Keep the prose response grouped by reviewer and comment ID.
4. Distinguish clearly between:
   - `addressed`
   - `partial`
   - `unresolved`
5. Add a brief change-log entry after each substantive edit.

## Guardrails
- Do not mark a comment as addressed unless the manuscript and evidence updates are both complete.
- If a comment is only partially addressed, say so directly and explain what remains.
- Keep rebuttal language factual, specific, and tied to actual manuscript changes.
