# Paired result check

All artifact checks passed for 20 HotpotQA questions.

This derives the one-draft baseline from recorded initial answers. It checks scores against saved gold with the matching scorer and audits counts; it does not independently judge the gold answers or run models.

| System | Answer EM | Answer F1 | Mean drafts | Judge calls | Repairs | Regressions | EM-mismatching approvals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One draft | 11/20 (55.0%) | 61.3% | 1.00 | 0 | — | — | — |
| fixed | 11/20 (55.0%) | 61.3% | 3.00 | 0 | 0 | 0 | — |
| llm | 13/20 (65.0%) | 71.3% | 1.20 | 24 | 2 | 0 | 7/19 |
| local-typed | 11/20 (55.0%) | 61.3% | 1.10 | 22 | 0 | 0 | 9/19 |
| openjev | 11/20 (55.0%) | 61.3% | 3.00 | 60 | 0 | 0 | — |

Repairs change an initial EM=0 answer to final EM=1; regressions do the reverse. These are paired comparisons against the same saved initial drafts.

Verified snapshot hashes, dataset selection, complete paired episodes, CSV/trace agreement, draft reuse, final scores and stopping, arm aggregates, and physical call counts/usage. Hashes and repaired/regressed task IDs are in `analysis.json`.

- Development results are exploratory; no significance or broad superiority claim.
- A repair or regression is defined by answer exact match, not manual factual adjudication.
- A non-matching approved answer can be a wording mismatch; inspect F1 and raw answers.
- Later drafts differ across arms because feedback differs.
- Local typed scores are uncalibrated Qwen outputs, not Jev results.
