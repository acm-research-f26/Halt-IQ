# Paired result check

All artifact checks passed for 8 HotpotQA questions.

This derives the one-draft baseline from recorded initial answers. It checks scores against saved gold with the matching scorer and audits counts; it does not independently judge the gold answers or run models.

| System | Answer EM | Answer F1 | Mean drafts | Judge calls | Repairs | Regressions | EM-mismatching approvals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One draft | 4/8 (50.0%) | 57.5% | 1.00 | 0 | — | — | — |
| fixed | 4/8 (50.0%) | 57.5% | 3.00 | 0 | 0 | 0 | — |
| llm | 5/8 (62.5%) | 70.0% | 1.38 | 11 | 1 | 0 | 3/7 |
| openjev | 4/8 (50.0%) | 57.5% | 3.00 | 24 | 0 | 0 | — |

Repairs change an initial EM=0 answer to final EM=1; regressions do the reverse. These are paired comparisons against the same saved initial drafts.

Verified snapshot hashes, dataset selection, complete paired episodes, CSV/trace agreement, draft reuse, final scores and stopping, arm aggregates, and physical call counts/usage. Hashes and repaired/regressed task IDs are in `analysis.json`.

- Development results are exploratory; no significance or broad superiority claim.
- A repair or regression is defined by answer exact match, not manual factual adjudication.
- A non-matching approved answer can be a wording mismatch; inspect F1 and raw answers.
- Later drafts differ across arms because feedback differs.
- Local typed scores are uncalibrated Qwen outputs, not Jev results.
- The OpenJev arm uses a separate community model; its scores and results do not establish official Jev performance.
