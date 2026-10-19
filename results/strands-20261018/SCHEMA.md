# Common result schema and comparison units

The three `exports/*-common-schema-results.jsonl` files each contain 140 rows:
20 original questions × seven policies. Five FixedK policies repeat byte-for-byte
in scientific fields across all three exports. The `source` field identifies the
owning replay export; it is not another independent experiment. Do not concatenate
duplicate FixedK rows into an accuracy denominator. Primary independent N = 20.

| Field | Meaning |
| --- | --- |
| `question_id` | Stable `hotpotqa:<source_id>` join key. |
| `split` | Export phase label; it does not make this an official hidden-test result. |
| `policy` | FixedK, head cap10/cap3, or hosted Jev cap10/cap3. |
| `source` | Relative owning replay label. |
| `final_answer` | Actual selected saved short answer. |
| `rounds` | Selected replay stop depth, including the initial draft. |
| `total_tokens` | Reported raw counterfactual prefix tokens, including visited decision calls. Different providers' tokenizers differ. |
| `seconds` | Counterfactual prefix model-request time; not measured live end-to-end latency. |
| `exact_match` | Boolean official-style HotpotQA answer agreement. |
| `token_f1` | Official-style token-overlap score, in [0, 1]. |
| `info_score` | Null: no independent information-score judge was run. |
| `cost_usd` | Known hosted prefix cost; local hardware/electricity unpriced. A local value zero is not total compute cost zero. |

`exports/*-first-draft-logits.jsonl` retains `question_id`, partition, evaluator
correctness label and actual logits. Hosted Jev logits are null because its API
supplies probabilities, not logits. Its observed probabilities/diagnostics are
retained in the independently audited optional analysis. No logits are manufactured.

The canonical primary analysis also contains four predeclared visited-confidence
retention policies not duplicated into these common-schema exports. All 13 primary
aggregate policies, paired comparisons and stop-depth/cost definitions remain in
`analyses/primary-paired-analysis.json`. Retention selects only among already visited
drafts, retains all visited costs, uses earliest ties and cannot rescue a failed
episode through retrospective gold-based selection.

Quality failures/missing episodes remain in planned denominators. Approval
precision is correct approvals / approvals (undefined if none); approval coverage
is approvals / all planned questions. These are separate from answer EM. First
decision approval counts are separate from eventual stopping approvals. AUROC,
Brier and NLL use available observed probabilities and report score coverage.
Paired intervals resample original questions, 5,000 draws, seed 42. Repeated drafts,
policies, aliases, cap3 derivations and repeated FixedK controls do not increase N.

Physical writer generation, real encoder forwards, CPU head scoring, hosted calls,
startup, idle and wall time have separate scopes. Replayed policy prefixes do not
create extra physical writer/encoder calls and do not prove live savings.
