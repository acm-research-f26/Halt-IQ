# HaltIQ experiment

Run kind: **live_models**. Status: completed.

**HotpotQA distractor, answer-only evaluation.** Complete supplied evidence is given to every writer and critic; gold answers and supporting-fact labels are evaluator-only.

These are local subsets of the public official development split. The local `test` label means held-out validation, not the hidden official test set. No supporting-fact/joint leaderboard metrics or statistical superiority are claimed.

| Arm | Completed / attempted | Answer EM | Answer F1 | Mean rounds | Judge calls | False approvals / approvals | API USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 20 / 20 | 55.0% | 61.3% | 3.00 | 0 | 0 / 0 | 0.000000 |
| llm | 20 / 20 | 65.0% | 71.3% | 1.20 | 24 | 7 / 19 | 0.000000 |
| local-typed | 20 / 20 | 55.0% | 61.3% | 1.10 | 22 | 9 / 19 | 0.000000 |
| openjev | 20 / 20 | 55.0% | 61.3% | 3.00 | 60 | 0 / 0 | 0.000000 |

Success is normalized exact match against the dataset reference, with failed episodes counted as unsuccessful. Critic approval is a stopping decision, not the success label. Token F1 is also saved.

Each round is one writer draft. Judge calls count attempted critic requests; each typed request contains three questions. The last round is reviewed as well. The fixed arm uses self-revision and no critic.

Every arm starts from the same draft; later drafts differ because feedback differs. This is a closed-loop system comparison, not a comparison of critics on identical full trajectories. Per-task arm order rotates. Latency includes loading and local contention; no latency significance claim is made.

First-draft operational tokens are charged to every arm for a standalone cost comparison. Actual experiment tokens count that shared request once. Costs exclude local electricity, taxes, prepaid credit purchases, and other programs.

See `manifest.json` for configuration/provenance, `traces.jsonl` for raw outputs and decisions, `episodes.csv` for paired per-task results, and `summary.json` for all aggregates.

HotpotQA answer EM/F1 follow the official scorer's normalization and yes/no/noanswer rules. This differs from the synthetic diagnostic normalizer. See `metrics.py` in `source_snapshot/`.

Benchmark passages, labels, and their derivatives are from [HotpotQA](https://hotpotqa.github.io/), Yang et al. (EMNLP 2018), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Preparation details and hashes are in `dataset_manifest.json` when a preparation manifest is available.
