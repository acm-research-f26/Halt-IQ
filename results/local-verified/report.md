# HaltIQ experiment

Run kind: **live_models**. Status: completed.

Small synthetic diagnostic run. These results do not establish Jev superiority or general benchmark accuracy.

| Arm | Completed / attempted | Success | Mean rounds | Judge calls | False approvals / approvals | API USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 / 8 | 87.5% | 3.00 | 0 | 0 / 0 | 0.000000 |
| llm | 8 / 8 | 100.0% | 1.75 | 14 | 0 / 5 | 0.000000 |
| local-typed | 8 / 8 | 87.5% | 1.00 | 8 | 1 / 8 | 0.000000 |

Success is normalized exact match against the dataset reference, with failed episodes counted as unsuccessful. Critic approval is a stopping decision, not the success label. Token F1 is also saved.

Each round is one writer draft. Judge calls count attempted critic requests; each typed request contains three questions. The last round is reviewed as well. The fixed arm uses self-revision and no critic.

Every arm starts from the same draft; later drafts differ because feedback differs. This is a closed-loop system comparison, not a comparison of critics on identical full trajectories. Per-task arm order rotates. Latency includes loading and local contention; no latency significance claim is made.

First-draft operational tokens are charged to every arm for a standalone cost comparison. Actual experiment tokens count that shared request once. Costs exclude local electricity, taxes, prepaid credit purchases, and other programs.

See `manifest.json` for configuration/provenance, `traces.jsonl` for raw outputs and decisions, `episodes.csv` for paired per-task results, and `summary.json` for all aggregates.
