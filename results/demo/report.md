# HaltIQ experiment

Run kind: **scripted_demo**. Status: completed.

**SCRIPTED SOFTWARE DEMO. No Jev or LLM inference occurred. These numbers validate control flow only.**

| Arm | Completed / attempted | Success | Mean rounds | Judge calls | False approvals / approvals | API USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 / 8 | 75.0% | 3.00 | 0 | 0 / 0 | 0.000000 |
| mock-typed | 8 / 8 | 75.0% | 2.00 | 16 | 1 / 7 | 0.000000 |

Success is normalized exact match against the dataset reference, with failed episodes counted as unsuccessful. Critic approval is a stopping decision, not the success label. Token F1 is also saved.

Each round is one writer draft. Judge calls count attempted critic requests; each typed request contains three questions. The last round is reviewed as well. The fixed arm uses self-revision and no critic.

Every arm starts from the same draft; later drafts differ because feedback differs. This is a closed-loop system comparison, not a comparison of critics on identical full trajectories. Per-task arm order rotates. Latency includes loading and local contention; no latency significance claim is made.

First-draft operational tokens are charged to every arm for a standalone cost comparison. Actual experiment tokens count that shared request once. Costs exclude local electricity, taxes, prepaid credit purchases, and other programs.

See `manifest.json` for configuration/provenance, `traces.jsonl` for raw outputs and decisions, `episodes.csv` for paired per-task results, and `summary.json` for all aggregates.
