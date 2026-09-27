# Datasets

The default research environment is now [HotpotQA distractor](hotpotqa/README.md), prepared in `hotpotqa/tasks.jsonl`. Use `python -m haltiq run --environment hotpotqa`. The document below describes the earlier synthetic diagnostics; they remain available via `--environment diagnostic` and the scripted `demo` command.

# HaltIQ diagnostic data

`diagnostic.jsonl` contains 24 original, AI-assisted synthetic QA diagnostics: 8 development items and 16 test items. They were created for this prototype and still require human review. They are not a published benchmark, a representative sample of real user requests, or evidence that any critic improves general reasoning.

All people, objects, places, and events are fictional. Each question is answerable using only its evidence, except for explicitly designated insufficient-evidence questions whose correct answer is `UNKNOWN`. No web lookup is needed. Categories cover relational multi-hop reasoning, latest applicable facts, negation, conjunctive constraints, arithmetic, insufficient evidence, and similar-name distractors. Some categories overlap; `category` names the primary diagnostic target.

Each JSONL line has these fields:

| Field | Purpose |
| --- | --- |
| `id` | Stable item identifier. |
| `split` | `dev` or `test`. |
| `category` | Primary diagnostic target. |
| `question` | Question shown to the writer and critic. |
| `evidence` | Ordered list of facts shown to the writer and critic. |
| `answers` | Acceptable short gold answers; evaluation only. |
| `demo_drafts` | Three scripted candidate answers; offline demonstration only. |
| `demo_scores` | Three matching scripted score dictionaries; offline demonstration only. |

The three scripted score dimensions are `supported`, `complete`, and `relevant`, on a 0–1 scale. They demonstrate the typed-critic interface; they are not predictions, confidence estimates, annotations from a real judge, or measurements of JEV. A high score can deliberately be wrong: `dev-003` and `test-012` begin with incorrect high-scoring drafts to check that false approvals are counted. Other trajectories improve, regress, or remain incorrect, so stopping behavior is visible without paid inference.

In a real run, construct model inputs from an explicit allowlist containing `question` and `evidence` (plus the current draft and permitted prior feedback). Never serialize the complete dataset row into a writer or critic prompt. Neither `answers`, `demo_drafts`, nor `demo_scores` belongs in a real model context. Only the offline scorer may access gold answers.

Use the development split to settle prompts, thresholds, parsing, and resource limits. Freeze those choices before running the test split. Once test examples or results have influenced a choice, treat that test evaluation as exploratory and collect a new held-out set for later confirmation. Because this tiny fixture is readable in the repository, it is a convenience split, not a secure blind evaluation.

Acceptable aliases are deliberately narrow. `UNKNOWN` is the only gold response for an indeterminate question; an unsupported guess is wrong. Return short answers, without reasoning, for the final answer field. Normalized exact match and token F1 are calculated against each accepted answer, taking the best score. Document the exact normalization used by the evaluator before reporting results.

See [the experiment protocol](../Docs/EXPERIMENT.md) for how to use these cases in the Monday prototype.

Real benchmark rows additionally carry `benchmark`, `metric`, and evaluator-only `metadata`. HotpotQA rows use `benchmark: "hotpotqa-distractor"` and `metric: "hotpotqa"`, so the harness selects official answer normalization automatically. Gold supporting-fact annotations stay inside metadata and never enter model prompts.
