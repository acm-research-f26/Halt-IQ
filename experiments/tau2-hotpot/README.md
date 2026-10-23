# HotpotQA inside Tau²

This is a custom closed-corpus question-answering environment using the native
Tau² orchestrator. It is not an official Tau² leaderboard score. The original
HotpotQA question, distractor passages, titles and sentence numbers are preserved.
The user is deterministic; there is no model-generated user or judge.

The agent searches and reads public passages, proposes an answer with citations,
and submits through a real environment tool. Rejected drafts go to a separate
reviewer before revision. A rejected or missing final submission scores zero.
Draft, review, tool and time limits apply to the whole episode.

Question-only MiniLM top-three retrieval supplies the initial evidence to every
method. The critic uses the partner's question-plus-draft top-three retrieval.
Gold answers and supporting facts are evaluator-only. They never enter retrieval,
model prompts, tool schemas or the public database.

## Qualification and training

The engineering pilot preselects 60 previously unused preparation questions:
40 for weight fitting, 10 for hyperparameter selection, and 10 for calibration.
Four separate questions, two of each type, compare fixed one-draft, fixed
three-draft, released-gate and trained-gate methods. All share the same initial
answers. Subsequent reviews and revisions are live model calls. This small
comparison checks integration, not population-level improvement.

The released Hobson model runs on all four comparison questions before any weight
update. Genuine native features are checked against the same head's CPU logits.
Training updates the existing 1,053,184-parameter pointer head, selects the fixed
learning-rate/epoch grid using tuning questions, and fits separate released and
trained temperatures on calibration questions. A completed proof requires changed
head weights, unchanged frozen files, exact saved-head reload, disjoint questions,
and all required native observations. No RL or adapter fine-tuning is claimed.

The larger prepared design uses 400 fit, 100 tuning and 100 calibration questions,
with a separate balanced comparison cohort selected from measured throughput.
It supports fixed1/2/3/5/10 and released, decomposed answer/evidence, and trained
gates. Ten drafts means one initial answer plus nine revisions. Preparation does
not start the 14-hour study.

## Common evaluation

Every method uses the same [official HotpotQA scoring definitions](https://github.com/hotpotqa/hotpot/blob/fa3a36370899e1d85822de61e58c85ea19993154/hotpot_evaluate_v1.py):
strict normalized answer EM/F1, supporting-fact EM/F1 and joint EM/F1. Joint F1
combines joint precision and recall. It is not the product of the two F1 scores.
Missing or failed submissions remain in the planned denominator.

Critic accuracy and binary F1 are separate from answer quality. Decision reporting
includes precision and coverage, AUROC, Brier and stable raw-logit NLL. One-class
AUROC and zero-approval precision are undefined. Repairs, regressions and paired
question-bootstrap intervals use the same questions; ties and negative results
are retained. Type breakdowns are descriptive, not proof that type has no effect.

## Local execution

The installed offline assets are reused. No hosted API is required. Model work
requires an adopted frozen plan and explicit model-call opt-in. Preparation:

```bash
python -B -m haltiq_tau_hotpot prepare-data --testing-root /absolute/testing --output /new/private/data
python -B -m haltiq_tau_hotpot prepare-machine --testing-root /absolute/testing --data-freeze /new/private/data/data-freeze.json --output /new/private/machine.json
python -B -m haltiq_tau_hotpot --help
```

The supervisor runs collection, native baseline/features, learning/calibration
and comparison without ChatGPT polling. A qualification purpose is restricted to
60 preparation questions, eight heldout questions and three drafts; it cannot
silently become the long study. Status, pause and stop use the supervisor CLI.
All model calls have fsynced pending/result receipts. Uncertain work stops without
retry. Runtime assets, machine paths, operator receipts and routine reports stay
private; publish only research source and reviewed numerical results.

GPU work has one OS lease owner. The writer unloads before Strands loads and vice
versa. Four CPU threads, 7 GiB available RAM, 3 GiB free VRAM, stable startup
reserves and approximately equal work/rest time preserve background use. These
are safe-boundary admission checks, not a hardware utilization quota.
