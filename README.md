# HaltIQ

Research on using Strands and Jev to decide when an agent should continue,
request a second opinion, or execute its next action.

The current direction is tool-using multi-agent orchestration with
[τ-bench](https://github.com/sierra-research/tau2-bench). The earlier HotpotQA
experiment remains available as a reproducible stopping-controller baseline.
Routine progress reports, presentation drafts and operator logs are kept outside
this repository.

## Research code

| Directory | Purpose |
| --- | --- |
| `haltiq_tau/` | Proposed-action checking and bounded reviewer feedback for τ-bench agents. |
| `haltiq_hotpot/` | Paired partner-method checks with source-rebuilt evidence and the compatible Hobson checkpoint. |
| `haltiq_tau_hotpot/` | Native custom Tau² HotpotQA tools, live revision teams, compatible Hobson features and head learning. |
| `pc_strands/` | Native Strands inference, head training, calibration and saved-draft evaluation. |
| `haltiq/` | Writer/provider adapters, answer metrics and persistent hosted-call budgeting. |
| `configs/` | Explicit model, resource and experiment settings. |
| `experiments/` | Small numerical experiments and controlled checks. |
| `tests/` | Metric, routing, provider and evaluator controls. |
| [HotpotQA data and results](results/strands-20261018/README.md) | Recorded candidates, critic outputs and reusable trained head. |

## Why the earlier answer scores matched

The critics selected from the same saved writer drafts. Every question kept the
same answer score through all ten drafts, so choosing a different draft could
not improve F1. That tests stopping choices, not whether feedback repairs an
answer. The critic judgments themselves were different:

| Critic | Critic accuracy at 0.8 | Critic binary F1 | Correctness ranking (AUROC) |
| --- | ---: | ---: | ---: |
| Original Strands | 50% | 0.500 | 0.380 |
| Fine-tuned Strands | 50% | 0.500 | 0.380 |
| Jev | 85% | 0.870 | 0.915 |

Here, critic accuracy measures whether the checker recognizes a correct first draft.
The writer's answers separately scored 50% EM and 0.54 answer F1. Critic binary
F1 and answer token-overlap F1 measure different things.

These are 20 questions, not a broad model ranking. Vedanshi's reranker study
also measures critic ranking; it does not demonstrate improved revised answers.
Her checkpoint and question set differ from this experiment.

Recompute the saved-answer audit without model calls:

```bash
python -B scripts/audit_trajectory_signal.py
python -B -m unittest tests.test_trajectory_signal -v
```

## HotpotQA method check

The small paired pilot selects ten bridge and ten comparison questions from
Vedanshi's saved first drafts using question IDs, without selecting by correctness.
It rebuilds ordinary evidence from the original dataset, then compares her
3,000-character baseline with MiniLM cross-encoder top-three evidence. The cached
CSV's context column is excluded because a later experiment replaced it with
gold-only evidence.

The pilot uses the compatible Hobson checkpoint, its served temperature, and the
same question format. It reports critic accuracy, binary F1, AUROC and calibration
separately from the unchanged writer's answer scores. Results are paired by
question, with question-type breakdowns and explicit failed-case coverage.
Historical notebook dependencies were unpinned, so this reproduces the declared
method rather than promising identical historical probabilities.

The [shared 20-question observations](experiments/hotpotqa/partner-method-20.json)
reproduced all 20 of the partner's top-three approval decisions. Local critic
accuracy was 60% with baseline evidence and 55% with reranking; AUROC was 0.479
and 0.552 respectively. Paired uncertainty includes no difference, so these
development observations establish no quality gain. All 40 scores were saved;
an in-process cleanup check failed, and separate process-exit checks verified
resource release. The original failure remains in the evidence.
These labels use strict normalized answer EM: a shorter name such as "Bing
Crosby" can fail against a full-name reference despite identifying the same
person. The labels remain unchanged for partner-method comparability; they are
not independent human judgments of semantic correctness.

```bash
python -m haltiq_hotpot --help
python -m unittest tests.test_hotpot_partner_pilot -v
python scripts/verify_partner_method.py verify experiments/hotpotqa/partner-method-20.json
```

`prepare` requires the pinned partner CSV and original dataset; `run` requires
local verified model directories, a frozen cohort hash, a new external output
directory and the shared GPU lease. It defaults to a dry run. Actual inference
requires `--allow-model-calls`, respects resource reserves and a bounded deadline,
and records every physical request without automatic retry.

## τ-bench setup

Use Python 3.12 or 3.13. This adapter checks both the upstream revision and the
actual imported checkout. Install that checkout in editable mode; its English
text scores are separate from older releases.

```bash
python -m pip install -e ".[tau]"
git clone https://github.com/sierra-research/tau2-bench.git ../tau2-bench
git -C ../tau2-bench checkout cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2
python -m pip install -e ../tau2-bench
python -m unittest tests.test_tau_routing tests.test_tau_official -v
```

Set `HALTIQ_TAU_UPSTREAM` to that checkout to enable the optional official
integration tests. The CLI keeps local writer/user generation on the CPU;
GPU generation requires a coordinator that owns the shared study GPU lease.

The writer proposes an action. Strands or Jev checks it before execution. A
rejection routes to a separate reviewer and then a revised proposal. Controls
omit the gate or always request one review. Only accepted actions reach the tools.

Task completion uses the official environment evaluator, including each task's
required checks. Word-overlap F1 is not the τ-bench success metric. Scripted
controls validate the harness; they are not model-performance results.

The [native model checks](experiments/strands/README.md) show strong sensitivity
to simple answer contradictions but weak sensitivity to retail order-status
rules. The four retail cases are controlled diagnostics, not a completed
multi-task benchmark or evidence of improved orchestration.

See [research methods](Docs/STRANDS_STUDY.md) for the comparison design and
[the pilot configuration](configs/tau2/pilot.example.json) for explicit tasks,
models and limits. Real runs need configured providers; no model or paid API is
called merely by importing the package or running its contract tests.

## Unattended local phases

`haltiq_supervisor/` runs reviewed phases locally without asking ChatGPT to poll.
It freezes every declared study package, checks input and completion receipts,
waits for resource admission, forwards pause/stop requests and contains its own
worker process trees. Successful phases advance automatically. Unknown work or
a declared review boundary stops for attention; failed phases are never retried.

```bash
python -B -m haltiq_supervisor inspect --plan /absolute/path/to/local-plan.json
python -B -m unittest tests.test_supervisor_engine tests.test_supervisor_handoff -v
```

Preparation and status commands make no model or Codex requests. Completion and
failure events go to a durable local outbox. An explicitly enabled callback can
attempt one [documented exact-session resume](https://learn.chatgpt.com/docs/non-interactive-mode#resume-a-non-interactive-session)
at chosen phase completions, review boundaries or final completion; it never
uses a polling timer or retries an uncertain
callback. CLI delivery is not yet verified against a live Codex session.

The custom **HotpotQA environment inside Tau²** implements actual search, passage
reading and final-answer submission under the native orchestrator. A separate
reviewer can revise an answer before the gate permits submission. The compatible
Hobson learner updates only its existing pointer head; the encoder and adapter
stay frozen. See [the integration method](experiments/tau2-hotpot/README.md).

A bounded qualification pilot may establish the missing live integration proof.
The full study still requires that real proof before dispatch. Passing offline
controls does not establish training completion or benchmark quality. Existing
static and retail results retain their original scope.
