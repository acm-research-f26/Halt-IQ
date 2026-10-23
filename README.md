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
