# Research methods

## Two different questions

**Critic quality:** given the same candidate answers, does Strands recognize
correctness as well as another critic? Compare AUROC, approval precision with
coverage, probability error and actual inference cost. This is comparable in
method to [Vedanshi's critic/reranker study](https://github.com/acm-research-f26/Halt-IQ/tree/vedanshi/strands-reranker),
but different checkpoints, questions and evidence presentation can change scores.

**Orchestration quality:** does using the critic to request a useful intervention
improve task completion? This requires a live interaction where a review can
change the next action. Replaying fixed candidates cannot answer that question.

The saved HotpotQA panel had no better later answer, even for an oracle selecting
the best recorded draft. Its equal answer F1 is therefore expected. The metric
control tests also include a wrong-then-correct sequence: early acceptance scores
zero and continuing scores one. This validates metric sensitivity, not a real
model improvement.

## Tool-agent experiment

The τ-bench adapter checks a proposed assistant response or tool call using
only public information. Each checker receives the complete policy and observed
conversation, all available tool names, and the complete schemas for the proposed
tools. Writer and reviewer agents retain every tool schema. This same declared
projection is used for every checker; it keeps the retail policy within the
released model's context window without silently cutting policy or history.
An oversized actual proposal still fails explicitly.
Hidden user instructions, reference actions, evaluation criteria and gold database
states are excluded. The decision model is a binary action checker, not a tool-call
generator. A separate reviewer supplies feedback when a proposal is rejected.

Compare the same writer and reviewer under four controllers: no gate, one fixed
review, Strands gating and Jev gating. Freeze task IDs, domain, language, seeds,
user simulator, scoring method and limits before dispatch. Interactions may
diverge because actions change the conversation; that is the intervention being
tested. A shared seed does not promise identical generated conversations.

Keep official task reward and final environment checks as the primary outcome.
Action-path matching is a diagnostic: a valid alternative tool sequence can still
complete a task. Preserve failed and missing planned trials in a separate
failure-inclusive success rate. Report completion coverage, reviewer/gate calls,
tokens, wall time and known hosted cost; unknown local pricing stays unknown.

The pinned upstream English text release differs from older τ² inputs. Do not
mix versions or silently replace semantic communication scoring. The scripted
official mock-environment check uses its declared database-only task and proves
that correct and incorrect actions receive different rewards. It does not establish
Strands or Jev benchmark quality.

## Model and data boundaries

The existing trained head learned from 60 HotpotQA questions, with 15 for model
selection and 15 for calibration. It did not improve the 20-question final panel.
Those temperatures are not calibrated for policy/tool decisions. New action
checks use an explicitly declared criterion and raw released-head temperature;
adaptation and calibration need separate τ-bench training data.

Expand a new evaluation only after small controlled runs establish valid tool
execution, scoring and physical usage. Select tasks without inspecting outcomes,
report domains/types separately, and keep development distinct from final testing.
A large numerical difference is an observation to test, not a required output.

## Controlled checks and next refinements

The [native sensitivity checks](../experiments/strands/README.md) separate two
capabilities. Strands strongly distinguishes correct from contradictory answers
on two simple facts. On two fictional retail-policy pairs, changing only the
observed order status produces very small confidence changes. At the declared
0.8 threshold it rejects both permitted and prohibited actions. These small
diagnostics identify a weakness to investigate; they are not task-success estimates.

The next controlled study should compare two refinements on separate tasks:

- **Action-specific head training:** learn from permitted/prohibited proposals
  and their observed prerequisites, then calibrate on a disjoint partition.
  This directly targets the weak distinction, but needs reliable labels and can
  overfit repeated task templates. Group related tasks before splitting data.
- **Separate policy checks:** score authorization, required confirmation and
  tool validity separately before routing to a reviewer. This makes rejection
  reasons easier to inspect, but adds inference cost and may reject more valid
  actions. Compare it with the single-check gate on the same task/seed pairs.

Neither refinement has been shown to improve official benchmark completion yet.
