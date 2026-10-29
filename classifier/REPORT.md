# Strands feature integration: controlled stopping-head ablation

Date: 29 October 2026. Integration base: `6c72991652400dcfec8f464a331efb95d5d1cf8b`. Original study: `5503cabeb2ec0094413b7d491ccb08ec9c991c61`.

## Requested comparison

| Variant | F1 (mean ± seed SD) | EM | Tokens (% of six rounds) | Average rounds |
|---|---:|---:|---:|---:|
| Original head | 0.3588 ± 0.0000 | 0.2125 | 16.61% | 1.00 |
| (a) Strands replaces approval | 0.3588 ± 0.0000 | 0.2125 | 16.61% | 1.00 |
| (b) Strands is sixth input | 0.3588 ± 0.0000 | 0.2125 | 16.61% | 1.00 |

Five seeds, 80 test questions per seed. Every seed of every variant stops at round one for every question. Paired F1 differences vs the original are identically zero, so their question-bootstrap intervals are [0,0]. This means exact agreement on this sample, not proof of population equivalence. The unchanged retraining matches the original weights bit-for-bit for all seeds.

## Fixed-round references under the same complete-round accounting

| Policy | F1 | Tokens (% of six rounds) | Rounds |
|---|---:|---:|---:|
| Fixed 1 | 0.3588 | 16.61% | 1 |
| Fixed 2 | 0.3849 | 33.29% | 2 |
| Fixed 3 | 0.3824 | 49.97% | 3 |
| Fixed 4 | 0.3849 | 66.64% | 4 |
| Fixed 5 | 0.3753 | 83.32% | 5 |
| Fixed 6 | 0.3921 | 100.00% | 6 |

The learned heads match Fixed 1 at the same cached Qwen-token budget. They do not beat it. They save 83.39% of this token proxy versus Fixed 6 while losing 0.0333 F1. No policy threshold was tuned against the test set to manufacture an equal-budget comparison.

## What the Strands input contributes

The replacing models still have five inputs and six parameters. The adding models have six inputs and seven parameters. Strands is a supplied continuous critic score, not a new trainable language model. The added/replaced heads learn different coefficients, but the selected checkpoints do not make different stopping decisions. This null result does not establish that Strands is useless: it shows that changing this input alone is insufficient under the current reward, training data and policy-gradient procedure.

The raw Strands score has AUROC 0.6704 against EM on the 80 test first drafts. This is a separate critic-discrimination diagnostic, using one draft per question. It is not the controller's AUROC, not a training objective, and not a live-system success measure. p_stop is an action probability and must not be relabeled p_correct.

## Cost and trajectory boundaries

Primary figures include the original Qwen writer and Qwen critic input/output tokens through the selected round, for every variant. This holds the reward and proxy cost definition fixed to isolate the feature ablation. Strands/reranker tokens, latency and hardware cost were not supplied; complete integrated-system cost cannot be established. The new variants each require one Strands score per stopped test question in this run, in addition to the recorded Qwen costs under the primary accounting.

The existing notebook uses a different scheduling assumption: its writer-only Fixed 1 and some gate policies omit the final Qwen critic call, while previous Qwen critic calls are retained to provide revision feedback. That is why its token percentages differ. For example, writer-only Fixed 1 costs about 8.31% of six complete rounds, versus 16.61% here for a complete writer–critic round. We do not change the notebook or claim these are interchangeable cost definitions. For replacement, evaluating Strands before the current Qwen critic could avoid that final critic call, but this would still require accounting for Strands overhead and validating a live execution schedule.

The supplied data are six-round Qwen trajectories scored post hoc by Strands, not five-round Strands writer/critic trajectories. A stronger critique could change future drafts in a live loop; prefix replay cannot measure those changes. Existing test results have already been examined historically; a fresh holdout is needed for future model/threshold selection. The original study also found two training IDs overlapping the team's separate holdout; that set must not be claimed untouched for these controllers.

## Training limits and next experiment

All variants use the original simplified group-relative policy gradient: eight sampled paths, 20 epochs, Adam 0.01, gradient norm clip 1, cost coefficient 0.1, validation-only checkpoint selection and 0.5 threshold. One update per sampled group means the probability ratio begins at 1; the clipping expression does not constrain that update. There is no reference-model KL term. Changing optimization, reward coefficient, or threshold would be a separate experiment, not this feature ablation.

A useful follow-up is to test training/validation-only changes that encourage learning the value of another revision, then evaluate on a fresh holdout. Preserve Fixed 1 and writer-only as essential baselines. Keep internal revision stopping separate from ending a full user task when moving to multi-agent orchestration.
