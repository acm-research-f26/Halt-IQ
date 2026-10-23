# HotpotQA inside Tau²

This is a custom closed-corpus question-answering environment using the native
Tau² orchestrator. It is not an official Tau² leaderboard score. The original
HotpotQA question, distractor passages, titles and sentence numbers are preserved.
The user is deterministic; there is no model-generated user or judge.
This local study partitions the public HotpotQA distractor development split into
disjoint preparation and evaluation cohorts; it is not a withheld test-set result.

The agent searches and reads public passages, proposes an answer with citations,
and submits through a real environment tool. Rejected drafts go to a separate
reviewer before revision. In the completed pilot, a rejected or missing final
submission scores zero. The future stopping-policy study instead returns the
last valid draft when its declared revision budget ends. Budget fallback is
separate from gate approval and never rescues a technical failure.
Draft, review, tool and time limits apply to the whole episode.

Question-only MiniLM top-three retrieval supplies the initial evidence to every
method. The critic uses the partner's question-plus-draft top-three retrieval.
The reviewer uses that same draft-dependent retrieval. Revisions still receive
the original writer passages plus the review; newly retrieved passages are not
silently added to the writer's tool view.
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
20 separate development questions, and a balanced comparison cohort of 20, 40 or
60 selected from measured throughput before its answers are generated.
It supports fixed1/2/3/5/10 and released, decomposed answer/evidence, and trained
gates. Ten drafts means one initial answer plus nine revisions. Preparation does
not start the 14-hour study.

The trained gate learns answer correctness. The decomposed gate requires both
released answer and released evidence predictions, with a separate calibration
temperature for each. Evidence labels use exact supporting-fact sets; they do
not establish semantic entailment. The minimum valid learning observations are
320/80/80 of the planned 400/100/100; all technical failures remain reported and
no replacement or correctness-based filtering is allowed. All 20 development
baselines must complete before updates.

The larger study memoizes only identical complete writer/reviewer requests and
native input features. Changed prompts, feedback, criteria or model identity do
not share an incompatible entry. Each feature starts with a genuine native
forward; a changed head is explicitly rescored on the CPU. This provides paired
trajectory prefixes while counting logical routes and actual physical work
separately. It does not measure independent live latency for each arm, and it
cannot guarantee different answer scores. A gate can select an unchanged answer
or reject a correct answer; both outcomes remain visible.
Known completed but invalid writer/reviewer responses are also replayed from
their original bytes and fail the same parser again. They are never resampled
for a later arm. An uncertain request has no reusable cache entry.

## Completed engineering pilot

The [four-question pilot](engineering-pilot-4.json) completed real collection,
untouched Strands inference, head training, calibration and native Tau² comparison.
Fixed1 scored 50% answer EM/F1, Fixed3 scored 25%, and both gates scored 0%.
Nine episodes failed on reviewer output limits and two ended without an accepted
answer at the cap. The nine reviewer failures shared three truncated physical
responses through exact request reuse. All 16 planned episodes remain counted;
no improvement is established.

Training used 40 fitting, 10 tuning and 10 calibration questions, separate from
the four comparison questions. The selected head received nine updates; the two
learning-rate candidates required 18 updates in total. The encoder and adapter
stayed unchanged, and saved-head logits reloaded exactly. Both fitted temperatures
reached the declared upper bound; at the unchanged 0.8 threshold, neither head
approved a first draft. Their probability metrics differ despite tied answer scores.

The original comparison stopped under RAM pressure. Its failed owner and costs
remain recorded; a separate comparison reused the verified preparation without
repeating training. Two questions of each type cannot rule out question-type effects.

Before the larger study, the reviewer uses a predeclared concise response and a
larger output allowance. A synthetic A/B checks termination rather than answer
quality. The future cap policy returns the last valid visited draft, matching the
stopping-policy objective. These changes do not alter the completed pilot.

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
60 preparation questions, four heldout questions and three drafts; it cannot
silently become the long study. Status, pause and stop use the supervisor CLI.
All model calls have fsynced pending/result receipts. Uncertain work stops without
retry. Runtime assets, machine paths, operator receipts and routine reports stay
private; publish only research source and reviewed numerical results.

The main worker has five finite phases: first answers, native features, learning
and calibration, paired comparison, and a model-free audit. Preparation writes
an untimed template; a separate compiler checks genuine pilot qualification,
reviewed source, fresh resource measurements and a full-route forecast. It
reserves 13 hours for model work and one hour for finalization. Missing proof or
an infeasible forecast prevents launch. The compiler itself starts no study.

```bash
python -B -m haltiq_tau_hotpot.planning --help
python -B -m haltiq_tau_hotpot.qualification --help
```

A separate bounded synthetic runtime check measures both criteria at 512, 2,048
and 4,096 tokens, verifies released/trained CPU feature replay and exact response
reuse, and separates cold placement from warm inference. These invented inputs
are timing and integration controls, with no benchmark quality claim. This
check requires the completed engineering pilot first; it cannot train weights,
use fresh benchmark questions or expand into the long study.

GPU work has one OS lease owner. The writer unloads before Strands loads and vice
versa. Four CPU threads, 7 GiB available RAM, 3 GiB free VRAM, stable startup
reserves and approximately equal work/rest time preserve background use. These
are safe-boundary admission checks, not a hardware utilization quota.

## Next architectural tests

- **Let the controller request more evidence.** Add a choice between reading
  another passage, requesting a review and submitting. This could repair missing
  reasoning hops and make the orchestration decision more useful. It also adds
  retrieval/tool costs and requires an ablation to distinguish better evidence
  from a better decision model.
- **Train on revision trajectories.** Add incorrect-to-correct and
  correct-to-incorrect states from fitting questions, then test supervised routing
  before a bounded RL experiment. This could teach the controller when another
  revision helps. Collecting trajectories costs more, and a poorly designed
  reward can favor easy approvals or needless revisions. Questions must stay
  grouped across fitting, tuning, calibration and held-out evaluation.
