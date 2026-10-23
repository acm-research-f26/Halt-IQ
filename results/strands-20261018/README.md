# Strands stopping-controller study

This completed HotpotQA experiment asks whether a small trained Strands decision
head can choose when a writer should stop revising. It is a saved-trajectory
stopping-policy experiment, not a live multi-agent orchestration benchmark.
The separate Tau² work uses its own tasks and evaluation rules.

## Experiment

- **Training:** 60 questions for head updates, 15 for model selection and 15 for
  calibration. The text encoder and released adapter stayed frozen.
- **Evaluation:** 20 different questions, with ten saved writer drafts each.
  Every critic judged the same candidate answers and supplied evidence.
- **Controls:** fixed draft counts, released Strands, trained Strands and an
  optional Jev comparator. Ten drafts means one initial answer plus nine revisions.
- **Metrics:** answer exact match and token F1; critic AUROC, Brier score,
  negative log-likelihood, approval precision and approval coverage. Paired
  intervals resample questions, not drafts or policy rows.

All primary stopping policies returned 50% exact match and token F1 of 0.54.
Eighteen questions repeated exactly the same answer across all ten drafts; two
changed wording but remained incorrect. A different stopping decision therefore
could not repair these saved answer trajectories. This is an uninformative test
of revision benefit, not evidence that the critics are equivalent.

The critics did differ when ranking first-draft correctness:

| Critic | AUROC | Approval precision | Approval coverage |
| --- | ---: | ---: | ---: |
| Released Strands | 0.380 | 50% | 50% |
| Trained Strands | 0.380 | 50% | 50% |
| Jev | 0.915 | 77% | 65% |

These are only 20 questions. Training did not improve Strands on this panel.
Approval precision describes approved answers, not all-question accuracy.

## Research artifacts

| Location | Contents |
| --- | --- |
| [data](data/) | Full public question/evidence snapshots, evaluator labels and all 200 writer drafts. |
| [exports](exports/) | Question-level answers, stopping choices and first-draft logits in a common schema. |
| [analyses](analyses/) | Numerical aggregate results, paired comparisons and hosted-call accounting. |
| [audits](audits/) | Machine-readable independent numerical checks. |
| [checkpoints/selected-head](checkpoints/selected-head/README.md) | Trained decision head, exact restore instructions and model licenses. |
| [proofs](proofs/) | Untouched-baseline, actual weight-update and separate calibration evidence. |
| [protocol](protocol/) | Dataset selection, frozen cohort, environment and model/writer settings. |
| [SCHEMA.md](SCHEMA.md) | Metric definitions, row meanings and comparison units. |

The reusable implementations are in [pc_strands](../../pc_strands/) and
[haltiq](../../haltiq/). They include native Strands feature extraction, supervised
head training, calibration, replay evaluation and provider adapters.

## Verify without model calls

From the repository root, with Python 3.10 or newer:

```bash
python -B results/strands-20261018/verify_bundle.py
```

This checks file hashes, dataset separation, all 200 answer scores, common-schema
exports, aggregate results and the selected head. It loads no model and makes no
API call. Five fixed controls repeat across three exports; 420 rows still represent
20 independent evaluation questions.

For a new native run, follow the selected-head restore instructions and
[recorded model/writer settings](protocol/source-and-writer-settings.json).
The [download configuration](../../configs/strands/windows312-20261018-downloads.json)
records public source/model pins and checksums; its historical deadlines are not
instructions to launch a new run. Use a new external runtime and new run settings.
Large upstream weights and private runtime journals are not included here.

## Scope and attribution

All ten writer drafts were physically generated before policy replay. Selected
prefix costs describe a counterfactual stopping policy; they do not demonstrate
live speed or compute savings. Optional Jev added 75 hosted calls and about
$0.00685. Local hardware costs were not priced.

HotpotQA is public validation data, not its hidden test set. Question-type
coverage is limited, and pretraining contamination cannot be ruled out. Gold
answers and supporting labels are evaluator-only and must never enter model
prompts. See the [dataset attribution](data/README.md) and
[citation](data/CITATION.bib); derived HotpotQA data remains CC BY-SA 4.0.
Model licensing and modification notices remain with the selected head.

`artifact-origins.json` records source hashes and export transformations for
retained evidence. `artifact-manifest.json` binds the files currently published
here. Numerical evidence was preserved during repository cleanup.
