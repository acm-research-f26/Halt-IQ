# October 18 completed Strands study — Yash's team handoff

The implementation, actual untouched baseline, decision-head training, separate
calibration, and matched fresh evaluation are complete. **There was no observed
answer-quality improvement:** all 13 primary policies achieved 10/20 exact matches
(50%) and token F1 0.54. Optional Jev achieved the same answer scores, with stronger
first-draft correctness ranking on this small panel. These results do not establish
population equivalence or broad model superiority.

## Read and share

- [Team write-up](reports/strands-head-finetuning-writeup.pdf) — four-page plain-language account of what was built, how it was tested, the results and two next ideas.
- [Editable team write-up](strands-head-finetuning-writeup.md), including the simple pipeline diagram.
- [Earlier technical summary](reports/team-progress.pdf) — preserved for reference.
- [Editable team progress text](team-progress.md).
- [Professor report](reports/professor-review.pdf) — approved 18-page primary report.
- [Optional Jev comparison](reports/optional-jev-companion.pdf) — approved four-page companion.
- [Independent primary results](audits/primary-independent-results.txt).
- [Independent optional Jev results](audits/optional-jev-independent-results.txt).
- [Reusable selected trained head and restore instructions](checkpoints/selected-head/README.md).

The private study website is supplementary. Teammates can use this repository's
documents and exports without access to the website.

## What was implemented and run

The local Qwen3 8B writer generated ten generic self-check drafts for each of 20
answer-blind, preselected HotpotQA validation questions. Both Strands heads judged
the **same saved trajectories**, so differences isolate stopping/selection on
common candidates. The decision model is used as the orchestrator's stop/continue
controller. It does not generate prose feedback or new evidence.

The released Strands model was genuinely evaluated on all 20 original development
questions before any weight update. A separate, answer-blind 90-question TRAIN
cohort preserved original 60 fit / 15 tuning / 15 calibration partitions. Training
updated the existing 3,151,360-parameter pointer head while freezing the encoder
and released adapter. Twelve candidate optimizer steps were performed in total;
the selected learning-rate 1e-5 / epoch-1 checkpoint contains **two steps**, not all
twelve. Reloaded logits matched, and the head changed while frozen tensors did not.
Released and trained heads received separate temperature fits on the same 15
calibration questions. This was supervised head training, not RL or new LoRA training.

Fresh comparison included Fixed1/2/3/5/10, both calibrated heads at cap10 and cap3,
and four predeclared visited-confidence retention variants. Ten drafts means one
initial answer plus nine revisions. All 20 planned questions remained in every
denominator. Eighteen questions repeated the same raw answer throughout; the two
wording-changing cases remained incorrect. There were no repairs, regressions,
failures, overflows or missing primary episodes.

| Fresh first-decision diagnostic | Released calibrated Strands | Trained calibrated Strands | Optional Jev |
| --- | ---: | ---: | ---: |
| Observed questions | 20 | 20 | 20 |
| AUROC | 0.380 | 0.380 | 0.915 |
| Brier score | 0.336260 | 0.336322 | 0.141200 |
| NLL | 0.910647 | 0.910822 | 0.469715 |
| Correct / first approvals | 5 / 10 | 5 / 10 | 10 / 13 |
| First approval coverage | 50% | 50% | 65% |

Approval precision is conditional on approval; it is not the all-question match
rate. Jev's eventual approvals were 10 correct out of 14, distinct from its first
10/13. Strands uses raw-logit NLL and saved temperatures; Jev supplies probabilities,
so logits and temperature stay null. The reports retain paired 5,000-draw seed-42
question-level intervals and their small-panel limitations.

## Files and provenance

- `data/*evaluator-only.jsonl`: exact public full-context fresh20 and TRAIN90
  snapshots. Gold answers/support labels are evaluator-only. Native model input
  construction allowlists question/evidence/draft; never paste whole rows into prompts.
- `data/fresh20-writer-trajectories.json`: all 20 × 10 actual drafts and reported
  call usage, with local response bodies removed. Its wrapper records the original
  raw trajectory hash and the transformation. It is a review export, not an alias
  satisfying strict original-trajectory hash gates.
- `exports/`: three compatible 140-row common-schema exports and first-draft logit
  records. **Fixed controls repeat across exports** and are the same observations.
  The 420 rows do not represent 420 independent questions. Independent fresh N is 20.
- `analyses/primary-paired-analysis.json`: all 13 primary policies and their paired
  contrasts, original SHA and explicit sanitization wrapper. `jev-*` files retain
  optional aggregates, paired contrasts and finalized accounting.
- `protocol/`: original selection history, freezes, full historical plans,
  selection proof, installed packages and writer/model/source settings. These are
  portable **DATA ONLY** snapshots under a wrapper; symbolic paths and dated
  deadlines must not be launched as a current plan. Preserved historical PROPOSED
  labels describe preparation metadata; actual completion belongs to the separate
  terminal campaign receipts in `proofs/`.
- `proofs/`: training/control/calibration proofs, native pretraining gate and actual
  owning terminal campaign snapshots. Original evidence hashes remain distinct
  from sanitized export hashes; the private original evidence is not overwritten.
- `artifact-origins.json`: original logical source/hash and transformation for each
  copied/derived evidence file. `artifact-manifest.json` binds this delivery's files.

Local machine paths were replaced with symbolic labels in derived JSON, including
path keys and historical argv strings. Scientific values and original SHA256s were
preserved. No credentials, original hosted HTTP bodies, live budget SQLite database,
raw paid journals, large upstream base/adapter weights, feature vectors or private
machine provenance are included. The selected trained head (12.6 MB) and exact
calibrated config are included under Apache-2.0 with licenses and modification
notices. The actual model/runtime source already lives in `pc_strands/`;
all 15 frozen source files remain byte-identical to the reviewed implementation.

## Offline verification and a future rerun

From the repository root with Python 3.10+:

```powershell
python -B results/strands-20261018/verify_bundle.py
```

This loads no model and calls no API. It checks file hashes/inventory, full cohort
counts and disjointness, all 200 draft scores, all common-schema rows, fixed-control
duplication, 13 aggregate policies, input allowlisting and the 15 frozen source hashes.
It verifies the delivered review bundle, not the authenticity of omitted private
raw network/model journals; those were independently audited before export.

The following actual CLI entry points were checked with `--help` when preparing
this delivery; help is read-only and imports no model:

```powershell
python -B -m pc_strands.runtime_setup --help
python -B -m pc_strands.evaluate --help
python -B -m pc_strands.decider --help
python -B -m pc_strands.train_head --help
python -B -m pc_strands.calibrate --help
python -B -m pc_strands.compare --help
python -B -m pc_strands.jev_replay --help
python -B -m pc_strands.campaign --help
```

A full native rerun needs downloaded pinned upstream source/base/checkpoint and
the installed environment captured here, a fresh external runtime directory,
newly reviewed absolute local paths/deadlines and resource admission. The source
download queue at `../../configs/strands/windows312-20261018-downloads.json` records
public URLs, sizes and SHA256s; its old deadline needs an explicit new copy.
Upstream source is `930e6d5de16309be32579f5631a7369a3322140c`, checkpoint revision
`216d1ccb46a9e9b036eb9187b248db98480f77d4`, base revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. Models and GPU runtime packages are not
installed by the dependency-free `haltiq` package. Native report reconstruction
requires its original hash-bound runtime receipts, beyond this compact public-data
review bundle. A new paid Jev rerun additionally needs the operator's own credential
and persistent cumulative ledger; no paid request is triggered by verification.

## Cost and historical boundaries

Primary fresh physical expense was 200 writer calls, 200 native encoder forwards
and 110 CPU scoring calls per Strands head. All writer drafts were generated even
when a replay controller would have stopped earlier. Mean selected drafts (5.5
for each Strands cap10 controller; 3.75 for Jev) and prefix token/time figures are
counterfactual replay estimates, not measured live latency or physical savings.
Startup, request time, intentional idle, phase wall and stage wall overlap and
are not added indiscriminately. Local hardware/electricity dollars remain unpriced.

Optional Jev added 75 calls, 163,044 input tokens, 1,500 output tokens and
$0.006847848, with no new writer or local encoder calls. The existing ledger ended
at 606 settled calls / $0.054730032 total and zero reserved/unsettled requests.
The earlier 531 calls / $0.047882184 are included in that cumulative amount.

Original TRAIN600 preparation is **historical separate expense**, not a completed
600-question training source: 590 saved trajectories, 591 completed physical
calls, one completed-but-unsaved call and nine undispatched questions. Its original
saved partitions were 400/100/90. Cumulative journals and invocation archives were
reconciled; those 591 calls are charged once. Actual TRAIN90 generation was new and
must not be pooled with historical600 as independent training examples. The before-
training development gate and reused post-training development diagnostic also
remain separate from this fresh20 confirmation.

## Attribution and two next ideas

HotpotQA: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop
Question Answering*, EMNLP 2018. [Homepage](https://hotpotqa.github.io/),
[citation](data/CITATION.bib), [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
The bundled derived data retains full public text; transformations are documented
sampling, partition annotations and formatting. Changes and attribution must be
retained under the same data license. This is public validation data, not the
official hidden test, and pretraining contamination cannot be ruled out.

1. **Evidence-specialist feedback under Strands control.** Route a rejected answer
   to a bounded verifier that identifies a cited contradiction or missing evidence,
   then revise. Pros: produces a different, grounded candidate instead of repeated
   generic self-checking. Cons: more calls, possible critic mistakes and another
   intervention requiring separate ablations/cost accounting.
2. **Budget-aware STOP/VERIFY learning.** Train the decision head on expected repair
   utility and explicit action cost, with supervised targets first and a later
   carefully bounded RL study. Pros: aligns decisions with whether another action
   can improve an answer. Cons: needs richer trajectories and counterfactual labels;
   reward errors and off-policy bias can reinforce poor actions. Neither extension
   was run in this study; a new held-out cohort is required after development.

Partner comparability means the same benchmark answer metrics and compatible row
schema. The different PC hardware, writer prompt and native/hosted context limits
are disclosed; this is not a claim of bit-identical historical partner environments.
