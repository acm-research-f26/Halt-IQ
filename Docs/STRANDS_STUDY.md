# Strands Decider study on the Windows PC

The study compares the released Strands decision head with actual head-weight
fine-tuning on HotpotQA. It uses the same official answer EM/token-F1 contract
and a compatible result export as the partner framework. The new writer prompt,
PC hardware and resource profile are declared rather than treated as identical
to historical partner runs. The primary model is Strands; Jev is a comparator.

This is an experimental implementation checkpoint. The actual untouched
20-question Strands development gate has its own receipt-bound native journal,
episodes and evaluator tasks. It is separate from control/protocol tests and the
Qwen writer pilot. Trained weights and fresh evaluation still require their own
receipts; unit tests do not establish model results.

The pinned runtime queue is in
`configs/strands/windows312-20261018-downloads.json`. It contains only public URLs,
file sizes and SHA256s, with this study's dated deadline. To reuse it later, copy
it to an external runtime directory and explicitly choose a new future deadline
with a UTC offset. Keep dependencies, datasets, caches, weights, keys and run
outputs outside this source checkout. This queue targets Windows/Python3.12.

Use a separate Python3.12 virtual environment. The downloader and offline install
watcher are separate commands; the watcher installs only verified local wheels
and the pinned upstream checkout. It does not download models automatically.

```powershell
python -m pc_strands.runtime_setup --root <runtime> --queue <queue.json> --rate 1048576
<venv-python> -m pc_strands.runtime_setup --root <runtime> --queue <queue.json> --install-when-ready --source <pinned-strands-checkout>
```

Upstream source commit: `930e6d5de16309be32579f5631a7369a3322140c`.
Released checkpoint revision: `216d1ccb46a9e9b036eb9187b248db98480f77d4`.
Base revision: `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
No LM Studio chat-model import is required: this is a local Transformers/PEFT
pointer model with the vocabulary output head replaced and input embeddings kept.
The strict4096token limit covers the complete decision prompt. Overflow is an
explicit failure; evidence is never silently truncated.

## Evaluation order

1. Generate TRAIN-only drafts with the frozen writer settings, keeping references
   outside all prompts. Store a durable started/completed request journal.
2. The adopted V3 baseline-only plan generates one draft for each of the original
   twenty development questions, retaining every question and full native context.
   It waits for truthful writer cleanup/ownership proof, then evaluates actual
   untouched Strands BEFORE any weight update, once verified weights, installation
   and CUDA health are ready. No quality-based or token-length case selection is
   allowed. The trainer checks the hash-bound baseline receipt, all twenty attempted
   questions and at least one real valid model forward; failures remain in the
   denominator. This cap-one gate measures first-draft decisions and native
   throughput, not repair or ten-draft benefit. It does not automatically launch
   training or fresh testing.
3. After measured native throughput, adopt a NEW downstream plan: use original
   TRAIN600 only if its actual generation is complete, or explicitly select a
   separate original-partition 120 protocol with 80 fit / 20 tune / 20 calibration and its
   own writer run. A proposed subset alone is not an adopted training protocol.
   Cache frozen torso/adapter representations in question-disjoint fit, tuning
   and calibration groups. Default cache CLI uses both option orders; this bounded
   study explicitly passes `--canonical-order-only`. No transformer backward graph
   is needed for head-only training.
4. Update the existing3151360parameter pointer head. Verify changed head tensors,
   unchanged released tensors/adapter and matching saved/reloaded logits. Select
   on tuning NLL; calibrate separately. Calibrate an unchanged released-head copy
   as a control so calibration gain is distinct from weight training.
5. Freeze one fresh20/50/100question panel from measured throughput before viewing
   its outcomes. The primary fresh protocol remains ten total drafts on common
   trajectories; compare Fixed1/2/3/5/10, released and tuned stopping policies,
   with derived cap-three partner parity. The cap-one development gate does not
   shorten this fresh protocol.
6. Report paired question-level intervals, failures, repairs/regressions, probability
   coverage and all physical work. A predeclared secondary confidence selector may
   retain an earlier visited draft at the same stop boundary, with no gold-based
   selection, extra inference, cost rewind or rescue of failed episodes.

The campaign runner accepts a reviewed JSON plan with explicit argv, file/hash
prerequisites and output contracts. A plan with fresh options requires exactly
one answer-blind fresh-N freeze; the adopted baseline-only V3 has no fresh branch.
It copies and hashes source before execution, contains only its own children and
refuses automatic retry of started/failed stages. Native health probes acquire the
same OS GPU lease as model workers. The operator hands off the CUDA-health marker
only after writer cleanup; the campaign does not create that marker itself.

## Interpretation and PC use

FixedK returns draftK even if it is worse. Rechecking can regress: in the earlier
200-question development panel, four correct first drafts became wrong at draft2
and none became correct, so EM fell57.5% to55.5%. Three is a historical experiment
cap, not a Strands limit. Ten total drafts means initial plus nine revisions.

Trajectory replay estimates selected answers and counterfactual prefix usage.
Physical writer generation, actual feature encoding and CPU-head scoring are
counted separately. Replayed prefix time is not measured live end-to-end latency
or proof of operational savings. Local electricity/hardware dollars are unpriced.

The background profile uses one model owner, four CPUthreads,7GiBavailableRAM and
3GiBfreeVRAM admission reserves, intentional rests and a1MiB/s serialized transfer.
These controls leave room for foreground work but are not hard GPU percentage
quotas. Profile changes require separate named runs. See `PC_STUDY_CONTROL.md` for
pause/stop commands. A pause can retain memory; normal stop/cleanup releases only
study-owned processes. No permanent power/driver/network settings are changed.

The current user deadline is8:30PMCentral onOctober18,2026, with new model work
ending around8:00 and paired outputs preserved by8:10 for reporting. A shorter
honest experiment is acceptable. Do not claim six hours of active testing from
setup/download/waiting time, a quality gain from training proof alone, or a
successful Strands run before actual native receipts exist.

## Professor report generation

The maintained builder reads explicit evidence without model loads or API calls.
Use the document Python with ReportLab and pypdf installed; study runtime packages
alone do not supply the PDF renderer. For a partial checkpoint, pass fresh N0:

```powershell
<document-python> -X utf8 -B tools/build_strands_report.py --input-root <runtime> --train-run study-runs/train600-fixed1 --fresh-n 0 --campaign-run <actual-campaign-directory> --as-of <checkpoint-ISO8601-with-offset> --output-dir <new-empty-report-directory>
```

Choose the actual campaign directory and checkpoint timestamp with a UTC offset;
do not substitute an unused or superseded campaign. For a resumed TRAIN source,
optionally append each preserved invocation archive in chronological order:

```powershell
--train-segment writer-supervision/train600-background-v2-ram7-q8-v5/segment-1-final
```

The archive must include its snapshot receipt and matching source/journal hashes.
The builder checks source identity, cumulative journal prefixes and nonoverlapping
worker intervals before summing selected wall/idle durations. Those sums exclude
gaps between invocations and wrapper startup/unload. The first selected snapshot's
physical counter is cumulative; subsequent counter differences are between
selected snapshots. Current cumulative physical calls/tokens are charged once,
not added to an archived cumulative counter. Without selected archives, only the
latest invocation timing is shown.

Completed provider calls and saved trajectories have separate counts. For example,
the original 600 run retained 590 trajectories but 591 completed calls after a
deadline during post-call rest: one completed call had no assembled trajectory,
nine questions were never dispatched, and ten trajectories remained missing.
All 591 calls still count as physical work. Zero saved trajectory failures does
not mean complete planned coverage or authorize relabeling a partial run complete.

For N20/50/100, the builder requires the explicit campaign plan, its adopted
answer-blind freeze, exact cohort IDs and pre-generation timing. It re-runs the
pure canonical comparison, verifies current trained/untouched checkpoints and
calibration, and omits quality tables if proof is incomplete. It emits a new PDF,
plain text and private provenance JSON; it refuses to overwrite older reports.
Public PDF/text references use relative source labels; provenance retains local
paths. Render every page with Poppler and inspect all pages before sharing.
Synthetic tests validate reporting contracts; they are not Strands results.

The report resolves artifact roles from the selected frozen plan's explicit stage
arguments: the training command supplies its baseline, source checkpoint, trained
checkpoint and fit/tune caches; native cache commands identify their TRAIN source;
calibration commands identify their checkpoint and temperature-fit output; the
selected fresh option's comparison and replay commands identify the shared writer
trajectories, features, heads and analysis. Nested paths such as
`checkpoints/train120-original-v4/head-trained` and `study-runs/fresh20-v4-trained`
therefore need no canonical aliases or copied results. Conflicting/missing declared
roles cannot fall back to older canonical artifacts. An explicit `--train-run`
must match the plan's declared TRAIN cache source. A baseline-only plan with no
TRAIN/cache/trained/fresh roles still permits a partial report on an explicitly
selected historical TRAIN source; it does not adopt unrelated old trained files.
Private provenance records the resolved role paths. The operator freeze comes
from `operator_fresh_freeze_path` (the historical default is `fresh-N-freeze.json`)
and must exactly match the campaign's copied, plan-bound freeze.

If genuine training/calibration and later fresh evaluation belong to separate
campaigns, keep both owners explicit:

```powershell
--campaign-run <actual-fresh-only-campaign> --training-campaign-run <actual-completed-training-only-campaign>
```

The fresh-only plan must have no imported base stages. Its
recorded stage IDs must be a subset of its uniquely declared fresh-option stages,
with no training-stage names imported into fresh state. Its
`training_campaign_reference` must name that training directory and bind its exact
frozen plan and terminal campaign hashes. The builder checks completed training
stage coverage, resolved actual argv, every captured output hash/output contract,
both frozen fifteen-file source snapshots and identical model pins. Training
commands supply TRAIN/cache/checkpoint/calibration roles; fresh commands supply
only the held-out roles. The unchanged-control copy remains owned by its actual
training stage, including the pre-calibration captured hashes. It is never copied
into fresh campaign state. Missing or tampered ownership is rejected, with no
canonical aliases or silent role fallback. All existing canonical rescoring,
baseline, weight-change, adapter, calibration, cohort-freeze and physical-expense
checks still apply. The existing single-campaign interface remains valid; use it
with fresh N0 to report a partial training-only checkpoint before a later fresh
campaign exists. The two-owner option does not establish training completion or
held-out improvement without the actual validated evidence.

The unchanged-control proof must link its head tensor hash to the trained proof's
original tensor hash. The released source head must also match the untouched gate
and training proof, and the released, control and trained adapter files must all
match the frozen adapter SHA256. A newly serialized unchanged head may have
different file bytes; its actual file hash remains bound by calibration and
checkpoint fingerprints. The actual control head and copy-proof hashes must also
match the completed copy stage's captured output hashes from before calibration;
rebinding later calibration or comparison fingerprints cannot hide a post-copy
mutation. Both calibration fits must use the same question IDs,
the predeclared `[0.25,6]` temperature bounds and threshold `0.8`. These checks add
no model loads. They check recorded tensor/weight provenance; the pre-fresh runtime
qualification separately performs the actual tensor comparison.

The physical-work table covers the explicit `--train-run`, an optional explicitly
selected historical TRAIN source, a validated untouched native development gate,
and selected confirmation phases. With a new
120 writer source, retain the earlier preparation expense through separate inputs:

```powershell
--train-run study-runs/train120-original-v4-fixed1 --historical-train-run study-runs/train600-fixed1 --historical-train-segment writer-supervision/train600-background-v2-ram7-q8-v5/segment-1-final
```

The historical source has its own journal, reconciliation, archive identity and
wall/idle table. The original 591 completed calls are charged once, separately
from new 120-run calls; archived cumulative counters are not added again. Questions
may repeat across real writer executions, but historical questions, references,
caches and denominators are never pooled into current training or confirmation.
An archive cannot bind to a different source; do not pass the original600 archive
as a current120 `--train-segment`. Selecting the same current/historical directory
twice, copied journals, or overlapping timestamped dispatch identities are rejected.
Historical missing trajectories remain missing, and an
incomplete source cannot be relabeled terminal. No source is inferred from plan
text or copied into a canonical alias to satisfy this option. Preserve the actual
receipts, and do not describe this phase table as a complete campaign-cost total:
unprovided setup, writer-generation, training or other phases remain separate
evidence.

The baseline table is explicitly **DEVELOPMENT / cap1 / before training**. Its
official-style answer EM/token F1 are recomputed from receipt-bound episodes and
the evaluator task snapshot, whose fingerprint must match the bound manifest.
The loose `summary.json` is never an authority. Both Fixed1 and completed Strands
episodes return the same first answer, so this gate cannot establish repair,
fresh accuracy or a fine-tuning gain. Approval precision and rejection diagnostics
are distinct from all-question answer accuracy, and failures keep the planned
denominator.

The native request journal also needs an exact SHA from the receipt, the reviewed
plan's explicit `baseline_reuse` hashes, or a completed baseline stage's captured
outputs under a matching executed plan. Every public question/evidence/draft input,
raw critic response, probability/threshold decision and physical usage total is
reconciled. Missing or inconsistent evidence suppresses baseline metrics/native
expense and invalidates the untouched gate; it never falls back to an unrelated
summary. These diagnostics can appear with the actual baseline-only V3 plan before
any trained checkpoint or fresh result exists.

Baseline native forwards/tokens/request seconds are charged once, separately from
current and historical TRAIN writer expense. Backend-startup wall and replay-phase
wall have separate recorded scopes; their sum excludes writer generation and
request seconds must not be added again. Phase idle includes rests, resource holds
and scheduling overshoot, so no residual is labeled exact RAM-pause time. Missing
startup timing remains unknown rather than zero. This remains a partial physical
phase table, not a complete campaign-cost or hardware/electricity estimate.

The optional report argument `--development-diagnostic <explicit-receipt.json>`
adds a separately labeled **secondary post-training DEVELOPMENT20 / cap1**
probability diagnostic, only after its actual three-stage native-cache and two
CPU head-replay campaign finishes. It reuses the same saved first drafts and the
matched, separately calibrated released/trained heads. This reused development
cohort was inspected before training; it cannot establish answer improvement,
repair benefit or fresh generalization, and never satisfies the primary fresh
confirmation gates. The saved writer answer EM/F1 stays distinct from
failure-inclusive head-episode quality and correctness-classification accuracy.

The consumer requires the exact reviewed stdlib audit and evidence helper hashes,
re-runs their read-only joins without models/Torch/network or bytecode writes,
and compares every normalized receipt field except its audit timestamp. Those
joins validate all 20 original tasks/full contexts, the three owning stages,
15 frozen source files, native features/events/overflows, raw head journals,
episodes, logits, labels and current calibrated checkpoints. The report also
requires its actual training owner and baseline/head roles to match the receipt,
checks every input-file hash, and rejects audit timestamps after the report.
Missing, changed or inconsistent explicit evidence omits **all secondary metrics
and secondary expense**, with the reason recorded privately. It does not change
the primary baseline/training/confirmation validation results. Neither a missing
receipt nor an unfinished campaign is replaced with synthetic scores.

AUROC/Brier/NLL use observed probabilities; missing/overflow scores remain absent.
Approval precision is conditional on approval, alongside approval coverage over
all 20 planned questions. Undefined AUROC/precision is displayed as undefined.
Question-paired differences use 5,000 draws/seed 42; NLL/Brier/classification accuracy
use only the observed-pair intersection. Approval coverage uses all 20, treating
an absent decision as no approval without inventing its probability. AUROC and
precision intervals use paired observed questions, exclude and count undefined
bootstrap draws, and show the actual pair denominator. Their point differences
are each arm's observed-cohort differences, explicitly distinguished from those
paired intervals when score availability differs. These small-cohort intervals
are descriptive. New native encoding and each CPU head replay are charged once
as separate physical phases; the reused writer drafts incur no new writer calls.
Private provenance and machine paths are not public website assets.

Public sources:
- https://github.com/strands-labs/strands-decider
- https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
- https://hotpotqa.github.io/
- https://github.com/acm-research-f26/Halt-IQ/tree/245973f27643a2ec5a48bca0a733edfed089bc57/eval-framework
