# Strands Decider study on the Windows PC

The study compares the released Strands decision head with actual head-weight
fine-tuning on HotpotQA. It uses the same official answer EM/token-F1 contract
and a compatible result export as the partner framework. The new writer prompt,
PC hardware and resource profile are declared rather than treated as identical
to historical partner runs. The primary model is Strands; Jev is a comparator.

This is an experimental implementation checkpoint. Control/protocol tests and a
20-request native Qwen writer pilot pass. Real Strands model inference, CUDA
compatibility, trained weights and fresh evaluation need their own receipts;
unit tests do not establish these results.

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
2. Generate the original twenty development questions up to ten total drafts.
   Evaluate actual untouched Strands BEFORE any weight update. The trainer checks
   the hash-bound baseline receipt, all twenty attempted questions and at least
   one real valid model forward; failures remain in the denominator.
3. Cache frozen torso/adapter representations in question-disjoint fit, tuning
   and calibration groups. Default cache CLI uses both option orders; this bounded
   study explicitly passes `--canonical-order-only`. No transformer backward graph
   is needed for head-only training.
4. Update the existing3151360parameter pointer head. Verify changed head tensors,
   unchanged released tensors/adapter and matching saved/reloaded logits. Select
   on tuning NLL; calibrate separately. Calibrate an unchanged released-head copy
   as a control so calibration gain is distinct from weight training.
5. Freeze one fresh20/50/100question panel from measured throughput before viewing
   its outcomes. Generate common trajectories; compare Fixed1/2/3/5/10, released
   and tuned stopping policies, with cap-three partner parity.
6. Report paired question-level intervals, failures, repairs/regressions, probability
   coverage and all physical work. A predeclared secondary confidence selector may
   retain an earlier visited draft at the same stop boundary, with no gold-based
   selection, extra inference, cost rewind or rescue of failed episodes.

The campaign runner accepts a reviewed JSON plan with explicit argv, file/hash
prerequisites, output contracts and exactly one answer-blind fresh-N freeze. It
copies and hashes source before execution, contains only its own children and
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
<document-python> -X utf8 -B tools/build_strands_report.py --input-root <runtime> --train-run study-runs/train600-fixed1 --fresh-n 0 --campaign-run campaign-main-v2-20261018 --as-of 2026-10-18T20:30:00-05:00 --output-dir <new-empty-report-directory>
```

Choose the actual checkpoint timestamp rather than copying the example date.
For N20/50/100, the builder requires the explicit campaign plan, its adopted
answer-blind freeze, exact cohort IDs and pre-generation timing. It re-runs the
pure canonical comparison, verifies current trained/untouched checkpoints and
calibration, and omits quality tables if proof is incomplete. It emits a new PDF,
plain text and private provenance JSON; it refuses to overwrite older reports.
Public PDF/text references use relative source labels; provenance retains local
paths. Render every page with Poppler and inspect all pages before sharing.
Synthetic tests validate reporting contracts; they are not Strands results.

Public sources:
- https://github.com/strands-labs/strands-decider
- https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
- https://hotpotqa.github.io/
- https://github.com/acm-research-f26/Halt-IQ/tree/245973f27643a2ec5a48bca0a733edfed089bc57/eval-framework
