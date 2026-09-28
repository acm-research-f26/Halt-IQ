# openjev-critic: Kev-0.8B as an open Jev-like critic

This test uses **Kev-0.8B**, an open-source Jev-like decision model, as the
critic in Yash's HotpotQA writer–critic loop, compared with the existing arms on
all 20 dev questions.

**Result:** Kev-0.8B works mechanically but not as a critic. At the 0.8
threshold it rejected every draft. Offline, its first-draft scores separate
correct from wrong answers at about chance level (AUROC 0.535).

## Why

- **Official Jev is blocked** on TypeSafe billing, so the `jev` arm can't run yet.
- **Tushaar suggested trying open-source Jev-like models** in the meantime.
- Yash's prototype already has an **`openjev` arm** that calls any local server
  implementing TypeSafe's `POST /v1/systemone`, so an open model can be
  plugged in without changing his code.

## Why Kev-0.8B

| Candidate | Verdict |
|---|---|
| **[Kev-0.8B](https://github.com/jaredpalmer/kev)** | **Chosen.** Its `/v1/systemone` matches exactly what Yash's `openjev` client sends and checks (details below). It runs on Apple Silicon through MLX and fits in memory beside `qwen3:8b`. |
| Kev-4B | Too big here. The model card lists about 9 GB of weights and about 14 GB with the server's batching buffers, and the README recommends a 32 GB Mac. With `qwen3:8b` at about 10 GB, that's roughly all of this 24 GB Mac. |
| [mini-jev](https://github.com/r-ms/mini-jev) | Not usable. Its server has no `/v1/systemone` endpoint, only `POST /run` and `GET /health`. It would need a custom server, and its own README says its scores are "a ranking with a confidence gap, not calibrated probabilities." |

**API match with Yash's `openjev` client** (checked in Kev's `kev/serve.py` and
`kev/api.py`, then live):
- The request `{"model", "state": {question, evidence, draft}, "questions": {…3 noul…}}`
  is accepted. `state` may be an object, and noul `criteria` may be `{true, false}`.
- The response contains `answers.<id> = {"type": "noul", "noul": p}`, integer
  `usage.input_tokens` and `usage.output_tokens`, and the `model` echoed back.
- Kev doesn't validate the request model name, and Yash's client doesn't check
  the returned name for local servers. With `KEV_API_KEY` unset, Kev requires no
  auth, which matches Yash's client (it sends no key to local servers).
- **No adapter or shim was needed.**

## Setup

| | |
|---|---|
| Kev | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) at commit `461b5ea` (2026-09-28), in its own folder `~/projects/haltiq/kev` |
| Model | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) (snapshot `9a45d25e`), a LoRA adapter on `Qwen/Qwen3.5-0.8B-Base` |
| Backend | MLX 0.32.2 on Apple Silicon (M4 Pro, 24 GB), bf16; built-in calibration temperature 2.35 |
| Writer / other critics | `qwen3:8b` via Ollama, digest `500a1f067a9f…`, Q4_K_M |
| Yash's code | `Yash` branch at `6113915`, unmodified (source hash `77de4bc9…`, the same as his pilot) |
| Memory during the run | about 3.2 GB for the Kev server + 9.8 GB for Ollama; memory pressure normal |

```bash
# Kev (its documented setup; KEV_API_KEY deliberately unset)
git clone https://github.com/jaredpalmer/kev.git ~/projects/haltiq/kev
cd ~/projects/haltiq/kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009

# From halt-iq-yash: smoke test, then the full comparison
python3 -m haltiq judge --backend openjev --openjev-url http://127.0.0.1:8009 \
  --openjev-model kev-0.8b --input data/example_state.json
python3 -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed,openjev \
  --openjev-url http://127.0.0.1:8009 --openjev-model kev-0.8b --split dev --limit 20
python scripts/analyze_run.py results/runs/run-20260928T234403183653Z

# Offline threshold analysis (no model calls; needs that run folder's traces.jsonl)
python3 first_draft_thresholds.py path/to/run-20260928T234403183653Z
```

`--openjev-model kev-0.8b` is a label that Kev echoes back into the traces. The
threshold stayed at Yash's 0.8, and every other setting was the default.

**Smoke test:** the pipeline passed with valid scores in about 180 ms at $0.
But Kev rejected both the wrong example answer ("Hillford": 0.51 / 0.43 / 0.41)
and the correct one ("Bellport": 0.63 / 0.53 / 0.50).

## Results: 20 dev questions

These are from `run-20260928T234403183653Z/analysis.md`. All 80 episodes
completed, all artifact checks passed, and hosted cost was $0.

| System | EM | F1 | Mean drafts | Critic calls | Repairs | Regressions | EM-wrong approvals |
|---|---|---|---|---|---|---|---|
| One draft (no critic) | 11/20 | 61.3% | 1.00 | 0 | — | — | — |
| fixed | 11/20 | 61.3% | 3.00 | 0 | 0 | 0 | — |
| llm | **13/20** | 71.3% | 1.20 | 24 | 2 | 0 | 7 of 19 |
| local-typed | 11/20 | 61.3% | 1.10 | 22 | 0 | 0 | 9 of 19 |
| **openjev (Kev-0.8B)** | 11/20 | 61.3% | 3.00 | 60 | 0 | 0 | none (0 approvals) |

- **Kev approved nothing.** Every question ran to the 3-draft cap, and its final
  answers match `fixed` on every question: its template feedback never changed
  correctness.
- The `fixed`, `llm` and `local-typed` results are identical to the earlier
  runs (see `../repro-yash-pilot/`).
- Kev averaged **0.41 s per call** (max 0.86 s), against about 4.5–5 s for the
  Qwen critics. Everything ran a bit slower than in the earlier runs (`fixed`
  7.2 s per question vs 5.3 s), probably because Kev and Ollama shared the GPU,
  so compare latency only within this run.

## Threshold analysis: first drafts only

`first_draft_thresholds.py` reads only the saved traces and makes no model
calls. For each question it takes the round-1 critic scores on the **shared first
draft** (11 EM-correct, 9 EM-wrong). "Approve" means all three checks are at or
above the threshold, which is the same as the lowest of the three scores being
at or above it. AUROC is the chance that a correct draft scores above a wrong
one: 0.5 is a coin flip and 1.0 is a perfect ranking.

**Kev-0.8B (openjev): the scores don't separate correct from wrong.**

| score | mean if correct | mean if wrong | AUROC |
|---|---|---|---|
| supported | 0.526 | 0.516 | 0.515 |
| complete | 0.526 | 0.493 | 0.545 |
| relevant | 0.430 | 0.435 | 0.515 |
| lowest of the three | 0.430 | 0.417 | **0.535** (53 of 99 correct/wrong pairs ranked correctly) |

| Threshold | First drafts approved | Wrong among approved | Correct drafts approved (of 11) |
|---|---|---|---|
| 0.45 | 8 | 4 of 8 | 4 |
| 0.50 | 7 | 3 of 7 | 4 |
| 0.55 | 4 | 2 of 4 | 2 |
| 0.60 | 2 | 1 of 2 | 1 |
| 0.65 | 2 | 1 of 2 | 1 |
| 0.70 | 1 | 0 of 1 | 1 |
| 0.75 | 0 | none | 0 |
| 0.80 | 0 | none | 0 |

At low thresholds, Kev's share of wrong approvals (about 40–50%) is just the
base rate of wrong first drafts (45%), which is what random approval gives.

**local-typed (Qwen writing scores as text): effectively a rubber stamp.** It
only ever outputs 1.0 or 0.0, and it gives every wrong first draft 1.0. Its only
0.0 went to a correct draft (#3, MedStar). AUROC is 0.455, slightly worse than
chance.

| Any threshold 0.45–0.80 | First drafts approved | Wrong among approved | Correct drafts approved (of 11) |
|---|---|---|---|
| all | 19 | 9 of 19 | 10 |

So the text-vs-probabilities comparison is: Qwen's written scores are
all-or-nothing and approve every wrong answer, while Kev's probabilities are
graded (20 distinct values) but carry almost no signal about correctness here.

**Temperature scaling can't fix this.** Kev's calibration divides the logits by
one temperature, which is monotonic, so it never changes the ranking of scores.
Applying the same increasing function to all three checks also preserves which
one is lowest. AUROC is therefore identical at any temperature, including the
raw `KEV_TEMPERATURE=1.0`. A different temperature would only move where a
threshold falls on the same ranking, not how well that ranking separates correct
from wrong answers.

## Kev's documented limits

These are from Kev's README and the [Kev-0.8B model card](https://huggingface.co/jaredpalmer/kev-0.8b),
and they fit what we saw:
- **Weak on knowledge and outside its domain:** "Out of domain it is a sub-1B
  model," with MMLU 0.41 and PAWS 0.59. HotpotQA's `supported` check requires
  multi-hop fact verification.
- **Trained on short texts:** at most 384 state tokens, while a HotpotQA state
  with about 10 paragraphs is much longer. Serving allows 65,536 tokens, but
  "longer context wasn't covered by training."
- **Calibration** is a single temperature fitted in distribution; "treat [probabilities]
  as advisory elsewhere."
- **Mac speed:** it runs through MLX, with answers in hundreds of milliseconds.
- **Kev-4B** is the recommended size, but it needs about 14 GB and wouldn't fit here.

## Caveats

- **n = 20** (11 correct, 9 wrong first drafts). An AUROC of 0.535 is
  consistent with anything from roughly 0.3 to 0.8, so this shows no useful
  separation *here*, not that Kev-0.8B is useless everywhere.
- **First-draft decisions only.** The threshold tables cover one decision per
  question on the shared first draft. They don't simulate the full loop: at a
  lower threshold, rejections would produce different later drafts, which these
  traces can't show.
- **Don't pick a threshold from these 20 questions.** That would be tuning on
  the evaluation data. Any threshold needs checking on fresh questions, and the
  80 held-out questions should stay untouched until settings are frozen.
- **This is not Jev.** Report the arm as "OpenJev: Kev-0.8B". Official Jev
  remains unrun.

## Files

- `run-20260928T234403183653Z/`: `report.md`, `episodes.csv`, `summary.json`,
  `analysis.md` and `analysis.json` from the run. The traces, manifest and
  source snapshot stay in the local `halt-iq-yash` clone.
- `first_draft_thresholds.py`: the offline threshold analysis. It needs the
  run's `traces.jsonl`.

## Credit

- **Yash Baruah:** the HotpotQA writer–critic prototype, including the
  `openjev` arm and its `/v1/systemone` client and `scripts/analyze_run.py`,
  which produced `analysis.md`, `analysis.json` and the one-draft baseline
  ([`Yash` branch](https://github.com/acm-research-f26/Halt-IQ/tree/Yash)).
- **Tushaar Sood:** the suggestion to try open Jev-like models.
- **Kev** by Jared Palmer: [GitHub](https://github.com/jaredpalmer/kev) and the
  [Kev-0.8B model card on Hugging Face](https://huggingface.co/jaredpalmer/kev-0.8b).
  Apache-2.0.
