# critic-bench: which critic should the HaltIQ loop use?

HaltIQ's writer–critic loop on HotpotQA stops when the critic approves a draft,
so the critic decides when to halt. A good critic approves right answers and
rejects wrong ones. This benchmark compares candidate critics (the loop's
current qwen3:8b critics, Kev-0.8B, Laya, and open LLM critics) on the **same
fixed drafts**, so every critic judges identical answers and nothing is
regenerated. The question it answers: which critic should the loop use?

**Credit.** The `first`/`all` drafts come from Yash's writer–critic loop (his
Sept 28 run, `run-20260928T234403183653Z`, `Yash` branch, read-only), and the
`extra` drafts are made with his unmodified `Writer`, selection helpers, and
scorer. The harness, the Laya and logprob critics, the extra draft set, and
the evaluation in this folder are Vedanshi's.

## Draft sets

| Set | Drafts | What it is |
|---|---:|---|
| `first` | 20 | The shared first draft for each of the 20 dev questions in the Sept 28 run |
| `all` | 25 | Every unique labeled draft from any round or arm of that run (the 20 + 5 revisions) |
| `extra` | ~200 | First drafts for 200 more HotpotQA dev questions (see `make_extra.py`) |
| `subset80` | 80 | The 20 `first` drafts + the first 60 `extra` drafts: the questions with consistency samples, so every critic can be compared on the same 80 |

All questions come from HotpotQA's official dev (validation) split. The 80
locally held-out test questions are never used: `make_extra.py` excludes them
and the 20 already-used questions by source ID and by question text.

HotpotQA (Yang et al., 2018) is licensed CC BY-SA 4.0. `extra/tasks.jsonl` contains question and evidence text from it, in Yash's task format.

**Labels.** *Strict* = HotpotQA exact match after normalization (what Yash's
`analyze_run.py` uses). *Lenient* = exact match or token F1 ≥ 0.8. Both come
from Yash's `haltiq/metrics.py`.

## Files

| File | What it does |
|---|---|
| `drafts.py` | Loads a draft set with its strict and lenient labels; reads saved critic decisions from the run. Imports Yash's scorer without writing into his folder. |
| `critics.py` | The critics. Each is a small function: draft in, P(correct) or approve/revise out. Add a critic by adding a function to `CRITICS`. |
| `score_drafts.py` | `python3 score_drafts.py CRITIC {first,all,extra}`. Scores a draft set and appends each score to `scores/CRITIC.jsonl` immediately, so reruns skip finished drafts. |
| `laya_critic.py` | Laya yes/no critic. `python3 laya_critic.py measure` reports token lengths vs Laya's 512-token limit. |
| `make_extra.py` | `select` / `estimate` / `write`: builds the `extra` set with Yash's downloader, selection helpers, and Writer. |
| `make_samples.py` | Draws 3 extra writer answers per `subset80` question at temperature 0.7 (Yash's Writer and prompt; seeds 1–3) into `samples/consistency.jsonl`. |
| `run_all.sh` | Runs the scoring stages in order, one Ollama job at a time, stopping at a set deadline. |
| `combos.py` | `python3 combos.py`: compares stop rules (e.g. evidence_match as a free gate before llm) on `extra` and `subset80`; writes `combos.md`. |
| `report.py` | `python3 report.py [--label strict\|lenient]`: writes `report.md` / `report-lenient.md` and the reliability charts. |
| `scores/` | Cached critic scores (critic, draft id, score or decision, seconds). |
| `extra/` | The 200 extra tasks (`tasks.jsonl`) and their first drafts (`drafts.jsonl`). |

## Critics

| Critic | Output | Source |
|---|---|---|
| `llm` | approve/revise | qwen3:8b with Yash's `LLMCritic` prompt. Saved decisions reused; live calls for drafts the run never judged. |
| `local-typed` | P = min of 3 scores | qwen3:8b self-reported 0–1 scores (saved only). |
| `kev` | P = min of 3 noul probabilities | Kev-0.8B through Yash's openjev arm. Saved scores reused; live calls reproduce saved scores exactly. |
| `laya` | P(yes) | Laya noul question "Is the proposed answer correct and supported by the evidence?" |
| `evidence_match` | 1 / 0 (0.5 for yes/no) | No model: 1 if the normalized answer appears word-for-word in the evidence the critic sees (all 10 paragraphs, not gold supporting facts). A "yes"/"no" answer can't be matched, so it gets 0.5; `UNKNOWN` gets 0. |
| `consistency` | 0, ⅓, ⅔, 1 | Share of 3 resampled writer answers (temperature 0.7) that match the first draft after normalization. Only for first drafts of the 80 `subset80` questions; its time is the 3 extra writer calls. |

The min of three scores is used because the loop approves only when all three
checks reach the threshold, so `min ≥ 0.8` is exactly the loop's rule.

## What the metrics mean

- **Accuracy when approved (@0.8):** of the drafts the critic would stop on, the share that are actually right. If this is no better than the set's overall correct rate, the critic adds nothing over "always stop".
- **Wrong approvals:** wrong drafts the critic approved. The loop stops and returns a wrong answer.
- **Wrongly rejected:** right drafts the critic rejected. The loop spends another round and may revise a right answer into a wrong one.
- **AUROC:** the chance that a random right answer gets a higher score than a random wrong one. 0.5 = coin flip, 1.0 = perfect ranking. It ignores the threshold.
- **Reliability chart:** drafts grouped by the critic's score. If a critic is calibrated, drafts it scores 0.7 are right about 70% of the time (the dashed diagonal).
- **Mean s/decision:** wall-clock seconds per judgment on this M4 Pro Mac. **Cost:** all critics run locally, so $0.

## Notes and caveats

- **Sample size.** With n = 20 an accuracy is uncertain by about ±22 points (95%, worst case); with n = 200 about ±7. The `first`/`all` results are anecdotal; read `extra` for decisions.
- **Laya truncation.** Laya reads 512 tokens (462 for the state after its question prefix). 19/20 first-draft states are longer (median 1,583 tokens), so we trim the evidence to the sentences sharing the most words with the answer and question (median ~25% of sentences kept). Laya cuts a JSON state from the end, so the proposed answer goes before the evidence. As an evaluator-only check, both gold supporting paragraphs are still represented in 15/20 first drafts.
- **Laya temperature warning.** Loading the model warns that the `choice:11+` calibration temperature (0.10) is clamped. We only ask a yes/no (noul) question, whose temperature (1.98) is within range and not clamped, so the warning does not apply.
- **Reused vs live scores.** Saved scores come from the Sept 28 run; live scores use the same classes and settings. Kev live scores matched saved ones exactly on a spot check.
- **Consistency samples.** Yash's client fixes temperature 0 and seed 42, so `make_samples.py` sends the same request with temperature 0.7 and seeds 1–3 (with one fixed seed, all samples would be identical). After the first sample, Ollama reuses the cached prompt, so 3 samples took about 8.7 s per question in total.
- **Environment.** System Python 3.14 with `laya_mlx`, plus `matplotlib` and `pyarrow` (installed with `pip3 install --user`). The HotpotQA download needed `SSL_CERT_FILE=$(python3 -m certifi)`, because python.org Python has no CA bundle; Yash's downloader still verifies the pinned sha256. The Kev server runs from `~/projects/haltiq/kev` on port 8009, as in the Sept 28 run.
- **Exact match is strict.** Some "wrong" drafts are near misses (e.g. `Wilmette, Illinois` vs gold `Wilmette`, F1 0.67), which is why a lenient label is also reported.

## Results (as of Oct 2, before Steps 5–6)

Full tables for every draft set: `report.md` (strict label) and `report-lenient.md`.

**Same 80 questions for every critic** (`subset80`, strict label; 45/80 = 56% of drafts correct, so "approve everything" is right 56% of the time; ±11 points at n = 80):

| Critic | n | Accuracy when approved (@0.8) | Wrong approvals @0.8 | Wrongly rejected @0.8 | AUROC | Mean s/decision |
|---|---:|---:|---:|---:|---:|---:|
| llm (qwen3:8b) | 80 | 41/66 (62%) | 25/35 | 4/45 | — | 9.59 |
| local-typed (qwen3:8b) | 20 | 10/19 (53%) | 9/9 | 1/11 | 0.45 | 7.26 |
| kev (Kev-0.8B) | 80 | approves none | 0/35 | 45/45 | 0.45 | 1.23 |
| laya | 80 | 12/23 (52%) | 11/35 | 33/45 | 0.50 | 0.26 |
| evidence_match | 80 | 43/69 (62%) | 26/35 | 2/45 | 0.61 | 0.00 |
| consistency | 80 | 41/68 (60%) | 27/35 | 4/45 | 0.57 | 8.82 |

**Largest set** (`extra`, 200 drafts, strict label; 115/200 = 57% correct; ±7 points):

| Critic | n | Accuracy when approved (@0.8) | Wrong approvals @0.8 | Wrongly rejected @0.8 | AUROC | Mean s/decision |
|---|---:|---:|---:|---:|---:|---:|
| llm (qwen3:8b) | 200 | 100/159 (63%) | 59/85 | 15/115 | — | 10.23 |
| kev (Kev-0.8B) | 200 | 0/1 | 1/85 | 115/115 | 0.48 | 0.86 |
| laya | 200 | 45/65 (69%) | 20/85 | 70/115 | 0.60 | 0.22 |
| evidence_match | 200 | 111/177 (63%) | 66/85 | 4/115 | 0.60 | 0.00 |
| consistency | 60 | 32/51 (63%) | 19/26 | 2/34 | 0.61 | 8.70 |

All critics run locally ($0).

![Reliability chart](reliability.png)

**Takeaways so far.**
- No critic is much better than "always approve": the best accuracy-when-approved is 62–69%, against a 56–57% baseline, and every AUROC is ≤ 0.61.
- The loop's current llm critic approves about 80% of drafts, including about 70% of the wrong ones.
- Kev-0.8B rejects almost everything at 0.8, and its ranking is at chance level.
- Laya is the only critic that is selective at 0.8, but it rejects most right answers too (70/115 on `extra`), and it is not better than chance on `subset80`.
- The free, model-free `evidence_match` ranks drafts about as well as anything else tested.
- With the lenient label (EM or F1 ≥ 0.8), 11 of the 85 wrong `extra` drafts become right. The ranking of critics does not change (see `report-lenient.md`).

**Stop rules** (`combos.md`): gating llm with evidence_match (send a draft to llm only if its answer appears in the evidence) gives the best accuracy when stopping (66% on `extra`, 69% on `subset80`, vs 62–63% for llm alone) and saves 9–12% of llm calls. Its 95% intervals still overlap "approve everything".

Not yet run: the qwen3:8b logprob critic (Step 5) and the qwen3:14b text critic (Step 6).
