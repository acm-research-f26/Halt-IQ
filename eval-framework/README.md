# eval-framework v0: every stopping policy on one cost–quality scale

**Goal.** Compare every way of deciding when the writer–critic loop stops on the same
two axes: how much cost a policy saves, and how much quality it gives up. Each team
exports its runs into one shared file format (below); `compare.py` turns all of them
into one table and one cost–quality plot.

```bash
cd eval-framework
python3 adapters/yash_run.py   # Yash's Sept 28 run -> results/yash-run-20260928T234403183653Z.jsonl
python3 compare.py             # every results/*.jsonl -> compare.md + cost_quality.png
```

No model calls anywhere in this folder. Yash's repo is only read.

## The results schema (`schema.py`)

JSONL: **one row per question per policy**. `compare.py` reads every `results/*.jsonl`.

| Field | Required | Type | Meaning |
|---|---|---|---|
| `question_id` | yes | str | Stable ID, e.g. `hotpotqa:5ab833ad5542990e739ec86f` (Yash's task IDs) |
| `split` | yes | str | `dev` or `test`. `compare.py` uses one split at a time (default `dev`); never mix them |
| `policy` | yes | str | Stopping policy name, e.g. `fixed-3`, `llm-critic`, `info-gain-0.1` |
| `source` | yes | str | Run ID or file it came from, for tracing |
| `final_answer` | yes | str | The answer the policy returns |
| `rounds` | yes | int | Drafts written before stopping (1 = stopped after the first draft) |
| `total_tokens` | no | int | All tokens spent on the question (input + output, writer + critic), or `null` |
| `seconds` | yes | float | Model time on the question: the sum of model-call latencies |
| `exact_match` | yes | bool | HotpotQA exact match after normalization |
| `token_f1` | yes | float | HotpotQA token F1, 0–1 |
| `info_score` | no | float | Optional information score (Sid's or Sri's method), or `null` |
| `cost_usd` | no | float | API spend (0 for local models), or `null` |

Example row:
```json
{"question_id": "hotpotqa:5ab833ad5542990e739ec86f", "split": "dev", "policy": "llm-critic", "source": "run-20260928T234403183653Z", "final_answer": "Aguascalientes", "rounds": 1, "total_tokens": 2353, "seconds": 5.1, "exact_match": false, "token_f1": 0.0, "info_score": null, "cost_usd": 0.0}
```

**Exporting your runs (Sid, Sri).** Write a small adapter (copy `adapters/yash_run.py`) or
build rows directly, then save them with `schema.write_rows(path, rows)`, which checks every
row. Put the file in `results/` and rerun `compare.py`. To be comparable:
- Use the **same questions** (Yash's 20 dev task IDs from `halt-iq-yash/data/hotpotqa/tasks.jsonl`, or the 200 `extra` questions in `critic-bench/extra/tasks.jsonl`), and the dev split only.
- Score answers with **Yash's `haltiq/metrics.py`** (`hotpot_evaluate`), so EM and F1 mean the same thing.
- Count **seconds** as the sum of model-call latencies, and **tokens** as input + output for every call, including the first draft and any critic or scoring calls.
- If your policy uses an information score to stop, put the value at the stopping round in `info_score`.

## What Yash's Sept 28 run records

| Recorded | Where | Used for |
|---|---|---|
| Rounds per episode, stop reason | `episode` events | `rounds` |
| Every round's draft, EM, F1 | `episode.round_trace` | final answer, EM, F1, oracles |
| Tokens (in/out) and latency of every call | `shared_writer`, `writer`, `critic` events | `total_tokens`, `seconds`, per-round cost |
| Cost in USD | every call | `cost_usd` (all 0: local models) |

Missing or limited:
- **No information score** is recorded, so `info_score` is `null`.
- **The round cap is 3**, so there is no fixed-6 baseline. The baseline here is `fixed-3` (always revise to the cap).
- **Kev token counts** use Kev's own tokenizer, so `kev-critic` tokens mix two tokenizers.
- **Only 20 questions**, so every difference in the table is within noise (±22 points on EM).
- Per-episode token totals match the sum of the individual calls in all 80 episodes, so the per-round costs used by the oracles are exact.

## Policies from Yash's run (`adapters/yash_run.py`)

| Policy | What it does |
|---|---|
| `one-draft` | Stop after the shared first draft (no critic) |
| `fixed-3` | Always revise to the cap (the run's `fixed` arm) |
| `llm-critic` | Stop when qwen3:8b approves (the `llm` arm) |
| `local-typed` | Stop when qwen3:8b's three self-scores are all ≥ 0.8 |
| `kev-critic` | Stop when Kev-0.8B's three noul scores are all ≥ 0.8 (the `openjev` arm) |
| `oracle-fixed` | Upper bound: on the fixed-3 trajectory, stop at the earliest round with the best F1 |
| `oracle-any-arm` | Upper bound: the best F1 among every draft any arm produced, reached most cheaply (cost = what that arm spent up to that round) |

The oracles look at the gold answer, so they show the best any stopping policy could do; they are not usable policies.

## Results v0 (`compare.md`, `cost_quality.png`)

| Policy | EM | F1 | Avg rounds | Avg tokens | Avg seconds | Tokens saved vs fixed-3 | Seconds saved | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one-draft | 55% | 0.613 | 1.00 | 1,777 | 6.24 | 67% | 13% | +0.000 |
| oracle-fixed (upper bound) | 55% | 0.613 | 1.00 | 1,777 | 6.24 | 67% | 13% | +0.000 |
| fixed-3 | 55% | 0.613 | 3.00 | 5,406 | 7.18 | 0% | 0% | +0.000 |
| oracle-any-arm (upper bound) | 65% | 0.713 | 1.10 | 2,333 | 7.36 | 57% | −3% | +0.100 |
| kev-critic | 55% | 0.613 | 3.00 | 11,144 | 8.65 | −106% | −20% | +0.000 |
| local-typed | 55% | 0.613 | 1.10 | 4,088 | 13.65 | 24% | −90% | +0.000 |
| llm-critic | 65% | 0.713 | 1.20 | 4,485 | 14.65 | 17% | −104% | +0.100 |

![Cost vs. quality](cost_quality.png)

**Reading it (20 questions, so treat as a direction, not a result):**
- **Revising without feedback adds nothing:** `fixed-3` never improves on the first draft, so `oracle-fixed` = `one-draft`.
- **The llm critic's feedback is what helps:** it repairs 2 answers (EM 55% → 65%). It reaches the same quality as the any-arm oracle, but at twice the time, because writing feedback is slow.
- **The headroom:** a stopping rule as good as the oracle would get llm-critic's quality at about `fixed-3`'s cost (7.4 s vs 14.7 s per question), revising only the 2 questions that need it.
- **Kev and local-typed** cost more than one-draft and gain nothing.
- **Tokens and seconds disagree** for critic arms: the llm critic spends fewer tokens than fixed-3 but takes twice as long. Report both.

## Critic-bench stop rules (separate section in `compare.md`)

Vedanshi's `critic-bench` stop rules were scored on 200 more dev questions, but only on
their **first drafts** (later rounds were never run). So they appear as *"stop after round 1
if the rule approves"*: how many drafts each rule stops on, the EM and F1 of those, and the
critic's cost. Rejected drafts would go on to round 2, which was not run, so these rows are
**not** on the same scale as the main table. The best rules (the evidence_match gate and
consistency + evidence_match) stop on about 75–78% of drafts at 66–67% EM, against 57% for
stopping on everything.

## Files

| File | What it does |
|---|---|
| `schema.py` | The schema (`FIELDS`), `validate`, `write_rows`, `read_rows` |
| `adapters/yash_run.py` | Converts a Yash harness run into schema rows (one-draft, fixed-3, the critic arms, two oracles) |
| `compare.py` | Per-policy table, cost–quality plot, critic-bench section → `compare.md`, `cost_quality.png` |
| `results/` | Schema files, one per source run |

## Next

- **Sid and Sri:** export runs on the same 20 dev questions (or the 200 `extra` ones) into `results/`.
- **Rerun the loop on the 200 `extra` questions** so the critic-bench rules get full trajectories (rounds 2–3) and land on the main scale.
- **Add confidence intervals** to the table once more questions are in.
