# repro-yash-pilot

A reproduction of Yash's HotpotQA dev pilot on my Mac, extended from 8 to all
20 dev questions. The arms are `fixed`, `llm` and `local-typed`. The Jev arm
wasn't run; see "Jev status" below.

## What I ran

This uses Yash's prototype from the [`Yash` branch](https://github.com/acm-research-f26/Halt-IQ/tree/Yash)
at commit `cd66129`. It's a clean clone, and the code hash matches his pilot
(`77de4bc938836f6f…`).

```bash
python3 -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed --split dev --limit 8
python3 -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed --split dev --limit 20
```

These commands are reconstructed from the run manifests. Every other setting
was the default: 3-draft cap, approval threshold 0.8, seed 42, temperature 0,
thinking off, 32,768-token context.

| | |
|---|---|
| Writer and critic model | `qwen3:8b` via Ollama, Q4_K_M, 8.2B |
| Model digest | `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` (same as Yash's pilot) |
| Dataset | `data/hotpotqa/tasks.jsonl`, sha256 `9bcdd373cb93d36c…` (same as the pilot) |
| Machine | macOS 15.6, Apple Silicon, Python 3.14.5 |
| Runs | `run-20260928T130835954064Z` (8 questions), `run-20260928T131217748792Z` (20 questions), 2026-09-28 |
| Hosted API cost | $0 (all local) |

## Files

- `run-20260928T130835954064Z/` and `run-20260928T131217748792Z/`:
  `report.md`, `episodes.csv` and `summary.json` copied from each run. Traces,
  manifests and source snapshots stay in the local clone.
- `ANALYSIS.md`: the full analysis, including the reproduction checks, a
  question-by-question look at disagreements, and EM-vs-F1 cases.

## Reproduction

- The **8-question run matches Yash's pilot exactly.** All 24 episodes and all
  29 drafts are identical, word for word.
- The **first 8 questions of the 20-question run match the 8-question run
  exactly.**

## Results: 20 dev questions

| Arm | EM | F1 | Avg drafts | Critic requests | False approvals | Avg latency per question |
|---|---|---|---|---|---|---|
| fixed | 11/20 (55%) | 61.3% | 3.00 | 0 | none (no critic) | 5.3 s |
| llm | 13/20 (65%) | 71.3% | 1.20 | 24 | 7 of 19 approvals | 10.4 s |
| local-typed | 11/20 (55%) | 61.3% | 1.10 | 22 | 9 of 19 approvals | 9.3 s |

A false approval means the critic approved an answer with EM = 0. Two of those
answers are probably correct but penalized by strict exact match: #19 "Wilmette,
Illinois" and #4 "translation of and commentary on…Principia". Counting them as
correct gives about 5 of 19 (`llm`) and 7 of 19 (`local-typed`).

## Takeaways

1. **Only the `llm` critic changed any outcome, and it did so twice** (#8 and #18,
   where it caught a wrong first draft and the writer fixed it). `fixed` and
   `local-typed` ended with exactly the first draft's accuracy.
2. **Both critics approve too readily.** Nearly every episode ended in an
   approval, and roughly a quarter to over a third of approvals were wrong.
   A better critic, such as Jev, would need to reject wrong answers more often.
3. **Fewer drafts did not mean faster.** Critics cut drafts from 3 to about
   1.1–1.2 but about doubled time per question, because a critic call (~4.4–5 s)
   costs more than a short-answer rewrite. Report latency and tokens, not just
   rounds.

**Too small to claim:** at n = 20, the 10-point `llm` advantage is two
questions, so it isn't evidence that `llm` beats `fixed`. False-approval rates
rest on 19 approvals each. Latency comes from one machine and one run. Nothing
here measures Jev.

## Jev status

**Blocked on TypeSafe billing.** The official Jev arm
(`--critics fixed,llm,jev --split dev --limit 8`) hasn't been run. Once billing is
resolved, the key must be exported in the shell as `TYPESAFE_API_KEY`; Yash's code
doesn't read `.env`. The first step is a single `python3 -m haltiq judge
--backend jev --input data/example_state.json` smoke test, because the adapter
has never been checked against a live Jev response.
