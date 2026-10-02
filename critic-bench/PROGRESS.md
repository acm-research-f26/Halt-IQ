# Progress report 10/02: critic benchmark

**My focus this week:** which critic should decide when the writer–critic loop stops. I tested open decision models, open LLM critics, and cheap signal-based checks on the same drafts.

All code is in `critic-bench/` on the `vedanshi` branch. Full tables: `critic-bench/report.md`, `combos.md`, `README.md`.

## Setup

- **Fixed drafts, every critic.** Drafts come from Yash's loop (his writer, qwen3:8b, unmodified). Every critic judges the same saved answers, so the comparison is fair and nothing is regenerated.
- **200 new dev questions.** Drawn from the official HotpotQA dev split (7,405 questions) with a seeded hash, excluding all 100 questions already in use (20 dev and 80 held-out). The held-out test set is untouched. This cut the uncertainty on accuracy from about ±22 points (20 drafts) to about ±7 points.
- **Label:** exact match, the same label as `analyze_run.py`. A lenient label (exact match or F1 ≥ 0.8) doesn't change the overall picture.

## What I built

- `score_drafts.py` / `critics.py`: a scoring harness that runs any critic on any saved draft set, with caching and timing. Adding a critic is one function.
- New critics:
  - **Laya**: an open decision model (yes/no probabilities), run locally on Apple silicon.
  - **evidence_match**: is the answer literally in the evidence? No model, free.
  - **consistency**: does the writer give the same answer when asked 3 more times?
  - **logprob**: qwen3:8b's probability of "yes" from token logprobs, turning an open LLM into a probability critic.
- `report.py`: AUROC, wrong approvals, wrong rejections, threshold sweep, reliability chart, confidence intervals.
- `combos.py`: stop rules that combine critics, with cost.
- `combine.py`: a learned combination (logistic regression, 5-fold cross-validation).

## Results: single critics (200 new questions)

Approving every draft is right 57% of the time. A useful critic has to beat that when it says "stop."

| Critic | Right when it approves | Wrong answers approved | Right answers rejected | AUROC | Seconds per decision |
|---|---:|---:|---:|---:|---:|
| Current LLM critic (qwen3:8b) | 63% | 59 / 85 | 15 / 115 | — | 10.2 |
| logprob (qwen3:8b P(yes)) | 65% | 53 / 85 | 18 / 115 | **0.63** | 7.1 |
| Laya (≥ 0.8) | 69% | 20 / 85 | 70 / 115 | 0.60 | 0.2 |
| evidence_match | 63% | 66 / 85 | 4 / 115 | 0.60 | **0** |
| consistency (all 3 agree) | 62% | 65 / 85 | 8 / 115 | 0.58 | 8.7 |
| Kev-0.8B (≥ 0.8) | approves 1 | 1 / 85 | 115 / 115 | 0.48 | 0.9 |

AUROC = chance a random right answer scores above a random wrong one (0.5 = coin flip). The current LLM critic only gives approve/revise, so it has no AUROC.

## Results: combining signals

| Stop rule (200 new questions) | Right when it stops | Wrong answers approved | Right answers rejected | Seconds per draft |
|---|---:|---:|---:|---:|
| Current LLM critic alone | 63% | 59 | 15 | 10.2 |
| Gate: evidence_match first, LLM only if it passes | 66% | **50** | 16 | 9.2 (9% fewer LLM calls) |
| consistency AND evidence_match (no critic model) | **67%** | 52 | **11** | 7.8 |
| Learned combination (cross-validated) | 65% | 56 | 12 | — |

Learned combination AUROC: **0.69**, the best of anything tested (best single critic: 0.63). evidence_match and consistency carry the most weight.

## What this means

1. **The current LLM critic barely beats approving everything** (63% vs 57%), approves 59 of 85 wrong answers, and costs about 10 s per decision.
2. **A free check matches it.** evidence_match is as accurate as the LLM critic at zero cost.
3. **Cheap signals combined beat the current critic.** Requiring the writer to agree with itself and the answer to appear in the evidence gives fewer wrong approvals, fewer wrong rejections, and about 25% less time, with no critic model.
4. **No single critic is reliable yet** (AUROC ≤ 0.63). The critic is the bottleneck, not the threshold, so tuning the threshold won't help much until a critic has more signal.
5. **This sets Jev's bar.** To be worth paying for, Jev has to beat a free evidence check and the combined rule.

## Caveats

- **Not statistically significant yet.** The improvements are consistent, but the 95% intervals overlap. They need a larger test to confirm.
- Small samples overstate results: consistency looked better on 60 questions (AUROC 0.61) than on 200 (0.58).
- Laya's 512-token limit truncates most inputs (19 of 20 on the first set), so it only sees part of the evidence.
- The logprob critic is very overconfident: 138 of 200 scores are ≥ 0.99.
- qwen3:14b as a critic ran out of memory on a 24 GB laptop (about 22 s per draft, rising past 115 s); stopped at 9 of 80. A bigger local critic isn't practical for us.
- Writer errors worth fixing separately: answering "UNKNOWN," and giving an entity for yes/no questions.

## Next steps

- **Score Jev on the same 200 drafts** for a direct comparison with the current critic and the free checks.
- **Jev inside the gate:** free check first, Jev only when the answer passes (cheaper routing).
- Confirm the combined rule on new dev questions before the team uses it.
- Possible inputs for the learned stop policy: evidence_match and consistency are cheap signals with real weight.
