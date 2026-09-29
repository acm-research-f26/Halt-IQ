# Progress report 09/28

**The question we're working on:** can a better *critic* stop a writer–critic loop at the right time, cheaply, without approving wrong answers? Jev (TypeSafe AI) is the candidate: it returns yes/no probabilities instead of text, and claims to be fast and calibrated.

Everything below is on the `vedanshi` branch of `acm-research-f26/Halt-IQ`.

| Folder | What it is |
|---|---|
| `critic-swap/` | Swappable critic patch for the SHP harness + first real run |
| `repro-yash-pilot/` | Reproduction and extension of Yash's HotpotQA pilot |
| `openjev-critic/` | Open-source Jev-like critic (Kev-0.8B) experiment |

---

## 1. Swappable critic in the SHP harness

**Problem:** in the Semantic Halting Problem (SHP) harness, the same model plays writer *and* critic, so you can't change only the critic.

**What I built** (a patch against upstream commit `6c3eb99`):
- `--critic {default, jev}` flag to choose the critic.
- A Jev critic: Jev decides approve/revise with a confidence; on approve the loop stops with no LLM call; on revise a normal LLM writes the feedback (Jev can't write text). Stubbed, since there's no API access.
- The Jev critic emits the same `APPROVED` signal as the original, so all halting logic works unchanged.
- Extra per-round logging: writer/critic latency, critic confidence, critic LLM calls.
- `compare_critics.py`: runs the same scenarios per critic and outputs a comparison table.

**Checks:** with the default critic, output is byte-identical to the original harness (no regression); theory checks pass.

**First real run** (default critic, 5 dev scenarios, gpt-oss-20b writer/critic, gpt-oss-120b judge, on Groq):

| Stop policy | Success | Avg rounds |
|---|---|---|
| critic_only | 3/5 | 2.6 |
| shp (full cascade) | 4/5 | 2.2 |
| fixed_k6 (always 6) | 3/5 | 6.0 |
| oracle_is (best round in hindsight) | 4/5 | 3.6 |

**Findings along the way:**
- Groq and NVIDIA no longer serve the paper's Llama models, so numbers aren't directly comparable to the paper.
- Answer-relevancy zeros all came from one question where the writer flip-flopped and hedged.
- "Critic" means an LLM reviewer in SHP but the PPO value head in Tushaar's veRL code.

---

## 2. Reproducing and extending Yash's pilot

Yash built a standalone prototype on HotpotQA: a local Qwen3-8B writer, swappable critics, and scoring by exact match (EM) against the gold answer.

**What I did:**
- Ran it on my Mac with the same model build (`500a1f067a9f`): the 8-question run **matches his pilot exactly, every draft word for word.**
- Extended it to all **20 dev questions.** (Yash's own 20-question run later matched mine exactly, so it reproduces across two machines.)
- Analyzed the results.

| Critic | Correct (EM) | Avg drafts | Wrong approvals | Time per question |
|---|---|---|---|---|
| fixed (no critic, 3 drafts) | 11/20 | 3.00 | n/a | 5.3 s |
| llm (Qwen writes a critique) | 13/20 | 1.20 | 7 of 19 | 10.4 s |
| local-typed (Qwen writes scores) | 11/20 | 1.10 | 9 of 19 | 9.3 s |

**Findings:**
1. **Only the LLM critic changed any outcome:** it fixed 2 wrong first drafts and broke none. Revising without a critic never changed correctness.
2. **Critics approve too easily:** roughly a quarter to a third of approvals were wrong answers, even after correcting for EM strictness (e.g. "Wilmette, Illinois" vs "Wilmette").
3. **Fewer drafts ≠ faster:** a critic call (~5 s) costs more than a short rewrite (~1.8 s, estimated), so critic arms took about twice as long. Report time and tokens, not just rounds.

---

## 3. Official Jev: blocked on access

- Yash's Jev adapter is built and tested against the API format, but nobody on the team has live access.
- I created a TypeSafe account. TypeSafe has suspended the free $5 credit for new accounts (citing misuse), and two payment attempts failed on their side.
- Cost once access works: about $0.0001 per request, so the full 100-question set costs a few cents.

---

## 4. Open-source Jev-like critic: Kev-0.8B

Following Tushaar's suggestion, I tested an open model that works like Jev: it reads the model's probability for "yes" vs "no" instead of generating text.

**Why Kev-0.8B:** it speaks the same API as Yash's `openjev` critic slot (no code changes), and fits in memory next to the writer. mini-jev had no compatible API; Kev-4B (~14 GB) doesn't fit alongside the 10 GB writer on a 24 GB Mac.

**Results (20 dev questions):**

| Critic | Correct | Avg drafts | Approvals | Time per decision |
|---|---|---|---|---|
| openjev (Kev-0.8B) | 11/20 | 3.00 | 0 | ~0.4 s |

- Kev is **fast** (~0.4 s vs ~5 s for the Qwen critics) but **approved nothing** at the 0.8 threshold.
- **Offline threshold analysis** (first drafts only, no new model calls): Kev's scores rank correct vs wrong answers at about chance (AUROC ≈ 0.53). No threshold from 0.45 to 0.80 helps; at low thresholds it approves wrong answers at the base rate.
- **local-typed** only ever outputs 1.0 or 0.0 and approved every wrong first draft (AUROC ≈ 0.46): a rubber stamp.
- Temperature scaling can't fix this: it changes how confident scores look, not their order, so AUROC stays the same.
- Likely cause, consistent with Kev's own docs: a sub-1B model, weak on knowledge, trained on short texts (≤ 384 tokens), while HotpotQA needs multi-hop checks over long evidence.

**Takeaway on "text vs probabilities":** probabilities give graded scores (the right shape) where text scores were all-or-nothing, but this small model's probabilities carry no signal on this task.

---

## 5. Caveats

- n = 20 dev questions is a pilot. The llm-vs-fixed gap is 2 questions.
- EM is strict and undercounts some correct answers; F1 is kept alongside.
- The threshold analysis covers first-draft decisions only, not the full loop.
- The 80 held-out questions are untouched, reserved until the threshold and prompts are frozen.

---

## 6. Questions for the professor

1. **Fair comparison:** should critics be compared on identical drafts, or as full feedback-and-stopping systems?
2. **Calibration:** is an external model's confidence a sound input for a learned stop policy? How should we test calibration here?
3. **Model size:** small open decision models were fast but showed no signal on multi-hop QA. Is it worth testing Kev-4B or official Jev on a bigger machine?
4. **Sample size:** how many questions would make a critic comparison credible, and which statistical test?

---

## 7. Next steps

- Get Jev access and run the official Jev arm on the same 20 dev questions.
- Test Kev-4B on the saved first drafts (fits in memory with the writer turned off).
- Tune the threshold on dev, then run the 80 held-out questions.
- Plug Yash's Jev adapter into the SHP harness, for results in two setups.

---

**Credits:** Yash built the HotpotQA prototype, the Jev adapter, the `openjev` arm and `analyze_run.py`. Tushaar suggested open Jev-like models. The SHP harness is by Sahil Shrivastava. Kev is by Jared Palmer ([GitHub](https://github.com/jaredpalmer/kev), [model card](https://huggingface.co/jaredpalmer/kev-0.8b)).
