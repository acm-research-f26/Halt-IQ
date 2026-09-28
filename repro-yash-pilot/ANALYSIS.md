# Analysis: reproduction of Yash's HotpotQA dev pilot

Runs: `run-20260928T130835954064Z` (8 dev questions) and
`run-20260928T131217748792Z` (all 20 dev questions). Both use the arms `fixed`,
`llm` and `local-typed` on `qwen3:8b`. Question numbers (#N) below are
positions in the 20-question dev split.

## 1. The 8-question run matches Yash's pilot exactly

The comparison is against `results/hotpotqa-dev-pilot/` on the `Yash` branch (commit `cd66129`).

- Settings, dataset hash (`9bcdd373…`) and code hash (`77de4bc9…`) are identical.
  Every call in the traces was answered by `qwen3:8b`.
- All 24 episodes (8 questions × 3 arms) are identical: final answer, EM, F1,
  drafts, stop reason and critic calls.
- **All 29 drafts are word-for-word identical** to the pilot's.

## 2. The first 8 questions of the 20-question run match the 8-question run

All 24 episodes and every draft are identical. The 20-question run is the
8-question run plus 12 more questions.

## 3. Results on 20 dev questions, one table per critic

Latency is the mean total time per question, including the shared first draft.

**fixed** (no critic; self-revision to 3 drafts)

| EM | F1 | Avg drafts | Critic requests | False approvals | Avg latency |
|---|---|---|---|---|---|
| 11/20 (55%) | 61.3% | 3.00 | 0 | none (no critic) | 5.3 s |

**llm** (Qwen writes a critique and approves or rejects)

| EM | F1 | Avg drafts | Critic requests | False approvals | Avg latency |
|---|---|---|---|---|---|
| 13/20 (65%) | 71.3% | 1.20 | 24 | 7 of 19 approvals | 10.4 s (4.97 s per critic call) |

**local-typed** (Qwen scores supported/complete/relevant; approves if all ≥ 0.8; not Jev)

| EM | F1 | Avg drafts | Critic requests | False approvals | Avg latency |
|---|---|---|---|---|---|
| 11/20 (55%) | 61.3% | 1.10 | 22 | 9 of 19 approvals | 9.3 s (4.43 s per critic call) |

## 4. Where the arms disagree

Only two questions split the arms. In both, the shared first draft is wrong and
`llm` catches it:

- **#8** "What year was a joint venture between RLJ Companies and this film
  studio founded in 2005, founded?" (gold: 2006). The first draft says 2005.
  `llm` rejects it ("…was founded in 2006") and the writer fixes it. `local-typed`
  approves 2005, and `fixed` keeps 2005 through all three drafts.
- **#18** "Gerd Neggo trained under the founder of which type of dance analysis?"
  (gold: Laban Movement Analysis). The first draft says "modern dance". `llm`
  rejects it and the writer corrects it. `local-typed` approves the wrong answer,
  and `fixed` keeps it.

## 5. EM says wrong, but the answer is probably right

In both cases every arm gives the same answer, so these aren't disagreements, but
they do inflate the false-approval counts:

- **#19** "…American rock band formed in which city?" (gold: Wilmette). All arms
  answer "Wilmette, Illinois" (F1 0.67). The answer is correct; exact match only
  penalizes the added state name.
- **#4** "How did Emilie du Chatelet … contribute to the basic laws of physics?"
  (gold: commentary on Isaac Newton's book "Principia"). All arms answer "translation of and
  commentary on Isaac Newton's book 'Principia' containing basic laws of…"
  (F1 0.60). This is essentially correct, just more detailed than the gold answer.

Counting these two as correct, false approvals are about **5 of 19** for `llm` and
**7 of 19** for `local-typed`.

## 6. Other observations

- The first draft was already right on 11 of 20 questions.
- **`fixed`:** self-revision changed the wording on #3 and #5 but never changed
  correctness. Its EM equals first-draft EM.
- **`local-typed`:** never rejected a wrong answer, so its EM also equals first-draft EM.
- **Correct answer rejected:** both critics rejected the correct answer to #3
  (MedStar Washington Hospital Center) on all three drafts. The episode used the
  full draft cap and still ended correct.
- **Latency:** a critic call (~4.4–5 s) costs more than a short-answer rewrite
  (~1.8 s per draft, inferred from `fixed`: 3 drafts in 5.3 s). So the critic
  arms use fewer drafts but take about twice as long per question.

## 7. Takeaways

1. **Only the `llm` critic changed any outcome, and it did so twice (#8, #18).**
   It fixed two wrong first drafts and broke none. `local-typed` and `fixed`
   ended with exactly the first draft's accuracy.
2. **Both critics approve too readily.** 19 of 20 episodes in each critic arm
   ended in an approval, mostly on the first draft. Roughly a quarter to over a
   third of approvals were wrong, even after adjusting for EM strictness. This
   is the gap a better critic such as Jev needs to close: rejecting wrong answers.
3. **Fewer drafts did not mean faster.** Critics cut drafts from 3 to about
   1.1–1.2 but doubled time per question, because critic calls cost more than
   short-answer rewrites. Rounds saved is the wrong way to measure cost here;
   report latency and tokens.

## 8. Too small to claim

- **"`llm` beats `fixed`"** isn't supported. The whole 10-point EM gap is two
  questions. It points the right way but proves nothing.
- **False-approval rates** rest on 19 approvals per arm and move a lot with the
  EM-strictness adjustment.
- **Exact reproduction** shows the setup is stable at temperature 0 on this
  hardware. It says nothing about variation across seeds, models or question sets.
- **Latency** comes from one Mac, one run and one model.
- **Nothing here measures Jev.** The Jev arm hasn't run.
