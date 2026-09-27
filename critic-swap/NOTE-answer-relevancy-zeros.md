# Note: answer_relevancy = 0 on real runs (not fixed)

**Status:** observed on the first real run. The cause is likely but not
confirmed, and nothing has been changed. The zeros are still in the data.

## Setup
- SHP harness at `6c3eb99` + `critic-swap.patch`
- Groq; writer and critic `openai/gpt-oss-20b`; RAGAS judge `openai/gpt-oss-120b`
- `reasoning_effort=low`
- These replace the paper's Llama models, which this Groq key no longer has.

## What we saw
Scenario `hotpot_5ae2070a5542994d89d5b313` (ground truth: "Badly Drawn Boy"),
6 rounds:

| round | answer_relevancy | draft opens with |
|---|---|---|
| 1 | 0.96 | "Badly Drawn Boy has a higher instrument-to-person ratio…" |
| 2 | **0.00** | "The context does not provide any quantitative information…" |
| 3 | 0.96 | "Badly Drawn Boy is a solo artist who plays multiple instruments…" |
| 4 | **0.00** | "The context does not give any numbers of instruments…" |
| 5 | 0.96 | "Badly Drawn Boy has a higher instrument-to-person ratio…" |
| 6 | **0.00** | "The context does not provide enough information to determine…" |

The zeros are exact, and they line up one-to-one with the drafts that hedge.
Drafts that commit to an answer score about 0.96.

**Across the 5-scenario run** (`real_default_5dev.csv`: 30 rounds, 24
distinct drafts), this is the **only scenario with any zeros**: 3 of its 6
rounds. The other four scenarios have no zeros in any round; their
answer_relevancy ranges from 0.48 to 0.99. `ar_zero_rate` = 0.200 in the table is
this one scenario, whose final draft is a hedge under every non-oracle
policy. So the zeros are real but concentrated in one scenario, not a
general problem with gpt-oss judging.

## Likely cause
RAGAS `AnswerRelevancy` has the judge LLM generate a question from the answer
and flag whether the answer is **noncommittal**. If it's flagged, the score is
forced to 0. So the zeros are probably intended RAGAS behavior, triggered by
hedged drafts, and not a crash or parse failure. The judge scores are genuine,
not failures silently turned into 0: the only RAGAS warning was the expected
"returned 1 generations instead of requested 3", from the harness stripping
`n` for Groq.

**Not yet confirmed:**
- The noncommittal flag itself wasn't inspected. The conclusion rests on the
  zeros lining up with hedged drafts.
- Whether this is specific to gpt-oss is unknown. It could be the gpt-oss writer
  hedging more, the gpt-oss-120b judge flagging more readily, or both. There's
  no Llama baseline on the same drafts to compare against.

## Why it matters
- `answer_relevancy` is 25% of the Information Score (equal weights), so one
  zero costs a draft about 0.24 IS. That distorts `final_is`, the IS-gain halt
  signal, and `oracle_is`.
- **The critic approved a hedged draft.** In round 4 the default critic said
  `APPROVED` on "the context does not give any numbers…", which is wrong against
  the ground truth. The writer also swings between committing and hedging as the
  critic pushes back on each.

## Possible fixes (not done)
- Log the noncommittal flag, or the RAGAS per-sample output, next to each score
  to confirm the cause.
- Re-judge the same cached drafts with a non-gpt-oss judge to see whether the
  flag depends on the judge.
- Report `answer_relevancy` and the IS both with and without zeroed rows, or
  decide whether hedging should count as a failure.

`compare_critics.py` now reports `ar_mean` and `ar_zero_rate` (the share of
final drafts with answer_relevancy = 0) so this stays visible in the tables.
