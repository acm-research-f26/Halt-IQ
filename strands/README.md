# strands/

Strands Decider 2B (`StrandsAgents/strands-decider-2B-hobson-v19`) as a HaltIQ critic, with a
MiniLM / bge reranker choosing the evidence paragraphs. Copied from `strands-reranker/` on the
`vedanshi` branch (commit d0688ae).

- `Vedanshi_StrandsCritic.ipynb`: Colab notebook (GPU) that serves the decider and scores the 220 dev drafts.
- `metrics.py`: recomputes `results/` from the saved CSVs, no model calls. It imports
  `critic-bench/report.py`, which is not on this branch yet, so it does not run here as-is.
- `results/`: per-draft scores (`haltiq_strands_experiments*.csv`), metrics tables, AUROC chart.
- `WRITEUP_DRAFT.md`, `WRITEUP_TEMPLATE.md`: write-up and the template `metrics.py` fills.
- `sri_trajectory_replay.ipynb`: Oct 23 Colab notebook (T4 GPU), one cell, run on Sri's 150 saved
  6-round trajectories (`implementation-3/study150/cache/` on the `sri` branch). No API keys.
  - Step A scores every unique draft with Strands + bge-reranker-base top-2 paragraphs
    (question + draft as the rerank query) and saves `../data/haltiq_sri_trajectory_strands.csv`
    (in Colab it writes to `MyDrive/`; the copy here is that output).
  - Step B replays stopping policies over the saved rounds, no model calls: `fixed1`, `fixed6`,
    `sri_llm_critic` (stop when Sri's LLM critic approves), `answer_unchanged` (stop once a revision
    repeats the answer), `strands_gate` (stop when Strands >= 0.8), and `orchestrator` (stop when
    Strands >= 0.8, otherwise revise until the answer repeats). The 0.8 threshold was fixed in advance.
  - On the 80 test questions, `orchestrator` reaches F1 0.378 at 19% of `fixed6` LLM tokens;
    `answer_unchanged` matches `fixed6` (F1 0.392) at 28%. Full tables are in the notebook output.

## `../data/haltiq_sri_trajectory_strands.csv`

Columns `id, answer, strands`: one Strands score per unique (question id, draft answer) pair.
Sri's cache has 900 drafts (150 x 6 rounds) that reduce to 166 unique pairs; all 166 match the
cache exactly by `id` and answer text (no extra, missing, or duplicate rows), checked against
`sri` at commit 5503cab. Look up a draft's score by `(id, answer)`, not by round.

Note: the notebook clones the `sri` branch at whatever its head is, not a pinned commit.
