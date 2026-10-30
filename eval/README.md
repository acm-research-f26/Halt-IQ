# eval/

Evaluation that combines the Strands critic (`strands/`) with the stopping classifier
(`classifier/`): stopping-rule replays, metrics, and reports. All scripts replay
`data/sri_rounds_with_strands.csv` on the 80 test questions; no model calls.

- `answer_selection.py` (`python3 eval/answer_selection.py`): for each stopping rule, returns the
  last answer vs. the best-Strands answer seen so far vs. an F1 oracle (upper bound only), with
  paired bootstrap CIs and answer-change counts. Writes `answer_selection.csv`.
- `compare_all.py` (`uv run --with matplotlib python eval/compare_all.py`): one table of fixed
  1-6 rounds, answer repeats, the routing rule, Sri's critic stop, the best answer-selection variant,
  and Sri's retrained stop heads (pending until `classifier/results/<head>/**/per_question.csv`
  exists), each against its closest-token fixed-round baseline. Writes `results_table.csv` and
  `f1_vs_tokens.png` (which also draws the routing threshold sweep from `headroom.py`).
- `headroom.py` (`python3 eval/headroom.py`; run after `compare_all.py`, whose table it reads):
  oracle stopping (earliest best-F1 round; upper bound) and each method's share of the possible
  gain over fixed 1; the routing threshold sweep (0.50-0.95, picked on train+dev at <= 30% tokens,
  reported on test); help vs. hurt counts per round transition; and the untuned rules on all
  150 questions. Writes `headroom.csv` (a `section` column says which part each row is) and
  `help_hurt.csv`.

Token accounting is shared and documented in `answer_selection.py`: writer tokens every round,
LLM critic tokens only when the critique is used, Strands calls counted separately, and
100% = fixed 6 rounds (writer 1-6 + critic 1-5).
