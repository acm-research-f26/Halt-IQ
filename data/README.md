# data/

Placeholder for shared inputs: HotpotQA question splits, saved trajectories, and per-draft critic scores.
Record the source and checksum of anything added here.

- `haltiq_sri_trajectory_strands.csv`: Strands + bge top-2 score per unique (id, answer) draft in
  Sri's study150 cache (166 pairs). Produced by `../strands/sri_trajectory_replay.ipynb`.
- `attach_strands.py`: joins those scores onto every round of Sri's cache, read from the `sri`
  commit 5503cab with git. Run `python3 data/attach_strands.py`; it exits without writing if any
  round lacks a score.
- `sri_rounds_with_strands.csv`: its output, 900 rows (150 questions x 6 rounds): id, split
  (train 50 / dev 20 / test 80), round, answer, strands, f1, em, critic_approved, tokens, and writer/critic token usage.
