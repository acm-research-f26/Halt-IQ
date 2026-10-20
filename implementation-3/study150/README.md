# 150-question real HotpotQA study

Time-bounded expansion requested 16 October 2026. 50 train, 20 validation, 80 test; all 80 old pilot IDs excluded. Selected by preserved partition order from the seeded 600-question design, before test generation or scoring. Exactly ten supplied passages per question. The original 600-question plan remains available in ../study.

## Resume (only if the current run has stopped)

Keep the MLX server running on port 8080 in its own terminal. In a second terminal:

```bash
cd Halt-IQ
source .venv/bin/activate
cd implementation-3
python run_study150.py
```

This skips complete trajectories, resumes partial questions, then trains five seeds and evaluates all seven policies. Do not launch multiple generators simultaneously. Logs: run.log, cache/calls.jsonl. Final results: results/all_seeds.json and results/seed0 through seed4. Each seed folder contains a checkpoint, history, REPORT.md, evaluation_detailed.json and per_question.csv.

All policies include critic calls through the selected round. F1/EM are answer metrics only. Tokens include both input and output, including cached input tokens. Timing is a sum of actual cached request durations, not independently measured live-policy latency. Report all seeds. Oracle F1 uses gold answers and is only a diagnostic. The test partition is a local holdout from official development data, not official hidden test data.

The research report explains the earlier pilot and the expanded study separately because their passage counts differ. Full methods and source references: ../study/README.md.
