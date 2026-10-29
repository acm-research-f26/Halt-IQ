# Sri: Strands feature ablation for the linear stopping head

This directory adds an isolated classifier experiment to the integration branch. The existing `data/`, `strands/`, and `eval/` files are unchanged. All code, checkpoints, reports, and derived inputs added by this experiment are under `classifier/`.

## Result

On the **same 80 test questions**, averaged across **seeds 0–4**:

| Head | Inputs / parameters | Answer F1 | EM | Qwen tokens (% of Fixed-6) | Average rounds |
|---|---:|---:|---:|---:|---:|
| Original | 5 / 6 | 0.3588 | 0.2125 | 16.61% | 1.00 |
| (a) Replace `critic_approved` with Strands | 5 / 6 | 0.3588 | 0.2125 | 16.61% | 1.00 |
| (b) Add Strands as a sixth input | 6 / 7 | 0.3588 | 0.2125 | 16.61% | 1.00 |

All 15 heads stop at round one on all 80 questions. Seed standard deviation is zero for these reported metrics; this is not zero population uncertainty. The retrained original heads match the pinned original checkpoint weights **exactly for all five seeds**. Adding the critic score did not improve the learned stopping decisions under this training setup. No test-based threshold tuning or best-test-seed selection was performed.

**Token definition:** sum of original cached Qwen writer + Qwen critic input/output tokens through the selected round, divided by the sum for six complete rounds over the same 80 questions. This is a ratio of sums, not an average of question-level percentages. Each head uses 3,027.7125 mean Qwen tokens versus 18,227.3625 for Fixed-6. Strands and reranker overhead is unavailable and excluded, so these are NOT end-to-end integrated savings. Both new variants require a Strands score at the selected round. See `REPORT.md` for the notebook accounting difference.

## What was trained

- Existing six-round Qwen trajectories; **no new Strands writer loop was generated**.
- Original 50 train / 20 validation / 80 test split, checked against `sri` commit `5503cabeb2ec0094413b7d491ccb08ec9c991c61`.
- Strands + BGE top-2 scores joined by exact `(id, answer)`: 166 distinct drafts covering 900 rows, no missing/duplicate keys.
- Original algorithm: 20 epochs, eight sampled stopping paths per question, Adam 0.01, gradient norm clip 1, reward `F1 - 0.1 * Qwen prefix tokens / Qwen six-round tokens`.
- Choose highest mean validation reward, keep earliest tie; deterministic stop threshold 0.5; force stop at round six.
- Single update per group, no reference-model KL penalty. This retains the original simplified group-relative policy-gradient algorithm, not full GRPO.
- All 15 training runs finish before test evaluation. The existing 80 test questions were previously evaluated in the original study; they are not a new untouched holdout.

## Prediction API: p(stop), not p(correct)

The classifier estimates whether to stop internal revision. Its score is not calibrated correctness confidence; do not use it as `p(correct)` or report correctness AUROC for it.

From the repository root:

```python
from classifier import StopPredictor

controller = StopPredictor('classifier/results/add/seed0/stop_head.pt')
rounds_so_far = [
    {'round': 1, 'answer': 'Example answer',
     'critic_approved': True, 'strands': 0.82}
]
p_stop = controller.predict(question='Example question?', rounds_so_far=rounds_so_far)
stop = controller.should_stop('Example question?', rounds_so_far)
```

Inputs must be the contiguous observed prefix from round one. `question` is an interface placeholder; the numeric model does not embed question text. No gold, F1, EM, future rounds, or future token totals are read at prediction time. Distances are recomputed from observed answer text. The round fraction uses the checkpoint's fixed cap **6**, never `len(rounds_so_far)` as the denominator. A new five-round live loop requires a separately specified/retrained configuration; do not silently swap caps.

Original and add variants need Boolean `critic_approved`. Replace needs only `answer` and a finite `strands` score in [0,1]; the original critic need not be called to compute that variant's current feature. However, cached later writer drafts were produced using original critic feedback, so this experiment is not evidence about a fully replaced live critic loop.

Equivalent CLI:

```bash
python -m classifier.predict --checkpoint classifier/results/add/seed0/stop_head.pt --input prefix.json
```

`prefix.json`: `{"question":"...", "rounds":[{"round":1,"answer":"...","critic_approved":true,"strands":0.82}]}`.

## Reproduce

Python 3.13 was used. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r classifier/requirements.txt
python -m unittest classifier.test_model
python -m classifier.train
python -m classifier.evaluate
```

The joined inputs and all checkpoints are included. Training/evaluation use CPU; no model API, GPU, or new scoring is required. Commands overwrite only the derived files under `classifier/results/`.

Optional: rebuild/audit joined inputs against immutable Git data (requires that original commit locally):

```bash
git fetch origin sri
python -m classifier.prepare_inputs
```

For a shallow clone that lacks the old commit, fetch `5503cabeb2ec0094413b7d491ccb08ec9c991c61` explicitly. Preparation does not modify the original CSVs. It also restores the original head code/checkpoints under `reference/` for parity checks.

## Files

- `model.py`, `predict.py`: prefix-only features, model and prediction API/CLI.
- `train.py`, `evaluate.py`: controlled training and locked test replay.
- `prepare_inputs.py`: original-data join, exact input checks and provenance hashes.
- `test_model.py`: feature/inference parity, missing-input validation, no-label feature dependence, prefix cap, forced stop tests.
- `reference/`: unmodified original head code and five old checkpoints, attributed to the pinned `sri` commit.
- `inputs/trajectories.json`: minimal training/evaluation replay inputs; includes labels for offline rewards only.
- `inputs/manifest.json`: source commits, hashes and dataset audit.
- `results/{original,replace,add}/seed{0..4}/`: trained weights and histories.
- `results/summary.csv`, `summary_by_seed.csv`: requested metrics and seed-level details.
- `results/per_question.csv`: 1,200 selected test outcomes (3 variants × 5 seeds × 80 questions).
- `results/per_round_predictions.csv`: 7,200 prefix-only p_stop values; labels are included strictly for offline diagnostics.
- `results/baselines.csv`, `baseline_per_question.csv`: fixed rounds 1–6 and pre-existing simple gate/distance policies under the same complete-round cost definition.
- `results/original_parity.json`: exact five-seed original-checkpoint reproduction.
- `results/paired_f1.json`: paired question-bootstrap differences vs original (5,000 resamples, seed 42; seed-averaged per question, n=80).
- `results/critic_diagnostic.json`: Strands-score AUROC on 80 test FIRST drafts, not controller correctness AUROC.
- `METHOD.md`: copy-ready short write-up paragraph.
- `REPORT.md`: results, limitations and interpretation.
