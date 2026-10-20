# Sri — HotpotQA learned stopping experiments

This branch contains my Implementation 3 code, the linear stopping classifier, tests, cached model responses, saved checkpoints, and per-question results. Qwen2.5-1.5B-Instruct-4bit stays frozen; only five classifier weights and one bias are trained.

## Start here

Use Python 3.13 (the recorded environment used 3.13.15). From this repository:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
cd implementation-3
python -m unittest test_hotpot test_study
```

`requirements-local.txt` pins the CPU evaluation/training dependencies. `requirements-local-lock.txt` records the original local environment. No model download or API key is needed to inspect or evaluate saved trajectories. Install `requirements-data.txt` only if rebuilding a question sample from the original parquet dataset.

## Code map

- `implementation-3/hotpot.py`: features, linear stop head, policy-gradient training, reward, stopping policies, pilot CLI.
- `implementation-3/study.py`: dataset preparation, live collection, detailed evaluation, uncertainty estimates, five-seed study.
- `implementation-3/run_study150.py`: resume the 150-question collection and run training/evaluation.
- `implementation-3/audit_study150.py`: dataset integrity, official answer-scorer parity, token accounting, combined results.
- `implementation-3/critic_diagnostic.py`: diagnostic on my local 80 test first drafts.
- `implementation-3/matched_benchmark.py`: my critic on the exact 220 saved benchmark drafts. Its comparison output also recomputes attributed reference scores; those are not my results.
- `implementation-3/export_team_framework.py`: shared-schema export and overlap checks.
- `implementation-3/test_hotpot.py`, `test_study.py`: correctness and data-boundary checks.

## Reproduce evaluation without generating answers

From `implementation-3` with the environment active:

```bash
python study.py report --cache study150/cache --checkpoint study150/results/seed0/stop_head.pt --output results/reproduced-seed0
python audit_study150.py
```

To retrain all five seeds using cached responses (this updates the saved study result directories):

```bash
python -c "import study; study.STUDY = study.ROOT / 'study150'; study.run()"
```

## Shared benchmark and evaluation framework

From the repository root:

```bash
python scripts/setup_team_references.py
cd implementation-3
python export_team_framework.py
python matched_benchmark.py
```

The setup script fetches the exact pinned `vedanshi` and `Yash` commits into ignored `.team-reference/` checkouts. It does not change those branches or copy their code into this branch. Existing 220 saved scores make the benchmark script a scoring-only run; missing scores trigger local model calls. Do not delete those scores unless intentionally running the benchmark again.

## Live generation on Apple Silicon

Use a separate environment for the MLX server. From the repository root, in terminal 1:

```bash
python3.13 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install mlx-lm
python -m mlx_lm.server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8080
```

In terminal 2, from the repository root:

```bash
source .venv/bin/activate
cd implementation-3
python run_study150.py
```

This resumes collection on port 8080, skips complete trajectories, then retrains and evaluates. The checked-in 150-question cache is already complete. Do not run simultaneous generators. The original software manifest records the generation environment/model snapshot; a new MLX installation is not guaranteed bitwise-identical output.

## Data and results

- `data/`, `cache/`, `results/`: original pilot inputs, responses, and results; synthetic demo directories are pipeline checks only.
- `study/`: original 600-question plan and official HotpotQA answer evaluator. The full downloaded validation parquet is omitted; its source is recorded in the protocol.
- `study150/questions.json`: 50 train / 20 validation / 80 local test questions, excluding the original 80 pilot IDs.
- `study150/cache/`: 150 six-round trajectories, writer/critic responses, token usage, and call timings.
- `study150/results/seed0` through `seed4`: six-parameter checkpoints, training histories, per-question CSVs, and evaluation summaries.
- `study150/results/matched220/`: my frozen critic's 220 benchmark responses, input hashes, usage, metrics, and source pins.
- `study150/results/reference_alignment/`: shared-schema JSONL and overlap audit.

The official development split supplies these local partitions; they are not the hidden official HotpotQA test set. Existing manifests retain original paths as historical provenance, but runnable entry points use repository-relative paths. Dataset source: https://hotpotqa.github.io/ . Respect upstream dataset terms and attribution.

## Findings and limitations

The learned controller stops after one round on all 80 local test questions across five seeds: F1 0.3588, EM 0.2125, 83.39% fewer replay tokens than Fixed-6. It matches Fixed-1 and loses F1 versus Fixed-6 (0.3921). Writer-only gives the same first-answer quality at about half the learned controller's token cost. The distance rule matches Fixed-6 F1/EM on this sample with 47.64% fewer replay tokens.

My local critic's 220-draft AUROC is 0.503; it approves 216 drafts, including 92 of 94 exact-match failures. This is weak discrimination, not evidence of a successful judge. Critic confidence and stopping probability are different quantities.

Savings are cached-trajectory replay estimates, not measured live stopping savings. Historical fixed-round policies include critic costs; writer-only excludes them. Two local training questions overlap the team's holdout, so that holdout is not untouched for this controller. A fresh shared holdout and a three-round live run are needed for full base-protocol alignment. Shared first-draft benchmark scoring is separate from six-round controller evaluation.

Current detailed write-up (my results): https://docs.google.com/document/d/1WV3WlsH-qnNuE6Nlev4QO81nT5kK2hkNqu2ZsaTcJGE/edit
