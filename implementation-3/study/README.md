# Expanded HotpotQA study

Prepared 16 October 2026. The original 80-question pilot and its checkpoint are preserved.

## Design

600 unique questions from the official HotpotQA distractor development set: 300 train, 100 validation, 200 local held-out test. Exactly ten supplied passages per question. All 80 prior pilot IDs are excluded. Other teammates' question lists were unavailable; cross-team overlap is not established. These are local partitions, not HotpotQA's hidden official test set. Dataset/protocol checksums and exact IDs are saved.

Frozen Qwen2.5-1.5B-Instruct-4bit, temperature 0, maximum 256 output tokens per call, six writer/critic rounds. CPU linear stopping head, five features and six parameters. Seeds 0-4; 20 epochs; group size 8; token penalty 0.1; stop threshold 0.5. Hyperparameters fixed before expanded test scoring. No checkpoint or seed selection by test performance.

## Running in VS Code

Do not start a second generator if one is already running. Check generation.log first.

Terminal 1, from the summer-tushaar project:

```bash
source .venv-mlx/bin/activate
python -m mlx_lm.server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8080
```

Terminal 2:

```bash
cd /Users/srisathiya/Documents/Codex/HaltIQ/summer-tushaar
source .venv/bin/activate
cd implementation-3
python study.py generate
```

After all 600 trajectories complete:

```bash
python study.py run
```

Generation resumes from complete questions and saved rounds. A pending writer response survives a failed critic call. Failed attempts are logged; they are not included in counterfactual policy-prefix costs. No automatic network retries are performed. Keep the laptop connected to power and awake. Stop with Ctrl+C if needed, then rerun the same command.

## Outputs

- questions.json: exact split assignments, passages and reference answers. Gold answers are never sent to the writer or critic.
- protocol.json: frozen design and provenance.
- software_manifest.json: source checksums and environment versions at launch.
- cache/calls.jsonl: real requests, responses, provider usage, latency and failed attempts. Contains public benchmark text, no API secrets.
- cache/hotpot_*.json: complete trajectories; .partial files are incomplete and excluded from evaluation.
- results/seed0 ... seed4: checkpoints, training history, detailed evaluation JSON, per-question CSV and summary Markdown.
- results/all_seeds.json: all seeds, with no best-test-seed selection.

## Metrics and interpretation

Primary: normalized answer token F1. Secondary: normalized answer exact match, rounds, LLM calls, writer/critic tokens, input/output tokens, and sum of request seconds. Report paired bootstrap 95% intervals against Fixed-1 and Fixed-6, resampling questions (not individual rounds). Baselines: Fixed-1, Fixed-3, Fixed-6, first critic approval, two consecutive lexical distances below 0.06, learned head, and a hindsight F1 oracle. The oracle cannot be deployed because it uses reference answers.

All policies include writer and critic cost through the chosen round, including the final critic. Fixed-1 is therefore a one-round writer/critic baseline, not a writer-only baseline. Total tokens count all provider-reported input and output tokens, including cached prompt tokens. This is not a money estimate. Request timing depends on MLX prompt caching and hardware state; replay prefix timing is not measured end-to-end runtime for independently deployed stopping policies. Failed/repeated data-collection attempts are retained in calls.jsonl and should be reported separately from replay cost.

Supporting-fact and joint HotpotQA scores are not claimed: the model outputs only an answer, not sentence-level support predictions. No live Jev integration or result is claimed for this experiment.

## Existing pilot audit

../results/pilot-audit-20261016 contains the original checkpoint audit with EM, F1, rounds, tokens, inferred calls, paired confidence intervals and a hindsight oracle. ../results/pilot-seed-audit-20261016 contains five repeated training seeds on that same small pilot. This is an exploratory robustness check, not five independent test datasets. Original context counts: 79 questions have four passages and one has two; this differs from the expanded ten-passage study.

## Data and scorer references

https://hotpotqa.github.io/
https://huggingface.co/datasets/hotpotqa/hotpot_qa
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

Official scorer parity checked for all 480 pilot answers. Original scores unchanged. A zero-overlap edge case in hotpot.py now matches official F1 semantics (empty normalized strings score F1 0).
