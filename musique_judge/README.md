# MuSiQue help-or-hurt judge (Vedanshi, Nov 2026)

A writer agent answers each MuSiQue question and revises its answer for 6 rounds. Whenever the answer changes, a judge decides whether the revision helped or hurt. Stopping rules are then replayed over the saved rounds and scored on answer quality (F1), accuracy (EM) and tokens saved, against fixed 1, 2 and 6 rounds.

## Files

| Path | What it is |
| --- | --- |
| `run_rounds.py` | Writer loop: Qwen2.5-1.5B-Instruct, greedy decoding, 6 rounds per question |
| `coct_judge.py` | Help-or-hurt judge: counterfactual (CoCT, after Zhang, Sun, Guo, Gogate, arXiv 2506.22385) and direct versions |
| `results.py` | Replays stopping rules and prints the results table (no GPU needed) |
| `common.py` | Scoring (official HotpotQA F1/EM), model loading, dataset loading |
| `data/musique_rounds.csv` | 200 questions x 6 rounds: answer, F1, EM, tokens, writer's raw output |
| `data/musique_paragraphs.jsonl` | Question, gold answers and evidence paragraphs for the 200 questions |
| `data/musique_judge_3b.csv` | Judge verdicts on the 130 answer changes: Qwen2.5-3B, answers only |
| `data/musique_judge_7b_reasoning.csv` | Judge verdicts: Qwen2.5-7B (4-bit), with the writer's reasoning |
| `results/results_musique_3b.csv`, `results/results_musique_7b.csv` | Results tables |

## Reproduce the results tables (no GPU)

```
python3 results.py --rounds data/musique_rounds.csv --judge data/musique_judge_3b.csv --name musique_3b
python3 results.py --rounds data/musique_rounds.csv --judge data/musique_judge_7b_reasoning.csv --name musique_7b
```

## Rerun from scratch (Colab T4)

```
pip install datasets transformers accelerate bitsandbytes
python run_rounds.py --n 200 --rounds 6
python coct_judge.py --rounds data/musique_rounds.csv --paras data/musique_paragraphs.jsonl --out data/musique_judge_3b.csv
python coct_judge.py --rounds data/musique_rounds.csv --paras data/musique_paragraphs.jsonl --out data/musique_judge_7b_reasoning.csv --model Qwen/Qwen2.5-7B-Instruct --four_bit --with_reasoning --decisive_first
```

Both scripts resume after a disconnect.

## Key results (200 questions, 130 answer changes: 31 helped, 22 hurt, 77 ties)

| Method | F1 | EM | Tokens saved |
| --- | --- | --- | --- |
| Fixed 6 rounds | 0.234 | 14.5% | 0% |
| Answer repeats | 0.234 | 14.5% | 57% |
| Repeats + 7B direct judge (with writer's reasoning) | 0.249 | 14.5% | 53% |
| Oracle stopping | 0.274 | 17.0% | 81% |

Judge accuracy on the 53 help/hurt cases: 3B direct 55%, 3B CoCT 47%; 7B direct 79%, 7B CoCT 58%; always "better" 58%.
