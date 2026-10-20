# Local 80 only — shared framework output

Replay timings, not measured live savings. Six-round local loop differs from team three-round loop.

Baseline: **fixed_6** (always revise to the round cap). "Saved" = 1 − cost / baseline cost (negative = costs more). ΔF1 = F1 − baseline F1.

| Policy | Source | n | EM | F1 | Avg rounds | Avg tokens | Avg seconds | Tokens saved | Seconds saved | ΔF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one-draft-writer-only | sri-local80-seed0-replay | 80 | 21% | 0.359 | 1.00 | 1,514 | 2.83 | 92% | 60% | -0.033 |
| fixed_1 | sri-local80-seed0-replay | 80 | 21% | 0.359 | 1.00 | 3,028 | 5.55 | 83% | 22% | -0.033 |
| learned | sri-local80-seed0-replay | 80 | 21% | 0.359 | 1.00 | 3,028 | 5.55 | 83% | 22% | -0.033 |
| oracle_f1 | sri-local80-seed0-replay | 80 | 24% | 0.392 | 1.06 | 3,203 | 5.58 | 82% | 22% | +0.000 |
| critic | sri-local80-seed0-replay | 80 | 21% | 0.359 | 1.12 | 3,320 | 5.61 | 82% | 21% | -0.033 |
| fixed_3 | sri-local80-seed0-replay | 80 | 22% | 0.382 | 3.00 | 9,108 | 6.24 | 50% | 12% | -0.010 |
| distance | sri-local80-seed0-replay | 80 | 24% | 0.392 | 3.14 | 9,544 | 6.29 | 48% | 12% | +0.000 |
| fixed_6 | sri-local80-seed0-replay | 80 | 24% | 0.392 | 6.00 | 18,227 | 7.12 | 0% | 0% | +0.000 |