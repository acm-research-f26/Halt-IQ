# Critic benchmark report

Label: **lenient** (exact match or token F1 >= 0.8).

## Draft set `first`: 20 drafts, 11 correct (55%)

Yes/no questions (by gold answer): 2/20. Drafts answering yes/no (evidence_match scores these 0.5): 0/20.

Approving everything would be right 11/20 (55%) of the time; a useful critic beats this on accuracy when approved.

| Critic | Source | n scored | Accuracy when approved (@0.8) | Wrong approvals @0.8 | Wrongly rejected @0.8 | AUROC | Mean s/decision | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llm | Yash's critic; saved + live | 20/20 | 10/17 (59%) | 7/9 | 1/11 | — (text only) | 7.81 | $0 (local) |
| local-typed | Yash's critic; saved | 20/20 | 10/19 (53%) | 9/9 | 1/11 | 0.45 | 7.26 | $0 (local) |
| kev | Kev-0.8B; saved + live | 20/20 | 0/0 | 0/9 | 11/11 | 0.54 | 1.01 | $0 (local) |
| laya | new | 20/20 | 0/2 (0%) | 2/9 | 11/11 | 0.48 | 0.17 | $0 (local) |
| evidence_match | new, no model | 20/20 | 11/17 (65%) | 6/9 | 0/11 | 0.67 | 0.00 | $0 (local) |
| consistency | new, 3 writer samples | 20/20 | 9/17 (53%) | 8/9 | 2/11 | 0.45 | 9.19 | $0 (local) |

Threshold sweep (probability critics): wrong approvals / wrongly rejected

| Critic | @0.5 | @0.6 | @0.7 | @0.8 | @0.9 |
|---|---:|---:|---:|---:|---:|
| local-typed | 9 / 1 | 9 / 1 | 9 / 1 | 9 / 1 | 9 / 1 |
| kev | 3 / 7 | 1 / 10 | 0 / 10 | 0 / 11 | 0 / 11 |
| laya | 7 / 1 | 5 / 1 | 5 / 5 | 2 / 11 | 0 / 11 |
| evidence_match | 6 / 0 | 6 / 0 | 6 / 0 | 6 / 0 | 6 / 0 |
| consistency | 9 / 2 | 9 / 2 | 8 / 2 | 8 / 2 | 8 / 2 |

Sample size: n = 20, so any accuracy here is uncertain by roughly ±22% (95% interval, worst case p = 0.5). Accuracy-when-approved uses even fewer drafts.

## Draft set `all`: 25 drafts, 13 correct (52%)

Yes/no questions (by gold answer): 3/25. Drafts answering yes/no (evidence_match scores these 0.5): 0/25.

Approving everything would be right 13/25 (52%) of the time; a useful critic beats this on accuracy when approved.

| Critic | Source | n scored | Accuracy when approved (@0.8) | Wrong approvals @0.8 | Wrongly rejected @0.8 | AUROC | Mean s/decision | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llm | Yash's critic; saved + live | 22/25 | 12/19 (63%) | 7/9 | 1/13 | — (text only) | 7.27 | $0 (local) |
| local-typed | Yash's critic; saved | 20/25 | 10/19 (53%) | 9/9 | 1/11 | 0.45 | 7.26 | $0 (local) |
| kev | Kev-0.8B; saved + live | 21/25 | 0/0 | 0/10 | 11/11 | 0.56 | 0.98 | $0 (local) |
| laya | new | 25/25 | 1/4 (25%) | 3/12 | 12/13 | 0.55 | 0.17 | $0 (local) |
| evidence_match | new, no model | 25/25 | 13/21 (62%) | 8/12 | 0/13 | 0.67 | 0.00 | $0 (local) |
| consistency | new, 3 writer samples | 20/25 | 9/17 (53%) | 8/9 | 2/11 | 0.45 | 9.19 | $0 (local) |

Threshold sweep (probability critics): wrong approvals / wrongly rejected

| Critic | @0.5 | @0.6 | @0.7 | @0.8 | @0.9 |
|---|---:|---:|---:|---:|---:|
| local-typed | 9 / 1 | 9 / 1 | 9 / 1 | 9 / 1 | 9 / 1 |
| kev | 3 / 7 | 1 / 10 | 0 / 10 | 0 / 11 | 0 / 11 |
| laya | 8 / 1 | 6 / 1 | 6 / 5 | 3 / 12 | 0 / 13 |
| evidence_match | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 |
| consistency | 9 / 2 | 9 / 2 | 8 / 2 | 8 / 2 | 8 / 2 |

Sample size: n = 25, so any accuracy here is uncertain by roughly ±20% (95% interval, worst case p = 0.5). Accuracy-when-approved uses even fewer drafts.

## Draft set `extra`: 200 drafts, 126 correct (63%)

Yes/no questions (by gold answer): 12/200. Drafts answering yes/no (evidence_match scores these 0.5): 4/200.

Approving everything would be right 126/200 (63%) of the time; a useful critic beats this on accuracy when approved.

| Critic | Source | n scored | Accuracy when approved (@0.8) | Wrong approvals @0.8 | Wrongly rejected @0.8 | AUROC | Mean s/decision | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llm | Yash's critic; saved + live | 200/200 | 106/159 (67%) | 53/74 | 20/126 | — (text only) | 10.23 | $0 (local) |
| kev | Kev-0.8B; saved + live | 200/200 | 0/1 (0%) | 1/74 | 126/126 | 0.50 | 0.86 | $0 (local) |
| laya | new | 200/200 | 50/65 (77%) | 15/74 | 76/126 | 0.64 | 0.22 | $0 (local) |
| evidence_match | new, no model | 200/200 | 119/177 (67%) | 58/74 | 7/126 | 0.58 | 0.00 | $0 (local) |
| consistency | new, 3 writer samples | 60/200 | 36/51 (71%) | 15/21 | 3/39 | 0.61 | 8.70 | $0 (local) |

Threshold sweep (probability critics): wrong approvals / wrongly rejected

| Critic | @0.5 | @0.6 | @0.7 | @0.8 | @0.9 |
|---|---:|---:|---:|---:|---:|
| kev | 22 / 101 | 10 / 121 | 3 / 125 | 1 / 126 | 0 / 126 |
| laya | 57 / 23 | 47 / 31 | 32 / 42 | 15 / 76 | 1 / 121 |
| evidence_match | 59 / 4 | 58 / 7 | 58 / 7 | 58 / 7 | 58 / 7 |
| consistency | 17 / 2 | 17 / 2 | 15 / 3 | 15 / 3 | 15 / 3 |

Sample size: n = 200, so any accuracy here is uncertain by roughly ±7% (95% interval, worst case p = 0.5). Accuracy-when-approved uses even fewer drafts.

## Draft set `subset80`: 80 drafts, 50 correct (62%)

Yes/no questions (by gold answer): 5/80. Drafts answering yes/no (evidence_match scores these 0.5): 1/80.

Approving everything would be right 50/80 (62%) of the time; a useful critic beats this on accuracy when approved.

| Critic | Source | n scored | Accuracy when approved (@0.8) | Wrong approvals @0.8 | Wrongly rejected @0.8 | AUROC | Mean s/decision | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llm | Yash's critic; saved + live | 80/80 | 44/66 (67%) | 22/30 | 6/50 | — (text only) | 9.59 | $0 (local) |
| local-typed | Yash's critic; saved | 20/80 | 10/19 (53%) | 9/9 | 1/11 | 0.45 | 7.26 | $0 (local) |
| kev | Kev-0.8B; saved + live | 80/80 | 0/0 | 0/30 | 50/50 | 0.51 | 1.23 | $0 (local) |
| laya | new | 80/80 | 15/23 (65%) | 8/30 | 35/50 | 0.55 | 0.26 | $0 (local) |
| evidence_match | new, no model | 80/80 | 48/69 (70%) | 21/30 | 2/50 | 0.63 | 0.00 | $0 (local) |
| consistency | new, 3 writer samples | 80/80 | 45/68 (66%) | 23/30 | 5/50 | 0.57 | 8.82 | $0 (local) |

Threshold sweep (probability critics): wrong approvals / wrongly rejected

| Critic | @0.5 | @0.6 | @0.7 | @0.8 | @0.9 |
|---|---:|---:|---:|---:|---:|
| local-typed | 9 / 1 | 9 / 1 | 9 / 1 | 9 / 1 | 9 / 1 |
| kev | 10 / 36 | 3 / 47 | 0 / 49 | 0 / 50 | 0 / 50 |
| laya | 25 / 8 | 20 / 11 | 16 / 21 | 8 / 35 | 0 / 49 |
| evidence_match | 21 / 1 | 21 / 2 | 21 / 2 | 21 / 2 | 21 / 2 |
| consistency | 26 / 4 | 26 / 4 | 23 / 5 | 23 / 5 | 23 / 5 |

Sample size: n = 80, so any accuracy here is uncertain by roughly ±11% (95% interval, worst case p = 0.5). Accuracy-when-approved uses even fewer drafts.

![Reliability chart](reliability-lenient.png)
