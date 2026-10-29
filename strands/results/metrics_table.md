220 HotpotQA dev drafts (qwen3:8b), 126 correct by strict exact match (57%). AUROC 95% CI: 1000-sample bootstrap over drafts, seed 0. Approve = score >= 0.8. Precision = share of approved drafts that are correct.

| Critic | Context | AUROC (95% CI) | Approve rate | Precision | Wrong approvals |
|---|---|---:|---:|---:|---:|
| Strands, truncated | first 3,000 chars of the 10 paragraphs | 0.434 (0.360–0.508) | 16/220 (7%) | 56% | 7/94 |
| Strands + MiniLM top-2 | MiniLM reranker top-2 paragraphs | 0.667 (0.593–0.736) | 103/220 (47%) | 70% | 31/94 |
| Strands + MiniLM top-3 | MiniLM reranker top-3 paragraphs (pre-chosen headline) | 0.623 (0.550–0.700) | 74/220 (34%) | 72% | 21/94 |
| Strands + MiniLM top-4 | MiniLM reranker top-4 paragraphs | 0.604 (0.523–0.679) | 77/220 (35%) | 65% | 27/94 |
| Strands + bge top-2 | bge-reranker-base top-2 paragraphs | 0.725 (0.655–0.792) | 128/220 (58%) | 72% | 36/94 |
| Strands + bge top-3 | bge-reranker-base top-3 paragraphs | 0.659 (0.590–0.729) | 90/220 (41%) | 70% | 27/94 |
| Strands + MiniLM pairmax | max over the pairs from MiniLM top-4 | 0.660 (0.582–0.730) | 134/220 (61%) | 67% | 44/94 |
| Strands + MiniLM pair mean | mean over the pairs from MiniLM top-4 | 0.548 (0.468–0.623) | 13/220 (6%) | 69% | 4/94 |
| Strands + MiniLM pair 2nd | 2nd-highest pair from MiniLM top-4 | 0.599 (0.521–0.671) | 77/220 (35%) | 68% | 25/94 |
| Strands, gold paragraphs | gold paragraphs only (oracle, not deployable) | 0.734 (0.669–0.799) | 135/220 (61%) | 72% | 38/94 |
| Jev | Jev's own harness context (not reranked) | 0.726 (0.652–0.797) | 146/220 (66%) | 71% | 42/94 |
