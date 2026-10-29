"""Plain-language, automatically scoped research writeup from saved receipts."""
from pathlib import Path
import json


def write_report(output, replay=None, live=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = ["HALTIQ: VEDANSHI STRANDS + SHREE LINEAR CLASSIFIER",
             "", "Purpose", "Test whether the teammates' checker and stopping classifier can work together in a live writer/reviewer loop.",
             "", "Components", "AWS Strands Agents SDK runs separate writer and reviewer agents using a local, pinned Qwen2.5-1.5B model.",
             "Vedanshi's Hobson-v19 Strands Decider scores each answer using BGE-ranked top-two public evidence paragraphs.",
             "Shree's pretrained six-parameter linear head uses five lexical/history/review features. Its seed and weights are frozen before testing.",
             "The combined transfer stops only when the head requests stopping AND Strands scores at least 0.8. This is a new AND rule around the teammates' existing models; it is not a newly trained paired head.",
             "", "Prior evidence used", "Only Vedanshi's and Shree's public code, checkpoints, questions and saved trajectories are inputs. Yash's earlier experiments, models and results are excluded.",
             "", "Inherited classifier limitation", "The frozen classifier requests stopping on the first round at its 0.5 cutoff. The combined AND rule may therefore equal Strands-only. Component ablations measure this directly; equal results do not establish synergy."]
    if replay:
        lines += ["", "Saved-trajectory replay (existing teammate evidence)",
                  f"Questions: {replay.get('unique_questions', 'unknown')}. Newly generated model calls: 0.",
                  "These are policy decisions replayed over the same saved drafts, not a live multi-agent run or a measured latency experiment."]
        for row in replay.get("summary", []):
            if row.get("seed") == 0:
                lines.append(json.dumps(row, ensure_ascii=False))
        lines += ["Saved LLM token-prefix estimates exclude Strands and reranker compute. They are not total compute savings."]
    lines += ["", "New live simulation"]
    if live:
        lines.append(json.dumps(live, indent=2, ensure_ascii=False))
    else:
        lines.append("Not run in this receipt. No new HotpotQA accuracy or Tau task-success result is claimed.")
    lines += ["", "Evaluation boundaries", "HotpotQA uses public distractor evidence while reference answers and supporting facts stay in the evaluator. Answer, supporting-fact and joint scores remain separate. Rejected or failed attempts remain in the planned denominator.",
              "Tau uses the pinned official English retail tasks, environment tools and evaluator. The local user simulator and local judges differ from released leaderboard conditions; report these as official-domain local simulations, not leaderboard-comparable scores.",
              "The original Shree writer used Apple MLX 4-bit inference. This Colab version uses Hugging Face inference; T4 serves Hobson in FP16 rather than native BF16. Record these numerical differences without claiming identical historical probabilities.",
              "", "Reproduction and evidence", "Use the two Colab notebooks in notebooks/. plan.json or manifest.json freezes the selected tasks, source and models before model calls. Journal/trace files, per-episode outcomes, runtime metadata and summary.json preserve failures and actual usage.",
              "The original teammate integration revision is 6c72991652400dcfec8f464a331efb95d5d1cf8b. The classifier revision is 5503cabeb2ec0094413b7d491ccb08ec9c991c61."]
    path = output / "Research Writeup.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
