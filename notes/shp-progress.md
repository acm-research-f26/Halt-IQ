# SHP Baseline — Working Setup

- Cloned semantic-halting-problem, confirmed NO GPU needed (API-based: NVIDIA/Groq/OpenAI + local embeddings)
- Environment fully installed (venv, requirements.txt)
- theory_checks.py: ALL proven guarantees pass — Termination, IS bounds, simplex weights, distance bounds, halt-priority consistency
- Fixed dependency issue: pinned datasets<4.0 (v5 dropped legacy loading-script support)
- Fixed dataset-name issue: HF hotpot_qa moved to hotpotqa/hotpot_qa — patched scripts/build_dataset.py
- Dataset build SUCCEEDED: 80 real HotpotQA scenarios (20 dev, 60 test-frozen), 64 bridge / 16 comparison type
- Next: add API key to .env, run experiments/run_experiment.py

## Research direction — GRPO-aligned plan
- SHP is our real harness (HotpotQA, no GPU) — HiPER's idea (hierarchical credit assignment) is a conceptual reference only
- Core plan: train a GRPO policy (critic-free, matches proposal) with potential-based reward shaping to replace SHP's hand-tuned thresholds (fixed epsilon/k/MAX_ROUNDS)
- Second piece: gated judge-consultation policy (FrugalGPT/LLaPipe-style) — decide if consulting RAGAS judge is worth it each round; SHP's own paper found naive always-judge backfires
- Oracle finding: best-round-in-hindsight beats every real policy — possible pivot: track running best-answer estimate instead of assuming last round is best
- C3 (counterfactual credit assignment) — expensive comparison point for whether our cheaper gated signal is sensible
