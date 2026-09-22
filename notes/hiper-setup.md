# HiPER Baseline — Findings

- Built on veRL + verl-agent, requires CUDA GPU (torch cu124 + flash-attn)
- NOT runnable on local Mac — no CUDA support
- Two environments: ALFWorld and WebShop (need separate conda envs)
- Entry points: example_scripts/HiPER_trainer/run_alfworld.sh / run_webshop.sh
- Blocker: need GPU access (lab machine / cloud instance) to actually run training

## Papers read tonight
- HiPER: hierarchical planner/executor split, Hierarchical Advantage Estimation for credit assignment
- C3: counterfactual credit assignment (exact but expensive) — our accuracy benchmark
- Orchestration-traces survey: confirms zero existing RL methods for "when to stop" — the gap HaltIQ fills
