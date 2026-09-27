# Notes on Tushaar's summer code (`summer-tushaar/`)

This is a summary of a local copy that isn't a git repo. None of the code itself is committed here.

## What it is

It's a separate project from the SHP harness. A multi-agent orchestrator learns
when to stop delegating to specialist agents, with a stop policy trained by
critic-free GRPO. There is no writer/critic loop, no HotpotQA, no RAGAS and no
import of the SHP harness.

- **implementation-1/**: vendored HiPER-agent codebase (a veRL / verl-agent fork,
  unmodified) plus a lightweight training config. It is Ray-based and needs CUDA.
- **implementation-2/**: Tushaar's own code on **WebShop**. The orchestrator chooses
  `{delegate-search, delegate-compare, stop}`.
- **implementation-3/**: the same code ported to **ALFWorld**, which is the newest
  version. The orchestrator chooses `{delegate-navigate, delegate-interact, stop}`.
  It adds a sub-agent credit trainer, a SEAL self-editing variant and run scripts
  under `runs/`. Its README refers to `PROJECT_GUIDE.md` and `EXPERIMENT_LOG.md`,
  which aren't in the copy.

## Key pieces (implementation-3/research/orchestration/)

- **Loop:** `episode_runner.run_episode`. Each round the orchestrator picks an
  action, a specialist acts in the environment, and the loop repeats until
  `stop`, the task is done, or the 6-round cap.
- **Models:** one Qwen2.5-0.5B-Instruct base with separate LoRA adapters: `shared`
  for the specialists, `orchestrator`, and `self_editor` (`model_pool.py`).
- **Stop policy:** there is no `StopPolicyHead` class anywhere. The stop policy is
  the orchestrator's LoRA adapter generating `<action>stop</action>`
  (`orchestrator.py`).
- **GRPO (Baseline B, `baseline_b_rl_stop.py`):**
  - Runs a group of G episodes and scores each with `R = 10·success − 0.5·rounds`.
  - Normalizes rewards within the group and applies a PPO-clipped update to
    the orchestrator adapter only, with the specialists frozen.
  - The old log-probs are recomputed right before a single optimizer step, so
    the ratio is always 1 and the clipping never activates. In effect it's
    policy-gradient with a group baseline.
- **Baseline A:** a fixed delegation sequence run through the same loop.
- **What's recorded:** each episode's `EpisodeLog` has `success`, `num_delegation_rounds`
  and `stopped_by`. The run `result.json` holds per-group success rate and
  average rounds. Timings are per training round, not per episode.

## "Critic" means something different here

In Tushaar's code, "critic" means the **PPO value head**: HiPER's three value
heads V_low, V_high and V_term in `hiper_hae/core_hae.py`, which his GRPO
variant deliberately removes. It does **not** mean an LLM reviewer.

So the Jev critic swap (`critic-swap/`) applies to the **SHP harness**, which is
the only codebase with a Writer→Critic loop. It doesn't apply to Tushaar's loop.
If Jev is used there later, it fits as a **stop gate** ("task done with
confidence ≥ τ → stop"). That would be a hook in `run_episode` and a third
baseline beside the heuristic and GRPO.
