# critic-swap

A patch for the upstream Semantic Halting Problem (SHP) harness that makes the
critic in the Writer→Critic loop swappable, so the default LLM critic can be
compared against Jev (TypeSafe AI) on the same scenarios.

- `critic-swap.patch`: the change, against upstream commit `6c3eb99`
- `mock_comparison.csv`: output of the mock comparison run (synthetic, see below)

## What the patch does

| File (under `backend/`) | Change |
|---|---|
| `shp/critics.py` (new) | Critic registry selected with `--critic {default,jev}`. `default` is the original LLM critic (`agents.make_critic_node`). `jev` asks `JevClient.decide()` for approve/revise plus a calibrated confidence. On approve it emits `APPROVED` with no LLM call. On revise, the regular agent LLM writes one critique, using a prompt that cannot approve. |
| `shp/trajectory.py` | `TrajectoryGenerator(critic=...)`. Logs per-round `writer_latency_s`, `critic_latency_s`, `critic_confidence` and `critic_llm_calls`. |
| `experiments/checkpoint.py` | Loads the new per-round fields with defaults, so older cached trajectories still load. |
| `experiments/run_experiment.py` | `--critic` flag. The critic name goes into the `run_id`, so a Jev run never reuses the default critic's cached trajectories. In `--mock` mode the Jev path runs the real critic node with a stub verdict. |
| `experiments/compare_critics.py` (new) | Runs the same scenarios once per critic and prints a table plus a CSV: success rate, avg rounds, judge calls, critic LLM calls, latency and final IS, for each critic × halt policy (`critic_only`, `shp`). |

The Jev stub (`JevClient`) works only in mock mode, where it returns a
deterministic verdict. The real call raises `NotImplementedError` until the Jev
API is wired in.

Because the Jev critic emits the literal `APPROVED` on approval, the halting
logic (`halting.shp_should_halt`) and every replay policy work unchanged.

## Apply and run the mock comparison

Environment setup is described in `../SETUP.md` (Python 3.12, `langchain-community<0.4`).

```bash
git clone https://github.com/SahilShrivastava-Dev/semantic-halting-problem
cd semantic-halting-problem
git checkout 6c3eb99
git apply --directory=backend /path/to/Halt-IQ/critic-swap/critic-swap.patch

cd backend
python experiments/compare_critics.py --mock --split dev --success is
```

The table prints to stdout and the CSV is written to
`results/compare_cmp_dev_groq_mr6_mock.csv`. This reproduces `mock_comparison.csv` exactly.

A single critic can also be run through the normal runner:

```bash
python experiments/run_experiment.py --mock --split dev --critic jev
```

Verified on a fresh checkout of `6c3eb99`:
- The patch applies cleanly and the mock comparison reproduces `mock_comparison.csv`.
- With `--critic default`, the mock `summary.csv` is byte-identical to the unpatched harness, so there's no regression.
- `python -m shp.theory_checks` passes.
- `tests/` was not run, because pytest isn't installed in the venv.

## The numbers are synthetic

`mock_comparison.csv` comes from `--mock` mode. The trajectories are fabricated
and the Jev verdicts come from a hash-based stub formula. Latency reads 0.000
because no model is called. The numbers show only that the pipeline runs end to
end. They say nothing about how Jev compares with the default critic.

| critic | policy | n | success_rate | avg_rounds | judge_calls | critic_llm | latency_s | final_is |
|---|---|---|---|---|---|---|---|---|
| default | critic_only | 20 | 0.500 | 3.55 | 0.00 | 3.55 | 0.000 | 0.6497 |
| default | shp | 20 | 0.500 | 3.55 | 3.55 | 3.55 | 0.000 | 0.6497 |
| jev | critic_only | 20 | 0.200 | 2.60 | 0.00 | 1.60 | 0.000 | 0.6311 |
| jev | shp | 20 | 0.300 | 2.95 | 2.95 | 1.80 | 0.000 | 0.6378 |

In this run, success means final IS ≥ 0.65 (`--success is`).

## Open questions

1. **How to define success.** The harness doesn't record success at all, only
   RAGAS metrics and the Information Score. `compare_critics.py` offers two stand-ins:
   - `--success contains`: the normalized ground-truth answer appears in the
     final answer. This is weak for yes/no comparison questions.
   - `--success is`: final IS ≥ `--is-threshold`. The default of 0.65 is arbitrary.

   The team needs to pick one, or define another.
2. **Jev API access.** `JevClient.decide()` needs the TypeSafe AI endpoint,
   credentials and response format before any real run.
3. **Jev's confidence doesn't affect halting yet.** It's logged per round
   (`critic_confidence`) but not used. An obvious next step is to approve only
   when confidence ≥ τ, or to feed the confidence into a learned stop policy.
4. **The live LangGraph path isn't wired.** `shp/agent_workflow.build_graph`,
   which the API/frontend uses, still calls `make_critic_node` directly. Only
   the experiment harness honors `--critic`. Wiring it is about a two-line change.

Also note that the comparison across critics isn't strictly paired. Each
critic's feedback changes the writer's later drafts, so each critic has its own
trajectories. Policies within one critic are still strictly paired.
