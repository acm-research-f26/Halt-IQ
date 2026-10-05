"""Convert one of Yash's harness runs (read-only) into the common schema.

    python3 adapters/yash_run.py [RUN_DIR]      # default: the Sept 28 run

Policies written (one row per question each):
- one-draft      stop after the shared first draft (no critic)
- fixed-3        always revise to the round cap (3), no critic
- llm-critic     stop when qwen3:8b approves (Yash's LLMCritic)
- local-typed    stop when qwen3:8b's 3 self-scores are all >= 0.8
- kev-critic     stop when Kev-0.8B's 3 noul scores are all >= 0.8 (the run's `openjev` arm)
- oracle-fixed   on the fixed-3 trajectory, stop at the earliest round with the best F1
- oracle-any-arm  over every draft any arm produced for the question, take the best F1,
                 reached as cheaply as possible (fewest tokens up to that round in that arm)
  Both oracles peek at the gold answer, so they are upper bounds, not usable policies.
  Their cost is what that arm actually spent up to the chosen round (incl. its critic calls).

Costs are rebuilt from the run's per-call records: tokens = input + output of every call
(shared first draft + revisions + critic calls), seconds = sum of call latencies.
Kev's token counts come from Kev's own tokenizer, so kev-critic tokens mix two tokenizers.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import write_rows  # noqa: E402

DEFAULT_RUN = (Path(__file__).resolve().parents[3] / "halt-iq-yash" / "results" / "runs"
               / "run-20260928T234403183653Z")
OUT = Path(__file__).resolve().parents[1] / "results"
POLICY = {"fixed": "fixed-3", "llm": "llm-critic", "local-typed": "local-typed", "openjev": "kev-critic"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def convert(run_dir):
    run_dir = Path(run_dir)
    events = read_jsonl(run_dir / "traces.jsonl")
    source = run_dir.name

    # Per-round cost for each (question, arm): round 1 = the shared first draft; critic calls join their round.
    shared = {e["task_id"]: e for e in events if e["event"] == "shared_writer"}
    round_cost = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))  # (task, arm) -> round -> [tokens, seconds]
    for e in events:
        if e["event"] == "writer":
            call, key, rnd = e["call"], (e["task_id"], e["arm"]), e["round"]
        elif e["event"] == "critic":
            call, key, rnd = e["review"]["call"], (e["task_id"], e["arm"]), e["round"]
        else:
            continue
        cost = round_cost[key][rnd]
        cost[0] += call["input_tokens"] + call["output_tokens"]
        cost[1] += call["latency_ms"] / 1000

    def cost_through(task_id, arm, last_round):
        first = shared[task_id]["call"]
        tokens, seconds = first["input_tokens"] + first["output_tokens"], first["latency_ms"] / 1000
        for rnd in range(1, last_round + 1):
            tokens += round_cost[(task_id, arm)][rnd][0]
            seconds += round_cost[(task_id, arm)][rnd][1]
        return tokens, seconds

    rows = []
    for ep in (e for e in events if e["event"] == "episode"):
        task_id, arm, trace = ep["task_id"], ep["arm"], ep["round_trace"]
        tokens, seconds = cost_through(task_id, arm, ep["rounds"])
        assert tokens == ep["input_tokens"] + ep["output_tokens"], "rebuilt tokens differ from the run's totals"
        base = {"question_id": task_id, "split": ep["split"], "source": source, "info_score": None, "cost_usd": ep["cost_usd"]}
        rows.append({**base, "policy": POLICY[arm], "final_answer": ep["final_answer"], "rounds": ep["rounds"],
                     "total_tokens": tokens, "seconds": round(seconds, 3),
                     "exact_match": ep["exact_match"], "token_f1": ep["token_f1"]})
        if arm == "fixed":
            first = trace[0]
            f_tokens, f_seconds = cost_through(task_id, "fixed", 0)
            rows.append({**base, "policy": "one-draft", "final_answer": first["draft"], "rounds": 1,
                         "total_tokens": f_tokens, "seconds": round(f_seconds, 3),
                         "exact_match": first["exact_match"], "token_f1": first["token_f1"]})
            best = max(r["token_f1"] for r in trace)
            stop = next(r for r in trace if r["token_f1"] == best)  # earliest round with the best F1
            o_tokens, o_seconds = cost_through(task_id, "fixed", stop["round"])
            rows.append({**base, "policy": "oracle-fixed", "final_answer": stop["draft"], "rounds": stop["round"],
                         "total_tokens": o_tokens, "seconds": round(o_seconds, 3),
                         "exact_match": stop["exact_match"], "token_f1": stop["token_f1"]})

    by_task = defaultdict(list)
    for ep in (e for e in events if e["event"] == "episode"):
        for rnd in ep["round_trace"]:
            by_task[ep["task_id"]].append((ep, rnd, *cost_through(ep["task_id"], ep["arm"], rnd["round"])))
    for task_id, options in by_task.items():
        best = max(rnd["token_f1"] for _, rnd, _, _ in options)
        ep, rnd, tokens, seconds = min((o for o in options if o[1]["token_f1"] == best), key=lambda o: (o[2], o[3]))
        rows.append({"question_id": task_id, "split": ep["split"], "source": source, "info_score": None,
                     "cost_usd": ep["cost_usd"], "policy": "oracle-any-arm", "final_answer": rnd["draft"],
                     "rounds": rnd["round"], "total_tokens": tokens, "seconds": round(seconds, 3),
                     "exact_match": rnd["exact_match"], "token_f1": rnd["token_f1"]})
    return rows, source


if __name__ == "__main__":
    rows, source = convert(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"yash-{source}.jsonl"
    print(f"wrote {write_rows(path, rows)} rows to {path.relative_to(OUT.parent)}")
