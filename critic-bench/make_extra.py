"""Build the `extra` draft set: first drafts for ~200 more HotpotQA dev questions.

    python3 make_extra.py select     # pick questions; prints exactly what is excluded
    python3 make_extra.py estimate   # runtime estimate from the Sept 28 run's timings
    python3 make_extra.py write      # first drafts with Yash's Writer (resumable)

Uses Yash's own code, unmodified: download_hotpot (pinned, checksum-verified source),
the same answer-blind selection helpers and task format as prepare_hotpot, and the
same Writer + OllamaClient settings the loop used (read from the run's manifest).
"""

import json
import sys
from collections import defaultdict

from drafts import EXTRA_DIR, RUN_DIR, read_jsonl, YASH_REPO  # also puts Yash's repo on sys.path

from haltiq.benchmark_download import download_hotpot
from haltiq.hotpot import _quotas, _rank, _task, _validate, normalize_question

SIZE, SEED = 200, 42
CACHE = EXTRA_DIR.parent / "data-cache"  # gitignored download cache
CONFIG = json.loads((RUN_DIR / "manifest.json").read_text())["config"]


def select():
    source, _ = download_hotpot(CACHE)
    rows = _validate(json.loads(source.read_text(encoding="utf-8")))

    # Exclude the 100 questions already in use: 20 dev (Sept 28 run) + 80 held-out test.
    used = read_jsonl(RUN_DIR / "dataset_snapshot.jsonl")
    assert used == read_jsonl(YASH_REPO / "data" / "hotpotqa" / "tasks.jsonl"), "run snapshot != Yash's task file"
    used_ids = {t["metadata"]["source_id"] for t in used}
    used_questions = {normalize_question(t["question"]) for t in used}

    # Same duplicate rule as prepare_hotpot: keep the smallest ID per normalized question.
    unique = {}
    for row in sorted(rows, key=lambda r: r["_id"]):
        unique.setdefault(normalize_question(row["question"]), row)
    eligible = [r for q, r in unique.items() if r["_id"] not in used_ids and q not in used_questions]

    split_counts = defaultdict(int)
    for t in used:
        split_counts[t["split"]] += 1
    print(f"official dev rows: {len(rows)}; unique questions: {len(unique)}")
    print(f"excluded: {split_counts['dev']} dev (already used) + {split_counts['test']} held-out test "
          f"= {len(used_ids)} source IDs; eligible: {len(eligible)}")

    # Same answer-blind selection as prepare_hotpot: type-proportional quotas, seeded hash ranking.
    groups = defaultdict(list)
    for row in eligible:
        groups[row["type"]].append(row)
    quotas = _quotas(SIZE, groups)
    chosen = [row for kind in sorted(groups)
              for row in sorted(groups[kind], key=lambda r: _rank(SEED, "extra", r))[:quotas[kind]]]
    chosen.sort(key=lambda r: _rank(SEED, "extra", r))
    tasks = [_task(row, "extra") for row in chosen]

    overlap = {t["metadata"]["source_id"] for t in tasks} & used_ids
    assert not overlap and not {normalize_question(t["question"]) for t in tasks} & used_questions
    print(f"selected {len(tasks)} ({dict(quotas)}); overlap with the 100 used/held-out questions: 0")
    EXTRA_DIR.mkdir(exist_ok=True)
    (EXTRA_DIR / "tasks.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks),
                                           encoding="utf-8")


def estimate():
    events = read_jsonl(RUN_DIR / "traces.jsonl")
    writer = [e["call"]["latency_ms"] / 1000 for e in events if e["event"] == "shared_writer"]
    llm = [e["review"]["call"]["latency_ms"] / 1000 for e in events if e["event"] == "critic" and e["arm"] == "llm"]
    kev = [e["review"]["call"]["latency_ms"] / 1000 for e in events if e["event"] == "critic" and e["arm"] == "openjev"]
    n = len(read_jsonl(EXTRA_DIR / "tasks.jsonl"))
    for name, times in (("writer (qwen3:8b)", writer), ("llm critic (qwen3:8b)", llm), ("kev", kev)):
        mean = sum(times) / len(times)
        print(f"{name}: {mean:.1f} s/call in the Sept 28 run -> ~{n * mean / 60:.0f} min for {n} questions")


def write():
    from haltiq.critics import Writer
    from haltiq.providers import OllamaClient

    writer = Writer(OllamaClient(base_url=CONFIG["ollama_url"], model=CONFIG["writer_model"], seed=CONFIG["seed"],
                                 timeout=CONFIG["timeout"], num_ctx=CONFIG["context_tokens"]))
    out_path = EXTRA_DIR / "drafts.jsonl"
    done = {row["task_id"] for row in read_jsonl(out_path)} if out_path.exists() else set()
    tasks = [t for t in read_jsonl(EXTRA_DIR / "tasks.jsonl") if t["id"] not in done]
    print(f"{len(done)} drafts cached, {len(tasks)} to write")
    with out_path.open("a", encoding="utf-8") as out:
        for i, task in enumerate(tasks, 1):
            try:
                draft, call = writer.write({"question": task["question"], "evidence": task["evidence"]})
            except Exception as error:  # record and move on, like the loop's error events
                print(f"  [{i}/{len(tasks)}] {task['id']}: {type(error).__name__}: {error}")
                continue
            out.write(json.dumps({"task_id": task["id"], "draft": draft, "seconds": round(call.latency_ms / 1000, 3),
                                  "model": call.model}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{i}/{len(tasks)}] {task['id']}: {draft!r}")


if __name__ == "__main__":
    {"select": select, "estimate": estimate, "write": write}[sys.argv[1]]()
