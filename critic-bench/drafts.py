"""Load the fixed test sets of drafts. Every critic judges exactly these drafts.

- first / all: drafts from Yash's Sept 28 run (read-only).
- extra: first drafts for ~200 more HotpotQA dev questions (made by make_extra.py).

Two labels per draft, both from Yash's HotpotQA scorer (haltiq/metrics.py):
- correct (strict): exact match after normalization, as analyze_run.py uses.
- lenient: exact match OR token F1 >= 0.8.
"""

import hashlib
import json
import re
import string
import sys
from pathlib import Path

YASH_REPO = Path(__file__).resolve().parents[2] / "halt-iq-yash"
RUN_DIR = YASH_REPO / "results" / "runs" / "run-20260928T234403183653Z"
EXTRA_DIR = Path(__file__).resolve().parent / "extra"
LENIENT_F1 = 0.8

sys.dont_write_bytecode = True  # import Yash's package without writing anything into his folder
sys.path.insert(0, str(YASH_REPO))
from haltiq.metrics import hotpot_evaluate  # noqa: E402


def labels(draft, gold):
    score = hotpot_evaluate(draft, gold)
    return {"correct": score["exact_match"], "f1": score["token_f1"],
            "lenient": score["exact_match"] or score["token_f1"] >= LENIENT_F1}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text):
    """HotpotQA answer normalization (same rule as Yash's haltiq/metrics.py)."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def draft_id(task_id, draft):
    """Same question + same normalized answer = same draft, whichever arm wrote it."""
    digest = hashlib.sha256(normalize(draft).encode()).hexdigest()[:8]
    return f"{task_id.split(':')[-1]}-{digest}"


def load_drafts(which="first", run_dir=RUN_DIR):
    """which = "first" (20 shared first drafts), "all" (every unique labeled draft), or "extra"."""
    if which == "extra":
        tasks = {t["id"]: t for t in read_jsonl(EXTRA_DIR / "tasks.jsonl")}
        return [{"draft_id": draft_id(row["task_id"], row["draft"]), "task_id": row["task_id"],
                 "question": tasks[row["task_id"]]["question"], "evidence": tasks[row["task_id"]]["evidence"],
                 "draft": row["draft"], "is_first": True, **labels(row["draft"], tasks[row["task_id"]]["answers"])}
                for row in read_jsonl(EXTRA_DIR / "drafts.jsonl")]

    tasks = {t["id"]: t for t in read_jsonl(run_dir / "dataset_snapshot.jsonl") if t["split"] == "dev"}
    events = read_jsonl(run_dir / "traces.jsonl")
    first = {(e["task_id"], e["draft"]) for e in events if e["event"] == "shared_writer"}

    drafts = {}
    for episode in (e for e in events if e["event"] == "episode"):
        task = tasks[episode["task_id"]]  # KeyError here would mean a non-dev question slipped in
        for rnd in episode["round_trace"]:
            is_first = (task["id"], rnd["draft"]) in first
            if which == "first" and not is_first:
                continue
            key = draft_id(task["id"], rnd["draft"])
            label = labels(rnd["draft"], task["answers"])
            assert label["correct"] == rnd["exact_match"], "rescored label differs from the saved one"
            drafts.setdefault(key, {
                "draft_id": key, "task_id": task["id"], "question": task["question"],
                "evidence": task["evidence"], "draft": rnd["draft"], "is_first": is_first, **label,
            })
    return list(drafts.values())


def saved_reviews(arm, run_dir=RUN_DIR):
    """Critic decisions already saved in the run for one arm: draft_id -> review."""
    events = read_jsonl(run_dir / "traces.jsonl")
    latency = {(e["task_id"], e["round"]): e["review"]["call"]["latency_ms"]
               for e in events if e["event"] == "critic" and e["arm"] == arm}
    reviews = {}
    for episode in (e for e in events if e["event"] == "episode" and e["arm"] == arm):
        for rnd in episode["round_trace"]:
            reviews.setdefault(draft_id(episode["task_id"], rnd["draft"]), {
                "approved": rnd["approved"], "scores": rnd["scores"],
                "seconds": latency[(episode["task_id"], rnd["round"])] / 1000,
            })
    return reviews
