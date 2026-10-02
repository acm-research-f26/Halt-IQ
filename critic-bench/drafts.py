"""Load the fixed test set of drafts from Yash's Sept 28 run (read-only).

Every critic judges exactly these drafts, so nothing is regenerated.
A draft is "correct" when its saved HotpotQA exact_match is True (the same
label Yash's analyze_run.py uses).
"""

import hashlib
import json
import re
import string
from pathlib import Path

RUN_DIR = (Path(__file__).resolve().parents[2] / "halt-iq-yash" / "results" / "runs"
           / "run-20260928T234403183653Z")


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
    """which = "first" (20 shared first drafts) or "all" (every unique labeled draft)."""
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
            drafts.setdefault(key, {
                "draft_id": key, "task_id": task["id"], "question": task["question"],
                "evidence": task["evidence"], "draft": rnd["draft"],
                "correct": rnd["exact_match"], "is_first": is_first,
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
