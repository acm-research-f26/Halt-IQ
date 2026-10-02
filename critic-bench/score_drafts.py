"""Score a draft set with one critic and cache every score to JSONL.

Usage: python3 score_drafts.py CRITIC {first,all}

Scores go to scores/CRITIC.jsonl, one line per draft, written as soon as each
draft is scored. Drafts already in the cache are skipped, so reruns are free
and a crash loses at most the draft in progress.
"""

import argparse
import json
import time
from pathlib import Path

from critics import CRITICS
from drafts import load_drafts

SCORES_DIR = Path(__file__).resolve().parent / "scores"


def load_cache(critic):
    path = SCORES_DIR / f"{critic}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["draft_id"]: row for row in rows}


def score(critic, which):
    cache = load_cache(critic)
    drafts = load_drafts(which)
    todo = [d for d in drafts if d["draft_id"] not in cache]
    print(f"{critic} on {which}: {len(drafts)} drafts, {len(drafts) - len(todo)} cached, {len(todo)} to score")

    SCORES_DIR.mkdir(exist_ok=True)
    with (SCORES_DIR / f"{critic}.jsonl").open("a", encoding="utf-8") as out:
        for i, draft in enumerate(todo, 1):
            start = time.perf_counter()
            result = CRITICS[critic](draft)
            seconds = time.perf_counter() - start
            if isinstance(result, tuple):
                result, seconds = result
            if result is None:
                print(f"  [{i}/{len(todo)}] {draft['draft_id']}: no score available, skipped")
                continue
            row = {"critic": critic, "draft_id": draft["draft_id"],
                   "score": result if isinstance(result, float) else None,
                   "decision": result if isinstance(result, str) else None,
                   "seconds": round(seconds, 3)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"  [{i}/{len(todo)}] {draft['draft_id']} {draft['draft']!r} -> {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("critic", choices=sorted(CRITICS))
    parser.add_argument("draft_set", choices=["first", "all"])
    args = parser.parse_args()
    score(args.critic, args.draft_set)
