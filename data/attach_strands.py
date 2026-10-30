"""Attach Strands scores to every round of Sri's study150 trajectories.

    python3 data/attach_strands.py

Reads Sri's cache (implementation-3/study150/cache/hotpot_*.json) straight from the
pinned `sri` commit with git, so no checkout of her branch is needed, and joins
haltiq_sri_trajectory_strands.csv on the exact (id, answer) pair. Writes
sri_rounds_with_strands.csv: one row per (question, round), 150 x 6 = 900 rows.

Columns:
- id, split, round (1-6), answer, strands
- f1, em: her per-round `quality` (answer F1) and `answer_em`
- critic_approved: 1 if her LLM critic returned APPROVED that round, else 0
- tokens: her per-round total, writer total_tokens + critic total_tokens
- writer_* / critic_*: prompt, completion, total and cached prompt tokens from each call's usage

`split` is the cache's own label (train 50 / dev 20 / test 80; `dev` is her validation
set), checked against study150/questions.json at the same commit.
Exits non-zero without writing the output if any round has no score.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRI_COMMIT = "5503cab"
STUDY = "implementation-3/study150"
SCORES = HERE / "haltiq_sri_trajectory_strands.csv"
OUT = HERE / "sri_rounds_with_strands.csv"
EXPECTED_QUESTIONS, EXPECTED_ROUNDS = 150, 6
ROLES = ("writer", "critic")
USAGE = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
COLUMNS = (["id", "split", "round", "answer", "strands", "f1", "em", "critic_approved", "tokens"]
           + [f"{role}_{field}" for role in ROLES for field in USAGE])


def git(*args):
    return subprocess.run(["git", "-C", str(HERE), *args], check=True,
                          capture_output=True, text=True).stdout


def usage(call):
    u = call["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", "")
    return [u["prompt_tokens"], u["completion_tokens"], u["total_tokens"], cached]


def main():
    with SCORES.open(newline="", encoding="utf-8") as f:
        scores = {}
        for r in csv.DictReader(f):
            key = (r["id"], r["answer"])
            if key in scores:
                sys.exit(f"duplicate score for {key}")
            scores[key] = r["strands"]

    splits = {q["id"]: q["split"] for q in json.loads(git("show", f"{SRI_COMMIT}:{STUDY}/questions.json"))}
    # --full-tree: without it, ls-tree run from data/ lists only entries under data/.
    paths = sorted(p for p in git("ls-tree", "--full-tree", "--name-only", f"{SRI_COMMIT}:{STUDY}/cache/").split()
                   if p.startswith("hotpot_") and p.endswith(".json"))
    if len(paths) != EXPECTED_QUESTIONS:
        sys.exit(f"expected {EXPECTED_QUESTIONS} trajectories at {SRI_COMMIT}, found {len(paths)}")

    rows, misses, used = [], [], set()
    for p in paths:
        t = json.loads(git("show", f"{SRI_COMMIT}:{STUDY}/cache/{p}"))
        if t["split"] != splits.get(t["id"]):
            sys.exit(f"{t['id']}: cache split {t['split']!r} != questions.json {splits.get(t['id'])!r}")
        if len(t["rounds"]) != EXPECTED_ROUNDS:
            sys.exit(f"{t['id']}: {len(t['rounds'])} rounds, expected {EXPECTED_ROUNDS}")
        for k, r in enumerate(t["rounds"], 1):
            key = (t["id"], r["answer"])
            if key not in scores:
                misses.append((t["id"], k, r["answer"]))
                continue
            used.add(key)
            rows.append([t["id"], t["split"], k, r["answer"], scores[key], r["quality"], r["answer_em"],
                         int(r["critic_approved"]), r["tokens"]]
                        + usage(r["writer"]) + usage(r["critic"]))

    print(f"{len(paths)} trajectories, {len(rows) + len(misses)} rounds, {len(rows)} scored, {len(misses)} missing")
    print(f"{len(used)}/{len(scores)} scores used")
    for qid, k, answer in misses:
        print(f"  missing: {qid} round {k} answer {answer!r}")
    if misses:
        sys.exit(f"not writing {OUT.name}: {len(misses)} rounds have no Strands score")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
