"""Offline first-draft threshold analysis from a saved haltiq run (no model calls).

For each question, the round-1 critic review of each typed arm judges the SAME shared
first draft. Pair its three noul scores with that draft's exact-match label, then:
  - mean scores for EM-correct vs EM-wrong first drafts, per check and for min(check)
  - AUROC / pairwise ranking of correct above wrong (min of the three is the gating score:
    "all three >= t" is equivalent to "min >= t")
  - approvals and wrong approvals at thresholds 0.45..0.80
Usage: python first_draft_thresholds.py RUN_DIR
"""

import json
import sys
from pathlib import Path
from statistics import mean

CHECKS = ("supported", "complete", "relevant")
ARMS = ("openjev", "local-typed")
THRESHOLDS = [round(0.45 + 0.05 * i, 2) for i in range(8)]


def load(run_dir: Path):
    reviews, first_em, first_draft = {}, {}, {}
    for line in (run_dir / "traces.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        if e["event"] == "shared_writer":
            first_draft[e["task_id"]] = e["draft"]
        elif e["event"] == "critic" and e["round"] == 1 and e["arm"] in ARMS:
            reviews[(e["task_id"], e["arm"])] = e["review"]
        elif e["event"] == "episode" and e["round_trace"]:
            r1 = e["round_trace"][0]
            first_em.setdefault(e["task_id"], set()).add(bool(r1["exact_match"]))
    for tid, labels in first_em.items():
        if len(labels) != 1:
            raise ValueError(f"{tid}: arms disagree on first-draft EM")
    return reviews, {t: labels.pop() for t, labels in first_em.items()}, first_draft


def auroc(pos, neg):
    """P(random correct scores above random wrong); ties count 1/2. Also the raw pair counts."""
    wins = ties = 0
    for p in pos:
        for n in neg:
            wins += p > n
            ties += p == n
    pairs = len(pos) * len(neg)
    return (wins + 0.5 * ties) / pairs if pairs else None, wins, ties, pairs


def main(run_dir: Path):
    reviews, em, drafts = load(run_dir)
    tasks = list(drafts)
    n_right = sum(em[t] for t in tasks)
    print(f"Run: {run_dir.name}")
    print(f"First drafts: {len(tasks)}, EM-correct {n_right}, EM-wrong {len(tasks) - n_right}")
    print("Scope: FIRST-DRAFT decisions only (round 1 on the shared draft), not the full revision loop.\n")
    for arm in ARMS:
        rows = [(reviews[(t, arm)]["scores"], em[t]) for t in tasks if (t, arm) in reviews]
        missing = [t for t in tasks if (t, arm) not in reviews]
        print(f"== {arm}  (n={len(rows)}{', MISSING ' + str(len(missing)) if missing else ''})")
        print(f"{'score':10s} {'mean correct':>13s} {'mean wrong':>11s} {'AUROC':>6s}  pairs (correct>wrong / ties / total)")
        for key in CHECKS + ("min",):
            val = (lambda s: min(s[c] for c in CHECKS)) if key == "min" else (lambda s, k=key: s[k])
            pos = [val(s) for s, ok in rows if ok]
            neg = [val(s) for s, ok in rows if not ok]
            a, w, ti, pr = auroc(pos, neg)
            print(f"{key:10s} {mean(pos):13.3f} {mean(neg):11.3f} {a:6.3f}  {w} / {ti} / {pr}")
        print(f"\n{'threshold':>9s} {'approved':>9s} {'wrong among approved':>21s} {'correct approved':>17s} {'correct rejected':>17s}")
        for t in THRESHOLDS:
            appr = [ok for s, ok in rows if all(s[c] >= t for c in CHECKS)]
            wrong = sum(not ok for ok in appr)
            print(f"{t:9.2f} {len(appr):9d} {f'{wrong} of {len(appr)}':>21s} {f'{len(appr) - wrong} of {n_right}':>17s} {n_right - (len(appr) - wrong):17d}")
        uniq = sorted({round(min(s[c] for c in CHECKS), 4) for s, _ in rows})
        print(f"distinct min-scores: {len(uniq)}  range {uniq[0]}–{uniq[-1]}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
