"""Replay stopping policies over saved rounds and print the results table.
No model calls. Works on MuSiQue rounds and on Sri's HotpotQA rounds.

python results.py --rounds data/musique_rounds.csv --judge data/musique_judge.csv --name musique
python results.py --rounds data/sri_rounds_with_strands.csv --judge data/hotpot_judge.csv --split test --name hotpot
"""
import argparse
import csv
import random
from collections import defaultdict

from common import normalize_answer


def load(path, split):
    by_id = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if split and r.get("split", "test") != split:
                continue
            by_id[r["id"]].append(r)
    for rows in by_id.values():
        rows.sort(key=lambda r: int(r["round"]))
    return dict(by_id)


def same(a, b):
    return normalize_answer(a["answer"]) == normalize_answer(b["answer"])


def tok(rows, upto):  # writer tokens for rounds 1..upto
    return sum(float(r["tokens"]) for r in rows[:upto])


# Each policy returns (f1, tokens) for one question.
def fixed(k):
    return lambda rows, j: (rows[min(k, len(rows)) - 1], tok(rows, min(k, len(rows))))


def oracle(rows, j):
    best = max(float(r["f1"]) for r in rows)
    t = next(i for i, r in enumerate(rows) if float(r["f1"]) == best) + 1
    return rows[t - 1], tok(rows, t)


def answer_repeats(rows, j):
    for t in range(2, len(rows) + 1):
        if same(rows[t - 2], rows[t - 1]):
            return rows[t - 1], tok(rows, t)
    return rows[-1], tok(rows, len(rows))


def routing_rule(threshold=0.8):
    """Friday's rule: stop if Strands is confident, else revise until the answer repeats."""
    def pol(rows, j):
        for t in range(1, len(rows) + 1):
            r = rows[t - 1]
            if float(r["strands"]) >= threshold or (t >= 2 and same(rows[t - 2], r)):
                return r, tok(rows, t)
        return rows[-1], tok(rows, len(rows))
    return pol


def judge_policy(col):
    """Orchestrator: stop when the answer repeats; when it changes, ask the judge.
    WORSE -> stop and return the previous answer. BETTER -> keep revising.
    Judge tokens are charged to the policy."""
    def pol(rows, j):
        extra = 0.0
        for t in range(2, len(rows) + 1):
            prev, cur = rows[t - 2], rows[t - 1]
            if same(prev, cur):
                return cur, tok(rows, t) + extra
            v = j.get((cur["id"], cur["round"]))
            if v is None:  # change not judged (should not happen): treat as BETTER
                continue
            extra += float(v[col + "_tokens"])
            if v[col] == "WORSE":
                return prev, tok(rows, t) + extra
        return rows[-1], tok(rows, len(rows)) + extra
    return pol


def paired_ci(a, b, n=2000, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(a)))
    diffs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        diffs.append(sum(a[i] - b[i] for i in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--split", default=None, help="e.g. test for Sri's HotpotQA file")
    ap.add_argument("--name", default="run")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    data = load(args.rounds, args.split)
    ids = sorted(data)
    R = max(len(r) for r in data.values())
    judge = {}
    if args.judge:
        with open(args.judge) as f:
            for r in csv.DictReader(f):
                judge[(r["id"], r["round"])] = r

    # ---- headroom: how much room is there for revision to help? ----
    changes = sum(not same(a, b) for rows in data.values() for a, b in zip(rows, rows[1:]))
    improvable = sum(max(float(r["f1"]) for r in rows) > float(rows[0]["f1"]) for rows in data.values())
    print(f"\n== {args.name}: {len(ids)} questions, {R} rounds ==")
    print(f"Answer changes between rounds: {changes}")
    all_rows = [r for rows in data.values() for r in rows]
    if "parsed" in all_rows[0]:
        ok = sum(r["parsed"] == "1" for r in all_rows)
        later = [r for r in all_rows if r["round"] != "1"]
        print(f"Rounds with a usable 'Answer:' line: {ok} of {len(all_rows)} ({100*ok/len(all_rows):.0f}%); "
              f"revision rounds only: {100*sum(r['parsed'] == '1' for r in later)/len(later):.0f}%")
    print(f"Questions where a later round beats round 1: {improvable} of {len(ids)}")

    policies = [("Fixed 1", fixed(1)), ("Fixed 2", fixed(2)), (f"Fixed {R}", fixed(R)),
                ("Oracle stopping", oracle), ("Answer repeats", answer_repeats)]
    if "strands" in next(iter(data.values()))[0] and all(r.get("strands") not in ("", None)
                                                          for rows in data.values() for r in rows):
        policies.append((f"Routing rule ({args.threshold})", routing_rule(args.threshold)))
    if judge and len(judge) < changes:
        print(f"NOTE: judge covered {len(judge)} of {changes} changes. Judge accuracy below is valid; "
              f"the judge rows in the F1 table are NOT (unjudged changes are treated as BETTER).")
    if judge:
        policies.append(("Repeats + CoCT judge", judge_policy("coct")))
        if any(v.get("direct") for v in judge.values()):
            policies.append(("Repeats + direct judge", judge_policy("direct")))

    full_tokens = sum(tok(data[i], len(data[i])) for i in ids)
    per_q, table = {}, []
    for name, pol in policies:
        res = [pol(data[i], judge) for i in ids]
        per_q[name] = [float(r[0]["f1"]) for r in res]
        f1 = sum(per_q[name]) / len(res)
        em = sum(float(r[0]["em"]) for r in res) / len(res)
        pct = 100 * sum(r[1] for r in res) / full_tokens
        table.append((name, f1, em, pct))

    print(f"\n| Method | Answer quality (F1) | Accuracy (EM) | Tokens (% of {R} rounds) | Tokens saved |\n|---|---|---|---|---|")
    for name, f1, em, pct in table:
        print(f"| {name} | {f1:.3f} | {100*em:.1f}% | {pct:.0f}% | {100-pct:.0f}% |")

    base = per_q["Answer repeats"]
    print("\nPaired 95% CI of F1 difference vs answer repeats:")
    for name in per_q:
        if name != "Answer repeats":
            lo, hi = paired_ci(per_q[name], base)
            d = sum(per_q[name]) / len(base) - sum(base) / len(base)
            print(f"  {name}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    if judge:
        js = [v for (qid, _), v in judge.items() if qid in data]
        lab = [v for v in js if v["label"] != "tie"]
        print(f"\nJudge: {len(js)} changes judged — help {sum(v['label']=='help' for v in js)}, "
              f"hurt {sum(v['label']=='hurt' for v in js)}, tie {sum(v['label']=='tie' for v in js)}")
        for col in ["coct", "direct"]:
            if lab and lab[0].get(col):
                ok = sum((v[col] == "BETTER") == (v["label"] == "help") for v in lab)
                hurt = [v for v in lab if v["label"] == "hurt"]
                caught = sum(v[col] == "WORSE" for v in hurt)
                print(f"  {col}: accuracy {ok}/{len(lab)} = {ok/len(lab):.0%}; "
                      f"harmful revisions caught {caught}/{len(hurt)}")

    out = f"results_{args.name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "f1", "em", "tokens_pct", "tokens_saved_pct"])
        for name, f1, em, pct in table:
            w.writerow([name, round(f1, 4), round(em, 4), round(pct, 1), round(100 - pct, 1)])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
