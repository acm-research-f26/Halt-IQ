"""Answer selection: return the best-Strands answer seen so far instead of the last one.

    python3 eval/answer_selection.py

Reads data/sri_rounds_with_strands.csv, test split only (80 questions x 6 saved rounds).
Replay only: stopping cuts a saved trajectory short and never changes earlier rounds.
For each stopping rule, compares three ways of picking the returned answer:
- last:         the answer from the round where the loop stops
- best_strands: highest Strands score among rounds seen so far (ties go to the earliest round)
- oracle:       highest F1 among rounds seen so far (uses gold answers; upper bound only)
Selection adds no LLM tokens. best_strands needs a Strands score for every round seen, so its
Strands calls equal the rounds used (the routing rule already makes those calls).

Token accounting (the same for every method):
- writer tokens for every round produced
- LLM critic tokens for a round only when that critique is used: as feedback for the next
  revision, or as the stop signal (Sri's critic rule, in compare_all.py)
- Strands calls are counted separately and add no LLM tokens
- 100% = fixed 6 rounds = writer rounds 1-6 + critic rounds 1-5
The replay notebook (strands/sri_trajectory_replay.ipynb) also charged the round-6 critic in
fixed6, so its token percentages are a little lower than these.

Paired bootstrap: 10,000 resamples of the 80 questions, seed 0, the same resamples for every
comparison. Writes eval/answer_selection.csv.
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "sri_rounds_with_strands.csv"
OUT = HERE / "answer_selection.csv"
SPLIT = "test"
T_HI = 0.8          # routing threshold, fixed in advance (same as the replay notebook)
B, SEED = 10_000, 0


def load(split=SPLIT):
    """Trajectories for one split: {id: [round dicts in round order]}."""
    traj = {}
    with DATA.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            traj.setdefault(r["id"], []).append({
                "round": int(r["round"]), "answer": r["answer"], "strands": float(r["strands"]),
                "f1": float(r["f1"]), "em": float(r["em"]), "critic_approved": r["critic_approved"] == "1",
                "writer": int(r["writer_total_tokens"]), "critic": int(r["critic_total_tokens"])})
    for rounds in traj.values():
        rounds.sort(key=lambda r: r["round"])
        assert [r["round"] for r in rounds] == list(range(1, len(rounds) + 1))
    return traj


def same_answer(a, b):
    return a["answer"].strip().lower() == b["answer"].strip().lower()


# Stopping rules: rounds -> number of rounds used (k, 1-based).
def fixed(k):
    return lambda R: min(k, len(R))


def answer_repeats(R):
    return next((k for k in range(2, len(R) + 1) if same_answer(R[k - 1], R[k - 2])), len(R))


def routing(R):
    """Stop if Strands >= T_HI, else revise until the answer repeats."""
    for k in range(1, len(R) + 1):
        if R[k - 1]["strands"] >= T_HI or (k > 1 and same_answer(R[k - 1], R[k - 2])):
            return k
    return len(R)


def sri_critic(R):
    """Sri's original rule: stop at the first round her LLM critic approves."""
    return next((k for k in range(1, len(R) + 1) if R[k - 1]["critic_approved"]), len(R))


# name -> (stop rule, critic read on the stopping round, Strands scores every round seen)
RULES = {**{f"fixed{k}": (fixed(k), False, False) for k in range(1, 7)},
         "answer_repeats": (answer_repeats, False, False),
         "routing": (routing, False, True)}

# Selectors: (rounds seen) -> index of the returned round.
SELECTORS = {
    "last": lambda R: len(R) - 1,
    "best_strands": lambda R: max(range(len(R)), key=lambda i: (R[i]["strands"], -i)),
    "oracle": lambda R: max(range(len(R)), key=lambda i: (R[i]["f1"], -i)),
}


def replay(traj, rule, critic_on_stop, strands_scored, selector="last"):
    """Per-question results, in a fixed id order."""
    out = []
    for qid in sorted(traj):
        R = traj[qid]
        k = rule(R)
        seen = R[:k]
        pick = seen[SELECTORS[selector](seen)]
        critic_rounds = k if critic_on_stop else k - 1
        out.append({"id": qid, "rounds": k, "answer": pick["answer"], "f1": pick["f1"], "em": pick["em"],
                    "tokens": sum(r["writer"] for r in seen) + sum(r["critic"] for r in R[:critic_rounds]),
                    "strands_calls": k if strands_scored or selector == "best_strands" else 0})
    return out


def mean(xs):
    return sum(xs) / len(xs)


_resamples = {}


def resamples(n):
    if n not in _resamples:
        rng = random.Random(SEED)
        _resamples[n] = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]
    return _resamples[n]


def boot_ci(values):
    """95% percentile bootstrap CI of the mean of per-question values (paired if values are differences)."""
    means = sorted(mean([values[i] for i in ix]) for ix in resamples(len(values)))
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def summary(res, full_tokens):
    return {"questions": len(res), "f1": mean([r["f1"] for r in res]), "em": mean([r["em"] for r in res]),
            "avg_tokens": mean([r["tokens"] for r in res]),
            "tokens_pct": 100 * mean([r["tokens"] for r in res]) / full_tokens,
            "avg_rounds": mean([r["rounds"] for r in res]),
            "avg_strands_calls": mean([r["strands_calls"] for r in res])}


def paired(a, b):
    """F1 difference a - b per question, with CI."""
    d = [x["f1"] - y["f1"] for x, y in zip(a, b)]
    assert [x["id"] for x in a] == [y["id"] for y in b]
    lo, hi = boot_ci(d)
    return mean(d), lo, hi


def full_tokens(traj):
    return mean([r["tokens"] for r in replay(traj, *RULES["fixed6"])])


def main():
    traj = load()
    full = full_tokens(traj)
    rows = []
    for name, spec in RULES.items():
        last = replay(traj, *spec)
        for sel in SELECTORS:
            res = last if sel == "last" else replay(traj, *spec, selector=sel)
            row = {"stop_rule": name, "selector": sel, **summary(res, full)}
            if sel != "last":
                d, lo, hi = paired(res, last)
                changed = [(x, y) for x, y in zip(res, last) if not same_answer(x, y)]
                row.update(f1_diff_vs_last=d, ci_low=lo, ci_high=hi, answers_changed=len(changed),
                           better=sum(x["f1"] > y["f1"] for x, y in changed),
                           worse=sum(x["f1"] < y["f1"] for x, y in changed),
                           same_f1=sum(x["f1"] == y["f1"] for x, y in changed))
            rows.append(row)

    cols = ["stop_rule", "selector", "questions", "f1", "em", "tokens_pct", "avg_tokens", "avg_rounds",
            "avg_strands_calls", "f1_diff_vs_last", "ci_low", "ci_high", "answers_changed", "better", "worse", "same_f1"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: round(v, 4) if isinstance(v, float) else v for c, v in r.items()})

    print(f"Answer selection, {SPLIT} split ({len(traj)} questions). Tokens: % of fixed 6 rounds "
          f"({full:.0f} LLM tokens/question). Oracle uses gold answers: upper bound only.\n")
    print(f"{'stop rule':15s} {'selector':12s} {'F1':>6s} {'EM':>6s} {'tokens':>7s} {'rounds':>6s} {'Strands':>7s}"
          f"  {'F1 vs last [95% CI]':26s} {'changed':>7s} {'better':>6s} {'worse':>5s} {'same F1':>7s}")
    for r in rows:
        diff = (f"{r['f1_diff_vs_last']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                if "f1_diff_vs_last" in r else "")
        changes = (f"{r['answers_changed']:7d} {r['better']:6d} {r['worse']:5d} {r['same_f1']:7d}" if diff else "")
        print(f"{r['stop_rule']:15s} {r['selector']:12s} {r['f1']:6.3f} {r['em']:6.3f} {r['tokens_pct']:6.0f}% "
              f"{r['avg_rounds']:6.2f} {r['avg_strands_calls']:7.2f}  {diff:26s} {changes}")
    print(f"\nwrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
