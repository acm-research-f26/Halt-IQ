"""One comparison table for every stopping method on the test split, plus an F1-vs-tokens chart.

    uv run --with matplotlib python eval/compare_all.py

(matplotlib is only needed for the chart; `uv run --with` uses a throwaway environment.)
Same data, replay, token accounting and bootstrap as eval/answer_selection.py (see its docstring).

Methods: fixed 1-6 rounds, stop once the answer repeats, the routing rule (Strands >= 0.8, else
revise until the answer repeats), Sri's original critic stopping (first round her LLM critic
approves), the best answer-selection variant from answer_selection.py, and Sri's retrained stop
heads. The selection variant is the best_strands row with the highest F1 (ties: fewer tokens),
picked on this same test split, so its row is optimistic.

Stop heads: for each head in HEADS, reads classifier/results/<head>/**/per_question.csv in Sri's
per-question format (columns id, rounds, and optionally seed and policy; rows with policy
"learned" are used when a policy column exists). Each question's stop round is replayed on the
saved trajectory with this accounting. Several seeds are averaged per question before the
bootstrap. A head with no such file is marked "pending".

For each method: F1, EM, tokens (% of fixed 6), avg rounds, F1 95% CI, and the fixed-round
baseline with the closest token cost with the paired F1 difference and CI.
Writes eval/results_table.csv and eval/f1_vs_tokens.png.
"""
import csv
import sys
from pathlib import Path

import answer_selection as sel

HERE = Path(__file__).resolve().parent
CLASSIFIER_RESULTS = HERE.parent / "classifier" / "results"
OUT = HERE / "results_table.csv"
CHART = HERE / "f1_vs_tokens.png"
# head -> (critic read on the stopping round, Strands scores every round seen).
# original and strands_add use her critic_approved feature every round; strands_replace swaps it for Strands.
HEADS = {"original": (True, False), "strands_replace": (False, True), "strands_add": (True, True)}


def head_stop_rounds(name):
    """{id: [stop round per seed]} from classifier/results/<name>/, or None if nothing is there."""
    stops = {}
    for path in sorted((CLASSIFIER_RESULTS / name).glob("**/per_question.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("policy", "learned") == "learned":
                    stops.setdefault(r["id"], []).append(int(r["rounds"]))
    return stops or None


def replay_head(traj, stops, critic_on_stop, strands_scored):
    """Replay each seed's stop round, then average per question across seeds."""
    missing = set(traj) - set(stops)
    if missing:
        sys.exit(f"stop rounds missing for {len(missing)} test questions, e.g. {sorted(missing)[0]}")
    out = []
    for qid in sorted(traj):
        runs = [sel.replay({qid: traj[qid]}, sel.fixed(k), critic_on_stop, strands_scored)[0] for k in stops[qid]]
        out.append({"id": qid, **{c: sel.mean([r[c] for r in runs]) for c in ("f1", "em", "tokens", "rounds", "strands_calls")}})
    return out


def main():
    traj = sel.load()
    full = sel.full_tokens(traj)
    methods = {}          # label -> per-question results, or None if pending
    notes = {}
    for k in range(1, 7):
        methods[f"fixed {k}"] = sel.replay(traj, *sel.RULES[f"fixed{k}"])
    methods["answer repeats"] = sel.replay(traj, *sel.RULES["answer_repeats"])
    methods["routing (Strands >= 0.8, else repeat)"] = sel.replay(traj, *sel.RULES["routing"])
    methods["Sri critic stop"] = sel.replay(traj, sel.sri_critic, True, False)

    variants = {rule: sel.replay(traj, *spec, selector="best_strands") for rule, spec in sel.RULES.items()}
    best = max(variants, key=lambda rule: (sel.mean([r["f1"] for r in variants[rule]]),
                                          -sel.mean([r["tokens"] for r in variants[rule]])))
    label = f"{best} + best-Strands answer"
    methods[label] = variants[best]
    notes[label] = "best of the best_strands variants, picked on this test split"

    for head, spec in HEADS.items():
        stops = head_stop_rounds(head)
        label = f"Sri stop head: {head}"
        if stops is None:
            methods[label] = None
            notes[label] = f"no per_question.csv under classifier/results/{head}/"
        else:
            methods[label] = replay_head(traj, stops, *spec)
            notes[label] = f"{max(map(len, stops.values()))} seed(s)"

    fixed_rows = {k: sel.summary(methods[f"fixed {k}"], full)["tokens_pct"] for k in range(1, 7)}
    rows = []
    for name, res in methods.items():
        row = {"method": name, "note": notes.get(name, "")}
        if res is None:
            row["status"] = "pending"
            rows.append(row)
            continue
        s = sel.summary(res, full)
        lo, hi = sel.boot_ci([r["f1"] for r in res])
        row.update(status="done", f1=s["f1"], em=s["em"], tokens_pct=s["tokens_pct"], avg_rounds=s["avg_rounds"],
                   avg_strands_calls=s["avg_strands_calls"], f1_ci_low=lo, f1_ci_high=hi)
        if not name.startswith("fixed "):
            k = min(fixed_rows, key=lambda k: abs(fixed_rows[k] - s["tokens_pct"]))
            d, dlo, dhi = sel.paired(res, methods[f"fixed {k}"])
            row.update(nearest_fixed=f"fixed {k}", nearest_fixed_tokens_pct=fixed_rows[k],
                       f1_diff_vs_nearest_fixed=d, diff_ci_low=dlo, diff_ci_high=dhi)
        rows.append(row)

    cols = ["method", "status", "f1", "em", "tokens_pct", "avg_rounds", "avg_strands_calls", "f1_ci_low", "f1_ci_high",
            "nearest_fixed", "nearest_fixed_tokens_pct", "f1_diff_vs_nearest_fixed", "diff_ci_low", "diff_ci_high", "note"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: round(v, 4) if isinstance(v, float) else v for c, v in r.items()})

    print(f"All methods, {sel.SPLIT} split ({len(traj)} questions). Tokens: % of fixed 6 rounds "
          f"({full:.0f} LLM tokens/question).\n")
    print(f"{'method':40s} {'F1':>6s} {'[95% CI]':16s} {'EM':>6s} {'tokens':>7s} {'rounds':>6s}  "
          f"{'equal-token baseline':22s} {'F1 diff [95% CI]':24s}")
    for r in rows:
        if r["status"] == "pending":
            print(f"{r['method']:40s} {'pending':>6s}  ({r['note']})")
            continue
        base = (f"{r['nearest_fixed']} ({r['nearest_fixed_tokens_pct']:.0f}%)" if "nearest_fixed" in r else "")
        diff = (f"{r['f1_diff_vs_nearest_fixed']:+.3f} [{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}]"
                if base else "")
        print(f"{r['method']:40s} {r['f1']:6.3f} [{r['f1_ci_low']:.3f}, {r['f1_ci_high']:.3f}] {r['em']:6.3f} "
              f"{r['tokens_pct']:6.0f}% {r['avg_rounds']:6.2f}  {base:22s} {diff}")
    for r in rows:
        if r["note"] and r["status"] == "done":
            print(f"note: {r['method']}: {r['note']}")

    plot([r for r in rows if r["status"] == "done"])
    print(f"\nwrote {OUT.relative_to(HERE.parent)}, {CHART.relative_to(HERE.parent)}")


def plot(rows):
    try:
        import matplotlib
    except ImportError:
        sys.exit("matplotlib is needed for the chart: uv run --with matplotlib python eval/compare_all.py")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = [r for r in rows if r["method"].startswith("fixed ")]
    other = [r for r in rows if not r["method"].startswith("fixed ")]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax.plot([r["tokens_pct"] for r in fixed], [r["f1"] for r in fixed], "-o", color="0.45", label="fixed k rounds", zorder=2)
    for r in fixed:
        ax.annotate(r["method"].split()[1], (r["tokens_pct"], r["f1"]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color="0.35")
    # Methods that land on the same point share one marker and one label.
    points = {}
    for r in other:
        points.setdefault((round(r["tokens_pct"], 2), round(r["f1"], 4)), []).append(r["method"])
    for i, ((x, y), names) in enumerate(points.items()):
        ax.scatter(x, y, s=45, zorder=3, color=f"C{i}")
        ax.annotate("\n".join(names), (x, y), textcoords="offset points", xytext=(8, -3), fontsize=8,
                    va="top", color=f"C{i}")
    ax.set_xlabel("LLM tokens (% of fixed 6 rounds)")
    ax.set_ylabel("F1 (test, 80 questions)")
    ax.set_xlim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("HotpotQA stopping: F1 vs. LLM token cost (replay of Sri's study150 test trajectories)", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART)


if __name__ == "__main__":
    main()
