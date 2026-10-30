"""Headroom for stopping on Sri's saved trajectories: oracle, threshold sweep, help vs. hurt, all 150.

    python3 eval/headroom.py

Reads data/sri_rounds_with_strands.csv. No model calls. Same replay, token accounting and paired
bootstrap (10,000 resamples, seed 0) as eval/answer_selection.py; tokens are % of fixed 6 rounds
computed on the same set of questions.

1. Oracle stopping (uses gold answers; upper bound only): stop at the earliest round that reaches
   the question's best F1 over rounds 1-6. For each method in eval/results_table.csv, the share of
   the possible gain it recovers: (F1 - fixed 1 F1) / (oracle F1 - fixed 1 F1), test split.
2. Routing threshold sweep (Strands >= t, else revise until the answer repeats) for t = 0.50-0.95.
   The threshold is picked on train+dev only (best F1 with tokens <= 30%; ties: fewer tokens),
   then reported on test. The test curve is also drawn by eval/compare_all.py.
3. Help vs. hurt: for each round transition k -> k+1, questions whose F1 went up, down or stayed
   the same, on all 150 questions and on test.
4. The rules with nothing tuned (fixed 1-6, answer repeats, routing at 0.8, Sri's critic stop)
   on all 150 questions. train and dev are included, but none of these rules was fit to them.

Writes eval/headroom.csv (one row per result; `section` says which part) and eval/help_hurt.csv.
"""
import csv
from pathlib import Path

import answer_selection as sel

HERE = Path(__file__).resolve().parent
RESULTS_TABLE = HERE / "results_table.csv"
OUT = HERE / "headroom.csv"
HELP_HURT = HERE / "help_hurt.csv"
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
MAX_TOKENS_PCT = 30


def oracle_stop(R):
    best = max(r["f1"] for r in R)
    return next(k for k in range(1, len(R) + 1) if R[k - 1]["f1"] == best)


def load(*splits):
    traj = {}
    for split in splits:
        traj.update(sel.load(split))
    return traj


def no_tuning(traj):
    """label -> per-question results for the rules with no fitted parameters."""
    methods = {f"fixed {k}": sel.replay(traj, *sel.RULES[f"fixed{k}"]) for k in range(1, 7)}
    methods["answer repeats"] = sel.replay(traj, *sel.RULES["answer_repeats"])
    methods[f"routing @ {sel.T_HI}"] = sel.replay(traj, *sel.RULES["routing"])
    methods["Sri critic stop"] = sel.replay(traj, sel.sri_critic, True, False)
    return methods


def sweep(traj):
    """Routing at each threshold: {threshold: per-question results}."""
    return {t: sel.replay(traj, sel.routing_at(t), False, True) for t in THRESHOLDS}


def pick(curve, full):
    """Best F1 with tokens <= MAX_TOKENS_PCT; ties go to fewer tokens."""
    ok = {t: sel.summary(res, full) for t, res in curve.items()}
    ok = {t: s for t, s in ok.items() if s["tokens_pct"] <= MAX_TOKENS_PCT}
    if not ok:
        raise SystemExit(f"no threshold stays within {MAX_TOKENS_PCT}% tokens")
    return max(ok, key=lambda t: (ok[t]["f1"], -ok[t]["tokens_pct"]))


def row(section, split, method, res, full, **extra):
    s = sel.summary(res, full)
    lo, hi = sel.boot_ci([r["f1"] for r in res])
    return {"section": section, "split": split, "method": method, "questions": s["questions"], "f1": s["f1"],
            "em": s["em"], "tokens_pct": s["tokens_pct"], "avg_rounds": s["avg_rounds"],
            "f1_ci_low": lo, "f1_ci_high": hi, **extra}


def help_hurt(traj, split):
    out = []
    for k in range(1, 6):
        d = [R[k]["f1"] - R[k - 1]["f1"] for R in traj.values()]
        changed = sum(not sel.same_answer(R[k], R[k - 1]) for R in traj.values())
        out.append({"split": split, "transition": f"{k}->{k + 1}", "questions": len(d),
                    "f1_up": sum(x > 0 for x in d), "f1_down": sum(x < 0 for x in d), "f1_same": sum(x == 0 for x in d),
                    "answer_text_changed": changed, "mean_f1_change": sel.mean(d)})
    return out


def fmt(x, spec):
    return format(x, spec) if isinstance(x, float) else str(x)


def main():
    test, tune, every = sel.load("test"), load("train", "dev"), load("train", "dev", "test")
    full_test, full_tune, full_all = sel.full_tokens(test), sel.full_tokens(tune), sel.full_tokens(every)
    rows = []

    # 1. Oracle stopping and share of the possible gain.
    oracle = sel.replay(test, oracle_stop, False, False)
    fixed1 = sel.summary(sel.replay(test, *sel.RULES["fixed1"]), full_test)["f1"]
    o = row("oracle", "test", "oracle stop (earliest best-F1 round)", oracle, full_test, gain_share=1.0)
    rows.append(o)
    gain = o["f1"] - fixed1
    with RESULTS_TABLE.open(newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    gains = []
    for r in table:
        done = r["status"] == "done"
        gains.append({"section": "gain_share", "split": "test", "method": r["method"],
                      "f1": float(r["f1"]) if done else "pending",
                      "tokens_pct": float(r["tokens_pct"]) if done else "pending",
                      "gain_share": (float(r["f1"]) - fixed1) / gain if done else "pending"})
    rows += gains

    # 2. Threshold sweep: pick on train+dev, report on test.
    tune_curve, test_curve = sweep(tune), sweep(test)
    chosen = pick(tune_curve, full_tune)
    for split, curve, full in (("train+dev", tune_curve, full_tune), ("test", test_curve, full_test)):
        for t, res in curve.items():
            rows.append(row("sweep", split, f"routing @ {t}", res, full, threshold=t,
                            note="chosen on train+dev" if t == chosen else ""))
    fixed_test = {k: sel.replay(test, *sel.RULES[f"fixed{k}"]) for k in range(1, 7)}
    fixed_pct = {k: sel.summary(res, full_test)["tokens_pct"] for k, res in fixed_test.items()}
    picked = test_curve[chosen]
    p = row("chosen_threshold", "test", f"routing @ {chosen} (chosen on train+dev)", picked, full_test,
            threshold=chosen, gain_share=(sel.mean([r["f1"] for r in picked]) - fixed1) / gain)
    k = min(fixed_pct, key=lambda k: abs(fixed_pct[k] - p["tokens_pct"]))
    d, lo, hi = sel.paired(picked, fixed_test[k])
    d8, lo8, hi8 = sel.paired(picked, test_curve[sel.T_HI])
    p.update(nearest_fixed=f"fixed {k}", f1_diff_vs_nearest_fixed=d, diff_ci_low=lo, diff_ci_high=hi,
             note=f"vs routing @ {sel.T_HI}: {d8:+.3f} [{lo8:+.3f}, {hi8:+.3f}]")
    rows.append(p)

    # 4. No-tuning rules on all 150 questions.
    for name, res in no_tuning(every).items():
        rows.append(row("all150", "all", name, res, full_all))

    # 3. Help vs. hurt per round transition.
    hh = help_hurt(every, "all") + help_hurt(test, "test")

    cols = ["section", "split", "method", "threshold", "questions", "f1", "em", "tokens_pct", "avg_rounds",
            "f1_ci_low", "f1_ci_high", "gain_share", "nearest_fixed", "f1_diff_vs_nearest_fixed",
            "diff_ci_low", "diff_ci_high", "note"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: round(v, 4) if isinstance(v, float) else v for c, v in r.items()})
    with HELP_HURT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(hh[0]))
        w.writeheader()
        for r in hh:
            w.writerow({c: round(v, 4) if isinstance(v, float) else v for c, v in r.items()})

    # Print.
    print(f"1. Oracle stopping, test ({len(test)} questions; uses gold answers, upper bound only)")
    print(f"   F1 {o['f1']:.3f} [{o['f1_ci_low']:.3f}, {o['f1_ci_high']:.3f}]  EM {o['em']:.3f}  "
          f"tokens {o['tokens_pct']:.0f}%  rounds {o['avg_rounds']:.2f}")
    print(f"   possible gain over fixed 1: {o['f1']:.3f} - {fixed1:.3f} = {gain:+.3f} F1\n")
    print(f"   {'method':40s} {'F1':>6s} {'tokens':>7s} {'gain recovered':>15s}")
    for r in gains:
        print(f"   {r['method']:40s} {fmt(r['f1'], '6.3f'):>6s} "
              f"{fmt(r['tokens_pct'], '6.0f') + ('%' if r['tokens_pct'] != 'pending' else ''):>7s} "
              f"{fmt(r['gain_share'], '14.0%'):>15s}")

    print(f"\n2. Routing threshold sweep (tokens <= {MAX_TOKENS_PCT}% on train+dev, best F1)")
    print(f"   {'threshold':>9s} | {'train+dev F1':>12s} {'tokens':>7s} | {'test F1':>7s} {'tokens':>7s} {'rounds':>6s}")
    for t in THRESHOLDS:
        a, b = sel.summary(tune_curve[t], full_tune), sel.summary(test_curve[t], full_test)
        mark = "  <- chosen" if t == chosen else ""
        print(f"   {t:9.2f} | {a['f1']:12.3f} {a['tokens_pct']:6.1f}% | {b['f1']:7.3f} {b['tokens_pct']:6.1f}% "
              f"{b['avg_rounds']:6.2f}{mark}")
    print(f"   chosen {chosen} on test: F1 {p['f1']:.3f} [{p['f1_ci_low']:.3f}, {p['f1_ci_high']:.3f}], tokens "
          f"{p['tokens_pct']:.0f}%, gain recovered {p['gain_share']:.0%}; vs {p['nearest_fixed']} "
          f"({fixed_pct[k]:.0f}%) {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; {p['note']}")

    print("\n3. Help vs. hurt per round transition (F1 of round k+1 vs. round k)")
    print(f"   {'split':5s} {'step':5s} {'up':>4s} {'down':>5s} {'same':>5s} {'answer changed':>15s} {'mean F1 change':>15s}")
    for r in hh:
        print(f"   {r['split']:5s} {r['transition']:5s} {r['f1_up']:4d} {r['f1_down']:5d} {r['f1_same']:5d} "
              f"{r['answer_text_changed']:15d} {r['mean_f1_change']:+15.3f}")

    print(f"\n4. Rules with nothing tuned, all {len(every)} questions (tokens % of fixed 6 on all 150)")
    print(f"   {'method':20s} {'F1':>6s} {'[95% CI]':16s} {'EM':>6s} {'tokens':>7s} {'rounds':>6s}")
    for r in rows:
        if r["section"] == "all150":
            print(f"   {r['method']:20s} {r['f1']:6.3f} [{r['f1_ci_low']:.3f}, {r['f1_ci_high']:.3f}] {r['em']:6.3f} "
                  f"{r['tokens_pct']:6.0f}% {r['avg_rounds']:6.2f}")
    print(f"\nwrote {OUT.relative_to(HERE.parent)}, {HELP_HURT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
