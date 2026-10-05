"""Compare every stopping policy on one cost–quality scale.

    python3 compare.py [--baseline fixed-3] [--split dev]

Reads every results/*.jsonl file (the common schema in schema.py), writes
compare.md (one row per policy) and cost_quality.png (cost on x, F1 on y).
Also adds a separate section for Vedanshi's critic-bench stop rules, which are
first-draft-only and therefore not directly comparable (see that section).
No model calls.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from schema import read_rows

HERE = Path(__file__).resolve().parent
CRITIC_BENCH = HERE.parent / "critic-bench"
ORACLES = {"oracle-fixed", "oracle-any-arm"}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    return {"n": len(rows), "em": mean([r["exact_match"] for r in rows]), "f1": mean([r["token_f1"] for r in rows]),
            "rounds": mean([r["rounds"] for r in rows]), "tokens": mean([r["total_tokens"] for r in rows]),
            "seconds": mean([r["seconds"] for r in rows]), "info": mean([r["info_score"] for r in rows])}


def policy_table(rows, baseline):
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[(row["policy"], row["source"])].append(row)
    stats = {key: summarize(group) for key, group in by_policy.items()}
    base = next((s for (policy, _), s in stats.items() if policy == baseline), None)
    if base is None:
        raise SystemExit(f"baseline policy {baseline!r} not found")
    saved = lambda value, ref: f"{1 - value / ref:+.0%}".replace("+", "") if value is not None and ref else "—"
    lines = [f"Baseline: **{baseline}** (always revise to the round cap). \"Saved\" = 1 − cost / baseline cost "
             "(negative = costs more). ΔF1 = F1 − baseline F1.", "",
             "| Policy | Source | n | EM | F1 | Avg rounds | Avg tokens | Avg seconds | Tokens saved | Seconds saved | ΔF1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (policy, source), s in sorted(stats.items(), key=lambda kv: kv[1]["seconds"]):
        label = policy + (" (upper bound)" if policy in ORACLES else "")
        tokens = f"{s['tokens']:,.0f}" if s["tokens"] is not None else "—"
        lines.append(f"| {label} | {source} | {s['n']} | {s['em']:.0%} | {s['f1']:.3f} | {s['rounds']:.2f} | {tokens} "
                     f"| {s['seconds']:.2f} | {saved(s['tokens'], base['tokens'])} | {saved(s['seconds'], base['seconds'])} "
                     f"| {s['f1'] - base['f1']:+.3f} |")
    return lines, stats


def scatter(stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=150)
    points = defaultdict(list)  # identical (x, y) points share one label
    for (policy, _), s in stats.items():
        points[(round(s["seconds"], 2), round(s["f1"], 3))].append(policy)
    same_quality = defaultdict(int)  # alternate labels above/below for points at the same F1
    for (x, y), policies in sorted(points.items()):
        oracle = all(p in ORACLES for p in policies)
        ax.scatter([x], [y], s=70, zorder=3, color="white" if oracle else "#2a78d6",
                   edgecolor="#7a7a7a" if oracle else "white", linewidth=1.8)
        below = same_quality[y] % 2 == 1
        same_quality[y] += 1
        ax.annotate(" / ".join(sorted(policies)), (x, y), textcoords="offset points",
                    xytext=(0, -14 if below else 9), ha="center", fontsize=8.5, color="#3b3b3b")
    ax.scatter([], [], s=60, color="#2a78d6", label="stopping policy")
    ax.scatter([], [], s=60, color="white", edgecolor="#7a7a7a", linewidth=1.8, label="oracle (upper bound)")
    ax.set(xlabel="Cost: model seconds per question (sum of call latencies)", ylabel="Quality: mean token F1")
    ax.set_title("Cost vs. quality per stopping policy (HotpotQA dev, n = 20)", fontsize=11, loc="left")
    ax.grid(color="#e6e5df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.25, y=0.2)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(path)


def api_cost(results, jev_cost, n):
    """Jev API spend per draft: mean cost per Jev call x share of drafts on which the rule calls Jev."""
    calls = sum(called == "jev" for *_, called in results)
    return f"${calls * jev_cost / n:.6f}" if calls and jev_cost else "$0"


def critic_bench_section():
    """Vedanshi's critic-bench stop rules on the 200 `extra` first drafts (cached scores, no model calls)."""
    if not (CRITIC_BENCH / "scores").exists():
        return []
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(CRITIC_BENCH))
    from combos import RULES  # noqa: E402  (critic-bench)
    from drafts import EXTRA_DIR, load_drafts, read_jsonl  # noqa: E402
    from report import jev_cost_per_decision, load_scores  # noqa: E402

    scores, drafts = load_scores(), load_drafts("extra")
    writer_seconds = mean([r["seconds"] for r in read_jsonl(EXTRA_DIR / "drafts.jsonl")])
    n, right = len(drafts), sum(d["correct"] for d in drafts)
    lines = ["## Critic-bench stop rules (first draft only, 200 `extra` dev questions)", "",
             "**Not on the same scale as the table above.** These 200 questions only have a first draft, so each rule is "
             "scored as *\"stop after round 1 if the rule approves\"*: how many drafts it stops on, and how good those are. "
             "What happens after a rejection (more rounds) was not run, so there is no final F1 for rejected drafts. "
             f"Writer cost for the first draft: {writer_seconds:.1f} s per question (qwen3:8b), the same for every rule; "
             "critic cost is extra. Jev is a paid API (cost from its budget ledger); every other critic runs locally. "
             "Strict exact-match label.", "",
             f"{n} drafts, {right} correct ({right / n:.0%}).", "",
             "| Rule | Stops at round 1 | EM of stopped drafts | Mean F1 of stopped drafts | Wrong stops | Right drafts sent on "
             "| Critic s per draft | API $ per draft |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    jev_cost = jev_cost_per_decision()
    f1 = {d["draft_id"]: d["f1"] for d in drafts}
    for name, rule, needs in RULES:
        usable = [d for d in drafts if all(d["draft_id"] in scores.get(c, {}) for c in needs)]
        if len(usable) < n:
            continue  # rule not scored on every extra draft
        results = [(d, *rule(scores, d["draft_id"])) for d in usable]
        stopped = [d for d, ok, _, _ in results if ok]
        em = sum(d["correct"] for d in stopped)
        lines.append(f"| {name} | {len(stopped)}/{n} | {em}/{len(stopped)} ({em / len(stopped):.0%}) "
                     f"| {mean([f1[d['draft_id']] for d in stopped]):.3f} | {len(stopped) - em}/{n - right} "
                     f"| {right - em}/{right} | {mean([sec for _, _, sec, _ in results]):.2f} "
                     f"| {api_cost(results, jev_cost, n)} |")
    return lines + [""]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="fixed-3")
    parser.add_argument("--split", default="dev", help="only compare rows from this split (never mix dev and test)")
    args = parser.parse_args()

    rows = [r for path in sorted((HERE / "results").glob("*.jsonl")) for r in read_rows(path) if r["split"] == args.split]
    questions = {r["question_id"] for r in rows}
    lines = ["# Stopping policies: cost vs. quality", "",
             f"Split: **{args.split}**, {len(questions)} questions, from {len({r['source'] for r in rows})} source(s). "
             "Generated by `compare.py` from `results/*.jsonl`; no model calls.", ""]
    table, stats = policy_table(rows, args.baseline)
    scatter(stats, HERE / "cost_quality.png")
    lines += table + ["", "![Cost vs. quality](cost_quality.png)", ""] + critic_bench_section()
    (HERE / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
