"""Compare every critic in scores/ on the same drafts; write report.md and reliability.png.

Usage: python3 report.py

"Approve" means score >= threshold for probability critics (same rule as Yash's
loop, default 0.8) or decision == "approve" for text-only critics.
"""

import json
import math
from pathlib import Path

from drafts import load_drafts

HERE = Path(__file__).resolve().parent
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
# Fixed order and color per critic, so a critic keeps its color across reruns.
ORDER = ["llm", "local-typed", "kev", "laya", "qwen8b-logprob", "qwen14b"]
COLORS = {"local-typed": "#2a78d6", "kev": "#eb6834", "laya": "#1baf7a", "qwen8b-logprob": "#eda100"}
MARKERS = {"local-typed": "o", "kev": "s", "laya": "^", "qwen8b-logprob": "D"}
SOURCE = {"llm": "reused (Yash run)", "local-typed": "reused (Yash run)", "kev": "reused (Sept 28 run)"}


def load_scores():
    scores = {}
    for path in sorted((HERE / "scores").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        scores[path.stem] = {row["draft_id"]: row for row in rows}
    return dict(sorted(scores.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def approves(row, threshold):
    return row["score"] >= threshold if row["score"] is not None else row["decision"] == "approve"


def auroc(pos, neg):
    """Chance a random correct draft scores above a random wrong one (ties count half)."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def errors(rows, threshold):
    wrong_ok = sum(approves(r, threshold) and not r["correct"] for r in rows)
    right_no = sum(not approves(r, threshold) and r["correct"] for r in rows)
    return wrong_ok, right_no


def evaluate(critic_rows, drafts):
    """Join cached scores to this draft set's labels; drafts a critic never scored are left out."""
    rows = [{**critic_rows[d["draft_id"]], "correct": d["correct"]} for d in drafts if d["draft_id"] in critic_rows]
    n_right = sum(r["correct"] for r in rows)
    approved = [r for r in rows if approves(r, 0.8)]
    wrong_ok, right_no = errors(rows, 0.8)
    probabilistic = all(r["score"] is not None for r in rows)
    return {
        "rows": rows, "n": len(rows), "probabilistic": probabilistic,
        "approval_acc": f"{sum(r['correct'] for r in approved)}/{len(approved)}"
                        + (f" ({sum(r['correct'] for r in approved) / len(approved):.0%})" if approved else ""),
        "wrong_ok": f"{wrong_ok}/{len(rows) - n_right}", "right_no": f"{right_no}/{n_right}",
        "auroc": auroc([r["score"] for r in rows if r["correct"]], [r["score"] for r in rows if not r["correct"]])
                 if probabilistic else None,
        "seconds": sum(r["seconds"] for r in rows) / len(rows),
    }


def reliability_chart(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=150)
    ax.plot([0, 1], [0, 1], color="#b8b7ae", linewidth=1.5, linestyle="--", label="perfectly calibrated")
    for critic, res in results.items():
        if not res["probabilistic"]:
            continue
        xs, ys, ns = [], [], []
        for low in (0.0, 0.2, 0.4, 0.6, 0.8):  # 5 equal-width bins of predicted P(correct)
            in_bin = [r for r in res["rows"] if low <= r["score"] < low + 0.2 or (low == 0.8 and r["score"] == 1.0)]
            if in_bin:
                xs.append(sum(r["score"] for r in in_bin) / len(in_bin))
                ys.append(sum(r["correct"] for r in in_bin) / len(in_bin))
                ns.append(len(in_bin))
        ax.plot(xs, ys, color=COLORS.get(critic, "#7a7a7a"), linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=[30 + 18 * n for n in ns], color=COLORS.get(critic, "#7a7a7a"),
                   marker=MARKERS.get(critic, "o"), edgecolor="white", linewidth=2, zorder=3,
                   label=f"{critic} (n={res['n']})")
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="Critic's predicted P(correct), binned",
           ylabel="Fraction actually correct (exact match)")
    ax.set_title("Critic reliability on all unique drafts\n(marker size = drafts in bin)", fontsize=11, loc="left")
    ax.grid(color="#e6e5df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, markerscale=0.6)
    fig.tight_layout()
    fig.savefig(path)


def main():
    scores = load_scores()
    lines = []
    for which in ("first", "all"):
        drafts = load_drafts(which)
        n, n_right = len(drafts), sum(d["correct"] for d in drafts)
        results = {critic: evaluate(rows, drafts) for critic, rows in scores.items()}
        lines += [f"## Draft set `{which}`: {n} drafts, {n_right} correct ({n_right / n:.0%})", "",
                  f"Approving everything would be right {n_right}/{n} ({n_right / n:.0%}) of the time; "
                  "a useful critic beats this on accuracy when approved.", "",
                  "| Critic | Source | n scored | Accuracy when approved (@0.8) | Wrong approvals @0.8 "
                  "| Wrongly rejected @0.8 | AUROC | Mean s/decision | Cost |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for critic, r in results.items():
            auc = f"{r['auroc']:.2f}" if r["auroc"] is not None else "— (text only)"
            lines.append(f"| {critic} | {SOURCE.get(critic, 'new (critic-bench)')} | {r['n']}/{n} | {r['approval_acc']} "
                         f"| {r['wrong_ok']} | {r['right_no']} | {auc} | {r['seconds']:.2f} | $0 (local) |")
        lines += ["", "Threshold sweep (probability critics): wrong approvals / wrongly rejected", "",
                  "| Critic | " + " | ".join(f"@{t}" for t in THRESHOLDS) + " |",
                  "|---|" + "---:|" * len(THRESHOLDS)]
        for critic, r in results.items():
            if r["probabilistic"]:
                lines.append(f"| {critic} | " + " | ".join("%d / %d" % errors(r["rows"], t) for t in THRESHOLDS) + " |")
        margin = 1.96 * math.sqrt(0.25 / n)
        lines += ["", f"Sample size: n = {n}, so any accuracy here is uncertain by roughly ±{margin:.0%} "
                      f"(95% interval, worst case p = 0.5). Accuracy-when-approved uses even fewer drafts.", ""]
        if which == "all":
            reliability_chart(results, HERE / "reliability.png")
    lines += ["![Reliability chart](reliability.png)", ""]
    text = "# Critic benchmark report\n\n" + "\n".join(lines)
    (HERE / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
