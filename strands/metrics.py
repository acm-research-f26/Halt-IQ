"""Score every Strands critic column (and Jev) on the 220 HotpotQA dev drafts.

    python3 metrics.py

Reads results/haltiq_strands_experiments2.csv (a superset of the first CSV; the
shared columns are checked to be identical) and writes:
- results/metrics_table.csv, results/metrics_table.md: AUROC with a 1000-sample
  bootstrap 95% CI (seed 0), approve rate, precision and wrong approvals at 0.8
- results/recall_table.md: recall@2/@3 of the gold paragraphs, MiniLM vs bge
- results/auroc_chart.png
- WRITEUP_DRAFT.md, filled from WRITEUP_TEMPLATE.md
AUROC and the approve rule come from critic-bench/report.py (eval-framework has no
AUROC code). Label = strict exact match (`em` column). No model calls.
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "critic-bench"))
from report import approves, auroc  # noqa: E402  (critic-bench)

THRESHOLD = 0.8
N_BOOT, SEED = 1000, 0
# column: (label, what the critic saw)
CRITICS = {
    "strands": ("Strands, truncated", "first 3,000 chars of the 10 paragraphs"),
    "strands_top2": ("Strands + MiniLM top-2", "MiniLM reranker top-2 paragraphs"),
    "strands_top3": ("Strands + MiniLM top-3", "MiniLM reranker top-3 paragraphs (pre-chosen headline)"),
    "strands_top4": ("Strands + MiniLM top-4", "MiniLM reranker top-4 paragraphs"),
    "bge_top2": ("Strands + bge top-2", "bge-reranker-base top-2 paragraphs"),
    "bge_top3": ("Strands + bge top-3", "bge-reranker-base top-3 paragraphs"),
    "strands_pairmax": ("Strands + MiniLM pairmax", "max over the pairs from MiniLM top-4"),
    "pair_mean": ("Strands + MiniLM pair mean", "mean over the pairs from MiniLM top-4"),
    "pair_2nd": ("Strands + MiniLM pair 2nd", "2nd-highest pair from MiniLM top-4"),
    "strands2": ("Strands, gold paragraphs", "gold paragraphs only (oracle, not deployable)"),
    "jev": ("Jev", "Jev's own harness context (not reranked)"),
}
CHART = ["strands", "strands_top2", "strands_top3", "strands_top4", "bge_top2", "bge_top3",
         "strands_pairmax", "strands2", "jev"]
ORACLE = "strands2"


def load():
    first = list(csv.DictReader((RESULTS / "haltiq_strands_experiments.csv").open(encoding="utf-8")))
    rows = list(csv.DictReader((RESULTS / "haltiq_strands_experiments2.csv").open(encoding="utf-8")))
    assert len(first) == len(rows) and all(a == {k: b[k] for k in a} for a, b in zip(first, rows)), \
        "shared columns differ between the two CSVs"
    return rows


def auroc_fast(scores, labels):
    """Same statistic as report.auroc (ties count half), vectorized for the bootstrap."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def metrics(rows):
    y = np.array([int(r["em"]) for r in rows])
    boot = np.random.default_rng(SEED).integers(0, len(y), size=(N_BOOT, len(y)))  # same resamples for every critic
    out = []
    for col, (label, context) in CRITICS.items():
        s = np.array([float(r[col]) for r in rows])
        point = auroc([v for v, c in zip(s, y) if c], [v for v, c in zip(s, y) if not c])
        assert abs(point - auroc_fast(s, y)) < 1e-12
        lo, hi = np.percentile([auroc_fast(s[i], y[i]) for i in boot], [2.5, 97.5])
        ok = np.array([approves({"score": v}, THRESHOLD) for v in s])
        out.append({"critic": col, "label": label, "context": context, "n": len(y), "n_correct": int(y.sum()),
                    "auroc": point, "auroc_ci_low": lo, "auroc_ci_high": hi,
                    "approved": int(ok.sum()), "approve_rate": ok.mean(),
                    "precision": y[ok].mean() if ok.any() else float("nan"),
                    "wrong_approvals": int((ok & (y == 0)).sum()), "n_wrong": int((y == 0).sum())})
    return out


def write_tables(out, rows):
    with (RESULTS / "metrics_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    n, right = out[0]["n"], out[0]["n_correct"]
    lines = [f"{n} HotpotQA dev drafts (qwen3:8b), {right} correct by strict exact match ({right / n:.0%}). "
             f"AUROC 95% CI: {N_BOOT}-sample bootstrap over drafts, seed {SEED}. "
             f"Approve = score >= {THRESHOLD}. Precision = share of approved drafts that are correct.", "",
             "| Critic | Context | AUROC (95% CI) | Approve rate | Precision | Wrong approvals |",
             "|---|---|---:|---:|---:|---:|"]
    for m in out:
        precision = f"{m['precision']:.0%}" if m["approved"] else "—"
        lines.append(f"| {m['label']} | {m['context']} | {m['auroc']:.3f} ({m['auroc_ci_low']:.3f}–{m['auroc_ci_high']:.3f}) "
                     f"| {m['approved']}/{n} ({m['approve_rate']:.0%}) | {precision} "
                     f"| {m['wrong_approvals']}/{m['n_wrong']} |")
    (RESULTS / "metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hit = lambda col: sum(r[col] == "True" for r in rows)
    recall = ["Share of drafts whose **both** gold paragraphs are in the reranker's top-k (diagnostic only; "
              "never used at inference).", "",
              "| Reranker | recall@2 | recall@3 |", "|---|---:|---:|"]
    for name, k2, k3 in (("MiniLM cross-encoder", "recall_top2", "recall_top3"),
                         ("bge-reranker-base", "bge_recall_top2", "bge_recall_top3")):
        recall.append(f"| {name} | {hit(k2)}/{n} ({hit(k2) / n:.0%}) | {hit(k3)}/{n} ({hit(k3) / n:.0%}) |")
    (RESULTS / "recall_table.md").write_text("\n".join(recall) + "\n", encoding="utf-8")
    return lines, recall


def chart(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {m["critic"]: m for m in out}
    order = sorted(CHART, key=lambda c: by[c]["auroc"])
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    for i, col in enumerate(order):
        m, oracle = by[col], col == ORACLE
        ink = "#6b6b6b" if oracle else "#111111"
        ax.errorbar(m["auroc"], i, xerr=[[m["auroc"] - m["auroc_ci_low"]], [m["auroc_ci_high"] - m["auroc"]]],
                    fmt="o", markersize=7, color=ink, ecolor=ink, elinewidth=1.6, capsize=3,
                    markerfacecolor="white" if oracle else ink, markeredgewidth=1.6)
        ax.annotate(f"{m['auroc']:.3f}", (m["auroc_ci_high"], i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color="#111111")
    labels = [by[c]["label"] + (" (oracle, not deployable)" if c == ORACLE else "") for c in order]
    ax.set_yticks(range(len(order)), labels, fontsize=9)
    ax.axvline(0.5, color="#999999", linewidth=1, linestyle="--")
    ax.annotate("chance", (0.5, -0.6), xytext=(3, 0), textcoords="offset points", fontsize=8, color="#555555")
    ax.set_xlim(0.3, 0.9)
    ax.set_xlabel(f"AUROC vs. strict exact match, 95% bootstrap CI ({N_BOOT} resamples)")
    ax.set_ylim(-0.9, len(order) - 0.5)
    ax.set_title(f"Strands Decider 2B as a critic, {out[0]['n']} HotpotQA dev drafts", fontsize=11)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "auroc_chart.png")


def writeup(out, rows, lines, recall):
    """Fill WRITEUP_TEMPLATE.md's {{...}} slots with computed numbers -> WRITEUP_DRAFT.md."""
    mean = lambda col: sum(float(r[col]) for r in rows) / len(rows)
    n, right = out[0]["n"], out[0]["n_correct"]
    values = {"METRICS": "\n".join(lines), "RECALL": "\n".join(recall), "N": str(n), "N_RIGHT": str(right),
              "BASE_RATE": f"{right / n:.0%}", "SEC_TOP3": f"{mean('sec_top3'):.2f}",
              "SEC_PAIRMAX": f"{mean('sec_pairmax'):.2f}"}
    for m in out:
        values[f"AUROC_{m['critic']}"] = f"{m['auroc']:.3f} ({m['auroc_ci_low']:.3f}–{m['auroc_ci_high']:.3f})"
        values[f"AUC_{m['critic']}"] = f"{m['auroc']:.3f}"  # point estimate only
        values[f"APPROVED_{m['critic']}"] = str(m["approved"])
    text = (HERE / "WRITEUP_TEMPLATE.md").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    assert "{{" not in text, "unfilled slot in WRITEUP_TEMPLATE.md"
    (HERE / "WRITEUP_DRAFT.md").write_text(text, encoding="utf-8")


def main():
    rows = load()
    out = metrics(rows)
    lines, recall = write_tables(out, rows)
    chart(out)
    writeup(out, rows, lines, recall)
    print("\n".join(lines + [""] + recall))


if __name__ == "__main__":
    main()
