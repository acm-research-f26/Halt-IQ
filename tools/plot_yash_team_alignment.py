"""Export a scientific figure from the frozen cached studies; no model calls."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "team-alignment-2026-10-30")
    args = parser.parse_args()
    root = args.artifact_root
    sources = {name: root / name / "test80.json"
               for name in ("Yash-Retrained-Strands-Head", "Yash-Quality-Strands-Head")}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in sources.items()}
    for report in reports.values():
        assert report["unique_test_questions"] == 80
        assert report["seeds"] == list(range(5))
        assert report["new_language_model_calls"] == 0
        for item in report["summary"]:
            rows = [row for row in report["rows"]
                    if row["seed"] == item["seed"] and row["policy"] == item["policy"]]
            assert len(rows) == len({row["id"] for row in rows}) == 80
            for metric in ("answer_f1", "llm_tokens", "rounds"):
                assert abs(sum(row[metric] for row in rows)/80 - item["mean_"+metric]) < 1e-9
    pg, bce = reports.values()

    def row(report, policy, seed=0):
        return next(item for item in report["summary"]
                    if item["seed"] == seed and item["policy"] == policy)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 7.5),
                                    gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=.18, right=.97, bottom=.26, top=.80, wspace=.42)
    fig.suptitle("Cached teammate trajectories: stopping changes without higher answer F1",
                 x=.04, y=.96, ha="left", fontsize=17, fontweight="bold")
    fig.text(.04, .905, "Yash | 80 previously viewed test questions | fitted on train50, selected on dev20",
             fontsize=11, color="#334155")
    names = ["Fixed1", "Fixed6", "Repeated answer", "Strands only", "PG head, seed0",
             "BCE head, seed0", "BCE AND Strands, seed0"]
    selected = [row(bce, p) for p in ("fixed1", "fixed6", "answer_unchanged", "strands_only")]
    selected += [row(pg, "retrained_head"), row(bce, "quality_head"), row(bce, "quality_head_and")]
    colors = ["#94a3b8", "#475569", "#111827", "#2563eb", "#c2410c", "#0f766e", "#7c3aed"]
    values = [item["mean_answer_f1"] for item in selected]
    left.barh(names, values, color=colors, height=.65)
    left.invert_yaxis()
    left.set_xlim(0, .45)
    left.set_xlabel("Mean answer F1 (0–1)")
    left.set_title("Primary seed0 comparison", loc="left", pad=15, fontweight="bold")
    left.set_axisbelow(True)
    left.grid(axis="x", color="#e2e8f0")
    for index, value in enumerate(values):
        left.text(value+.005, index, f"{value:.4f}", va="center", fontsize=9)

    right.set_title("All seeds: quality and saved prefix cost", loc="left", pad=15, fontweight="bold")
    right.set_xlabel("Mean saved writer/reviewer tokens (thousands)")
    right.set_ylabel("Mean answer F1")
    right.set_xlim(0, 20)
    right.set_ylim(.345, .405)
    right.grid(color="#e2e8f0")
    for policy, label, color, marker, offset in (
            ("fixed1", "Fixed1 / PG seeds0–4", "#c2410c", "v", (8, -18)),
            ("fixed6", "Fixed6", "#475569", "s", (-46, 10)),
            ("answer_unchanged", "Repeated answer", "#111827", "D", (7, 10)),
            ("strands_only", "Strands", "#2563eb", "s", (8, -16))):
        item = row(bce, policy)
        xy = (item["mean_llm_tokens"]/1000, item["mean_answer_f1"])
        right.scatter(*xy, s=65, c=color, marker=marker, zorder=4)
        right.annotate(label, xy, xytext=offset, textcoords="offset points", fontsize=9)
    for seed in range(5):
        item = row(bce, "quality_head", seed)
        right.scatter(item["mean_llm_tokens"]/1000, item["mean_answer_f1"],
                      s=140 if seed == 0 else 55, c="#0f766e", marker="*" if seed == 0 else "o",
                      edgecolors="white", linewidths=.5, zorder=5)
    first, capped = row(bce, "quality_head"), row(bce, "quality_head", 3)
    assert all(row(pg, "retrained_head", seed)["mean_llm_tokens"] == row(bce, "fixed1")["mean_llm_tokens"]
               and row(pg, "retrained_head", seed)["mean_answer_f1"] == row(bce, "fixed1")["mean_answer_f1"]
               for seed in range(5))
    assert all(row(bce, "quality_head", seed)["forced_cap_returns"] == 80 for seed in (3, 4))
    right.annotate("BCE seeds0–2\n★ predeclared seed0",
                   (first["mean_llm_tokens"]/1000, first["mean_answer_f1"]),
                   xytext=(25, -55), textcoords="offset points", fontsize=9, ha="center",
                   arrowprops={"arrowstyle": "-", "color": "#0f766e"})
    right.annotate("BCE seeds3/4\nall returns at cap",
                   (capped["mean_llm_tokens"]/1000, capped["mean_answer_f1"]),
                   xytext=(-28, -38), textcoords="offset points", fontsize=9, ha="center")
    fig.text(.04, .15, "Saved prefix estimates exclude verifier/reranker compute. Fixed6 includes its terminal reviewer; adaptive arms stop reviewing after return.", fontsize=9)
    fig.text(.04, .115, "All five seeds reuse the same 80 questions. Overlapping seed points are shown without jitter. This is cached replay, not fresh blind inference.", fontsize=9)
    fig.text(.04, .08, "BCE is a separate class-weighted correctness objective; its scores are not calibrated probabilities. No live speedup or noninferiority claim.", fontsize=9)
    files = []
    for extension in ("png", "pdf"):
        path = root / ("Yash-Cached-Quality-Cost-Figure-20261030." + extension)
        fig.savefig(path, dpi=220, facecolor="white")
        files.append(path)
    plt.close(fig)
    receipt = {"verified_local": datetime.now().astimezone().isoformat(),
        "matplotlib": matplotlib.__version__, "model_calls": 0,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources.values()},
        "figure_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "all_source_summary_values_recomputed": True, "unique_questions": 80,
        "scope": "known cached test80, all seeds; excludes verifier/reranker compute"}
    (root/"Yash-Cached-Figure-Verification.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
