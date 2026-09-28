"""Plot checked HotpotQA results. Install the optional .[plots] dependencies first."""

import argparse
import json
from pathlib import Path

from analyze_run import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    checked = analyze(args.run_directory)
    manifest = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator

    cap = manifest["config"]["max_rounds"]
    names = ["One draft", f"Fixed {cap} drafts", "Regular LLM critic", "Local typed critic"]
    baseline = checked["one_draft_baseline"]
    count = baseline["episodes"]
    values = [baseline] + [checked["arms"][arm] for arm in ("fixed", "llm", "local-typed")]
    correct = [v["correct"] for v in values]
    writer = [round(v["mean_drafts"] * count) for v in values]
    judges = [v["judge_calls"] for v in values]
    positions = list(range(len(names)))
    colors = ["#778697", "#778697", "#227f80", "#c28237"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.25, top=0.76, wspace=0.42)
    fig.suptitle("HotpotQA development results", x=0.035, y=0.97, ha="left", fontsize=21, weight="bold")
    model = manifest["config"]["writer_model"]
    date = manifest["started_at"][:10]
    fig.text(0.035, 0.88, f"Writer: {model} | {count} questions | at most {cap} drafts | {date}",
             fontsize=11, color="#475569")

    ax = axes[0]
    bars = ax.barh(positions, [v["answer_em"] * 100 for v in values], color=colors, height=0.56)
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Answer exact match")
    ax.set_title("Accuracy", loc="left", pad=12)
    ax.bar_label(bars, labels=[f"{n}/{count}" for n in correct], padding=6)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.barh(positions, writer, color="#3b596f", height=0.56, label="Writer")
    ax.barh(positions, judges, left=writer, color="#c6d2da", height=0.56, label="Critic")
    for position, first, second in zip(positions, writer, judges):
        ax.text(first + second + 1, position, str(first + second), va="center")
    ax.set_yticks(positions, [])
    ax.invert_yaxis()
    ax.set_xlim(0, max(a + b for a, b in zip(writer, judges)) * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.set_xlabel(f"Model calls across {count} questions")
    ax.set_title("Calls if each system ran separately", loc="left", pad=12)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.04, -0.22))

    fig.text(0.035, 0.10, "One-draft scores come from saved initial answers. Shared first drafts were called once in the actual run.", fontsize=10)
    fig.text(0.035, 0.055, "Development results only. No live Jev run; local typed scores are from Qwen. Call counts do not measure equal computation.",
             fontsize=10, color="#475569")
    for extension in ("png", "svg"):
        fig.savefig(args.run_directory / f"comparison.{extension}", dpi=180, facecolor="white")
    svg = args.run_directory / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"Saved comparison.png and comparison.svg in {args.run_directory}")


if __name__ == "__main__":
    main()
