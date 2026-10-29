"""Actual Sri-head ablations over the teammates' 900 joined saved rounds.

This is a cached-trajectory stopping simulation. It does not create revised
answers, measure live speed, or silently turn saved scores into new model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

from .linear import LinearHead, SRI_REVISION, make_features

TEAM_REVISION = "6c72991652400dcfec8f464a331efb95d5d1cf8b"
CSV_SHA256 = "15edd9ea608e231ccd8f74fe461f2b5f917f80187140a4edb303fc1bf93ce394"
POLICIES = ("fixed1", "fixed6", "answer_unchanged", "strands_only", "linear_transfer", "pair_and")


def load_trajectories(path: str | Path, *, verify_pin=True) -> list[dict]:
    path = Path(path)
    if verify_pin and hashlib.sha256(path.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError("Joined teammate CSV differs from its pinned SHA256")
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {"round": int(raw["round"]), "answer": raw["answer"], "split": raw["split"],
                   "strands": float(raw["strands"]), "f1": float(raw["f1"]), "em": float(raw["em"]),
                   "tokens": int(raw["tokens"]), "writer_tokens": int(raw["writer_total_tokens"]),
                   "critic_tokens": int(raw["critic_total_tokens"])}
            if (row["split"] not in {"train", "dev", "test"} or not raw["id"]
                    or any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ("strands", "f1", "em"))
                    or row["em"] not in (0., 1.) or min(row["tokens"], row["writer_tokens"], row["critic_tokens"]) < 0
                    or row["tokens"] != row["writer_tokens"] + row["critic_tokens"]):
                raise ValueError("Invalid saved probability, evaluator metric, split or token accounting")
            grouped[raw["id"]].append(row)
    result = []
    for qid, rounds in sorted(grouped.items()):
        rounds.sort(key=lambda r: r["round"])
        if len(rounds) != 6 or [r["round"] for r in rounds] != list(range(1, 7)) or len({r["split"] for r in rounds}) != 1:
            raise ValueError("Each question requires exactly six unique, consistently assigned rounds")
        result.append({"id": qid, "split": rounds[0]["split"], "rounds": rounds})
    if verify_pin and Counter(t["split"] for t in result) != {"train": 50, "dev": 20, "test": 80}:
        raise ValueError("The pinned 50/20/80 question partition changed")
    if not result:
        raise ValueError("No teammate trajectories were loaded")
    return result


def replay_one(trajectory: dict, policy: str, head: LinearHead, *, strands_threshold=.8,
               stop_threshold=.5) -> dict:
    if policy not in POLICIES:
        raise ValueError("Unknown predeclared replay policy")
    if strands_threshold != .8 or stop_threshold != .5:
        raise ValueError("The teammate's declared thresholds are frozen at .8 and .5")
    rounds = trajectory["rounds"]
    writer_tokens = critic_tokens = strands_calls = head_calls = 0
    previous_distance = None
    decisions = []
    for i, row in enumerate(rounds):
        writer_tokens += row["writer_tokens"]
        final = i == len(rounds) - 1
        previous = rounds[i - 1]["answer"] if i else None
        approved = row["strands"] >= strands_threshold
        p_stop = None
        if policy in {"linear_transfer", "pair_and"}:
            features = make_features(row["answer"], previous, i, len(rounds), approved, previous_distance)
            p_stop = head.probability(features)
            previous_distance = None if i == 0 else features[0]
            head_calls += 1
        if policy in {"strands_only", "linear_transfer", "pair_and"}:
            strands_calls += 1
        if policy == "fixed1":
            stop = True
        elif policy == "fixed6":
            # Matches the teammate's recorded Fixed-6 baseline, including its final critic.
            critic_tokens += row["critic_tokens"]
            stop = final
        elif policy == "answer_unchanged":
            stop = final or (i > 0 and row["answer"].strip().lower() == previous.strip().lower())
        elif policy == "strands_only":
            stop = final or approved
        elif policy == "linear_transfer":
            stop = final or p_stop >= stop_threshold
        else:
            stop = final or (approved and p_stop >= stop_threshold)
        decisions.append({"round": i + 1, "strands_probability": row["strands"] if strands_calls else None,
                          "p_stop": p_stop, "stop": stop})
        # A continuing trajectory requires the saved reviewer's feedback. This is
        # a counterfactual prefix cost, not a newly measured execution duration.
        if not stop and policy != "fixed6":
            critic_tokens += row["critic_tokens"]
        if stop:
            return {"id": trajectory["id"], "split": trajectory["split"], "seed": head.seed,
                    "policy": policy, "rounds": i + 1, "answer": row["answer"],
                    "answer_f1": row["f1"], "answer_em": row["em"],
                    "llm_tokens": writer_tokens + critic_tokens, "writer_tokens": writer_tokens,
                    "reviewer_tokens": critic_tokens, "strands_calls": strands_calls,
                    "head_calls": head_calls, "decisions": decisions}
    raise ValueError("The trajectory had no terminal round")


def _ci(values, *, count=2000):
    if not values:
        raise ValueError("A paired comparison needs questions")
    rng = random.Random(42)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(count))
    return [means[int(count * .025)], means[min(count - 1, int(count * .975))]]


def analyze(trajectories: list[dict], *, split="test", seeds=(0, 1, 2, 3, 4)) -> dict:
    if split not in {"train", "dev", "test"}:
        raise ValueError("Report one original split at a time")
    selected = [t for t in trajectories if t["split"] == split]
    if not selected or len({t["id"] for t in selected}) != len(selected):
        raise ValueError("A nonempty, unique question partition is required")
    rows, summaries = [], []
    for seed in seeds:
        head = LinearHead.pretrained(seed)
        seed_rows = [replay_one(t, p, head) for t in selected for p in POLICIES]
        rows.extend(seed_rows)
        fixed = {r["id"]: r for r in seed_rows if r["policy"] == "fixed6"}
        for policy in POLICIES:
            pp = [r for r in seed_rows if r["policy"] == policy]
            n = len(pp)
            differences = [r["answer_f1"] - fixed[r["id"]]["answer_f1"] for r in pp]
            baseline_tokens = sum(r["llm_tokens"] for r in fixed.values())
            summary = {"seed": seed, "policy": policy, "questions": n,
                       **{f"mean_{key}": sum(r[key] for r in pp) / n for key in
                          ("answer_f1", "answer_em", "rounds", "llm_tokens", "writer_tokens", "reviewer_tokens", "strands_calls", "head_calls")},
                       "f1_difference_vs_fixed6": sum(differences) / n,
                       "paired_f1_difference_ci95_vs_fixed6": _ci(differences),
                       "llm_token_savings_percent_vs_fixed6": 100 * (1 - sum(r["llm_tokens"] for r in pp) / baseline_tokens),
                       "first_round_stops": sum(r["rounds"] == 1 for r in pp)}
            summaries.append(summary)
    paired = [r for r in rows if r["policy"] == "pair_and"]
    strands = {(r["seed"], r["id"]): r for r in rows if r["policy"] == "strands_only"}
    return {"schema": "haltiq-teammate-pair-replay-v1", "execution": "cached_trajectory_replay",
            "source_revision": TEAM_REVISION, "sri_revision": SRI_REVISION,
            "joined_csv_sha256": CSV_SHA256, "split": split, "unique_questions": len(selected),
            "seeds": list(seeds), "physical_new_model_calls": 0, "thresholds": {"strands": .8, "linear": .5},
            "method": "Pair requires both frozen Sri head stop and Strands approval; continue uses saved LLM-review feedback; final cap returns last answer.",
            "transfer_note": "The pretrained head's critic_approved feature originally came from an LLM reviewer. Here it is Strands >= .8; this is frozen transfer, not a newly trained paired controller.",
            "cost_note": "LLM tokens are saved writer/reviewer prefix estimates. Strands and reranker compute costs are unavailable in this CSV and reported as call counts, so token savings are not total-compute or live-latency savings.",
            "sample_note": "Seeds reuse the same questions; they are not additional independent samples. Local test80 is from official development data, not hidden official test data.",
            "pair_identical_to_strands_only": all(r["rounds"] == strands[(r["seed"], r["id"])]["rounds"] for r in paired),
            "inherited_classifier_first_round_collapse": all(r["rounds"] == 1 for r in rows if r["policy"] == "linear_transfer"),
            "summary": summaries, "rows": rows,
            "checkpoints": [LinearHead.pretrained(seed).as_dict() for seed in seeds]}


def run_replay(output: str | Path, data: str | Path | None = None, *, split="test") -> dict:
    """Write the five-seed teammate-only replay; return its complete report.

    ``output`` accepts either a JSON filename or an output directory.
    """
    data = Path(data) if data else Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv"
    report = analyze(load_trajectories(data), split=split)
    target = Path(output)
    if target.suffix.lower() != ".json":
        target = target / "teammate-replay.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    args = parser.parse_args()
    report = run_replay(args.output, args.data, split=args.split)
    print(json.dumps({k: report[k] for k in ("execution", "split", "unique_questions", "physical_new_model_calls", "pair_identical_to_strands_only", "inherited_classifier_first_round_collapse")}))


if __name__ == "__main__":
    main()
