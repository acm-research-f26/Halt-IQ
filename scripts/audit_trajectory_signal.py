"""Recompute the completed HotpotQA panel without loading models or calling APIs.

Answer quality measures the writer's selected draft. Critic diagnostics measure
how well a probability separates correct from incorrect drafts. This audit keeps
those outcomes separate and checks whether the saved trajectory can improve at
all, even under an oracle selector.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pc_strands.compare import first_draft_diagnostics
from pc_strands.data import load_tasks
from pc_strands.decider import probability_true
from pc_strands.metrics import evaluate

ARMS = {"untuned": "strands-untuned", "trained": "strands-trained",
        "jev": "jev-1.13-single-correctness"}
FIXED = {f"fixed-{n}-prefix" for n in (1, 2, 3, 5, 10)}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value["data"] if isinstance(value, dict) and "export_kind" in value else value


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    if actual is None or expected is None:
        require(actual is expected, message)
    else:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), message)


def analyze_trajectories(tasks, trajectories, *, expected_drafts):
    """Rescore every saved candidate; no favorable-candidate filtering."""
    ids = [task.id for task in tasks]
    require(bool(ids) and len(ids) == len(set(ids)), "empty or duplicate task cohort")
    require([row["task_id"] for row in trajectories] == ids, "trajectory cohort/order mismatch")
    by_id, quality_by_round = {}, [[] for _ in range(expected_drafts)]
    raw_counts, em_changes, f1_changes = Counter(), 0, 0
    best_em, best_f1 = [], []
    for task, trajectory in zip(tasks, trajectories):
        rounds = trajectory["rounds"]
        require(trajectory["status"] == "completed" and len(rounds) == expected_drafts,
                "incomplete saved trajectory")
        require([row["round"] for row in rounds] == list(range(1, expected_drafts + 1)),
                "noncontiguous saved rounds")
        rescored = []
        for index, row in enumerate(rounds):
            require(row["status"] == "completed" and isinstance(row["draft"], str),
                    "invalid saved draft")
            score = evaluate(row["draft"], task.answers)
            require(type(row["exact_match"]) is bool and row["exact_match"] == score["exact_match"],
                    "stored exact match differs from reference rescore")
            close(row["token_f1"], score["token_f1"], "stored F1 differs from reference rescore")
            rescored.append(score)
            quality_by_round[index].append(score)
        raw_counts[len({row["draft"] for row in rounds})] += 1
        em_changes += len({row["exact_match"] for row in rescored}) > 1
        f1_changes += len({row["token_f1"] for row in rescored}) > 1
        best_em.append(max(row["exact_match"] for row in rescored))
        best_f1.append(max(row["token_f1"] for row in rescored))
        by_id[task.id] = trajectory
    n = len(ids)
    round_scores = [{"draft": index + 1,
                     "em": sum(row["exact_match"] for row in rows) / n,
                     "token_f1": sum(row["token_f1"] for row in rows) / n}
                    for index, rows in enumerate(quality_by_round)]
    return by_id, {"questions": n, "saved_draft_states": n * expected_drafts,
        "drafts_per_question": expected_drafts,
        "raw_answer_variants_per_question": {str(k): v for k, v in sorted(raw_counts.items())},
        "questions_with_changing_em": em_changes, "questions_with_changing_f1": f1_changes,
        "stored_metric_mismatches": 0, "quality_by_draft": round_scores,
        "oracle_any_saved_draft_em": sum(best_em) / n,
        "oracle_best_saved_draft_f1": sum(best_f1) / n,
        "oracle_em_gain_over_first": sum(best_em) / n - round_scores[0]["em"],
        "oracle_f1_gain_over_first": sum(best_f1) / n - round_scores[0]["token_f1"]}


def check_exports(bundle, tasks, trajectories):
    task_map = {task.id: task for task in tasks}
    ids, policies, fixed_rows, row_count = set(task_map), {}, {}, 0
    for arm, primary in ARMS.items():
        rows = read_rows(bundle / f"exports/{arm}-common-schema-results.jsonl")
        expected = FIXED | {primary, primary + "-cap3"}
        seen, groups = set(), {}
        for row in rows:
            key = (row["question_id"], row["policy"])
            require(key not in seen and key[0] in ids and key[1] in expected,
                    "duplicate or foreign exported question/policy")
            seen.add(key)
            depth = row["rounds"]
            require(type(depth) is int and 1 <= depth <= len(trajectories[key[0]]["rounds"]),
                    "export selects an invalid draft")
            selected = trajectories[key[0]]["rounds"][depth - 1]
            require(row["final_answer"] == selected["draft"], "export answer differs from selected draft")
            score = evaluate(row["final_answer"], task_map[key[0]].answers)
            require(type(row["exact_match"]) is bool and row["exact_match"] == score["exact_match"],
                    "export exact match differs from reference rescore")
            close(row["token_f1"], score["token_f1"], "export F1 differs from reference rescore")
            if key[1].startswith("fixed-"):
                require(depth == int(key[1].split("-")[1]), "fixed control selects wrong depth")
                canonical = {k: v for k, v in row.items() if k != "source"}
                require(key not in fixed_rows or fixed_rows[key] == canonical,
                        "shared fixed controls disagree between exports")
                fixed_rows[key] = canonical
            groups.setdefault(key[1], []).append(row)
        require(set(groups) == expected and all({r["question_id"] for r in group} == ids
                for group in groups.values()), "missing exported cohort or policy")
        for name, group in groups.items():
            n = len(tasks)
            policies[name] = {"questions": n, "em": sum(r["exact_match"] for r in group) / n,
                "token_f1": sum(r["token_f1"] for r in group) / n,
                "mean_selected_drafts": sum(r["rounds"] for r in group) / n}
        row_count += len(rows)
    return policies, row_count


def audit_bundle(bundle):
    bundle = Path(bundle)
    names = ["data/fresh20-evaluator-only.jsonl", "data/fresh20-writer-trajectories.json",
        "analyses/primary-paired-analysis.json", "audits/optional-jev-independent-audit.json"]
    names += [f"exports/{arm}-{kind}.jsonl" for arm in ARMS
              for kind in ("common-schema-results", "first-draft-logits")]
    names += [f"proofs/{name}-calibration-proof.json" for name in ("released-calibrated", "head-trained")]
    before = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in names}
    tasks = load_tasks(bundle / names[0])
    require(len(tasks) == 20, "this completed study requires all twenty questions")
    ids = [task.id for task in tasks]
    trajectories, signal = analyze_trajectories(tasks, read_json(bundle / names[1]), expected_drafts=10)
    policies, row_count = check_exports(bundle, tasks, trajectories)
    primary = read_json(bundle / "analyses/primary-paired-analysis.json")
    hosted = read_json(bundle / "audits/optional-jev-independent-audit.json")
    require(primary["planned_question_ids"] == ids and primary["planned_n"] == len(ids),
            "primary comparison cohort mismatch")
    require(hosted["planned_n"] == len(ids) and hosted["threshold"] == .8,
            "hosted comparison scope mismatch")
    observations = hosted["per_question"]
    require([r["question_id"] for r in observations] == ids, "hosted probability cohort/order mismatch")
    for task, row in zip(tasks, observations):
        rounds = trajectories[task.id]["rounds"]
        require(row["question"] == task.question and row["references"] == task.answers
                and row["first_answer"] == rounds[0]["draft"]
                and row["tenth_answer"] == rounds[-1]["draft"]
                and row["first_em"] == rounds[0]["exact_match"], "hosted state/reference mismatch")
    first_results, logits_by_arm, probabilities = {}, {}, {}
    for arm, policy in ARMS.items():
        rows = read_rows(bundle / f"exports/{arm}-first-draft-logits.jsonl")
        require([r["question_id"] for r in rows] == ids, "logit cohort/order mismatch")
        temperature = None
        if arm != "jev":
            proof_name = "released-calibrated" if arm == "untuned" else "head-trained"
            proof = read_json(bundle / f"proofs/{proof_name}-calibration-proof.json")
            require(not (set(proof["question_ids"]) & set(ids)), "calibration overlaps fresh questions")
            temperature = proof["temperature"]
        decision_rows, probs = {}, []
        for qid, row, observation in zip(ids, rows, observations):
            label = int(trajectories[qid]["rounds"][0]["exact_match"])
            require(type(row["label"]) is int and row["label"] == label, "first logit label mismatch")
            if arm == "jev":
                require(row["logits"] is None, "hosted logits are unavailable, not reconstructed")
                p = observation["jev_first_p"]
            else:
                p = probability_true(row["logits"], temperature)
                close(p, observation["released_first_p" if arm == "untuned" else "trained_first_p"],
                      "saved probability differs from logits and temperature")
            require(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1,
                    "invalid stored probability")
            probs.append(p)
            decision_rows[qid] = {"decisions": [{"round": 1, "draft_exact_match": bool(label),
                "approved": p >= .8, "score": {"p_true": p, "logits": row["logits"],
                                                "temperature": temperature}}]}
        diagnostic = first_draft_diagnostics(decision_rows, ids)
        recorded = (hosted["policies"][policy]["first_draft_probability_diagnostics"] if arm == "jev"
                    else primary["policies"][policy]["first_draft_probabilities"])
        for key in ("auroc", "brier", "nll", "approved_first_drafts", "first_draft_approval_precision"):
            close(diagnostic[key], recorded[key], "recomputed first-decision diagnostic mismatch: " + key)
        first_results[arm] = {key: diagnostic[key] for key in ("scored_first_drafts", "auroc", "brier", "nll",
            "approved_first_drafts", "first_draft_approval_precision", "approval_coverage_of_planned")}
        first_results[arm]["wrong_first_approvals"] = sum(p >= .8 and not decision_rows[qid]["decisions"][0]["draft_exact_match"]
                                                                 for qid, p in zip(ids, probs))
        logits_by_arm[arm], probabilities[arm] = rows, probs
    for name, result in primary["policies"].items():
        require(result["planned_n"] == len(ids) and result["failed_or_missing"] == 0,
                "primary result missing questions")
        if name in policies:
            close(result["em"], policies[name]["em"], "primary EM differs from exports")
            close(result["token_f1"], policies[name]["token_f1"], "primary F1 differs from exports")
    after = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in names}
    require(before == after, "audit inputs changed while reading")
    changed = [a["logits"] != b["logits"] for a, b in zip(logits_by_arm["untuned"], logits_by_arm["trained"])]
    max_delta = max(abs(a - b) for row_a, row_b in zip(logits_by_arm["untuned"], logits_by_arm["trained"])
                    for a, b in zip(row_a["logits"], row_b["logits"]))
    return {"status": "PASS", "scope": "existing_outputs_no_new_model_calls",
        "trajectory_signal": signal, "export_rows_checked": row_count,
        "distinct_export_policies": len(policies), "answer_policies": policies,
        "first_decision_threshold": .8, "first_decision_metrics": first_results,
        "head_change": {"first_logit_pairs_changed": sum(changed), "max_absolute_logit_change": max_delta,
            "first_approval_disagreements": sum((a >= .8) != (b >= .8)
                                                for a, b in zip(probabilities["untuned"], probabilities["trained"]))},
        "input_sha256": before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=REPO / "results/strands-20261018")
    parser.add_argument("--output", type=Path, help="Optional aggregate JSON; contains no new model results")
    args = parser.parse_args()
    encoded = json.dumps(audit_bundle(args.bundle), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
