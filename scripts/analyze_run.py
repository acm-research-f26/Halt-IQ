"""Check a completed run's saved artifacts and derive a one-draft baseline.

Rechecks recorded scores with the matching saved scorer; it does not independently
adjudicate gold answers or make model calls. Run: python scripts/analyze_run.py RUN_DIRECTORY
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{label}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}")
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"{label}: {actual!r} != {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{label}: values differ")


def aggregate(items):
    approvals = [ep for ep in items if ep["stop_reason"] == "critic_approved"]
    judgments = [r for ep in items for r in ep["round_trace"] if "approved" in r]
    judge_calls = sum(ep["judge_calls"] for ep in items)
    false_approvals = sum(not ep["exact_match"] for ep in approvals)
    return {
        "episodes": len(items), "completed": len(items), "failures": 0,
        "success_rate": mean(ep["success"] for ep in items), "completion_rate": 1.0,
        "completed_accuracy": mean(ep["exact_match"] for ep in items),
        "mean_token_f1": mean(ep["token_f1"] for ep in items),
        "mean_rounds": mean(ep["rounds"] for ep in items),
        "mean_revisions": mean(ep["rounds"] - 1 for ep in items),
        "mean_writer_calls": mean(ep["writer_calls"] for ep in items),
        "mean_judge_calls": mean(ep["judge_calls"] for ep in items),
        "judge_calls": judge_calls, "typed_questions": sum(ep["typed_questions"] for ep in items),
        "mean_judge_latency_ms": sum(ep["judge_latency_ms"] for ep in items) / judge_calls if judge_calls else None,
        "operational_input_tokens": sum(ep["input_tokens"] for ep in items),
        "operational_output_tokens": sum(ep["output_tokens"] for ep in items),
        "api_cost_usd": sum(ep["cost_usd"] for ep in items),
        "approvals": len(approvals), "false_approvals": false_approvals,
        "false_approval_rate": false_approvals / len(approvals) if approvals else None,
        "false_approvals_per_episode": false_approvals / len(items),
        "judgments": len(judgments),
        "correct_draft_rejections": sum(r["exact_match"] and not r["approved"] for r in judgments),
        "round_cap_episodes": sum(ep["stop_reason"] == "round_cap" for ep in items),
    }


def analyze(directory):
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    manifest, summary = read("manifest.json"), read("summary.json")
    require(manifest["status"] == "completed", "Only a fully completed, error-free run can be analyzed")
    require(manifest["run_kind"] == "live_models" and manifest["metrics"] == ["hotpotqa"],
            "Expected a live HotpotQA run")
    for key in ("status", "run_kind", "benchmarks", "metrics"):
        same(summary[key], manifest[key], f"summary.{key}")

    source_hash = hashlib.sha256()
    sources = sorted((directory / "source_snapshot").glob("*.py"))
    require(bool(sources), "Missing source snapshot")
    for path in sources:
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    require(source_hash.hexdigest() == manifest["source_sha256"], "Source snapshot hash mismatch")
    scorer_path = Path(__file__).resolve().parents[1] / "haltiq" / "metrics.py"
    require(scorer_path.read_bytes() == (directory / "source_snapshot" / "metrics.py").read_bytes(),
            "Current scorer differs from the saved scorer; use the matching project version")
    spec = importlib.util.spec_from_file_location("verified_hotpot_metrics", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    require(sha256(directory / "dataset_snapshot.jsonl") == manifest["dataset_sha256"], "Dataset snapshot hash mismatch")
    same(read("dataset_manifest.json")["output"]["sha256"], manifest["dataset_sha256"], "Dataset preparation hash")
    dataset = [json.loads(line) for line in (directory / "dataset_snapshot.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {task["id"]: task for task in dataset}
    require(len(by_id) == len(dataset), "Duplicate dataset task IDs")
    ids, arms = manifest["task_ids"], manifest["config"]["arms"]
    require(ids and arms and len(set(ids)) == len(ids) and len(set(arms)) == len(arms), "Invalid task IDs or arms")
    selected = [task["id"] for task in dataset if manifest["config"]["split"] == "all" or task["split"] == manifest["config"]["split"]]
    if manifest["config"].get("limit") is not None:
        selected = selected[:manifest["config"]["limit"]]
    same(ids, selected, "Manifest task selection")

    events = [json.loads(line) for line in (directory / "traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(not any(e["event"] in {"error", "invalid_output"} for e in events), "Trace contains an error")
    shared_events = [e for e in events if e["event"] == "shared_writer"]
    shared = {e["task_id"]: e for e in shared_events}
    require(len(shared) == len(shared_events) == len(ids) and set(shared) == set(ids), "Missing or duplicate first drafts")
    episodes = [e for e in events if e["event"] == "episode"]
    keyed = {(ep["task_id"], ep["arm"]): ep for ep in episodes}
    expected_pairs = {(task_id, arm) for task_id in ids for arm in arms}
    require(len(keyed) == len(episodes) and set(keyed) == expected_pairs, "Missing or duplicate episodes")
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    csv_keyed = {(row["task_id"], row["arm"]): row for row in rows}
    require(len(csv_keyed) == len(rows) and set(csv_keyed) == expected_pairs, "CSV episode coverage differs")

    first_scores = {}
    for pair, ep in keyed.items():
        task_id, arm = pair
        require(ep["status"] == "completed" and ep["error"] == "", f"{pair}: failed episode")
        expected_csv = {key: str(value) for key, value in ep.items() if key not in {"event", "round_trace"}}
        same(csv_keyed[pair], expected_csv, f"{pair}: CSV versus trace")
        rounds = ep["round_trace"]
        require(1 <= len(rounds) == ep["rounds"] <= manifest["config"]["max_rounds"], f"{pair}: invalid round count")
        same([r["round"] for r in rounds], list(range(1, len(rounds) + 1)), f"{pair}: round sequence")
        same(rounds[0]["draft"], shared[task_id]["draft"], f"{pair}: first draft reuse")
        for score in rounds:
            require(type(score["exact_match"]) is bool and type(score["token_f1"]) in (int, float)
                    and 0 <= score["token_f1"] <= 1, f"{pair}: invalid recorded score")
            rescored = scorer.hotpot_evaluate(score["draft"], by_id[task_id]["answers"])
            same({key: score[key] for key in rescored}, rescored, f"{pair}: round score versus saved gold")
        first = {key: rounds[0][key] for key in ("exact_match", "token_f1")}
        if task_id in first_scores:
            same(first, first_scores[task_id], f"{pair}: first-draft scoring agreement")
        first_scores[task_id] = first
        for key in ("exact_match", "token_f1"):
            same(ep[key], rounds[-1][key], f"{pair}: final score")
        same(ep["success"], ep["exact_match"], f"{pair}: success")
        same(ep["final_answer"], rounds[-1]["draft"], f"{pair}: final answer")
        same(ep["split"], by_id[task_id]["split"], f"{pair}: split")
        require(ep["writer_calls"] == len(rounds), f"{pair}: writer count")
        reviewed = [r for r in rounds if "approved" in r]
        require(ep["judge_calls"] == ep["judge_successful_calls"] == len(reviewed), f"{pair}: critic count")
        require(len(reviewed) == (0 if arm == "fixed" else len(rounds)), f"{pair}: missing decisions")
        if reviewed:
            require(all(type(r["approved"]) is bool for r in reviewed), f"{pair}: invalid decision")
            require(not any(r["approved"] for r in reviewed[:-1]), f"{pair}: continued after approval")
            require((ep["stop_reason"] == "critic_approved") == reviewed[-1]["approved"], f"{pair}: inconsistent stopping")
        if ep["stop_reason"] == "round_cap":
            require(len(rounds) == manifest["config"]["max_rounds"], f"{pair}: early cap")

    aggregates = {arm: aggregate([keyed[task_id, arm] for task_id in ids]) for arm in arms}
    same(summary["arms"], aggregates, "Recomputed summary")
    calls = [e["review"]["call"] if e["event"] == "critic" else e["call"] for e in events
             if e["event"] in {"shared_writer", "writer", "critic"}]
    actual = summary["actual_experiment"]
    require(len(calls) == len(ids) + sum(ep["writer_calls"] - 1 + ep["judge_calls"] for ep in episodes), "Physical call count mismatch")
    same(actual["successful_model_calls"], len(calls), "Physical model calls")
    for field, call_field in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"), ("api_cost_usd", "cost_usd")):
        same(actual[field], sum(call[call_field] for call in calls), f"Actual usage: {field}")

    baseline = {
        "episodes": len(ids), "correct": sum(v["exact_match"] for v in first_scores.values()),
        "answer_em": mean(v["exact_match"] for v in first_scores.values()),
        "answer_f1": mean(v["token_f1"] for v in first_scores.values()),
        "mean_drafts": 1.0, "judge_calls": 0,
        "note": "Derived from the saved shared first drafts, not an additional model run.",
    }
    comparisons = {}
    for arm in arms:
        repairs = [task_id for task_id in ids if not first_scores[task_id]["exact_match"] and keyed[task_id, arm]["exact_match"]]
        regressions = [task_id for task_id in ids if first_scores[task_id]["exact_match"] and not keyed[task_id, arm]["exact_match"]]
        comparisons[arm] = {
            "correct": sum(keyed[task_id, arm]["exact_match"] for task_id in ids),
            "answer_em": aggregates[arm]["success_rate"], "answer_f1": aggregates[arm]["mean_token_f1"],
            "mean_drafts": aggregates[arm]["mean_rounds"], "judge_calls": aggregates[arm]["judge_calls"],
            "repairs": len(repairs), "regressions": len(regressions),
            "repair_task_ids": repairs, "regression_task_ids": regressions,
            "em_mismatching_approvals": aggregates[arm]["false_approvals"], "approvals": aggregates[arm]["approvals"],
        }
    return {
        "verification": "passed",
        "method": "Artifact consistency audit; round EM/F1 are rechecked with the current scorer only after verifying it matches the saved scorer. No independent gold adjudication or new inference. Checksums establish agreement with the saved manifest, not external authenticity.",
        "checks": ["source and dataset snapshot hashes", "matching scorer and round scores versus saved gold", "prepared dataset hash and task selection", "complete paired episodes", "CSV versus trace fields", "shared first-draft reuse and scoring", "final scores and stopping", "recomputed arm aggregates", "physical model calls and usage"],
        "input_sha256": {name: sha256(directory / name) for name in ("manifest.json", "summary.json", "episodes.csv", "traces.jsonl", "dataset_snapshot.jsonl")},
        "analysis_script_sha256": sha256(Path(__file__)),
        "source_sha256": source_hash.hexdigest(), "task_ids": ids,
        "one_draft_baseline": baseline, "arms": comparisons,
        "limitations": ["Development results are exploratory; no significance or broad superiority claim.",
                        "A repair or regression is defined by answer exact match, not manual factual adjudication.",
                        "A non-matching approved answer can be a wording mismatch; inspect F1 and raw answers.",
                        "Later drafts differ across arms because feedback differs.",
                        "Local typed scores are uncalibrated Qwen outputs, not Jev results."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.run_directory)
        baseline = result["one_draft_baseline"]
        count = baseline["episodes"]
        lines = ["# Paired result check", "", f"All artifact checks passed for {count} HotpotQA questions.", "",
                 "This derives the one-draft baseline from recorded initial answers. It checks scores against saved gold with the matching scorer and audits counts; it does not independently judge the gold answers or run models.", "",
                 "| System | Answer EM | Answer F1 | Mean drafts | Judge calls | Repairs | Regressions | EM-mismatching approvals |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                 f"| One draft | {baseline['correct']}/{count} ({baseline['answer_em']:.1%}) | {baseline['answer_f1']:.1%} | 1.00 | 0 | — | — | — |"]
        for arm, item in result["arms"].items():
            approvals = f"{item['em_mismatching_approvals']}/{item['approvals']}" if item["approvals"] else "—"
            lines.append(f"| {arm} | {item['correct']}/{count} ({item['answer_em']:.1%}) | {item['answer_f1']:.1%} | {item['mean_drafts']:.2f} | {item['judge_calls']} | {item['repairs']} | {item['regressions']} | {approvals} |")
        lines += ["", "Repairs change an initial EM=0 answer to final EM=1; regressions do the reverse. These are paired comparisons against the same saved initial drafts.", "",
                  "Verified snapshot hashes, dataset selection, complete paired episodes, CSV/trace agreement, draft reuse, final scores and stopping, arm aggregates, and physical call counts/usage. Hashes and repaired/regressed task IDs are in `analysis.json`.", ""]
        lines += [f"- {note}" for note in result["limitations"]]
        (args.run_directory / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.run_directory / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Verified {count} questions; saved {args.run_directory / 'analysis.md'}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")


if __name__ == "__main__":
    main()
