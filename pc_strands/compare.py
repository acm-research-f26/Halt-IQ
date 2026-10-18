"""Question-paired analysis of terminal, matched untouched/trained head replays.

No model is loaded. Quality includes every frozen planned question; physical
evaluation work and counterfactual operational prefixes remain separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .data import fingerprint
from .decider import probability_true, sha256_file
from .metrics import evaluate, paired_delta

TERMINAL = {"completed", "completed_with_errors"}
SHARED_CONFIG = ("mode", "dataset_sha256", "max_drafts", "seed", "feedback", "phase",
                 "resource_profile", "trajectory_sha256", "feature_cache_sha256",
                 "feature_manifest_sha256", "backend_source_sha256", "threshold")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value, name: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"invalid {name}")
    return float(value)


def auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """Mann-Whitney AUC with half-credit for tied probabilities."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("nonempty aligned labels/probabilities required")
    if any(type(y) not in (int, bool) or y not in (0, 1) for y in labels) or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("binary labels and finite probabilities required")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    lower_negatives, favorable, position = 0, 0.0, 0
    pairs = sorted(zip(probabilities, labels))
    while position < len(pairs):
        end = position + 1
        while end < len(pairs) and pairs[end][0] == pairs[position][0]:
            end += 1
        positive = sum(label for _, label in pairs[position:end])
        negative = end - position - positive
        favorable += positive * (lower_negatives + negative / 2)
        lower_negatives += negative
        position = end
    return favorable / (positives * negatives)


def first_draft_diagnostics(rows: Mapping[str, dict], planned_ids: Sequence[str]) -> dict:
    labels, probabilities, losses = [], [], []
    approved = correct_approved = clipped_nll = 0
    threshold_values = set()
    for qid in planned_ids:
        first = next((d for d in rows.get(qid, {}).get("decisions", []) if d.get("round") == 1), None)
        if first is None:
            continue
        score = first["score"]
        p = _finite(score.get("p_true"), "first-draft probability")
        if p > 1 or type(first.get("draft_exact_match")) is not bool:
            raise ValueError("invalid first-draft diagnostic label/probability")
        y = int(first["draft_exact_match"])
        if type(first.get("approved")) is not bool:
            raise ValueError("explicit first-draft approval required")
        labels.append(y)
        probabilities.append(p)
        approved += first["approved"]
        correct_approved += first["approved"] and y == 1
        logits, temperature = score.get("logits"), score.get("temperature")
        if logits is not None and temperature is not None:
            if (len(logits) != 2 or not all(type(x) in (int, float) and math.isfinite(x) for x in logits)
                    or _finite(temperature, "temperature") == 0):
                raise ValueError("invalid raw logits/temperature")
            delta = (logits[1] - logits[0]) / temperature
            if abs(probability_true(logits, temperature) - p) > 1e-8:
                raise ValueError("probability differs from raw logits and declared temperature")
            losses.append(max(delta, 0.0) - y * delta + math.log1p(math.exp(-abs(delta))))
            threshold_values.add(float(temperature))
        else:
            clipped_nll += int(p in (0.0, 1.0))
            safe = min(1 - 1e-15, max(1e-15, p))
            losses.append(-math.log(safe if y else 1 - safe))
    count = len(labels)
    return {"planned_n": len(planned_ids), "scored_first_drafts": count,
            "missing_first_decisions": len(planned_ids) - count,
            "probability_coverage_of_planned": count / len(planned_ids),
            "auroc": auroc(labels, probabilities) if count else None,
            "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / count if count else None,
            "nll": sum(losses) / count if count else None,
            "nll_clipped_probability_only_rows": clipped_nll,
            "observed_temperatures": sorted(threshold_values),
            "approved_first_drafts": approved,
            "approval_coverage_of_scored": approved / count if count else None,
            "approval_coverage_of_planned": approved / len(planned_ids),
            "first_draft_approval_precision": correct_approved / approved if approved else None,
            "scope": "Probability metrics use observed first decisions only; absent/overflow decisions are coverage failures, not imputed probabilities."}


def policy_statistics(rows: Mapping[str, dict], ids: Sequence[str]) -> dict:
    available = [rows[qid] for qid in ids if qid in rows]
    complete = [r for r in available if r["status"] == "completed"]
    correct = sum(bool(r["exact_match"]) for r in complete)
    missing = len(ids) - len(available)
    unknown = missing + sum(bool(r.get("usage_unknown")) for r in available)
    def total(key):
        return sum(_finite(r.get(key, 0), key) for r in available)
    tokens = total("writer_input_tokens") + total("writer_output_tokens") + total("decision_input_tokens")
    approvals = [r for r in complete if r.get("stop_reason") == "approved"]
    return {"planned_n": len(ids), "recorded_n": len(available), "completed": len(complete),
            "failed_or_missing": len(ids) - len(complete), "correct": correct,
            "em": correct / len(ids), "token_f1": sum(r["token_f1"] for r in complete) / len(ids),
            "em_completed_only": correct / len(complete) if complete else None,
            "token_f1_completed_only": sum(r["token_f1"] for r in complete) / len(complete) if complete else None,
            "mean_actual_drafts": total("rounds") / len(ids),
            "mean_counterfactual_prefix_model_seconds": total("observed_prefix_model_seconds") / len(ids),
            "mean_raw_reported_tokens": None if unknown else tokens / len(ids),
            "known_raw_tokens_lower_bound": tokens, "episodes_with_unknown_usage": unknown,
            "missing_episode_rows": missing,
            "operational_usage_complete": unknown == 0,
            "counterfactual_hosted_cost_usd": total("cost_usd"),
            "repairs_completed": sum(not r["first_draft_exact_match"] and r["exact_match"] for r in complete),
            "regressions_completed": sum(r["first_draft_exact_match"] and not r["exact_match"] for r in complete),
            "correct_first_draft_lost_including_failures": sum(r["first_draft_exact_match"] and
                (r["status"] != "completed" or not r["exact_match"]) for r in available),
            "approvals": len(approvals), "em_wrong_approvals": sum(not r["exact_match"] for r in approvals),
            "approval_precision": sum(r["exact_match"] for r in approvals) / len(approvals) if approvals else None,
            "approval_coverage": len(approvals) / len(ids),
            "rejections_of_EM_correct_drafts": sum(d["draft_exact_match"] and not d["approved"]
                for r in available for d in r.get("decisions", [])),
            "stop_depths": {str(n): sum(r["rounds"] == n for r in complete) for n in range(1, 11)},
            "first_draft_probabilities": first_draft_diagnostics(rows, ids),
            "measurement_mode": "counterfactual_trajectory_prefix_not_live_latency",
            "usage_caveat": "Known prefix costs are lower bounds when a policy episode/usage record is missing; quality failures remain zero."}


def _journal(path: Path) -> dict:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete request journal")
    pending, completed = set(), set()
    for line in raw.decode("utf-8").splitlines():
        event = json.loads(line)
        key = event["request_id"]
        if event["event"] == "request_started":
            if key in pending or key in completed:
                raise ValueError("duplicate dispatched request")
            pending.add(key)
        elif event["event"] == "request_completed":
            if key not in pending:
                raise ValueError("completion without dispatch")
            pending.remove(key)
            completed.add(key)
        elif event["event"] == "request_error":
            if key not in pending:
                raise ValueError("request error without dispatch")
        else:
            raise ValueError("unknown request event")
    if pending:
        raise ValueError("pending/uncertain requests prevent terminal comparison")
    return {"completed_requests": len(completed), "unreconciled_requests": 0}


def _load_run(directory: Path) -> dict:
    manifest = _read(directory / "manifest.json")
    if manifest.get("status") not in TERMINAL:
        raise ValueError("terminal completed/completed_with_errors result sets required")
    frozen, config = manifest["frozen"], manifest["frozen"]["config"]
    ids = frozen["task_ids"]
    if (not ids or len(ids) != len(set(ids)) or not all(isinstance(x, str) and x for x in ids)
            or config.get("phase") != "confirmation" or config.get("mode") != "head-replay"):
        raise ValueError("frozen unique confirmation IDs and head-replay mode required")
    if any(k not in config for k in SHARED_CONFIG):
        raise ValueError("incomplete shared protocol fingerprints")
    tasks = _read(directory / "tasks-evaluator-only.json")
    if [r["id"] for r in tasks] != ids or fingerprint(tasks) != frozen["dataset_state_sha256"]:
        raise ValueError("task snapshot differs from frozen planned cohort")
    source_files = {p.name: sha256_file(p) for p in (directory / "source_snapshot").glob("*.py")}
    if not source_files or fingerprint(source_files) != frozen["source_sha256"]:
        raise ValueError("frozen evaluator source snapshot mismatch")
    journal = _journal(directory / "requests.jsonl")
    usage = manifest.get("actual_evaluation_usage", {})
    if usage.get("unreconciled_requests") != 0 or usage.get("completed_requests_including_invalid_outputs") != journal["completed_requests"]:
        raise ValueError("physical usage and reconciled request count differ")
    if usage.get("reported_physical_input_tokens") != 0:
        raise ValueError("head replay unexpectedly reports physical encoder tokens")
    by_policy = {}
    for row in _read(directory / "episodes.json"):
        qid, policy = row["task_id"], row["policy"]
        if qid not in ids or row["status"] not in {"completed", "failed"}:
            raise ValueError("foreign episode or nonterminal episode status")
        if qid in by_policy.setdefault(policy, {}):
            raise ValueError("duplicate question-policy result")
        if type(row["exact_match"]) is not bool or not 0 <= _finite(row["token_f1"], "token F1") <= 1:
            raise ValueError("invalid answer metric")
        by_policy[policy][qid] = row
    primary = config["policy_name"]
    if primary not in by_policy:
        raise ValueError("primary policy absent")
    return {"directory": directory, "manifest": manifest, "frozen": frozen, "config": config,
            "ids": ids, "tasks": tasks, "policies": by_policy, "primary": primary}


def compare_runs(untuned_dir: str | Path, tuned_dir: str | Path, trajectory_dir: str | Path,
                 feature_cache: str | Path, *, draws: int = 5000, seed: int = 42,
                 retain_best: bool = False) -> dict:
    if type(draws) is not int or draws < 1:
        raise ValueError("positive bootstrap draw count required")
    paths = [Path(untuned_dir), Path(tuned_dir)]
    trajectory, cache = Path(trajectory_dir), Path(feature_cache)
    tracked = [p / name for p in paths for name in ("manifest.json", "episodes.json", "requests.jsonl", "tasks-evaluator-only.json")]
    tracked += [trajectory / "manifest.json", trajectory / "trajectories.json", trajectory / "tasks-evaluator-only.json",
                cache, Path(str(cache) + ".manifest.json")]
    tracked += [p for run in paths for p in (run / "source_snapshot").glob("*.py")]
    tracked += [p for p in (Path(str(cache) + ".events.jsonl"), Path(str(cache) + ".failures.jsonl")) if p.exists()]
    tracked += [Path(__file__), Path(__file__).with_name("metrics.py")]
    if retain_best:
        tracked.append(Path(__file__).with_name("retention.py"))
    before = {str(p): sha256_file(p) for p in tracked}
    untuned, tuned = (_load_run(p) for p in paths)
    if untuned["ids"] != tuned["ids"]:
        raise ValueError("planned question IDs/order differ")
    ids = untuned["ids"]
    for key in ("source_sha256", "dataset_state_sha256"):
        if untuned["frozen"][key] != tuned["frozen"][key]:
            raise ValueError(f"shared {key} differs")
    for key in SHARED_CONFIG:
        if untuned["config"][key] != tuned["config"][key]:
            raise ValueError(f"shared protocol differs: {key}")
    if untuned["primary"] == tuned["primary"]:
        raise ValueError("untuned/tuned policy names must be distinct")
    config = untuned["config"]
    generation = _read(trajectory / "manifest.json")
    if generation.get("status") not in TERMINAL or generation.get("writer_unload_verified") is not True:
        raise ValueError("terminal unloaded writer generation required")
    if config["trajectory_sha256"] != sha256_file(trajectory / "trajectories.json"):
        raise ValueError("shared trajectory hash does not match actual source")
    features = _read(Path(str(cache) + ".manifest.json"))
    if (not features.get("status", "").startswith("completed") or features.get("unreconciled_inferences") != 0
            or features.get("cached_rows_sha256") != sha256_file(cache)
            or config["feature_cache_sha256"] != sha256_file(cache)
            or config["feature_manifest_sha256"] != sha256_file(Path(str(cache) + ".manifest.json"))
            or features["frozen"]["trajectories_sha256"] != config["trajectory_sha256"]
            or features["frozen"]["backend_sha256"] != config["backend_source_sha256"]
            or features["frozen"]["tasks_sha256"] != sha256_file(trajectory / "tasks-evaluator-only.json")):
        raise ValueError("shared real encoder feature provenance mismatch")
    for suffix, key in ((".events.jsonl", "events_sha256"), (".failures.jsonl", "failures_sha256")):
        path = Path(str(cache) + suffix)
        if features.get(key) != (sha256_file(path) if path.exists() else None):
            raise ValueError("feature journal/failure checksum mismatch")
    original_tasks = {r["id"]: r for r in _read(trajectory / "tasks-evaluator-only.json")}
    if any(original_tasks.get(r["id"]) != r for r in untuned["tasks"]):
        raise ValueError("source trajectory question/evidence/reference snapshot differs")
    trajectories = _read(trajectory / "trajectories.json")
    source = {r["task_id"]: r for r in trajectories}
    if len(source) != len(trajectories) or any(qid not in source for qid in ids):
        raise ValueError("source trajectory IDs missing/duplicated")
    for qid in ids:
        rounds = source[qid]["rounds"]
        if len(rounds) > 10 or [r["round"] for r in rounds] != list(range(1, len(rounds) + 1)):
            raise ValueError("source rounds must be a contiguous bounded prefix")
        for row in rounds:
            if row.get("status") == "completed":
                gold = evaluate(row["draft"], original_tasks[qid]["answers"])
                if any(row[key] != value for key, value in gold.items()):
                    raise ValueError("source round scores differ from original references")
    combined, deduplicated = {}, []
    for run in (untuned, tuned):
        for name, rows in run["policies"].items():
            if name in combined:
                if not name.startswith("fixed-") or fingerprint(combined[name]) != fingerprint(rows):
                    raise ValueError("duplicated policy is not an identical shared fixed control")
                deduplicated.append(name)
            else:
                combined[name] = rows
            for qid, row in rows.items():
                first = source[qid]["rounds"]
                if row["first_draft_exact_match"] != bool(first and first[0].get("exact_match")):
                    raise ValueError("episode first-draft label differs from shared source")
                if row["status"] == "completed":
                    number = row["rounds"]
                    if type(number) is not int or not 1 <= number <= min(config["max_drafts"], len(first)):
                        raise ValueError("completed selected round outside shared source")
                    if row["final_answer"] != first[number - 1].get("draft"):
                        raise ValueError("policy answer is not its selected shared draft")
                    gold = evaluate(row["final_answer"], original_tasks[qid]["answers"])
                    if any(row[key] != value for key, value in gold.items()):
                        raise ValueError("episode scores differ from frozen references")
                for decision in row.get("decisions", []):
                    number = decision["round"]
                    if type(number) is not int or not 1 <= number <= row["rounds"]:
                        raise ValueError("decision sees a future/unvisited draft")
                    if (decision["draft_exact_match"] != first[number - 1]["exact_match"]
                            or decision["approved"] != (decision["score"]["p_true"] >= config["threshold"])):
                        raise ValueError("decision labels or approval rule differ from frozen protocol")
    secondary = []
    if retain_best:
        from .retention import derive_retained_episode
        for run in (untuned, tuned):
            for name in (run["primary"], run["primary"] + "-cap3"):
                if name not in combined:
                    continue
                derived = {qid: derive_retained_episode(row, source[qid], original_tasks[qid]["answers"])
                           for qid, row in combined[name].items()}
                combined[name + "-retainbest"] = derived
                secondary.append(name + "-retainbest")
    statistics = {name: policy_statistics(rows, ids) for name, rows in combined.items()}
    categories = sorted({original_tasks[qid].get("category", "unknown") for qid in ids})
    for name, rows in combined.items():
        statistics[name]["categories"] = {}
        for category in categories:
            group = [qid for qid in ids if original_tasks[qid].get("category", "unknown") == category]
            completed = [rows[qid] for qid in group if qid in rows and rows[qid]["status"] == "completed"]
            statistics[name]["categories"][category] = {"planned_n": len(group), "completed": len(completed),
                "em": sum(r["exact_match"] for r in completed) / len(group),
                "token_f1": sum(r["token_f1"] for r in completed) / len(group)}
    contrasts = [(tuned["primary"], untuned["primary"])]
    if tuned["primary"] + "-cap3" in combined and untuned["primary"] + "-cap3" in combined:
        contrasts.append((tuned["primary"] + "-cap3", untuned["primary"] + "-cap3"))
    contrasts += [(name, fixed) for name in (untuned["primary"], tuned["primary"])
                  for fixed in sorted(combined) if fixed.startswith("fixed-")]
    if retain_best:
        for ending in ("", "-cap3"):
            candidate, baseline = tuned["primary"] + ending + "-retainbest", untuned["primary"] + ending + "-retainbest"
            if candidate in combined and baseline in combined:
                contrasts.append((candidate, baseline))
        contrasts += [(name, name.removesuffix("-retainbest")) for name in secondary]
    comparisons = {candidate + "_minus_" + baseline: {metric: paired_delta(combined[candidate], combined[baseline], ids,
        metric, draws=draws, seed=seed) for metric in ("exact_match", "token_f1")} for candidate, baseline in contrasts}
    after = {str(p): sha256_file(p) for p in tracked}
    if before != after:
        raise ValueError("inputs changed while comparing; no stable result produced")
    return {"status": "independently_joined_terminal_paired_replays", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "planned_n": len(ids), "planned_question_ids": ids, "shared_protocol": {k: config[k] for k in SHARED_CONFIG},
            "source_sha256": untuned["frozen"]["source_sha256"], "input_sha256": before,
            "model_fingerprints": {r["primary"]: r["config"].get("model_fingerprint") for r in (untuned, tuned)},
            "deduplicated_identical_fixed_controls": sorted(set(deduplicated)),
            "secondary_retain_best_policies": secondary,
            "policies": statistics, "paired_comparisons": comparisons,
            "bootstrap": {"draws": draws, "seed": seed, "unit": "original question", "rng": "python.random.Random",
                          "method": "paired percentile bootstrap; failed/missing quality=0 against frozen planned N"},
            "physical_evaluation_usage": {"writer_generation_once": generation.get("actual_evaluation_usage"),
                "analysis_planned_questions": len(ids), "source_generation_questions": len(trajectories),
                "source_feature_selected_questions": features.get("selected_questions"),
                "real_encoder_features_once": features.get("physical_encoding_usage"),
                "untuned_head_scoring": untuned["manifest"]["actual_evaluation_usage"],
                "tuned_head_scoring": tuned["manifest"]["actual_evaluation_usage"],
                "note": "Shared writer/features counted once. Head-scoring phases add zero physical encoder tokens; prefix estimates are not extra physical calls."},
            "temperatures": {"untuned_override": untuned["config"].get("temperature"),
                "tuned_override": tuned["config"].get("temperature"),
                "released_encoder_metadata": features.get("encoder", {}).get("temperature"),
                "published_released_noul": 1.0704394404346953,
                "published_released_global": 1.0545210242263665,
                "published_config_revision": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
                "local_fit_predeclared_bounds": [.25, 6.0],
                "note": "Actual used first-decision temperatures appear per policy; metadata is not a new calibration fit."},
            "limitations": ["Frozen generic trajectories estimate stopping decisions; no live end-to-end latency or operational cost saving is measured.",
                "Local hardware/electricity dollar cost is unpriced; raw model tokenizers need not measure equal compute.",
                "Probability diagnostics cover observed first decisions only; report failures/coverage alongside them.",
                "Question-paired intervals are descriptive; training proof and better train reward do not establish held-out improvement."]}


def write_report(result: dict, output: str | Path) -> None:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "paired-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["HaltIQ terminal paired head replay comparison", f"Planned questions: {result['planned_n']}",
             "Quality uses frozen planned N, with failed/missing episodes zero-scored.",
             "Policy | Completed | EM | F1 | Mean drafts | Raw tokens/q | Counterfactual model s/q"]
    for name, row in result["policies"].items():
        tokens = "unknown" if row["mean_raw_reported_tokens"] is None else f"{row['mean_raw_reported_tokens']:.2f}"
        lines.append(f"{name} | {row['completed']}/{row['planned_n']} | {100*row['em']:.2f}% | {row['token_f1']:.6f} | {row['mean_actual_drafts']:.3f} | {tokens} | {row['mean_counterfactual_prefix_model_seconds']:.6f}")
    lines.append("\nPaired candidate-minus-baseline quality in percentage points;95%question-bootstrap intervals:")
    for name, values in result["paired_comparisons"].items():
        text = []
        for metric, row in values.items():
            low, high = row["bootstrap95"]
            text.append(f"{metric} {100*row['candidate_minus_baseline']:+.3f}[{100*low:+.3f},{100*high:+.3f}]")
        lines.append(name + ": " + "; ".join(text))
    lines.extend(["", *result["limitations"], "Physical shared generation/encoding and head replay costs are separate in paired-analysis.json."])
    (directory / "PAIRED-RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("untuned", "tuned", "trajectory-dir", "feature-cache", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--retain-best", action="store_true", help="Add predeclared secondary visited-confidence selectors; primary contrast unchanged")
    args = parser.parse_args()
    result = compare_runs(args.untuned, args.tuned, args.trajectory_dir, args.feature_cache,
                          retain_best=args.retain_best)
    write_report(result, args.output)
    print(json.dumps({"status": result["status"], "planned_n": result["planned_n"], "output": args.output}))


if __name__ == "__main__":
    main()
