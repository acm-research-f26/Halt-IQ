"""Paired initial drafts, bounded feedback loops, and inspectable result artifacts."""

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
from statistics import mean
import time

from . import __version__
from .critics import RUBRIC_VERSION, typed_review
from .providers import CallResult


def json_write(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summarize(episodes: list[dict]) -> dict:
    grouped = defaultdict(list)
    for ep in episodes:
        grouped[ep["arm"]].append(ep)
    result = {}
    for arm, items in grouped.items():
        approvals = [ep for ep in items if ep["stop_reason"] == "critic_approved"]
        completed = [ep for ep in items if ep["status"] == "completed"]
        judged = [r for ep in items for r in ep["round_trace"] if "approved" in r]
        result[arm] = {
            "episodes": len(items),
            "completed": sum(ep["status"] == "completed" for ep in items),
            "failures": sum(ep["status"] != "completed" for ep in items),
            "success_rate": mean(ep["success"] for ep in items),
            "completion_rate": len(completed) / len(items),
            "completed_accuracy": mean(ep["exact_match"] for ep in completed) if completed else None,
            "mean_token_f1": mean(ep["token_f1"] if ep["status"] == "completed" else 0.0 for ep in items),
            "mean_rounds": mean(ep["rounds"] for ep in items),
            "mean_revisions": mean(max(0, ep["rounds"] - 1) for ep in items),
            "mean_writer_calls": mean(ep["writer_calls"] for ep in items),
            "mean_judge_calls": mean(ep["judge_calls"] for ep in items),
            "judge_calls": sum(ep["judge_calls"] for ep in items),
            "typed_questions": sum(ep["typed_questions"] for ep in items),
            "mean_judge_latency_ms": (sum(ep["judge_latency_ms"] for ep in items) /
                                      sum(ep["judge_successful_calls"] for ep in items)
                                      if sum(ep["judge_successful_calls"] for ep in items) else None),
            "operational_input_tokens": sum(ep["input_tokens"] for ep in items),
            "operational_output_tokens": sum(ep["output_tokens"] for ep in items),
            "api_cost_usd": sum(ep["cost_usd"] for ep in items),
            "approvals": len(approvals),
            "false_approvals": sum(not ep["exact_match"] for ep in approvals),
            "false_approval_rate": (mean(not ep["exact_match"] for ep in approvals) if approvals else None),
            "false_approvals_per_episode": sum(not ep["exact_match"] for ep in approvals) / len(items),
            "judgments": len(judged),
            "correct_draft_rejections": sum(r["exact_match"] and not r["approved"] for r in judged),
            "round_cap_episodes": sum(ep["stop_reason"] == "round_cap" for ep in items),
        }
    return result


def write_reports(output: Path, episodes: list[dict], manifest: dict, actual_calls: list[CallResult]):
    aggregates = summarize(episodes)
    summary = {
        "run_kind": manifest["run_kind"], "status": manifest["status"],
        "benchmarks": manifest.get("benchmarks", ["diagnostic"]),
        "metrics": manifest.get("metrics", ["diagnostic"]),
        "arms": aggregates,
        "actual_experiment": {
            "successful_model_calls": len(actual_calls),
            "input_tokens": sum(c.input_tokens for c in actual_calls),
            "output_tokens": sum(c.output_tokens for c in actual_calls),
            "api_cost_usd": sum(c.cost_usd for c in actual_calls),
            "note": "Provider-validated responses, including application-level validation failures. Shared first drafts are counted once here, once per arm in operational metrics. Unvalidated/network failures may have unreported token usage and retain budget reservations; see manifest budget.",
        },
    }
    json_write(output / "summary.json", summary)
    json_write(output / "manifest.json", manifest)
    if episodes:
        fields = [k for k in episodes[0] if k != "round_trace"]
        with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: ep[k] for k in fields} for ep in episodes)
    lines = ["# HaltIQ experiment", "", f"Run kind: **{manifest['run_kind']}**. Status: {manifest['status']}.", ""]
    if manifest["run_kind"] == "scripted_demo":
        lines += ["**SCRIPTED SOFTWARE DEMO. No Jev or LLM inference occurred. These numbers validate control flow only.**", ""]
    elif "hotpotqa-distractor" in manifest.get("benchmarks", []):
        lines += ["**HotpotQA distractor, answer-only evaluation.** Complete supplied evidence is given to every writer and critic; gold answers and supporting-fact labels are evaluator-only.", "",
                  "These are local subsets of the public official development split. The local `test` label means held-out validation, not the hidden official test set. "
                  "No supporting-fact/joint leaderboard metrics or statistical superiority are claimed.", ""]
    else:
        lines += ["Small synthetic diagnostic run. These results do not establish Jev superiority or general benchmark accuracy.", ""]
    lines += ["| Arm | Completed / attempted | Answer EM | Answer F1 | Mean rounds | Judge calls | False approvals / approvals | API USD |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, a in aggregates.items():
        lines.append(f"| {name} | {a['completed']} / {a['episodes']} | {a['success_rate']:.1%} | {a['mean_token_f1']:.1%} | {a['mean_rounds']:.2f} | {a['judge_calls']} | {a['false_approvals']} / {a['approvals']} | {a['api_cost_usd']:.6f} |")
    lines += ["", "Success is normalized exact match against the dataset reference, with failed episodes counted as unsuccessful. "
              "Critic approval is a stopping decision, not the success label. Token F1 is also saved.", "",
              "Each round is one writer draft. Judge calls count attempted critic requests; each typed request contains three questions. "
              "The last round is reviewed as well. The fixed arm uses self-revision and no critic.", "",
              "Every arm starts from the same draft; later drafts differ because feedback differs. This is a closed-loop system comparison, "
              "not a comparison of critics on identical full trajectories. Per-task arm order rotates. Latency includes loading and local contention; no latency significance claim is made.", "",
              "First-draft operational tokens are charged to every arm for a standalone cost comparison. "
              "Actual experiment tokens count that shared request once. Costs exclude local electricity, taxes, prepaid credit purchases, and other programs.", "",
              "See `manifest.json` for configuration/provenance, `traces.jsonl` for raw outputs and decisions, "
              "`episodes.csv` for paired per-task results, and `summary.json` for all aggregates."]
    if "hotpotqa-distractor" in manifest.get("benchmarks", []):
        lines += ["", "HotpotQA answer EM/F1 follow the official scorer's normalization and yes/no/noanswer rules. "
                  "This differs from the synthetic diagnostic normalizer. See `metrics.py` in `source_snapshot/`.", "",
                  "Benchmark passages, labels, and their derivatives are from [HotpotQA](https://hotpotqa.github.io/), "
                  "Yang et al. (EMNLP 2018), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). "
                  "Preparation details and hashes are in `dataset_manifest.json` when a preparation manifest is available."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(tasks, writer, critics: dict, output: Path, config: dict, budget=None, demo=False, progress=print) -> dict:
    max_rounds, threshold = config["max_rounds"], config["threshold"]
    if not 1 <= max_rounds <= 10:
        raise ValueError("max_rounds must be between 1 and 10")
    if not critics:
        raise ValueError("At least one arm is required")
    if demo and any(len(t.demo_drafts) < max_rounds or len(t.demo_scores) < max_rounds for t in tasks):
        raise ValueError("Demo requires one scripted draft and score set per round")
    dataset_path = Path(config["dataset"])
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset_manifest = None
    sidecar = dataset_path.parent / "manifest.json"
    if sidecar.is_file():
        possible_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if possible_manifest.get("benchmark") == "hotpotqa-distractor":
            if possible_manifest.get("output", {}).get("sha256") != dataset_hash:
                raise ValueError("Dataset checksum does not match its preparation manifest; prepare a new subset instead of editing it in place")
            dataset_manifest = possible_manifest
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256()
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
        shutil.copyfile(path, snapshot / path.name)
    shutil.copyfile(config["dataset"], output / "dataset_snapshot.jsonl")
    if dataset_manifest is not None:
        json_write(output / "dataset_manifest.json", dataset_manifest)
        for filename in ("CITATION.bib", "README.md"):
            source_file = dataset_path.parent / filename
            if source_file.is_file():
                shutil.copyfile(source_file, output / ("dataset_" + filename))
    manifest = {
        "version": __version__, "run_kind": "scripted_demo" if demo else "live_models",
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "python": platform.python_version(), "platform": platform.platform(),
        "source_sha256": source_hash.hexdigest(), "rubric_version": RUBRIC_VERSION,
        "benchmarks": sorted({task.benchmark for task in tasks}),
        "metrics": sorted({task.metric for task in tasks}),
        "evaluation_scope": "answer-only",
        "config": config, "task_ids": [t.id for t in tasks],
        "dataset_sha256": dataset_hash,
    }
    json_write(output / "manifest.json", manifest)
    episodes, actual_calls = [], []

    with (output / "traces.jsonl").open("w", encoding="utf-8") as trace:
        def event(data):
            trace.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()

        def demo_call():
            return CallResult(data={"scripted": True}, input_tokens=0, output_tokens=0,
                              latency_ms=0.0, cost_usd=0.0, model="scripted-fixture")

        def meter(ep, call, role, physical=True):
            if physical and not demo:
                actual_calls.append(call)
            for key in ("input_tokens", "output_tokens"):
                ep[key] += getattr(call, key)
            ep["cost_usd"] += call.cost_usd
            if role == "critic":
                ep["judge_successful_calls"] += 1
                ep["judge_latency_ms"] += call.latency_ms

        try:
            for task_index, task in enumerate(tasks):
                progress(f"[{task_index + 1}/{len(tasks)}] {task.id}: shared initial draft", flush=True)
                first, first_call, first_error = "", None, None
                try:
                    event({"event": "request", "task_id": task.id, "role": "shared_writer"})
                    first, first_call = ((task.demo_drafts[0], demo_call()) if demo else writer.write(task.state()))
                    if not demo:
                        actual_calls.append(first_call)
                    event({"event": "shared_writer", "task_id": task.id, "draft": first, "call": asdict(first_call)})
                except Exception as exc:
                    first_call = getattr(exc, "call", None)
                    if first_call is not None and not demo:
                        actual_calls.append(first_call)
                    first_error = f"{type(exc).__name__}: {exc}"
                    event({"event": "error", "task_id": task.id, "role": "shared_writer", "error": first_error,
                           "call": asdict(first_call) if first_call else None})

                arms = list(critics)
                offset = task_index % len(arms)
                for arm in arms[offset:] + arms[:offset]:
                    started = time.perf_counter()
                    ep = {"task_id": task.id, "split": task.split, "category": task.category,
                          "benchmark": task.benchmark, "metric": task.metric,
                          "source_id": task.metadata.get("source_id", task.id),
                          "arm": arm, "status": "completed", "stop_reason": "round_cap", "rounds": 0,
                          "writer_calls": 1, "judge_calls": 0, "judge_successful_calls": 0, "typed_questions": 0,
                          "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "judge_latency_ms": 0.0,
                          "elapsed_ms": 0.0, "final_answer": first, "exact_match": False, "success": False,
                          "token_f1": 0.0, "error": "", "round_trace": []}
                    draft, feedback = first, None
                    if first_call:
                        meter(ep, first_call, "writer", physical=False)
                    active_role = "writer"
                    try:
                        if first_error:
                            raise RuntimeError(first_error)
                        for round_number in range(1, max_rounds + 1):
                            if round_number > 1:
                                active_role = "writer"
                                ep["writer_calls"] += 1
                                event({"event": "request", "task_id": task.id, "arm": arm, "round": round_number, "role": "writer"})
                                draft, call = ((task.demo_drafts[round_number - 1], demo_call()) if demo else
                                               writer.write(task.state(), draft, feedback))
                                meter(ep, call, "writer")
                                event({"event": "writer", "task_id": task.id, "arm": arm,
                                       "round": round_number, "draft": draft, "call": asdict(call)})
                            ep["rounds"] = round_number
                            ep["final_answer"] = draft
                            observation = {"round": round_number, "draft": draft, **task.evaluate(draft)}
                            if arm == "fixed":
                                feedback = "Recheck your answer against all the evidence and constraints; correct any errors. Keep it if correct."
                            else:
                                active_role = "critic"
                                ep["judge_calls"] += 1
                                if arm in {"jev", "openjev", "local-typed", "mock-typed"}:
                                    ep["typed_questions"] += 3
                                event({"event": "request", "task_id": task.id, "arm": arm, "round": round_number, "role": "critic"})
                                review = (typed_review(task.demo_scores[round_number - 1], threshold, demo_call()) if demo else
                                          critics[arm].review(task.state(draft)))
                                meter(ep, review.call, "critic")
                                observation.update(approved=review.approved, feedback=review.feedback, scores=review.scores)
                                event({"event": "critic", "task_id": task.id, "arm": arm,
                                       "round": round_number, "review": asdict(review)})
                                feedback = review.feedback
                                if review.approved:
                                    ep["stop_reason"] = "critic_approved"
                            ep["round_trace"].append(observation)
                            if ep["stop_reason"] == "critic_approved":
                                break
                    except Exception as exc:
                        failed_call = getattr(exc, "call", None)
                        if failed_call is not None:
                            meter(ep, failed_call, active_role)
                            event({"event": "invalid_output", "task_id": task.id, "arm": arm,
                                   "role": active_role, "call": asdict(failed_call)})
                        ep.update(status="failed", stop_reason="provider_error", error=f"{type(exc).__name__}: {exc}")
                    ep.update(task.evaluate(ep["final_answer"]))
                    ep["success"] = ep["status"] == "completed" and ep["exact_match"]
                    ep["elapsed_ms"] = (time.perf_counter() - started) * 1000 + (first_call.latency_ms if first_call else 0)
                    episodes.append(ep)
                    event({"event": "episode", **ep})
                    progress(f"  {arm}: {ep['stop_reason']}, {ep['rounds']} rounds, exact={ep['exact_match']}", flush=True)
                    if budget:
                        manifest["budget"] = budget.snapshot()
                    write_reports(output, episodes, manifest, actual_calls)
            manifest["status"] = "completed_with_errors" if any(ep["status"] != "completed" for ep in episodes) else "completed"
        except BaseException:
            manifest["status"] = "interrupted"
            raise
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            if budget:
                manifest["budget"] = budget.snapshot()
            write_reports(output, episodes, manifest, actual_calls)
    return {"manifest": manifest, "episodes": episodes, "summary": summarize(episodes)}
