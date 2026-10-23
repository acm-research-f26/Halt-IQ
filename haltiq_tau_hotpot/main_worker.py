"""Finite main-study phases, executable only from a separately adopted snapshot.

Authoring/importing this worker never adopts or starts the fourteen-hour study.
Actual local provider ownership and per-call accounting remain in LocalRuntime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from pc_strands.control import GpuLease, RunControl
from pc_strands.metrics import paired_delta
from .journal import CallJournal, durable_write, fingerprint
from .protocol import ARMS, MAIN_SETTINGS, PHASES, execution_cutoff, load_main, pinned, read, verify_dispatch
from .worker import bare_case, completion, hydrate, initial_drafts


def preparation_cases(context):
    cohorts = context["cohorts"]
    return [case for key in ("main_development", "main_fit", "main_tune", "main_calibration")
            for case in cohorts[key]]


def collect(args, context, control):
    # Reuse the exact shared first-draft collection implementation; it does not
    # perform qualification-plan validation or select/replace any questions.
    from .worker import collect as collect_first_drafts
    c = context["cohorts"]
    return collect_first_drafts(args, context["config"], context["freeze"], {
        "pilot_heldout": c["main_development"], "pilot_fit": c["main_fit"],
        "pilot_tune": c["main_tune"], "pilot_calibration": c["main_calibration"]}, control)


def _runtime_terminal(args, runtime, control):
    durable_write(args.output / "runtime-terminal.json", {"cleanup": runtime.cleanup,
        "calls": runtime.journal.verify(), "timings": runtime.timings,
        "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})


def features(args, context, control):
    from .runtime import LocalRuntime
    from .scoring import score
    drafts = initial_drafts(args.previous / "collect")
    rows, baseline = [], []
    runtime = LocalRuntime(context["config"], args.output, control)
    try:
        with runtime:
            for case in preparation_cases(context):
                qid = case["public"]["id"]
                draft = drafts[qid]["episode"]["submitted"]
                for criterion in MAIN_SETTINGS["criteria"]:
                    row = {"question_id": qid, "partition": case["partition"], "criterion": criterion}
                    if draft is None:
                        row.update(status="failed", failure="no_initial_submission", label=0)
                    else:
                        row["label"] = int(score(draft, case["evaluator"])["em" if criterion == "answer" else "sp_em"])
                        try:
                            row.update(status="completed", observed=runtime.observe(case["public"], draft, criterion=criterion))
                        except ValueError as error:
                            row.update(status="failed", failure=type(error).__name__, message=str(error))
                    durable_write(args.output / "rows" / (qid + "-" + criterion + ".json"), row)
                    rows.append(row)
                    if case["partition"] == "development" and criterion == "answer":
                        baseline.append(row)
    finally:
        _runtime_terminal(args, runtime, control)
    durable_write(args.output / "features.json", rows)
    config = context["config"]
    proof = {"phase": "untouched_pretraining", "terminal": True,
        "pending_or_uncertain": runtime.journal.summary()["pending_or_uncertain"],
        "cleanup_verified": runtime.cleanup["verified"],
        "checkpoint_head_sha256": _digest(Path(config["checkpoint"]) / "head.safetensors"),
        "valid_native_forwards": sum(r["status"] == "completed" and r["observed"]["physical_forwards"] == 1 for r in baseline),
        "planned_question_ids": [c["public"]["id"] for c in context["cohorts"]["main_development"]],
        "completed": sum(r["status"] == "completed" for r in baseline),
        "failed": sum(r["status"] != "completed" for r in baseline),
        "rows_sha256": fingerprint(baseline), "finished_at": datetime.now(timezone.utc).isoformat()}
    durable_write(args.output / "untouched-baseline.json", proof)
    return completion(args, planned=1240, completed=sum(r["status"] == "completed" for r in rows),
        failed=sum(r["status"] != "completed" for r in rows), cleanup=runtime.cleanup["verified"],
        pending=runtime.journal.summary()["pending_or_uncertain"])


def _digest(path):
    from .hobson import file_hash
    return file_hash(path)


def verify_feature_rows(args, context, rows, baseline):
    """Join each planned state with its actual native result and external label."""
    from .hobson import stable_hash
    from .scoring import score
    journal = CallJournal(args.previous / "features/physical")
    if journal.verify()["pending_or_uncertain"]:
        raise ValueError("Native source journal is unresolved")
    responses = {fingerprint(read(journal.directory / event["response"]))
        for event in journal.records if event["event"] == "completed" and event["role"] in {"native_answer", "native_evidence"}}
    cases = {c["public"]["id"]: c for c in preparation_cases(context)}
    expected = {(qid, criterion) for qid in cases for criterion in MAIN_SETTINGS["criteria"]}
    seen = set()
    drafts = initial_drafts(args.previous / "collect")
    for row in rows:
        key = row["question_id"], row["criterion"]
        if (key not in expected or key in seen or row["partition"] != cases[key[0]]["partition"]
                or row["status"] not in {"completed", "failed"}):
            raise ValueError("Foreign, duplicate or repurposed main feature row")
        seen.add(key)
        if row["status"] != "completed":
            continue
        case, observed = cases[key[0]], row["observed"]
        public, draft = case["public"], drafts[key[0]]["episode"]["submitted"]
        evidence = "\n".join(public["context"][i][0] + ": " + " ".join(public["context"][i][1])
                             for i in observed["retrieval_indices"])
        state = f"Question: {public['question']}\nEvidence: {evidence}\nDraft answer: {draft['answer']}"
        if key[1] == "evidence":
            state += "\nCitations: " + json.dumps(draft["supporting_facts"], ensure_ascii=False)
        feature = observed["feature"]
        metric = "em" if key[1] == "answer" else "sp_em"
        if (fingerprint(observed) not in responses or observed["physical_forwards"] != 1
                or feature["provenance"]["state_sha256"] != stable_hash(state)
                or feature["criterion"] != key[1] or row["label"] != int(score(draft, case["evaluator"])[metric])
                or _digest(feature["path"]) != feature["sha256"]):
            raise ValueError("Main native feature/physical response/public input/label join failed")
    if seen != expected or baseline["rows_sha256"] != fingerprint([
            r for r in rows if r["partition"] == "development" and r["criterion"] == "answer"]):
        raise ValueError("Every planned state and the actual untouched baseline must remain bound")


def valid_learning_rows(rows, criterion, partition, minimum):
    planned = [r for r in rows if r["criterion"] == criterion and r["partition"] == partition]
    valid = [r for r in planned if r["status"] == "completed"]
    if len({r["question_id"] for r in valid}) < minimum:
        raise ValueError("Frozen technical-validity minima not met; never replace or quality-filter questions")
    return valid, {"planned": len(planned), "valid": len(valid), "failed": len(planned) - len(valid)}


def train(args, context, control):
    config = context["config"]
    rows = read(args.previous / "features/features.json")
    baseline = read(args.previous / "features/untouched-baseline.json")
    verify_feature_rows(args, context, rows, baseline)
    minima = MAIN_SETTINGS["minimum_valid"]
    source, counts = {}, {}
    for partition, minimum in minima.items():
        source[partition], counts[partition] = valid_learning_rows(rows, "answer", partition, minimum)
    evidence, evidence_count = valid_learning_rows(rows, "evidence", "calibration",
        MAIN_SETTINGS["minimum_evidence_calibration"])
    with GpuLease(Path(config["gpu_lease"]), "CPU-only main Hobson head learning and calibration"):
        from .learning import train_heads, validate_baseline, fit_temperature
        forbidden = [r["question_id"] for partition in source.values() for r in partition]
        validate_baseline(baseline, _digest(Path(config["checkpoint"]) / "head.safetensors"),
                          forbidden, minimum_questions=20)
        material = {name: [hydrate(r) for r in part] for name, part in source.items()}
        journal = CallJournal(args.output / "physical")
        result = journal.call("head_training_and_calibration", {
            "features_sha256": _digest(args.previous / "features/features.json"),
            "baseline_sha256": _digest(args.previous / "features/untouched-baseline.json")},
            lambda: train_heads(config["checkpoint"], config["base"], material["fit"], material["tune"],
                material["calibration"], args.output / "learned", baseline, control=control, minimum_questions=minima))
        # These raw released-head logits came from the already journaled native
        # evidence CAL observations. Scalar fitting performs no new model forward.
        control.before_work(require_gpu=False)
        started = time.monotonic()
        fitted = fit_temperature([r["observed"]["raw_logits_false_true"] for r in evidence], evidence)
        result["released_evidence_calibration"] = {**fitted, **evidence_count,
            "source_observation_sha256": fingerprint(evidence), "weights_updated": False,
            "physical_new_head_forwards": 0, "physical_new_encoder_forwards": 0,
            "label_counts": {str(y): sum(r["label"] == y for r in evidence) for y in (0, 1)},
            "warnings": ["one-class evidence CAL retained; AUROC undefined"] if len({r["label"] for r in evidence}) < 2 else [],
            "goal": "strict supporting-facts EM proxy; not a separate entailment judgment"}
        control.rest_after_block(time.monotonic() - started)
    result["planned_answer_coverage"] = counts
    durable_write(args.output / "learning.json", result)
    return completion(args, planned=1, completed=1, failed=0, cleanup=True,
                      pending=journal.summary()["pending_or_uncertain"])


def analyse(episodes, context, initial):
    from .comparison import analyse as common_analysis, diagnostics
    from .scoring import score
    cases = context["cohorts"]["main_fresh"]
    result = common_analysis(episodes, cases, ARMS, initial_drafts=initial)
    ids, by_arm = [c["public"]["id"] for c in cases], {arm: {} for arm in ARMS}
    for row in episodes:
        by_arm[row["arm"]][row["question_id"]] = {**row, **row["metrics"]}
    contrasts = [(a, "fixed1") for a in ARMS if a != "fixed1"] + [
        ("released", "fixed10"), ("decomposed", "released"), ("trained", "released")]
    result["paired"] = {a + "_minus_" + b: {metric: paired_delta(by_arm[a], by_arm[b], ids,
        metric, draws=5000, seed=42) for metric in ("em", "f1", "sp_em", "sp_f1", "joint_em", "joint_f1")}
        for a, b in contrasts}
    gold = {c["public"]["id"]: c["evaluator"] for c in cases}
    observations = []
    for row in episodes:
        if row["arm"] != "decomposed":
            continue
        gate = next((e for e in row["trace"] if e["event"] == "gate" and e.get("criterion") == "evidence"
                     and e.get("status") == "completed"), None)
        if gate is not None:
            observations.append({"label": int(score(initial[row["question_id"]], gold[row["question_id"]])["sp_em"]),
                                 "score": gate["score"]})
    result.update(scope="heldout_custom_HotpotQA_in_Tau2_main_study", fresh_n_freeze=context["fresh_freeze"],
        evidence_first_draft_decision=diagnostics(observations, len(ids)),
        submission_routes={arm: {"planned_n": len(ids),
            "submitted": sum(row["status"] == "completed" and row["submitted"] is not None for row in by_arm[arm].values()),
            "submission_coverage": sum(row["status"] == "completed" and row["submitted"] is not None for row in by_arm[arm].values()) / len(ids),
            "budget_fallback_submissions": sum(row.get("budget_fallback_used") is True for row in by_arm[arm].values()),
            "gate_approved_submissions": sum(row.get("gate_approved") is True for row in by_arm[arm].values()) if arm not in ARMS[:5] else None,
            "gate_approval_coverage_of_planned": sum(row.get("gate_approved") is True for row in by_arm[arm].values()) / len(ids) if arm not in ARMS[:5] else None}
            for arm in ARMS},
        policy_definitions={"released": "calibrated released answer gate", "trained": "calibrated trained answer gate only",
                            "decomposed": "separately calibrated released answer AND released evidence gates"},
        interpretation="All planned questions/failures retained. Budget fallback submits the last valid draft without claiming gate approval. Paired evidence may be positive, negative or inconclusive; tied answer scores are possible.")
    return result


def compare(args, context, control):
    """Run one owned child per question; never retain model imports in parent."""
    import subprocess
    from .main_question import await_process_binding, exited_process_binding, question_directory, stamp
    from .hobson import file_hash
    from haltiq_supervisor.engine import check_handoff
    config, rows, initial, receipts, terminals = context["config"], [], {}, [], []
    if config.get("compare_process_layout") != "per_question":
        raise ValueError("Main compare requires the predeclared per-question process layout")
    plan, state = read(args.previous / "plan.json"), read(args.previous / "supervisor.json")
    for index, case in enumerate(context["cohorts"]["main_fresh"]):
        control._check_stop()
        qid = case["public"]["id"]
        directory = question_directory(args.output, index)
        directory.mkdir(parents=True, exist_ok=False)
        command = [plan["python"], "-B", "-m", "haltiq_tau_hotpot.main_question",
            "--config", str(args.config), "--output", str(args.output), "--previous", str(args.previous),
            "--control", str(args.control), "--deadline", args.deadline, "--question-index", str(index),
            "--allow-model-calls"]
        started, at = time.monotonic(), stamp()
        durable_write(directory / "dispatch-pending.json", {"run_id": state["run_id"],
            "plan_sha256": state["plan_sha256"], "question_index": index, "question_id": qid,
            "argv": command, "at": at, "compare_parent_pid": os.getpid(), "automatic_retry": False})
        remaining = (control.deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("Model cutoff reached before question dispatch")
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, cwd=args.previous / "source", stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, creationflags=0x08000000 if os.name == "nt" else 0)
            try:
                import sys
                identities = await_process_binding(child, directory, read(directory / "dispatch-pending.json"),
                    executables=list(dict.fromkeys([plan["python"], sys.executable, sys._base_executable])),
                    timeout_seconds=min(300, remaining))
                code = child.wait(timeout=max(0, (control.deadline - datetime.now(timezone.utc)).total_seconds()) + 60)
            except BaseException:
                # Only this owned handle is touched. The outer pre-GO OS job also
                # contains its descendants and reconciles them on parent failure.
                if child.poll() is None:
                    child.terminate()
                raise
        cleared = exited_process_binding(identities)
        exited = {"launcher_pid": child.pid, "worker_pid": identities["worker"]["pid"], "identity_exit_check": cleared,
            "identity_sha256": file_hash(directory / "identity-verified.json"), "exit_code": code, "process_exit_observed": True,
            "started_at": at, "finished_at": stamp(), "wall_seconds": time.monotonic() - started,
            "question_index": index, "question_id": qid, "run_id": state["run_id"], "plan_sha256": state["plan_sha256"]}
        durable_write(directory / "process-exit.json", exited)
        if code:
            raise RuntimeError("Question worker exited nonzero; preserve evidence and never retry it")
        receipt = read(directory / "receipt.json")
        if (receipt.get("pid") != identities["worker"]["pid"] or receipt.get("launcher_pid") != child.pid
                or receipt.get("parent_pid") != identities["worker"]["parent_pid"]
                or receipt.get("identity_sha256") != exited["identity_sha256"] or receipt.get("question_id") != qid
                or receipt.get("question_index") != index or receipt.get("run_id") != state["run_id"]
                or receipt.get("plan_sha256") != state["plan_sha256"] or receipt.get("terminal") is not True
                or receipt.get("cleanup_verified") is not True or receipt.get("pending_or_uncertain") is not False):
            raise ValueError("Question terminal receipt differs from the actual owned child")
        check_handoff(plan)
        terminal = read(directory / "runtime-terminal.json")
        if (terminal.get("cleanup", {}).get("verified") is not True
                or terminal.get("calls", {}).get("pending_or_uncertain") is not False):
            raise ValueError("Question cleanup or physical journal is unresolved")
        for key in ("admission", "runtime_terminal"):
            target = directory / (key.replace("_", "-") + ".json")
            if file_hash(target) != receipt[key + "_sha256"]:
                raise ValueError("Question admission/runtime evidence changed")
        for relative, expected in receipt["outputs_sha256"].items():
            if file_hash(args.output / relative) != expected:
                raise ValueError("Question source outcome changed")
        first = read(args.output / "initial" / (qid + ".json"))
        initial[qid] = first["submitted"]
        rows.extend(read(args.output / "episodes" / (qid + "-" + arm + ".json")) for arm in ARMS)
        control.active_seconds += receipt["active_seconds"]
        control.idle_seconds += receipt["intentional_idle_seconds"]
        receipts.append({"question_index": index, "question_id": qid, "path": str(directory / "receipt.json"),
            "sha256": file_hash(directory / "receipt.json"), "process_exit_sha256": file_hash(directory / "process-exit.json"),
            "dispatch_sha256": file_hash(directory / "dispatch-pending.json"),
            "identity_sha256": file_hash(directory / "identity-verified.json"), "owned_runtime_clear": True})
        terminals.append(terminal)
    # Each child journal summary is cumulative. Read the serial journal ONCE;
    # only independent child wall/idle/startup/switch durations may be summed.
    calls = CallJournal(args.output / "physical").verify()
    cleanup = {"verified": all(t["cleanup"]["verified"] for t in terminals),
        "private_port_closed": all(t["cleanup"]["private_port_closed"] for t in terminals),
        "owned_server_exited": all(t["cleanup"]["owned_server_exited"] for t in terminals),
        "gpu_lease_released": all(t["cleanup"]["gpu_lease_released"] for t in terminals),
        "native_unloads": [u for t in terminals for u in t["cleanup"]["native_unloads"]],
        "errors": [error for t in terminals for error in t["cleanup"]["errors"]]}
    durable_write(args.output / "runtime-terminal.json", {"cleanup": cleanup, "calls": calls,
        "timings": {key: sum(t["timings"][key] for t in terminals) for key in ("startup_seconds", "switch_seconds")},
        "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds,
        "comparison_process_layout": "per_question", "question_receipts": receipts,
        "timing_scope": "Nonoverlapping child durations are summed; physical journal totals are cumulative and counted once"})
    durable_write(args.output / "episodes.json", rows)
    durable_write(args.output / "initial-drafts.json", initial)
    result = analyse(rows, context, initial)
    result.update(wrapper_plan_sha256=os.environ["HALTIQ_PLAN_SHA256"], fresh_n_freeze_sha256=config["fresh_n_freeze_sha256"])
    durable_write(args.output / "comparison.json", result)
    completed = sum(row["status"] == "completed" for row in rows)
    return completion(args, planned=context["fresh_n"] * len(ARMS), completed=completed,
        failed=len(rows) - completed, cleanup=cleanup["verified"], pending=calls["pending_or_uncertain"])


def audit(args, context, control):
    from .main_audit import verify_compare
    snapshots = {}
    for phase in PHASES[:-1]:
        if not (args.previous / phase / "physical/calls.jsonl").is_file():
            raise ValueError("Final audit requires every original physical journal")
        snapshots[phase] = CallJournal(args.previous / phase / "physical").verify()
        if snapshots[phase]["pending_or_uncertain"]:
            raise ValueError("An unresolved actual call prevents final qualification")
    rows = read(args.previous / "features/features.json")
    baseline = read(args.previous / "features/untouched-baseline.json")
    verify_feature_rows(args, context, rows, baseline)
    observed = read(args.previous / "compare/comparison.json")
    recomputed = analyse(read(args.previous / "compare/episodes.json"), context,
                         read(args.previous / "compare/initial-drafts.json"))
    compare_proof = verify_compare(args, context, read(args.previous / "compare/episodes.json"),
                                  read(args.previous / "compare/initial-drafts.json"))
    if any(observed.get(k) != value for k, value in recomputed.items()):
        raise ValueError("Final comparison does not reproduce from all actual episodes")
    learning = read(args.previous / "train/learning.json")
    if (_digest(args.previous / "train/learned/answer-head.safetensors") != learning["criteria"]["answer"]["trained_head_sha256"]
            or _digest(Path(context["config"]["checkpoint"]) / "head.safetensors") != learning["released_head_sha256"]):
        raise ValueError("Final released/trained head files changed after inference")
    result = {"terminal": True, "pending_or_uncertain": False, "research_audit_passed": True,
              "physical_journals": snapshots, "comparison_sha256": _digest(args.previous / "compare/comparison.json"),
              "native_compare_audit": compare_proof,
              "fresh_n_freeze_sha256": context["config"]["fresh_n_freeze_sha256"],
              "scope": "recomputed actual outcomes; no new model calls"}
    durable_write(args.output / "audit.json", result)
    return completion(args, planned=1, completed=1, failed=0, cleanup=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    for flag in ("config", "output", "previous", "control"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    if args.phase != "audit" and not args.allow_model_calls:
        raise ValueError("Main model phases need explicit opt-in")
    plan, state = read(args.previous / "plan.json"), read(args.previous / "supervisor.json")
    verify_dispatch(args, plan, state)
    config = pinned(args.config, plan["main_protocol"]["sha256"])
    import importlib.util
    for module, expected in config["runtime_modules"].items():
        found = importlib.util.find_spec(module)
        if found is None or Path(found.origin).resolve() != Path(expected).resolve():
            raise ValueError("Main runtime imported a different pinned package: " + module)
    context = load_main(config)
    cutoff = execution_cutoff(config, args.deadline, args.phase)
    args.output.mkdir(parents=True, exist_ok=False)
    control, started = RunControl(args.control, cutoff), time.monotonic()
    try:
        value = {"collect": collect, "features": features, "train": train, "compare": compare, "audit": audit}[args.phase](args, context, control)
        durable_write(args.output / "phase-timing.json", {"wall_seconds": time.monotonic() - started,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
        print(json.dumps({"phase": args.phase, "completed": value["completed_units"], "failed": value["failed_units"]}))
    except BaseException as error:
        durable_write(args.output / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "phase": args.phase, "no_automatic_retry": True, "actual_partial_episode": getattr(error, "haltiq_episode_partial", None)})
        raise


if __name__ == "__main__":
    main()
