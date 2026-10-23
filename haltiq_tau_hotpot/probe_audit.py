"""Terminal-only synthetic runtime qualification; no model loads or head execution."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import re

from . import qualification as q

ROLES = {"native_startup": 1, "native_answer": 3, "native_evidence": 3,
         "cached_head_answer": 6, "cached_head_evidence": 6,
         "writer_load": 1, "writer": 1, "reviewer": 4, "writer_unload": 1}
SCHEMA = "haltiq-qualified-runtime-probe-v2"


def terminal_owner(run, expected_plan_sha256):
    """Reject active ownership before opening any model-result artifacts."""
    plan, state = q.read(run / "plan.json"), q.read(run / "supervisor.json")
    q.require(q.sha(run / "plan.json") == expected_plan_sha256 == state.get("plan_sha256"), "Probe owner/plan changed")
    q.require(plan.get("purpose") == "runtime_probe" and plan.get("scope") == "custom_tau2_hotpot"
              and [s["id"] for s in plan["stages"]] == ["probe"]
              and state.get("status") == "completed" and set(state.get("stages", {})) == {"probe"},
              "The single probe phase must be genuinely terminal")
    row = state["stages"]["probe"]
    q.require(row.get("status") == "completed" and row.get("exit_code") == 0
              and row.get("owned_tree_clear") is True and row.get("os_handoff_verified") is True,
              "Probe owned-process cleanup is not verified")
    expected = {str(q.within(s["path"].replace("{run}", str(run)), run)) for s in plan["stages"][0]["outputs"]}
    canonical = {str(run / "probe" / name) for name in ("probe.json", "completion.json", "runtime-terminal.json", "phase-timing.json")}
    q.require(canonical <= expected and set(row.get("output_sha256", {})) == expected, "Missing probe output binding")
    for path, hashed in row["output_sha256"].items():
        q.require(q.sha(q.within(path, run)) == hashed, "Probe output changed")
    actual = {p.relative_to(run / "source").as_posix(): q.sha(p) for p in (run / "source").rglob("*") if p.is_file()}
    q.require(actual == state.get("source_sha256") == plan.get("reviewed_source_sha256"), "Probe frozen source changed")
    q.require(q.sha(plan["python"]) == state.get("python_sha256"), "Probe interpreter changed")
    return plan, state


def physical(directory, journal_class):
    """Use the actual cache-aware journal verifier, then add serial/time checks."""
    q.require((directory / "calls.jsonl").is_file(), "Missing physical probe journal")
    try:
        journal = journal_class(directory)
        summary = journal.verify()
    except (ValueError, RuntimeError) as error:
        raise q.QualificationError("Invalid/unresolved physical probe journal: " + str(error)) from error
    q.require(summary == {"dispatched": 26, "completed": 26, "uncertain": 0,
              "pending_or_uncertain": False, "exact_response_reuses": 2, "by_role": ROLES}, "Wrong bounded physical route")
    active, pending, calls, reuses, previous = None, {}, [], [], None
    for event in journal.records:
        at = q.aware(event["at"])
        q.require(previous is None or at >= previous, "Nonmonotonic physical journal")
        previous = at
        if event["event"] == "pending":
            q.require(active is None, "Overlapping physical probe dispatch")
            pending[event["id"]], active = event, event["id"]
        elif event["event"] == "completed":
            q.require(active == event["id"] and type(event.get("elapsed_seconds")) in (int, float)
                      and math.isfinite(event["elapsed_seconds"]) and event["elapsed_seconds"] >= 0, "Invalid physical completion")
            calls.append({"pending": pending[event["id"]], "completed": event,
                          "value": q.read(directory / event["response"])})
            active = None
        else:
            q.require(event["event"] == "exact_response_reuse" and active is None, "Uncertain/overlapping reuse")
            reuses.append(event)
    q.require(Counter(e["role"] for e in reuses) == {"writer": 1, "reviewer": 1}, "Wrong exact-response reuse roles")
    return {"summary": summary, "calls": calls, "reuses": reuses, "journal_sha256": q.sha(journal.path)}


def bound(calls, used, role, request, value=None):
    matches = [c for c in calls if c["completed"]["role"] == role and c["pending"]["request_sha256"] == q.fingerprint(request)
               and (value is None or c["value"] == value)]
    q.require(len(matches) == 1 and matches[0]["completed"]["id"] not in used, "Unbound or multiply consumed probe response")
    call = matches[0]
    used.add(call["completed"]["id"])
    return call


def points(run, probe, config, encoder, trained, calls, used, runtime, hobson):
    result, feature_paths = [], set()
    expected = [(target, criterion) for target in (512, 2048, 4096) for criterion in ("answer", "evidence")]
    q.require([(p["target_tokens"], p["criterion"]) for p in probe["points"]] == expected, "Probe point coverage/order changed")
    for index, row in enumerate(probe["points"]):
        criterion, target, public, draft = [row[k] for k in ("criterion", "target_tokens", "public", "draft")]
        sentence = public["context"][0][1][0]
        q.require(public == {"id": f"synthetic-{criterion}-{target}", "type": "synthetic_control", "level": "engineering_only",
                  "question": "What synthetic key is stated in the note?", "context": [["Synthetic runtime note", [sentence]]]}
                  and re.fullmatch(r"The synthetic key is control\.(?: filler)*", sentence)
                  and draft == {"answer": "control", "supporting_facts": [["Synthetic runtime note", 0]]}, "Non-synthetic probe input")
        q.require(row == q.read(run / "probe/points" / f"{criterion}-{target}.json")
                  and row["cold_first_forward_after_placement"] is (index == 0)
                  and type(row["input_tokens"]) is int and target - 8 <= row["input_tokens"] <= target,
                  "Changed point artifact, cold flag or token boundary")
        state = runtime.decision_state(public, draft, [0], criterion)
        actual, same, learned = [row[k] for k in ("actual", "cached_released", "cached_trained")]
        native = bound(calls, used, "native_" + criterion,
                       {"state": state, "criterion": criterion, "head": "released", "temperature": 1.}, actual)
        feature, provenance = actual["feature"], actual["feature"]["provenance"]
        path = q.within(feature["path"], run / "probe/features")
        q.require(path.name == q.fingerprint({"state": state, "criterion": criterion}) + ".safetensors"
                  and q.sha(path) == feature["sha256"] and feature["schema"] == hobson.FEATURE_SCHEMA
                  and feature["criterion"] == criterion and feature["option_order"] == ["false", "true"]
                  and feature["encoder_sha256"] == encoder and provenance["state_sha256"] == q.fingerprint(state)
                  and provenance["criterion"] == criterion and provenance["option_order"] == ["false", "true"]
                  and provenance["instructions_sha256"] == q.fingerprint(hobson.INSTRUCTIONS_BY_CRITERION[criterion])
                  and provenance["source_sha256"] == q.sha(hobson.__file__)
                  and feature["provenance_sha256"] == q.fingerprint(provenance)
                  and all(re.fullmatch(r"[0-9a-f]{64}", provenance[k]) for k in ("token_ids_sha256", "rendered_question_sha256")),
                  "Native feature/public-input provenance changed")
        q.safe_tensors(path, {"decide": [2048], "options": [2, 2048]})
        feature_paths.add(path)
        q.probability(actual, native=True)
        q.require(type(actual["input_tokens"]) is int and actual["input_tokens"] == row["input_tokens"] and actual["temperature"] == 1.
                  and actual["retrieval_indices"] == [0] and actual["cached_replay_max_logit_error"] >= 0,
                  "Native token count, criterion or technical replay changed")
        for cached, head in ((same, Path(config["checkpoint"]) / "head.safetensors"), (learned, trained)):
            q.probability(cached, native=False)
            q.require(cached["temperature"] == 1. and cached["cache_reused"] is True and cached["feature"] == feature
                      and cached["retrieval_indices"] == [0] and cached["original_native_call_id"] == native["completed"]["id"]
                      and cached["original_native_response_sha256"] == native["completed"]["response_sha256"], "Cached head ancestry changed")
            bound(calls, used, "cached_head_" + criterion, {"original_native_call_id": native["completed"]["id"],
                  "feature_sha256": feature["sha256"], "head_sha256": q.sha(head), "criterion": criterion, "temperature": 1.}, cached)
        q.require(same["raw_logits_false_true"] == actual["raw_logits_false_true"], "Released cached logits changed")
        result.append({"id": native["completed"]["id"], "criterion": criterion, "target_tokens": target,
                       "input_tokens": row["input_tokens"], "cold_first_forward_after_placement": index == 0})
    q.require(feature_paths == {p.resolve() for p in (run / "probe/features").rglob("*") if p.is_file()}, "Unaccounted feature artifact")
    return result


def generation(run, probe, config, physical, used, module):
    state = {"question": "What is the synthetic key?", "evidence": [{"title": "Synthetic runtime note",
        "sentences": [{"index": 0, "text": "The synthetic key is control."}]}], "previous_draft": None}
    calls = physical["calls"]
    writer = bound(calls, used, "writer", module.generation_payload(config, state))
    answer = module.parse_response(writer["value"])
    q.require(probe["generation_results"] == [answer], "Writer generation result is not the committed response")
    states, rows = module.synthetic_reviews(), probe["reviewer_ab"]
    q.require([(r["case_id"], r["profile"]) for r in rows] == list(module.REVIEW_SEQUENCE), "Synthetic reviewer A/B order changed")
    measured, ids = [], []
    for index, row in enumerate(rows):
        case_id, profile = row["case_id"], row["profile"]
        q.require(row == q.read(run / "probe/reviewer-ab" / f"{index}.json")
            and row["state"] == states[case_id], "Synthetic A/B cell/public state changed")
        payload = module.generation_payload(config, states[case_id], review=True, profile=profile)
        call = bound(calls, used, "reviewer", payload)
        raw, failure, feedback = call["value"], None, None
        try:
            feedback = module.parse_response(raw, review=True)
        except ValueError as error:
            failure = str(error)
        metadata = module.response_metadata(raw)
        q.require(row["physical_call_id"] == call["completed"]["id"]
            and row["request_sha256"] == q.fingerprint(payload) and row["metadata"] == metadata
            and row["feedback"] == feedback and row["failure"] == failure
            and row["status"] == ("completed" if failure is None else "known_terminal_failure"),
            "A/B result differs from its committed response/technical status")
        if profile == "main":
            q.require(failure is None and raw.get("done") is True and raw.get("done_reason") == "stop" and metadata["words"] <= 120,
                      "Predeclared main reviewer did not satisfy its technical completion/length check")
        ids.append(call["completed"]["id"])
        measured.append({"id": call["completed"]["id"], "case_id": case_id, "profile": profile,
            "state_sha256": q.fingerprint(states[case_id]), "request_sha256": q.fingerprint(payload),
            "seconds": call["completed"]["elapsed_seconds"], "metadata": metadata, "status": row["status"]})
    expected_reuses = {"writer": writer["completed"]["id"], "reviewer": ids[1]}
    q.require({e["role"]: e["id"] for e in physical["reuses"]} == expected_reuses,
              "Exact generation reuse must bind writer and the first main-profile review")
    q.require({p.name for p in (run / "probe/reviewer-ab").rglob("*") if p.is_file()} == {f"{i}.json" for i in range(4)},
              "Unexpected/missing synthetic review cell artifact")
    # The intended memory route places writer once, completes all generation,
    # unloads it, then places native once; no generation follows native startup.
    roles = [c["completed"]["role"] for c in calls]
    q.require(roles[:8] == ["writer_load", "writer", "reviewer", "reviewer", "reviewer", "reviewer", "writer_unload", "native_startup"]
        and not any(role in {"writer", "reviewer", "writer_load"} for role in roles[8:]),
        "All generation must precede the single native placement")
    native_start = calls[7]["pending"]["at"]
    q.require(all(q.aware(e["at"]) <= q.aware(native_start) for e in physical["reuses"]),
              "Generation reuse must finish before native placement")
    return measured


def timing(run, plan, state, physical, terminal, point_rows):
    value, owner = q.read(run / "probe/phase-timing.json"), state["stages"]["probe"]
    q.require(all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
                  for k in ("wall_seconds", "active_seconds", "intentional_idle_seconds"))
              and value["active_seconds"] + value["intentional_idle_seconds"] <= value["wall_seconds"] + 2
              and all(terminal[k] == value[k] for k in ("active_seconds", "intentional_idle_seconds")), "Invalid phase timing")
    start, end = q.aware(owner["started_at"]), q.aware(owner["finished_at"])
    q.require(start <= q.aware(q.read(run / "probe/completion.json")["finished_at"]) <= end
              and end <= min(q.aware(plan["hard_deadline"]), datetime.now(timezone.utc)), "Probe completion outside its owned clock")
    details = {p["id"]: p for p in point_rows}
    for call in physical["calls"]:
        pending, completed = call["pending"], call["completed"]
        q.require(start <= q.aware(pending["at"]) <= q.aware(completed["at"]) <= end
                  and (pending["role"] == "writer_unload" or q.aware(pending["at"]) <= q.aware(plan["deadline"])), "Call outside its phase/cutoff")
    q.require(all(start <= q.aware(r["at"]) <= end for r in physical["reuses"]), "Reuse outside its phase")
    q.require(type(state.get("supervisor_wall_seconds")) in (int, float)
              and math.isfinite(state["supervisor_wall_seconds"]) and state["supervisor_wall_seconds"] >= 0, "Missing supervisor timing")
    q.require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in terminal["timings"].values()), "Invalid startup/switch timing")
    return {**value, "supervisor_wall_seconds": state["supervisor_wall_seconds"], "startup_and_switch": terminal["timings"],
        "physical_calls": [{"id": c["completed"]["id"], "role": c["completed"]["role"],
            "seconds": c["completed"]["elapsed_seconds"], **details.get(c["completed"]["id"], {})} for c in physical["calls"]],
        "scope": "Containing wall/active/idle/startup/physical durations overlap; do not add them."}


def audit_run(run, *, expected_plan_sha256):
    run = Path(run).resolve()
    plan, state = terminal_owner(run, expected_plan_sha256)
    with q.frozen_helpers(run) as helpers:
        protocol, engine = helpers["haltiq_supervisor.plan"], helpers["haltiq_supervisor.engine"]
        protocol.validate(plan)
        q.require(protocol.READINESS <= plan.get("readiness", {}).keys()
                  and all(specs and all(s.get("sha256") for s in specs) for specs in plan["readiness"].values()), "Missing implementation/pilot readiness")
        protocol.verify_frozen(plan, state, run)
        runtime, worker, generation_module = [importlib.import_module("haltiq_tau_hotpot." + name) for name in ("runtime", "runtime_probe", "generation")]
        journal_module = importlib.import_module("haltiq_tau_hotpot.journal")
        for module in (runtime, worker, journal_module, generation_module):
            q.within(module.__file__, run / "source")
        config = q.read(plan["runtime_probe"]["path"])
        pinned_inputs = {str(Path(s["path"]).resolve()): s["sha256"] for s in
            [*plan.get("inputs", []), *(s for specs in plan["readiness"].values() for s in specs)]}
        q.require(pinned_inputs.get(str(Path(config["combined_pilot_proof"]).resolve())) == config["combined_pilot_proof_sha256"], "Pilot proof is not a frozen readiness/input")
        pilot, trained = worker.pilot_binding(config)
        hobson = helpers["haltiq_tau_hotpot.hobson"]
        encoder = hobson.encoder_binding(config["checkpoint"], config["base"])
        binding = q.model_binding(config, {"encoder_sha256": encoder, "released_head_sha256": pilot["model_binding"]["released_head_sha256"]})
        q.require(binding == pilot["model_binding"] and pilot.get("terminal") is True
                  and pilot.get("actual_os_handoff_verified") is True, "Qualified pilot/model mismatch")
        for head in (Path(config["checkpoint"]) / "head.safetensors", trained):
            q.safe_tensors(head, q.HEAD_SHAPES)
        engine.verify_completion(plan["stages"][0], state, q.read(run / "probe/completion.json"))
        completion = q.read(run / "probe/completion.json")
        q.require((completion["completed_units"], completion["failed_units"], completion["not_run_units"]) == (25, 0, 0), "Incomplete logical probe units")
        probe, terminal = q.read(run / "probe/probe.json"), q.read(run / "probe/runtime-terminal.json")
        q.require(probe.get("schema") == "haltiq-runtime-probe-v2" and probe.get("status") == "PASS_RUNTIME_PROBE"
                  and probe.get("engineering_only") is True and all(probe.get(k) is False for k in ("benchmark_quality_claim", "new_learning", "long_study_started"))
                  and probe["wrapper_plan_sha256"] == state["plan_sha256"] and probe["source_sha256"] == state["source_sha256"]
                  and probe["pilot_proof_sha256"] == config["combined_pilot_proof_sha256"] and probe["pilot_run_id"] == pilot["run_id"]
                  and probe["released_head_sha256"] == binding["released_head_sha256"] and probe["trained_head_sha256"] == q.sha(trained)
                  and probe["generation_profile"] == config["generation_profile"] == generation_module.MAIN_PROFILE, "Probe provenance changed")
        cleanup = terminal["cleanup"]
        q.require(all(cleanup.get(k) is True for k in ("verified", "private_port_closed", "owned_server_exited", "gpu_lease_released"))
                  and cleanup["errors"] == [] and cleanup["native_unloads"]
                  and all(v["allocated_bytes"] == v["reserved_bytes"] == 0 for v in cleanup["native_unloads"]), "Runtime cleanup failed")
        physical_work, used = physical(run / "probe/physical", journal_module.CallJournal), set()
        q.require(probe["calls"] == terminal["calls"] == physical_work["summary"], "Physical summaries differ")
        rows = points(run, probe, config, encoder, trained, physical_work["calls"], used, runtime, hobson)
        reviewer_ab = generation(run, probe, config, physical_work, used, generation_module)
        startup = bound(physical_work["calls"], used, "native_startup", {"checkpoint": config["checkpoint"]})["value"]["metadata"]
        q.require(startup["encoder_sha256"] == encoder and startup["released_head_sha256"] == binding["released_head_sha256"]
                  and startup["head_parameters"] == 1053184, "Native placement binding changed")
        for role in ("writer_load", "writer_unload"):
            bound(physical_work["calls"], used, role, {"model": "qwen3:8b"})
        q.require(len(used) == 26, "An actual physical operation was not bound")
        timing_value = timing(run, plan, state, physical_work, terminal, rows)
        for lease in (Path(plan["study_lease"]), run / "supervisor.lock"):
            q.require(lease.is_file(), "Missing actual ownership lease")
            with helpers["pc_strands.control"].GpuLease(lease, "terminal synthetic probe audit"):
                pass
        engine.check_handoff(plan)
        protocol.verify_frozen(plan, state, run)
        q.require((plan, state) == terminal_owner(run, expected_plan_sha256), "Terminal probe owner changed during audit")
        return {"status": "PASS_QUALIFIED_RUNTIME_PROBE", "qualified": True, "terminal": True, "cleanup_verified": True,
            "pending_or_uncertain": False, "actual_os_handoff_verified": True, "os_handoff_verified_at": datetime.now(timezone.utc).isoformat(),
            "run_id": state["run_id"], "plan_sha256": state["plan_sha256"], "supervisor_sha256": q.sha(run / "supervisor.json"),
            "source_sha256": state["source_sha256"], "model_binding": binding, "trained_head_sha256": q.sha(trained),
            "pinned_inputs_sha256": pinned_inputs,
            "pilot_proof_sha256": config["combined_pilot_proof_sha256"], "physical_calls": physical_work["summary"],
            "journal_sha256": physical_work["journal_sha256"], "timing": timing_value,
            "generation_profile": config["generation_profile"], "reviewer_ab": reviewer_ab,
            "output_sha256": {p.relative_to(run).as_posix(): q.sha(p) for p in (run / "probe").rglob("*") if p.is_file()},
            "new_model_forwards": 0, "new_head_scores": 0, "long_study_started_by_audit": False}


def qualify(run, output, *, expected_plan_sha256):
    run, output = Path(run).resolve(), Path(output).resolve()
    q.require(not output.exists() and output != run and run / "source" not in output.parents
              and run / "probe" not in output.parents, "Use a new receipt outside frozen source/probe artifacts")
    try:
        value = audit_run(run, expected_plan_sha256=expected_plan_sha256)
    except Exception as error:
        value = {"status": "BLOCKED_NO_QUALIFICATION", "qualified": False, "error_type": type(error).__name__,
                 "reason": str(error), "new_model_forwards": 0, "new_head_scores": 0, "long_study_started_by_audit": False}
    value.update(schema=SCHEMA, run=str(run), reviewed_at=datetime.now(timezone.utc).isoformat(), auditor_source_sha256=q.sha(__file__))
    from .journal import durable_write
    durable_write(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    result = qualify(args.run, args.output, expected_plan_sha256=args.expected_plan_sha256)
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": q.sha(args.output)}))
    raise SystemExit(0 if result["qualified"] else 1)


if __name__ == "__main__":
    main()
