"""Read-only, terminal-only qualification of the small native Tau2-Hotpot pilot.

This audit never loads a model, imports Torch, retries calls, or starts a study.
Recorded callback results may be replayed through the frozen native Tau2 tools;
that is a deterministic protocol check, not new model evidence.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import sys

EXPECTED_PLAN_SHA256 = "d0bb017313610c656c7efe6c347939eb510185ed15ba66084b7f87c607eea161"
PHASES = ("collect", "features", "train", "compare")
PACKAGES = ("haltiq", "pc_strands", "haltiq_hotpot", "haltiq_tau", "haltiq_supervisor", "haltiq_tau_hotpot")
HEAD_SHAPES = {"k.bias": [256], "k.weight": [256, 2048], "norm.bias": [2048],
               "norm.weight": [2048], "q.bias": [256], "q.weight": [256, 2048]}


class QualificationError(ValueError):
    """Missing or inconsistent evidence suppresses combined-pilot readiness."""


def require(value, message):
    if not value:
        raise QualificationError(message)


def read(path):
    def invalid(value):
        raise QualificationError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def aware(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Offset-bearing receipt timestamp required")
    return parsed.astimezone(timezone.utc)


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path == root or root in path.parents, "Evidence path escapes the adopted run")
    return path


def terminal_owner(run, expected_plan_sha256):
    """Check terminal ownership before inspecting any question/model outcomes."""
    plan_path, state_path = run / "plan.json", run / "supervisor.json"
    plan, state = read(plan_path), read(state_path)
    require(sha(plan_path) == expected_plan_sha256 == state.get("plan_sha256"), "Plan/owner hash mismatch")
    require(plan.get("purpose") == "qualification_pilot" and plan.get("scope") == "custom_tau2_hotpot",
            "Only the explicitly bounded qualification pilot can be qualified")
    require(state.get("status") == "completed" and set(state.get("stages", {})) == set(PHASES),
            "All four supervisor phases must be genuinely terminal and completed")
    require(tuple(s["id"] for s in plan["stages"]) == PHASES, "Wrong phase order")
    for stage in plan["stages"]:
        row = state["stages"][stage["id"]]
        require(row.get("status") == "completed" and row.get("exit_code") == 0
                and row.get("owned_tree_clear") is True and row.get("os_handoff_verified") is True,
                "A phase lacks successful owned-process cleanup")
        require(row.get("output_sha256"), "Missing generated-output binding")
        expected_outputs = {str(within(spec["path"].replace("{run}", str(run)), run)) for spec in stage["outputs"]}
        require(set(row["output_sha256"]) == expected_outputs, "A planned generated-output binding is missing")
        for path, expected in row["output_sha256"].items():
            require(sha(within(path, run)) == expected, "Completed generated evidence changed")
    actual = {p.relative_to(run / "source").as_posix(): sha(p)
              for p in (run / "source").rglob("*") if p.is_file()}
    require(actual == state.get("source_sha256") == plan.get("reviewed_source_sha256"),
            "Frozen source/manifest changed")
    require(sha(plan["python"]) == state.get("python_sha256"), "Interpreter changed")
    return plan, state


@contextmanager
def frozen_helpers(run):
    """Import reviewed helpers from this run even as the checkout evolves.

    This command is single-threaded. Restore previous import state on exit; no
    caller can accidentally retain a partly replaced study package namespace.
    """
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    selected = lambda name: name.split(".", 1)[0] in PACKAGES
    old = {name: module for name, module in sys.modules.items() if selected(name)}
    for name in old:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(run / "source"))
    before = set(sys.modules)
    try:
        result = {name: importlib.import_module(name) for name in (
            "haltiq_supervisor.plan", "haltiq_supervisor.engine", "pc_strands.control",
            "haltiq_tau_hotpot.worker", "haltiq_tau_hotpot.agent", "haltiq_tau_hotpot.domain",
            "haltiq_tau_hotpot.comparison", "haltiq_tau_hotpot.learning", "haltiq_tau_hotpot.hobson")}
        require(not (set(sys.modules) - before) & {"torch", "transformers", "peft"},
                "Qualification helpers unexpectedly imported a model runtime")
        for module in result.values():
            if module.__name__.split(".", 1)[0] in PACKAGES:
                within(module.__file__, run / "source")
        yield result
    finally:
        for name in list(sys.modules):
            if selected(name):
                sys.modules.pop(name, None)
        sys.modules.update(old)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def journal(directory):
    """Stricter independent journal reconciliation, including duplicate IDs."""
    directory = Path(directory)
    records = [json.loads(line) for line in (directory / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    pending, complete, responses, active = {}, {}, [], set()
    for event in records:
        identifier, kind = event["id"], event["event"]
        require(kind in {"pending", "completed"}, "Pending/uncertain/foreign physical journal event")
        if kind == "pending":
            require(identifier not in pending and identifier not in complete and not active,
                    "Duplicate/overlapping physical dispatch")
            require(isinstance(event.get("request_sha256"), str) and len(event["request_sha256"]) == 64,
                    "Unbound physical request")
            pending[identifier] = event
            active.add(identifier)
        else:
            require(identifier in pending and identifier not in complete, "Unmatched/duplicate physical completion")
            request = pending[identifier]
            require(event["role"] == request["role"] and aware(event["at"]) >= aware(request["at"]),
                    "Physical role/time mismatch")
            path = within(directory / event["response"], directory)
            require(sha(path) == event["response_sha256"], "Physical response changed")
            require(type(event.get("elapsed_seconds")) in (int, float)
                    and math.isfinite(event["elapsed_seconds"]) and event["elapsed_seconds"] >= 0,
                    "Invalid physical duration")
            complete[identifier] = event
            active.remove(identifier)
            responses.append({"pending": request, "completed": event, "value": read(path)})
    require(set(pending) == set(complete), "Unresolved physical dispatch")
    return {"calls": len(complete), "by_role": dict(Counter(v["completed"]["role"] for v in responses)),
            "physical_call_seconds": sum(v["completed"]["elapsed_seconds"] for v in responses),
            "responses": responses, "sha256": sha(directory / "calls.jsonl")}


def safe_tensors(path, shapes):
    """Inspect exact finite F32 tensor bytes with no Torch/model imports."""
    payload = Path(path).read_bytes()
    require(len(payload) >= 8, "Truncated tensor artifact")
    length = struct.unpack("<Q", payload[:8])[0]
    require(0 < length <= 1024 * 1024 and 8 + length <= len(payload), "Invalid tensor header")
    header = json.loads(payload[8:8 + length])
    require(set(header) - {"__metadata__"} == set(shapes), "Wrong tensor names")
    blocks, offsets = {}, []
    for name, shape in shapes.items():
        spec = header[name]
        require(spec.get("dtype") == "F32" and spec.get("shape") == shape, "Wrong tensor dtype/shape")
        start, stop = spec["data_offsets"]
        require(type(start) is int and type(stop) is int and 0 <= start <= stop
                and stop - start == math.prod(shape) * 4 and 8 + length + stop <= len(payload), "Invalid tensor span")
        block = payload[8 + length + start:8 + length + stop]
        values = array("f")
        values.frombytes(block)
        if sys.byteorder != "little":
            values.byteswap()
        require(all(math.isfinite(value) for value in values), "Nonfinite cached/head tensor")
        offsets.append((start, stop))
        blocks[name] = (shape, block, values)
    ordered = sorted(offsets)
    require(ordered[0][0] == 0 and all(a[1] == b[0] for a, b in zip(ordered, ordered[1:]))
            and ordered[-1][1] == len(payload) - 8 - length, "Tensor payload gaps/overlap/trailing bytes")
    tensor_hash = hashlib.sha256()
    for name in sorted(blocks):
        shape, block, _ = blocks[name]
        tensor_hash.update(name.encode("utf-8"))
        tensor_hash.update(str(tuple(shape)).encode("ascii"))
        tensor_hash.update(block)
    return blocks, tensor_hash.hexdigest()


def probability(result, *, native):
    logits, temperature = result["raw_logits_false_true"], result["temperature"]
    require(len(logits) == 2 and all(type(x) in (float, int) and math.isfinite(x) for x in logits)
            and type(temperature) in (float, int) and math.isfinite(temperature) and temperature > 0,
            "Invalid gate logits/temperature")
    margin = (logits[1] - logits[0]) / temperature
    expected = 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))
    require(math.isclose(result["p_true_unrounded"], expected, rel_tol=2e-6, abs_tol=2e-7)
            and result["p_true"] == round(result["p_true_unrounded"], 4), "Gate probability/rounding mismatch")
    require(result["physical_forwards"] == int(native) and result["head_scores"] == (2 if native else 1),
            "Physical encoder/head accounting mismatch")
    if native:
        require(len(result["native_logits_false_true"]) == 2
                and max(abs(a - b) for a, b in zip(logits, result["native_logits_false_true"])) <= 2e-4
                and result["cached_replay_max_logit_error"] <= 2e-4, "Native/cached head logits differ")


def native_episode(row, case, indices, helpers, *, arm, max_drafts):
    """Replay callbacks only; every native tool/submission must match the record."""
    agent, scorer = helpers["haltiq_tau_hotpot.agent"], helpers["haltiq_tau_hotpot.domain"]
    require(row["question_id"] == case["public"]["id"] and row["arm"] == arm
            and row["passage_indices"] == indices and row["seed"] == 42
            and row["critic_threshold"] == .8 and row["caps"]["answer_drafts"] == max_drafts,
            "Episode identity/retrieval/caps changed")
    trace = iter(row["trace"])
    # Callbacks consume only their own trace events; native tool events are
    # reconstructed from the declared original public passages below.
    events = [v for v in trace if v["event"] in {"draft", "review", "gate"}]
    queue = iter(events)
    def next_event(kind):
        event = next(queue)
        require(event["event"] == kind, "Recorded callback order differs")
        return event
    def writer(*_):
        event = next_event("draft")
        if event["status"] == "failed":
            raise ValueError("Recorded known writer failure")
        return {} if event["status"] == "invalid_submission" else event["draft"]
    def reviewer(*_):
        event = next_event("review")
        if event["status"] == "failed":
            raise ValueError("Recorded known reviewer failure")
        return event["feedback"]
    def gate(_, __, criterion):
        event = next_event("gate")
        require(event["criterion"] == criterion, "Gate criterion changed")
        if event["status"] == "failed":
            raise ValueError("Recorded known gate failure")
        return event["score"]
    fallback = {"cap_fallback": row["cap_fallback"]} if "cap_fallback" in row else {}
    if not fallback:
        require(not any(key in row for key in ("submission_reason", "budget_fallback_used", "gate_approved")),
                "Legacy episode cannot acquire undeclared fallback metadata")
    result = agent.run_episode({"public": case["public"], "evaluator": case["evaluator"]}, writer,
        reviewer, gate, arm=arm, max_drafts=max_drafts, threshold=.8, passage_indices=indices,
        caps={k: v for k, v in row["caps"].items() if k != "answer_drafts"}, seed=42, timeout_seconds=1800, **fallback)
    require(next(queue, None) is None, "Extra unconsumed callback event")
    for key in ("scope", "upstream_revision", "scorer", "status", "failure_category", "submitted",
                "metrics", "native_reward", "counts", "caps", "trace", "physical_cost_scope"):
        require(result[key] == row[key], "Native episode replay differs: " + key)
    if "cap_fallback" in row:
        for key in ("cap_fallback", "submission_reason", "budget_fallback_used", "gate_approved"):
            require(result[key] == row[key], "Native fallback replay differs: " + key)
    actual, replay = row["native_simulation"], result["native_simulation"]
    require(actual is not None and actual["task_id"] == case["public"]["id"]
            and actual["mode"] == "half_duplex" and actual["seed"] == 42,
            "Missing native Tau simulation")
    require(actual["termination_reason"] == replay["termination_reason"]
            and actual["reward_info"] == replay["reward_info"] and actual["policy"] == scorer.POLICY,
            "Native termination/evaluator result changed")
    keep = ("role", "content", "tool_calls", "id", "requestor", "error", "execution_status", "turn_idx")
    simplify = lambda messages: [{k: message[k] for k in keep if k in message} for message in messages]
    require(simplify(actual["messages"]) == simplify(replay["messages"]), "Native tool dispatch/results differ")
    return result


def feature_rows(run, cohorts, rows, baseline, helpers):
    from argparse import Namespace
    helpers["haltiq_tau_hotpot.worker"].verify_native_rows(Namespace(previous=run), {}, cohorts, rows, baseline)
    expected = {c["public"]["id"]: c for group in cohorts.values() for c in group}
    require(len(rows) == len(expected) and {r["question_id"] for r in rows} == set(expected)
            and len({r["question_id"] for r in rows}) == len(rows), "Feature denominator/identity mismatch")
    for row in rows:
        require(row["status"] == "completed" and row["criterion"] == "answer"
                and row["partition"] == expected[row["question_id"]]["partition"], "Required native feature missing")
        value, feature = row["observed"], row["observed"]["feature"]
        probability(value, native=True)
        within(feature["path"], run / "features")
        safe_tensors(feature["path"], {"decide": [2048], "options": [2, 2048]})
        provenance = feature["provenance"]
        require(set(feature) == {"schema", "criterion", "option_order", "encoder_sha256", "provenance",
                                 "provenance_sha256", "path", "sha256"}
                and set(provenance) == {"criterion", "state_sha256", "instructions_sha256",
                    "rendered_question_sha256", "token_ids_sha256", "option_order", "source_sha256"}
                and feature["schema"] == "haltiq-hobson-pointer-feature-v1"
                and feature["criterion"] == "answer" and feature["option_order"] == ["false", "true"]
                and fingerprint(provenance) == feature["provenance_sha256"]
                and provenance["source_sha256"] == sha(run / "source/haltiq_tau_hotpot/hobson.py")
                and provenance["criterion"] == "answer"
                and provenance["instructions_sha256"] == fingerprint(helpers["haltiq_tau_hotpot.hobson"].INSTRUCTIONS_BY_CRITERION["answer"]),
                "Feature schema/input-source provenance mismatch")
    return expected


def learning_proof(run, config, cohorts, rows, baseline, physical, helpers):
    proof = read(run / "train/learning.json")
    require(proof == read(run / "train/learned/learning-proof.json") and proof["status"] == "completed",
            "Missing matching completed learning proof")
    hobson, learning = helpers["haltiq_tau_hotpot.hobson"], helpers["haltiq_tau_hotpot.learning"]
    frozen = hobson.encoder_binding(config["checkpoint"], config["base"])
    require(proof["encoder_sha256"] == frozen and proof["frozen_torso_adapter_files_unchanged"] is True
            and proof["released_head_file_unchanged"] is True and proof["head_parameters"] == 1053184,
            "Frozen encoder/head capacity proof failed")
    require(all(r["observed"]["feature"]["encoder_sha256"] == frozen for r in rows),
            "Native features belong to another encoder")
    require(proof["baseline_receipt_sha256"] == fingerprint(baseline), "Learning baseline receipt changed")
    prep_ids = {c["public"]["id"] for name, group in cohorts.items() if name != "pilot_heldout" for c in group}
    learning.validate_baseline(baseline, sha(Path(config["checkpoint"]) / "head.safetensors"), prep_ids)
    heldout_ids = sorted(c["public"]["id"] for c in cohorts["pilot_heldout"])
    require(sorted(baseline["planned_question_ids"]) == heldout_ids == proof["baseline_question_ids"], "Heldout baseline IDs changed")
    for partition in ("fit", "tune", "calibration"):
        ids = sorted(c["public"]["id"] for c in cohorts["pilot_" + partition])
        require(proof["partitions"][partition] == ids, "Learning partitions changed")
        material = []
        for row in rows:
            if row["partition"] != partition:
                continue
            feature = row["observed"]["feature"]
            _, tensor_sha = safe_tensors(feature["path"], {"decide": [2048], "options": [2, 2048]})
            material.append({k: row[k] for k in ("question_id", "partition", "criterion", "label")} | {
                "provenance_sha256": feature["provenance_sha256"], "tensor_sha256": tensor_sha})
        require(proof["cache_sha256"][partition] == fingerprint(material), "Learning used different feature tensors")
    require(set(proof["criteria"]) == {"answer"}, "Qualification criterion changed")
    result = proof["criteria"]["answer"]
    require(result["reload_verified"] is True and result["reload_max_logit_error"] == 0
            and result["physical_encoder_forwards"] == 0 and result["torso_loaded"] is False,
            "Head reload/frozen-torso execution proof missing")
    released_path, trained_path = Path(config["checkpoint"]) / "head.safetensors", run / "train/learned/answer-head.safetensors"
    released, before_sha = safe_tensors(released_path, HEAD_SHAPES)
    trained, after_sha = safe_tensors(trained_path, HEAD_SHAPES)
    require(sha(released_path) == proof["released_head_sha256"]
            and sha(trained_path) == result["trained_head_sha256"]
            and before_sha == proof["initial_head_tensor_sha256"]
            and after_sha == result["trained_head_tensor_sha256"] and before_sha != after_sha,
            "Unchanged or incompatible trained head bytes")
    delta = math.sqrt(sum(sum((a - b) ** 2 for a, b in zip(released[n][2], trained[n][2])) for n in released))
    require(delta > 0 and math.isclose(delta, result["head_delta_l2"], rel_tol=1e-5, abs_tol=1e-8), "Head delta does not match weights")
    history = result["history"]
    grid = [(lr, epoch) for lr in (1e-5, 5e-5) for epoch in (1, 2, 3)]
    require([(h["learning_rate"], h["epoch"]) for h in history] == grid
            and all(h["optimizer_steps"] == 3 and math.isfinite(h["tune_nll"]) and h["nonzero_gradient_seen"] is True for h in history)
            and result["total_optimizer_steps"] == result["backward_passes"] == 18,
            "Actual bounded gradient/update history missing")
    best = min(history, key=lambda h: h["tune_nll"])
    require(result["selected"] == {"learning_rate": best["learning_rate"], "epoch": best["epoch"]}, "TUNE selection changed")
    calibration = [r for r in rows if r["partition"] == "calibration"]
    labels = [int(r["label"]) for r in calibration]
    require(result["calibration_labels"] == labels and result["calibration_questions"] == 10, "CAL labels/count changed")
    for partition in ("fit", "tune", "calibration"):
        counts = {str(label): sum(r["partition"] == partition and r["label"] == label for r in rows) for label in (0, 1)}
        require(result["label_counts"][partition] == counts, "Learning label counts changed")
        if partition == "fit":
            require(all(counts.values()), "FIT requires both labels")
    for kind in ("released", "trained"):
        logits = result["calibration_logits_false_true"][kind]
        fitted = learning.fit_temperature(logits, calibration)
        require(fitted == result[kind + "_calibration"], "Separate calibration temperature/NLL mismatch")
    for actual, row in zip(result["calibration_logits_false_true"]["released"], calibration):
        require(max(abs(a - b) for a, b in zip(actual, row["observed"]["raw_logits_false_true"])) <= 2e-4,
                "Released calibration does not match actual native feature scores")
    expected_work = {"fit": (18, 240), "tune": (6, 60), "reload_before": (1, 10),
                     "reload_after": (1, 10), "released_calibration": (1, 10), "trained_calibration": (1, 10)}
    require(result["physical_head_work"] == {k: {"head_forward_batches": v[0], "head_scored_states": v[1]}
            for k, v in expected_work.items()}, "Head work/grid accounting changed")
    calls = physical["train"]["responses"]
    require(len(calls) == 1 and calls[0]["completed"]["role"] == "head_training_and_calibration"
            and calls[0]["value"] == proof, "Training physical receipt missing")
    require(calls[0]["pending"]["request_sha256"] == fingerprint({
        "features_sha256": sha(run / "features/features.json"),
        "baseline_sha256": sha(run / "features/untouched-baseline.json")}), "Training request used different source receipts")
    require(aware(baseline["finished_at"]) <= aware(calls[0]["pending"]["at"]), "Weight updates preceded untouched baseline")
    return proof


def model_binding(config, proof):
    checkpoint, base = Path(config["checkpoint"]), Path(config["base"])
    frozen = proof["encoder_sha256"]
    base_name = "model.safetensors-00001-of-00001.safetensors"
    require(sha(checkpoint / "head.safetensors") == proof["released_head_sha256"], "Released head changed")
    require(sha(base / base_name) == frozen["base/" + base_name]
            and sha(checkpoint / "lora/adapter_model.safetensors") == frozen["checkpoint/lora/adapter_model.safetensors"],
            "Model binding changed")
    return {"released_head_sha256": proof["released_head_sha256"],
        "adapter_sha256": frozen["checkpoint/lora/adapter_model.safetensors"],
        "base_weights_sha256": frozen["base/" + base_name],
        "upstream_sha256": {name: frozen["upstream/" + name] for name in
                            ("modeling.py", "infer.py", "prompting.py", "schema.py")},
        "head_parameters": 1053184}


def timings(run, plan, state, physical):
    phases = {}
    for phase in PHASES:
        value = read(run / phase / "phase-timing.json")
        require(all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
                    for k in ("wall_seconds", "active_seconds", "intentional_idle_seconds")), "Invalid phase timing")
        require(value["active_seconds"] + value["intentional_idle_seconds"] <= value["wall_seconds"] + 2,
                "Active/idle accounting exceeds its containing phase wall")
        stage = state["stages"][phase]
        completion = read(run / phase / "completion.json")
        require(aware(stage["started_at"]) <= aware(completion["finished_at"]) <= aware(stage["finished_at"]),
                "Phase completion does not belong to its current supervisor dispatch")
        for call in physical[phase]["responses"]:
            require(aware(call["pending"]["at"]) >= aware(stage["started_at"])
                    and aware(call["completed"]["at"]) <= aware(stage["finished_at"]),
                    "Physical call lies outside its phase owner")
            if call["pending"]["role"] != "writer_unload":
                require(aware(call["pending"]["at"]) <= aware(plan["deadline"]), "New work began after the cutoff")
        value["receipt_sha256"] = sha(run / phase / "phase-timing.json")
        if phase != "train":
            runtime = read(run / phase / "runtime-terminal.json")
            value["runtime_startup_and_switch"] = runtime["timings"]
            require(runtime["active_seconds"] == value["active_seconds"]
                    and runtime["intentional_idle_seconds"] == value["intentional_idle_seconds"],
                    "Runtime/phase active and intentional idle accounting differ")
        phases[phase] = value
    require(type(state.get("supervisor_wall_seconds")) in (float, int)
            and math.isfinite(state["supervisor_wall_seconds"]) and state["supervisor_wall_seconds"] >= 0,
            "Supervisor wall accounting missing")
    return {"supervisor_wall_seconds": state["supervisor_wall_seconds"], "phases": phases,
        "scope": "Supervisor wall contains workers/admission/idle; phase wall contains active/idle; startup/switch and physical-call wall are overlapping diagnostics. Never add these scopes together. Training call wall includes its internal rests."}


def audit_run(run, *, expected_plan_sha256=EXPECTED_PLAN_SHA256):
    """Return readiness only after every terminal integrity/cleanup check passes."""
    run = Path(run).resolve()
    plan, state = terminal_owner(run, expected_plan_sha256)
    source_before = fingerprint(state["source_sha256"])
    with frozen_helpers(run) as helpers:
        protocol = helpers["haltiq_supervisor.plan"]
        protocol.verify_frozen(plan, state, run)
        protocol.qualification(plan)
        for stage in plan["stages"]:
            helpers["haltiq_supervisor.engine"].verify_completion(stage, state, read(run / stage["id"] / "completion.json"))
        config = read(plan["qualification_protocol"]["path"])
        freeze, cohorts = helpers["haltiq_tau_hotpot.worker"].qualification_protocol(config)
        require([s["planned_units"] for s in plan["stages"]] == [64, 64, 1, 16], "Qualification counts differ")
        physical = {phase: journal(run / phase / "physical") for phase in PHASES}
        allowed_roles = {"collect": {"retrieval", "writer_load", "writer", "writer_unload"},
            "features": {"retrieval", "native_startup", "native_answer"},
            "train": {"head_training_and_calibration"},
            "compare": {"retrieval", "writer_load", "writer", "reviewer", "writer_unload", "native_startup", "native_answer", "cpu_head"}}
        require(all(set(physical[p]["by_role"]) <= allowed_roles[p] for p in PHASES), "Undeclared physical model work")
        collect = read(run / "collect/episodes.json")
        cases = cohorts["pilot_heldout"] + cohorts["pilot_fit"] + cohorts["pilot_tune"] + cohorts["pilot_calibration"]
        require([r["question_id"] for r in collect] == [c["public"]["id"] for c in cases], "Initial cohort/order/denominator changed")
        for row, case in zip(collect, cases):
            require(row == read(run / "collect/episodes" / (row["question_id"] + ".json")), "Initial row file changed")
            native_episode(row["episode"], case, row["initial_retrieval"]["indices"], helpers, arm="fixed1", max_drafts=1)
            require(row["episode"]["status"] == "completed", "All64 initial submissions are required for this completed training route")
        require(physical["collect"]["by_role"].get("writer") == 64, "Initial writer physical calls differ")
        writer_responses = [call["value"] for call in physical["collect"]["responses"] if call["completed"]["role"] == "writer"]
        require(all(raw.get("done") is True and raw.get("done_reason") != "length" for raw in writer_responses),
                "Initial writer did not actually finish")
        require(Counter(fingerprint(json.loads(raw["message"]["content"])) for raw in writer_responses)
                == Counter(fingerprint(row["episode"]["submitted"]) for row in collect),
                "Initial submissions differ from actual writer responses")
        features, baseline = read(run / "features/features.json"), read(run / "features/untouched-baseline.json")
        feature_rows(run, cohorts, features, baseline, helpers)
        require(physical["features"]["by_role"].get("native_answer") == 64, "Native feature forwards differ")
        native_calls = [v for v in physical["features"]["responses"] if v["completed"]["role"] == "native_answer"]
        require([v["value"] for v in native_calls] == [r["observed"] for r in features],
                "Native first-baseline/all-feature call order changed")
        for row, call in zip(features, native_calls):
            case = next(c for c in cases if c["public"]["id"] == row["question_id"])
            draft = next(r["episode"]["submitted"] for r in collect if r["question_id"] == row["question_id"])
            evidence = "\n".join(case["public"]["context"][i][0] + ": " + " ".join(case["public"]["context"][i][1])
                for i in row["observed"]["retrieval_indices"])
            public_state = f"Question: {case['public']['question']}\nEvidence: {evidence}\nDraft answer: {draft['answer']}"
            require(call["pending"]["request_sha256"] == fingerprint({"state": public_state,
                "criterion": "answer", "head": "released", "temperature": None}),
                "Untouched native feature request used another state/head")
        proof = learning_proof(run, config, cohorts, features, baseline, physical, helpers)
        episodes = read(run / "compare/episodes.json")
        heldout = cohorts["pilot_heldout"]
        arms = freeze["pilot"]["arms"]
        planned_pairs = [(c["public"]["id"], arm) for c in heldout for arm in arms]
        require([(e["question_id"], e["arm"]) for e in episodes] == planned_pairs, "Missing/duplicate/foreign paired episode")
        saved = {r["question_id"]: r for r in collect}
        initial, gate_results = {}, []
        for case in heldout:
            qid = case["public"]["id"]
            initial[qid] = saved[qid]["episode"]["submitted"]
        for episode in episodes:
            qid, arm = episode["question_id"], episode["arm"]
            case = next(c for c in heldout if c["public"]["id"] == qid)
            require(episode == read(run / "compare/episodes" / f"{qid}-{arm}.json"), "Paired row file changed")
            require(episode["shared_initial_draft_sha256"] == fingerprint(initial[qid])
                    and episode["shared_initial_physical_calls"] == 0, "Shared initial-draft cost/provenance mismatch")
            native_episode(episode, case, saved[qid]["initial_retrieval"]["indices"], helpers, arm=arm, max_drafts=3)
            drafts = [e for e in episode["trace"] if e["event"] == "draft"]
            require(drafts and drafts[0].get("draft") == initial[qid], "Arms did not share actual initial draft")
            gate_results += [e["score"] for e in episode["trace"] if e["event"] == "gate" and e["status"] == "completed"]
        comparison = helpers["haltiq_tau_hotpot.comparison"].analyse(episodes, heldout, arms, initial_drafts=initial)
        require(comparison == read(run / "compare/comparison.json"), "Common metrics/paired intervals differ")
        for phase, completed in (("collect", 64), ("features", 64), ("train", 1),
                                 ("compare", sum(e["status"] == "completed" for e in episodes))):
            receipt = read(run / phase / "completion.json")
            require(receipt["completed_units"] == completed and receipt["not_run_units"] == 0
                    and receipt["failed_units"] == receipt["planned_units"] - completed,
                    "Physical/native outcome counts differ from phase completion")
        compare_roles = physical["compare"]["by_role"]
        require(compare_roles.get("writer", 0) == sum(e["counts"]["answer_drafts"] - 1 for e in episodes)
                and compare_roles.get("reviewer", 0) == sum(e["counts"]["reviewer_calls"] for e in episodes),
                "Shared initial/new revision/reviewer physical counts differ")
        expected_writers = [event["draft"] for episode in episodes for event in episode["trace"]
            if event["event"] == "draft" and event["number"] > 1 and event["status"] == "completed"]
        actual_writers = []
        for call in physical["compare"]["responses"]:
            if call["completed"]["role"] == "writer":
                raw = call["value"]
                if raw.get("done") and raw.get("done_reason") != "length":
                    try:
                        value = json.loads(raw["message"]["content"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    if type(value) is dict and set(value) == {"answer", "supporting_facts"}:
                        # A syntactically valid response may still have invalid
                        # citations; those remain explicit failed draft events.
                        actual_writers.append(value)
        require(not (Counter(map(fingerprint, expected_writers)) - Counter(map(fingerprint, actual_writers))),
                "A completed revision has no matching physical writer response")
        expected_reviews = [event["feedback"] for episode in episodes for event in episode["trace"]
                            if event["event"] == "review" and event["status"] == "completed"]
        actual_reviews = [call["value"].get("message", {}).get("content") for call in physical["compare"]["responses"]
                          if call["completed"]["role"] == "reviewer"]
        require(not (Counter(expected_reviews) - Counter(actual_reviews)), "Review feedback has no matching physical response")
        actual_gates = [r["value"] for r in physical["compare"]["responses"]
                        if r["completed"]["role"] in {"cpu_head", "native_answer"}]
        require(Counter(map(fingerprint, gate_results)) == Counter(map(fingerprint, actual_gates)),
                "Recorded route gates do not match completed physical head/native results")
        for result in actual_gates:
            probability(result, native=result["physical_forwards"] == 1)
        for phase in ("collect", "features", "compare"):
            terminal = read(run / phase / "runtime-terminal.json")
            require(terminal["cleanup"]["verified"] is True and terminal["cleanup"]["private_port_closed"] is True
                    and terminal["cleanup"]["gpu_lease_released"] is True
                    and terminal["cleanup"]["owned_server_exited"] is True
                    and not terminal["cleanup"]["errors"]
                    and len(terminal["cleanup"]["native_unloads"]) == physical[phase]["by_role"].get("native_startup", 0)
                    and all(v["allocated_bytes"] == 0 for v in terminal["cleanup"]["native_unloads"])
                    and terminal["calls"]["pending_or_uncertain"] is False
                    and terminal["calls"]["completed"] == physical[phase]["calls"], "Runtime terminal cleanup/call reconciliation failed")
        helpers["haltiq_supervisor.engine"].check_handoff(plan)
        os_handoff_at = datetime.now(timezone.utc).isoformat()
        protocol.verify_frozen(plan, state, run)
        require(fingerprint(state["source_sha256"]) == source_before, "Source identity changed during audit")
        timing = timings(run, plan, state, physical)
        output_hashes = {str(p.relative_to(run)): sha(p) for p in (run / "collect/episodes.json",
            run / "features/features.json", run / "features/untouched-baseline.json", run / "train/learning.json",
            run / "train/learned/answer-head.safetensors", run / "compare/episodes.json", run / "compare/comparison.json")}
        return {"status": "PASS_QUALIFIED_ENGINEERING_PILOT", "qualified": True,
            "terminal": True, "cleanup_verified": True, "pending_or_uncertain": False,
            "ready": {"combined_pilot": True},
            "scope": "Four heldout engineering questions; no population improvement or official Tau2 leaderboard claim",
            "run_id": state["run_id"], "plan_sha256": state["plan_sha256"],
            "supervisor_sha256": sha(run / "supervisor.json"), "source_manifest_sha256": source_before,
            "source_sha256": dict(state["source_sha256"]), "model_binding": model_binding(config, proof),
            "pinned_inputs_sha256": {s["path"]: s["sha256"] for s in plan["inputs"]},
            "output_sha256": output_hashes, "physical_calls": {p: {k: v for k, v in j.items() if k != "responses"} for p, j in physical.items()},
            "timing": timing,
            "planned_units": [64, 64, 1, 16], "heldout_questions": 4,
            "comparison": comparison, "trained_head_sha256": proof["criteria"]["answer"]["trained_head_sha256"],
            "untouched_baseline_sha256": sha(run / "features/untouched-baseline.json"),
            "actual_os_handoff_verified": True, "os_handoff_verified_at": os_handoff_at,
            "all_native_episodes_replayed_without_model_calls": 80,
            "new_model_forwards": 0, "new_head_scores": 0, "torch_imported_by_audit": False,
            "long_study_started_by_audit": False}


def qualify(run, output, *, expected_plan_sha256=EXPECTED_PLAN_SHA256):
    """Write one NEW private receipt, including an honest blocking result."""
    run, output = Path(run).resolve(), Path(output).resolve()
    require(not output.exists() and output != run and run / "source" not in output.parents,
            "Qualification output must be a new receipt outside frozen source")
    try:
        value = audit_run(run, expected_plan_sha256=expected_plan_sha256)
    except Exception as error:
        value = {"status": "BLOCKED_NO_QUALIFICATION", "qualified": False, "ready": {}, "error_type": type(error).__name__,
                 "reason": str(error), "new_model_forwards": 0, "new_head_scores": 0,
                 "long_study_started_by_audit": False}
    value.update(schema="haltiq-tau-hotpot-qualified-pilot-v1", run=str(run),
                 reviewed_at=datetime.now(timezone.utc).isoformat(), auditor_source_sha256=sha(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        import os
        stream.write((json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", default=EXPECTED_PLAN_SHA256)
    args = parser.parse_args()
    result = qualify(args.run, args.output, expected_plan_sha256=args.expected_plan_sha256)
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": sha(args.output)}))
    raise SystemExit(0 if result["status"] == "PASS_QUALIFIED_ENGINEERING_PILOT" else 1)


if __name__ == "__main__":
    main()
