"""Hash-bound scheduling measurements and conservative whole-route forecasts.

No model loading, outcome-based selection, resource probing, process dispatch,
or adoption occurs here. The pilot remains a separate engineering experiment.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import math
from pathlib import Path

from haltiq_supervisor.plan import READINESS, satisfied
from .hobson import file_hash
from .journal import CallJournal, fingerprint
from .protocol import PHASES, pinned, read
from .qualification import journal, terminal_owner


def _spec(path):
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Existing real scheduling evidence file required")
    return {"path": str(path), "sha256": file_hash(path)}


def _checked(spec):
    if (not isinstance(spec, dict) or not Path(spec.get("path", "")).is_absolute()
            or not spec.get("sha256") or not satisfied(spec)):
        raise ValueError("Scheduling evidence is missing or changed")
    return spec


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


FORECAST_POLICY = {"method": "terminal_pilot_warm_role_maxima_cold_placement_full_window_proxy_v1",
    "max_native_window": 4096, "minimum_measured_native_tokens": 1024,
    "evidence_native_proxy_multiplier": 2, "safety_fraction": .25,
    "fixed_safety_seconds": 600, "minimum_local_overhead_per_operation_seconds": .1,
    "compilation_dispatch_reserve_seconds": 300,
    "known_terminal_failure_cache": True,
    "failure_cache_scope": "Exact payload/model only; revalidate committed raw terminal responses, including known failures; never reuse or retry uncertain calls",
    "scope": "Upper scheduling estimate with explicit unmeasured evidence proxy; no quality-based selection"}

def pilot_measurements(path):
    """Read only verified scheduling metadata; never inspect labels/quality to select N."""
    proof = read(path)
    if (proof.get("schema") != "haltiq-tau-hotpot-qualified-pilot-v1"
            or proof.get("status") != "PASS_QUALIFIED_ENGINEERING_PILOT"
            or any(proof.get(key) is not True for key in ("qualified", "terminal", "cleanup_verified", "actual_os_handoff_verified"))
            or proof.get("pending_or_uncertain") is not False or proof.get("planned_units") != [64, 64, 1, 16]):
        raise ValueError("A real terminal qualified four-phase pilot is required")
    run = Path(proof["run"]).resolve()
    _, state = terminal_owner(run, proof["plan_sha256"])
    if (proof.get("run_id") != state["run_id"] or proof.get("supervisor_sha256") != file_hash(run / "supervisor.json")
            or proof.get("source_sha256") != state["source_sha256"]):
        raise ValueError("Qualified pilot owner/source receipt changed")
    for input_path, expected in proof["pinned_inputs_sha256"].items():
        _checked({"path": input_path, "sha256": expected})
    for relative, expected in proof["output_sha256"].items():
        target = (run / relative).resolve()
        if run not in target.parents:
            raise ValueError("Pilot generated-output path escapes its run")
        _checked({"path": str(target), "sha256": expected})
    verification = [_spec(path), _spec(run / "plan.json"), _spec(run / "supervisor.json")]
    verification += [{"path": str(run / "source" / key), "sha256": value} for key, value in state["source_sha256"].items()]
    verification += [{"path": str(run / key), "sha256": value} for key, value in proof["output_sha256"].items()]
    verification += [{"path": key, "sha256": value} for key, value in proof["pinned_inputs_sha256"].items()]
    durations, warm, cold, native_rates, total_calls = {}, {}, {"writer": [], "native": []}, [], 0
    timing = proof["timing"]["phases"]
    phase_walls, startup, switch_residual, local_residual = {}, [], [], 0.
    journal_specs = []
    for phase in PHASES[:-1]:
        physical = journal(run / phase / "physical")
        declared = {key: value for key, value in physical.items() if key != "responses"}
        if declared != proof["physical_calls"][phase]:
            raise ValueError("Pilot physical timing/journal binding changed")
        phase_timing = pinned(run / phase / "phase-timing.json", timing[phase]["receipt_sha256"])
        if any(phase_timing.get(key) != timing[phase].get(key)
                for key in ("wall_seconds", "active_seconds", "intentional_idle_seconds")):
            raise ValueError("Pilot phase timing changed")
        phase_walls[phase] = phase_timing["wall_seconds"]
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in phase_timing.values())
                or phase_timing["active_seconds"] + phase_timing["intentional_idle_seconds"] > phase_timing["wall_seconds"] + 2):
            raise ValueError("Pilot timing scopes are invalid or overlap beyond their containing wall")
        first_writer = first_native = False
        for call in physical["responses"]:
            event, response = call["completed"], call["value"]
            role, seconds = event["role"], event["elapsed_seconds"]
            verification.append({"path": str(run / phase / "physical" / event["response"]), "sha256": event["response_sha256"]})
            durations.setdefault(role, []).append(seconds)
            if role == "writer_load":
                first_writer = True
            elif role == "native_startup":
                first_native = True
            elif role in {"writer", "reviewer"}:
                if first_writer:
                    cold["writer"].append(seconds)
                    first_writer = False
                else:
                    warm.setdefault(role, []).append(seconds)
            if role == "native_answer":
                tokens = response.get("input_tokens")
                if type(tokens) is not int or not 0 < tokens <= 4096:
                    raise ValueError("Actual native token-count metadata is required")
                native_rates.append({"input_tokens": tokens, "seconds": seconds, "cold_after_placement": first_native})
                if first_native:
                    cold["native"].append(seconds)
                    first_native = False
                else:
                    warm.setdefault(role, []).append(seconds)
        if phase != "train":
            terminal = read(run / phase / "runtime-terminal.json")
            runtime = timing[phase]["runtime_startup_and_switch"]
            if terminal["timings"] != runtime:
                raise ValueError("Pilot startup/switch timing binding changed")
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in runtime.values()):
                raise ValueError("Invalid measured runtime startup/switch timing")
            startup.append(runtime["startup_seconds"])
            contained = sum(v["completed"]["elapsed_seconds"] for v in physical["responses"]
                            if v["completed"]["role"] in {"writer_load", "writer_unload", "native_startup"})
            switch_residual.append(max(0, runtime["switch_seconds"] - contained))
            local_residual += max(0, phase_timing["wall_seconds"]
                - phase_timing["active_seconds"] - phase_timing["intentional_idle_seconds"])
        total_calls += physical["calls"]
        journal_specs.append(_spec(run / phase / "physical/calls.jsonl"))
        verification += [_spec(run / phase / "physical/calls.jsonl"), _spec(run / phase / "phase-timing.json")]
        if phase != "train":
            verification.append(_spec(run / phase / "runtime-terminal.json"))
    required = {"writer", "reviewer", "retrieval", "native_answer", "cpu_head", "writer_load", "native_startup", "writer_unload", "head_training_and_calibration"}
    if required - durations.keys() or not native_rates or not warm.get("native_answer") or not warm.get("writer"):
        raise ValueError("Pilot lacks required real cold/warm role measurements")
    if len(durations["writer"]) < 64 or len(native_rates) < 64:
        raise ValueError("Full planned pilot writer/native measurements are required")
    return {"role_max_seconds": {key: max(values) for key, values in durations.items()},
        "steady_role_max_seconds": {key: max(warm.get(key, values)) for key, values in durations.items()},
        "cold_placement_max_seconds": {key: max(values, default=0) for key, values in cold.items()},
        "role_counts": {key: len(values) for key, values in durations.items()},
        "native_samples": native_rates, "phase_wall_seconds": phase_walls, "physical_calls": total_calls,
        "runtime_startup_max_seconds": max(startup), "switch_residual_max_seconds": max(switch_residual),
        "local_residual_seconds": local_residual,
        "pilot_source_sha256": fingerprint(state["source_sha256"]), "journals": journal_specs,
        "qualification": _spec(path), "verification_inputs": verification, "quality_values_used": False}


def runtime_probe_measurements(path, pilot_path):
    """Accept only an independently qualified actual six-point/cache probe."""
    proof, pilot = read(path), read(pilot_path)
    if (proof.get("schema") != "haltiq-qualified-runtime-probe-v1"
            or proof.get("status") != "PASS_QUALIFIED_RUNTIME_PROBE"
            or any(proof.get(key) is not True for key in ("qualified", "terminal", "cleanup_verified"))
            or proof.get("pending_or_uncertain") is not False
            or proof.get("pilot_proof_sha256") != file_hash(pilot_path)
            or proof.get("model_binding") != pilot["model_binding"]):
        raise ValueError("Actual qualified same-model runtime probe is required")
    run = Path(proof["run"]).resolve()
    plan, state = read(run / "plan.json"), read(run / "supervisor.json")
    if (file_hash(run / "plan.json") != proof.get("plan_sha256") or state.get("plan_sha256") != proof["plan_sha256"]
            or file_hash(run / "supervisor.json") != proof.get("supervisor_sha256")
            or state.get("status") != "completed" or set(state.get("stages", {})) != {"probe"}
            or plan.get("purpose") != "runtime_probe" or [s["id"] for s in plan["stages"]] != ["probe"]
            or file_hash(plan["python"]) != state.get("python_sha256")):
        raise ValueError("Runtime probe owner/terminal/interpreter binding changed")
    actual_source = {p.relative_to(run / "source").as_posix(): file_hash(p)
                     for p in (run / "source").rglob("*") if p.is_file()}
    if actual_source != state.get("source_sha256") or actual_source != proof.get("source_sha256") or actual_source != plan.get("reviewed_source_sha256"):
        raise ValueError("Runtime probe frozen source changed")
    stage = state["stages"]["probe"]
    expected = {str(Path(s["path"].replace("{run}", str(run))).resolve()) for s in plan["stages"][0]["outputs"]}
    if (stage.get("status") != "completed" or stage.get("exit_code") != 0
            or stage.get("owned_tree_clear") is not True or stage.get("os_handoff_verified") is not True
            or set(stage.get("output_sha256", {})) != expected):
        raise ValueError("Runtime probe requires exact completed outputs and owned-runtime cleanup")
    from haltiq_supervisor.engine import verify_completion
    if plan["stages"][0].get("planned_units") != 22:
        raise ValueError("Runtime probe scope must remain exactly22planned units")
    verify_completion(plan["stages"][0], state, read(run / "probe/completion.json"))
    verification = [_spec(path), _spec(run / "plan.json"), _spec(run / "supervisor.json")]
    if not plan.get("inputs") or set(plan.get("readiness", {})) != READINESS:
        raise ValueError("Runtime probe needs its original input and five readiness contracts")
    original_inputs = list(plan["inputs"])
    for specs in plan["readiness"].values():
        if not isinstance(specs, list) or not specs or any(not spec.get("json_equals") for spec in specs):
            raise ValueError("Runtime probe readiness contracts must remain explicit")
        original_inputs.extend(specs)
    for spec in original_inputs:
        verification.append(_checked(spec))
    original_hashes = {str(Path(spec["path"]).resolve()): spec["sha256"] for spec in original_inputs}
    proof_hashes = proof.get("pinned_inputs_sha256", {})
    if {str(Path(key).resolve()): value for key, value in proof_hashes.items()} != original_hashes:
        raise ValueError("Qualified probe original-input binding differs from its adopted plan")
    for input_path, expected_hash in proof_hashes.items():
        verification.append(_checked({"path": input_path, "sha256": expected_hash}))
    maps = [stage["output_sha256"], {str(run / key): value for key, value in proof["output_sha256"].items()},
            {str(run / "source" / key): value for key, value in actual_source.items()}]
    for mapping in maps:
        for name, expected_hash in mapping.items():
            target = Path(name).resolve()
            if run not in target.parents:
                raise ValueError("Probe evidence path escapes its frozen run")
            verification.append(_checked({"path": str(target), "sha256": expected_hash}))
    value, directory = read(run / "probe/probe.json"), run / "probe/physical"
    if (value.get("schema") != "haltiq-runtime-probe-v1" or value.get("status") != "PASS_RUNTIME_PROBE"
            or value.get("engineering_only") is not True or value.get("benchmark_quality_claim") is not False
            or value.get("new_learning") is not False or value.get("long_study_started") is not False
            or value.get("wrapper_plan_sha256") != proof["plan_sha256"]
            or value.get("source_sha256") != actual_source
            or value.get("pilot_proof_sha256") != file_hash(pilot_path)
            or value.get("released_head_sha256") != pilot["model_binding"]["released_head_sha256"]
            or value.get("trained_head_sha256") != pilot["trained_head_sha256"]):
        raise ValueError("Probe scope/source/pilot binding changed")
    if not (directory / "calls.jsonl").is_file():
        raise ValueError("Runtime probe physical journal is missing")
    physical = CallJournal(directory)
    stats = physical.verify()
    expected_roles = {"native_startup": 1, "native_answer": 3, "native_evidence": 3,
        "cached_head_answer": 6, "cached_head_evidence": 6, "writer_load": 1, "writer": 1, "reviewer": 1, "writer_unload": 1}
    if stats != value.get("calls") or stats["by_role"] != expected_roles or stats["exact_response_reuses"] != 2 or stats["pending_or_uncertain"]:
        raise ValueError("Runtime probe physical calls/reuses do not reconcile")
    responses, seconds = {}, {}
    for event in physical.records:
        if event["event"] != "completed":
            continue
        elapsed = event["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Runtime probe duration is invalid")
        response = directory / event["response"]
        verification.append(_spec(response))
        responses.setdefault(event["role"], []).append(read(response))
        seconds.setdefault(event["role"], []).append(elapsed)
    verification.append(_spec(directory / "calls.jsonl"))
    points = value.get("points", [])
    expected_points = [(criterion, target) for target in (512, 2048, 4096) for criterion in ("answer", "evidence")]
    if [(r["criterion"], r["target_tokens"]) for r in points] != expected_points:
        raise ValueError("Exactly three length points per criterion are required")
    native, cached = {c: [] for c in ("answer", "evidence")}, {c: [] for c in ("answer", "evidence")}
    for index, row in enumerate(points):
        c, target, observed = row["criterion"], row["target_tokens"], row["actual"]
        if (type(row["input_tokens"]) is not int or not target - 8 <= row["input_tokens"] <= target
                or observed["input_tokens"] != row["input_tokens"] or observed["physical_forwards"] != 1
                or observed["head_scores"] != 2 or observed["temperature"] != 1.
                or row["cold_first_forward_after_placement"] is not (index == 0)
                or row["public"].get("type") != "synthetic_control"
                or row["cached_released"]["raw_logits_false_true"] != observed["raw_logits_false_true"]
                or any(v["physical_forwards"] != 0 or v["head_scores"] != 1 or v["temperature"] != 1.
                       for v in (row["cached_released"], row["cached_trained"]))):
            raise ValueError("Native/cache probe point does not preserve its declared physical scope")
        native[c].append(observed)
        cached[c] += [row["cached_released"], row["cached_trained"]]
    for c in native:
        if (Counter(map(fingerprint, native[c])) != Counter(map(fingerprint, responses["native_" + c]))
                or Counter(map(fingerprint, cached[c])) != Counter(map(fingerprint, responses["cached_head_" + c]))):
            raise ValueError("Probe points differ from durable native/cached responses")
    profile = {}
    cold_by_response = {fingerprint(row["actual"]): row["cold_first_forward_after_placement"] for row in points}
    for c in native:
        profile[c] = [{"input_tokens": row["input_tokens"], "seconds": elapsed,
            "cold_after_placement": cold_by_response[fingerprint(row)]}
            for row, elapsed in zip(responses["native_" + c], seconds["native_" + c])]
    terminal = read(run / "probe/runtime-terminal.json")
    if (terminal.get("cleanup", {}).get("verified") is not True or terminal["calls"] != stats
            or not _finite(terminal["timings"]["startup_seconds"])):
        raise ValueError("Runtime probe cleanup/physical terminal receipt changed")
    return {"qualification": _spec(path), "verification_inputs": verification, "native_by_criterion": profile,
        "cpu_head_max_seconds": max(seconds["cached_head_answer"] + seconds["cached_head_evidence"]),
        "role_max_seconds": {k: max(v) for k, v in seconds.items()},
        "cold_placement_max_seconds": {"writer": seconds["writer"][0],
            "native": next(row["seconds"] for rows in profile.values() for row in rows if row["cold_after_placement"])},
        "startup_seconds": terminal["timings"]["startup_seconds"], "quality_values_used": False}


def forecast_route(measurements, *, policy=None, available_seconds=46800, runtime_probe=None):
    """Price the complete route, including rests and both exact-cache policies."""
    policy = policy or FORECAST_POLICY
    if policy != FORECAST_POLICY:
        raise ValueError("Forecast policy must match the predeclared canonical protocol")
    if not _finite(available_seconds) or available_seconds > 46800:
        raise ValueError("Remaining model window must be finite and cannot exceed13hours")
    rates = measurements["steady_role_max_seconds"]
    if (any(not _finite(value) for value in rates.values())
            or any(type(r["input_tokens"]) is not int or not 0 < r["input_tokens"] <= 4096
                or not _finite(r["seconds"]) or type(r["cold_after_placement"]) is not bool
                for r in measurements["native_samples"])
            or any(not _finite(measurements[key]) for key in
                ("runtime_startup_max_seconds", "switch_residual_max_seconds", "local_residual_seconds"))):
        raise ValueError("Measured durations must be finite and nonnegative")
    # Full-window quadratic extrapolation is deliberately conservative. Evidence
    # has not been measured in the answer-only pilot: keep its explicit 2x proxy.
    native = max(row["seconds"] * max(1, (4096 / row["input_tokens"]) ** 2)
                 for row in measurements["native_samples"] if not row["cold_after_placement"])
    sufficient = max(row["input_tokens"] for row in measurements["native_samples"]
                     if not row["cold_after_placement"]) >= 1024
    evidence, evidence_proxy = native * 2, True
    if runtime_probe is not None:
        measured_native = runtime_probe["native_by_criterion"]
        native = max(v["seconds"] for v in measured_native["answer"] if not v["cold_after_placement"])
        evidence = max(v["seconds"] for v in measured_native["evidence"])
        sufficient, evidence_proxy = True, False
        rates = {key: max(value, runtime_probe["role_max_seconds"].get(key, 0))
                 if key in {"writer_load", "native_startup", "writer_unload", "reviewer"} else value
                 for key, value in rates.items()}
        rates["cpu_head"] = max(rates["cpu_head"], runtime_probe["cpu_head_max_seconds"])
    overhead = max(.1, measurements["local_residual_seconds"] / max(1, measurements["physical_calls"]))
    def duty(seconds):
        return seconds + max(5, seconds)
    candidates = []
    for n in (20, 40, 60):
        # These prefix bounds require durable response reuse for known terminal
        # validation failures as well as successes. Each arm keeps the failure;
        # an identical later callback cannot resample it into a new trajectory.
        counts = {"writer": 620 + 10 * n, "reviewer": 9 * n, "retrieval": 1240 + 11 * n,
            "native_answer": 620 + 10 * n, "native_evidence": 620 + 10 * n,
            "cpu_head": 20 * n, "writer_load": 1 + n, "native_startup": 1 + n,
            "writer_unload": 1 + n}
        seconds = {role: count * duty(evidence if role == "native_evidence" else native
                    if role.startswith("native_") and role != "native_startup" else rates[role])
                   for role, count in counts.items()}
        # Training wall already contains its actual internal rests. Scale the
        # whole pilot training phase by ten; never add its overlapping call wall.
        seconds["training_and_answer_calibration"] = measurements["phase_wall_seconds"]["train"] * 10
        seconds["evidence_calibration_cpu"] = max(5, rates["head_training_and_calibration"])
        seconds["runtime_initialization"] = 3 * duty(max(measurements["runtime_startup_max_seconds"],
            runtime_probe["startup_seconds"] if runtime_probe else 0))
        seconds["switch_non_call_overhead"] = (2 + 2 * n) * 2 * measurements["switch_residual_max_seconds"]
        seconds["cold_first_forward_after_each_placement"] = sum(
            (1 + n) * max(measurements["cold_placement_max_seconds"][family],
                runtime_probe["cold_placement_max_seconds"][family] if runtime_probe else 0) * 2
            for family in ("writer", "native"))
        # Logical gate/callback/cache checks and fsync/Tau work remain physical
        # CPU/wall expense even when an exact cache avoids another model call.
        logical = sum(counts.values()) + 68 * n + 40 * n
        seconds["local_control_and_cache_overhead"] = logical * overhead
        subtotal = sum(seconds.values())
        total = subtotal * 1.25 + 600
        candidates.append({"n": n, "counts": counts, "components_seconds": seconds,
            "before_safety_seconds": subtotal, "safety_seconds": subtotal * .25 + 600,
            "model_route_seconds": total, "fits_model_budget": sufficient and total + 300 <= available_seconds})
    eligible = [row["n"] for row in candidates if row["fits_model_budget"]]
    return {"schema": "haltiq-main-whole-route-forecast-v1", "answer_blind": True,
        "quality_values_used": False, "fresh_scoring_started": False,
        "basis": "measured_throughput_and_resources", "whole_route_fits": bool(eligible),
        "selected_n": max(eligible) if eligible else None, "policy": deepcopy(policy),
        "measurements": measurements, "candidates": candidates, "model_budget_seconds": 46800,
        "available_model_seconds_at_compile": available_seconds, "native_context_measurement_sufficient": sufficient,
        "runtime_probe": runtime_probe, "evidence_cost_is_unmeasured_proxy": evidence_proxy,
        "measurement_blocker": None if sufficient else "Warm native samples below1024tokens need a separately reviewed full-window technical measurement",
        "finalization_reserve_seconds": 3600, "native_full_window_proxy_seconds": native,
        "unmeasured_components": (["evidence criterion uses explicit 2x full-window answer proxy"] if evidence_proxy else []) + [
            "future sustained resource pressure cannot be guaranteed by a forecast"],
        "cache_scope": "Exact complete payloads/features only; logical paths share physical work. No independent live-arm latency claim.",
        "overlapping_scopes_added": False}
