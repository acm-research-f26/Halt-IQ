"""Prepare a blocked study, then separately compile a reviewed timed plan.

Preparation/compilation never adopt a run, start a worker or load models. The
separate resource check uses read-only OS telemetry, including nvidia-smi. A
forecast is a scheduling estimate, never a quality or live-arm latency result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import socket
import time

from haltiq_supervisor.plan import PACKAGES, READINESS, inspect, satisfied, source_files
from .hobson import file_hash
from .journal import durable_write
from .protocol import MAIN_SETTINGS, PHASES, _cohort, load_main, pinned, read, settings_match
from .forecasting import FORECAST_POLICY, pilot_measurements, runtime_probe_measurements, forecast_route

RESOURCES = {"ram_gib": 7, "vram_gib": 3, "startup_ram_gib": 11,
    "startup_vram_gib": 9, "stability_seconds": 30, "minimum_rest_seconds": 5,
    "cpu_threads": 4, "gpu_owners": 1, "duty_cycle": .5, "admission_timeout_seconds": 1200}
RUNTIME_KEYS = ("checkpoint", "base", "reranker", "ollama", "ollama_models", "writer_digest",
    "gpu_lease", "study_lease", "tau_revision", "model_inputs", "runtime_modules")
COHORTS = (("main_development", 20, "development"), ("main_fit", 400, "fit"),
    ("main_tune", 100, "tune"), ("main_calibration", 100, "calibration"),
    ("main_fresh20", 20, "fresh"), ("main_fresh40", 40, "fresh"), ("main_fresh60", 60, "fresh"))
OUTPUTS = {"collect": ["episodes.json", "runtime-terminal.json", "physical/calls.jsonl"],
    "features": ["features.json", "untouched-baseline.json", "runtime-terminal.json", "physical/calls.jsonl"],
    "train": ["learning.json", "learned/answer-head.safetensors", "learned/learning-proof.json", "physical/calls.jsonl"],
    "compare": ["episodes.json", "initial-drafts.json", "comparison.json", "runtime-terminal.json", "physical/calls.jsonl"],
    "audit": ["audit.json"]}


def _now(value=None):
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Explicit-offset clock required")
    return value.astimezone(timezone.utc)


def _aware(value):
    return _now(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _spec(path):
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("A real, existing pinned file is required")
    return {"path": str(path), "sha256": file_hash(path)}


def _checked(spec):
    if (not isinstance(spec, dict) or not isinstance(spec.get("path"), str)
            or not Path(spec["path"]).is_absolute() or not spec.get("sha256")
            or not satisfied(spec)):
        raise ValueError("A declared input or readiness receipt is missing or changed")
    return spec


def _new_directory(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Preparation/compilation output must be a NEW external directory")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _data(path):
    freeze = read(path)
    if freeze.get("schema") != "tau2-hotpot-main-v1":
        raise ValueError("Prepared main-study data schema required")
    settings_match(freeze.get("main"))
    selection = freeze.get("selection", {})
    if (selection.get("answer_blind") is not True or any(selection.get(key) is not False
            for key in ("quality_filtering", "length_filtering", "replacement_allowed"))):
        raise ValueError("Main data must remain answer-blind, unfiltered and without replacements")
    seen_ids, seen_questions, previous = set(), set(), {}
    for name, count, partition in COHORTS:
        rows, ids, questions = _cohort(freeze["files"][name], count, partition)
        if seen_ids & ids or seen_questions & questions:
            raise ValueError("Main development/learning/fresh partitions overlap")
        if partition == "fresh":
            current = {r["public"]["id"]: r for r in rows}
            if any(current.get(q) != row for q, row in previous.items()):
                raise ValueError("Fresh candidates must be exact nested balanced cohorts")
            previous = current
        else:
            seen_ids.update(ids)
            seen_questions.update(questions)
    return freeze


def _plan(config, source, python, *, n=None, start=None, review=None, inputs=()):
    stages = []
    for phase, units in zip(PHASES, (620, 1240, 1, None if n is None else n * 8, 1)):
        completion = "{run}/" + phase + "/completion.json"
        outputs = [{"path": "{run}/" + phase + "/" + name}
                   for name in OUTPUTS[phase] + ["phase-timing.json"]]
        outputs.append({"path": completion, "json_equals": {"terminal": True,
            "pending_or_uncertain": False, "cleanup_verified": True, "planned_units_accounted": True}})
        stages.append({"id": phase, "model_work": phase != "audit", "gpu": phase in {"collect", "features", "compare"},
            "planned_units": units, "completion_receipt": completion, "check_handoff": True,
            "argv": [str(python), "-B", "-m", "haltiq_tau_hotpot.main_worker", phase,
                "--config", str(config), "--output", "{run}/" + phase, "--previous", "{run}",
                "--deadline", "{deadline}", "--control", "{control}", "--allow-model-calls"],
            "outputs": outputs})
    runtime = read(config)
    return {"version": 1, "scope": "custom_tau2_hotpot", "purpose": "main_study",
        "status": "ready" if start is not None else "blocked_preparation_only",
        "source_root": str(source), "source_paths": list(PACKAGES) + ["pyproject.toml", "scripts/start_tau_hotpot_main.ps1"],
        "python": str(python), "start_time": start.isoformat() if start else None,
        "deadline": (start + timedelta(hours=14)).isoformat() if start else None,
        "hard_deadline": (start + timedelta(hours=14, minutes=5)).isoformat() if start else None,
        "model_deadline": (start + timedelta(hours=13)).isoformat() if start else None,
        "stop_grace_seconds": 60, "private_port": 11435,
        "gpu_lease": runtime["gpu_lease"], "study_lease": runtime["study_lease"],
        "resources": deepcopy(RESOURCES), "codex_argv": None, "local_only": True,
        "readiness": review["readiness"] if review else {},
        "reviewed_source_sha256": review["source_sha256"] if review else {},
        "main_protocol": _spec(config), "stages": stages, "inputs": list(inputs)}


def prepare_template(runtime_config, data_freeze, source_root, python, output):
    """Write inert canonical inputs; never invent readiness or an execution clock."""
    source, python = Path(source_root).resolve(), Path(python).resolve()
    if not source.is_dir() or not python.is_file():
        raise ValueError("Existing explicit source root and interpreter required")
    runtime_spec, freeze_spec = _spec(runtime_config), _spec(data_freeze)
    runtime, freeze = read(runtime_config), _data(data_freeze)
    if runtime.get("hosted_services") is not False or any(key not in runtime for key in RUNTIME_KEYS):
        raise ValueError("A complete local-only machine configuration is required")
    output = _new_directory(output, source)
    research = deepcopy(MAIN_SETTINGS)
    research["forecast_policy"] = deepcopy(FORECAST_POLICY)
    durable_write(output / "research-protocol.json", research)
    config = {key: deepcopy(runtime[key]) for key in RUNTIME_KEYS}
    config.update(purpose="main_study", preparation_status="blocked_no_launch", start_time=None,
        model_deadline=None, data_freeze=freeze_spec["path"], data_freeze_sha256=freeze_spec["sha256"],
        protocol=str(output / "research-protocol.json"), protocol_sha256=file_hash(output / "research-protocol.json"),
        combined_pilot_proof=None, combined_pilot_proof_sha256=None, fresh_n_freeze=None,
        fresh_n_freeze_sha256=None, native_feature_cache=True, response_cache=True,
        compare_process_layout="per_question",
        generation_profile=deepcopy(research["generation_profile"]),
        cap_fallback=research["cap_fallback"],
        hosted_services=False, new_downloads=False, long_study_started=False)
    durable_write(output / "main-config-template.json", config)
    plan = _plan(output / "main-config-template.json", source, python)
    durable_write(output / "plan-template.json", plan)
    manifest = {key: file_hash(path) for key, path in source_files(plan).items()}
    durable_write(output / "unreviewed-source-manifest.json", {"source_sha256": manifest,
        "reviewed": False, "scope": "Snapshot for an independent review, never a PASS receipt"})
    instructions = {"status": "blocked_no_launch", "model_calls": 0, "processes_started": 0,
        "adopted": False, "prepared_at": _now().isoformat(), "runtime_config": runtime_spec,
        "data_freeze": freeze_spec, "template": _spec(output / "main-config-template.json"),
        "protocol": _spec(output / "research-protocol.json"), "source_root": str(source), "python": str(python),
        "requires": ["Actual completed combined-pilot qualification; static tests are insufficient",
            "Independent complete source review with all five pinned readiness contracts",
            "Fresh stable resource/OS-lease/private-port proof", "Measured answer-blind conservative whole-route forecast",
            "Separate explicit timed compilation, final plan review, manual supervisor adoption and launch"],
        "operator_steps": ["Keep this template inert; do not fill readiness or timestamps by assumption.",
            "Obtain the reviewed source and actual terminal pilot/resource receipts, then run compile-main explicitly.",
            "Review the generated plan and N freeze. A compiled plan still has no adopted run or started process.",
            "Only an authorized operator may subsequently adopt and launch its frozen supervisor.",
            "Never retry started, failed or uncertain stages; preserve all planned questions and failures."],
        "fresh_candidates": {str(n): freeze["files"]["main_fresh" + str(n)]["sha256"] for n in (20, 40, 60)}}
    durable_write(output / "preparation.json", instructions)
    return instructions


def verify_review(path, plan, pilot_spec):
    value = read(path)
    actual = {key: file_hash(file) for key, file in source_files(plan).items()}
    if (value.get("schema") != "haltiq-main-source-review-v1" or value.get("status") != "PASS_REVIEWED_SOURCE"
            or Path(value.get("source_root", "")).resolve() != Path(plan["source_root"]).resolve()
            or value.get("source_sha256") != actual or value.get("exact_prefix_cache_validated") is not True
            or set(value.get("readiness", {})) != READINESS):
        raise ValueError("An independently reviewed complete current source/cache/readiness manifest is required")
    for specs in value["readiness"].values():
        if not isinstance(specs, list) or not specs:
            raise ValueError("Every readiness contract requires at least one pinned JSON receipt")
        for spec in specs:
            _checked(spec)
            if not spec.get("json_equals"):
                raise ValueError("Readiness needs explicit typed JSON assertions, not just file existence")
    if not any(spec.get("path") == pilot_spec["path"] and spec.get("sha256") == pilot_spec["sha256"]
            for spec in value["readiness"]["combined_pilot"]):
        raise ValueError("Combined-pilot readiness must bind the actual selected qualification proof")
    return value


def verify_resources(path, *, now=None, config=None):
    now, value = _now(now), read(path)
    if (value.get("schema") != "haltiq-main-resource-admission-v1"
            or value.get("os_gpu_lease_free") is not True or value.get("study_lease_free") is not True
            or value.get("private_port_closed") is not True or value.get("status") != "PASS_RESOURCE_ADMISSION"):
        raise ValueError("Explicit free OS GPU lease/private-port resource receipt required")
    samples = value.get("samples", [])
    if len(samples) != 4:
        raise ValueError("Exactly four fresh resource samples are required")
    times = [_aware(sample["at"]) for sample in samples]
    if (times != sorted(set(times)) or any((b - a).total_seconds() < 10 for a, b in zip(times, times[1:]))
            or (times[-1] - times[0]).total_seconds() < 30
            or not 0 <= (now - times[-1]).total_seconds() <= 600
            or times[0] < now - timedelta(seconds=900)):
        raise ValueError("Resource samples must span 30 seconds and remain fresh")
    for sample in samples:
        values = [sample.get(key) for key in ("available_ram_gib", "free_vram_gib", "gpu_temperature_c")]
        if (any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
                or values[0] < 11 or values[1] < 9 or not 0 <= values[2] < 80):
            raise ValueError("Startup resource reserves must hold throughout admission")
    if config is not None:
        origin = _checked(value.get("config"))
        measured_config = read(origin["path"])
        if (measured_config.get("hosted_services") is not False
                or any(measured_config.get(key) != config.get(key) for key in ("gpu_lease", "study_lease"))):
            raise ValueError("Resource receipt belongs to a different local runtime/OS lease")
    return value


def sample_resources(runtime_config, output, *, sampler=None, sleep=time.sleep, now=None, lease_factory=None):
    """One bounded, resource-only admission attempt; no model/worker dispatch."""
    from pc_strands.control import GpuLease, resource_snapshot
    config, output = read(runtime_config), Path(output).resolve()
    source = Path(__file__).resolve().parents[1]
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Resource receipt must be a NEW external file")
    if config.get("hosted_services") is not False:
        raise ValueError("Local-only machine configuration required")
    lease_factory, sampler = lease_factory or GpuLease, sampler or resource_snapshot
    clock = now or (lambda: datetime.now(timezone.utc))
    result = {"schema": "haltiq-main-resource-admission-v1", "status": "BLOCKED_RESOURCE_ADMISSION",
        "samples": [], "os_gpu_lease_free": False, "study_lease_free": False, "private_port_closed": False,
        "model_calls": 0, "model_processes_started": 0, "automatic_retry": False, "config": _spec(runtime_config),
        "telemetry_scope": "Read-only memory/GPU queries may invoke nvidia-smi; no study/model process starts"}
    def check_port():
        with socket.socket() as probe:
            probe.settimeout(.5)
            if probe.connect_ex(("127.0.0.1", 11435)) == 0:
                raise ValueError("Private11435 is already in use")
    try:
        check_port()
        with lease_factory(Path(config["study_lease"]), "main resource-only study admission"):
            with lease_factory(Path(config["gpu_lease"]), "main resource-only GPU admission"):
                for index in range(4):
                    if index:
                        sleep(10)
                    measured = sampler()
                    sample = {"at": _now(clock()).isoformat(),
                        "available_ram_gib": measured["available_ram_bytes"] / 1024**3,
                        "free_vram_gib": measured["gpu_free_mib"] / 1024,
                        "gpu_temperature_c": measured["gpu_temperature_c"]}
                    # Preserve unavailable/nonfinite telemetry as null, so a
                    # failed observation still has a valid durable JSON receipt.
                    sample = {key: value if key == "at" or type(value) in (int, float) and math.isfinite(value)
                              else None for key, value in sample.items()}
                    result["samples"].append(sample)
                    check_port()
        # Reacquire/release the actual OS locks after the observation; a lease
        # filename or an old process identifier is not an ownership proof.
        with lease_factory(Path(config["study_lease"]), "verify main study admission release"):
            with lease_factory(Path(config["gpu_lease"]), "verify main GPU admission release"):
                pass
        check_port()
        result.update(os_gpu_lease_free=True, study_lease_free=True, private_port_closed=True,
                      status="PASS_RESOURCE_ADMISSION")
        samples = result["samples"]
        elapsed = (_aware(samples[-1]["at"]) - _aware(samples[0]["at"])).total_seconds()
        if elapsed < 30 or any(type(v) not in (int, float) or not math.isfinite(v)
                for row in samples for key, v in row.items() if key != "at"):
            raise ValueError("Four finite resource observations must span at least30seconds")
        if any(row["available_ram_gib"] < 11 or row["free_vram_gib"] < 9
                or not 0 <= row["gpu_temperature_c"] < 80 for row in samples):
            raise ValueError("Resource reserve/temperature admission did not pass")
    except Exception as error:
        result.update(status="BLOCKED_RESOURCE_ADMISSION", error_type=type(error).__name__, reason=str(error))
    result["recorded_at"] = _now(clock()).isoformat()
    durable_write(output, result)
    return result


def compile_main(template, output, *, start_time, review_receipt, pilot_proof, resource_receipt, runtime_probe=None, now=None):
    """Compile files only, after actual receipts; never call supervisor.prepare/run."""
    fixed_clock = now
    clock = lambda: _now(fixed_clock)
    now, template = clock(), Path(template).resolve()
    prepared = read(template / "preparation.json")
    if prepared.get("status") != "blocked_no_launch" or prepared.get("adopted") is not False:
        raise ValueError("An inert prepared template is required")
    config = deepcopy(pinned(prepared["template"]["path"], prepared["template"]["sha256"]))
    research = pinned(prepared["protocol"]["path"], prepared["protocol"]["sha256"])
    settings_match(research)
    if research.get("forecast_policy") != FORECAST_POLICY:
        raise ValueError("Canonical forecast policy changed")
    _checked(prepared["runtime_config"])
    freeze = _data(config["data_freeze"])
    if file_hash(config["data_freeze"]) != config["data_freeze_sha256"]:
        raise ValueError("Prepared data-freeze binding changed")
    start = _aware(start_time)
    if not 0 <= (now - start).total_seconds() <= 300:
        raise ValueError("Timed compilation requires a truthful actual start within the last five minutes")
    source, python = Path(prepared["source_root"]), Path(prepared["python"])
    base = _plan(prepared["template"]["path"], source, python)
    pilot_spec = _spec(pilot_proof)
    review = verify_review(review_receipt, base, pilot_spec)
    verify_resources(resource_receipt, now=now, config=config)
    output = _new_directory(output, source)
    try:
        measured = pilot_measurements(pilot_proof)
        technical = runtime_probe_measurements(runtime_probe, pilot_proof) if runtime_probe is not None else None
        now = clock()
        verify_resources(resource_receipt, now=now, config=config)
        throughput = forecast_route(measured, policy=research["forecast_policy"],
                                    available_seconds=46800 - (now - start).total_seconds(), runtime_probe=technical)
    except (ValueError, KeyError, TypeError, OSError) as error:
        value = {"status": "blocked_no_launch", "reason": str(error), "model_calls": 0,
            "adopted": False, "processes_started": 0, "selected_n": None,
            "compiled_at": now.isoformat(), "planned_start_time": start.isoformat()}
        durable_write(output / "compilation.json", value)
        return value
    throughput.update(resource_proof=_spec(resource_receipt), source_review=_spec(review_receipt),
        compiled_at=now.isoformat(), planned_start_time=start.isoformat())
    durable_write(output / "throughput-forecast.json", throughput)
    if not throughput["whole_route_fits"]:
        value = {"status": "blocked_no_launch", "reason": "No complete N20/40/60 ten-draft route fits the 13-hour model budget",
            "forecast": _spec(output / "throughput-forecast.json"), "selected_n": None,
            "model_calls": 0, "adopted": False, "processes_started": 0}
        durable_write(output / "compilation.json", value)
        return value
    n = throughput["selected_n"]
    fresh = {"schema": "haltiq-main-fresh-n-freeze-v1", "n": n, "selected_n": n,
        "answer_blind": True, "fresh_scoring_started": False, "basis": "measured_throughput_and_resources",
        "data_freeze_sha256": config["data_freeze_sha256"], "protocol_sha256": config["protocol_sha256"],
        "dataset_sha256": freeze["files"]["main_fresh" + str(n)]["sha256"],
        "throughput_proof": str(output / "throughput-forecast.json"),
        "throughput_proof_sha256": file_hash(output / "throughput-forecast.json"), "frozen_at": now.isoformat()}
    durable_write(output / "fresh-n-freeze.json", fresh)
    config.update(start_time=start.isoformat(), model_deadline=(start + timedelta(hours=13)).isoformat(),
        preparation_status="compiled_not_adopted", combined_pilot_proof=pilot_spec["path"],
        combined_pilot_proof_sha256=pilot_spec["sha256"], fresh_n_freeze=str(output / "fresh-n-freeze.json"),
        fresh_n_freeze_sha256=file_hash(output / "fresh-n-freeze.json"))
    durable_write(output / "main-config.json", config)
    load_main(config)
    inputs = [_spec(output / name) for name in ("main-config.json", "fresh-n-freeze.json", "throughput-forecast.json")]
    inputs += [_checked({"path": config[key], "sha256": config[key + "_sha256"]}) for key in ("protocol", "data_freeze")]
    inputs += [_spec(review_receipt), pilot_spec, _spec(resource_receipt)]
    inputs += [_checked(spec) for spec in measured["verification_inputs"]]
    if technical is not None:
        inputs += [_checked(spec) for spec in technical["verification_inputs"]]
    inputs += [_checked(spec) for spec in freeze["files"].values()]
    inputs += [_checked(spec) for spec in config["model_inputs"]]
    pinned_paths = {str(Path(spec["path"]).resolve()) for spec in config["model_inputs"]}
    if any(str(Path(path).resolve()) not in pinned_paths for path in config["runtime_modules"].values()):
        raise ValueError("Runtime module origins must be included in frozen model/source inputs")
    verify_review(review_receipt, base, pilot_spec)
    verify_resources(resource_receipt, now=clock(), config=config)
    selected_route = next(row["model_route_seconds"] for row in throughput["candidates"] if row["n"] == n)
    if (clock() - start).total_seconds() + selected_route + 300 > 46800:
        raise ValueError("Compilation consumed the remaining full-route window; no executable plan written")
    plan = _plan(output / "main-config.json", source, python, n=n, start=start, review=review, inputs=inputs)
    report = inspect(plan)
    if report["status"] != "ready_not_started":
        raise ValueError("Compiled plan does not satisfy the supervisor contracts: " + "; ".join(report["blockers"]))
    durable_write(output / "plan.json", plan)
    value = {"status": "compiled_not_adopted", "plan": _spec(output / "plan.json"), "selected_n": n,
        "model_calls": 0, "processes_started": 0, "adopted": False, "codex_callbacks": False,
        "compiled_at": now.isoformat(), "requires_manual_plan_review_and_adoption": True}
    durable_write(output / "compilation.json", value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-main", help="Write only blocked, untimed main-study artifacts")
    for flag in ("runtime-config", "data-freeze", "source-root", "python", "output"):
        prepare.add_argument("--" + flag, required=True, type=Path)
    compile_parser = commands.add_parser("compile-main", help="Compile reviewed timed files; do not adopt or launch")
    for flag in ("template", "output", "review-receipt", "pilot-proof", "resource-receipt"):
        compile_parser.add_argument("--" + flag, required=True, type=Path)
    compile_parser.add_argument("--start-time", required=True)
    compile_parser.add_argument("--runtime-probe", type=Path)
    resources = commands.add_parser("resources", help="One30-second resource-only check; no model start")
    resources.add_argument("--config", dest="runtime_config", required=True, type=Path)
    resources.add_argument("--output", required=True, type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    function = {"prepare-main": prepare_template, "compile-main": compile_main, "resources": sample_resources}[command]
    result = function(**args)
    print(json.dumps({"status": result["status"], "model_calls": 0, "adopted": False}))
    success = (command == "prepare-main" and result["status"] == "blocked_no_launch"
        or command == "compile-main" and result["status"] == "compiled_not_adopted"
        or command == "resources" and result["status"] == "PASS_RESOURCE_ADMISSION")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
