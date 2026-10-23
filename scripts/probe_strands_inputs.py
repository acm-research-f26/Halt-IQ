"""Four real Strands forwards for input sensitivity, not benchmark quality.

Requires the existing external runtime and local model snapshots. It never
downloads, trains, edits weights, retries an uncertain forward, or changes other
applications. The deadline is cooperative at safe boundaries; an operator must
reconcile a process killed during an uncertain native call before any new run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pc_strands.control import (GpuLease, RunControl, StudyStopped, atomic_json,
                                parse_deadline, utc_now)
from pc_strands.data import public_state

DEFAULT_CONFIG = REPO / "configs/strands/input-probe.json"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_admission = {"samples": 4, "sample_interval_seconds": 10,
        "minimum_ram_gib": 8.5, "minimum_free_vram_gib": 9}
    expected_runtime = {"minimum_ram_gib": 7, "minimum_free_vram_gib": 3,
        "duty_cycle": .5, "minimum_rest_seconds": 5}
    if (config.get("scope") != "mechanistic_input_sensitivity_not_benchmark_quality"
            or config.get("max_wall_seconds") != 600 or config.get("threads") != 4
            or config.get("threshold") != .8 or config.get("startup_admission") != expected_admission
            or config.get("runtime_limits") != expected_runtime):
        raise ValueError("The bounded probe resource profile must not change")
    cases = config.get("cases", [])
    if len(cases) != 4:
        raise ValueError("Exactly four predeclared cases are required")
    pairs = {}
    for case in cases:
        if set(case) != {"pair", "evaluator_expected_correct", "public_state"}:
            raise ValueError("Unexpected probe-case fields")
        if type(case["evaluator_expected_correct"]) is not bool:
            raise ValueError("Evaluator labels must be explicit booleans")
        state = case["public_state"]
        if set(state) != {"question", "evidence", "draft"}:
            raise ValueError("Evaluator labels must not enter model state")
        public_state(state)
        pairs.setdefault(case["pair"], []).append(case)
    if len(pairs) != 2:
        raise ValueError("Exactly two paired facts are required")
    for group in pairs.values():
        if (len(group) != 2 or {case["evaluator_expected_correct"] for case in group} != {False, True}
                or group[0]["public_state"]["question"] != group[1]["public_state"]["question"]
                or group[0]["public_state"]["evidence"] != group[1]["public_state"]["evidence"]
                or group[0]["public_state"]["draft"] == group[1]["public_state"]["draft"]):
            raise ValueError("A pair must change only the draft and contain both evaluator labels")
    return config


def ensure_port_closed():
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", 11435)) == 0:
            raise RuntimeError("Private writer port is open; native probe will not start")


def startup_admission(control):
    """Four current samples separated by ten seconds; caller holds OS lease."""
    samples = []
    started = time.monotonic()
    for index in range(4):
        if index:
            control._sleep(10)
        control.before_work(require_gpu=True)
        samples.append(dict(control.last_resources))
    return {"samples": samples, "wall_seconds": time.monotonic() - started}


def load_backend(checkpoint, base):
    # The caller must own the shared OS lease and pass admission before this
    # function imports the backend and its Torch-dependent constructor.
    from pc_strands.decider import DeciderPaths, StrandsDecider
    return StrandsDecider(DeciderPaths(checkpoint, base), device="cuda", cpu_threads=4)


def model_input_hashes(checkpoint, comparison):
    paths = {"script": Path(__file__), "backend": REPO / "pc_strands/decider.py",
             "controls": REPO / "pc_strands/control.py"}
    for label, directory in (("released", checkpoint), ("comparison", comparison)):
        if directory is None:
            continue
        for name in ("head.safetensors", "strands_decider_config.json", "lora/adapter_model.safetensors"):
            path = directory / name
            if name == "head.safetensors" and not path.exists():
                path = directory / "slot_head.pt"
            if not path.is_file():
                raise ValueError("Required local head/config/adapter is missing")
            paths[f"{label}/{name}"] = path
    return paths, {key: digest(path) for key, path in paths.items()}


def summarize_pairs(records):
    results = []
    for pair in dict.fromkeys(row["pair"] for row in records):
        group = [row for row in records if row["pair"] == pair and row["status"] == "completed"]
        if len(group) != 2:
            continue
        correct = next(row for row in group if row["evaluator_expected_correct"])
        wrong = next(row for row in group if not row["evaluator_expected_correct"])
        results.append({"pair": pair, "correct_minus_contradictory_probability": {
            name: correct["scores"][name]["p_true"] - wrong["scores"][name]["p_true"]
            for name in correct["scores"]}})
    return results


@contextmanager
def native_lifetime(result):
    """Exit inside GpuLease so placement/offload always retain ownership."""
    holder = {}
    try:
        yield holder
    finally:
        if "decider" in holder:
            try:
                result["offload_seconds"] = holder["decider"].offload()
                result["owned_model_offload_completed"] = True
            except Exception as exc:
                result["owned_model_offload_completed"] = False
                result["offload_error_type"] = type(exc).__name__
                raise


def run_probe(args, *, backend_factory=load_backend, control_class=RunControl):
    config = load_config(args.config)
    checkpoint, base = args.checkpoint_dir.resolve(), args.base_dir.resolve()
    comparison = args.comparison_head.resolve() if args.comparison_head else None
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Native probe runs belong outside the source repository")
    if not args.lease_path.is_file():
        raise ValueError("Use the existing study shared GPU lease file")
    if not checkpoint.is_dir() or not base.is_dir() or (comparison and not comparison.is_dir()):
        raise ValueError("Existing external local model directories are required")
    if any(output == model or model in output.parents or output in model.parents
           for model in (checkpoint, base, comparison) if model is not None):
        raise ValueError("Output must not overlap any model directory")
    now = datetime.now(timezone.utc)
    deadline = parse_deadline(args.deadline) if args.deadline else now + timedelta(seconds=600)
    if not 0 < (deadline - now).total_seconds() <= config["max_wall_seconds"]:
        raise ValueError("Deadline must be in the next ten minutes with an explicit offset")
    output.mkdir(parents=True, exist_ok=False)
    paths, hashes = model_input_hashes(checkpoint, comparison)
    hashes["config"] = digest(args.config)
    record_path = output / "probe.json"
    result = {"scope": config["scope"], "status": "waiting_for_owned_admission",
        "started_at": utc_now(), "deadline": deadline.isoformat(),
        "deadline_kind": "cooperative_safe_boundary", "config_sha256": hashes["config"],
        "input_sha256_before": hashes, "physical_encoder_forwards": 0,
        "native_forward_pending_or_uncertain": False, "head_score_calls": 0,
        "records": [], "pair_differences": [], "gpu_lease_released": False,
        "model_work_started": False}
    atomic_json(record_path, result)
    guard = control_class(output / "control", deadline.isoformat(), duty_cycle=.5,
        min_rest_seconds=5, minimum_ram_gib=7, minimum_free_vram_gib=3)
    admission = control_class(output / "admission", deadline.isoformat(), duty_cycle=.5,
        min_rest_seconds=5, minimum_ram_gib=8.5, minimum_free_vram_gib=9, maximum_pause_seconds=0)
    decider, comparison_head, model_error = None, None, None
    wall_started = time.perf_counter()
    try:
        with GpuLease(args.lease_path, "four-state-strands-input-diagnostic"), native_lifetime(result) as holder:
            result["gpu_lease_held_before_backend_import"] = True
            ensure_port_closed()
            result["startup_admission"] = startup_admission(admission)
            ensure_port_closed()
            if (deadline - datetime.now(timezone.utc)).total_seconds() < 30:
                raise StudyStopped("Insufficient remaining time for native startup")
            started = time.perf_counter()
            result.update(status="native_startup", model_work_started=True)
            atomic_json(record_path, result)
            decider = backend_factory(checkpoint, base)
            holder["decider"] = decider
            # Placement precedes the post-load RAM guard: moving weights off CPU
            # releases their host storage before regular per-call admission.
            decider.activate()
            if comparison:
                comparison_head = decider.make_head(comparison)
            result["native_startup_seconds"] = time.perf_counter() - started
            result["released_metadata"] = decider.metadata()
            atomic_json(record_path, result)
            guard.rest_after_block(result["native_startup_seconds"])
            for index, case in enumerate(config["cases"]):
                guard.before_work(require_gpu=True)
                row = {"case": index + 1, "pair": case["pair"], "status": "native_pending",
                       "evaluator_expected_correct": case["evaluator_expected_correct"], "scores": {}}
                result["records"].append(row)
                result.update(status="running", native_forward_pending_or_uncertain=True)
                atomic_json(record_path, result)
                started = time.perf_counter()
                feature = decider.extract(case["public_state"])
                result["physical_encoder_forwards"] += 1
                result["native_forward_pending_or_uncertain"] = False
                row.update(status="native_completed", input_tokens=feature.input_tokens,
                           encoding_seconds=feature.encoding_seconds)
                atomic_json(record_path, result)
                heads = [("released", None)] + ([("comparison", comparison_head)] if comparison else [])
                for name, head in heads:
                    score = decider.score_features(feature, head=head)
                    values = asdict(score)
                    row["scores"][name] = {key: values[key] for key in (
                        "p_true", "logits", "temperature", "elapsed_seconds")}
                    row["scores"][name]["approved_at_0_8"] = values["p_true"] >= .8
                    result["head_score_calls"] += 1
                row["status"] = "completed"
                result["pair_differences"] = summarize_pairs(result["records"])
                # Persist completed work before cooperative rest/deadline checks.
                atomic_json(record_path, result)
                guard.rest_after_block(time.perf_counter() - started)
            result["status"] = "completed"
    except BaseException as exc:
        model_error = exc
        result.update(status="interrupted", error_type=type(exc).__name__)
    finally:
        # The with block has exited. Observe actual re-acquisition rather than
        # treating an absent process or stale lock metadata as ownership proof.
        try:
            with GpuLease(args.lease_path, "completed-probe-release-verification"):
                pass
            result["gpu_lease_released"] = True
        except Exception as exc:
            result["release_verification_error_type"] = type(exc).__name__
            model_error = model_error or exc
            result["status"] = "interrupted"
        try:
            after = {key: digest(path) for key, path in paths.items()}
            after["config"] = digest(args.config)
        except Exception as exc:
            after = {}
            result["final_hash_error_type"] = type(exc).__name__
            model_error = model_error or exc
        result.update(input_sha256_after=after, input_files_unchanged=after == hashes,
            finished_at=utc_now(), wall_seconds=time.perf_counter() - wall_started,
            intentional_idle_seconds=guard.idle_seconds + admission.idle_seconds)
        if after != hashes:
            result["status"] = "interrupted"
            model_error = model_error or RuntimeError("Probe input files changed")
        atomic_json(record_path, result)
    if model_error is not None:
        raise model_error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--comparison-head", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New external run directory")
    parser.add_argument("--lease-path", type=Path, required=True, help="Existing study shared GPU lease")
    parser.add_argument("--deadline", help="Optional ISO8601 with offset, at most ten minutes from now")
    args = parser.parse_args()
    result = run_probe(args)
    print(json.dumps({"status": result["status"], "physical_encoder_forwards": result["physical_encoder_forwards"],
                      "head_score_calls": result["head_score_calls"], "pair_differences": result["pair_differences"]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
