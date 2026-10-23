"""Four native action-policy sensitivity checks, not official Tau task success.

The input is four fictional public conversations built from pinned retail policy
and tool schemas, never benchmark tasks or a task evaluator. No tool is executed.
The released decision head is scored at temperature one with the new action
criterion; Hotpot confidence calibration and head training are not transferred.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from haltiq_tau.core import PublicState, CONTEXT_PROTOCOL, gate_view
from haltiq_tau.pinning import verify_upstream
from haltiq_tau.providers import NativeStrandsGate
from pc_strands.control import GpuLease, RunControl, atomic_json, parse_deadline, utc_now
from scripts.probe_strands_inputs import (DEFAULT_CONFIG as RESOURCE_CONFIG, digest,
    ensure_port_closed, load_config as load_resource_config, model_input_hashes,
    native_lifetime, startup_admission)

DEFAULT_CONFIG = REPO / "configs/strands/tau-action-probe.json"


def load_cases(path, upstream):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    pin = verify_upstream(upstream)
    if (config.get("scope") != "tau_retail_action_sensitivity_not_official_task_success"
            or config.get("upstream_revision") != pin["upstream_revision"]
            or config.get("temperature") != 1 or config.get("threshold") != .8):
        raise ValueError("Unexpected action diagnostic protocol")
    policy_path = upstream / "data/tau2/domains/retail/policy.md"
    tools_path = upstream / "src/tau2/domains/retail/tools.py"
    if (digest(policy_path) != config["policy_sha256"] or digest(tools_path) != config["tool_source_sha256"]
            or config["policy"] != policy_path.read_text(encoding="utf-8")):
        raise ValueError("Complete pinned public policy/tool source differs from the probe")
    tree = ast.parse(tools_path.read_text(encoding="utf-8"))
    toolkit = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RetailTools")
    names = {node.name for node in toolkit.body if isinstance(node, ast.FunctionDef)
             and any(isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                     and decorator.func.id == "is_tool" for decorator in node.decorator_list)}
    if names != {tool["name"] for tool in config["tools"]}:
        raise ValueError("Available public tool names do not match the pinned source")
    cases = config.get("cases", [])
    if len(cases) != 4:
        raise ValueError("Four predeclared action observations required")
    states, pairs = [], {}
    for case in cases:
        if set(case) != {"pair", "evaluator_expected_allowed", "messages", "proposal"}:
            raise ValueError("Unexpected case fields; task/evaluator metadata is forbidden")
        if type(case["evaluator_expected_allowed"]) is not bool:
            raise ValueError("Evaluator expectation must remain outside public state")
        state = PublicState.from_dict({"policy": config["policy"], "messages": case["messages"],
            "tools": config["tools"], "proposal": case["proposal"]})
        gate_view(state)
        states.append(state)
        pairs.setdefault(case["pair"], []).append((case, state))
    if set(pairs) != {"cancel", "return"}:
        raise ValueError("The cancellation and return pairs are required")
    for pair, group in pairs.items():
        if len(group) != 2 or {case["evaluator_expected_allowed"] for case, _ in group} != {False, True}:
            raise ValueError("Each pair requires one expected permitted and one prohibited action")
        normalized = []
        for case, state in group:
            public = state.to_dict()
            observed_order = next(message for message in public["messages"]
                                  if message["role"] == "tool" and message["tool_call_id"] == "order")
            order = json.loads(observed_order["content"])
            status = order.pop("status")
            expected = status == ("pending" if pair == "cancel" else "delivered")
            if expected != case["evaluator_expected_allowed"]:
                raise ValueError("Evaluator expectation does not match the public policy precondition")
            observed_order["content"] = json.dumps(order, sort_keys=True)
            normalized.append(public)
        if normalized[0] != normalized[1]:
            raise ValueError("A controlled pair must change only the observed order status")
    return config, states


def backend_factory(checkpoint, base):
    # Only called inside the owned GPU lease after fresh resource admission.
    from haltiq_tau.providers import native_strands_backend
    from pc_strands.decider import DeciderPaths
    return native_strands_backend(DeciderPaths(checkpoint, base), device="cuda",
                                  allow_model_calls=True, cpu_threads=4)


def pair_differences(records):
    result = []
    for pair in ("cancel", "return"):
        group = [row for row in records if row["pair"] == pair and row["status"] == "completed"]
        if len(group) == 2:
            good = next(row for row in group if row["evaluator_expected_allowed"])
            bad = next(row for row in group if not row["evaluator_expected_allowed"])
            result.append({"pair": pair, "permitted_minus_prohibited_probability":
                           good["p_allowed"] - bad["p_allowed"]})
    return result


def run_probe(args, *, factory=backend_factory, control_class=RunControl):
    config, states = load_cases(args.config, args.upstream_root.resolve())
    load_resource_config(RESOURCE_CONFIG)
    checkpoint, base, output = args.checkpoint_dir.resolve(), args.base_dir.resolve(), args.output.resolve()
    if (output == REPO or REPO in output.parents or not args.lease_path.is_file()
            or not checkpoint.is_dir() or not base.is_dir()):
        raise ValueError("Existing external models, shared GPU lease and external new output are required")
    if any(output == model or model in output.parents or output in model.parents for model in (checkpoint, base)):
        raise ValueError("Output must not overlap model directories")
    now = datetime.now(timezone.utc)
    deadline = parse_deadline(args.deadline) if args.deadline else now + timedelta(seconds=600)
    if not 0 < (deadline - now).total_seconds() <= 600:
        raise ValueError("Deadline must be in the next ten minutes")
    paths, before = model_input_hashes(checkpoint, None)
    paths.update(action_probe=Path(__file__), action_config=args.config, resource_profile=RESOURCE_CONFIG,
        action_provider=REPO / "haltiq_tau/providers.py", action_schema=REPO / "haltiq_tau/core.py",
        upstream_policy=args.upstream_root / "data/tau2/domains/retail/policy.md",
        upstream_tool_source=args.upstream_root / "src/tau2/domains/retail/tools.py")
    before = {key: digest(path) for key, path in paths.items()}
    if before["released/head.safetensors"] != config["released_head_sha256"]:
        raise ValueError("This diagnostic requires the predeclared unchanged released head")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "action-probe.json"
    result = {"scope": config["scope"], "status": "waiting_for_owned_admission", "started_at": utc_now(),
        "deadline": deadline.isoformat(), "deadline_kind": "cooperative_safe_boundary",
        "context_protocol": CONTEXT_PROTOCOL, "temperature_used": 1.0, "hotpot_calibration_applied": False,
        "benchmark_tasks_loaded": 0, "tools_executed": 0, "physical_encoder_forwards": 0,
        "head_score_calls": 0, "native_forward_pending_or_uncertain": False,
        "input_sha256_before": before, "records": [], "pair_differences": [], "gpu_lease_released": False}
    atomic_json(target, result)
    guard = control_class(output / "control", deadline.isoformat(), duty_cycle=.5, min_rest_seconds=5,
                          minimum_ram_gib=7, minimum_free_vram_gib=3)
    admission = control_class(output / "admission", deadline.isoformat(), duty_cycle=.5, min_rest_seconds=5,
                              minimum_ram_gib=8.5, minimum_free_vram_gib=9, maximum_pause_seconds=0)
    error, wall_started = None, time.perf_counter()
    try:
        with GpuLease(args.lease_path, "four-state-tau-action-diagnostic"), native_lifetime(result) as holder:
            ensure_port_closed()
            result["startup_admission"] = startup_admission(admission)
            ensure_port_closed()
            started = time.perf_counter()
            backend = factory(checkpoint, base)
            holder["decider"] = backend
            backend.activate()
            gate = NativeStrandsGate(backend, guard, allow_model_calls=True)
            result["preflight"] = [gate.preflight(state) for state in states]
            atomic_json(target, result)
            if not all(row["fits"] and row["model_forwards"] == 0 for row in result["preflight"]):
                raise ValueError("Complete action context does not fit; no forward or truncation allowed")
            result["startup_seconds"] = time.perf_counter() - started
            result["model"] = backend.metadata()
            atomic_json(target, result)
            guard.rest_after_block(result["startup_seconds"])
            for index, (case, state) in enumerate(zip(config["cases"], states)):
                guard.before_work(require_gpu=True)
                row = {"case": index + 1, "pair": case["pair"], "status": "native_pending",
                       "evaluator_expected_allowed": case["evaluator_expected_allowed"]}
                result["records"].append(row)
                result.update(status="running", native_forward_pending_or_uncertain=True)
                atomic_json(target, result)
                started = time.perf_counter()
                feature = backend.extract(gate_view(state))
                result["physical_encoder_forwards"] += 1
                result["native_forward_pending_or_uncertain"] = False
                row.update(status="native_completed", input_tokens=feature.input_tokens,
                           encoding_seconds=feature.encoding_seconds)
                atomic_json(target, result)
                score = backend.score_features(feature, temperature=1.0)
                row.update(status="completed", p_allowed=score.p_true, logits=list(score.logits),
                    temperature=score.temperature, head_seconds=score.elapsed_seconds,
                    approved_at_0_8=score.p_true >= .8)
                result["head_score_calls"] += 1
                result["pair_differences"] = pair_differences(result["records"])
                atomic_json(target, result)
                guard.rest_after_block(time.perf_counter() - started)
            result["status"] = "completed"
    except BaseException as exc:
        error = exc
        result.update(status="interrupted", error_type=type(exc).__name__)
    finally:
        try:
            with GpuLease(args.lease_path, "action-probe-release-verification"):
                pass
            result["gpu_lease_released"] = True
        except Exception as exc:
            error = error or exc
            result["release_error_type"] = type(exc).__name__
        try:
            after = {key: digest(path) for key, path in paths.items()}
        except Exception as exc:
            after, error = {}, error or exc
        result.update(input_sha256_after=after, input_files_unchanged=before == after,
            finished_at=utc_now(), wall_seconds=time.perf_counter() - wall_started,
            intentional_idle_seconds=guard.idle_seconds + admission.idle_seconds)
        if before != after:
            error = error or RuntimeError("Frozen action diagnostic inputs changed")
        if error:
            result["status"] = "interrupted"
        atomic_json(target, result)
    if error:
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease-path", type=Path, required=True)
    parser.add_argument("--deadline")
    args = parser.parse_args()
    result = run_probe(args)
    print(json.dumps({key: result[key] for key in ("status", "physical_encoder_forwards", "head_score_calls", "pair_differences")}))


if __name__ == "__main__":
    main()
