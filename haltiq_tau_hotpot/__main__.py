"""Prepare explicit local qualification plans; preparation never starts a job."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from haltiq_hotpot.protocol import digest
from haltiq_supervisor.plan import PACKAGES, inspect, source_files
from .data import prepare
from .journal import durable_write


def build_pilot_plan(config_path, source, python, readiness, deadline, hard_deadline):
    config_path, source, python = Path(config_path).resolve(), Path(source).resolve(), Path(python).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from .worker import qualification_protocol
    freeze, _ = qualification_protocol(config)
    p = freeze["pilot"]
    prep, heldout = sum(p["training"].values()), p["heldout"]
    inputs = [{"path": str(config_path), "sha256": digest(config_path)},
              {"path": config["data_freeze"], "sha256": config["data_freeze_sha256"]}]
    inputs += [freeze["files"][name] for name in ("pilot_fit", "pilot_tune", "pilot_calibration", "pilot_heldout")]
    inputs += config["model_inputs"]
    plan = {"version": 1, "scope": "custom_tau2_hotpot", "purpose": "qualification_pilot",
        "status": "ready", "source_root": str(source), "python": str(python),
        "source_paths": list(PACKAGES) + ["pyproject.toml"], "deadline": deadline,
        "hard_deadline": hard_deadline, "stop_grace_seconds": 60,
        "private_port": 11435, "gpu_lease": config["gpu_lease"], "study_lease": config["study_lease"],
        "resources": {"ram_gib": 7, "vram_gib": 3, "startup_ram_gib": 8.5,
            "startup_vram_gib": 9, "stability_seconds": 30, "admission_timeout_seconds": 1800,
            "cpu_threads": 4, "gpu_owners": 1, "minimum_rest_seconds": 5, "duty_cycle": .5},
        "codex_argv": None, "readiness": readiness, "inputs": inputs,
        "qualification_protocol": {"path": str(config_path), "sha256": digest(config_path)}, "stages": []}
    for name, count in (("collect", prep + heldout), ("features", prep + heldout), ("train", 1), ("compare", heldout * 4)):
        plan["stages"].append({"id": name, "model_work": True, "gpu": name != "train", "planned_units": count,
            "argv": [str(python), "-B", "-m", "haltiq_tau_hotpot.worker", name,
                "--config", str(config_path), "--output", "{run}/" + name, "--previous", "{run}",
                "--deadline", "{deadline}", "--control", "{control}", "--allow-model-calls"],
            "completion_receipt": "{run}/" + name + "/completion.json",
            "outputs": [{"path": "{run}/" + name + "/completion.json", "json_equals": {
                "terminal": True, "pending_or_uncertain": False, "cleanup_verified": True}},
                {"path": "{run}/" + name + "/" + {"collect": "episodes.json", "features": "features.json",
                    "train": "learning.json", "compare": "comparison.json"}[name]}],
            "notify_after": False})
        if name in {"collect", "features", "compare"}:
            plan["stages"][-1]["outputs"] += [{"path": "{run}/" + name + "/runtime-terminal.json"},
                {"path": "{run}/" + name + "/physical/calls.jsonl"}]
        if name == "features":
            plan["stages"][-1]["outputs"].append({"path": "{run}/features/untouched-baseline.json"})
    plan["reviewed_source_sha256"] = {name: digest(path) for name, path in source_files(plan).items()}
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    data = sub.add_parser("prepare-data")
    data.add_argument("--testing-root", required=True, type=Path)
    data.add_argument("--output", required=True, type=Path)
    machine = sub.add_parser("prepare-machine")
    machine.add_argument("--testing-root", required=True, type=Path)
    machine.add_argument("--data-freeze", required=True, type=Path)
    machine.add_argument("--output", required=True, type=Path)
    plan = sub.add_parser("prepare-pilot")
    for name in ("config", "source", "python", "readiness", "output"):
        plan.add_argument("--" + name, required=True, type=Path)
    plan.add_argument("--deadline", required=True)
    plan.add_argument("--hard-deadline", required=True)
    args = parser.parse_args()
    if args.action == "prepare-data":
        result = prepare(args.testing_root, args.output)
        print(json.dumps({"status": result["status"], "pilot": result["pilot"],
                          "main_status": result["main"]["status"]}))
    elif args.action == "prepare-machine":
        from .setup import machine_config
        result = machine_config(args.testing_root, args.data_freeze, args.output)
        print(json.dumps({"status": "cached_assets_verified", "pinned_inputs": len(result["model_inputs"]), "model_calls": 0}))
    else:
        readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
        result = build_pilot_plan(args.config, args.source, args.python, readiness, args.deadline, args.hard_deadline)
        report = inspect(result)
        if report["blockers"]:
            raise ValueError("Pilot is not ready: " + "; ".join(report["blockers"]))
        durable_write(args.output, result)
        print(json.dumps({"status": "prepared_not_started", "plan_sha256": digest(args.output),
                          "model_calls": 0, "long_study_started": False}))


if __name__ == "__main__":
    main()
