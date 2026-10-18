"""Run reviewed, explicit study stages; never infer readiness or retry a stage.

Plan: working_directory, evaluation_deadline, optional hard_deadline, steps,
prerequisites, fresh_options. A step has id, argv, requires, outputs and calls
(default true). File conditions have path, optional sha256, json_equals/json_in
(dotted keys). Use {source} and {run} in argv for the frozen source/run paths.
An operator fresh-freeze JSON selects exactly one proposed N; see choose_fresh.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .control import GpuLease
from .windows_job import ProcessContainment


def stamp():
    return datetime.now(timezone.utc).isoformat()


def clock(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Every deadline/freeze timestamp needs an explicit UTC offset")
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def condition(spec):
    path = Path(spec["path"])
    if not path.is_file():
        return False
    if spec.get("json_equals") or spec.get("json_in"):
        try:
            data = load(path)
            checks = {key: [value] for key, value in spec.get("json_equals", {}).items()}
            checks.update(spec.get("json_in", {}))
            for dotted, expected in checks.items():
                actual = data
                for key in dotted.split("."):
                    actual = actual[key]
                if actual not in expected:
                    return False
        except (ValueError, KeyError, TypeError):
            return False
    if spec.get("sha256") and digest(path) != spec["sha256"]:
        raise ValueError(f"Pinned prerequisite/output changed: {path}")
    return True


def choose_fresh(plan, freeze_path, plan_sha256):
    freeze = load(freeze_path)
    if (freeze.get("plan_sha256") != plan_sha256 or freeze.get("answer_blind") is not True
            or freeze.get("basis") != "measured_throughput_and_resources"
            or freeze.get("fresh_scoring_started") is not False
            or clock(freeze["frozen_at"]) > datetime.now(timezone.utc)):
        raise ValueError("Fresh freeze must bind this plan and pre-score throughput/resource selection")
    choices = [x for x in plan["fresh_options"] if type(freeze.get("n")) is int and x["n"] == freeze["n"]]
    if len(choices) != 1 or freeze["n"] not in (20, 50, 100):
        raise ValueError("Choose exactly one proposed fresh N: 20, 50 or 100")
    option = choices[0]
    if digest(option["dataset"]) != freeze.get("dataset_sha256"):
        raise ValueError("Fresh dataset fingerprint differs from operator freeze")
    return option


def initialize(plan_path, output, resume):
    plan, plan_hash = load(plan_path), digest(plan_path)
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "campaign.json"
    if receipt.exists():
        if not resume:
            raise ValueError("Existing campaign requires explicit --resume")
        state = load(receipt)
        if state["plan_sha256"] != plan_hash:
            raise ValueError("Campaign plan changed; no implicit adoption")
        if any(row["status"] != "completed" for row in state["stages"].values()):
            raise ValueError("Failed/unfinished stage requires operator reconciliation; never auto-retry")
    else:
        if any(x.name != "campaign.lock" for x in output.iterdir()):
            raise ValueError("New campaign output must be empty")
        package = Path(plan["working_directory"]) / "pc_strands"
        target = output / "source" / "pc_strands"
        target.mkdir(parents=True)
        for source in sorted(package.glob("*.py")):
            shutil.copyfile(source, target / source.name)
        if not (target / "campaign.py").is_file():
            raise ValueError("Reviewed source must contain campaign.py")
        shutil.copyfile(plan_path, output / "plan.json")
        state = {"plan_sha256": plan_hash, "started_at": stamp(), "stages": {},
                 "source_sha256": {p.name: digest(p) for p in target.glob("*.py")}}
    return plan, state


def run(plan_path, output, *, resume=False, selected=None, freeze_path=None, poll_seconds=2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with GpuLease(output / "campaign.lock", "campaign supervisor (not the GPU lock)"):
        plan, state = initialize(Path(plan_path).resolve(), output, resume)
        source = output / "source"
        if ({p.name: digest(p) for p in (source / "pc_strands").glob("*.py")} != state["source_sha256"]
                or digest(output / "plan.json") != state["plan_sha256"]):
            raise ValueError("Frozen source/plan changed")
        deadline = clock(plan["evaluation_deadline"])
        hard = clock(plan.get("hard_deadline", (deadline + timedelta(minutes=10)).isoformat()))
        if not deadline < hard <= deadline + timedelta(minutes=10):
            raise ValueError("Hard deadline must follow evaluation cutoff by at most ten minutes")
        steps = plan["steps"]
        all_steps = steps + [s for option in plan.get("fresh_options", []) for s in option["steps"]]
        ids = [s["id"] for s in all_steps]
        if len(ids) != len(set(ids)) or any(not x or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in x) for x in ids):
            raise ValueError("Stage IDs must be unique safe filenames")
        if selected and (len(selected) != len(set(selected)) or not set(selected).issubset(s["id"] for s in steps)):
            raise ValueError("Unknown selected stage")
        queue = [next(s for s in steps if s["id"] == name) for name in selected] if selected else list(steps)
        if plan.get("fresh_options") and not selected:
            queue.append({"id": "_fresh_gate"})
        if state.get("selected_stages", selected) != selected:
            raise ValueError("Selected stage order changed on resume")
        state["selected_stages"] = selected
        receipt = output / "campaign.json"

        def publish(status, **extra):
            state.update(status=status, updated_at=stamp(), **extra)
            save(receipt, state)

        def stopped():
            return datetime.now(timezone.utc) >= deadline or (output / "stop.request").exists()

        try:
            for step in queue:
                name = step["id"]
                if name == "_fresh_gate":
                    freeze_path = Path(freeze_path) if freeze_path else output / "fresh-freeze.json"
                    while not freeze_path.is_file():
                        publish("awaiting_fresh_freeze")
                        if stopped():
                            publish("stopped_before_new_stage", next_stage="_fresh_gate")
                            return state
                        time.sleep(poll_seconds)
                    option = choose_fresh(plan, freeze_path, state["plan_sha256"])
                    freeze_sha = digest(freeze_path)
                    if state.get("fresh_freeze_sha256", freeze_sha) != freeze_sha:
                        raise ValueError("Operator fresh freeze changed")
                    if "fresh_freeze_sha256" not in state:
                        if freeze_path.resolve() != (output / "fresh-freeze.json").resolve():
                            shutil.copyfile(freeze_path, output / "fresh-freeze.json")
                    state.update(fresh_freeze_sha256=freeze_sha, fresh_n=option["n"])
                    queue.extend(option["steps"])
                    publish("fresh_selected")
                    continue
                if name in state["stages"]:
                    for path, sha in state["stages"][name]["output_sha256"].items():
                        if digest(path) != sha:
                            raise ValueError("Completed stage output changed")
                    continue
                if stopped():
                    publish("stopped_before_new_stage", next_stage=name)
                    return state
                dependencies = step.get("requires", [])
                while True:
                    waiting = []
                    for dependency in dependencies:
                        if dependency in state["stages"]:
                            continue
                        spec = plan.get("prerequisites", {}).get(dependency)
                        if spec is None:
                            raise ValueError(f"Unresolved prerequisite needs explicit file contract: {dependency}")
                        if not all(condition(x) for x in (spec if isinstance(spec, list) else [spec])):
                            waiting.append(dependency)
                    if not waiting and not (output / "pause.request").exists():
                        break
                    publish("waiting_prerequisite", next_stage=name, waiting=waiting)
                    if stopped():
                        publish("stopped_before_new_stage", next_stage=name)
                        return state
                    time.sleep(poll_seconds)
                current = {p.name: digest(p) for p in (source / "pc_strands").glob("*.py")}
                if current != state["source_sha256"] or digest(output / "plan.json") != state["plan_sha256"]:
                    raise ValueError("Frozen source/plan changed")
                argv = [x.replace("{source}", str(source)).replace("{run}", str(output)) for x in step["argv"]]
                if not argv or any(not isinstance(x, str) or not x for x in argv):
                    raise ValueError("Explicit nonempty argv required; shell commands unsupported")
                if step.get("calls", True) and plan["evaluation_deadline"] not in argv:
                    raise ValueError("Every inference stage must explicitly receive the evaluation cutoff")
                if stopped():
                    publish("stopped_before_new_stage", next_stage=name)
                    return state
                row = {"status": "started", "argv": argv, "started_at": stamp(), "output_sha256": {}}
                state["stages"][name] = row
                publish("running", next_stage=name)
                child, containment = None, ProcessContainment()
                try:
                    environment = os.environ.copy()
                    for key in ("TYPESAFE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH"):
                        environment.pop(key, None)
                    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
                    with (output / f"{name}.log").open("ab", buffering=0) as log:
                        child = subprocess.Popen(argv, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                                 creationflags=0x08004000 if os.name == "nt" else 0)
                        containment.attach(child.pid)
                        row["pid"] = child.pid
                        publish("running")
                        while child.poll() is None:
                            if datetime.now(timezone.utc) >= hard or (output / "stop.request").exists():
                                raise TimeoutError("Campaign hard cutoff/operator stop; owned child terminated")
                            time.sleep(min(poll_seconds, 1))
                        row["exit_code"] = child.returncode
                        if child.returncode:
                            raise RuntimeError(f"Stage {name} exited {child.returncode}; no retry")
                    for spec in step.get("outputs", []):
                        if not condition(spec):
                            raise ValueError(f"Stage output contract failed: {spec['path']}")
                        row["output_sha256"][spec["path"]] = digest(spec["path"])
                    if {p.name: digest(p) for p in (source / "pc_strands").glob("*.py")} != state["source_sha256"]:
                        raise ValueError("Frozen source changed during child execution")
                    row.update(status="completed", finished_at=stamp())
                    publish("between_stages")
                finally:
                    containment.close()
                    if child is not None and child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)
            publish("completed")
            return state
        except BaseException as error:
            publish("failed", error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages", help="Comma-separated explicit stage order; excludes fresh options")
    parser.add_argument("--fresh-freeze", type=Path)
    args = parser.parse_args()
    run(args.plan, args.run_dir, resume=args.resume,
        selected=args.stages.split(",") if args.stages else None, freeze_path=args.fresh_freeze)


if __name__ == "__main__":
    main()
