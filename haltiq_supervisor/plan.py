"""Validate and freeze explicit phase commands, without loading any model."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import uuid

from pc_strands.campaign import clock, digest, load, save
from .handoff import validate_callback

PACKAGES = ("haltiq", "pc_strands", "haltiq_hotpot", "haltiq_tau", "haltiq_supervisor")
READINESS = {"custom_domain", "durable_call_journals", "hobson_training", "gpu_coordinator", "combined_pilot"}
SAFE_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}\Z")


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path escapes its declared directory")
    return path


def expand(value, directory):
    return value.replace("{run}", str(directory)).replace("{source}", str(directory / "source"))


def satisfied(spec, directory=None):
    path = Path(expand(spec["path"], directory) if directory else spec["path"])
    if not path.is_file():
        return False
    if spec.get("sha256") and digest(path) != spec["sha256"]:
        raise ValueError("Pinned file changed: " + str(path))
    if spec.get("json_equals"):
        try:
            data = load(path)
            for dotted, expected in spec["json_equals"].items():
                actual = data
                for key in dotted.split("."):
                    actual = actual[key]
                if type(actual) is not type(expected) or actual != expected:
                    return False
        except (ValueError, TypeError, KeyError):
            return False
    return True


def validate(plan):
    if plan.get("version") != 1 or plan.get("scope") not in {"custom_tau2_hotpot", "offline_supervision_fixture"}:
        raise ValueError("Explicit supported supervision schema/scope required")
    source = Path(plan["source_root"])
    if not source.is_absolute() or not source.is_dir() or not Path(plan["python"]).is_file():
        raise ValueError("Existing absolute source and Python paths required")
    if not Path(plan["python"]).is_absolute():
        raise ValueError("Python must be absolute")
    if plan.get("deadline") is None and plan.get("hard_deadline") is None and plan.get("status") != "ready":
        pass  # A preparation-only template has no adopted execution clock.
    else:
        deadline, hard = clock(plan["deadline"]), clock(plan["hard_deadline"])
        if not 0 < (hard - deadline).total_seconds() <= 600:
            raise ValueError("Hard cutoff must follow worker deadline by at most600seconds")
    if not 0 <= plan.get("stop_grace_seconds", 60) <= 600:
        raise ValueError("Stop grace must be bounded")
    policy = plan["resources"]
    floors = {"ram_gib": 7, "vram_gib": 3, "startup_ram_gib": 8.5,
              "startup_vram_gib": 9, "stability_seconds": 30, "minimum_rest_seconds": 5}
    if any(type(policy.get(k)) not in (int, float) or not math.isfinite(policy[k]) or policy[k] < floor for k, floor in floors.items()):
        raise ValueError("Resource reserves/rest must not be lowered")
    if policy.get("cpu_threads") != 4 or policy.get("gpu_owners") != 1 or policy.get("duty_cycle") != .5:
        raise ValueError("Keep four threads, one GPU owner and1:1work/rest")
    if not 30 <= policy.get("admission_timeout_seconds", 1200) <= 7200:
        raise ValueError("Bounded local admission wait required")
    stages = plan.get("stages", [])
    ids = [s["id"] for s in stages]
    if not stages or len(set(ids)) != len(ids) or any(not SAFE_ID.fullmatch(x) for x in ids):
        raise ValueError("Explicit unique safe stage IDs required")
    for stage in stages:
        if type(stage.get("model_work")) is not bool or type(stage.get("gpu")) is not bool:
            raise ValueError("Declare model_work and gpu for every stage")
        argv = stage.get("argv")
        if argv is not None:
            if (not isinstance(argv, list) or not argv or argv[0] != plan["python"]
                    or any(not isinstance(x, str) or not x or "\0" in x for x in argv)):
                raise ValueError("Explicit Python argv required; shell commands unsupported")
            if stage["model_work"] and not all(x in argv for x in ("{deadline}", "{control}", "--allow-model-calls")):
                raise ValueError("Model workers require explicit deadline/control/allow arguments")
            if stage["model_work"]:
                if (len(argv) < 4 or argv[1:3] != ["-B", "-m"]
                        or not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+", argv[3])
                        or argv[3].split(".")[0] not in PACKAGES):
                    raise ValueError("Model entrypoints must be frozen package modules, never live scripts or -c")
                module = source / Path(*argv[3].split("."))
                if not module.with_suffix(".py").is_file() and not (module / "__main__.py").is_file():
                    raise ValueError("Declared model worker module is not implemented")
        if not stage.get("outputs") or any(not s["path"].startswith("{run}/") for s in stage["outputs"]):
            raise ValueError("Every stage needs fresh output contracts inside this run")
        if stage["model_work"] and plan["scope"] != "custom_tau2_hotpot":
            raise ValueError("Offline fixtures cannot dispatch model work")
        unselected = stage.get("planned_units") is None and plan.get("status") != "ready"
        if stage["model_work"] and ((not unselected and (type(stage.get("planned_units")) is not int or stage["planned_units"] < 1))
                or stage.get("completion_receipt") not in [s["path"] for s in stage["outputs"]]):
            raise ValueError("Model stages need planned counts and a bound completion receipt")
    if plan["scope"] == "custom_tau2_hotpot" and not set(PACKAGES + ("pyproject.toml",)) <= set(plan["source_paths"]):
        raise ValueError("Freeze every study package and pyproject, not only pc_strands")
    validate_callback(plan.get("codex_argv"))
    if any(s["model_work"] for s in stages):
        if plan.get("private_port") != 11435 or not Path(plan.get("gpu_lease", "")).is_absolute():
            raise ValueError("Use the declared private11435 and existing absolute shared GPU lease")
        if (not Path(plan.get("study_lease", "")).is_absolute()
                or plan["study_lease"] == plan["gpu_lease"]):
            raise ValueError("Use one absolute shared study lease separate from GPU ownership")
    return plan


def inspect(plan):
    validate(plan)
    blockers = list(plan.get("blockers", []))
    if plan.get("status") != "ready":
        blockers.append("Plan is preparation-only, not an adopted executable study")
    if any(s.get("argv") is None for s in plan["stages"]):
        blockers.append("One or more real worker commands remain unimplemented")
    if any(s["model_work"] and s.get("planned_units") is None for s in plan["stages"]):
        blockers.append("Question/episode counts require answer-blind throughput selection")
    ready = plan.get("readiness", {})
    if any(s["model_work"] for s in plan["stages"]):
        blockers.extend("Missing verified readiness: " + key for key in sorted(READINESS - ready.keys()))
        if not plan.get("reviewed_source_sha256"):
            blockers.append("Reviewed complete source manifest is not bound")
    for key, specs in ready.items():
        if not specs or not all(s.get("sha256") and satisfied(s) for s in specs):
            blockers.append("Readiness receipt not satisfied: " + key)
    for spec in plan.get("inputs", []):
        if not spec.get("sha256") or not satisfied(spec):
            blockers.append("Pinned input unavailable or unbound: " + spec["path"])
    if plan.get("hard_deadline") is None:
        blockers.append("Execution window not selected; set deadlines only at reviewed adoption")
    elif clock(plan["hard_deadline"]) <= datetime.now(timezone.utc):
        blockers.append("Execution window has expired; review a new plan before adoption")
    return {"status": "blocked_no_launch" if blockers else "ready_not_started", "blockers": blockers,
            "model_calls": 0, "codex_calls": 0}


def source_files(plan):
    root, files = Path(plan["source_root"]).resolve(), {}
    for name in plan["source_paths"]:
        candidate = within(root / name, root)
        if candidate.is_dir() and name not in PACKAGES + ("scripts",):
            raise ValueError("Declare configuration/data files explicitly; directory snapshots are Python packages only")
        paths = sorted(candidate.rglob("*.py")) if candidate.is_dir() else [candidate]
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Only real source files can be frozen")
            within(path, root)
            files[path.relative_to(root).as_posix()] = path
    if "haltiq_supervisor/engine.py" not in files or "pc_strands/control.py" not in files:
        raise ValueError("Supervisor and resource-control source must be frozen")
    return files


def prepare(plan_path, directory, *, allow_model_calls=False):
    plan_path, directory = Path(plan_path).resolve(), Path(directory).resolve()
    original = plan_path.read_bytes()
    plan = json.loads(original)
    report = inspect(plan)
    if report["blockers"]:
        raise ValueError("Study is blocked: " + "; ".join(report["blockers"]))
    if any(s["model_work"] for s in plan["stages"]) and not allow_model_calls:
        raise ValueError("Explicit --allow-model-calls required even for live-plan adoption")
    source_root = Path(plan["source_root"]).resolve()
    if directory == source_root or source_root in directory.parents or directory in source_root.parents:
        raise ValueError("Use a new external run directory")
    files = source_files(plan)
    hashes = {key: digest(path) for key, path in files.items()}
    if plan.get("reviewed_source_sha256") and plan["reviewed_source_sha256"] != hashes:
        raise ValueError("Source differs from the reviewed manifest")
    directory.mkdir(parents=True, exist_ok=False)
    for key, path in files.items():
        target = directory / "source" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (directory / "plan.json").write_bytes(original)
    state = {"status": "prepared_not_started", "run_id": str(uuid.uuid4()), "plan_sha256": hashlib.sha256(original).hexdigest(),
             "source_sha256": hashes, "stages": {}, "python_sha256": digest(plan["python"]),
             "prepared_at": datetime.now(timezone.utc).isoformat()}
    save(directory / "supervisor.json", state)
    verify_frozen(plan, state, directory)
    return state


def verify_frozen(plan, state, directory):
    if digest(directory / "plan.json") != state["plan_sha256"] or digest(plan["python"]) != state["python_sha256"]:
        raise ValueError("Frozen plan or interpreter changed")
    actual = {p.relative_to(directory / "source").as_posix(): digest(p)
              for p in (directory / "source").rglob("*") if p.is_file()}
    if actual != state["source_sha256"]:
        raise ValueError("Frozen source changed")
    for spec in plan.get("inputs", []):
        if not satisfied(spec):
            raise ValueError("Pinned input changed")
    for specs in plan.get("readiness", {}).values():
        if not all(s.get("sha256") and satisfied(s) for s in specs):
            raise ValueError("Pinned readiness proof changed")
