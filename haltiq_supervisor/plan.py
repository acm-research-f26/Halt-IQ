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

PACKAGES = ("haltiq", "pc_strands", "haltiq_hotpot", "haltiq_tau", "haltiq_supervisor", "haltiq_tau_hotpot")
READINESS = {"custom_domain", "durable_call_journals", "hobson_training", "gpu_coordinator", "combined_pilot"}
SAFE_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}\Z")
QUALIFICATION_PHASES = ("collect", "features", "train", "compare")
QUALIFICATION_ARMS = ["fixed1", "fixed3", "released", "trained"]


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


def _pinned_json(spec, label):
    if (not isinstance(spec, dict) or not isinstance(spec.get("path"), str)
            or not Path(spec["path"]).is_absolute()
            or not isinstance(spec.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
        raise ValueError(label + " requires an absolute path and SHA256")
    path = Path(spec["path"])
    if not path.is_file() or digest(path) != spec["sha256"]:
        raise ValueError(label + " is missing or changed")
    return load(path)


def qualification(plan):
    """Permit only the bounded pilot that establishes the missing live proof."""
    spec = plan.get("qualification_protocol")
    config = _pinned_json(spec, "Qualification config")
    if not isinstance(config, dict) or config.get("purpose") != "qualification_pilot":
        raise ValueError("Qualification config must explicitly declare its purpose")
    freeze_spec = {"path": config.get("data_freeze"), "sha256": config.get("data_freeze_sha256")}
    freeze = _pinned_json(freeze_spec, "Qualification data freeze")
    if not isinstance(freeze, dict) or freeze.get("schema") != "tau2-hotpot-qualification-v1":
        raise ValueError("Explicit qualification data schema required")
    pilot = freeze.get("pilot", {})
    if not isinstance(pilot, dict) or not isinstance(pilot.get("training"), dict):
        raise ValueError("Qualification training counts must be explicit")
    training = pilot.get("training", {})
    if (set(training) != {"fit", "tune", "calibration"}
            or any(type(n) is not int or n <= 0 for n in training.values())
            or sum(training.values()) > 60
            or type(pilot.get("heldout")) is not int or not 1 <= pilot["heldout"] <= 8
            or type(pilot.get("max_drafts")) is not int or not 1 <= pilot["max_drafts"] <= 3
            or pilot.get("arms") != QUALIFICATION_ARMS or pilot.get("threshold") != .8
            or pilot.get("criteria") != ["answer"]
            or pilot.get("engineering_only") is not True
            or pilot.get("baseline_before_weight_updates") is not True):
        raise ValueError("Qualification counts, arms or baseline contract exceed the bounded pilot")
    selection = freeze.get("selection", {})
    if (selection.get("answer_blind") is not True
            or any(selection.get(key) is not False for key in ("quality_filtering", "length_filtering", "replacement_allowed"))):
        raise ValueError("Qualification selection must remain answer-blind without filtering or replacement")
    required = [spec, freeze_spec]
    seen_ids, seen_questions = set(), set()
    for partition, count in {**training, "heldout": pilot["heldout"]}.items():
        cohort = freeze.get("files", {}).get("pilot_" + partition)
        cases = _pinned_json(cohort, "Qualification cohort")
        if (not isinstance(cases, list) or type(cohort.get("count")) is not int
                or cohort["count"] != count or len(cases) != count):
            raise ValueError("Qualification cohort counts differ from the frozen protocol")
        counts = {kind: 0 for kind in ("bridge", "comparison")}
        for case in cases:
            public = case.get("public", {}) if isinstance(case, dict) else {}
            identifier, question, kind = (public.get(k) for k in ("id", "question", "type"))
            expected_partition = "engineering_heldout" if partition == "heldout" else partition
            if (not isinstance(identifier, str) or not identifier or not isinstance(question, str)
                    or not question.strip() or kind not in counts or case.get("partition") != expected_partition):
                raise ValueError("Qualification cohort has invalid question identity or partition")
            from pc_strands.data import question_key
            key = question_key(question)
            if identifier in seen_ids or key in seen_questions:
                raise ValueError("Qualification question-grouped partitions overlap")
            seen_ids.add(identifier)
            seen_questions.add(key)
            counts[kind] += 1
        if cohort.get("type_counts") != counts or counts["bridge"] != counts["comparison"]:
            raise ValueError("Qualification question-type counts must match balanced frozen cohorts")
        required.append(cohort)
    inputs = {(str(Path(s["path"]).resolve()), s.get("sha256")) for s in plan.get("inputs", [])}
    if any((str(Path(s["path"]).resolve()), s["sha256"]) not in inputs for s in required):
        raise ValueError("Qualification config, data freeze and cohorts must also be pinned inputs")
    stages = plan["stages"]
    if tuple(s["id"] for s in stages) != QUALIFICATION_PHASES:
        raise ValueError("Qualification permits exactly collect/features/train/compare in order")
    prep, heldout = sum(training.values()), pilot["heldout"]
    units = {"collect": prep + heldout, "features": prep + heldout, "train": 1, "compare": 4 * heldout}
    for stage in stages:
        argv = stage.get("argv") or []
        expected = [plan["python"], "-B", "-m", "haltiq_tau_hotpot.worker", stage["id"]]
        if (not stage["model_work"] or argv[:5] != expected
                or stage.get("planned_units") != units[stage["id"]]):
            raise ValueError("Qualification phase commands/counts must match the frozen pilot")
        flags = argv[5:]
        expected_flags = {"--config", "--output", "--previous", "--deadline", "--control", "--allow-model-calls"}
        if (len(flags) != 11 or {x for x in flags if x.startswith("--")} != expected_flags
                or flags.count("--allow-model-calls") != 1):
            raise ValueError("Qualification worker arguments must be explicit and unique")
        values = {}
        index = 0
        while index < len(flags):
            flag = flags[index]
            if flag == "--allow-model-calls":
                index += 1
                continue
            if flag not in expected_flags or index + 1 >= len(flags) or flag in values:
                raise ValueError("Qualification worker arguments must be explicit and unique")
            values[flag] = flags[index + 1]
            index += 2
        if (values.get("--config") != spec["path"] or values.get("--deadline") != "{deadline}"
                or values.get("--control") != "{control}" or values.get("--previous") != "{run}"
                or values.get("--output") != "{run}/" + stage["id"]
                or stage.get("completion_receipt") != "{run}/" + stage["id"] + "/completion.json"):
            raise ValueError("Qualification worker must use its pinned config and owned phase output")
    return freeze


def validate(plan):
    if plan.get("version") != 1 or plan.get("scope") not in {"custom_tau2_hotpot", "offline_supervision_fixture"}:
        raise ValueError("Explicit supported supervision schema/scope required")
    purpose = plan.get("purpose", "main_study")
    if purpose not in {"main_study", "qualification_pilot", "runtime_probe"} or (
            purpose in {"qualification_pilot", "runtime_probe"} and plan["scope"] != "custom_tau2_hotpot"):
        raise ValueError("Explicit supported supervision purpose required")
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
    if purpose == "qualification_pilot":
        qualification(plan)
    if purpose == "runtime_probe":
        spec = plan.get("runtime_probe")
        config = _pinned_json(spec, "Synthetic runtime probe config")
        stages = plan["stages"]
        start, end = clock(plan["start_time"]), clock(plan["deadline"])
        if (config.get("purpose") != "runtime_probe" or config.get("targets") != [512, 2048, 4096]
                or config.get("criteria") != ["answer", "evidence"]
                or config.get("native_feature_cache") is not True or config.get("response_cache") is not True
                or config.get("new_learning") is not False or type(config.get("benchmark_questions")) is not int
                or config.get("benchmark_questions") != 0
                or not 0 < (end - start).total_seconds() <= 1800
                or len(stages) != 1 or stages[0]["id"] != "probe" or stages[0].get("planned_units") != 22
                or stages[0]["argv"][:5] != [plan["python"], "-B", "-m", "haltiq_tau_hotpot.runtime_probe", "probe"]
                or not stages[0]["model_work"] or not stages[0]["gpu"]
                or {"path": spec["path"], "sha256": spec["sha256"]} not in plan.get("inputs", [])):
            raise ValueError("Only the fixed22-operation synthetic runtime probe is permitted")
        flags = stages[0]["argv"][5:]
        expected = ["--config", spec["path"], "--output", "{run}/probe", "--previous", "{run}",
                    "--deadline", "{deadline}", "--control", "{control}", "--allow-model-calls"]
        if flags != expected or stages[0].get("completion_receipt") != "{run}/probe/completion.json":
            raise ValueError("Synthetic probe arguments must match its owned invocation exactly")
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
        required = READINESS - {"combined_pilot"} if plan.get("purpose") == "qualification_pilot" else READINESS
        blockers.extend("Missing verified readiness: " + key for key in sorted(required - ready.keys()))
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
