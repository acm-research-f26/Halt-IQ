"""Sequential frozen phases with local waits, durable dispatch and no retries."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from pc_strands.campaign import digest, load, save, stamp
from pc_strands.control import GpuLease, resource_snapshot
from pc_strands.windows_job import ProcessContainment
from .handoff import emit_handoff
from .plan import clock, expand, satisfied, verify_frozen, within


def journal(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": stamp(), **value}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_handoff(plan):
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", plan["private_port"])) == 0:
            raise RuntimeError("Private worker port remains open; no handoff")
    lease = Path(plan["gpu_lease"])
    if not lease.is_file():
        raise ValueError("Existing shared OS GPU lease required")
    with GpuLease(lease, "unattended phase handoff check"):
        pass


def descendants_remain(containment, child):
    if os.name != "nt":
        try:
            os.killpg(child.pid, 0)
            return True
        except ProcessLookupError:
            return False
    import ctypes as c
    from ctypes import wintypes as w
    class Accounting(c.Structure):
        _fields_ = [(name, c.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")] + [
            (name, w.DWORD) for name in ("faults", "total", "active", "terminated")]
    value = Accounting()
    api = containment.api
    api.QueryInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p]
    api.QueryInformationJobObject.restype = w.BOOL
    if not api.QueryInformationJobObject(containment.handle, 1, c.byref(value), c.sizeof(value), None):
        raise c.WinError(c.get_last_error())
    return value.active != 0


def wait_owned_tree_clear(containment, child, *, hard_monotonic, stop_deadline=None, sleep=time.sleep):
    """Allow Windows exit accounting to drain, never a lingering owned process."""
    cutoff = min(time.monotonic() + 2, hard_monotonic,
                 stop_deadline if stop_deadline is not None else math.inf)
    while descendants_remain(containment, child):
        remaining = cutoff - time.monotonic()
        if os.name != "nt" or remaining <= 0:
            return False
        sleep(min(.02, remaining))
    return True


def close_owned(containment, child, *, clean=False):
    if os.name != "nt" and child is not None and not clean:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    containment.close()
    if child is not None and child.poll() is None:
        child.kill()
        child.wait(timeout=10)


def mirror_flags(directory, control, run_id):
    control.mkdir(parents=True, exist_ok=True)
    for name in ("pause.request", "stop.request"):
        target = control / name
        if (directory / name).exists():
            if not target.exists():
                save(target, {"source": "supervisor", "run_id": run_id, "at": stamp()})
        elif name == "pause.request" and target.is_file():
            value = load(target)
            if value.get("source") == "supervisor" and value.get("run_id") == run_id:
                target.unlink()


def verify_completion(step, state, completion):
    expected = {"run_id": state["run_id"], "plan_sha256": state["plan_sha256"], "phase": step["id"],
                "terminal": True, "pending_or_uncertain": False, "cleanup_verified": True,
                "planned_units_accounted": True, "planned_units": step["planned_units"]}
    if any(type(completion.get(k)) is not type(v) or completion.get(k) != v for k, v in expected.items()):
        raise ValueError("Worker completion lacks current-run accounting/cleanup proof")
    counts = [completion.get(k) for k in ("completed_units", "failed_units", "not_run_units")]
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != step["planned_units"]:
        raise ValueError("Worker planned-unit accounting does not reconcile")


def admission(plan, directory, publish, *, sampler=None, sleep=time.sleep, cutoff_monotonic=None):
    sampler = sampler or resource_snapshot
    policy = plan["resources"]
    started, stable = time.monotonic(), []
    while True:
        if (datetime.now(timezone.utc) >= clock(plan["deadline"]) or (directory / "stop.request").exists()
                or (cutoff_monotonic is not None and time.monotonic() >= cutoff_monotonic)):
            raise TimeoutError("Stopped before model dispatch")
        if time.monotonic() - started >= policy.get("admission_timeout_seconds", 1200):
            raise TimeoutError("Resource admission expired; no worker dispatched")
        snapshot = sampler()
        observed = [snapshot.get(key) for key in ("available_ram_bytes", "gpu_free_mib", "gpu_temperature_c")]
        measured = all(type(x) in (int, float) and math.isfinite(x) for x in observed)
        good = (not (directory / "pause.request").exists()
                and measured and observed[0] >= policy["startup_ram_gib"] * 1024**3
                and observed[1] >= policy["startup_vram_gib"] * 1024
                and observed[2] < 80)
        if good:
            stable.append({"monotonic": time.monotonic(), "resources": snapshot})
        else:
            stable = []
        publish("waiting_resources", last_resources=snapshot, stable_samples=len(stable))
        if len(stable) >= 4 and stable[-1]["monotonic"] - stable[0]["monotonic"] >= policy["stability_seconds"]:
            return {"samples": stable, "wait_wall_seconds": time.monotonic() - started}
        sleep(10)


def run(directory, *, resume=False, allow_model_calls=False, allow_codex_handoff=False, poll_seconds=1):
    directory = Path(directory).resolve()
    if Path(__file__).resolve().parents[1] != directory / "source":
        raise ValueError("Run the copied supervisor from this run's frozen source directory")
    state, plan = load(directory / "supervisor.json"), load(directory / "plan.json")
    verify_frozen(plan, state, directory)
    shared = GpuLease(Path(plan["study_lease"]), "one unattended study") if plan.get("study_lease") else nullcontext()
    with GpuLease(directory / "supervisor.lock", "local supervisor; not GPU ownership"), shared:
        state, plan = load(directory / "supervisor.json"), load(directory / "plan.json")
        verify_frozen(plan, state, directory)
        if state["status"] != "prepared_not_started" and not resume:
            raise ValueError("Existing execution needs explicit --resume")
        if state["status"] in {"failed", "stopped"} or any(r["status"] != "completed" for r in state["stages"].values()):
            raise ValueError("Unfinished/failed dispatch requires reconciliation; never automatically retry")
        if any(s["model_work"] for s in plan["stages"]) and not allow_model_calls:
            raise ValueError("Explicit --allow-model-calls required")
        callback = plan.get("codex_argv") if allow_codex_handoff else None
        started = time.monotonic()
        invocation = {"started_at": stamp(), "wall_seconds": 0}
        state.setdefault("invocations", []).append(invocation)
        soft_monotonic = started + max(0, (clock(plan["deadline"]) - datetime.now(timezone.utc)).total_seconds())
        hard_monotonic = started + max(0, (clock(plan["hard_deadline"]) - datetime.now(timezone.utc)).total_seconds())
        receipt = directory / "supervisor.json"

        def publish(status, **extra):
            if status != "needs_review":
                state.pop("review_request", None)
            invocation["wall_seconds"] = time.monotonic() - started
            if status in {"completed", "needs_review", "failed", "stopped"}:
                invocation["last_terminal_snapshot_at"] = stamp()
            state.update(status=status, updated_at=stamp(),
                         supervisor_wall_seconds=sum(x["wall_seconds"] for x in state["invocations"]),
                         timing_scope="Supervisor invocation wall includes workers, local waits and callbacks; do not add worker timings again",
                         **extra)
            save(receipt, state)

        def notify(kind, name=None, remote=False):
            event_id = (name or "run") + "-" + kind
            try:
                return emit_handoff(directory, event_id=event_id, kind=kind, stage=name,
                    state_path=receipt, codex_argv=callback if remote else None,
                    codex_cwd=Path(plan["source_root"]))
            except Exception as error:
                # Scientific completion is not rewritten because a notification failed.
                state.setdefault("notification_errors", []).append({"event_id": event_id, "error_type": type(error).__name__})
                save(receipt, state)

        def approved(name, row):
            approval = within(directory / "approvals" / (name + ".json"), directory)
            expected = {"approved": True, "plan_sha256": state["plan_sha256"],
                        "output_sha256": row["output_sha256"]}
            value = load(approval) if approval.is_file() else None
            return type(value) is dict and value.get("approved") is True and value == expected

        def review_request(name, row):
            return {"approval_path": str(directory / "approvals" / (name + ".json")),
                    "expected_payload": {"approved": True, "plan_sha256": state["plan_sha256"],
                                         "output_sha256": row["output_sha256"]}}

        name = None
        try:
            for step in plan["stages"]:
                name = step["id"]
                previous = state["stages"].get(name)
                if previous:
                    for path, sha in previous["output_sha256"].items():
                        if digest(path) != sha:
                            raise ValueError("Completed output changed")
                    if step.get("review_after"):
                        if not approved(name, previous):
                            publish("needs_review", next_stage=name, review_request=review_request(name, previous))
                            notify("action_required", name, True)
                            publish("needs_review", next_stage=name, review_request=review_request(name, previous))
                            return state
                    continue
                if (datetime.now(timezone.utc) >= clock(plan["deadline"]) or time.monotonic() >= soft_monotonic
                        or (directory / "stop.request").exists()):
                    publish("stopped", next_stage=name)
                    notify("action_required", name, True)
                    return state
                while (directory / "pause.request").exists():
                    publish("paused_before_stage", next_stage=name)
                    if (datetime.now(timezone.utc) >= clock(plan["deadline"]) or time.monotonic() >= soft_monotonic
                            or (directory / "stop.request").exists()):
                        raise TimeoutError("Stopped while paused")
                    time.sleep(poll_seconds)
                verify_frozen(plan, state, directory)
                if not all(satisfied(s, directory) for s in step.get("requires", [])):
                    publish("needs_review", next_stage=name)
                    notify("action_required", name, True)
                    return state
                outputs = [dict(s, path=str(within(Path(expand(s["path"], directory)), directory))) for s in step["outputs"]]
                if any(Path(s["path"]).exists() for s in outputs):
                    raise ValueError("Output exists before dispatch; cannot reuse stale success")
                if step["model_work"]:
                    observed = admission(plan, directory, publish, cutoff_monotonic=soft_monotonic)
                    state.setdefault("admissions", {})[name] = observed
                checked = step["model_work"] or step.get("check_handoff", False)
                if checked:
                    check_handoff(plan)
                verify_frozen(plan, state, directory)
                if (time.monotonic() >= soft_monotonic or datetime.now(timezone.utc) >= clock(plan["deadline"])
                        or (directory / "stop.request").exists()):
                    publish("stopped", next_stage=name)
                    notify("action_required", name, True)
                    return state
                control = directory / "workers" / name
                control.mkdir(parents=True, exist_ok=True)
                argv = [expand(x, directory).replace("{deadline}", plan["deadline"]).replace("{control}", str(control)) for x in step["argv"]]
                command = directory / (name + ".command.json")
                save(command, {"argv": argv, "cwd": str(directory / "source")})
                row = {"status": "started", "started_at": stamp(), "output_sha256": {}}
                state["stages"][name] = row
                journal(directory / "events.jsonl", {"kind": "stage_dispatch_pending", "stage": name, "run_id": state["run_id"]})
                publish("running", next_stage=name)
                child, containment, clean = None, ProcessContainment(), False
                try:
                    environment = os.environ.copy()
                    for key in ("TYPESAFE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"):
                        environment.pop(key, None)
                    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONPATH=str(directory / "source"),
                                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                                       HALTIQ_RUN_ID=state["run_id"], HALTIQ_PLAN_SHA256=state["plan_sha256"], HALTIQ_PHASE=name)
                    with (directory / (name + ".log")).open("ab", buffering=0) as log:
                        child = subprocess.Popen([plan["python"], "-B", "-m", "haltiq_supervisor.worker", str(command)],
                            cwd=directory / "source", env=environment, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=0x08004000 if os.name == "nt" else 0, start_new_session=os.name != "nt")
                        containment.attach(child.pid)
                        row["bootstrap_pid"] = child.pid
                        publish("running")
                        # No worker can import a model or spawn before its bootstrap belongs to our OS job.
                        verify_frozen(plan, state, directory)
                        if (time.monotonic() >= soft_monotonic or datetime.now(timezone.utc) >= clock(plan["deadline"])
                                or (directory / "stop.request").exists()):
                            raise TimeoutError("Cutoff before bootstrap release; no worker command started")
                        mirror_flags(directory, control, state["run_id"])
                        child.stdin.write(b"GO\n")
                        child.stdin.close()
                        stop_at = None
                        while child.poll() is None:
                            mirror_flags(directory, control, state["run_id"])
                            now = datetime.now(timezone.utc)
                            if now >= clock(plan["deadline"]) or time.monotonic() >= soft_monotonic or (directory / "stop.request").exists():
                                if not (control / "stop.request").exists():
                                    save(control / "stop.request", {"source": "supervisor", "run_id": state["run_id"], "at": stamp()})
                                stop_at = stop_at or time.monotonic()
                            if (now >= clock(plan["hard_deadline"]) or time.monotonic() >= hard_monotonic
                                    or (stop_at is not None and time.monotonic() - stop_at >= plan.get("stop_grace_seconds", 60))):
                                raise TimeoutError("Safe-stop grace expired; terminate only owned process tree")
                            time.sleep(poll_seconds)
                        row["exit_code"] = child.returncode
                        if child.returncode:
                            raise RuntimeError("Worker exited nonzero; outputs retained and never retried")
                    stop_deadline = stop_at + plan.get("stop_grace_seconds", 60) if stop_at is not None else None
                    if not wait_owned_tree_clear(containment, child, hard_monotonic=hard_monotonic,
                                                 stop_deadline=stop_deadline):
                        raise RuntimeError("Worker left owned descendants; cleanup failure blocks handoff")
                    if checked:
                        check_handoff(plan)
                    for spec in outputs:
                        within(spec["path"], directory)
                        if not satisfied(spec):
                            raise ValueError("Worker output contract failed: " + spec["path"])
                        row["output_sha256"][spec["path"]] = digest(spec["path"])
                    if step["model_work"]:
                        completion = load(expand(step["completion_receipt"], directory))
                        verify_completion(step, state, completion)
                    verify_frozen(plan, state, directory)
                    row.update(status="completed", finished_at=stamp(), owned_tree_clear=True, os_handoff_verified=checked)
                    journal(directory / "events.jsonl", {"kind": "stage_completed", "stage": name, "run_id": state["run_id"]})
                    publish("between_stages")
                    clean = True
                finally:
                    close_owned(containment, child, clean=clean)
                if step.get("review_after"):
                    publish("needs_review", next_stage=name, review_request=review_request(name, row))
                    delivered = notify("action_required", name, True)
                    if not (delivered and delivered.get("status") in {"completed", "prepared_local_only"} and approved(name, row)):
                        publish("needs_review", next_stage=name, review_request=review_request(name, row))
                        return state
                notify("phase_completed", name, step.get("notify_after", False))
            publish("completed", next_stage=None)
            notify("final_completed", remote=True)
            publish("completed", next_stage=None)
            return state
        except BaseException as error:
            publish("failed", error_type=type(error).__name__)
            notify("failure", name, True)
            publish("failed", error_type=type(error).__name__)
            raise
