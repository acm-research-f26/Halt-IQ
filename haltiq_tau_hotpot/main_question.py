"""One fresh question and all eight methods in a separately owned process.

The outer frozen supervisor owns this descendant before it starts. This module
imports no model framework; public task/model work starts only after a new
four-sample admission and LocalRuntime's independent shared GPU acquisition.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

from pc_strands.control import GpuLease, ResourcePressure, RunControl, resource_snapshot
from .hobson import file_hash
from .journal import CallJournal, durable_write, fingerprint
from .protocol import ARMS, execution_cutoff, load_main, pinned, read, verify_dispatch
from .runtime import LocalRuntime, port_open
from .worker import bare_case


def stamp():
    return datetime.now(timezone.utc).isoformat()


def question_directory(output, index):
    return Path(output) / "questions" / f"{index:03d}"


def publish_signal(path, value):
    """A reader can see only complete fsynced JSON; a signal never overwrites."""
    temporary = path.with_name(path.name + ".publishing")
    durable_write(temporary, value)
    os.link(temporary, path)  # Atomic exclusive publication on the same volume.
    temporary.unlink()


def process_identity(pid):
    """Read a live process identity without a shell, model import or PID cleanup."""
    if os.name != "nt":
        root = Path("/proc") / str(pid)
        if not root.is_dir():
            raise ProcessLookupError("Owned process is absent")
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        created = boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return {"pid": pid, "parent_pid": int(fields[1]), "created_at": datetime.fromtimestamp(created, timezone.utc).isoformat(),
            "executable": str((root / "exe").resolve(strict=True)), "active": fields[0] != "Z"}
    import ctypes as c
    from ctypes import wintypes as w
    class Entry(c.Structure):
        _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("pid", w.DWORD), ("heap", c.c_size_t),
            ("module", w.DWORD), ("threads", w.DWORD), ("parent", w.DWORD), ("priority", w.LONG),
            ("flags", w.DWORD), ("name", w.WCHAR * 260)]
    api = c.WinDLL("kernel32", use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes, api.CreateToolhelp32Snapshot.restype = [w.DWORD, w.DWORD], w.HANDLE
    api.Process32FirstW.argtypes = api.Process32NextW.argtypes = [w.HANDLE, c.POINTER(Entry)]
    api.Process32FirstW.restype = api.Process32NextW.restype = w.BOOL
    api.CloseHandle.argtypes, api.CloseHandle.restype = [w.HANDLE], w.BOOL
    snapshot, parent = api.CreateToolhelp32Snapshot(2, 0), None
    if snapshot == c.c_void_p(-1).value:
        raise c.WinError(c.get_last_error())
    try:
        entry = Entry(); entry.size = c.sizeof(entry)
        present = api.Process32FirstW(snapshot, c.byref(entry))
        while present:
            if entry.pid == pid:
                parent = int(entry.parent); break
            present = api.Process32NextW(snapshot, c.byref(entry))
    finally:
        api.CloseHandle(snapshot)
    if parent is None:
        raise ProcessLookupError("Owned process exited before identity verification")
    api.OpenProcess.argtypes, api.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        if c.get_last_error() == 87:
            raise ProcessLookupError("Owned process exited before handle acquisition")
        raise c.WinError(c.get_last_error())
    try:
        api.GetProcessTimes.argtypes, api.GetProcessTimes.restype = [w.HANDLE] + [c.POINTER(w.FILETIME)] * 4, w.BOOL
        times = [w.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(c.byref(t) for t in times)):
            raise c.WinError(c.get_last_error())
        api.QueryFullProcessImageNameW.argtypes, api.QueryFullProcessImageNameW.restype = [w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD)], w.BOOL
        size, buffer = w.DWORD(32768), c.create_unicode_buffer(32768)
        if not api.QueryFullProcessImageNameW(handle, 0, buffer, c.byref(size)):
            raise c.WinError(c.get_last_error())
        api.GetExitCodeProcess.argtypes, api.GetExitCodeProcess.restype = [w.HANDLE, c.POINTER(w.DWORD)], w.BOOL
        code = w.DWORD()
        if not api.GetExitCodeProcess(handle, c.byref(code)):
            raise c.WinError(c.get_last_error())
        ticks = times[0].dwLowDateTime + (times[0].dwHighDateTime << 32)
        return {"pid": pid, "parent_pid": parent,
            "created_at": datetime.fromtimestamp(ticks / 10_000_000 - 11644473600, timezone.utc).isoformat(),
            "executable": str(Path(buffer.value).resolve()), "active": code.value == 259}
    finally:
        api.CloseHandle(handle)


def validate_process_binding(launcher, worker, started, dispatched, observed_at, *, launcher_pid, parent_pid, executables, compare_parent=None):
    """A venv redirector and its actual Python worker have distinct identities."""
    times = [datetime.fromisoformat(v) for v in (dispatched["at"], launcher["created_at"],
        worker["created_at"], started["at"], observed_at)]
    allowed = {str(Path(p).resolve()).casefold() for p in executables}
    if (any(t.tzinfo is None for t in times) or times != sorted(times)
            or any(started.get(k) != dispatched.get(k) for k in ("run_id", "plan_sha256", "question_index"))
            or dispatched.get("compare_parent_pid") != parent_pid
            or launcher.get("pid") != launcher_pid or launcher.get("parent_pid") != parent_pid
            or worker.get("pid") != started.get("pid") or worker.get("parent_pid") != started.get("parent_pid")
            or worker.get("parent_pid") != (parent_pid if worker["pid"] == launcher_pid else launcher_pid)
            or launcher.get("active") is not True or worker.get("active") is not True
            or any(str(Path(p["executable"]).resolve()).casefold() not in allowed for p in (launcher,worker))):
        raise ValueError("Live question worker ancestry/creation/executable differs from its owned launcher")
    if compare_parent is not None:
        created = datetime.fromisoformat(compare_parent["created_at"])
        if (compare_parent.get("pid") != parent_pid or compare_parent.get("active") is not True
                or created.tzinfo is None or created > times[0]):
            raise ValueError("Compare parent identity changed before its owned question dispatch")


def await_process_binding(child, directory, dispatched, *, executables, timeout_seconds=300):
    """Verify while worker waits before admission; never authorize a raced PID."""
    end = time.monotonic() + timeout_seconds
    while not (directory / "started.json").is_file():
        if child.poll() is not None or time.monotonic() >= end:
            raise RuntimeError("Owned question exited or timed out before its identity handshake")
        time.sleep(.05)
    started = read(directory / "started.json")
    observed = stamp()
    compare_parent, launcher, worker = process_identity(os.getpid()), process_identity(child.pid), process_identity(started["pid"])
    validate_process_binding(launcher, worker, started, dispatched, observed, launcher_pid=child.pid,
        parent_pid=os.getpid(), executables=executables, compare_parent=compare_parent)
    value = {"schema": "haltiq-main-question-process-binding-v1", "launcher": launcher, "worker": worker,
        "compare_parent_pid": os.getpid(), "compare_parent": compare_parent, "observed_at": observed,
        "started_sha256": file_hash(directory / "started.json"), "dispatch_sha256": file_hash(directory / "dispatch-pending.json"),
        "executables": list(executables), "executable_sha256": {str(Path(p).resolve()): file_hash(p) for p in executables},
        "verified_before_admission": True}
    publish_signal(directory / "identity-verified.json", value)
    return value


def exited_process_binding(binding):
    """No PID is stopped: a reused PID is distinct from the observed process."""
    result = {}
    for role in ("launcher", "worker"):
        original = binding[role]
        try:
            current = process_identity(original["pid"])
        except ProcessLookupError:
            current = None
        absent = current is None or current["created_at"] != original["created_at"]
        result[role] = {"pid": original["pid"], "original_created_at": original["created_at"],
            "original_identity_absent": absent, "current_identity": current}
    if not all(v["original_identity_absent"] for v in result.values()):
        raise RuntimeError("Observed owned question process remains after launcher exit")
    return result


class QuestionRuntime(LocalRuntime):
    """One writer placement, followed by one native placement, per question."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_start = len(self.journal.records)

    def roles(self):
        return [e["role"] for e in self.journal.records[self.record_start:] if e["event"] == "pending"]

    def _writer_ready(self):
        roles = self.roles()
        if "native_startup" in roles:
            raise ResourcePressure("Question writer-to-native placement order is exhausted")
        if "writer_load" in roles:
            resident = self._http("/api/ps", timeout=5).get("models")
            if (not self.writer_loaded or not isinstance(resident, list) or len(resident) != 1
                    or resident[0].get("digest") != self.config["writer_digest"]):
                raise ResourcePressure("Question writer was evicted; never perform an unbudgeted reload")
            return
        return super()._writer_ready()

    def _native_ready(self):
        if self.backend is None and "native_startup" in self.roles():
            raise ResourcePressure("Question native placement was lost; never reload it")
        return super()._native_ready()

    def assert_envelope(self):
        roles = self.roles()
        if (roles.count("writer_load") > 1 or roles.count("native_startup") > 1
                or "native_startup" in roles and "writer_load" in roles[roles.index("native_startup") + 1:]):
            raise RuntimeError("Question exceeded its frozen placement envelope")


def admission(config, control, output, *, sampler=resource_snapshot, sleep=None, clock=None, lease_factory=GpuLease):
    """One attempt, no retry:11GiB RAM/9GiB VRAM for four samples over30s."""
    clock = clock or (lambda: datetime.now(timezone.utc))
    sleep = sleep or control._sleep
    result = {"schema": "haltiq-main-question-admission-v1", "status": "BLOCKED_ADMISSION",
        "samples": [], "ram_gib": 11, "vram_gib": 9, "gpu_lease_released": False,
        "private_port_closed": False, "model_calls": 0, "automatic_retry": False}
    try:
        if port_open(11435):
            raise ResourcePressure("Private11435 is already owned")
        with lease_factory(Path(config["gpu_lease"]), "main question startup admission"):
            for i in range(4):
                if i:
                    sleep(10)
                control._check_stop()
                observed = sampler()
                ram, vram = observed.get("available_ram_bytes"), observed.get("gpu_free_mib")
                row = {"at": clock().isoformat(), "available_ram_gib": ram / 1024**3 if type(ram) in (int,float) else None,
                    "free_vram_gib": vram / 1024 if type(vram) in (int,float) else None,
                    "gpu_temperature_c": observed.get("gpu_temperature_c")}
                row = {k: v if k == "at" or type(v) in (int, float) and math.isfinite(v) else None for k, v in row.items()}
                result["samples"].append(row)
                if (row["available_ram_gib"] is None or row["free_vram_gib"] is None or row["gpu_temperature_c"] is None
                        or row["available_ram_gib"] < 11 or row["free_vram_gib"] < 9
                        or not 0 <= row["gpu_temperature_c"] < 80 or port_open(11435)):
                    raise ResourcePressure("Question startup reserves or private port did not pass")
        with lease_factory(Path(config["gpu_lease"]), "verify main question admission release"):
            pass
        result["gpu_lease_released"] = True
        result["private_port_closed"] = not port_open(11435)
        times = [datetime.fromisoformat(r["at"]) for r in result["samples"]]
        if (not result["private_port_closed"] or any(t.tzinfo is None for t in times)
                or any((b-a).total_seconds() < 10 for a,b in zip(times,times[1:]))
                or (times[-1]-times[0]).total_seconds() < 30):
            raise ResourcePressure("Question admission timing/ownership did not pass")
        result["status"] = "PASS_ADMISSION"
    except BaseException as error:
        result.update(error_type=type(error).__name__, reason=str(error))
        durable_write(output, result)
        raise
    durable_write(output, result)
    return result


def run_question(args, context, control):
    config = context["config"]
    case = context["cohorts"]["main_fresh"][args.question_index]
    qid, public = case["public"]["id"], case["public"]
    learning = read(args.previous / "train/learning.json")
    criterion = learning["criteria"]["answer"]
    trained = args.previous / "train/learned/answer-head.safetensors"
    if (learning["status"] != "completed" or not learning["frozen_torso_adapter_files_unchanged"]
            or file_hash(trained) != criterion["trained_head_sha256"]):
        raise ValueError("Genuine trained head and separate calibration proof required")
    runtime, rows = QuestionRuntime(config, args.output, control), []
    try:
        with runtime:
            from .agent import run_episode
            ranked = runtime.rank(public)
            first = run_episode(bare_case(case), runtime.generate, arm="fixed1", max_drafts=1,
                                passage_indices=ranked["indices"], cap_fallback="last_valid")
            draft = first["submitted"]
            durable_write(args.output / "initial" / (qid + ".json"), first)
            for arm in ARMS:
                def writer(state, feedback):
                    if state["previous_draft"] is None:
                        if draft is None:
                            raise ValueError("Shared initial draft failed; never redispatch it")
                        return deepcopy(draft)
                    return runtime.generate(state, feedback)
                def reviewer(state, proposed):
                    selection = runtime.rank(public, proposed["answer"])
                    evidence = [{"title": public["context"][i][0], "sentences": [
                        {"index": n, "text": text} for n,text in enumerate(public["context"][i][1])]}
                        for i in selection["indices"]]
                    return runtime.generate({"question": public["question"], "evidence": evidence, "draft": proposed}, review=True)
                def gate(state, proposed, component):
                    if component == "evidence":
                        if arm != "decomposed":
                            raise ValueError("Only decomposed uses an evidence gate")
                        head, temperature = None, learning["released_evidence_calibration"]["temperature"]
                    else:
                        head = trained if arm == "trained" else None
                        temperature = criterion[("trained" if arm == "trained" else "released") + "_calibration"]["temperature"]
                    return runtime.observe(public, proposed, criterion=component, head_path=head, temperature=temperature)
                row = run_episode(bare_case(case), writer, reviewer, gate, arm=arm, max_drafts=10,
                    threshold=.8, passage_indices=ranked["indices"], timeout_seconds=3600, cap_fallback="last_valid")
                row.update(shared_initial_draft_sha256=fingerprint(draft), shared_initial_physical_calls=0)
                durable_write(args.output / "episodes" / (qid + "-" + arm + ".json"), row)
                runtime.assert_envelope()
                rows.append(row)
    finally:
        terminal = {"cleanup": runtime.cleanup, "calls": runtime.journal.verify(), "timings": runtime.timings,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds}
        durable_write(question_directory(args.output, args.question_index) / "runtime-terminal.json", terminal)
    return {"question_id": qid, "initial": draft, "episodes": rows, "terminal": terminal,
        "record_start": runtime.record_start, "record_end": len(runtime.journal.records),
        "record_slice_sha256": fingerprint(runtime.journal.records[runtime.record_start:]),
        "record_prefix_sha256": fingerprint(runtime.journal.records[:runtime.record_start])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output", "previous", "control"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--question-index", type=int, required=True)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_calls:
        raise ValueError("Explicit child model-call opt-in required")
    started, at = time.monotonic(), stamp()
    plan, state = read(args.previous / "plan.json"), read(args.previous / "supervisor.json")
    args.phase = "compare"
    if Path(__file__).resolve().parents[1] != args.previous.resolve() / "source":
        raise ValueError("Question worker must use its actual frozen source")
    if (state["run_id"] != os.environ.get("HALTIQ_RUN_ID")
            or state["plan_sha256"] != os.environ.get("HALTIQ_PLAN_SHA256")
            or os.environ.get("HALTIQ_PHASE") != "compare"
            or args.config.resolve() != Path(plan["main_protocol"]["path"]).resolve()
            or args.output.resolve() != args.previous.resolve() / "compare"
            or args.control.resolve() != args.previous.resolve() / "workers/compare"
            or args.deadline != plan["deadline"] or args.question_index < 0):
        raise ValueError("Question identity handshake must belong to its current frozen parent")
    directory = question_directory(args.output, args.question_index)
    pending = read(directory / "dispatch-pending.json")
    if (pending["run_id"] != state["run_id"] or pending["plan_sha256"] != state["plan_sha256"]
            or pending["question_index"] != args.question_index):
        raise ValueError("Question dispatch does not match its current parent owner")
    publish_signal(directory / "started.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "at": at,
        "run_id": state["run_id"], "plan_sha256": state["plan_sha256"], "question_index": args.question_index})
    limit = time.monotonic() + min(300, max(0,
        (datetime.fromisoformat(plan["model_deadline"]) - datetime.now(timezone.utc)).total_seconds()))
    while not (directory / "identity-verified.json").is_file():
        if time.monotonic() >= limit:
            raise TimeoutError("Parent did not verify the current worker before admission")
        time.sleep(.05)
    identity = read(directory / "identity-verified.json")
    if (identity.get("worker", {}).get("pid") != os.getpid()
            or identity["worker"].get("parent_pid") != os.getppid()
            or identity.get("started_sha256") != file_hash(directory / "started.json")
            or identity.get("dispatch_sha256") != file_hash(directory / "dispatch-pending.json")
            or identity.get("verified_before_admission") is not True):
        raise ValueError("Parent identity approval belongs to a different question worker")
    # The live identity handshake precedes full recursive source/pilot hashing.
    # It authorizes no model work: every existing scientific guard still runs.
    verify_dispatch(args, plan, state)
    config = pinned(args.config, plan["main_protocol"]["sha256"])
    context = load_main(config)
    if (config.get("compare_process_layout") != "per_question"
            or not 0 <= args.question_index < context["fresh_n"]
            or pending["question_id"] != context["cohorts"]["main_fresh"][args.question_index]["public"]["id"]):
        raise ValueError("Question worker must match the frozen layout and selected cohort")
    control = RunControl(args.control, execution_cutoff(config, args.deadline, "compare"))
    try:
        import importlib.util
        for module, expected in config["runtime_modules"].items():
            found = importlib.util.find_spec(module)
            if found is None or Path(found.origin).resolve() != Path(expected).resolve():
                raise ValueError("Question imported a different pinned runtime package")
        admission(config, control, directory / "admission.json")
        result = run_question(args, context, control)
        if result["terminal"]["cleanup"]["verified"] is not True:
            raise RuntimeError("Question runtime cleanup failed")
        bindings = {"initial/" + result["question_id"] + ".json": file_hash(args.output / "initial" / (result["question_id"] + ".json"))}
        bindings.update({"episodes/" + result["question_id"] + "-" + arm + ".json":
            file_hash(args.output / "episodes" / (result["question_id"] + "-" + arm + ".json")) for arm in ARMS})
        value = {"schema": "haltiq-main-question-v1", "terminal": True, "cleanup_verified": True,
            "pending_or_uncertain": False, "run_id": state["run_id"], "plan_sha256": state["plan_sha256"],
            "config_sha256": file_hash(args.config), "source_sha256": fingerprint(state["source_sha256"]),
            "question_index": args.question_index, "question_id": result["question_id"], "pid": os.getpid(),
            "parent_pid": os.getppid(), "launcher_pid": identity["launcher"]["pid"],
            "identity_sha256": file_hash(directory / "identity-verified.json"),
            "started_at": at, "finished_at": stamp(), "wall_seconds": time.monotonic() - started,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds,
            "physical_record_start": result["record_start"], "physical_record_end": result["record_end"],
            "physical_record_slice_sha256": result["record_slice_sha256"],
            "physical_record_prefix_sha256": result["record_prefix_sha256"], "outputs_sha256": bindings,
            "admission_sha256": file_hash(directory / "admission.json"),
            "runtime_terminal_sha256": file_hash(directory / "runtime-terminal.json")}
        durable_write(directory / "receipt.json", value)
    except BaseException as error:
        durable_write(directory / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "pid": os.getpid(), "no_automatic_retry": True, "wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
