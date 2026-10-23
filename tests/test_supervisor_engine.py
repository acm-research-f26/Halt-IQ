"""Real short owned Python fixtures; no models, GPU probes, APIs or Codex calls."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from haltiq_supervisor.engine import admission, mirror_flags
from haltiq_supervisor.plan import prepare, source_files
from pc_strands.campaign import digest, load, save

REPO = Path(__file__).resolve().parents[1]
PROFILE = {"ram_gib": 7, "vram_gib": 3, "startup_ram_gib": 8.5,
    "startup_vram_gib": 9, "stability_seconds": 30, "minimum_rest_seconds": 5,
    "cpu_threads": 4, "gpu_owners": 1, "duty_cycle": .5, "admission_timeout_seconds": 1200}


def running(pid):
    """Read-only liveness for an owned fixture; os.kill(pid,0) is unsafe on Windows."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259
        finally:
            api.CloseHandle(handle)
    status = Path(f"/proc/{pid}/stat")
    if status.exists() and status.read_text().split(")", 1)[1].split()[0] == "Z":
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class FrozenEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "run"
        self.plan_path = self.root / "plan.json"
        now = datetime.now(timezone.utc)
        self.plan = {"version": 1, "scope": "offline_supervision_fixture", "status": "ready",
            "source_root": str(REPO), "python": sys.executable,
            "source_paths": ["haltiq_supervisor", "pc_strands"], "inputs": [], "readiness": {},
            "resources": deepcopy(PROFILE), "deadline": (now + timedelta(seconds=40)).isoformat(),
            "hard_deadline": (now + timedelta(seconds=45)).isoformat(), "stop_grace_seconds": 2,
            "stages": []}

    def step(self, name, *, code=None, review=False, requires=None):
        script = code or (
            "import json,os,sys; from pathlib import Path; "
            "p=Path(sys.argv[1]); counter=Path(sys.argv[2]); "
            "counter.write_text((counter.read_text() if counter.exists() else '')+'dispatch\\n'); "
            "p.write_text(json.dumps({'ok':True,'cwd':str(Path.cwd()),'phase':os.environ['HALTIQ_PHASE']}))")
        return {"id": name, "model_work": False, "gpu": False,
            "argv": [sys.executable, "-B", "-c", script, "{run}/" + name + ".json", "{run}/" + name + ".dispatches", "{control}"],
            "outputs": [{"path": "{run}/" + name + ".json", "json_equals": {"ok": True}}],
            "requires": requires or [], "review_after": review}

    def adopt(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        return prepare(self.plan_path, self.directory)

    def execute(self, *, resume=False):
        command = [sys.executable, "-B", "-m", "haltiq_supervisor", "run", "--run-dir", str(self.directory)]
        if resume:
            command.append("--resume")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(self.directory / "source"))
        return subprocess.run(command, cwd=self.directory / "source", env=environment,
            capture_output=True, text=True, timeout=20,
            creationflags=0x08000000 if os.name == "nt" else 0)

    def test_two_phase_autochain_uses_frozen_source_and_never_starts_codex(self):
        self.plan["stages"] = [self.step("first"), self.step("second", requires=[
            {"path": "{run}/first.json", "json_equals": {"ok": True}}])]
        self.adopt()
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        state = load(self.directory / "supervisor.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(list(state["stages"]), ["first", "second"])
        self.assertTrue(all(row["owned_tree_clear"] for row in state["stages"].values()))
        for name in ("first", "second"):
            output = load(self.directory / (name + ".json"))
            self.assertEqual(output["cwd"], str(self.directory / "source"))
            self.assertEqual(output["phase"], name)
        events = [json.loads(line) for line in (self.directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["kind"] for e in events], ["stage_dispatch_pending", "stage_completed"] * 2)
        outbox = [load(p) for p in (self.directory / "outbox").glob("*.json")]
        self.assertTrue(outbox)
        self.assertTrue(all(e["dispatch_intents"] == 0 and not e["dispatch_attempted"] for e in outbox))

    def test_completed_resume_skips_exact_output_and_tampering_never_redispatches(self):
        self.plan["stages"] = [self.step("first")]
        self.adopt()
        self.assertEqual(self.execute().returncode, 0)
        self.assertEqual(self.execute(resume=True).returncode, 0)
        counter = self.directory / "first.dispatches"
        self.assertEqual(counter.read_text().splitlines(), ["dispatch"])
        (self.directory / "first.json").write_text('{"ok":true,"changed":true}')
        result = self.execute(resume=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Completed output changed", result.stderr)
        self.assertEqual(counter.read_text().splitlines(), ["dispatch"])

    def test_unknown_started_dispatch_blocks_resume_without_any_new_child(self):
        self.plan["stages"] = [self.step("first")]
        self.adopt()
        receipt = self.directory / "supervisor.json"
        state = load(receipt)
        state.update(status="running", stages={"first": {"status": "started", "output_sha256": {}}})
        save(receipt, state)
        result = self.execute(resume=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("never automatically retry", result.stderr)
        self.assertFalse((self.directory / "first.dispatches").exists())

    def test_nonzero_child_is_terminal_and_no_retry_is_inferred_from_exit(self):
        code = ("import sys;from pathlib import Path;p=Path(sys.argv[2]);"
                "p.write_text((p.read_text() if p.exists() else '')+'attempt\\n');raise SystemExit(7)")
        self.plan["stages"] = [self.step("first", code=code), self.step("second")]
        self.adopt()
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        state = load(self.directory / "supervisor.json")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stages"]["first"]["exit_code"], 7)
        self.assertFalse((self.directory / "second.json").exists())
        self.assertNotEqual(self.execute(resume=True).returncode, 0)
        self.assertEqual((self.directory / "first.dispatches").read_text().splitlines(), ["attempt"])

    def test_stale_existing_output_blocks_dispatch_even_if_contract_would_pass(self):
        self.plan["stages"] = [self.step("first")]
        self.adopt()
        (self.directory / "first.json").write_text('{"ok":true}')
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output exists before dispatch", result.stderr)
        self.assertEqual(load(self.directory / "supervisor.json")["stages"], {})
        self.assertFalse((self.directory / "first.dispatches").exists())

    def test_review_gate_requires_exact_plan_and_completed_output_binding(self):
        self.plan["stages"] = [self.step("first", review=True), self.step("second")]
        self.adopt()
        self.assertEqual(self.execute().returncode, 0)
        state = load(self.directory / "supervisor.json")
        self.assertEqual(state["status"], "needs_review")
        self.assertEqual(self.execute(resume=True).returncode, 0)
        self.assertFalse((self.directory / "second.json").exists())
        approval = self.directory / "approvals/first.json"
        approval.parent.mkdir()
        save(approval, {"approved": True, "plan_sha256": "wrong", "output_sha256": {}})
        self.assertEqual(self.execute(resume=True).returncode, 0)
        self.assertFalse((self.directory / "second.json").exists())
        save(approval, {"approved": True, "plan_sha256": state["plan_sha256"],
                        "output_sha256": state["stages"]["first"]["output_sha256"]})
        result = self.execute(resume=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load(self.directory / "supervisor.json")["status"], "completed")
        self.assertEqual((self.directory / "first.dispatches").read_text().splitlines(), ["dispatch"])

    def test_changed_copied_source_is_rejected_before_any_stage(self):
        self.plan["stages"] = [self.step("first")]
        self.adopt()
        target = self.directory / "source/pc_strands/control.py"
        target.write_bytes(target.read_bytes() + b"\n# synthetic tamper\n")
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Frozen source changed", result.stderr)
        self.assertFalse((self.directory / "first.dispatches").exists())

    def test_soft_cutoff_is_forwarded_and_no_later_stage_dispatches(self):
        now = datetime.now(timezone.utc)
        self.plan.update(deadline=(now + timedelta(seconds=2)).isoformat(),
                         hard_deadline=(now + timedelta(seconds=7)).isoformat())
        code = ("import json,sys,time;from pathlib import Path;control=Path(sys.argv[3]);"
                "\nwhile not (control/'stop.request').exists():time.sleep(.01)"
                "\nPath(sys.argv[1]).write_text(json.dumps({'ok':True,'saw_stop':True}))")
        self.plan["stages"] = [self.step("first", code=code), self.step("second")]
        self.adopt()
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        state = load(self.directory / "supervisor.json")
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["stages"]["first"]["status"], "completed")
        self.assertTrue(load(self.directory / "first.json")["saw_stop"])
        self.assertFalse((self.directory / "second.json").exists())

    def test_hard_cutoff_kills_owned_stubborn_worker_and_blocks_later_phase(self):
        now = datetime.now(timezone.utc)
        self.plan.update(deadline=(now + timedelta(seconds=2)).isoformat(),
            hard_deadline=(now + timedelta(seconds=3)).isoformat(), stop_grace_seconds=.2)
        code = ("import json,os,sys,time;from pathlib import Path;"
                "Path(sys.argv[1]).write_text(json.dumps({'ok':True,'pid':os.getpid()}));time.sleep(30)")
        self.plan["stages"] = [self.step("first", code=code), self.step("second")]
        self.adopt()
        started = time.monotonic()
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn("Safe-stop grace expired", result.stderr)
        self.assertFalse(running(load(self.directory / "first.json")["pid"]))
        self.assertFalse((self.directory / "second.json").exists())

    def test_lingering_owned_grandchild_blocks_handoff_and_unrelated_fixture_survives(self):
        code = ("import json,subprocess,sys,os;from pathlib import Path;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,"
            "creationflags=0x08000000 if os.name=='nt' else 0);"
            "Path(sys.argv[1]).write_text(json.dumps({'ok':True,'grandchild_pid':p.pid}))")
        self.plan["stages"] = [self.step("first", code=code), self.step("second")]
        self.adopt()
        sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x08000000 if os.name == "nt" else 0)
        try:
            result = self.execute()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("left owned descendants", result.stderr)
            self.assertFalse((self.directory / "second.json").exists())
            pid = load(self.directory / "first.json")["grandchild_pid"]
            until = time.monotonic() + 3
            while running(pid) and time.monotonic() < until:
                time.sleep(.05)
            self.assertFalse(running(pid))
            self.assertIsNone(sentinel.poll())
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=5)

    def test_preparation_uses_one_plan_byte_snapshot_and_copies_explicit_configuration(self):
        self.plan["stages"] = [self.step("first")]
        self.plan["source_paths"].append("configs/tau2/pilot.example.json")
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        expected = self.plan_path.read_bytes()
        original = source_files
        def mutate_live_plan(value):
            replacement = deepcopy(self.plan)
            replacement["stages"][0]["argv"] = [sys.executable, "-c", "raise RuntimeError('unreviewed')"]
            self.plan_path.write_text(json.dumps(replacement), encoding="utf-8")
            return original(value)
        with patch("haltiq_supervisor.plan.source_files", side_effect=mutate_live_plan):
            state = prepare(self.plan_path, self.directory)
        self.assertEqual((self.directory / "plan.json").read_bytes(), expected)
        self.assertNotEqual(digest(self.plan_path), state["plan_sha256"])
        self.assertEqual(digest(REPO / "configs/tau2/pilot.example.json"),
                         digest(self.directory / "source/configs/tau2/pilot.example.json"))


class AdmissionAndFlagTests(unittest.TestCase):
    def plan(self):
        return {"resources": deepcopy(PROFILE),
                "deadline": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}

    def test_nonfinite_telemetry_and_pressure_reset_stability_before_fresh30second_admission(self):
        healthy = {"available_ram_bytes": 10 * 1024**3, "gpu_free_mib": 12 * 1024, "gpu_temperature_c": 40}
        samples = [healthy, healthy, {"available_ram_bytes": None, "gpu_free_mib": math.nan,
                   "gpu_temperature_c": 40}, healthy, healthy, healthy, healthy]
        clock, published = {"now": 0}, []
        def sleep(seconds): clock["now"] += seconds
        with tempfile.TemporaryDirectory() as temp, patch("haltiq_supervisor.engine.time.monotonic", side_effect=lambda: clock["now"]):
            result = admission(self.plan(), Path(temp), lambda status, **kw: published.append(kw),
                               sampler=iter(samples).__next__, sleep=sleep)
        self.assertEqual(result["wait_wall_seconds"], 60)
        self.assertEqual(len(result["samples"]), 4)
        self.assertEqual(published[2]["stable_samples"], 0)
        self.assertEqual(result["samples"][-1]["monotonic"] - result["samples"][0]["monotonic"], 30)

    def test_monotonic_cutoff_ends_resource_wait_without_dispatch_or_real_sleep(self):
        clock = {"now": 0}
        def sleep(seconds): clock["now"] += seconds
        with tempfile.TemporaryDirectory() as temp, patch("haltiq_supervisor.engine.time.monotonic", side_effect=lambda: clock["now"]):
            with self.assertRaisesRegex(TimeoutError, "before model dispatch"):
                admission(self.plan(), Path(temp), lambda *a, **kw: None, sampler=lambda: {},
                          sleep=sleep, cutoff_monotonic=15)
        self.assertEqual(clock["now"], 20)

    def test_pause_mirroring_removes_only_supervisor_owned_pause_and_never_foreign_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control = root / "worker"
            (root / "pause.request").write_text("{}")
            mirror_flags(root, control, "this-run")
            self.assertEqual(load(control / "pause.request")["run_id"], "this-run")
            (root / "pause.request").unlink()
            mirror_flags(root, control, "this-run")
            self.assertFalse((control / "pause.request").exists())
            save(control / "pause.request", {"source": "user", "run_id": "other-run"})
            mirror_flags(root, control, "this-run")
            self.assertTrue((control / "pause.request").exists())


if __name__ == "__main__":
    unittest.main()
