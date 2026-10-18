"""Protocol tests using only owned, short-lived Python children; no models/network."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from pc_strands.campaign import choose_fresh, condition, digest, load, run, save


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.repository = self.root / "repository"
        (self.repository / "pc_strands").mkdir(parents=True)
        (self.repository / "pc_strands" / "campaign.py").write_text("# frozen fixture\n")
        self.deadline = (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()
        self.plan = {"working_directory": str(self.repository), "evaluation_deadline": self.deadline,
                     "steps": [], "prerequisites": {}}
        self.path = self.root / "plan.json"
        self.output = self.root / "run"

    def write_plan(self):
        self.path.write_text(json.dumps(self.plan), encoding="utf-8")

    def step(self, name, target, *, code=None, **extra):
        return {"id": name, "calls": False,
                "argv": [sys.executable, "-B", "-c", code or "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(str(Path.cwd()))", str(target)],
                "outputs": [{"path": str(target)}], **extra}

    def test_source_frozen_child_cwd_and_resume_exact_outputs(self):
        target = self.root / "result.txt"
        self.plan["steps"] = [self.step("writer", target)]
        self.write_plan()
        result = run(self.path, self.output, poll_seconds=.01)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(Path(target.read_text()), self.output / "source")
        self.assertEqual(result["plan_sha256"], digest(self.path))
        # A later live-source change cannot mutate the campaign's source.
        (self.repository / "pc_strands" / "campaign.py").write_text("# changed live\n")
        self.assertEqual(run(self.path, self.output, resume=True)["status"], "completed")
        target.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "output changed"):
            run(self.path, self.output, resume=True)

    def test_prerequisite_waits_for_explicit_verified_contract(self):
        marker, target = self.root / "health.json", self.root / "result"
        self.plan["prerequisites"]["health"] = {"path": str(marker), "json_equals": {"cuda.available": True}}
        self.plan["steps"] = [self.step("work", target, requires=["health"])]
        self.write_plan()
        def ready():
            time.sleep(.05)
            marker.write_text('{"cuda":{"available":true}}')
        worker = threading.Thread(target=ready)
        worker.start()
        try:
            result = run(self.path, self.output, poll_seconds=.01)
        finally:
            worker.join()
        self.assertEqual(result["stages"]["work"]["status"], "completed")

    def test_unknown_prerequisite_never_dispatches(self):
        target = self.root / "must-not-exist"
        self.plan["steps"] = [self.step("work", target, requires=["unverified_runtime"])]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "explicit file contract"):
            run(self.path, self.output)
        self.assertFalse(target.exists())
        self.assertEqual(load(self.output / "campaign.json")["stages"], {})

    def test_failed_child_and_uncertain_dispatch_never_retry(self):
        target = self.root / "unused"
        self.plan["steps"] = [self.step("bad", target, code="raise SystemExit(7)")]
        self.write_plan()
        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            run(self.path, self.output, poll_seconds=.01)
        self.assertEqual(load(self.output / "campaign.json")["stages"]["bad"]["exit_code"], 7)
        with self.assertRaisesRegex(ValueError, "never auto-retry"):
            run(self.path, self.output, resume=True)

    def test_inference_requires_explicit_offset_deadline(self):
        target = self.root / "unused"
        self.plan["steps"] = [dict(self.step("missing_deadline", target), calls=True)]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "explicitly receive"):
            run(self.path, self.output)
        self.assertFalse(target.exists())

    def test_cutoff_prevents_new_child(self):
        target = self.root / "unused"
        self.plan["evaluation_deadline"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        self.plan["steps"] = [self.step("late", target)]
        self.write_plan()
        self.assertEqual(run(self.path, self.output)["status"], "stopped_before_new_stage")
        self.assertFalse(target.exists())

    def test_hard_cutoff_terminates_only_owned_child(self):
        self.plan["evaluation_deadline"] = (datetime.now(timezone.utc) + timedelta(seconds=.4)).isoformat()
        self.plan["hard_deadline"] = (datetime.now(timezone.utc) + timedelta(seconds=.8)).isoformat()
        self.plan["steps"] = [self.step("slow", self.root / "unused", code="import time; time.sleep(30)")]
        self.write_plan()
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "hard cutoff"):
            run(self.path, self.output, poll_seconds=.01)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(load(self.output / "campaign.json")["status"], "failed")

    def test_fresh_freeze_runs_exactly_one_option_and_binds_dataset(self):
        panels = []
        for n in (20, 50, 100):
            data, target = self.root / f"data{n}", self.root / f"result{n}"
            data.write_text(str(n))
            panels.append({"n": n, "dataset": str(data), "steps": [self.step(f"fresh{n}", target)]})
        self.plan["fresh_options"] = panels
        self.write_plan()
        freeze = self.root / "freeze.json"
        freeze.write_text(json.dumps({"n": 50, "plan_sha256": digest(self.path), "dataset_sha256": digest(panels[1]["dataset"]),
                                     "answer_blind": True, "fresh_scoring_started": False,
                                     "basis": "measured_throughput_and_resources", "frozen_at": datetime.now(timezone.utc).isoformat()}))
        result = run(self.path, self.output, freeze_path=freeze, poll_seconds=.01)
        self.assertEqual(result["fresh_n"], 50)
        self.assertEqual(list(result["stages"]), ["fresh50"])
        self.assertFalse((self.root / "result20").exists())
        self.assertFalse((self.root / "result100").exists())
        (self.root / "data50").write_text("changed")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            choose_fresh(self.plan, freeze, digest(self.path))

    def test_no_freeze_means_no_fresh_calls(self):
        self.plan["evaluation_deadline"] = (datetime.now(timezone.utc) + timedelta(seconds=.1)).isoformat()
        self.plan["fresh_options"] = [{"n": 20, "dataset": "not-accessed", "steps": [self.step("fresh", self.root / "unused")]}]
        self.write_plan()
        result = run(self.path, self.output, poll_seconds=.01)
        self.assertEqual(result["status"], "stopped_before_new_stage")
        self.assertEqual(result["stages"], {})

    def test_sha_and_type_contracts_and_plan_tampering(self):
        marker = self.root / "marker"
        marker.write_text("ok")
        with self.assertRaisesRegex(ValueError, "Pinned"):
            condition({"path": str(marker), "sha256": "0" * 64})
        self.write_plan()
        run(self.path, self.output)
        self.plan["extra"] = True
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "plan changed"):
            run(self.path, self.output, resume=True)

    def test_frozen_source_tamper_is_rejected_even_after_completion(self):
        self.write_plan()
        run(self.path, self.output)
        (self.output / "source" / "pc_strands" / "campaign.py").write_text("# tampered")
        with self.assertRaisesRegex(ValueError, "Frozen source"):
            run(self.path, self.output, resume=True)

    def test_explicit_selected_order_is_immutable_on_resume(self):
        self.plan["steps"] = [self.step("a", self.root / "a"), self.step("b", self.root / "b")]
        self.write_plan()
        result = run(self.path, self.output, selected=["b", "a"], poll_seconds=.01)
        self.assertEqual(list(result["stages"]), ["b", "a"])
        with self.assertRaisesRegex(ValueError, "order changed"):
            run(self.path, self.output, resume=True, selected=["a", "b"])

    def test_status_save_retries_transient_windows_reader_lock(self):
        original, attempts = Path.replace, []
        def reader_lock(path, target):
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError("simulated Windows status reader")
            return original(path, target)
        target = self.root / "receipt.json"
        with patch.object(Path, "replace", reader_lock):
            save(target, {"status": "completed"})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(load(target), {"status": "completed"})
        self.assertFalse(target.with_suffix(".tmp").exists())

    def test_terminal_status_set_accepts_recorded_failures_but_not_interruption(self):
        marker = self.root / "manifest.json"
        spec = {"path": str(marker), "json_in": {"status": ["completed", "completed_with_errors"]},
                "json_equals": {"completed_questions": 600, "writer_unload_verified": True}}
        for status, expected in (("completed", True), ("completed_with_errors", True), ("interrupted", False)):
            marker.write_text(json.dumps({"status": status, "completed_questions": 600, "writer_unload_verified": True}))
            self.assertEqual(condition(spec), expected)


if __name__ == "__main__":
    unittest.main()
