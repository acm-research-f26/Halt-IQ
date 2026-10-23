"""Adoption boundaries; no models, providers or Codex requests."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from haltiq_supervisor.plan import PACKAGES, inspect, prepare, source_files, validate, verify_frozen
from pc_strands.campaign import digest, load


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.plan = {"version": 1, "scope": "offline_supervision_fixture", "status": "ready",
            "source_root": str(self.repo), "python": sys.executable,
            "source_paths": ["haltiq_supervisor", "pc_strands"], "inputs": [], "readiness": {},
            "deadline": (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
            "hard_deadline": (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat(),
            "resources": {"ram_gib": 7, "vram_gib": 3, "startup_ram_gib": 8.5, "startup_vram_gib": 9,
                "stability_seconds": 30, "minimum_rest_seconds": 5, "cpu_threads": 4, "gpu_owners": 1,
                "duty_cycle": .5},
            "stages": [{"id": "first", "model_work": False, "gpu": False,
                "argv": [sys.executable, "-c", "pass"], "outputs": [{"path": "{run}/result.json"}]}]}
        self.file, self.output = self.root / "plan.json", self.root / "run"

    def write(self):
        self.file.write_text(json.dumps(self.plan), encoding="utf-8")

    def test_preparation_never_dispatches_and_copies_one_inspected_plan_snapshot(self):
        self.write()
        original = self.file.read_bytes()
        from haltiq_supervisor.plan import inspect as real_inspect
        def mutate(value):
            result = real_inspect(value)
            changed = dict(value, injected_after_validation=True)
            self.file.write_text(json.dumps(changed), encoding="utf-8")
            return result
        with patch("haltiq_supervisor.plan.inspect", side_effect=mutate), patch("subprocess.Popen") as spawn:
            state = prepare(self.file, self.output)
        self.assertEqual(state["status"], "prepared_not_started")
        self.assertEqual((self.output / "plan.json").read_bytes(), original)
        self.assertEqual(state["plan_sha256"], hashlib.sha256(original).hexdigest())
        spawn.assert_not_called()

    def test_config_directory_cannot_silently_omit_nonpython_settings(self):
        self.plan["source_paths"].append("configs")
        with self.assertRaisesRegex(ValueError, "configuration/data files explicitly"):
            source_files(self.plan)

    def test_resource_reserves_cannot_be_weakened(self):
        for key, value in (("ram_gib", 6), ("vram_gib", 2), ("startup_ram_gib", 8),
                           ("minimum_rest_seconds", 0), ("cpu_threads", 8), ("gpu_owners", 2),
                           ("ram_gib", float("nan")), ("vram_gib", float("inf"))):
            before = self.plan["resources"][key]
            self.plan["resources"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(self.plan)
            self.plan["resources"][key] = before

    def test_prepare_only_null_window_and_missing_workers_report_blockers(self):
        self.plan.update(status="prepared_blocked", deadline=None, hard_deadline=None)
        self.plan["stages"][0]["argv"] = None
        report = inspect(self.plan)
        self.assertEqual(report["status"], "blocked_no_launch")
        self.assertGreaterEqual(len(report["blockers"]), 3)
        self.write()
        with self.assertRaisesRegex(ValueError, "blocked"):
            prepare(self.file, self.output)
        self.assertFalse(self.output.exists())

    def test_readiness_receipt_is_hash_pinned_and_rechecked_after_adoption(self):
        proof = self.root / "proof.json"
        proof.write_text('{"ready": true}')
        self.plan["readiness"] = {"fixture": [{"path": str(proof), "sha256": digest(proof), "json_equals": {"ready": True}}]}
        self.write()
        state = prepare(self.file, self.output)
        proof.write_text('{"ready": false}')
        with self.assertRaisesRegex(ValueError, "Pinned"):
            verify_frozen(self.plan, state, self.output)

    def test_source_change_during_copy_is_rejected_before_launch(self):
        self.write()
        from haltiq_supervisor.plan import shutil
        real = shutil.copyfile
        def changed(source, target):
            real(source, target)
            if Path(target).name == "engine.py":
                Path(target).write_text("# unreviewed copy\n")
        with patch("haltiq_supervisor.plan.shutil.copyfile", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Frozen source"):
                prepare(self.file, self.output)

    def test_typed_receipt_cannot_substitute_integer_for_boolean(self):
        proof = self.root / "proof.json"
        proof.write_text('{"ready": 1}')
        self.plan["readiness"] = {"fixture": [{"path": str(proof), "sha256": digest(proof), "json_equals": {"ready": True}}]}
        self.assertEqual(inspect(self.plan)["status"], "blocked_no_launch")

    def test_custom_model_stages_require_all_packages_and_real_module_entrypoint(self):
        self.plan.update(scope="custom_tau2_hotpot", private_port=11435, gpu_lease=str(self.root / "gpu.lock"),
                         study_lease=str(self.root / "study.lock"))
        stage = self.plan["stages"][0]
        stage.update(model_work=True, planned_units=20, completion_receipt="{run}/result.json",
                     argv=[sys.executable, "-c", "pass", "{deadline}", "{control}", "--allow-model-calls"])
        with self.assertRaisesRegex(ValueError, "frozen package modules"):
            validate(self.plan)
        stage["argv"] = [sys.executable, "-B", "-m", "haltiq_hotpot.runner", "{deadline}", "{control}", "--allow-model-calls"]
        with self.assertRaisesRegex(ValueError, "Freeze every study package"):
            validate(self.plan)
        self.plan["source_paths"] = [*PACKAGES, "pyproject.toml"]
        report = inspect(self.plan)
        self.assertEqual(report["status"], "blocked_no_launch")
        self.assertTrue(any("combined_pilot" in s for s in report["blockers"]))
        self.assertTrue(any("manifest" in s for s in report["blockers"]))


if __name__ == "__main__":
    unittest.main()
