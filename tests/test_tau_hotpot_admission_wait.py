"""Synthetic scheduling/ownership controls; no models, callbacks or OS leases."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pc_strands.control import ResourcePressure, StudyStopped
from haltiq_tau_hotpot.admission_wait import verify_wait_history, wait_limit
from haltiq_tau_hotpot.hobson import file_hash
from haltiq_tau_hotpot.main_question import admission


class AdmissionWaitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.elapsed, self.idle, self.active, self.holds, self.entries = 0., 0., 0., False, 0
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.config = {"gpu_lease": str(self.root / "synthetic.lock"), "question_admission_wait_seconds": 120}
        self.control = self
        self.directory = self.root
        self.deadline = self.start + timedelta(hours=1)
        self.idle_seconds, self.active_seconds = 0., 0.
        self.samples, self.sample_calls, self.sleep_hook, self.blocked_lease_attempts = [11.1], 0, None, 0
        self.pressure_observed = False
        self.output = self.root / "admission.json"

    def _check_stop(self):
        if (self.root / "stop.request").exists() or self.now() >= self.deadline:
            raise StudyStopped("Synthetic stop or deadline")

    def now(self):
        return self.start + timedelta(seconds=self.elapsed)

    def sleep(self, seconds):
        if self.pressure_observed:
            self.assertFalse(self.holds, "Pressure wait must release admission ownership")
            self.pressure_observed = False
        self.elapsed += seconds
        self.idle_seconds += seconds
        if self.sleep_hook:
            self.sleep_hook()

    def sampler(self):
        self.assertTrue(self.holds)
        value = self.samples[min(self.sample_calls, len(self.samples) - 1)]
        self.sample_calls += 1
        self.pressure_observed = value is None or value < 11
        return {"available_ram_bytes": None if value is None else value * 1024**3,
                "gpu_free_mib": 9.1 * 1024, "gpu_temperature_c": 40}

    def lease(self, *args):
        owner = self
        class Lease:
            def __enter__(self):
                if owner.blocked_lease_attempts:
                    owner.blocked_lease_attempts -= 1
                    raise ResourcePressure("Synthetic foreign GPU owner")
                owner.entries += 1
                owner.assertFalse(owner.holds)
                owner.holds = True
            def __exit__(self, *args):
                owner.assertTrue(owner.holds)
                owner.holds = False
        return Lease()

    def run_admission(self, port=None):
        with patch("haltiq_tau_hotpot.main_question.port_open", side_effect=port or (lambda _: False)), \
                patch("subprocess.Popen") as process:
            try:
                result = admission(self.config, self, self.output, sampler=self.sampler, sleep=self.sleep,
                    clock=self.now, lease_factory=self.lease, monotonic=lambda: self.elapsed)
            finally:
                process.assert_not_called()
                self.assertFalse(self.holds)
                self.assertFalse(list(self.root.rglob("calls.jsonl")))
        verify_wait_history(self.config, result, self.root)
        return result

    def test_low_ram_recovers_to_four_fresh_samples_and_counts_wait_once(self):
        self.samples = [10.1, 11.1, 11.1, 11.1, 11.1]
        result = self.run_admission()
        self.assertEqual(result["status"], "PASS_ADMISSION")
        self.assertEqual([s["at"] for s in result["samples"]], [(self.start + timedelta(seconds=n)).isoformat() for n in (10,20,30,40)])
        self.assertEqual(result["wait"]["idle_seconds"], self.idle_seconds)
        self.assertEqual(self.idle_seconds, 40)
        self.assertEqual(self.active_seconds, 0)
        self.assertEqual(result["model_calls"], 0)
        self.assertTrue(result["gpu_lease_released"])
        self.assertTrue(result["private_port_closed"])

    def test_bad_fourth_sample_discards_three_good_samples(self):
        self.samples = [11.1, 11.1, 11.1, 10.1, 11.1, 11.1, 11.1, 11.1]
        result = self.run_admission()
        self.assertEqual(self.sample_calls, 8)
        self.assertEqual(result["samples"][0]["at"], (self.start + timedelta(seconds=40)).isoformat())
        self.assertEqual(result["wait"]["observations"], 8)

    def test_exact30second_optin_finishes_zero_overhead_window_inclusively(self):
        self.config["question_admission_wait_seconds"] = 30
        result = self.run_admission()
        self.assertEqual(result["status"], "PASS_ADMISSION")
        self.assertEqual(self.elapsed, 30)

    def test30second_budget_with_sampling_overhead_conservatively_expires(self):
        self.config["question_admission_wait_seconds"] = 30
        original = self.sampler
        def sample():
            self.elapsed += .25
            return original()
        self.sampler = sample
        with self.assertRaisesRegex(ResourcePressure, "expired"):
            self.run_admission()
        result = json.loads(self.output.read_text())
        self.assertEqual(result["status"], "BLOCKED_ADMISSION")
        self.assertEqual(result["model_calls"], 0)

    def test_permanent_pressure_expires_without_models_or_claimed_cleanup(self):
        self.samples, self.config["question_admission_wait_seconds"] = [10.1], 40
        with self.assertRaisesRegex(ResourcePressure, "expired"):
            self.run_admission()
        result = json.loads(self.output.read_text())
        self.assertEqual(self.elapsed, 40)
        self.assertIsNone(result["gpu_lease_released"])
        self.assertIsNone(result["private_port_closed"])
        self.assertEqual(result["model_calls"], 0)

    def test_unknown_telemetry_waits_but_never_admits_it(self):
        self.samples = [None, 11.1, 11.1, 11.1, 11.1]
        result = self.run_admission()
        history = [json.loads(line) for line in (self.root / result["wait"]["history_file"]).read_text().splitlines()]
        self.assertIsNone(history[0]["sample"]["available_ram_gib"])
        self.assertEqual(history[0]["kind"], "resource_pressure")

    def test_model_deadline_clips_wait_before_sampling_again(self):
        self.samples = [10.1]
        self.deadline = self.start + timedelta(seconds=15)
        with self.assertRaises(StudyStopped):
            self.run_admission()
        self.assertEqual(self.elapsed, 15)
        self.assertEqual(self.sample_calls, 2)

    def test_explicit_stop_after_pressure_aborts_without_dispatch(self):
        self.samples = [10.1]
        self.sleep_hook = lambda: (self.root / "stop.request").touch()
        with self.assertRaises(StudyStopped):
            self.run_admission()
        self.assertEqual(self.sample_calls, 1)

    def test_pause_before_sampling_releases_no_model_and_resets_window(self):
        (self.root / "pause.request").touch()
        self.sleep_hook = lambda: (self.root / "pause.request").unlink(missing_ok=True)
        result = self.run_admission()
        self.assertEqual(self.sample_calls, 4)
        self.assertEqual(self.idle_seconds, 31)
        self.assertEqual(result["wait"]["observations"], 5)

    def test_pause_during_good_window_discards_the_partial_window(self):
        def pause_then_release():
            path = self.root / "pause.request"
            if self.elapsed == 10:
                path.touch()
            elif self.elapsed >= 11:
                path.unlink(missing_ok=True)
        self.sleep_hook = pause_then_release
        result = self.run_admission()
        self.assertEqual(result["samples"][0]["at"], (self.start + timedelta(seconds=11)).isoformat())
        self.assertEqual(self.sample_calls, 5)

    def test_temperature80_never_enters_the_final_good_window(self):
        original = self.sampler
        def sample():
            value = original()
            if self.sample_calls == 1:
                value["gpu_temperature_c"] = 80
                self.pressure_observed = True
            return value
        self.sampler = sample
        result = self.run_admission()
        self.assertTrue(all(s["gpu_temperature_c"] < 80 for s in result["samples"]))
        self.assertEqual(self.idle_seconds, 40)

    def test_low_free_vram_never_enters_the_final_good_window(self):
        original = self.sampler
        def sample():
            value = original()
            if self.sample_calls == 1:
                value["gpu_free_mib"] = 8.9 * 1024
                self.pressure_observed = True
            return value
        self.sampler = sample
        result = self.run_admission()
        self.assertTrue(all(s["free_vram_gib"] >= 9 for s in result["samples"]))
        self.assertEqual(self.idle_seconds, 40)

    def test_foreign_gpu_lease_waits_before_any_sample(self):
        self.blocked_lease_attempts = 1
        result = self.run_admission()
        self.assertEqual(self.sample_calls, 4)
        self.assertEqual(self.idle_seconds, 40)
        self.assertEqual(result["wait"]["observations"], 5)

    def test_busy_port_waits_without_closing_or_taking_gpu_ownership(self):
        result = self.run_admission(port=lambda _: self.elapsed < 10)
        self.assertEqual(self.sample_calls, 4)
        self.assertEqual(self.idle_seconds, 40)
        self.assertEqual(result["wait"]["observations"], 5)

    def test_history_tamper_and_undeclared_wait_cannot_pass_audit(self):
        result = self.run_admission()
        with self.assertRaisesRegex(ValueError, "not declared"):
            verify_wait_history({}, result, self.root)
        history = self.root / result["wait"]["history_file"]
        rows = [json.loads(line) for line in history.read_text().splitlines()]
        rows[-1]["sample"]["available_ram_gib"] = 10.9
        history.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        result["wait"]["history_sha256"] = file_hash(history)
        with self.assertRaisesRegex(ValueError, "startup reserves"):
            verify_wait_history(self.config, result, self.root)

    def test_rebound_clock_or_elapsed_cannot_fabricate_a_wait_budget(self):
        result = self.run_admission()
        result["wait"]["elapsed_seconds"] += 5
        with self.assertRaisesRegex(ValueError, "timing"):
            verify_wait_history(self.config, result, self.root)
        result["wait"]["elapsed_seconds"] -= 5
        history = self.root / result["wait"]["history_file"]
        rows = [json.loads(line) for line in history.read_text().splitlines()]
        rows[1]["elapsed_seconds"] += 3
        history.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        result["wait"]["history_sha256"] = file_hash(history)
        with self.assertRaisesRegex(ValueError, "clock"):
            verify_wait_history(self.config, result, self.root)

    def test_invalid_optin_rejected_at_worker_and_main_protocol_boundary(self):
        from haltiq_tau_hotpot.protocol import load_main
        for value in (True, -1, 1, 1201, float("nan"), "120"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                wait_limit({"question_admission_wait_seconds": value})
            with self.subTest(protocol=value), self.assertRaisesRegex(ValueError, "wait must"):
                load_main({"purpose": "main_study", "question_admission_wait_seconds": value})
        self.assertEqual(wait_limit({}), 0)

    def test_actual_source_export_closure_includes_wait_helper(self):
        from haltiq_supervisor.plan import source_files
        root = Path(__file__).resolve().parents[1]
        files = source_files({"source_root": str(root), "source_paths": ["haltiq_tau_hotpot", "haltiq_supervisor", "pc_strands"]})
        self.assertIn("haltiq_tau_hotpot/admission_wait.py", files)

    def test_existing_history_is_never_reused_or_overwritten(self):
        self.run_admission()
        before = {p.name: p.read_bytes() for p in (self.output, self.root / "admission.observations.jsonl")}
        with self.assertRaises(FileExistsError):
            self.run_admission()
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.output, self.root / "admission.observations.jsonl")})


if __name__ == "__main__":
    unittest.main()
