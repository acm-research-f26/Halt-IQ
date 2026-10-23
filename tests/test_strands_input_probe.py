"""Synthetic lifecycle controls; these tests execute no native model."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pc_strands.control import GpuLease, ResourcePressure
from scripts.probe_strands_inputs import DEFAULT_CONFIG, load_config, run_probe


class SyntheticControl:
    def __init__(self, directory, deadline, **limits):
        self.limits = limits
        self.idle_seconds = 0
        self.last_resources = {"available_ram_bytes": 10 * 1024**3, "gpu_free_mib": 12 * 1024}
    def before_work(self, require_gpu=False):
        pass
    def _sleep(self, seconds):
        self.idle_seconds += seconds
    def rest_after_block(self, seconds):
        self.idle_seconds += max(5, seconds)


@dataclass
class SyntheticScore:
    p_true: float = .6
    logits: tuple = (0, .4)
    temperature: float = 1
    elapsed_seconds: float = .001


class InputProbeTests(unittest.TestCase):
    def arguments(self, root, *, comparison=True):
        for name in ("released", "base", "comparison"):
            folder = root / name
            folder.mkdir()
            if name != "base":
                (folder / "lora").mkdir()
                for file in ("head.safetensors", "strands_decider_config.json", "lora/adapter_model.safetensors"):
                    (folder / file).write_bytes(b"synthetic-test-fixture-not-model-weights")
        (root / "shared.lock").write_bytes(b"0")
        return argparse.Namespace(config=DEFAULT_CONFIG, checkpoint_dir=root / "released", base_dir=root / "base",
            comparison_head=root / "comparison" if comparison else None, output=root / "output",
            lease_path=root / "shared.lock", deadline=None)

    def factory(self, args, calls, *, fail_at=None):
        test = self
        def assert_owned():
            with test.assertRaises(ResourcePressure):
                with GpuLease(args.lease_path, "conflicting-unit-control"):
                    pass
        class SyntheticBackend:
            def __init__(self):
                assert_owned()
                calls.append("constructed")
                self.states = []
            def activate(self):
                assert_owned()
                calls.append("activated")
            def make_head(self, path):
                assert_owned()
                return object()
            def metadata(self):
                return {"scope": "synthetic-unit-control"}
            def extract(self, state):
                assert_owned()
                test.assertEqual(set(state), {"question", "evidence", "draft"})
                self.states.append(state)
                calls.append("extract")
                if len(self.states) == fail_at:
                    raise RuntimeError("synthetic uncertain native failure")
                return SimpleNamespace(input_tokens=10, encoding_seconds=.001)
            def score_features(self, feature, head=None):
                assert_owned()
                calls.append("score")
                return SyntheticScore()
            def offload(self):
                assert_owned()
                calls.append("offloaded")
                return .001
        return lambda checkpoint, base: SyntheticBackend()

    def test_single_encoding_serves_both_heads_and_cleanup_keeps_os_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            calls = []
            test = self
            class PlacementObservedControl(SyntheticControl):
                def before_work(self, require_gpu=False):
                    if self.limits["minimum_ram_gib"] == 7:
                        test.assertIn("activated", calls)
            with patch("scripts.probe_strands_inputs.ensure_port_closed"):
                result = run_probe(args, backend_factory=self.factory(args, calls), control_class=PlacementObservedControl)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(calls.count("extract"), 4)
            self.assertEqual(calls.count("score"), 8)
            self.assertEqual(calls[-1], "offloaded")
            self.assertTrue(result["gpu_lease_released"])
            self.assertTrue(result["input_files_unchanged"])
            self.assertEqual(len(result["startup_admission"]["samples"]), 4)
            self.assertEqual(result["pair_differences"][0]["correct_minus_contradictory_probability"],
                             {"released": 0, "comparison": 0})

    def test_failure_is_preserved_without_a_retry_and_completed_work_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            calls = []
            with patch("scripts.probe_strands_inputs.ensure_port_closed"):
                with self.assertRaisesRegex(RuntimeError, "uncertain"):
                    run_probe(args, backend_factory=self.factory(args, calls, fail_at=2), control_class=SyntheticControl)
            result = json.loads((args.output / "probe.json").read_text())
            self.assertEqual(result["physical_encoder_forwards"], 1)
            self.assertEqual(calls.count("extract"), 2)
            self.assertTrue(result["native_forward_pending_or_uncertain"])
            self.assertEqual(result["records"][0]["status"], "completed")
            self.assertTrue(result["owned_model_offload_completed"])
            self.assertTrue(result["gpu_lease_released"])
            with self.assertRaises(FileExistsError):
                run_probe(args, backend_factory=self.factory(args, calls), control_class=SyntheticControl)

    def test_failed_startup_admission_never_constructs_backend(self):
        class Denied(SyntheticControl):
            def before_work(self, require_gpu=False):
                raise ResourcePressure("synthetic RAM pressure")
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            calls = []
            with patch("scripts.probe_strands_inputs.ensure_port_closed"):
                with self.assertRaises(ResourcePressure):
                    run_probe(args, backend_factory=self.factory(args, calls), control_class=Denied)
            result = json.loads((args.output / "probe.json").read_text())
            self.assertEqual(calls, [])
            self.assertFalse(result["model_work_started"])
            self.assertTrue(result["gpu_lease_released"])

    def test_extended_deadline_and_lower_resource_reserves_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            args.deadline = (datetime.now(timezone.utc) + timedelta(minutes=11)).isoformat()
            with self.assertRaisesRegex(ValueError, "ten minutes"):
                run_probe(args, control_class=SyntheticControl)
            config = json.loads(DEFAULT_CONFIG.read_text())
            config["runtime_limits"]["minimum_ram_gib"] = 1
            path = Path(directory) / "changed-profile.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "must not change"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
