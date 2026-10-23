"""Synthetic action-probe lifecycle checks; no native model or tool execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pc_strands.control import GpuLease, ResourcePressure
from scripts.probe_tau_actions import DEFAULT_CONFIG, load_cases, run_probe
from tests.test_strands_input_probe import SyntheticControl


class TauActionProbeTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(DEFAULT_CONFIG.read_text())
        upstream = root / "upstream"
        policy = upstream / "data/tau2/domains/retail/policy.md"
        tools = upstream / "src/tau2/domains/retail/tools.py"
        policy.parent.mkdir(parents=True)
        tools.parent.mkdir(parents=True)
        policy.write_text(config["policy"], encoding="utf-8")
        tools.write_text("class RetailTools:\n" + "".join(
            f"    @is_tool('synthetic')\n    def {tool['name']}(self): pass\n" for tool in config["tools"]), encoding="utf-8")
        config["policy_sha256"] = hashlib.sha256(policy.read_bytes()).hexdigest()
        config["tool_source_sha256"] = hashlib.sha256(tools.read_bytes()).hexdigest()
        checkpoint, base = root / "head", root / "base"
        checkpoint.mkdir()
        base.mkdir()
        (checkpoint / "lora").mkdir()
        for name in ("head.safetensors", "strands_decider_config.json", "lora/adapter_model.safetensors"):
            (checkpoint / name).write_bytes(b"synthetic-not-model-weights")
        config["released_head_sha256"] = hashlib.sha256((checkpoint / "head.safetensors").read_bytes()).hexdigest()
        target = root / "cases.json"
        target.write_text(json.dumps(config), encoding="utf-8")
        lock = root / "shared.lock"
        lock.write_bytes(b"0")
        return argparse.Namespace(config=target, upstream_root=upstream, checkpoint_dir=checkpoint,
            base_dir=base, output=root / "output", lease_path=lock, deadline=None), config

    def backend(self, args, calls, *, overflow=False):
        test = self
        class FakeBackend:
            max_length = 4096
            def __init__(self):
                self.assert_owned()
            def assert_owned(self):
                with test.assertRaises(ResourcePressure):
                    with GpuLease(args.lease_path, "conflicting-synthetic-test"):
                        pass
            def activate(self):
                calls.append("activate")
            def _encode(self, view, order):
                test.assertEqual(set(view), {"context_protocol", "policy", "messages", "available_tool_names", "tools", "proposal"})
                test.assertNotIn("evaluator_expected_allowed", json.dumps(view))
                proposed = {call["name"] for call in view["proposal"]["tool_calls"]}
                test.assertEqual({tool["name"] for tool in view["tools"]}, proposed)
                return view, "synthetic-prompt", None, None, 5000 if overflow else 3000
            def extract(self, view):
                self.assert_owned()
                self._encode(view, (0, 1))
                calls.append("extract")
                return SimpleNamespace(input_tokens=3000, encoding_seconds=.001)
            def score_features(self, feature, *, temperature):
                self.assert_owned()
                test.assertEqual(temperature, 1)
                calls.append("score")
                return SimpleNamespace(p_true=.6, logits=(0, .4), temperature=1, elapsed_seconds=.001)
            def metadata(self):
                return {"criterion": "tau-policy-action-v1", "temperature": 1}
            def offload(self):
                self.assert_owned()
                calls.append("offload")
                return .001
        return lambda checkpoint, base: FakeBackend()

    def test_native_route_uses_public_projection_and_raw_temperature_without_forced_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = self.fixture(Path(directory))
            calls = []
            with patch("scripts.probe_tau_actions.verify_upstream", return_value={"upstream_revision": config["upstream_revision"]}), \
                 patch("scripts.probe_tau_actions.ensure_port_closed"):
                result = run_probe(args, factory=self.backend(args, calls), control_class=SyntheticControl)
            self.assertEqual(result["physical_encoder_forwards"], 4)
            self.assertEqual(result["head_score_calls"], 4)
            self.assertEqual(result["benchmark_tasks_loaded"], 0)
            self.assertEqual(result["tools_executed"], 0)
            self.assertEqual([row["permitted_minus_prohibited_probability"] for row in result["pair_differences"]], [0, 0])
            self.assertEqual(calls[-1], "offload")
            self.assertTrue(result["gpu_lease_released"])
            self.assertTrue(result["input_files_unchanged"])

    def test_complete_context_overflow_stops_before_any_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = self.fixture(Path(directory))
            calls = []
            with patch("scripts.probe_tau_actions.verify_upstream", return_value={"upstream_revision": config["upstream_revision"]}), \
                 patch("scripts.probe_tau_actions.ensure_port_closed"):
                with self.assertRaisesRegex(ValueError, "does not fit"):
                    run_probe(args, factory=self.backend(args, calls, overflow=True), control_class=SyntheticControl)
            result = json.loads((args.output / "action-probe.json").read_text())
            self.assertEqual(result["physical_encoder_forwards"], 0)
            self.assertNotIn("extract", calls)
            self.assertTrue(result["owned_model_offload_completed"])

    def test_hidden_metadata_or_an_extra_pair_change_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = self.fixture(Path(directory))
            for mutate in (lambda value: value["cases"][0].update(task_id="forbidden"),
                           lambda value: value["cases"][1]["proposal"]["tool_calls"][0]["arguments"].update(reason="ordered by mistake")):
                changed = copy.deepcopy(config)
                mutate(changed)
                args.config.write_text(json.dumps(changed))
                with patch("scripts.probe_tau_actions.verify_upstream", return_value={"upstream_revision": config["upstream_revision"]}):
                    with self.assertRaises(ValueError):
                        load_cases(args.config, args.upstream_root)


if __name__ == "__main__":
    unittest.main()
