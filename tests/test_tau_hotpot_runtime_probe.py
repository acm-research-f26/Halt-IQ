"""Synthetic probe scope controls, without model imports or OS probes."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from haltiq_supervisor.plan import PACKAGES, inspect, validate
from haltiq_tau_hotpot.hobson import file_hash
from haltiq_tau_hotpot.runtime import decision_state
from haltiq_tau_hotpot.runtime_probe import synthetic_input, verify_dispatch, probe
from tests import test_supervisor_plan as plan_tests


class SyntheticInputTests(unittest.TestCase):
    def test_token_fitting_changes_only_invented_filler_and_preserves_explicit_criterion(self):
        requests = []
        def prepare(state, criterion):
            requests.append((state, criterion))
            n = 71 + state.count(" filler")
            if n > 4096: raise ValueError("strict window")
            return {"input_tokens": n}
        for criterion in ("answer", "evidence"):
            for target in (512, 2048, 4096):
                public, draft, n = synthetic_input(SimpleNamespace(prepare_state=prepare), target, criterion)
                self.assertEqual(n, target)
                self.assertEqual(public["type"], "synthetic_control")
                self.assertNotIn("evaluator", public)
                state = decision_state(public, draft, [0], criterion)
                self.assertEqual("Citations:" in state, criterion == "evidence")
                self.assertEqual(prepare(state, criterion)["input_tokens"], n)
        self.assertEqual({criterion for _, criterion in requests}, {"answer", "evidence"})

    def test_insufficient_context_coverage_fails_instead_of_shrinking_target(self):
        with self.assertRaisesRegex(ValueError, "declared token boundary"):
            synthetic_input(SimpleNamespace(prepare_state=lambda *a, **k: {"input_tokens": 10}), 4096, "answer")

    def test_probe_materializes_generation_before_its_single_native_placement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trained = root / "head.safetensors"
            trained.write_bytes(b"synthetic head, never loaded")
            (root / "supervisor.json").write_text(json.dumps({"source_sha256": {"synthetic.py": "a" * 64}}))
            events, observed = [], set()
            def generation(state, **kwargs):
                events.append("generation")
                return "retain" if kwargs.get("review") else {"answer": "control", "supporting_facts": []}
            def observe(public, draft, **kwargs):
                events.append("native_or_cpu")
                key = (public["id"], kwargs["criterion"])
                fresh = key not in observed
                observed.add(key)
                return {"physical_forwards": int(fresh), "head_scores": 2 if fresh else 1,
                    "input_tokens": 71 + public["context"][0][1][0].count(" filler"),
                    "raw_logits_false_true": [0., 1.]}
            runtime = Mock()
            runtime.__enter__ = Mock(return_value=runtime)
            runtime.__exit__ = Mock(return_value=False)
            runtime.backend.prepare_state.side_effect = lambda state, **_: {"input_tokens": 71 + state.count(" filler")}
            runtime._native_ready.side_effect = lambda: events.append("native_placement")
            runtime.generate.side_effect = generation
            runtime.observe.side_effect = observe
            runtime.cleanup = {"verified": True}
            runtime.timings = {}
            runtime.journal.verify.return_value = {"by_role": {"native_startup": 1, "native_answer": 3,
                "native_evidence": 3, "cached_head_answer": 6, "cached_head_evidence": 6,
                "writer_load": 1, "writer": 1, "reviewer": 1, "writer_unload": 1},
                "exact_response_reuses": 2, "pending_or_uncertain": False}
            proof = {"run_id": "synthetic", "model_binding": {"released_head_sha256": "a" * 64}}
            args = SimpleNamespace(output=root / "probe", previous=root)
            control = SimpleNamespace(active_seconds=0., idle_seconds=0.)
            with patch("haltiq_tau_hotpot.runtime_probe.pilot_binding", return_value=(proof, trained)), \
                    patch("haltiq_tau_hotpot.runtime_probe.LocalRuntime", return_value=runtime), \
                    patch("haltiq_tau_hotpot.runtime_probe.completion"), \
                    patch.dict("os.environ", {"HALTIQ_PLAN_SHA256": "a" * 64}):
                probe(args, {"combined_pilot_proof_sha256": "b" * 64}, control)
            self.assertEqual(events, ["generation"] * 4 + ["native_placement"] + ["native_or_cpu"] * 18)
            runtime._native_ready.assert_called_once()

    def test_import_has_no_model_runtime_or_tau_initialization(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import sys;import haltiq_tau_hotpot.runtime_probe;assert not any(x in sys.modules for x in ('torch','transformers','tau2'))"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ProbeScopeTests(unittest.TestCase):
    def setUp(self):
        self.base = plan_tests.PlanTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root, self.plan = self.base.root, self.base.plan
        config = {"purpose": "runtime_probe", "targets": [512, 2048, 4096], "criteria": ["answer", "evidence"],
            "native_feature_cache": True, "response_cache": True, "new_learning": False, "benchmark_questions": 0}
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        spec = {"path": str(self.config), "sha256": file_hash(self.config)}
        start = datetime.now(timezone.utc)
        self.plan.update(scope="custom_tau2_hotpot", purpose="runtime_probe", start_time=start.isoformat(),
            deadline=(start + timedelta(minutes=30)).isoformat(), hard_deadline=(start + timedelta(minutes=35)).isoformat(),
            source_paths=[*PACKAGES, "pyproject.toml"], runtime_probe=spec, inputs=[spec],
            private_port=11435, gpu_lease=str(self.root / "gpu.lock"), study_lease=str(self.root / "study.lock"))
        self.plan["stages"] = [{"id": "probe", "model_work": True, "gpu": True, "planned_units": 22,
            "argv": [sys.executable, "-B", "-m", "haltiq_tau_hotpot.runtime_probe", "probe", "--config", str(self.config),
                "--output", "{run}/probe", "--previous", "{run}", "--deadline", "{deadline}",
                "--control", "{control}", "--allow-model-calls"],
            "completion_receipt": "{run}/probe/completion.json", "outputs": [{"path": "{run}/probe/completion.json"}]}]

    def test_fixed_synthetic_scope_still_needs_all_five_readiness_gates(self):
        validate(self.plan)
        result = inspect(self.plan)
        self.assertEqual(result["status"], "blocked_no_launch")
        self.assertTrue(any("combined_pilot" in message for message in result["blockers"]))

    def test_extra_question_training_phase_or_clock_extension_cannot_expand_probe(self):
        original = deepcopy(self.plan)
        for kind in ("phase", "count", "clock", "config", "argv"):
            self.plan = deepcopy(original)
            if kind == "phase": self.plan["stages"].append(deepcopy(self.plan["stages"][0]))
            elif kind == "count": self.plan["stages"][0]["planned_units"] = 23
            elif kind == "clock": self.plan["start_time"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            elif kind == "argv": self.plan["stages"][0]["argv"][-2] = "wrong-control"
            else:
                value = json.loads(self.config.read_text()); value["benchmark_questions"] = 1
                self.config.write_text(json.dumps(value))
                self.plan["runtime_probe"]["sha256"] = file_hash(self.config)
                self.plan["inputs"] = [deepcopy(self.plan["runtime_probe"])]
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate(self.plan)

    def test_changed_actual_worker_arguments_or_owner_refuse_before_frozen_model_io(self):
        run = self.root / "run"
        args = SimpleNamespace(previous=run, config=self.config, output=run / "probe",
            control=run / "workers/probe", deadline=self.plan["deadline"])
        state = {"run_id": "fixture", "plan_sha256": "0" * 64, "stages": {"probe": {"status": "started"}}}
        env = {"HALTIQ_RUN_ID": "fixture", "HALTIQ_PLAN_SHA256": "0" * 64, "HALTIQ_PHASE": "probe"}
        with patch("haltiq_tau_hotpot.runtime_probe.__file__", str(run / "source/haltiq_tau_hotpot/runtime_probe.py")), patch.dict("os.environ", env), patch("haltiq_supervisor.plan.verify_frozen") as verified:
            verify_dispatch(args, self.plan, state)
            verified.assert_called_once()
            args.output = run / "different"
            with self.assertRaisesRegex(ValueError, "frozen invocation"): verify_dispatch(args, self.plan, state)


if __name__ == "__main__":
    unittest.main()
