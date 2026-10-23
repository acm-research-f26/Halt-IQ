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
from haltiq_tau_hotpot.generation import MAIN_PROFILE, generation_payload
from haltiq_tau_hotpot.journal import CallJournal
from haltiq_tau_hotpot.runtime import LocalRuntime, decision_state
from haltiq_tau_hotpot.runtime_probe import generation_check, synthetic_input, verify_dispatch, probe, pilot_binding
from tests import test_supervisor_plan as plan_tests


class SyntheticInputTests(unittest.TestCase):
    def technical_generation(self, root, *, bad_main=False):
        config = {"purpose": "runtime_probe", "generation_profile": deepcopy(MAIN_PROFILE),
                  "writer_digest": "synthetic-not-a-model", "response_cache": True}
        control = SimpleNamespace(before_work=lambda **_: None, rest_after_block=lambda _: None)
        runtime = LocalRuntime(config, root, control)
        runtime._writer_ready = Mock()
        def raw(content, reason="stop"):
            return {"done": True, "done_reason": reason, "prompt_eval_count": 20, "eval_count": 10,
                    "message": {"content": content}}
        runtime._http = Mock(side_effect=[raw(json.dumps({"answer": "control", "supporting_facts": []})),
            raw("Incomplete old review.", "length"), raw("word " * 121 if bad_main else "Retain."),
            raw("Revise the cited entity."), raw("Incomplete old review.", "length")])
        return runtime

    def test_ab_keeps_known_legacy_failures_and_reuses_only_predeclared_main_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.technical_generation(root)
            answer, rows = generation_check(runtime, root)
            self.assertEqual([row["status"] for row in rows], ["known_terminal_failure", "completed", "completed", "known_terminal_failure"])
            self.assertEqual(len(answer), 1)
            self.assertEqual(runtime._http.call_count, 5)
            self.assertEqual(runtime.journal.verify()["by_role"], {"writer": 1, "reviewer": 4})
            self.assertEqual(runtime.journal.summary()["exact_response_reuses"], 2)
            reused = [e for e in runtime.journal.records if e["event"] == "exact_response_reuse" and e["role"] == "reviewer"]
            self.assertEqual(reused[0]["id"], rows[1]["physical_call_id"])

    def test_failed_main_word_limit_preserves_cell_then_stops_without_resampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.technical_generation(root, bad_main=True)
            with self.assertRaisesRegex(ValueError, "Predeclared main reviewer"):
                generation_check(runtime, root)
            self.assertEqual(runtime._http.call_count, 3)
            row = json.loads((root / "reviewer-ab/1.json").read_text())
            self.assertEqual(row["metadata"]["words"], 121)
            self.assertEqual(row["status"], "known_terminal_failure")
            self.assertFalse(runtime.journal.verify()["pending_or_uncertain"])
            self.assertFalse((root / "reviewer-ab/2.json").exists())

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
            journal, responses = CallJournal(root / "probe/physical"), {}
            config = {"purpose": "runtime_probe", "generation_profile": deepcopy(MAIN_PROFILE), "combined_pilot_proof_sha256": "b" * 64}
            def generation(state, **kwargs):
                events.append("generation")
                payload = generation_payload(config, state, **kwargs)
                key = json.dumps(payload, sort_keys=True)
                value = "Retain." if kwargs.get("review") else {"answer": "control", "supporting_facts": []}
                if key in responses:
                    journal.reuse(responses[key], payload)
                else:
                    journal.call("reviewer" if kwargs.get("review") else "writer", payload,
                        lambda: {"done": True, "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 10,
                            "message": {"content": value if isinstance(value, str) else json.dumps(value)}})
                    responses[key] = deepcopy(journal.records[-1])
                return value
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
            runtime.config = config
            runtime.journal.records = journal.records
            runtime.journal.directory = journal.directory
            runtime.journal.verify.return_value = {"by_role": {"native_startup": 1, "native_answer": 3,
                "native_evidence": 3, "cached_head_answer": 6, "cached_head_evidence": 6,
                "writer_load": 1, "writer": 1, "reviewer": 4, "writer_unload": 1},
                "exact_response_reuses": 2, "pending_or_uncertain": False}
            proof = {"run_id": "synthetic", "model_binding": {"released_head_sha256": "a" * 64}}
            args = SimpleNamespace(output=root / "probe", previous=root)
            control = SimpleNamespace(active_seconds=0., idle_seconds=0.)
            with patch("haltiq_tau_hotpot.runtime_probe.pilot_binding", return_value=(proof, trained)), \
                    patch("haltiq_tau_hotpot.runtime_probe.LocalRuntime", return_value=runtime), \
                    patch("haltiq_tau_hotpot.runtime_probe.completion"), \
                    patch.dict("os.environ", {"HALTIQ_PLAN_SHA256": "a" * 64}):
                probe(args, config, control)
            self.assertEqual(events, ["generation"] * 7 + ["native_placement"] + ["native_or_cpu"] * 18)
            runtime._native_ready.assert_called_once()

    def test_recovered_pilot_uses_verified_two_owner_binding_and_original_trained_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proof = {"schema": "haltiq-tau-hotpot-qualified-pilot-v2", "run": str(root / "comparison-only-owner")}
            path = root / "qualified.json"
            path.write_text(json.dumps(proof))
            original_head = root / "original-preparation-owner/train/learned/answer-head.safetensors"
            config = {"combined_pilot_proof": str(path), "combined_pilot_proof_sha256": file_hash(path)}
            with patch("haltiq_tau_hotpot.recovery.verify_qualified_pilot", return_value=(proof, original_head, {})) as verify, \
                    patch("haltiq_tau_hotpot.qualification.terminal_owner", side_effect=AssertionError("V1 owner route is forbidden")):
                actual, head = pilot_binding(config)
            verify.assert_called_once_with(proof, config=config)
            self.assertEqual(actual, proof)
            self.assertEqual(head, original_head)
            self.assertNotIn("comparison-only-owner", str(head))

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
            "native_feature_cache": True, "response_cache": True, "new_learning": False, "benchmark_questions": 0,
            "generation_profile": deepcopy(MAIN_PROFILE)}
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        spec = {"path": str(self.config), "sha256": file_hash(self.config)}
        start = datetime.now(timezone.utc)
        self.plan.update(scope="custom_tau2_hotpot", purpose="runtime_probe", start_time=start.isoformat(),
            deadline=(start + timedelta(minutes=30)).isoformat(), hard_deadline=(start + timedelta(minutes=35)).isoformat(),
            source_paths=[*PACKAGES, "pyproject.toml"], runtime_probe=spec, inputs=[spec],
            private_port=11435, gpu_lease=str(self.root / "gpu.lock"), study_lease=str(self.root / "study.lock"))
        self.plan["stages"] = [{"id": "probe", "model_work": True, "gpu": True, "planned_units": 25,
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
            elif kind == "count": self.plan["stages"][0]["planned_units"] = 26
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
