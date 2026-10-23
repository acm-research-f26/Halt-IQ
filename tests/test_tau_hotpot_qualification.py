"""Synthetic/offline audit controls; no benchmark model or readiness is emitted."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from haltiq_tau_hotpot import qualification as q
from haltiq_tau_hotpot.journal import CallJournal


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def tensors(path, specs):
    header, raw = {}, bytearray()
    for name, (shape, values) in specs.items():
        start = len(raw)
        raw.extend(struct.pack("<" + "f" * len(values), *values))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(raw)]}
    encoded = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)


class QualificationBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def owner(self):
        source = self.root / "source"
        source.mkdir()
        (source / "synthetic.py").write_text("# fixture, no executable model source\n")
        python = self.root / "fixture-python.exe"
        python.write_bytes(b"SYNTHETIC-NEVER-EXECUTED")
        plan = {"purpose": "qualification_pilot", "scope": "custom_tau2_hotpot", "python": str(python),
                "stages": [{"id": name, "outputs": [{"path": "{run}/" + name + "/receipt.json"}]} for name in q.PHASES],
                "reviewed_source_sha256": {"synthetic.py": q.sha(source / "synthetic.py")}}
        dump(self.root / "plan.json", plan)
        state = {"status": "completed", "plan_sha256": q.sha(self.root / "plan.json"),
            "source_sha256": plan["reviewed_source_sha256"], "python_sha256": q.sha(python), "stages": {}}
        for name in q.PHASES:
            path = self.root / name / "receipt.json"
            dump(path, {"synthetic": name})
            state["stages"][name] = {"status": "completed", "exit_code": 0, "owned_tree_clear": True,
                "os_handoff_verified": True, "output_sha256": {str(path): q.sha(path)}}
        dump(self.root / "supervisor.json", state)
        return plan, state

    def test_terminal_owner_rejects_running_before_reading_missing_outcomes(self):
        _, state = self.owner()
        state["status"] = "running"
        dump(self.root / "supervisor.json", state)
        with self.assertRaisesRegex(q.QualificationError, "genuinely terminal"):
            q.terminal_owner(self.root, state["plan_sha256"])
        self.assertFalse((self.root / "collect/episodes.json").exists())

    def test_owner_accepts_bound_fixture_then_rejects_output_source_and_cleanup_mutations(self):
        _, state = self.owner()
        expected = state["plan_sha256"]
        q.terminal_owner(self.root, expected)
        for mutation in ("output", "source", "cleanup", "interpreter", "hash"):
            with self.subTest(mutation=mutation):
                if mutation == "output":
                    p = self.root / "collect/receipt.json"
                    old = p.read_bytes(); p.write_bytes(b"changed")
                elif mutation == "source":
                    p = self.root / "source/synthetic.py"
                    old = p.read_bytes(); p.write_bytes(b"changed")
                elif mutation == "interpreter":
                    p = self.root / "fixture-python.exe"
                    old = p.read_bytes(); p.write_bytes(b"changed")
                else:
                    p = self.root / "supervisor.json"
                    old = p.read_bytes()
                    altered = deepcopy(state)
                    if mutation == "cleanup": altered["stages"]["collect"]["owned_tree_clear"] = False
                    else: altered["plan_sha256"] = "0" * 64
                    dump(p, altered)
                try:
                    with self.assertRaises(q.QualificationError):
                        q.terminal_owner(self.root, expected)
                finally:
                    p.write_bytes(old)

    def test_qualify_writes_blocked_without_readiness_and_never_retries_existing_receipt(self):
        output = self.root / "private-review.json"
        with patch.object(q, "audit_run", side_effect=q.QualificationError("fixture incomplete")) as audit:
            result = q.qualify(self.root, output)
            self.assertEqual(result["status"], "BLOCKED_NO_QUALIFICATION")
            self.assertEqual(result["ready"], {})
            self.assertFalse(result["qualified"])
            self.assertEqual(result["schema"], "haltiq-tau-hotpot-qualified-pilot-v1")
            self.assertEqual(result["new_model_forwards"], 0)
            with self.assertRaises(q.QualificationError):
                q.qualify(self.root, output)
            self.assertEqual(audit.call_count, 1)
        self.assertNotIn("combined_pilot", output.read_text())

    def test_qualification_cannot_write_into_frozen_source(self):
        with self.assertRaises(q.QualificationError), patch.object(q, "audit_run") as audit:
            q.qualify(self.root, self.root / "source/review.json")
        audit.assert_not_called()

    def test_physical_journal_reconciles_real_fsync_fixture_and_rejects_changed_response(self):
        directory = self.root / "physical"
        journal = CallJournal(directory)
        journal.call("synthetic_control", {"not_model": True}, lambda: {"value": 1})
        result = q.journal(directory)
        self.assertEqual(result["by_role"], {"synthetic_control": 1})
        path = directory / journal.records[-1]["response"]
        path.write_text("{}")
        with self.assertRaisesRegex(q.QualificationError, "response changed"):
            q.journal(directory)

    def test_duplicate_completed_pending_and_uncertain_physical_events_fail(self):
        for kind in ("duplicate", "pending", "uncertain"):
            directory = self.root / kind
            journal = CallJournal(directory)
            journal.call("synthetic", {}, lambda: {})
            event = journal.records[-1] if kind == "duplicate" else {**journal.records[0], "id": "new", "event": kind}
            with journal.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            with self.subTest(kind=kind), self.assertRaises(q.QualificationError):
                q.journal(directory)

    def test_finite_safetensors_inspection_is_actual_bytes_not_file_size_alone(self):
        path = self.root / "synthetic.safetensors"
        tensors(path, {"a": ([2], [1., 2.])})
        blocks, hashed = q.safe_tensors(path, {"a": [2]})
        self.assertEqual(list(blocks["a"][2]), [1., 2.])
        self.assertEqual(len(hashed), 64)
        with self.assertRaises(q.QualificationError): q.safe_tensors(path, {"a": [3]})
        tensors(path, {"a": ([2], [1., float("nan")])})
        with self.assertRaisesRegex(q.QualificationError, "Nonfinite"):
            q.safe_tensors(path, {"a": [2]})
        tensors(path, {"a": ([2], [1., 2.])})
        with path.open("ab") as stream: stream.write(b"UNBOUND")
        with self.assertRaisesRegex(q.QualificationError, "trailing"):
            q.safe_tensors(path, {"a": [2]})
        self.assertEqual(sum(__import__('math').prod(s) for s in q.HEAD_SHAPES.values()), 1053184)

    def test_probability_uses_raw_logits_temperature_rounding_and_physical_counts(self):
        result = {"raw_logits_false_true": [0., 0.], "temperature": 1., "p_true_unrounded": .5,
            "p_true": .5, "physical_forwards": 0, "head_scores": 1}
        q.probability(result, native=False)
        for update in ({"p_true": .9}, {"p_true_unrounded": .9}, {"temperature": 0},
                       {"physical_forwards": 1}, {"raw_logits_false_true": [float('nan'), 0]}):
            with self.subTest(update=update), self.assertRaises(q.QualificationError):
                q.probability({**result, **update}, native=False)

    def test_frozen_helpers_use_isolated_source_without_bytecode_and_restore_imports_on_failure(self):
        import haltiq_tau_hotpot.worker as original
        modules = ("haltiq_supervisor.plan", "haltiq_supervisor.engine", "pc_strands.control",
            "haltiq_tau_hotpot.worker", "haltiq_tau_hotpot.agent", "haltiq_tau_hotpot.domain",
            "haltiq_tau_hotpot.comparison", "haltiq_tau_hotpot.learning", "haltiq_tau_hotpot.hobson")
        for name in modules:
            path = self.root / "source" / Path(*name.split('.')).with_suffix('.py')
            path.parent.mkdir(parents=True, exist_ok=True)
            (path.parent / '__init__.py').write_text('')
            path.write_text('marker = "synthetic-frozen-fixture"\n')
        before_path, before_bytecode = list(sys.path), sys.dont_write_bytecode
        with self.assertRaisesRegex(RuntimeError, "intentional test failure"):
            with q.frozen_helpers(self.root) as helpers:
                self.assertTrue(sys.dont_write_bytecode)
                self.assertEqual(helpers['haltiq_tau_hotpot.worker'].marker, 'synthetic-frozen-fixture')
                self.assertIsNot(helpers['haltiq_tau_hotpot.worker'], original)
                raise RuntimeError('intentional test failure')
        self.assertIs(sys.modules['haltiq_tau_hotpot.worker'], original)
        self.assertEqual(sys.path, before_path)
        self.assertEqual(sys.dont_write_bytecode, before_bytecode)
        self.assertFalse(list((self.root / 'source').rglob('__pycache__')))


class LearningArtifactTests(unittest.TestCase):
    """Inspect actual F32 artifact bytes, using no Torch or numerical model."""
    def setUp(self):
        from haltiq_tau_hotpot import learning
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        checkpoint, base = self.root / "checkpoint", self.root / "base"
        checkpoint.mkdir(); base.mkdir()
        (checkpoint / "lora").mkdir()
        (checkpoint / "lora/adapter_model.safetensors").write_bytes(b"SYNTHETIC-NOT-A-REAL-ADAPTER")
        (base / "model.safetensors-00001-of-00001.safetensors").write_bytes(b"SYNTHETIC-NOT-A-REAL-TORSO")
        head = {name: (shape, [0.] * __import__('math').prod(shape)) for name, shape in q.HEAD_SHAPES.items()}
        tensors(checkpoint / "head.safetensors", head)
        (self.root / "train/learned").mkdir(parents=True)
        head["q.bias"][1][0] = .001
        tensors(self.root / "train/learned/answer-head.safetensors", head)
        feature_path = self.root / "features/synthetic.safetensors"
        feature_path.parent.mkdir()
        tensors(feature_path, {"decide": ([2048], [0.] * 2048), "options": ([2, 2048], [0.] * 4096)})
        frozen = {"checkpoint/lora/adapter_model.safetensors": q.sha(checkpoint / "lora/adapter_model.safetensors"),
                  "base/model.safetensors-00001-of-00001.safetensors": q.sha(base / "model.safetensors-00001-of-00001.safetensors")}
        frozen.update({"upstream/" + name: str(i) * 64 for i, name in enumerate(("modeling.py", "infer.py", "prompting.py", "schema.py"), 1)})
        rows, cohorts = [], {}
        for partition, count in (("fit", 40), ("tune", 10), ("calibration", 10), ("engineering_heldout", 4)):
            key = "pilot_heldout" if partition == "engineering_heldout" else "pilot_" + partition
            cohorts[key] = []
            for i in range(count):
                identifier = partition + f"{i:02}"
                cohorts[key].append({"public": {"id": identifier}})
                rows.append({"question_id": identifier, "partition": partition, "criterion": "answer", "label": i % 2,
                    "observed": {"raw_logits_false_true": [0., 1. if i % 2 else -1.], "feature": {
                        "path": str(feature_path), "encoder_sha256": frozen, "provenance_sha256": q.fingerprint(identifier)}}})
        now = datetime.now(timezone.utc).isoformat()
        baseline = {"phase": "untouched_pretraining", "terminal": True, "pending_or_uncertain": False,
            "cleanup_verified": True, "checkpoint_head_sha256": q.sha(checkpoint / "head.safetensors"),
            "planned_question_ids": sorted(c["public"]["id"] for c in cohorts["pilot_heldout"]),
            "completed": 4, "failed": 0, "valid_native_forwards": 4, "finished_at": now}
        dump(self.root / "features/features.json", rows)
        dump(self.root / "features/untouched-baseline.json", baseline)
        _, before = q.safe_tensors(checkpoint / "head.safetensors", q.HEAD_SHAPES)
        trained, after = q.safe_tensors(self.root / "train/learned/answer-head.safetensors", q.HEAD_SHAPES)
        _, tensor_sha = q.safe_tensors(feature_path, {"decide": [2048], "options": [2, 2048]})
        cal = [r for r in rows if r["partition"] == "calibration"]
        logits = [r["observed"]["raw_logits_false_true"] for r in cal]
        history = [{"learning_rate": lr, "epoch": epoch, "optimizer_steps": 3, "tune_nll": .1 + epoch / 100,
                    "nonzero_gradient_seen": True} for lr in (1e-5, 5e-5) for epoch in (1, 2, 3)]
        result = {"reload_verified": True, "reload_max_logit_error": 0, "physical_encoder_forwards": 0,
            "torso_loaded": False, "trained_head_sha256": q.sha(self.root / "train/learned/answer-head.safetensors"),
            "trained_head_tensor_sha256": after, "head_delta_l2": abs(trained["q.bias"][2][0]),
            "history": history, "selected": {"learning_rate": 1e-5, "epoch": 1}, "total_optimizer_steps": 18,
            "backward_passes": 18, "calibration_labels": [r["label"] for r in cal], "calibration_questions": 10,
            "calibration_logits_false_true": {"released": logits, "trained": logits},
            "released_calibration": learning.fit_temperature(logits, cal), "trained_calibration": learning.fit_temperature(logits, cal),
            "label_counts": {p: {str(y): sum(r["partition"] == p and r["label"] == y for r in rows) for y in (0, 1)}
                             for p in ("fit", "tune", "calibration")},
            "physical_head_work": {k: {"head_forward_batches": a, "head_scored_states": b} for k, (a, b) in {
                "fit": (18, 240), "tune": (6, 60), "reload_before": (1, 10), "reload_after": (1, 10),
                "released_calibration": (1, 10), "trained_calibration": (1, 10)}.items()}}
        proof = {"status": "completed", "encoder_sha256": frozen, "frozen_torso_adapter_files_unchanged": True,
            "released_head_file_unchanged": True, "head_parameters": 1053184, "baseline_receipt_sha256": q.fingerprint(baseline),
            "baseline_question_ids": baseline["planned_question_ids"], "released_head_sha256": q.sha(checkpoint / "head.safetensors"),
            "initial_head_tensor_sha256": before, "criteria": {"answer": result},
            "partitions": {p: sorted(c["public"]["id"] for c in cohorts["pilot_" + p]) for p in ("fit", "tune", "calibration")},
            "cache_sha256": {p: q.fingerprint([{k: r[k] for k in ("question_id", "partition", "criterion", "label")} | {
                "provenance_sha256": r["observed"]["feature"]["provenance_sha256"], "tensor_sha256": tensor_sha}
                for r in rows if r["partition"] == p]) for p in ("fit", "tune", "calibration")}}
        self.rows, self.cohorts, self.baseline, self.proof = rows, cohorts, baseline, proof
        self.config = {"checkpoint": str(checkpoint), "base": str(base)}
        self.helpers = {"haltiq_tau_hotpot.hobson": SimpleNamespace(encoder_binding=lambda *_: frozen),
                        "haltiq_tau_hotpot.learning": learning}

    def audit(self, proof=None):
        proof = proof or self.proof
        dump(self.root / "train/learning.json", proof)
        dump(self.root / "train/learned/learning-proof.json", proof)
        call = {"pending": {"at": datetime.now(timezone.utc).isoformat(), "request_sha256": q.fingerprint({
                    "features_sha256": q.sha(self.root / "features/features.json"),
                    "baseline_sha256": q.sha(self.root / "features/untouched-baseline.json")})},
                "completed": {"role": "head_training_and_calibration"}, "value": proof}
        return q.learning_proof(self.root, self.config, self.cohorts, self.rows, self.baseline,
                                {"train": {"responses": [call]}}, self.helpers)

    def test_actual_head_artifact_shapes_change_hashes_delta_and_calibration_recompute(self):
        self.assertEqual(self.audit(), self.proof)
        binding = q.model_binding(self.config, self.proof)
        self.assertEqual(binding["head_parameters"], 1053184)
        self.assertEqual(set(binding["upstream_sha256"]), {"modeling.py", "infer.py", "prompting.py", "schema.py"})

    def test_mutated_learning_selection_temperature_cache_update_or_partitions_fail(self):
        for kind in ("selection", "temperature", "cache", "steps", "partitions", "delta", "labels"):
            proof = deepcopy(self.proof)
            result = proof["criteria"]["answer"]
            if kind == "selection": result["selected"]["epoch"] = 3
            if kind == "temperature": result["trained_calibration"]["temperature"] = 9
            if kind == "cache": proof["cache_sha256"]["fit"] = "0" * 64
            if kind == "steps": result["backward_passes"] = 0
            if kind == "partitions": proof["partitions"]["tune"] = proof["partitions"]["fit"][:10]
            if kind == "delta": result["head_delta_l2"] = 0
            if kind == "labels": result["calibration_labels"] = [0] * 10
            with self.subTest(kind=kind), self.assertRaises(q.QualificationError):
                self.audit(proof)

    def test_forged_baseline_time_and_changed_actual_head_cannot_qualify(self):
        self.baseline["finished_at"] = "2999-01-01T00:00:00+00:00"
        self.proof["baseline_receipt_sha256"] = q.fingerprint(self.baseline)
        with self.assertRaisesRegex(q.QualificationError, "preceded"):
            self.audit()
        self.baseline["finished_at"] = "2000-01-01T00:00:00+00:00"
        self.proof["baseline_receipt_sha256"] = q.fingerprint(self.baseline)
        (self.root / "train/learned/answer-head.safetensors").write_bytes(b"FAKE TINY HEAD")
        with self.assertRaises(q.QualificationError):
            self.audit()


class NativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from loguru import logger
        logger.disable("tau2")
        import haltiq_tau_hotpot.agent as agent
        import haltiq_tau_hotpot.domain as domain
        cls.agent, cls.domain = agent, domain
        cls.helpers = {"haltiq_tau_hotpot.agent": agent, "haltiq_tau_hotpot.domain": domain}

    def setUp(self):
        self.case = {"public": {"id": "synthetic", "question": "Who?", "type": "bridge", "level": "hard",
            "context": [["A", ["Alice is here."]]]}, "evaluator": {"answer": "Alice", "supporting_facts": [["A", 0]]}}
        self.draft = {"answer": "Alice", "supporting_facts": [["A", 0]]}

    def episode(self, arm="fixed1", rejected=False):
        return self.agent.run_episode(self.case, lambda *_: deepcopy(self.draft), lambda *_: "retain",
            lambda *_: {"p_true": .1 if rejected else .9}, arm=arm, max_drafts=3, passage_indices=[0])

    def test_actual_native_fixed_replay_checks_tools_submission_before_stop_and_counts(self):
        row = self.episode("fixed3")
        replay = q.native_episode(row, self.case, [0], self.helpers, arm="fixed3", max_drafts=3)
        self.assertEqual(replay["counts"]["answer_drafts"], 3)
        self.assertEqual(replay["counts"]["reviewer_calls"], 2)
        self.assertEqual(replay["native_reward"], 1)
        self.assertEqual(replay["native_simulation"]["messages"][-1]["execution_status"], "executed")

    def test_rejected_gate_cap_is_legitimate_failure_not_a_missing_episode(self):
        row = self.episode("released", rejected=True)
        q.native_episode(row, self.case, [0], self.helpers, arm="released", max_drafts=3)
        self.assertEqual(row["status"], "completed_with_failure")
        self.assertEqual(row["failure_category"], "draft_cap_no_accepted_submission")
        self.assertEqual(row["counts"]["gate_calls"], 3)
        self.assertEqual(row["native_reward"], 0)

    def test_known_parse_failure_replays_as_recorded_failure_without_new_callback_call(self):
        def fail(*_): raise ValueError("synthetic parse failure")
        row = self.agent.run_episode(self.case, fail, lambda *_: "retain", arm="fixed3", max_drafts=3, passage_indices=[0])
        q.native_episode(row, self.case, [0], self.helpers, arm="fixed3", max_drafts=3)
        self.assertEqual(row["failure_category"], "writer_value_error")

    def test_mutated_native_reward_metrics_counts_or_tool_result_is_rejected(self):
        row = self.episode()
        for kind in ("reward", "metric", "count", "tool", "gold"):
            altered, case = deepcopy(row), deepcopy(self.case)
            if kind == "reward": altered["native_reward"] = 0
            if kind == "metric": altered["metrics"]["f1"] = 0
            if kind == "count": altered["counts"]["answer_drafts"] = 10
            if kind == "tool": altered["native_simulation"]["messages"][-1]["content"] = '{"submitted":false}'
            if kind == "gold": case["evaluator"]["answer"] = "Bob"
            with self.subTest(kind=kind), self.assertRaises(q.QualificationError):
                q.native_episode(altered, case, [0], self.helpers, arm="fixed1", max_drafts=3)


if __name__ == "__main__":
    unittest.main()
