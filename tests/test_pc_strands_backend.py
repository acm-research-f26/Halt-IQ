"""Protocol tests run without downloading torch or any model weights."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pc_strands.calibrate import binary_metrics, fit_temperature, validate_rows
from pc_strands.decider import (
    ContextWindowError, FeatureRecord, _checkpoint_config, cache_trajectories, label_for_order,
    option_token_indices, probability_true, validate_public_state,
)
from pc_strands.train_head import load_feature_rows, validate_baseline_receipt, validate_disjoint


def feature_row(qid="q1", partition="fit", label=1, order=(0, 1)):
    row = FeatureRecord([0.0] * 2048, [[0.0] * 2048, [1.0] * 2048], order,
                        200, "prompt-hash", "state-hash", 0.1).to_dict()
    return {**row, "question_id": qid, "partition": partition, "label": label}


class PublicInputTests(unittest.TestCase):
    def setUp(self):
        self.state = {"question": "Who?", "evidence": ["Title: all sentences."], "draft": "A person"}

    def test_gold_and_metadata_are_rejected(self):
        for key in ("answer", "supporting_facts", "exact_match", "question_id", "split", "api_key"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_public_state({**self.state, key: "hidden"})

    def test_evidence_order_and_complete_strings_are_preserved(self):
        self.state["evidence"] = ["Second title.\nAll sentences.", "First title.\nMore sentences."]
        self.assertEqual(validate_public_state(self.state), self.state)

    def test_history_requires_explicit_separate_arm(self):
        state = {**self.state, "history": [{"draft": "earlier", "feedback": "public"}],
                 "round": 10, "budget": {"drafts_remaining": 0}}
        with self.assertRaises(ValueError):
            validate_public_state(state)
        self.assertEqual(validate_public_state(state, allow_history=True)["round"], 10)
        state["history"][0]["gold"] = "not public"
        with self.assertRaises(ValueError):
            validate_public_state(state, allow_history=True)

    def test_full_window_error_keeps_real_counts(self):
        error = ContextWindowError(4097, 4096)
        self.assertEqual((error.token_count, error.max_length), (4097, 4096))

    def test_pointer_uses_last_complete_option_token(self):
        offsets = [(0, 0), (0, 2), (2, 4), (5, 8), (8, 11)]
        self.assertEqual(option_token_indices(offsets, [(0, 4), (5, 11)], 0), [2, 4])
        with self.assertRaises(ValueError):
            option_token_indices(offsets, [(12, 15)], 0)

    def test_reverse_order_remaps_label(self):
        self.assertEqual(label_for_order(1, (1, 0)), 0)
        self.assertEqual(label_for_order(0, (1, 0)), 1)
        with self.assertRaises(ValueError):
            label_for_order(1, (1, 1))


class CachedTrainingTests(unittest.TestCase):
    def test_cache_cli_activates_admitted_cpu_load_before_rechecking_ram(self):
        from pc_strands.decider import main
        from pc_strands.control import GpuLease as ActualLease, ResourcePressure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks-evaluator-only.json").write_text(json.dumps([{"id": "q", "question": "Q?",
                "evidence": ["E"], "answers": ["reference"], "partition": "fit"}]))
            (root / "trajectories.json").write_text(json.dumps([{"task_id": "q", "status": "completed",
                "rounds": [{"round": 1, "status": "completed", "draft": "draft"}]}]))
            (root / "manifest.json").write_text(json.dumps({"status": "completed", "writer_unload_verified": True}))
            lock, output = root / "owned.lock", root / "features.jsonl"
            state, events = {"held": False, "loaded": False, "active": False}, []
            test = self
            class Lease:
                def __init__(self, path, _): self.actual = ActualLease(path, "test")
                def __enter__(self):
                    self.actual.__enter__(); state["held"] = True; events.append("owned")
                def __exit__(self, *args):
                    self.actual.__exit__(*args); state["held"] = False; events.append("released")
            class Admission:
                def __init__(self, *args): pass
                def before_work(self, **kwargs):
                    test.assertTrue(state["held"])
                    if state["loaded"] and not state["active"]:
                        raise RuntimeError("simulated host RAM below threshold until placement")
                    events.append("admitted")
                def rest_after_block(self, _):
                    test.assertTrue(state["active"])
            class Backend:
                device, max_length = "cuda", 4096
                def __init__(self, *args, **kwargs):
                    test.assertTrue(state["held"])
                    test.assertEqual(events[-1], "admitted")
                    with test.assertRaises(ResourcePressure):
                        with ActualLease(lock, "conflict"): pass
                    state["loaded"] = True; events.append("loaded_cpu")
                def activate(self): state["active"] = True; events.append("activated"); return 0
                def metadata(self): return {"adapter_sha256": "mock-only"}
                def extract(self, state_value, order):
                    test.assertTrue(state["active"])
                    return FeatureRecord([0.] * 2048, [[0.] * 2048, [1.] * 2048], tuple(order), 100, "p", "s", .1)
                def offload(self): state["active"] = False; events.append("offloaded")
            argv = ["decider", "cache", "--checkpoint", str(root / "checkpoint"), "--base", str(root / "base"),
                    "--trajectory-dir", str(root), "--output", str(output), "--partition", "fit",
                    "--canonical-order-only", "--deadline", "2099-01-01T20:00:00-05:00", "--gpu-lock", str(lock)]
            with patch("sys.argv", argv), patch("pc_strands.control.GpuLease", Lease), \
                 patch("pc_strands.control.RunControl", Admission), patch("pc_strands.decider.StrandsDecider", Backend):
                main()
            self.assertEqual(events[:4], ["owned", "admitted", "loaded_cpu", "activated"])
            self.assertEqual(events[-1], "released")
            saved = json.loads(Path(str(output) + ".manifest.json").read_text())
            self.assertTrue(saved["backend_startup_this_invocation"]["explicit_activation"])
            self.assertEqual(saved["physical_encoding_usage"]["completed_encoder_forwards"], 1)

    def test_cache_activation_failure_cleans_owned_backend_and_releases_lease(self):
        from pc_strands.decider import main
        from pc_strands.control import GpuLease as ActualLease
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lock = root / "owned.lock"; events = []
            class Admission:
                def __init__(self, *args): pass
                def before_work(self, **kwargs): pass
            class Backend:
                def __init__(self, *args, **kwargs): events.append("constructed")
                def activate(self): raise RuntimeError("mock placement failure")
                def offload(self): events.append("offloaded")
            argv = ["decider", "cache", "--checkpoint", "unused", "--base", "unused", "--trajectory-dir", str(root),
                    "--output", str(root / "features"), "--deadline", "2099-01-01T20:00:00-05:00", "--gpu-lock", str(lock)]
            with patch("sys.argv", argv), patch("pc_strands.control.RunControl", Admission), \
                 patch("pc_strands.decider.StrandsDecider", Backend), self.assertRaisesRegex(RuntimeError, "placement failure"):
                main()
            self.assertEqual(events, ["constructed", "offloaded"])
            with ActualLease(lock, "successfully reacquired"):
                pass

    def test_cache_phase_idle_excludes_prior_startup_wait(self):
        clock = {"seconds": 0.0}
        class Control:
            idle_seconds = 100.0
            def before_work(self, **kwargs):
                clock["seconds"] += 3.0
                self.idle_seconds += 3.0
            def rest_after_block(self, elapsed):
                clock["seconds"] += elapsed
                self.idle_seconds += elapsed
        class Backend:
            device, max_length = "cpu", 4096
            def metadata(self): return {"adapter_sha256": "mock-only"}
            def extract(self, state, order):
                clock["seconds"] += 2.0
                return FeatureRecord([0.] * 2048, [[0.] * 2048, [1.] * 2048], tuple(order), 100, "p", "s", 2.0)
            def offload(self): pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks-evaluator-only.json").write_text(json.dumps([{"id": "q", "question": "Q?",
                "evidence": ["E"], "answers": ["reference"], "partition": "fit"}]))
            (root / "trajectories.json").write_text(json.dumps([{"task_id": "q", "status": "completed",
                "rounds": [{"round": 1, "status": "completed", "draft": "draft"}]}]))
            (root / "manifest.json").write_text(json.dumps({"status": "completed", "writer_unload_verified": True}))
            with patch("pc_strands.decider.time.perf_counter", side_effect=lambda: clock["seconds"]):
                result = cache_trajectories(Backend(), root, root / "features.jsonl", control=Control(),
                                            startup_metadata={"intentional_idle_seconds": 100.0})
            self.assertEqual(result["this_invocation_wall_seconds"], 7.0)
            self.assertEqual(result["this_invocation_intentional_idle_seconds"], 3.0)
            self.assertLessEqual(result["this_invocation_intentional_idle_seconds"], result["this_invocation_wall_seconds"])
            self.assertEqual(result["backend_startup_this_invocation"]["intentional_idle_seconds"], 100.0)

    def test_baseline_keeps_failure_denominator_but_requires_real_finished_forward(self):
        receipt = {"phase": "untuned_pretraining", "planned_task_ids": [f"q{i}" for i in range(20)],
                   "attempted": 20, "completed": 0, "failed": 20, "valid_decision_forwards": 1,
                   "pending_requests": 0, "uncertain_requests": 0, "checkpoint_head_sha256": "original"}
        self.assertEqual(len(validate_baseline_receipt(receipt, "original")), 20)
        for key, value in (("attempted", 19), ("valid_decision_forwards", 0), ("pending_requests", 1),
                           ("uncertain_requests", 1), ("failed", 19), ("completed", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_baseline_receipt({**receipt, key: value}, "original")

    def test_cache_refuses_confirmation_and_malformed_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(feature_row(partition="confirmation")) + "\n")
            with self.assertRaises(ValueError):
                load_feature_rows(path, "fit")
            with self.assertRaises(ValueError):
                load_feature_rows(path, "calibration")
        row = feature_row()
        row["decide"] = [1.0]
        with self.assertRaises(ValueError):
            FeatureRecord.from_dict(row)

    def test_cache_rejects_nonfinite_and_oversized_states(self):
        row = feature_row()
        row["options"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            FeatureRecord.from_dict(row)
        row = feature_row()
        row["input_tokens"] = True
        with self.assertRaises(ValueError):
            FeatureRecord.from_dict(row)
        row = feature_row(order=(False, True))
        with self.assertRaises(ValueError):
            FeatureRecord.from_dict(row)
        row = feature_row()
        row["input_tokens"] = 4097
        with self.assertRaises(ValueError):
            FeatureRecord.from_dict(row)

    def test_question_group_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_disjoint([{"question_id": "same"}], [{"question_id": "same"}])
        self.assertEqual(validate_disjoint([{"question_id": "fit"}], [{"question_id": "tune"}])["fit_questions"], 1)

    def test_released_pointer_shape_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strands_decider_config.json"
            path.write_text(json.dumps({"head_type": "slot", "pointer_dim": 768}))
            with self.assertRaises(ValueError):
                _checkpoint_config(Path(directory))

    def test_cache_public_input_and_resume_ledger_preserve_overflows(self):
        class FakeDecider:
            device, max_length = "cpu", 4096
            def __init__(self):
                self.calls = 0
            def metadata(self):
                return {"adapter_sha256": "test-only"}
            def extract(self, state, order):
                self.calls += 1
                if set(state) != {"question", "evidence", "draft"}:
                    raise AssertionError("private metadata entered model")
                if state["question"] == "long":
                    raise ContextWindowError(4097, 4096)
                return FeatureRecord([0.0] * 2048, [[0.0] * 2048, [1.0] * 2048],
                                     tuple(order), 200, "prompt", "state", 0.1)
            def offload(self):
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [{"id": name, "question": question, "evidence": ["public evidence"],
                      "answers": ["private reference"], "partition": "fit"}
                     for name, question in [("q1", "short"), ("q2", "long")]]
            rounds = [{"task_id": t["id"], "status": "completed", "rounds": [
                {"round": 1, "status": "completed", "draft": "public draft"}]} for t in tasks]
            (root / "tasks-evaluator-only.json").write_text(json.dumps(tasks))
            (root / "trajectories.json").write_text(json.dumps(rounds))
            (root / "manifest.json").write_text(json.dumps({"status": "completed", "writer_unload_verified": True}))
            output = root / "features.jsonl"
            backend = FakeDecider()
            result = cache_trajectories(backend, root, output, partition="fit")
            self.assertEqual(result["status"], "completed_with_overflows")
            self.assertEqual(result["physical_encoding_usage"]["completed_encoder_forwards"], 2)
            self.assertEqual(result["total_overflow_rows"], 2)
            self.assertEqual(backend.calls, 4)
            backend.calls = 0
            resumed = cache_trajectories(backend, root, output, partition="fit", resume=True)
            self.assertEqual(resumed["status"], "completed_with_overflows")
            self.assertEqual(backend.calls, 0)
            self.assertEqual(resumed["total_overflow_rows"], 2)
            self.assertEqual(len(output.read_text().splitlines()), 2)
            # A complete-line truncation must also fail; newline checks alone are insufficient.
            original = output.read_text()
            output.write_text(original.splitlines()[0] + "\n")
            with self.assertRaises(ValueError):
                cache_trajectories(backend, root, output, partition="fit", resume=True)
            self.assertEqual(backend.calls, 0)
            output.write_text(original)
            failures = Path(str(output) + ".failures.jsonl")
            failures.write_text("")
            with self.assertRaises(ValueError):
                cache_trajectories(backend, root, output, partition="fit", resume=True)
            self.assertEqual(backend.calls, 0)


class CalibrationTests(unittest.TestCase):
    def test_probability_is_canonical_stable_and_unrounded(self):
        self.assertEqual(probability_true([1000.0, -1000.0]), 0.0)
        self.assertEqual(probability_true([-1000.0, 1000.0]), 1.0)
        self.assertAlmostEqual(probability_true([0, math.log(4)]), 0.8)
        with self.assertRaises(ValueError):
            probability_true([0, 1], 0)

    def test_fit_reduces_held_calibration_nll_and_preserves_argmax(self):
        rows = [{"question_id": f"q{i}", "partition": "calibration", "label": y,
                 "logits": [0.0, z]} for i, (y, z) in enumerate([(1, 8), (0, -8), (0, 8), (1, -8)])]
        result = fit_temperature(rows)
        self.assertLess(result["after"]["nll"], result["before"]["nll"])
        self.assertEqual(result["after"]["accuracy"], result["before"]["accuracy"])
        self.assertGreaterEqual(result["temperature"], 0.25)
        self.assertLessEqual(result["temperature"], 6)

    def test_reliability_bins_form_disjoint_partition(self):
        rows = [{"label": 1, "logits": [0.0, math.log(p / (1 - p))]}
                for p in (0.1, 0.2, 0.3, 0.8, 0.9)]
        metrics = binary_metrics(rows, 1)
        self.assertEqual(sum(x["n"] for x in metrics["reliability"]), len(rows))

    def test_calibration_rejects_wrong_split_and_duplicate_question(self):
        row = {"question_id": "q", "partition": "fit", "label": 1, "logits": [0, 1]}
        with self.assertRaises(ValueError):
            validate_rows([row])
        row["partition"] = "calibration"
        with self.assertRaises(ValueError):
            validate_rows([row, row])


if __name__ == "__main__":
    unittest.main()
