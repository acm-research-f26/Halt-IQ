"""Protocol tests run without downloading torch or any model weights."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

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
