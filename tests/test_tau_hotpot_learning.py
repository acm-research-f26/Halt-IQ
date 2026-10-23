"""Synthetic CPU tensors test mechanics only; none are model-quality evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from haltiq_tau_hotpot.hobson import (FEATURE_SCHEMA, HEAD_PARAMETERS, HOBSON_TEMPERATURE,
    INSTRUCTIONS_BY_CRITERION, HobsonBackend, encoder_binding, feature_tensors, file_hash,
    load_head, score_head, stable_hash)
from haltiq_tau_hotpot.learning import (fit_temperature, grouped_nll, tensor_hash,
    train_heads, validate_baseline, validate_partitions)
import haltiq_tau_hotpot.hobson as hobson_module

TORCH_RUNTIME = all(importlib.util.find_spec(name) is not None for name in
                    ("torch", "safetensors", "strands_decider"))


class ImportBoundaryTests(unittest.TestCase):
    def test_resource_pause_does_not_become_active_training_or_an_extra_work_rest(self):
        from haltiq_tau_hotpot.learning import _Budget
        now, rests = [0.], []
        class Control:
            idle_seconds = 0.
            def before_work(self, **kwargs):
                now[0] += 30.
                self.idle_seconds += 30.
            def rest_after_block(self, seconds):
                rests.append(seconds)
                now[0] += seconds
                self.idle_seconds += seconds
        with patch("haltiq_tau_hotpot.learning.time.perf_counter", side_effect=lambda: now[0]):
            budget = _Budget(Control(), 900)
            now[0] += 6.
            budget.check()
            now[0] += 4.
            budget.finish()
        self.assertEqual(rests, [6., 4.])

    def test_importing_interfaces_does_not_import_torch_transformers_or_peft(self):
        code = ("import sys;import haltiq_tau_hotpot.hobson,haltiq_tau_hotpot.learning;"
                "assert not any(x in sys.modules for x in ('torch','transformers','peft'));print('safe')")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "safe")

    def test_calibration_uses_finite_raw_logits_and_does_not_increase_fitting_nll(self):
        rows = [{"question_id": "one", "label": 1}, {"question_id": "two", "label": 0}]
        logits = [[-1000., 1000.], [1000., -1000.]]
        fitted = fit_temperature(logits, rows)
        self.assertTrue(.05 <= fitted["temperature"] <= 20)
        self.assertLessEqual(fitted["calibration_fitting_nll"], grouped_nll(logits, rows))
        with self.assertRaises(ValueError):
            fit_temperature([[float("nan"), 0], [0, 1]], rows)

    def test_question_grouped_nll_does_not_overweight_duplicate_variants(self):
        rows = [{"question_id": "a", "label": 1}, {"question_id": "a", "label": 1},
                {"question_id": "b", "label": 0}]
        logits = [[0., 1.], [0., 1.], [0., 1.]]
        self.assertAlmostEqual(grouped_nll(logits, rows), grouped_nll([logits[0], logits[2]], [rows[0], rows[2]]))


@unittest.skipUnless(TORCH_RUNTIME, "CPU synthetic learning tests need the installed Strands runtime")
class SyntheticPointerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        from strands_decider.modeling import PointerHead
        self.torch = torch
        torch.set_num_threads(4)
        torch.manual_seed(42)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.checkpoint, self.base = self.root / "released", self.root / "base"
        (self.checkpoint / "lora").mkdir(parents=True)
        self.base.mkdir()
        config = {"base_model": "Qwen/Qwen3.5-2B-Base", "head_type": "pointer", "pointer_dim": 256,
            "lora_r": 16, "head_dropout": .05, "temperature_by_kind": {"noul": HOBSON_TEMPERATURE}}
        (self.checkpoint / "hobson_config.json").write_text(json.dumps(config))
        for name in ("tokenizer.json", "tokenizer_config.json", "lora/adapter_config.json"):
            (self.checkpoint / name).write_text("{}")
        # These bytes are never loaded as a torso/adapter. They test immutable file binding.
        (self.checkpoint / "lora/adapter_model.safetensors").write_bytes(b"SYNTHETIC-NOT-A-MODEL")
        (self.base / "config.json").write_text("{}")
        (self.base / "model.safetensors").write_bytes(b"SYNTHETIC-NOT-A-MODEL")
        self.head = PointerHead(2048, dim=256, dropout=.05).float().eval()
        save_file(self.head.state_dict(), str(self.checkpoint / "head.safetensors"))
        self.encoder = encoder_binding(self.checkpoint, self.base)
        self.rows = {}
        for partition in ("fit", "tune", "calibration"):
            self.rows[partition] = [self.row(partition + str(i), partition, i % 2) for i in range(2)]
        self.baseline = {"phase": "untouched_pretraining", "terminal": True,
            "pending_or_uncertain": False, "cleanup_verified": True,
            "checkpoint_head_sha256": file_hash(self.checkpoint / "head.safetensors"),
            "valid_native_forwards": 4, "completed": 4, "failed": 0,
            "planned_question_ids": ["baseline" + str(i) for i in range(4)]}

    def feature(self, name="synthetic", criterion="answer"):
        torch = self.torch
        provenance = {"criterion": criterion, "state_sha256": stable_hash(name),
            "instructions_sha256": stable_hash(INSTRUCTIONS_BY_CRITERION[criterion]),
            "rendered_question_sha256": "1" * 64, "token_ids_sha256": "2" * 64,
            "source_sha256": file_hash(hobson_module.__file__), "option_order": ["false", "true"]}
        return {"schema": FEATURE_SCHEMA, "criterion": criterion,
            "option_order": ["false", "true"], "encoder_sha256": dict(self.encoder),
            "provenance": provenance, "provenance_sha256": stable_hash(provenance),
            "decide": torch.randn(2048), "options": torch.randn(2, 2048)}

    def row(self, name, partition, label, criterion="answer"):
        return {"question_id": name, "partition": partition, "criterion": criterion,
                "label": label, "feature": self.feature(name, criterion)}

    def train(self, output="trained", **kwargs):
        return train_heads(self.checkpoint, self.base, self.rows["fit"], self.rows["tune"],
            self.rows["calibration"], self.root / output, self.baseline,
            learning_rates=(1e-4,), epochs=1, batch_size=2, max_seconds=60,
            minimum_questions={"fit": 2, "tune": 2, "calibration": 2}, **kwargs)

    def test_exact_compatible_head_and_canonical_probability_from_cpu_cache(self):
        head = load_head(self.checkpoint)
        self.assertEqual(sum(p.numel() for p in head.parameters()), HEAD_PARAMETERS)
        self.assertEqual(tensor_hash(head.state_dict()), tensor_hash(self.head.state_dict()))
        feature = self.feature()
        result = score_head(head, feature, temperature=1)
        logits = head(feature["decide"][None], feature["options"][None])[0]
        expected = float(self.torch.softmax(logits, dim=-1)[1])
        self.assertAlmostEqual(result["p_true_unrounded"], expected)
        self.assertEqual(result["head_scores"], 1)
        self.assertEqual(result["physical_forwards"], 0)

    def test_wrong_encoder_option_order_criterion_shape_and_nonfinite_rejected(self):
        feature = self.feature()
        for change in ({"option_order": ["true", "false"]}, {"criterion": "evidence"},
                       {"decide": [0.]}, {"options": self.torch.full((2, 2048), float("nan"))}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                feature_tensors({**feature, **change}, expected_encoder=self.encoder, criterion="answer")
        with self.assertRaisesRegex(ValueError, "different frozen encoder"):
            feature_tensors(feature, expected_encoder={})

    def test_frozen_old768_features_and_repurposed_partitions_cannot_train(self):
        self.rows["fit"][0]["feature"]["schema"] = "old-rank48-feature"
        with self.assertRaises(ValueError):
            self.train()
        self.rows["fit"][0]["feature"] = self.feature()
        self.rows["calibration"][0]["question_id"] = self.rows["fit"][0]["question_id"]
        with self.assertRaisesRegex(ValueError, "question overlap"):
            self.train()

    def test_baseline_missing_forward_cleanup_pending_or_training_overlap_blocks_updates(self):
        for changes in ({"valid_native_forwards": 0}, {"pending_or_uncertain": True},
                        {"cleanup_verified": False}, {"checkpoint_head_sha256": "old"},
                        {"planned_question_ids": ["fit0", "b", "c", "d"]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_baseline({**self.baseline, **changes}, self.baseline["checkpoint_head_sha256"], ["fit0"])

    def test_actual_synthetic_gradient_updates_existing_head_and_exact_reload(self):
        original = file_hash(self.checkpoint / "head.safetensors")
        proof = self.train()
        result = proof["criteria"]["answer"]
        self.assertEqual(proof["status"], "completed")
        self.assertEqual(proof["head_parameters"], HEAD_PARAMETERS)
        self.assertGreater(result["head_delta_l2"], 0)
        self.assertEqual(result["reload_max_logit_error"], 0)
        self.assertTrue(result["reload_verified"])
        self.assertEqual(file_hash(self.checkpoint / "head.safetensors"), original)
        self.assertEqual(encoder_binding(self.checkpoint, self.base), self.encoder)
        self.assertIn("released_calibration", result)
        self.assertIn("trained_calibration", result)
        self.assertNotEqual(result["trained_head_sha256"], original)
        with self.assertRaisesRegex(ValueError, "new output directory"):
            self.train()

    def test_evidence_has_separate_head_and_temperature_not_shared_answer_label(self):
        for partition in self.rows:
            self.rows[partition].extend(self.row(partition + str(i), partition, 1 - i % 2, "evidence") for i in range(2))
        proof = self.train()
        self.assertEqual(set(proof["criteria"]), {"answer", "evidence"})
        self.assertNotEqual(proof["criteria"]["answer"]["trained_head_sha256"],
                            proof["criteria"]["evidence"]["trained_head_sha256"])
        self.assertTrue((self.root / "trained/evidence-head.safetensors").is_file())

    def test_one_class_tune_and_calibration_are_retained_with_explicit_warnings(self):
        for partition in ("tune", "calibration"):
            for row in self.rows[partition]:
                row["label"] = 1
        proof = self.train()
        result = proof["criteria"]["answer"]
        self.assertEqual(result["label_counts"]["calibration"], {"0": 0, "1": 2})
        self.assertEqual(len(result["warnings"]), 2)
        self.assertEqual(len(proof["partitions"]["calibration"]), 2)
        self.assertTrue(all("AUROC is undefined" in message for message in result["warnings"]))

    def test_physical_head_work_counts_all_grid_reload_and_calibration_calls(self):
        proof = self.train()
        result = proof["criteria"]["answer"]
        self.assertEqual(result["total_optimizer_steps"], 1)
        self.assertEqual(result["backward_passes"], 1)
        self.assertTrue(result["history"][0]["nonzero_gradient_seen"])
        self.assertEqual(set(result["physical_head_work"]), {"fit", "tune", "reload_before", "reload_after",
                                                          "released_calibration", "trained_calibration"})
        for role in result["physical_head_work"].values():
            self.assertEqual(role, {"head_forward_batches": 1, "head_scored_states": 2})
        self.assertEqual(result["physical_encoder_forwards"], 0)
        self.assertFalse(result["torso_loaded"])

    def test_calibration_labels_do_not_affect_selected_head_or_tuning_grid(self):
        first = self.train("one")
        for row in self.rows["calibration"]:
            row["label"] = 1 - row["label"]
        second = self.train("two")
        a, b = first["criteria"]["answer"], second["criteria"]["answer"]
        self.assertEqual(a["trained_head_tensor_sha256"], b["trained_head_tensor_sha256"])
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["selected"], b["selected"])
        self.assertNotEqual(a["trained_calibration"]["temperature"], b["trained_calibration"]["temperature"])

    def test_file_mutation_during_cpu_training_prevents_complete_proof(self):
        def changed(*args, **kwargs):
            (self.base / "model.safetensors").write_bytes(b"CHANGED-SYNTHETIC-FIXTURE")
        class Control:
            def before_work(self, **kwargs): changed()
            def rest_after_block(self, seconds): pass
        with self.assertRaisesRegex(RuntimeError, "Immutable released"):
            self.train(control=Control())
        self.assertFalse((self.root / "trained/learning-proof.json").exists())

    def test_observer_hook_caches_real_head_inputs_without_second_encoder_forward(self):
        torch, test = self.torch, self
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = copy.deepcopy(test.head).requires_grad_(False)
                self.calls = 0
            def forward(self, **kwargs):
                self.calls += 1
                return {"logits": self.head(torch.ones(1, 2048), torch.arange(4096.).reshape(1, 2, 2048))}
        class Engine:
            device = "cpu"
            def _fit(self, state, question):
                return [1, 2], [[3, 4]], [[(0, 1), (1, 2)]]
            def _pad(self, sequences):
                return torch.tensor(sequences), torch.ones(1, len(sequences[0]))
            def _option_idx(self, *args):
                return torch.tensor([[2, 3]])
        class Official:
            def __init__(self, *args, **kwargs):
                self.torch, self.model, self.engine, self.metadata = torch, Model(), None, {}
            def activate(self): self.engine = Engine()
            def close(self): return {"allocated_after_cleanup_bytes": 0}
        with patch("haltiq_tau_hotpot.hobson.OfficialDecider", Official), patch.object(torch.cuda, "synchronize"):
            backend = HobsonBackend(self.checkpoint, self.base)
            backend.activate()
            prepared = backend.prepare_state("Synthetic public observation")
            result = backend.observe("Synthetic public observation", prepared=prepared)
            self.assertEqual(backend._official.model.calls, 1)
            self.assertEqual(result["physical_forwards"], 1)
            self.assertEqual(result["head_scores"], 2)
            self.assertEqual(result["cached_replay_max_logit_error"], 0)
            self.assertEqual(result["feature"]["decide"].device.type, "cpu")
            self.assertNotIn("label", result["feature"])
            with self.assertRaisesRegex(ValueError, "Prepared tokens"):
                backend.observe("different state", prepared=prepared)
            self.assertEqual(backend._official.model.calls, 1)
            self.assertEqual(backend.close()["allocated_after_cleanup_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
