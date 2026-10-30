"""Controls for a separately declared exact-match correctness classifier."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from haltiq_pair import quality_training as quality
from haltiq_pair.replay import load_trajectories
from haltiq_pair.team_training import StrandsHead, make_strands_features, replay_one


def trajectory(qid, split):
    return {"id": qid, "split": split, "rounds": [
        {"round": i + 1, "answer": "same", "split": split,
         "strands": .1 if i % 2 == 0 else .9, "em": float(i % 2), "f1": 0.,
         "tokens": 30, "writer_tokens": 10, "critic_tokens": 20}
        for i in range(6)]}


class CorrectnessObjectiveControls(unittest.TestCase):
    def test_balanced_bce_has_equal_question_weight_and_train_ratio(self):
        logits = torch.zeros(6)
        labels = torch.tensor([0., 0., 0., 1., 1., 1.])
        weights = torch.full((6,), 1/6)
        value = quality._balanced_bce(logits, labels, weights, 2., 1).item()
        self.assertAlmostEqual(value, 1.5 * math.log(2), places=6)

    def test_actual_fit_learns_correctness_even_when_f1_is_constant(self):
        train = [trajectory("t1", "train"), trajectory("t2", "train")]
        dev = [trajectory("d1", "dev")]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "head"
            fitted = quality._fit_seed(train, dev, 0, quality.QualityConfig(learning_rate=.05), folder, {})
            low = make_strands_features("same", "same", 2, 6, .1, 0.)
            high = make_strands_features("same", "same", 2, 6, .9, 0.)
            self.assertLess(fitted.probability(low), .5)
            self.assertGreater(fitted.probability(high), .5)
            self.assertGreater(fitted.probability(high) - fitted.probability(low), .25)
            self.assertGreater(fitted.weights[3], 0)
            self.assertEqual(fitted.metadata["objective"], quality.OBJECTIVE)
            history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(history["optimizer_steps_completed"], 200)
            self.assertGreater(history["final_parameter_l2_from_initial"], 0)
            self.assertTrue(all(row["optimizer_parameter_step_l2"] > 0 for row in history["epochs"]))

    def test_class_balance_and_optimizer_steps_never_use_dev_labels(self):
        train = [trajectory("train", "train")]
        dev = [trajectory("dev", "dev")]
        changed = copy.deepcopy(dev)
        for row in changed[0]["rounds"]:
            row["em"] = 0.
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            config = quality.QualityConfig(epochs=5)
            aa = quality._fit_seed(train, dev, 1, config, a, {})
            bb = quality._fit_seed(train, changed, 1, config, b, {})
            ah = json.loads((a / "history.json").read_text(encoding="utf-8"))
            bh = json.loads((b / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(aa.metadata["train_balance"], bb.metadata["train_balance"])
            self.assertEqual([r["weights"] for r in ah["epochs"]], [r["weights"] for r in bh["epochs"]])
            self.assertEqual([r["bias"] for r in ah["epochs"]], [r["bias"] for r in bh["epochs"]])

    def test_one_class_or_overlapping_partitions_fail_before_output(self):
        train, dev = [trajectory("same", "train")], [trajectory("same", "dev")]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "bad"
            with self.assertRaises(ValueError):
                quality._fit_seed(train, dev, 0, quality.QualityConfig(), folder, {})
            dev[0]["id"] = "different"
            for row in train[0]["rounds"]:
                row["em"] = 0.
            with self.assertRaises(ValueError):
                quality._fit_seed(train, dev, 0, quality.QualityConfig(), folder, {})
            self.assertFalse(folder.exists())

    def test_reloaded_head_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "head"
            fitted = quality._fit_seed([trajectory("t", "train")], [trajectory("d", "dev")],
                                      0, quality.QualityConfig(epochs=2), folder, {})
            self.assertEqual(fitted, StrandsHead.from_json(folder / "head.json"))
            data = json.loads((folder / "head.json").read_text(encoding="utf-8"))
            data["bias"] += .01
            (folder / "head.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                StrandsHead.from_json(folder / "head.json")

    def test_inference_actions_cannot_access_gold_labels(self):
        original = trajectory("q", "test")
        changed = copy.deepcopy(original)
        for row in changed["rounds"]:
            row["em"], row["f1"] = 1-row["em"], 1-row["f1"]
        head = StrandsHead((0., 0., 0., 10., 0.), -5., metadata={"objective": quality.OBJECTIVE})
        for policy in ("retrained_head", "retrained_head_and"):
            a, b = replay_one(original, policy, head), replay_one(changed, policy, head)
            self.assertEqual((a["rounds"], a["decisions"], a["llm_tokens"]),
                             (b["rounds"], b["decisions"], b["llm_tokens"]))


class QualityFreezeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv"
        cls.items = load_trajectories(cls.data)
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "study"
        cls.probed = False

        def before_test(items, directory):
            freeze = json.loads((directory / "training-freeze.json").read_text(encoding="utf-8"))
            assert freeze["seeds"] == list(quality.SEEDS)
            assert freeze["all_selected_checkpoints_frozen_before_test_scoring"]
            assert not (directory / "test80.json").exists()
            expected_pos = sum(r["em"] for t in items if t["split"] == "train" for r in t["rounds"])
            for entry in freeze["checkpoints"]:
                path = directory / entry["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
                head = StrandsHead.from_json(path)
                assert len(head.metadata["train_ids"]) == 50 and len(head.metadata["dev_ids"]) == 20
                assert head.metadata["train_balance"]["pos_weight"] == (300-expected_pos)/expected_pos
                assert head.metadata["objective"] == quality.OBJECTIVE
            cls.probed = True
            return {"unique_test_questions": 80}

        with patch.object(quality, "evaluate_frozen", side_effect=before_test):
            quality.train_study(cls.data, cls.output, quality.QualityConfig(epochs=2))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_five_development_selected_heads_exist_before_test_scoring(self):
        self.assertTrue(self.probed)

    def test_test_label_mutation_leaves_every_fitted_checkpoint_identical(self):
        changed = copy.deepcopy(self.items)
        for item in changed:
            if item["split"] == "test":
                for row in item["rounds"]:
                    row["em"], row["f1"] = 1-row["em"], 1-row["f1"]
        output = Path(self.temp.name) / "mutated-test"
        with patch.object(quality, "load_trajectories", return_value=changed), \
                patch.object(quality, "evaluate_frozen", return_value={}):
            quality.train_study(self.data, output, quality.QualityConfig(epochs=2))
        for seed in quality.SEEDS:
            self.assertEqual((self.output / f"seed{seed}/head.json").read_bytes(),
                             (output / f"seed{seed}/head.json").read_bytes())

    def test_changed_frozen_checkpoint_is_rejected(self):
        folder = Path(self.temp.name) / "changed-head"
        shutil.copytree(self.output, folder)
        path = folder / "seed0/head.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            quality.evaluate_frozen(self.items, folder)

    def test_other_eighty_question_partition_is_rejected(self):
        changed = copy.deepcopy(self.items)
        next(t for t in changed if t["split"] == "test")["id"] = "other-question"
        with self.assertRaises(ValueError):
            quality.evaluate_frozen(changed, self.output)

    def test_existing_output_is_preserved(self):
        before = (self.output / "training-freeze.json").read_bytes()
        with self.assertRaises(FileExistsError):
            quality.train_study(self.data, self.output)
        self.assertEqual(before, (self.output / "training-freeze.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
