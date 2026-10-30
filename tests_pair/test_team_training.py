"""Scientific controls for the newly trained Strands-feature stopping head."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from haltiq_pair import team_training as training
from haltiq_pair.replay import load_trajectories


def trajectory(qid="q", split="test", rounds=6):
    return {"id": qid, "split": split, "rounds": [
        {"round": i + 1, "answer": "wrong" if i == 0 else "right", "split": split,
         "strands": .1 if i == 0 else .9, "f1": float(i > 0), "em": float(i > 0),
         "tokens": 30, "writer_tokens": 10, "critic_tokens": 20}
        for i in range(rounds)]}


class EncoderAndPortableHead(unittest.TestCase):
    def test_continuous_score_preserves_source_history_conventions(self):
        f = training.make_strands_features
        self.assertEqual(f("cat", None, 0, 6, .31), [1., 1/6, 0., .31, 0.])
        self.assertEqual(f("dog", "cat", 1, 6, .32, None), [1., 2/6, 0., .32, 1.])
        self.assertEqual(f("dog", "dog", 2, 6, .33, 1.), [0., .5, -1., .33, 1.])
        self.assertEqual(training.FEATURES[3], "strands_probability")

    def test_encoder_rejects_nonfinite_boolean_and_invalid_round_inputs(self):
        for score in [True, "0.5", float("nan"), float("inf"), -.01, 1.01]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                training.make_strands_features("x", None, 0, 6, score)
        for index, cap in [(True, 6), (0, True), (-1, 6), (6, 6)]:
            with self.assertRaises(ValueError):
                training.make_strands_features("x", None, index, cap, .5)

    def test_inference_rejects_wrong_shape_dtype_and_feature_ranges(self):
        head = training.StrandsHead((0.,) * 5, 0.)
        for values in [[1., .5], [True, .5, 0., .5, 0.],
                       [1., .5, 0., float("nan"), 0.], [1., 0., 0., .5, 0.],
                       [1., .5, 2., .5, 0.], [1., .5, 0., .5, .3]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                head.probability(values)

    def test_portable_roundtrip_and_tamper_detection(self):
        head = training.StrandsHead((1., 2., 3., 4., 5.), -.5, 0, {"trained": True})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.json"
            path.write_text(json.dumps(head.as_dict()), encoding="utf-8")
            loaded = training.StrandsHead.from_json(path)
            vector = [1., 1/6, 0., .25, 0.]
            self.assertEqual(head.probability(vector), loaded.probability(vector))
            modified = head.as_dict()
            modified["weights"][0] += .01
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaises(ValueError):
                training.StrandsHead.from_json(path)


class PolicyAndLearningControls(unittest.TestCase):
    def test_gold_mutation_does_not_change_stopping_actions(self):
        original = trajectory()
        changed = copy.deepcopy(original)
        for row in changed["rounds"]:
            row["f1"], row["em"] = 1 - row["f1"], 1 - row["em"]
        head = training.StrandsHead((0., 0., 0., 10., 0.), -5.)
        for policy in training.POLICIES:
            a = training.replay_one(original, policy, head)
            b = training.replay_one(changed, policy, head)
            self.assertEqual((a["rounds"], a["answer"], a["decisions"], a["llm_tokens"]),
                             (b["rounds"], b["answer"], b["decisions"], b["llm_tokens"]))

    def test_and_ablation_differs_from_head_and_caps_are_not_approvals(self):
        t = trajectory()
        for row in t["rounds"]:
            row["strands"] = .1
        head = training.StrandsHead((0.,) * 5, 100.)
        alone = training.replay_one(t, "retrained_head", head)
        paired = training.replay_one(t, "retrained_head_and", head)
        self.assertEqual(alone["rounds"], 1)
        self.assertEqual(paired["rounds"], 6)
        self.assertTrue(paired["forced_cap"])
        self.assertFalse(paired["decisions"][-1]["gate_stop"])
        self.assertIsNone(paired["decisions"][-1]["p_stop"])
        self.assertEqual(paired["head_calls"], 5)

    def test_source_reward_and_operational_cost_are_explicitly_distinct(self):
        t = trajectory()
        self.assertAlmostEqual(training.reward(t["rounds"], 0, .1), -.1/6)
        self.assertAlmostEqual(training.reward(t["rounds"], 5, .1), .9)
        head = training.StrandsHead((0.,) * 5, 100.)
        row = training.replay_one(t, "retrained_head", head)
        self.assertEqual(row["llm_tokens"], 10)
        self.assertEqual(row["reviewer_tokens"], 0)
        self.assertEqual(training.replay_one(t, "fixed6", head)["llm_tokens"], 180)

    def test_policy_gradient_uses_later_quality_signal_and_reloads(self):
        train = [trajectory("train", "train", rounds=2)]
        dev = [trajectory("dev", "dev", rounds=2)]
        config = training.TrainingConfig(epochs=1, group_size=64)
        torch.manual_seed(0)
        initial = torch.nn.Linear(5, 1)
        vector = training.trajectory_features(train[0]["rounds"])[0]
        before = torch.sigmoid(initial(torch.tensor(vector)).squeeze()).item()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "seed0"
            fitted = training._fit_seed(train, dev, 0, config, folder, {})
            loaded = training.StrandsHead.from_json(folder / "head.json")
            self.assertLess(fitted.probability(vector), before)
            self.assertEqual(fitted, loaded)
            trace = json.loads((folder / "training-trace.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(trace["group_has_signal"])
            self.assertGreater(trace["gradient_norm_before_clip"], 0)
            self.assertEqual(len(trace["sampled_actions"]), 64)

    def test_train_dev_overlap_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "bad"
            with self.assertRaises(ValueError):
                training._fit_seed([trajectory("same", "train")], [trajectory("same", "dev")],
                                   0, training.TrainingConfig(epochs=1), folder, {})
            self.assertFalse(folder.exists())


class FrozenStudyControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv"
        cls.trajectories = load_trajectories(cls.data)
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "study"
        cls.probed = False

        def check_all_frozen(items, directory):
            marker = json.loads((directory / "training-freeze.json").read_text(encoding="utf-8"))
            assert marker["seeds"] == list(training.SEEDS)
            assert marker["all_selected_checkpoints_frozen_before_test_scoring"]
            assert len(marker["checkpoints"]) == 5
            assert not (directory / "test80.json").exists()
            for entry in marker["checkpoints"]:
                path = directory / entry["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
                head = training.StrandsHead.from_json(path)
                assert len(head.metadata["train_ids"]) == 50
                assert len(head.metadata["dev_ids"]) == 20
                assert not {t["id"] for t in items if t["split"] == "test"}.intersection(
                    head.metadata["train_ids"] + head.metadata["dev_ids"])
            cls.probed = True
            return {"unique_test_questions": 80}

        with patch.object(training, "evaluate_frozen", side_effect=check_all_frozen):
            training.train_study(cls.data, cls.output, training.TrainingConfig(epochs=1))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_five_checkpoints_freeze_before_any_test_scoring(self):
        self.assertTrue(self.probed)

    def test_mutated_heldout_labels_cannot_change_fitted_checkpoints(self):
        changed = copy.deepcopy(self.trajectories)
        for t in changed:
            if t["split"] == "test":
                for row in t["rounds"]:
                    row["f1"], row["em"] = 1-row["f1"], 1-row["em"]
        output = Path(self.temp.name) / "mutated-test"
        with patch.object(training, "load_trajectories", return_value=changed), \
                patch.object(training, "evaluate_frozen", return_value={}):
            training.train_study(self.data, output, training.TrainingConfig(epochs=1))
        for seed in training.SEEDS:
            self.assertEqual((self.output / f"seed{seed}/head.json").read_bytes(),
                             (output / f"seed{seed}/head.json").read_bytes())

    def test_checkpoint_changed_after_freeze_is_rejected(self):
        folder = Path(self.temp.name) / "tampered"
        import shutil
        shutil.copytree(self.output, folder)
        head = folder / "seed0/head.json"
        head.write_text(head.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaises(ValueError):
            training.evaluate_frozen(self.trajectories, folder)

    def test_existing_study_output_is_never_overwritten(self):
        before = (self.output / "training-freeze.json").read_bytes()
        with self.assertRaises(FileExistsError):
            training.train_study(self.data, self.output)
        self.assertEqual(before, (self.output / "training-freeze.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
