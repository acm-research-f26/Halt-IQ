"""Action-policy isolation/learning controls, using synthetic CPU data only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from haltiq_pair import nov2_policy as policy
from haltiq_pair.nov2_orchestration import ACTIONS, State
from haltiq_pair.hotpot import PublicQuestion


def records(prefix, split):
    result = []
    for i, action in enumerate(ACTIONS):
        features = [0.] * 11
        features[i] = 1.
        result.append({"question_id": prefix + str(i), "state_id": "initial", "split": split,
            "features": features, "available_actions": list(ACTIONS),
            "utilities": {candidate: float(candidate == action) for candidate in ACTIONS}})
    return result


def public_state():
    return State(PublicQuestion("id", "Which city?", (("Evidence", ("Paris",)),)),
        {"answer": "Paris", "supporting_facts": [["Evidence", 0]]},
        {"approved": False, "feedback": "Missing evidence; check conflicting passage."}, .6, 1, 3,
        usage=[{"role": "writer"}, {"role": "reviewer"}])


class PolicyControls(unittest.TestCase):
    def test_cpu_fit_learns_three_actions_and_freezes_all_five(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fit"
            summary = policy.fit_policy(records("train", "train"), records("dev", "dev"), output,
                                        policy.PolicyConfig(epochs=60, learning_rate=.1))
            self.assertEqual(len(summary["checkpoints"]), 5)
            self.assertEqual(summary["primary_head_path"], str(output / "seed0/head.json"))
            self.assertTrue((output / "training-freeze.json").exists())
            self.assertFalse(any("test" in path.name for path in output.rglob("*") if path.is_file()))
            for entry in summary["checkpoints"]:
                path = output / entry["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
                head = policy.LinearPolicy.from_json(path)
                self.assertEqual([head._decision(r["features"], ACTIONS)["action"]
                                  for r in records("unused", "dev")], list(ACTIONS))
                history = json.loads((path.parent / "history.json").read_text(encoding="utf-8"))
                self.assertEqual(history["optimizer_steps_completed"], 60)
                self.assertGreater(history["epochs"][-1]["parameter_l2_from_initial"], .1)
                self.assertTrue(all(r["optimizer_parameter_step_l2"] > 0 for r in history["epochs"]))

    def test_missing_counterfactual_overlap_and_test_fields_rejected_before_output(self):
        train, dev = records("train", "train"), records("dev", "dev")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bad"
            changed = copy.deepcopy(train)
            del changed[0]["utilities"]["INVOKE"]
            with self.assertRaises(ValueError):
                policy.fit_policy(changed, dev, output)
            changed = copy.deepcopy(dev)
            changed[0]["question_id"] = train[0]["question_id"]
            with self.assertRaises(ValueError):
                policy.fit_policy(train, changed, output)
            changed = copy.deepcopy(train)
            changed[0]["reference_answer"] = "private"
            with self.assertRaises(ValueError):
                policy.fit_policy(changed, dev, output)
            changed = copy.deepcopy(dev)
            changed[0]["split"] = "test"
            with self.assertRaises(ValueError):
                policy.fit_policy(train, changed, output)
            self.assertFalse(output.exists())

    def test_dev_utilities_can_select_but_cannot_change_optimizer_updates(self):
        training, dev = records("train", "train"), records("dev", "dev")
        changed = copy.deepcopy(dev)
        for row in changed:
            row["utilities"] = {action: -value for action, value in row["utilities"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            policy.fit_policy(training, dev, a, policy.PolicyConfig(epochs=4))
            policy.fit_policy(training, changed, b, policy.PolicyConfig(epochs=4))
            for seed in policy.SEEDS:
                aa = json.loads((a / f"seed{seed}/history.json").read_text(encoding="utf-8"))
                bb = json.loads((b / f"seed{seed}/history.json").read_text(encoding="utf-8"))
                self.assertEqual([row["weights"] for row in aa["epochs"]], [row["weights"] for row in bb["epochs"]])
                self.assertEqual([row["bias"] for row in aa["epochs"]], [row["bias"] for row in bb["epochs"]])

    def test_question_weighting_handles_unequal_state_counts(self):
        items = policy._records(records("train", "train"), "train")
        extra = copy.deepcopy(items[0])
        extra["state_id"] = "second"
        _, _, _, weights = policy._partition(items + [extra])
        self.assertAlmostEqual(float(weights[0] + weights[-1]), 1/3, places=6)
        self.assertAlmostEqual(float(weights[1]), 1/3, places=6)

    def test_public_features_and_masked_ties_do_not_use_identity_or_gold(self):
        state = public_state()
        head = policy.LinearPolicy(tuple((0.,) * 11 for _ in ACTIONS), (0., 0., 0.), 0,
                                   {"objective": policy.OBJECTIVE})
        decision = head.decide(state)
        self.assertEqual(decision["action"], "STOP")
        self.assertEqual(len(decision["features"]), 11)
        self.assertEqual(decision["features"][7:9], [1., 1.])
        changed = state.to_dict()
        changed["public"]["id"] = "different-unseen-question"
        self.assertEqual(head.decide(State.from_dict(changed)), decision)
        self.assertEqual(head._decision([0.] * 11, ("STOP",))["probabilities"],
                         {"STOP": 1., "REVISE": 0., "INVOKE": 0.})
        state.review["approved"] = "private-label"
        with self.assertRaises(ValueError):
            head.decide(state)

    def test_tampering_and_nonfinite_features_rejected(self):
        head = policy.LinearPolicy(tuple((0.,) * 11 for _ in ACTIONS), (0., 0., 0.), 0,
                                   {"objective": policy.OBJECTIVE})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.json"
            path.write_text(json.dumps(head.as_dict()), encoding="utf-8")
            self.assertEqual(policy.LinearPolicy.from_json(path), head)
            changed = head.as_dict()
            changed["bias"][0] = .1
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                policy.LinearPolicy.from_json(path)
        with self.assertRaises(ValueError):
            head._decision([float("nan")] * 11, ACTIONS)
        with self.assertRaises(ValueError):
            head._decision([0.] * 11, ("STOP", "INVOKE", "REVISE"))

    def test_utility_ties_stop_and_missing_action_is_not_imputed(self):
        row = records("train", "train")[0]
        row["utilities"] = {action: 1. for action in ACTIONS}
        self.assertEqual(policy._records([row], "train")[0]["target"], "STOP")
        row["available_actions"] = ["STOP", "REVISE"]
        with self.assertRaises(ValueError):
            policy._records([row], "train")


if __name__ == "__main__":
    unittest.main()
