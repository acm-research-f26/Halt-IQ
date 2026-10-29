"""Controls for inherited classifier fidelity, public-state isolation and replay."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from haltiq_pair.critic import Critic, INSTRUCTION
from haltiq_pair.linear import LinearHead, FEATURES, lexical_distance, make_features
from haltiq_pair.replay import analyze, load_trajectories, replay_one


def trajectory():
    return {"id": "q1", "split": "test", "rounds": [
        {"round": i + 1, "answer": answer, "split": "test", "strands": score,
         "f1": float(i > 0), "em": float(i > 0), "tokens": 30,
         "writer_tokens": 10, "critic_tokens": 20}
        for i, (answer, score) in enumerate(zip(["wrong", "right", "right", "right", "right", "right"], [.1, .9, .9, .9, .9, .9]))]}


class LinearFidelity(unittest.TestCase):
    def test_exact_sri_normalization_and_feature_history(self):
        self.assertEqual(lexical_distance("The A. Cat!", "cat"), 0)
        self.assertEqual(make_features("cat", None, 0, 6, False), [1, 1/6, 0, 0, 0])
        self.assertEqual(make_features("dog", "cat", 1, 6, True, 1), [1, 2/6, 0, 1, 1])
        self.assertEqual(make_features("dog", "dog", 2, 6, False, 1), [0, .5, -1, 0, 1])

    def test_all_real_pretrained_seeds_have_first_round_collapse(self):
        self.assertEqual(len(FEATURES), 5)
        for seed in range(5):
            head = LinearHead.pretrained(seed)
            self.assertEqual(len(head.checkpoint_sha256), 64)
            self.assertGreater(head.probability(make_features("wrong", None, 0, 6, False)), .5)
            self.assertTrue(head.as_dict()["trained"])

    def test_json_export_rejects_modified_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.json"
            data = LinearHead.pretrained().as_dict()
            path.write_text(json.dumps(data))
            self.assertEqual(LinearHead.from_json(path), LinearHead.pretrained())
            data["weights"][0] += 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                LinearHead.from_json(path)


class CriticIsolation(unittest.TestCase):
    def test_teammate_exact_query_evidence_and_http_format(self):
        captured = {}
        def ranker(query, passages):
            captured["query"] = query
            return [0, 3, 2]
        def transport(payload):
            captured["payload"] = payload
            return {"answers": {"sufficient": {"noul": .8123}}, "usage": {"input_tokens": 30, "output_tokens": 1}}
        critic = Critic(ranker=ranker, transport=transport)
        self.assertEqual(critic.score("Q?", ["A: zero", "B: one", "C: two"], "draft"), .8123)
        self.assertEqual(captured["query"], "Q? draft")
        self.assertEqual(captured["payload"], {"state": "Question: Q?\nEvidence: B: one\nC: two\nDraft answer: draft",
                         "questions": {"sufficient": {"type": "noul", "instructions": INSTRUCTION}}})
        self.assertEqual(critic.last_record["selected_indices"], [1, 2])
        self.assertEqual(critic.last_record["usage"]["input_tokens"], 30)

    def test_invalid_probability_fails_without_fallback(self):
        critic = Critic(ranker=lambda *_: [1], transport=lambda _: {"answers": {"sufficient": {"noul": float("nan")}}})
        with self.assertRaises(ValueError):
            critic.score("q", ["p"], "a")
        self.assertIsNone(critic.last_record)


class ReplayIntegrity(unittest.TestCase):
    def test_pair_really_uses_head_and_strands(self):
        t = trajectory()
        head = LinearHead.pretrained()
        self.assertEqual(replay_one(t, "linear_transfer", head)["rounds"], 1)
        paired = replay_one(t, "pair_and", head)
        self.assertEqual(paired["rounds"], 2)
        self.assertEqual(paired["llm_tokens"], 40)  # two writers and one reviewer
        rejecting = LinearHead((0, 0, 0, 0, 0), -10)
        self.assertEqual(replay_one(t, "pair_and", rejecting)["rounds"], 6)

    def test_gold_metrics_cannot_change_actions(self):
        t, changed = trajectory(), trajectory()
        for r in changed["rounds"]:
            r["f1"], r["em"] = 1 - r["f1"], 1 - r["em"]
        for policy in ("fixed1", "fixed6", "answer_unchanged", "strands_only", "linear_transfer", "pair_and"):
            a = replay_one(t, policy, LinearHead.pretrained())
            b = replay_one(changed, policy, LinearHead.pretrained())
            self.assertEqual((a["rounds"], a["decisions"], a["llm_tokens"]), (b["rounds"], b["decisions"], b["llm_tokens"]))

    def test_pinned_actual_teammate_csv(self):
        data = Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv"
        rows = load_trajectories(data)
        self.assertEqual(len(rows), 150)
        self.assertEqual(sum(t["split"] == "test" for t in rows), 80)
        # This checks the inherited negative result on real saved inputs, no new inference.
        for seed in range(5):
            head = LinearHead.pretrained(seed)
            for t in rows:
                self.assertEqual(replay_one(t, "linear_transfer", head)["rounds"], 1)
                self.assertEqual(replay_one(t, "pair_and", head)["rounds"], replay_one(t, "strands_only", head)["rounds"])


if __name__ == "__main__":
    unittest.main()
