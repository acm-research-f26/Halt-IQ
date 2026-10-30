"""Gold boundaries, prefix selection and actual-call controls for fresh tests."""
import copy
import inspect
import json
import unittest
from unittest.mock import patch

from haltiq_pair.aligned_gate import Gate
from haltiq_pair.aligned_hotpot import (bounded_call, execute_episode, replay_prefix,
                                       run, select_seen, summarize, token_totals, validate_fresh_inputs)
from haltiq_pair import hotpot
from haltiq_pair.hotpot import PublicQuestion
from haltiq_pair.replay import load_trajectories
from haltiq_pair.team_training import StrandsHead
from pathlib import Path


class Head:
    provenance = {"schema": "synthetic_control_only", "seed": 0}

    def __init__(self, stop_round=2):
        self.stop_round = stop_round
        self.vectors = []

    def probability(self, vector):
        self.vectors.append(list(vector))
        return float(vector[1] >= self.stop_round / 6)


class Critic:
    provenance = {"fixture": True}

    def __init__(self, scores, fail=False):
        self.scores = scores
        self.calls = 0
        self.fail = fail
        self.last_record = None

    def score(self, question, passages, answer):
        self.calls += 1
        if self.fail:
            raise ValueError("synthetic verifier failure")
        score = self.scores[answer]
        self.last_record = {"p_true": score, "physical_critic_calls": 1}
        return score


class Agents:
    def __init__(self, drafts, reviews=None):
        self.drafts = iter(drafts)
        self.reviews = iter(reviews) if reviews is not None else None
        self.usage = []
        self.prompts = []

    def generate(self, role, system, prompt):
        self.prompts.append((role, system, prompt))
        self.usage.append({"role": role, "status": "completed", "usage": {
            "inputTokens": 7, "outputTokens": 3, "totalTokens": 10}})
        value = next(self.drafts) if role == "writer" else (
            next(self.reviews) if self.reviews is not None else {"approved": False, "feedback": "Check the evidence."})
        if isinstance(value, Exception):
            self.usage[-1] = {"role": role, "status": "failed"}
            raise value
        if isinstance(value, str):
            return value
        return json.dumps(value)


def draft(answer):
    return {"answer": answer, "supporting_facts": [["Public", 0]]}


PUBLIC = PublicQuestion("synthetic", "Which answer follows from the passage?",
                        (("Public", ("The answer is in this public sentence.",)),))


class AlignedGateControls(unittest.TestCase):
    def test_continuous_feature_is_not_reviewer_boolean_or_extra_and(self):
        head = Head(stop_round=1)
        critic = Critic({"A": .23})
        gate = Gate(critic=critic, head=head)
        one = gate.decide("retrained", "A", None, 1, 6, "q", ["p"], critic_approved=True)
        both = gate.decide("retrained_AND", "A", None, 1, 6, "q", ["p"], critic_approved=True)
        self.assertEqual(head.vectors[0], [1., 1/6, 0., .23, 0.])
        self.assertTrue(one["stop"])
        self.assertFalse(both["stop"])
        self.assertTrue(gate.provenance["new_training"])
        self.assertFalse(gate.provenance["language_model_weights_trained"])

    def test_final_cap_never_becomes_a_learned_approval(self):
        head, critic = Head(1), Critic({"A": .99})
        gate = Gate(critic=critic, head=head)
        result = gate.decide("retrained", "A", "B", 6, 6, "q", ["p"])
        self.assertFalse(result["stop"])
        self.assertIsNone(result["p_stop"])
        self.assertEqual(result["physical_head_calls"], 0)
        self.assertEqual(head.vectors, [])

    def test_baseline_and_repeat_do_not_call_verifier_or_head(self):
        head, critic = Head(), Critic({}, fail=True)
        gate = Gate(critic=critic, head=head)
        self.assertTrue(gate.decide("baseline", "A", None, 1, 6, "q", ["p"])["stop"])
        self.assertTrue(gate.decide("repeat", " A ", "a", 2, 6, "q", ["p"])["stop"])
        self.assertEqual((critic.calls, head.vectors), (0, []))


class FreshCohortControls(unittest.TestCase):
    def setUp(self):
        items = load_trajectories(Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv")
        self.metadata = {"trained": True, **{split + "_ids": sorted(t["id"] for t in items if t["split"] == split)
                                            for split in ("train", "dev")}}
        self.test_id = next(t["id"] for t in items if t["split"] == "test")
        self.head = StrandsHead((0.,)*5, 0., seed=0, metadata=self.metadata)
        self.cohort = {"dataset": hotpot.DATASET_ID, "revision": hotpot.DATASET_REVISION,
                       "limit": 1, "question_ids": ["prospective-only"], "primary_live_head_sha256": "frozen"}

    def test_fresh_identity_and_predeclared_head_are_enforced_without_model_calls(self):
        validate_fresh_inputs(self.head, self.cohort, "frozen")
        for qid in (self.metadata["train_ids"][0], self.metadata["dev_ids"][0], self.test_id):
            for spelling in {qid, hotpot._bare_id(qid)}:
                with self.assertRaises(ValueError):
                    validate_fresh_inputs(self.head, dict(self.cohort, question_ids=[spelling]), "frozen")
        with self.assertRaises(ValueError):
            validate_fresh_inputs(self.head, self.cohort, "different")

    def test_fake_fit_metadata_and_duplicate_policy_cannot_start_runtime(self):
        wrong = StrandsHead((0.,)*5, 0., seed=0,
                            metadata=dict(self.metadata, train_ids=["fake"]*50))
        with self.assertRaises(ValueError):
            validate_fresh_inputs(wrong, self.cohort, "frozen")
        with patch("haltiq_pair.aligned_hotpot.Runtime") as runtime:
            with self.assertRaises(ValueError):
                run("unused", "unused", "unused", mode="online", policies=("fixed1", "fixed1"))
            runtime.assert_not_called()


class FormatAndGoldBoundaryControls(unittest.TestCase):
    def test_exactly_one_format_correction_retains_original_and_all_usage(self):
        agents, events = Agents(["not JSON", draft("A")]), []
        value = bounded_call(agents, "writer", "public system", "public question",
                             json.loads, events.append)
        self.assertEqual(value["answer"], "A")
        self.assertEqual(len(agents.usage), 2)
        self.assertEqual(events[0]["raw"], "not JSON")
        self.assertEqual(events[1]["event"], "format_failure")
        self.assertIn("not JSON", agents.prompts[1][2])

    def test_provider_failure_is_not_retried_as_formatting(self):
        agents = Agents([RuntimeError("offline transport failure"), draft("A")])
        with self.assertRaises(RuntimeError):
            bounded_call(agents, "writer", "s", "p", json.loads, lambda event: None)
        self.assertEqual(len(agents.usage), 1)
        self.assertIsNone(token_totals(agents.usage)["total_tokens"])

    def test_gold_is_not_an_argument_and_early_stop_does_not_call_reviewer(self):
        self.assertNotIn("gold", inspect.signature(execute_episode).parameters)
        agents = Agents([draft("A")])
        gate = Gate(critic=Critic({"A": .9}), head=Head(1))
        row = execute_episode(agents, gate, PUBLIC, "retrained")
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["generated_rounds"], 1)
        self.assertEqual([x[0] for x in agents.prompts], ["writer"])
        self.assertEqual(row["reported_agent_tokens"]["total_tokens"], 10)


class PrefixAndFailureControls(unittest.TestCase):
    def setUp(self):
        self.answers = ["A", "B", "C", "D", "E", "F"]
        self.scores = dict(zip(self.answers, [.9, .8, .1, .2, .99, .3]))
        self.agents = Agents([draft(a) for a in self.answers])
        self.critic, self.head = Critic(self.scores), Head(2)
        self.gate = Gate(critic=self.critic, head=self.head)
        self.collection = execute_episode(self.agents, self.gate, PUBLIC, "fixed6", collect=True)
        for step in self.collection["steps"]:
            step["metrics"] = {"f1": .5, "em": 0.}

    def test_collection_pays_six_writers_five_reviews_six_verifiers(self):
        self.assertEqual(self.collection["status"], "completed")
        self.assertEqual(self.collection["generated_rounds"], 6)
        self.assertEqual(len(self.agents.usage), 11)
        self.assertEqual(self.critic.calls, 6)
        self.assertEqual(self.collection["reported_agent_tokens"]["total_tokens"], 110)

    def test_prefix_selection_cannot_see_better_future_draft(self):
        row = replay_prefix(self.collection, "retrained_best", head=Head(2))
        self.assertEqual((row["generated_rounds"], row["selected_round"]), (2, 1))
        self.assertEqual(row["prediction"]["answer"], "A")
        self.assertEqual(row["writer_reviewer_tokens_estimated"]["total_tokens"], 30)
        full = replay_prefix(self.collection, "fixed6_best", head=Head(2))
        self.assertEqual((full["generated_rounds"], full["selected_round"]), (6, 5))
        self.assertEqual(full["writer_reviewer_tokens_estimated"]["total_tokens"], 110)

    def test_selector_ignores_gold_metrics_and_ties_choose_earlier(self):
        seen = copy.deepcopy(self.collection["steps"][:2])
        seen[0]["decision"]["p_approve"] = .8
        seen[1]["decision"]["p_approve"] = .8
        seen[0]["metrics"]["f1"], seen[1]["metrics"]["f1"] = 0., 1.
        self.assertEqual(select_seen(seen, "best_seen")["round"], 1)
        seen[0]["metrics"]["f1"], seen[1]["metrics"]["f1"] = 1., 0.
        self.assertEqual(select_seen(seen, "best_seen")["round"], 1)

    def test_verifier_failure_keeps_valid_draft_for_verifier_free_policy(self):
        agents = Agents([draft("A")])
        gate = Gate(critic=Critic({}, fail=True), head=Head())
        collection = execute_episode(agents, gate, PUBLIC, "fixed6", collect=True)
        self.assertEqual(collection["status"], "failed")
        self.assertEqual(collection["critic_call_attempts"], 1)
        self.assertEqual(replay_prefix(collection, "fixed1", head=Head())["status"], "completed")
        self.assertEqual(replay_prefix(collection, "strands", head=Head())["status"], "failed")

    def test_missing_future_is_failed_without_silent_last_draft_fallback(self):
        agents = Agents([draft("A"), "bad", "still bad"])
        gate = Gate(critic=Critic({"A": .1}), head=Head())
        collection = execute_episode(agents, gate, PUBLIC, "fixed6", collect=True)
        self.assertEqual(collection["status"], "failed")
        first = replay_prefix(collection, "fixed1", head=Head())
        final = replay_prefix(collection, "fixed6", head=Head())
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["writer_reviewer_tokens_estimated"]["total_tokens"], 10)
        self.assertEqual(final["status"], "failed")
        self.assertIsNone(final["prediction"])
        self.assertEqual(final["writer_reviewer_tokens_estimated"]["total_tokens"], 40)

    def test_failures_stay_zero_in_planned_denominator(self):
        rows = [{"question_id": "a", "policy": "retrained", "status": "completed",
                 "metrics": {"f1": .8}, "generated_rounds": 1, "forced_cap_return": False},
                {"question_id": "b", "policy": "retrained", "status": "failed",
                 "metrics": {}, "generated_rounds": 1, "forced_cap_return": False}]
        result = summarize(rows, ["retrained"], 2)["policies"]["retrained"]
        self.assertEqual((result["valid"], result["failed"], result["metrics"]["f1"]), (1, 1, .4))
        self.assertIsNone(result["mean_writer_reviewer_tokens"])


if __name__ == "__main__":
    unittest.main()
