"""Native Tau2/custom HotpotQA controls using scripted callbacks, never models."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from haltiq_tau_hotpot.scoring import answer_scores, score, support_scores

AVAILABLE = importlib.util.find_spec("tau2") is not None
if AVAILABLE:
    from tau2.data_model.message import ToolCall
    from haltiq_tau_hotpot.agent import run_episode
    from haltiq_tau_hotpot.domain import get_environment, make_task, validate_case, validate_task


def case():
    return {"public": {"id": "fixture-question", "question": "Who founded the Alpha company?",
        "type": "bridge", "level": "hard", "context": [
            ["Alpha", ["Alpha was founded by Alice.", "It opened in 2000."]],
            ["Alice", ["Alice founded Alpha."]]]},
        "evaluator": {"answer": "Alice", "supporting_facts": [["Alpha", 0], ["Alice", 0]]}}


def draft(answer="Alice"):
    return {"answer": answer, "supporting_facts": [["Alpha", 0], ["Alice", 0]]}


class MetricControls(unittest.TestCase):
    def test_strict_answer_aliases_and_boolean_special_cases_match_official_definitions(self):
        self.assertEqual(answer_scores(" The, ALICE! ", "Alice")["em"], 1)
        self.assertEqual(answer_scores("Bing Crosby", 'Harry Lillis "Bing" Crosby Jr.')["em"], 0)
        self.assertEqual(answer_scores("yes indeed", "yes")["f1"], 0)
        self.assertEqual(answer_scores("no", "not no")["f1"], 0)
        self.assertEqual(answer_scores("", "")["em"], 1)
        self.assertEqual(answer_scores("", "")["f1"], 0)

    def test_support_ids_are_exact_sets_and_duplicates_do_not_change_scores(self):
        expected = [["Title", 0], ["Other", 1]]
        self.assertEqual(support_scores(expected + [expected[0]], expected)["sp_f1"], 1)
        self.assertEqual(support_scores([["title", 0]], [["Title", 0]])["sp_f1"], 0)
        self.assertEqual(support_scores([["Title", 1]], [["Title", 0]])["sp_em"], 0)

    def test_joint_f1_uses_precision_recall_construction_not_f1_product(self):
        predicted = {"answer": "alpha", "supporting_facts": [["A", 0], ["B", 0]]}
        gold = {"answer": "alpha beta", "supporting_facts": [["A", 0]]}
        measured = score(predicted, gold)
        self.assertAlmostEqual(measured["f1"], 2 / 3)
        self.assertAlmostEqual(measured["sp_f1"], 2 / 3)
        self.assertEqual(measured["joint_f1"], .5)
        self.assertNotEqual(measured["joint_f1"], measured["f1"] * measured["sp_f1"])
        self.assertTrue(all(value == 0 for value in score(None, gold).values()))


@unittest.skipUnless(AVAILABLE, "requires the isolated pinned Tau2 runtime")
class NativeDomainControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from loguru import logger
        logger.disable("tau2")

    @classmethod
    def tearDownClass(cls):
        from loguru import logger
        logger.enable("tau2")

    def run_fixed(self, answer="Alice", **kwargs):
        return run_episode(case(), lambda state, feedback: draft(answer), arm="fixed1", max_drafts=1,
                           passage_indices=[1, 0], **kwargs)

    def test_native_search_read_and_submit_execute_before_agent_stop(self):
        result = self.run_fixed()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["submitted"], draft())
        self.assertEqual(result["native_reward"], 1)
        self.assertEqual(result["metrics"]["joint_f1"], 1)
        simulation = result["native_simulation"]
        self.assertEqual(simulation["termination_reason"], "agent_stop")
        messages = simulation["messages"]
        names = [call["name"] for m in messages for call in m.get("tool_calls") or []]
        self.assertEqual(names, ["search_passages", "read_passage", "read_passage", "submit_answer"])
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(json.loads(messages[-1]["content"]), {"submitted": True})
        json.dumps(result, allow_nan=False)

    def test_exact_two_passage_case_and_original_sentence_indices_are_preserved(self):
        observed = []
        run_episode(case(), lambda state, feedback: observed.append(state) or draft(),
                    arm="fixed1", max_drafts=1, passage_indices=[1, 0])
        evidence = observed[0]["evidence"]
        self.assertEqual([row["paragraph_index"] for row in evidence], [1, 0])
        self.assertEqual(evidence[1]["sentences"], [{"sentence_id": 0, "text": "Alpha was founded by Alice."},
                                                  {"sentence_id": 1, "text": "It opened in 2000."}])
        self.assertEqual(set(observed[0]), {"question", "evidence", "previous_draft"})

    def test_gold_and_hidden_evaluator_methods_are_not_public_tools_or_db(self):
        source = case()
        source["evaluator"]["answer"] = "PRIVATE_REFERENCE_CANARY"
        callbacks = []
        env = get_environment(public_case=source["public"], passage_indices=[0, 1])
        self.assertNotIn("PRIVATE_REFERENCE_CANARY", json.dumps(env.tools.db.model_dump()))
        self.assertEqual({tool.name for tool in env.get_tools()}, {"search_passages", "read_passage", "submit_answer"})
        response = env.get_response(ToolCall(id="bad", name="assert_answer_em",
                                            arguments={"expected_answer": "PRIVATE_REFERENCE_CANARY"}))
        self.assertTrue(response.error)
        run_episode(source, lambda state, feedback: callbacks.append(deepcopy(state)) or draft(),
                    arm="fixed1", max_drafts=1, passage_indices=[0, 1])
        self.assertNotIn("PRIVATE_REFERENCE_CANARY", json.dumps(callbacks))
        self.assertNotIn("supporting_facts", json.dumps(callbacks[0]))

    def test_evaluator_labels_do_not_change_writer_or_gate_inputs(self):
        observed = []
        for reference in ("Alice", "Bob"):
            source = case()
            source["evaluator"]["answer"] = reference
            seen = []
            def writer(state, feedback):
                seen.append(["writer", deepcopy(state), feedback])
                return draft()
            def gate(state, proposed, criterion):
                seen.append(["gate", deepcopy(state), deepcopy(proposed), criterion])
                return {"p_true": .9}
            result = run_episode(source, writer, gate=gate, arm="released", max_drafts=1,
                                 passage_indices=[0, 1])
            observed.append((seen, result["native_reward"]))
        self.assertEqual(observed[0][0], observed[1][0])
        self.assertEqual([row[1] for row in observed], [1, 0])

    def test_fixed_review_can_repair_answer_and_changes_real_native_metrics(self):
        first = self.run_fixed("Bob")
        seen = []
        def writer(state, feedback):
            seen.append((deepcopy(state), feedback))
            return draft("Alice" if feedback else "Bob")
        revised = run_episode(case(), writer, lambda state, current: "The public evidence names Alice.",
            arm="fixed3", max_drafts=3, passage_indices=[1, 0])
        self.assertEqual(first["metrics"]["em"], 0)
        self.assertEqual(revised["metrics"]["em"], 1)
        self.assertEqual(revised["counts"]["answer_drafts"], 3)
        self.assertEqual(revised["counts"]["reviewer_calls"], 2)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[1][0]["previous_draft"]["answer"], "Bob")
        self.assertTrue(seen[1][1])

    def test_all_rejected_stops_without_any_submission_or_fake_completion(self):
        result = run_episode(case(), lambda state, feedback: draft(), lambda state, current: "Check the evidence.",
            lambda state, current, criterion: {"p_true": .1}, arm="released", max_drafts=3,
            passage_indices=[0, 1])
        self.assertEqual(result["status"], "completed_with_failure")
        self.assertEqual(result["failure_category"], "draft_cap_no_accepted_submission")
        self.assertIsNone(result["submitted"])
        self.assertEqual(result["native_reward"], 0)
        self.assertEqual(result["counts"]["answer_drafts"], 3)
        self.assertEqual(result["counts"]["gate_calls"], 3)
        self.assertTrue(all(value == 0 for value in result["metrics"].values()))
        self.assertFalse(any(call["name"] == "submit_answer" for message in result["native_simulation"]["messages"]
                             for call in message.get("tool_calls") or []))

    def test_all_eight_declared_methods_are_real_different_control_routes_with_honest_ties(self):
        results = []
        for arm in ("fixed1", "fixed2", "fixed3", "fixed5", "fixed10", "released", "decomposed", "trained"):
            results.append(run_episode(case(), lambda state, feedback: draft(), lambda state, current: "Retain Alice.",
                lambda state, current, criterion: {"p_true": .9}, arm=arm, max_drafts=10, passage_indices=[0]))
        self.assertEqual([row["counts"]["answer_drafts"] for row in results], [1, 2, 3, 5, 10, 1, 1, 1])
        self.assertEqual([row["counts"]["gate_calls"] for row in results], [0, 0, 0, 0, 0, 1, 2, 1])
        self.assertTrue(all(row["metrics"]["em"] == 1 for row in results))

    def test_global_review_gate_tool_and_native_step_caps_preserve_failures(self):
        settings = [("reviewer_calls", 0), ("gate_calls", 0), ("tool_calls", 1), ("agent_steps", 1), ("tau_steps", 1)]
        for key, limit in settings:
            with self.subTest(cap=key):
                result = run_episode(case(), lambda state, feedback: draft(), lambda state, current: "Review.",
                    lambda state, current, criterion: {"p_true": .1}, arm="released", max_drafts=3,
                    passage_indices=[0, 1], caps={key: limit})
                self.assertIsNone(result["submitted"])
                self.assertEqual(result["native_reward"], 0)
                self.assertEqual(result["metrics"]["em"], 0)
                if key in result["counts"]:
                    self.assertLessEqual(result["counts"][key], limit + int(key == "agent_steps"))

    def test_invalid_public_citations_never_execute_even_when_gate_would_approve(self):
        gate_calls = []
        bad = draft()
        bad["supporting_facts"] = [["Alpha", 999]]
        result = run_episode(case(), lambda state, feedback: bad, gate=lambda *a: gate_calls.append(a) or {"p_true": 1},
                             arm="released", max_drafts=1, passage_indices=[0])
        self.assertEqual(result["failure_category"], "invalid_submission")
        self.assertIsNone(result["submitted"])
        self.assertEqual(gate_calls, [])

    def test_public_unknown_indices_and_empty_evaluator_criteria_fail_before_callbacks(self):
        source = case()
        source["public"]["answer"] = "must not leak"
        with self.assertRaises(ValueError):
            validate_case(source)
        public, gold = case()["public"], case()["evaluator"]
        task = make_task(public, gold)
        for change in (None, task.evaluation_criteria.model_copy(update={"env_assertions": []})):
            task.evaluation_criteria = change
            with self.assertRaises(ValueError):
                validate_task(task)
        with self.assertRaises(ValueError):
            run_episode(case(), lambda *args: self.fail("no callback"), passage_indices=[2])

    def test_provider_uncertainty_propagates_with_actual_partial_trace_and_no_retry(self):
        class UncertainCall(RuntimeError):
            pass
        calls = []
        def writer(state, feedback):
            calls.append(state)
            raise UncertainCall("uncertain model request")
        with self.assertRaises(UncertainCall) as failure:
            run_episode(case(), writer, arm="fixed1", max_drafts=1, passage_indices=[0])
        self.assertEqual(len(calls), 1)
        partial = failure.exception.haltiq_episode_partial
        self.assertEqual(partial["counts"]["answer_drafts"], 1)
        self.assertEqual(partial["trace"][-1]["status"], "pending")
        self.assertNotIn("native_simulation", partial)
        self.assertTrue(partial["native_partial_messages"])

    def test_known_callback_value_errors_are_native_zero_reward_failures_without_retry(self):
        for role in ("writer", "reviewer", "gate"):
            with self.subTest(role=role):
                calls = []
                def fail(*args):
                    calls.append(role)
                    raise ValueError("known schema or context failure")
                result = run_episode(case(), fail if role == "writer" else lambda state, feedback: draft(),
                    fail if role == "reviewer" else lambda state, current: "Review.",
                    fail if role == "gate" else lambda state, current, criterion: {"p_true": .1},
                    arm="released", max_drafts=3, passage_indices=[0])
                self.assertEqual(calls, [role])
                self.assertEqual(result["failure_category"], role + "_value_error")
                self.assertIsNone(result["submitted"])
                self.assertEqual(result["native_reward"], 0)
                self.assertEqual(result["native_simulation"]["termination_reason"], "agent_stop")

    def test_callback_mutation_cannot_corrupt_original_public_data_or_next_revision(self):
        source = case()
        original = deepcopy(source)
        seen = []
        def writer(state, feedback):
            seen.append(deepcopy(state))
            state["question"] = "corrupted"
            state["evidence"].clear()
            return draft()
        result = run_episode(source, writer, lambda state, current: "Retain the answer.",
                             arm="fixed2", max_drafts=2, passage_indices=[0, 1])
        self.assertEqual(source, original)
        self.assertEqual(seen[0]["question"], seen[1]["question"])
        self.assertTrue(seen[1]["evidence"])
        self.assertEqual(result["metrics"]["joint_em"], 1)

    def test_episode_wall_cap_stops_new_gate_review_or_submission_after_a_slow_callback(self):
        clock, gates, reviews = [0.0], [], []
        def writer(state, feedback):
            clock[0] = 2.0
            return draft()
        with patch("haltiq_tau_hotpot.agent.time.monotonic", side_effect=lambda: clock[0]):
            result = run_episode(case(), writer, lambda *args: reviews.append(args) or "Review.",
                lambda *args: gates.append(args) or {"p_true": .9}, arm="released", max_drafts=3,
                passage_indices=[0], timeout_seconds=1)
        self.assertEqual(result["failure_category"], "episode_wall_timeout")
        self.assertIsNone(result["submitted"])
        self.assertEqual(result["metrics"]["em"], 0)
        self.assertEqual(gates, [])
        self.assertEqual(reviews, [])


if __name__ == "__main__":
    unittest.main()
