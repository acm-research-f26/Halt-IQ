"""Behavioral safeguards for the critic experiment; no live model calls."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from haltiq.critics import (
    CHECKS,
    JevCritic,
    LLMCritic,
    LocalTypedCritic,
    QUESTIONS,
    Review,
    Writer,
    public_state,
    typed_review,
)
from haltiq.providers import CallResult
from haltiq.runner import run_experiment
from haltiq.tasks import Task, evaluate, load_tasks


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "diagnostic.jsonl"
GOLD_SENTINEL = "PRIVATE_GOLD_SHOULD_NEVER_REACH_A_MODEL"
DEMO_SENTINEL = "PRIVATE_SCRIPTED_DRAFT_SHOULD_NEVER_REACH_A_MODEL"


def call(data=None, *, inputs=0, outputs=0, cost=0.0, latency=0.0):
    return CallResult(
        data={} if data is None else data,
        input_tokens=inputs,
        output_tokens=outputs,
        latency_ms=latency,
        cost_usd=cost,
        model="unit-test-fake",
    )


def private_task():
    return Task(
        id="test-private",
        split="test",
        category="gold_isolation",
        question="Which room is assigned to the rover?",
        evidence=["The rover is assigned to Room Vale."],
        answers=[GOLD_SENTINEL],
        demo_drafts=[DEMO_SENTINEL] * 3,
        demo_scores=[{"supported": 0.12, "complete": 0.34, "relevant": 0.56}] * 3,
    )


def contaminated_state():
    task = private_task()
    return {**task.__dict__, "draft": "Room Vale"}


def no_progress(*args, **kwargs):
    pass


class EvaluationTests(unittest.TestCase):
    def test_aliases_case_and_punctuation_but_not_numeric_signs(self):
        self.assertTrue(evaluate("  THE building CEDAR! ", ["Cedar", "Building Cedar"])["exact_match"])
        self.assertFalse(evaluate("-3", ["3"])["exact_match"])
        self.assertFalse(evaluate("1.5", ["15"])["exact_match"])
        self.assertTrue(evaluate("15.", ["15"])["exact_match"])

    def test_partial_token_overlap_is_not_exact_correctness(self):
        result = evaluate("Dock West", ["Dock East"])
        self.assertFalse(result["exact_match"])
        self.assertAlmostEqual(result["token_f1"], 0.5)
        self.assertEqual(evaluate("", ["UNKNOWN"])["token_f1"], 0.0)

    def test_unicode_minus_preserves_negative_value(self):
        self.assertFalse(evaluate("\N{MINUS SIGN}3", ["3"])["exact_match"])
        self.assertTrue(evaluate("\N{MINUS SIGN}3", ["-3"])["exact_match"])
        self.assertFalse(evaluate("\N{MINUS SIGN}1.5", ["1.5"])["exact_match"])
        self.assertTrue(evaluate("\N{MINUS SIGN}1.5", ["-1.5"])["exact_match"])

    def test_token_f1_uses_multiset_overlap(self):
        # One matched occurrence out of two candidate tokens and one gold token.
        result = evaluate("red red", ["red"])
        self.assertFalse(result["exact_match"])
        self.assertAlmostEqual(result["token_f1"], 2 / 3)

    def test_diagnostic_fixture_has_intended_split_and_scripts(self):
        rows = load_tasks(DATASET, split="all")
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(row.split == "dev" for row in rows), 8)
        self.assertEqual(sum(row.split == "test" for row in rows), 16)
        self.assertEqual(len({row.id for row in rows}), len(rows))
        for row in rows:
            with self.subTest(task=row.id):
                self.assertEqual(len(row.demo_drafts), 3)
                self.assertEqual(len(row.demo_scores), 3)
                for scores in row.demo_scores:
                    typed_review(scores, 0.8, call())


class GoldIsolationTests(unittest.TestCase):
    def assert_public_payload(self, payload, *, draft=True):
        expected = {"question", "evidence", "draft"} if draft else {"question", "evidence"}
        self.assertEqual(set(payload), expected)
        serialized = json.dumps(payload)
        self.assertNotIn(GOLD_SENTINEL, serialized)
        self.assertNotIn(DEMO_SENTINEL, serialized)
        self.assertNotIn("demo_scores", serialized)
        self.assertNotIn("answers", payload)

    def test_task_and_critic_allowlists_remove_private_columns(self):
        self.assert_public_payload(private_task().state(), draft=False)
        self.assert_public_payload(private_task().state("Room Vale"))
        self.assert_public_payload(public_state(contaminated_state()))

    def test_writer_allows_only_public_state_and_explicit_revision_context(self):
        client = Mock()
        client.chat.return_value = call({"answer": "Room Vale"})
        writer = Writer(client)
        writer.write(contaminated_state())
        payload = json.loads(client.chat.call_args.args[0][-1]["content"])
        self.assert_public_payload(payload, draft=False)
        writer.write(contaminated_state(), previous="Room Ash", feedback="Check the assigned room.")
        payload = json.loads(client.chat.call_args.args[0][-1]["content"])
        self.assertEqual(set(payload), {"question", "evidence", "previous_answer", "revision_feedback"})
        self.assertEqual(payload["previous_answer"], "Room Ash")
        self.assertNotIn(GOLD_SENTINEL, json.dumps(client.chat.call_args.args))
        self.assertNotIn(DEMO_SENTINEL, json.dumps(client.chat.call_args.args))

    def test_each_real_critic_sends_only_public_state(self):
        for critic_type, data in (
            (LLMCritic, {"approved": True, "feedback": "Supported."}),
            (LocalTypedCritic, {key: 1.0 for key in CHECKS}),
        ):
            with self.subTest(critic=critic_type.__name__):
                client = Mock()
                client.chat.return_value = call(data)
                critic_type(client).review(contaminated_state())
                self.assert_public_payload(json.loads(client.chat.call_args.args[0][-1]["content"]))
                self.assertNotIn(GOLD_SENTINEL, json.dumps(client.chat.call_args.args))
        client = Mock()
        client.judge.return_value = call({"answers": {key: {"noul": 1.0} for key in CHECKS}})
        JevCritic(client).review(contaminated_state())
        payload, questions = client.judge.call_args.args
        self.assert_public_payload(payload)
        self.assertEqual(set(questions), {"supported", "complete", "relevant"})
        self.assertNotIn(GOLD_SENTINEL, json.dumps(questions))

    def test_runner_never_passes_gold_or_scripts_to_live_components(self):
        task = private_task()
        seen_writer, seen_critic = [], []

        class RecordingWriter:
            def write(self, state, previous=None, feedback=None):
                seen_writer.append(deepcopy(state))
                return "Room Vale", call({"answer": "Room Vale"})

        class RecordingCritic:
            def review(self, state):
                seen_critic.append(deepcopy(state))
                return Review(False, "Recheck the answer.", None, call())

        with tempfile.TemporaryDirectory() as tmp:
            run_experiment(
                [task], RecordingWriter(), {"fixed": None, "llm": RecordingCritic()},
                Path(tmp) / "run", config(), progress=no_progress,
            )
        self.assertTrue(seen_writer)
        self.assertTrue(seen_critic)
        for payload in seen_writer:
            self.assert_public_payload(payload, draft=False)
        for payload in seen_critic:
            self.assert_public_payload(payload)


class CriticValidationTests(unittest.TestCase):
    def test_typed_threshold_is_inclusive_and_requires_every_dimension(self):
        boundary = {key: 0.8 for key in CHECKS}
        self.assertTrue(typed_review(boundary, 0.8, call()).approved)
        for dimension in CHECKS:
            scores = {**boundary, dimension: 0.799999}
            with self.subTest(dimension=dimension):
                review = typed_review(scores, 0.8, call())
                self.assertFalse(review.approved)
                self.assertTrue(review.feedback.strip())

    def test_nonfinite_invalid_and_missing_scores_cannot_approve(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), -0.01, 1.01, True, "0.9", None):
            with self.subTest(score=repr(invalid)):
                with self.assertRaises(ValueError):
                    typed_review({"supported": invalid, "complete": 1, "relevant": 1}, 0.8, call())
        with self.assertRaises(ValueError):
            typed_review({"supported": 1, "complete": 1}, 0.8, call())
        for threshold in (float("nan"), float("inf"), -0.1, 0.0, 1.01):
            with self.subTest(threshold=repr(threshold)):
                with self.assertRaises(ValueError):
                    typed_review({key: 1 for key in CHECKS}, threshold, call())

    def test_jev_request_uses_named_typed_checks_and_preserves_continuous_scores(self):
        for key, question in QUESTIONS.items():
            with self.subTest(check=key):
                self.assertEqual(question["type"], "noul")
                self.assertEqual(set(question["criteria"]), {"true", "false"})
                self.assertTrue(question["instructions"])
        client = Mock()
        client.judge.return_value = call({"answers": {
            "supported": {"noul": 0.8}, "complete": {"noul": 0.92}, "relevant": {"noul": 1.0},
        }})
        review = JevCritic(client, threshold=0.8).review(contaminated_state())
        self.assertTrue(review.approved)
        self.assertEqual(review.scores, {"supported": 0.8, "complete": 0.92, "relevant": 1.0})

    def test_malformed_jev_answers_never_become_approval(self):
        invalid_responses = (
            {},
            {"answers": {}},
            {"answers": {key: {"value": 1} for key in CHECKS}},
            {"answers": {key: {"noul": True} for key in CHECKS}},
            {"answers": {key: {"noul": float("nan")} for key in CHECKS}},
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                client = Mock()
                client.judge.return_value = call(response)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    JevCritic(client).review(contaminated_state())

    def test_conventional_critic_requires_boolean_and_rejection_feedback(self):
        for response in ({"approved": "false", "feedback": "Fix it."},
                         {"approved": False, "feedback": "  "},
                         {"approved": True, "feedback": None}):
            with self.subTest(response=response):
                client = Mock()
                client.chat.return_value = call(response)
                with self.assertRaises(ValueError):
                    LLMCritic(client).review(contaminated_state())


def config(**overrides):
    return {"dataset": str(DATASET), "max_rounds": 3, "threshold": 0.8, **overrides}


class RunnerTests(unittest.TestCase):
    def run_in_temp(self, tasks, writer, critics, *, demo=False, **overrides):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output = Path(tmp.name) / "run"
        result = run_experiment(tasks, writer, critics, output, config(**overrides),
                                demo=demo, progress=no_progress)
        summary = json.loads((output / "summary.json").read_text())
        traces = [json.loads(line) for line in (output / "traces.jsonl").read_text().splitlines()]
        return result, summary, traces

    def test_demo_fixed_cap_early_stop_and_false_approval(self):
        tasks = [task for task in load_tasks(DATASET) if task.id in {"dev-002", "dev-003"}]
        result, summary, _ = self.run_in_temp(tasks, None, {"fixed": None, "mock-typed": None}, demo=True)
        episodes = {(ep["task_id"], ep["arm"]): ep for ep in result["episodes"]}
        for task in tasks:
            fixed = episodes[task.id, "fixed"]
            self.assertEqual(fixed["rounds"], 3)
            self.assertEqual(fixed["judge_calls"], 0)
            self.assertEqual(fixed["stop_reason"], "round_cap")
            critic = episodes[task.id, "mock-typed"]
            self.assertEqual(critic["rounds"], 1)
            self.assertEqual(critic["judge_calls"], 1)
            self.assertEqual(critic["typed_questions"], 3)
        false_approval = episodes["dev-003", "mock-typed"]
        self.assertEqual(false_approval["stop_reason"], "critic_approved")
        self.assertFalse(false_approval["success"])
        self.assertFalse(false_approval["exact_match"])
        self.assertEqual(summary["arms"]["mock-typed"]["false_approvals"], 1)
        self.assertEqual(summary["arms"]["mock-typed"]["false_approval_rate"], 0.5)
        self.assertEqual(summary["actual_experiment"]["successful_model_calls"], 0)

    def test_shared_first_draft_is_one_physical_call_but_one_per_arm_operationally(self):
        task = load_tasks(DATASET)[0]
        writer = Mock()
        writer.write.return_value = ("Niko", call({"answer": "Niko"}, inputs=10, outputs=2, cost=0.01))
        critic = Mock()
        critic.review.return_value = Review(True, "Approved.", None,
                                             call({"approved": True}, inputs=3, outputs=1, cost=0.002))
        result, summary, traces = self.run_in_temp([task], writer,
                                                  {"fixed": None, "llm": critic}, max_rounds=1)
        self.assertEqual(writer.write.call_count, 1)
        self.assertEqual(critic.review.call_count, 1)
        episodes = {ep["arm"]: ep for ep in result["episodes"]}
        self.assertEqual({ep["final_answer"] for ep in episodes.values()}, {"Niko"})
        self.assertEqual([ep["writer_calls"] for ep in episodes.values()], [1, 1])
        self.assertEqual(episodes["fixed"]["input_tokens"], 10)
        self.assertEqual(episodes["llm"]["input_tokens"], 13)
        physical = summary["actual_experiment"]
        self.assertEqual(physical["successful_model_calls"], 2)
        self.assertEqual(physical["input_tokens"], 13)
        self.assertEqual(physical["output_tokens"], 3)
        self.assertAlmostEqual(physical["api_cost_usd"], 0.012)
        self.assertAlmostEqual(sum(ep["cost_usd"] for ep in episodes.values()), 0.022)
        self.assertEqual(sum(event["event"] == "shared_writer" for event in traces), 1)

    def test_rejected_draft_at_cap_is_reviewed_and_not_silently_approved(self):
        task = load_tasks(DATASET)[0]
        writer = Mock()
        writer.write.return_value = ("Rami", call({"answer": "Rami"}))
        critic = Mock()
        critic.review.return_value = Review(False, "Check the supervisor.", None, call())
        result, _, _ = self.run_in_temp([task], writer, {"llm": critic}, max_rounds=2)
        ep = result["episodes"][0]
        self.assertEqual(ep["rounds"], 2)
        self.assertEqual(ep["judge_calls"], 2)
        self.assertEqual(ep["stop_reason"], "round_cap")
        self.assertFalse(ep["success"])
        self.assertFalse(ep["round_trace"][-1]["approved"])

    def test_critic_failure_stays_in_results_even_when_draft_is_correct(self):
        task = load_tasks(DATASET)[0]
        writer = Mock()
        writer.write.return_value = ("Niko", call({"answer": "Niko"}))
        critic = Mock()
        critic.review.side_effect = RuntimeError("Provider unavailable")
        result, summary, traces = self.run_in_temp([task], writer, {"llm": critic})
        ep = result["episodes"][0]
        self.assertEqual(ep["status"], "failed")
        self.assertEqual(ep["stop_reason"], "provider_error")
        self.assertTrue(ep["exact_match"])
        self.assertFalse(ep["success"])
        self.assertEqual(ep["judge_calls"], 1)
        self.assertEqual(ep["judge_successful_calls"], 0)
        self.assertEqual(summary["arms"]["llm"]["episodes"], 1)
        self.assertEqual(summary["arms"]["llm"]["failures"], 1)
        self.assertEqual(summary["arms"]["llm"]["success_rate"], 0)
        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertTrue(any(event["event"] == "episode" and event["status"] == "failed" for event in traces))

    def test_shared_writer_failure_retains_all_arms_and_continues_next_item(self):
        tasks = load_tasks(DATASET)[:2]
        writer = Mock()
        writer.write.side_effect = [RuntimeError("Model unavailable"), ("Ben", call({"answer": "Ben"}))]
        critic = Mock()
        critic.review.return_value = Review(True, "Approved.", None, call())
        result, summary, _ = self.run_in_temp(tasks, writer,
                                             {"fixed": None, "llm": critic}, max_rounds=1)
        self.assertEqual(len(result["episodes"]), 4)
        first = [ep for ep in result["episodes"] if ep["task_id"] == tasks[0].id]
        self.assertEqual(len(first), 2)
        self.assertTrue(all(ep["status"] == "failed" and not ep["success"] for ep in first))
        second = [ep for ep in result["episodes"] if ep["task_id"] == tasks[1].id]
        self.assertTrue(all(ep["status"] == "completed" and ep["success"] for ep in second))
        self.assertEqual(writer.write.call_count, 2)
        for arm in ("fixed", "llm"):
            self.assertEqual(summary["arms"][arm]["episodes"], 2)
            self.assertEqual(summary["arms"][arm]["failures"], 1)
            self.assertEqual(summary["arms"][arm]["success_rate"], 0.5)

    def test_invalid_writer_output_retains_completed_response_usage(self):
        task = load_tasks(DATASET)[0]
        invalid = call({"answer": "  "}, inputs=17, outputs=5, cost=0.03)

        # A failed shared first draft is a single physical request, even though
        # both arms retain the failed episode and its standalone usage.
        client = Mock()
        client.chat.return_value = invalid
        unused_critic = Mock()
        result, summary, traces = self.run_in_temp(
            [task], Writer(client), {"fixed": None, "llm": unused_critic}, max_rounds=1,
        )
        self.assertEqual(client.chat.call_count, 1)
        unused_critic.review.assert_not_called()
        physical = summary["actual_experiment"]
        self.assertEqual(physical["successful_model_calls"], 1)
        self.assertEqual(physical["input_tokens"], 17)
        self.assertEqual(physical["output_tokens"], 5)
        self.assertAlmostEqual(physical["api_cost_usd"], 0.03)
        for ep in result["episodes"]:
            self.assertEqual(ep["status"], "failed")
            self.assertFalse(ep["success"])
            self.assertEqual(ep["input_tokens"], 17)
            self.assertEqual(ep["output_tokens"], 5)
            self.assertAlmostEqual(ep["cost_usd"], 0.03)
        shared_errors = [event for event in traces if event["event"] == "error"]
        self.assertEqual(shared_errors[0]["call"]["input_tokens"], 17)

        # A malformed later revision also costs a request; the valid first
        # draft must remain inspectable without becoming a successful episode.
        client = Mock()
        client.chat.side_effect = [call({"answer": "Niko"}, inputs=10, outputs=2, cost=0.01), invalid]
        result, summary, traces = self.run_in_temp(
            [task], Writer(client), {"fixed": None}, max_rounds=2,
        )
        ep = result["episodes"][0]
        self.assertEqual(ep["status"], "failed")
        self.assertEqual(ep["writer_calls"], 2)
        self.assertEqual(ep["final_answer"], "Niko")
        self.assertTrue(ep["exact_match"])
        self.assertFalse(ep["success"])
        self.assertEqual(ep["input_tokens"], 27)
        self.assertEqual(ep["output_tokens"], 7)
        self.assertAlmostEqual(ep["cost_usd"], 0.04)
        self.assertEqual(summary["actual_experiment"]["successful_model_calls"], 2)
        self.assertAlmostEqual(summary["actual_experiment"]["api_cost_usd"], 0.04)
        self.assertTrue(any(event["event"] == "invalid_output" and event["role"] == "writer"
                            for event in traces))

    def test_invalid_critic_output_retains_usage_and_cannot_make_episode_successful(self):
        task = load_tasks(DATASET)[0]
        for arm, critic_type, invalid_data in (
            ("llm", LLMCritic, {"approved": "yes", "feedback": "Looks correct."}),
            ("local-typed", LocalTypedCritic, {"supported": True, "complete": 1, "relevant": 1}),
        ):
            with self.subTest(arm=arm):
                writer_client = Mock()
                writer_client.chat.return_value = call({"answer": "Niko"}, inputs=10, outputs=2, cost=0.01)
                critic_client = Mock()
                critic_client.chat.return_value = call(invalid_data, inputs=19, outputs=7, cost=0.04, latency=42)
                result, summary, traces = self.run_in_temp(
                    [task], Writer(writer_client), {arm: critic_type(critic_client)}, max_rounds=1,
                )
                ep = result["episodes"][0]
                self.assertEqual(ep["status"], "failed")
                self.assertEqual(ep["stop_reason"], "provider_error")
                self.assertTrue(ep["exact_match"])
                self.assertFalse(ep["success"])
                self.assertEqual(ep["judge_calls"], 1)
                self.assertEqual(ep["judge_successful_calls"], 1)
                self.assertEqual(ep["judge_latency_ms"], 42)
                self.assertEqual(ep["input_tokens"], 29)
                self.assertEqual(ep["output_tokens"], 9)
                self.assertAlmostEqual(ep["cost_usd"], 0.05)
                self.assertEqual(summary["actual_experiment"]["successful_model_calls"], 2)
                self.assertEqual(summary["actual_experiment"]["input_tokens"], 29)
                self.assertAlmostEqual(summary["actual_experiment"]["api_cost_usd"], 0.05)
                self.assertEqual(summary["arms"][arm]["completion_rate"], 0)
                self.assertIsNone(summary["arms"][arm]["completed_accuracy"])
                invalid_events = [event for event in traces if event["event"] == "invalid_output"]
                self.assertEqual(len(invalid_events), 1)
                self.assertEqual(invalid_events[0]["role"], "critic")
                self.assertEqual(invalid_events[0]["call"]["input_tokens"], 19)


if __name__ == "__main__":
    unittest.main()
