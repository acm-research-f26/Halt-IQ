"""Small controlled tests for intervention, leakage, accounting and strict IO."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from haltiq_tau.core import Action, Controller, GateScore, PublicState, RejectedAction, RouteFailure, fingerprint, gate_view, json_copy, summarize_planned, validate_gate_view
from haltiq_tau.providers import CommandGate, NativeStrandsGate, native_strands_backend
from haltiq_tau.__main__ import validate_pilot


POLICY = "Only set status after the user requests it; do not delete tasks."
TOOLS = [{"name": "update_task_status", "description": "Set status", "parameters": {
    "type": "object", "properties": {"status": {"enum": ["pending", "completed"]}}}}]
MESSAGES = [{"role": "user", "content": "Mark task_1 completed."}]


def action(status="completed"):
    return Action(tool_calls=({"id": "call-1", "name": "update_task_status",
                              "arguments": {"task_id": "task_1", "status": status}},))


class SequenceGate:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.states = []

    def score(self, state):
        self.states.append(state.to_dict())
        return GateScore(next(self.probabilities), "scripted-test-gate")


class PublicContractTests(unittest.TestCase):
    def test_private_gold_and_metadata_are_rejected(self):
        public = PublicState(POLICY, tuple(MESSAGES), tuple(TOOLS), action()).to_dict()
        for key in ("task", "evaluation_criteria", "expected_actions", "gold_db", "reward", "user_scenario", "task_id"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                PublicState.from_dict({**public, key: "private"})

    def test_message_metadata_cannot_cross_boundary(self):
        for key in ("raw_data", "cost", "reward", "hidden_user_instructions"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                PublicState(POLICY, ({**MESSAGES[0], key: "secret"},), tuple(TOOLS), action())

    def test_hidden_labels_cannot_change_same_public_payload(self):
        payloads = []
        for private_label in (0, 1):
            private_task = {"evaluation_criteria": {"reward": private_label}}
            # Agent boundary intentionally has no parameter through which private_task can pass.
            self.assertNotIn("evaluation_criteria", PublicState.__dataclass_fields__)
            payloads.append(fingerprint(PublicState(POLICY, tuple(MESSAGES), tuple(TOOLS), action()).to_dict()))
        self.assertEqual(payloads[0], payloads[1])

    def test_json_rejects_nan_and_non_string_keys(self):
        for value in (float("nan"), float("inf"), {1: "value"}, (1, 2), object()):
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                json_copy(value)

    def test_mutating_input_or_export_does_not_change_state(self):
        rows = deepcopy(MESSAGES)
        state = PublicState(POLICY, tuple(rows), tuple(TOOLS), action())
        rows[0]["content"] = "changed"
        exported = state.to_dict()
        exported["tools"][0]["parameters"]["type"] = "changed"
        self.assertEqual(state.messages[0]["content"], MESSAGES[0]["content"])
        self.assertEqual(state.to_dict()["tools"][0]["parameters"]["type"], "object")

    def test_text_and_tool_calls_cannot_mix(self):
        with self.assertRaises(ValueError):
            Action("I did it", action().tool_calls)
        with self.assertRaises(ValueError):
            Action()

    def test_projection_preserves_policy_history_proposed_arguments_and_all_names(self):
        extra = {"name": "other_tool", "description": "Other", "parameters": {"type": "object", "x": "long"}}
        full = PublicState(POLICY, tuple(MESSAGES), tuple(TOOLS + [extra]), action())
        view = gate_view(full)
        self.assertEqual(view["policy"], POLICY)
        self.assertEqual(view["messages"], MESSAGES)
        self.assertEqual(view["tools"], TOOLS)
        self.assertEqual(view["proposal"], action().to_dict())
        self.assertEqual(view["available_tool_names"], ["update_task_status", "other_tool"])
        extra["parameters"] = {"different": "unproposed schema detail"}
        self.assertEqual(gate_view(PublicState(POLICY, tuple(MESSAGES), tuple(TOOLS + [extra]), action())), view)

    def test_text_projection_has_no_tool_schemas_but_keeps_all_names(self):
        view = gate_view(PublicState(POLICY, tuple(MESSAGES), tuple(TOOLS), Action("What is your task ID?")))
        self.assertEqual(view["tools"], [])
        self.assertEqual(view["available_tool_names"], ["update_task_status"])
        with self.assertRaises(ValueError):
            validate_gate_view({**view, "evaluation_criteria": "private"})

    def test_score_rejects_invalid_probability_and_counts(self):
        for value in (True, -1, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GateScore(value, "model")
        with self.assertRaises(ValueError):
            GateScore(.5, "model", input_tokens=True)


class InterventionTests(unittest.TestCase):
    def test_reject_routes_to_independent_review_then_changed_action(self):
        calls = []
        def writer(policy, messages, tools, feedback):
            calls.append(feedback)
            return action("pending" if feedback is None else "completed")
        gate = SequenceGate([.1, .9])
        control = Controller(writer, lambda public: "The user requested completed, not pending.", gate)
        result = control.route(POLICY, MESSAGES, TOOLS)
        self.assertEqual(result.action.tool_calls[0]["arguments"]["status"], "completed")
        self.assertEqual([row["route"] for row in result.events], ["review", "execute"])
        self.assertEqual(calls, [None, "The user requested completed, not pending."])
        self.assertEqual(gate.states[1]["messages"], MESSAGES)

    def test_positive_gate_accepts_without_reviewer(self):
        reviews = []
        control = Controller(lambda *args: action(), lambda s: reviews.append(s), SequenceGate([.8]))
        result = control.route(POLICY, MESSAGES, TOOLS)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(reviews, [])

    def test_negative_gate_never_emits_rejected_action(self):
        control = Controller(lambda *args: action(), lambda state: "Check the request.", SequenceGate([.1] * 3))
        with self.assertRaises(RejectedAction) as raised:
            control.route(POLICY, MESSAGES, TOOLS)
        self.assertEqual(len(raised.exception.events), 3)
        self.assertFalse(any(row["route"] == "execute" for row in raised.exception.events))

    def test_no_gate_and_fixed_review_have_distinct_call_counts(self):
        calls = []
        def writer(*args):
            calls.append("writer")
            return action()
        def reviewer(state):
            calls.append("reviewer")
            return "Check status."
        Controller(writer, mode="no_gate").route(POLICY, MESSAGES, TOOLS)
        self.assertEqual(calls, ["writer"])
        calls.clear()
        Controller(writer, reviewer, mode="fixed_review").route(POLICY, MESSAGES, TOOLS)
        self.assertEqual(calls, ["writer", "reviewer", "writer"])

    def test_feedback_may_change_route_without_forcing_a_score_gap(self):
        # Different controller paths can validly return the same correct action.
        no_gate = Controller(lambda *args: action(), mode="no_gate").route(POLICY, MESSAGES, TOOLS)
        gated = Controller(lambda *args: action(), lambda state: "Check.", SequenceGate([.2, .9])).route(POLICY, MESSAGES, TOOLS)
        self.assertEqual(no_gate.action.to_dict(), gated.action.to_dict())
        self.assertNotEqual(len(no_gate.events), len(gated.events))

    def test_reviewer_cannot_mutate_executed_history(self):
        messages = deepcopy(MESSAGES)
        def writer(policy, rows, tools, feedback):
            rows[0]["content"] = "bad mutation"
            return action()
        def reviewer(state):
            state.messages[0]["content"] = "other mutation"
            return "Reconsider."
        Controller(writer, reviewer, SequenceGate([.1, .9])).route(POLICY, messages, TOOLS)
        self.assertEqual(messages, MESSAGES)

    def test_unknown_tool_is_never_returned_even_without_gate(self):
        unknown = Action(tool_calls=({"id": "x", "name": "delete_task", "arguments": {}},))
        with self.assertRaises(RejectedAction):
            Controller(lambda *args: unknown, mode="no_gate").route(POLICY, MESSAGES, TOOLS)

    def test_provider_exception_is_not_retried(self):
        calls = []
        class FailedGate:
            def score(self, state):
                calls.append(state)
                raise TimeoutError("uncertain dispatch")
        with self.assertRaises(RouteFailure) as raised:
            Controller(lambda *args: action(), lambda state: "review", FailedGate()).route(POLICY, MESSAGES, TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(raised.exception.events[0]["gate_status"], "failed_or_uncertain")

    def test_paid_gate_cost_survives_a_later_reviewer_failure(self):
        class PaidGate:
            def score(self, state):
                return GateScore(.1, "paid-test-double", input_tokens=100, output_tokens=2, cost_usd=.01)
        def failed_reviewer(state):
            raise TimeoutError("uncertain reviewer request")
        with self.assertRaises(RouteFailure) as raised:
            Controller(lambda *args: action(), failed_reviewer, PaidGate()).route(POLICY, MESSAGES, TOOLS)
        event = raised.exception.events[0]
        self.assertEqual(event["gate"]["cost_usd"], .01)
        self.assertEqual(event["gate_status"], "completed")
        self.assertEqual(event["feedback_status"], "failed_or_uncertain")


class AccountingTests(unittest.TestCase):
    def test_missing_and_infrastructure_failures_stay_in_denominator(self):
        planned = [("a", 42), ("b", 42), ("c", 42)]
        rows = [{"task_id": "a", "seed": 42, "status": "completed", "reward": 1.0},
                {"task_id": "b", "seed": 42, "status": "infrastructure_error", "reward": None}]
        result = summarize_planned(planned, rows)
        self.assertEqual(result["failure_inclusive_success"], 1/3)
        self.assertEqual(result["evaluated_success"], 1)
        self.assertEqual(result["missing_or_failed"], 2)

    def test_duplicate_or_unplanned_results_are_rejected(self):
        row = {"task_id": "a", "seed": 42, "status": "completed", "reward": 1}
        for rows in ([row, row], [{**row, "task_id": "other"}]):
            with self.assertRaises(ValueError):
                summarize_planned([("a", 42)], rows)

    def test_all_failed_evaluated_success_is_unavailable(self):
        result = summarize_planned([("a", 42)], [])
        self.assertEqual(result["failure_inclusive_success"], 0)
        self.assertIsNone(result["evaluated_success"])


class ProviderBoundaryTests(unittest.TestCase):
    def test_native_metadata_reports_action_temperature_not_checkpoint_calibration(self):
        class UnloadedBackend:
            def __init__(self, *args, **kwargs): pass
            def metadata(self):
                return {"temperature": 1.438, "checkpoint_head_sha256": "head",
                        "adapter_sha256": "adapter", "base_revision": "base"}
        with patch("pc_strands.decider.StrandsDecider", UnloadedBackend):
            backend = native_strands_backend(None, allow_model_calls=True)
            metadata = backend.metadata()
        self.assertEqual(metadata["temperature"], 1.0)
        self.assertEqual(metadata["checkpoint_stored_temperature"], 1.438)
        self.assertEqual(metadata["tau_calibration"], "not_fitted")
        self.assertFalse(metadata["hotpot_calibration_applied"])

    def test_execution_is_off_without_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            CommandGate(["unused"])
        with self.assertRaises(ValueError):
            native_strands_backend(None)
        with self.assertRaises(ValueError):
            NativeStrandsGate(None, None)

    def test_command_timeout_never_retries_or_prints_child_stderr(self):
        gate = CommandGate(["explicit-model-process"], allow_model_calls=True)
        state = PublicState(POLICY, tuple(MESSAGES), tuple(TOOLS), action())
        with patch("haltiq_tau.providers.subprocess.run", side_effect=subprocess.TimeoutExpired("hidden", 1)) as run:
            with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                gate.score(state)
        self.assertEqual(run.call_count, 1)

    def test_command_receives_new_public_action_question_not_qa(self):
        gate = CommandGate(["explicit-model-process"], allow_model_calls=True)
        state = PublicState(POLICY, tuple(MESSAGES), tuple(TOOLS), action())
        response = subprocess.CompletedProcess([], 0, json.dumps({"p_allowed": .9, "model": "test"}), "")
        with patch("haltiq_tau.providers.subprocess.run", return_value=response) as run:
            score = gate.score(state)
        payload = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(set(payload["state"]), {"context_protocol", "policy", "messages", "tools", "available_tool_names", "proposal"})
        self.assertIn("next assistant action", payload["instructions"])
        self.assertEqual(score.p_allowed, .9)
        self.assertIsNone(score.logits)

    def test_example_pin_validates_and_rejects_version_drift(self):
        path = Path(__file__).resolve().parents[1] / "configs/tau2/pilot.example.json"
        example = json.loads(path.read_text())
        validate_pilot(example)
        with self.assertRaises(ValueError):
            validate_pilot({**example, "upstream_revision": "different"})
        with self.assertRaises(ValueError):
            validate_pilot({**example, "evaluation_type": "all_ignore_basis"})
        validate_pilot({**example, "arms": ["no_gate", "fixed_review", "strands"]})

    def test_live_cli_refuses_local_gpu_before_dispatch(self):
        from haltiq_tau.__main__ import main
        path = Path(__file__).resolve().parents[1] / "configs/tau2/pilot.example.json"
        config = json.loads(path.read_text())
        config.update(arms=["no_gate", "strands"], writer_model="ollama_chat/qwen3:8b",
                      reviewer_model="ollama_chat/qwen3:8b", user_model="ollama_chat/qwen3:8b")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            argv = ["haltiq_tau", "run", "--config", str(config_path), "--upstream-root", directory,
                    "--output", str(Path(directory) / "output"), "--control-root", str(Path(directory) / "control"),
                    "--deadline", "2099-01-01T20:00:00-05:00", "--allow-model-calls",
                    "--strands-command", '["unused"]', "--strands-preflight-command", '["unused"]']
            with patch("sys.argv", argv), patch("haltiq_tau.pilot.run_pilot") as run, self.assertRaises(SystemExit):
                main()
            run.assert_not_called()
            self.assertFalse((Path(directory) / "output").exists())


if __name__ == "__main__":
    unittest.main()
