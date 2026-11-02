"""Synthetic engineering controls; no model generation or benchmark claims."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from haltiq_pair import hotpot
from haltiq_pair.nov2_orchestration import Orchestrator, OrchestrationError, RoleAgents, State, valid_actions


PUBLIC = hotpot.PublicQuestion("public-id", "Which city?", (("Evidence", ("The city is Paris.",)),))


class FakeAgents:
    def __init__(self, outputs=None):
        self.usage, self.prompts = [], []
        self.outputs = outputs or {}

    def generate(self, role, system, prompt):
        self.prompts.append((role, system, prompt))
        self.usage.append({"role": role, "status": "completed", "usage": {"totalTokens": 10}})
        queue = self.outputs.get(role, [])
        value = queue.pop(0) if queue else {
            "writer": '{"answer":"Paris","supporting_facts":[["Evidence",0]]}',
            "reviewer": '{"approved":false,"feedback":"Check supporting evidence."}',
            "challenger": '{"alternative":"Paris","evidence":[["Evidence",0]],"feedback":"ACTUAL_CHALLENGE: confirm the city."}',
        }[role]
        if isinstance(value, Exception):
            self.usage[-1]["status"] = "failed"
            raise value
        return value


class FakeCritic:
    last_record = {"synthetic": True}

    def score(self, question, passages, answer):
        self.last = (question, passages, answer)
        return .6


class OrchestrationControls(unittest.TestCase):
    def test_three_actual_roles_and_challenge_reaches_writer(self):
        agents = FakeAgents()
        engine = Orchestrator(agents, FakeCritic())
        initial = engine.begin(PUBLIC)
        saved = initial.to_dict()
        revised = engine.transition(initial, "INVOKE")
        self.assertEqual([p[0] for p in agents.prompts], ["writer", "reviewer", "challenger", "writer", "reviewer"])
        self.assertIn("ACTUAL_CHALLENGE", agents.prompts[3][2])
        self.assertIn("reviewer_feedback", agents.prompts[3][2])
        self.assertNotIn("ACTUAL_CHALLENGE", agents.prompts[0][2])
        self.assertEqual(initial.to_dict(), saved)
        self.assertTrue(revised.challenger_used)
        self.assertEqual(revised.agent_calls, 5)
        self.assertEqual(revised.history[-1]["challenger_response"]["alternative"], "Paris")
        self.assertEqual(valid_actions(revised), ("STOP", "REVISE"))
        self.assertEqual(len(revised.critic_records), 2)

    def test_stop_has_no_new_calls_and_cap_is_not_learned_stop(self):
        agents = FakeAgents()
        engine = Orchestrator(agents, FakeCritic(), max_drafts=2)
        state = engine.begin(PUBLIC)
        stopped = engine.transition(state, "STOP")
        self.assertEqual(stopped.terminal_reason, "policy_stop")
        self.assertEqual(len(agents.usage), 2)
        capped = engine.transition(state, "REVISE")
        self.assertEqual(capped.terminal_reason, "budget_cap")
        self.assertEqual(capped.draft_number, 2)
        self.assertEqual(valid_actions(capped), ("STOP",))
        with self.assertRaises(ValueError):
            engine.transition(capped, "INVOKE")

    def test_format_corrections_count_and_never_overrun_mandatory_chain(self):
        agents = FakeAgents({"writer": ["bad"]})
        engine = Orchestrator(agents, FakeCritic(), max_agent_calls=2)
        with self.assertRaises(OrchestrationError) as context:
            engine.begin(PUBLIC)
        self.assertEqual(len(context.exception.usage), 1)
        self.assertEqual([p[0] for p in agents.prompts], ["writer"])
        self.assertIsNone(context.exception.state)
        self.assertEqual(context.exception.critic_records, [])

    def test_revision_failure_keeps_attempts_and_original_public_state(self):
        agents = FakeAgents()
        engine = Orchestrator(agents, FakeCritic())
        state = engine.begin(PUBLIC)
        before = state.to_dict()
        agents.outputs["writer"] = [RuntimeError("synthetic transport failure")]
        with self.assertRaises(OrchestrationError) as context:
            engine.transition(state, "REVISE")
        self.assertEqual(len(context.exception.usage), 3)
        self.assertEqual(len(context.exception.critic_records), 1)
        self.assertEqual(state.to_dict(), before)
        self.assertEqual(context.exception.error_type, "RuntimeError")

    def test_invalid_challenger_evidence_fails_without_writer_or_gold(self):
        agents = FakeAgents({"challenger": ['{"alternative":"Rome","evidence":[["SECRET",0]],"feedback":"try"}'] * 2})
        engine = Orchestrator(agents, FakeCritic())
        state = engine.begin(PUBLIC)
        with self.assertRaises(OrchestrationError) as context:
            engine.transition(state, "INVOKE")
        self.assertEqual([p[0] for p in agents.prompts], ["writer", "reviewer", "challenger", "challenger"])
        self.assertEqual(len(context.exception.usage), 4)

    def test_state_round_trip_and_rejects_evaluator_fields(self):
        engine = Orchestrator(FakeAgents(), FakeCritic())
        state = engine.begin(PUBLIC)
        self.assertEqual(State.from_dict(state.to_dict()), state)
        value = state.to_dict()
        value["public"]["answer"] = "private reference"
        with self.assertRaises(ValueError):
            State.from_dict(value)
        with self.assertRaises(ValueError):
            engine.begin({"question": "Which city?", "answer": "private reference"})

    def test_agent_budget_limits_mask_invocation_and_cap(self):
        engine = Orchestrator(FakeAgents(), FakeCritic(), max_agent_calls=4)
        state = engine.begin(PUBLIC)
        self.assertEqual(valid_actions(state), ("STOP", "REVISE"))
        end = engine.transition(state, "REVISE")
        self.assertEqual(end.agent_calls, 4)
        self.assertEqual(end.terminal_reason, "agent_budget_cap")

    def test_verifier_failure_is_retained(self):
        critic = FakeCritic()
        critic.score = lambda *args: float("nan")
        with self.assertRaises(OrchestrationError) as context:
            Orchestrator(FakeAgents(), critic).begin(PUBLIC)
        self.assertEqual(len(context.exception.usage), 2)
        self.assertEqual(context.exception.critic_records[0]["status"], "failed")

    def test_sdk_metadata_and_failure_receipt_keep_real_tokens(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            def model(**kwargs):
                captured.update(kwargs)
                return object()

            class Agent:
                def __init__(self, **kwargs):
                    captured["role"] = kwargs["name"]

                def __call__(self, prompt):
                    metadata = captured["params"]["metadata"]
                    receipt = {"call_id": metadata["yash_call_id"], "role": metadata["yash_role"],
                        "response_id": "response", "primary_receipt": "requests/response.json",
                        "usage": {"prompt_tokens": 30, "completion_tokens": 256, "total_tokens": 286},
                        "finish_reason": "length"}
                    Path(tmp, receipt["call_id"] + ".json").write_text(json.dumps(receipt), encoding="utf-8")
                    return SimpleNamespace(metrics=SimpleNamespace(accumulated_usage={}),
                        message={"content": [{"text": "truncated"}]}, stop_reason="length")

            with patch.dict("sys.modules", {"strands": SimpleNamespace(Agent=Agent),
                    "strands.models.openai": SimpleNamespace(OpenAIModel=model)}):
                adapter = RoleAgents(receipt_root=tmp)
                with self.assertRaises(ValueError):
                    adapter.generate("challenger", "system", "public prompt")
            row = adapter.usage[0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["usage"]["total_tokens"], 286)
            self.assertTrue(row["token_accounting_known"])
            self.assertEqual(len(row["call_id"]), 32)
            self.assertEqual(captured["params"]["max_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
