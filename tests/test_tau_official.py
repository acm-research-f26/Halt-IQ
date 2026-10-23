"""Optional official dependency integration. No model or paid API calls."""
from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = importlib.util.find_spec("tau2") is not None
CHECKOUT = os.environ.get("HALTIQ_TAU_UPSTREAM")


@unittest.skipUnless(AVAILABLE and CHECKOUT, "requires isolated pinned Tau runtime and HALTIQ_TAU_UPSTREAM")
class OfficialIntegrationTests(unittest.TestCase):
    def test_generation_hooks_wrap_actual_official_user_dispatch_and_no_retry(self):
        from datetime import datetime, timedelta, timezone
        from tau2.data_model.message import AssistantMessage
        from tau2.user import user_simulator
        from haltiq_tau.dispatch import guard_official_calls
        events = []
        def original(**kwargs):
            events.append("dispatch")
            self.assertEqual(kwargs["num_retries"], 0)
            self.assertEqual(kwargs["response_validation_retries"], 0)
            self.assertLessEqual(kwargs["timeout"], 10)
            return AssistantMessage(role="assistant", content="Hello", cost=0,
                                    usage={"prompt_tokens": 4, "completion_tokens": 2})
        with patch.object(user_simulator, "generate", original):
            with guard_official_calls(datetime.now(timezone.utc) + timedelta(seconds=10),
                                     before_call=lambda: events.append("admit"),
                                     after_call=lambda elapsed: events.append("rest")) as calls:
                user_simulator.generate(model="scripted", messages=[], timeout=60, num_retries=3)
            self.assertIs(user_simulator.generate, original)
        self.assertEqual(events, ["admit", "dispatch", "rest"])
        self.assertEqual(calls[0]["input_tokens"], 4)
        self.assertIsNone(calls[0]["cost_usd"])

    def test_expired_deadline_blocks_official_judge_before_any_call(self):
        from tau2.evaluator import evaluator_communicate
        from haltiq_tau.dispatch import guard_official_calls
        with patch.object(evaluator_communicate, "generate") as called:
            with guard_official_calls(datetime.now(timezone.utc) - timedelta(seconds=1)) as calls:
                with self.assertRaisesRegex(RuntimeError, "before official"):
                    evaluator_communicate.generate(model="unused", messages=[])
            self.assertEqual(calls, [])
            called.assert_not_called()

    def test_local_ollama_resource_arguments_reach_the_request(self):
        """Inspect an actual LiteLLM request transformation without a network call."""
        import httpx
        import litellm
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        captured = []
        def fake_post(self, url, **kwargs):
            captured.append(json.loads(kwargs["data"]))
            return httpx.Response(200, request=httpx.Request("POST", url), json={
                "model": "qwen3:8b", "created_at": "2026-10-22T00:00:00Z", "done": True,
                "message": {"role": "assistant", "content": "fixture"},
                "prompt_eval_count": 4, "eval_count": 2})
        with patch.object(HTTPHandler, "post", fake_post):
            litellm.completion(model="ollama_chat/qwen3:8b", api_base="http://127.0.0.1:11435",
                messages=[{"role": "user", "content": "fixture"}], num_thread=4,
                num_ctx=16384, num_gpu=0, think=False, num_retries=0, max_tokens=512)
        self.assertEqual(captured[0]["options"]["num_thread"], 4)
        self.assertEqual(captured[0]["options"]["num_ctx"], 16384)
        self.assertEqual(captured[0]["options"]["num_gpu"], 0)
        self.assertEqual(captured[0]["think"], False)

    def test_official_environment_evaluator_separates_known_paths(self):
        from haltiq_tau.mock_check import run_mock_check
        result = run_mock_check(CHECKOUT)
        self.assertEqual([row["reward"] for row in result["results"]], [1, 0])
        self.assertFalse(result["model_quality_result"])

    def test_actual_agent_only_returns_approved_tool_action(self):
        from tau2.data_model.message import UserMessage, ToolMessage
        from tau2.domains.mock.environment import get_environment
        from haltiq_tau.agent import TauPolicyAgent
        from haltiq_tau.core import Action, Controller, GateScore
        env = get_environment()
        def writer(policy, messages, tools, feedback):
            return Action(tool_calls=({"id": "write-1", "name": "create_task", "arguments": {
                "user_id": "user_1", "title": "Important Meeting" if feedback else "Wrong Meeting"}},))
        class Gate:
            def score(self, public):
                title = public.proposal.tool_calls[0]["arguments"]["title"]
                return GateScore(.9 if title == "Important Meeting" else .1, "scripted-not-a-model")
        agent = TauPolicyAgent(env.get_tools(), env.get_policy(), Controller(writer, lambda state: "Use requested title.", Gate()))
        message, state = agent.generate_next_message(UserMessage.text("Create Important Meeting for user_1."), agent.get_init_state())
        self.assertEqual(message.tool_calls[0].arguments["title"], "Important Meeting")
        self.assertEqual(len(state.messages), 2)
        self.assertEqual([row["route"] for row in state.route_events], ["review", "execute"])
        response = env.get_response(message.tool_calls[0])
        self.assertFalse(response.error)
        self.assertNotIn("Wrong Meeting", str(env.tools.db.model_dump()))
        self.assertIn("Important Meeting", str(env.tools.db.model_dump()))
        with self.assertRaises(ValueError):
            agent.get_init_state([ToolMessage(role="tool", id="private", requestor="user", content="hidden")])

    def test_multiunit_pilot_keeps_every_arm_and_unit_with_scripted_generators(self):
        """Actual official domains/orchestrators/evaluators; scripted IO, NOT model quality."""
        from tau2.data_model.message import AssistantMessage
        from haltiq_tau.core import Action, GateScore
        from haltiq_tau.pilot import run_pilot
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/tau2/pilot.example.json").read_text())
        config.update(writer_model="scripted-fixture-writer", reviewer_model="scripted-fixture-reviewer",
                      user_model="scripted-fixture-user", max_steps=4)
        class Writer:
            def __init__(self, *args, **kwargs): self.calls = []
            def set_seed(self, seed): pass
            def __call__(self, *args):
                self.calls.append({"status": "completed", "scripted": True})
                return Action("I can help with your request.")
        class Reviewer(Writer):
            def __call__(self, *args):
                self.calls.append({"status": "completed", "scripted": True})
                return "Check the customer's request and policy."
        class Gate:
            def score(self, state): return GateScore(.9, "scripted-not-a-model")
            def identity(self): return {"provider": "scripted-test-double-not-a-model"}
            def preflight(self, state): return {"fits": True, "model_forwards": 0, "scripted_test_double": True}
        counter = {"calls": 0}
        def scripted_user(**kwargs):
            counter["calls"] += 1
            # The first arm reaches a known step-cap failure; later arms still run.
            content = "Please help me." if counter["calls"] <= 2 or counter["calls"] % 2 else "###STOP###"
            return AssistantMessage(role="assistant", content=content,
                                    cost=0, usage={"prompt_tokens": 0, "completion_tokens": 0})
        with tempfile.TemporaryDirectory() as directory, \
             patch("haltiq_tau.llm.TauWriter", Writer), patch("haltiq_tau.llm.TauReviewer", Reviewer), \
             patch("tau2.user.user_simulator.generate", side_effect=scripted_user), \
             patch("tau2.utils.llm_utils.completion", side_effect=AssertionError("No live calls in fixture test")):
            result = run_pilot(config, CHECKOUT, Path(directory) / "pilot", {"strands": Gate, "jev": Gate},
                               allow_model_calls=True, deadline=datetime.now(timezone.utc) + timedelta(minutes=2),
                               before_call=lambda: None, after_call=lambda elapsed: None)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(set(result["arms"]), set(config["arms"]))
            self.assertTrue(all(row["planned"] == 4 and row["reported"] == 4 for row in result["arms"].values()))
            outcomes = (Path(directory) / "pilot/outcomes.jsonl").read_text().splitlines()
            self.assertEqual(len(outcomes), 16)
            first = json.loads(outcomes[0])
            self.assertEqual(first["arm"], "no_gate")
            self.assertEqual(first["termination_reason"], "max_steps")
            self.assertEqual(first["reward"], 0)


if __name__ == "__main__":
    unittest.main()
