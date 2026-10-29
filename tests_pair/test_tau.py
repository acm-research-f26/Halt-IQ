"""Action isolation plus optional integration with the actual pinned Tau tools."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from haltiq_pair.tau import (critic_inputs, local_model_args, make_agent, parse_proposal,
                            local_judges, run)


class ProposalTests(unittest.TestCase):
    def test_exclusive_public_tool_proposal(self):
        row = {"content": None, "tool_calls": [{"name": "get_order", "arguments": {"id": "one"}}]}
        self.assertEqual(parse_proposal(json.dumps(row), {"get_order"}), row)
        for bad in ({**row, "gold": "hidden"}, {**row, "content": "Also tell the user"},
                    {"content": None, "tool_calls": []},
                    {"content": None, "tool_calls": [{"name": "delete_all", "arguments": {}}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_proposal(json.dumps(bad), {"get_order"})

    def test_only_loopback_generation_is_allowed(self):
        self.assertEqual(local_model_args()["num_retries"], 0)
        for endpoint in ("https://api.openai.com/v1", "http://localhost.example.com/v1",
                         "http://user:password@localhost/v1", "http://localhost/v1?target=remote"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                local_model_args(endpoint)

    def test_tau_critic_query_does_not_repeat_history_and_all_policy_is_a_candidate(self):
        policy = "0123456789" * 500
        schemas = [{"name": "read", "parameters": {}}, {"name": "write", "parameters": {"x": "complete"}}]
        messages = [{"role": "user", "content": "Earlier request"},
                    {"role": "tool", "content": "Visible result"},
                    {"role": "user", "content": "Latest request"}]
        question, passages = critic_inputs(policy, schemas, messages,
            {"content": None, "tool_calls": [{"name": "write", "arguments": {}}]})
        self.assertIn("Latest request", question)
        self.assertNotIn("Earlier request", question)
        self.assertNotIn("Visible result", question)
        self.assertIn("complete", "\n".join(passages))
        self.assertNotIn('"name": "read"', "\n".join(passages))
        policy_chunks = [p.split("\n", 1)[1] for p in passages if p.startswith("Retail policy excerpt")]
        reconstructed = policy_chunks[0] + "".join(p[150:] for p in policy_chunks[1:])
        self.assertEqual(reconstructed, policy)


HAS_TAU = importlib.util.find_spec("tau2") is not None


@unittest.skipUnless(HAS_TAU, "optional pinned official Tau dependency")
class AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        from tau2.domains.mock.environment import get_environment
        self.env = get_environment()

    def test_rejected_action_does_not_execute_and_review_changes_action(self):
        from tau2.data_model.message import UserMessage
        class Agents:
            usage = []
            prompts = []
            def generate(self, role, system, prompt):
                self.prompts.append(prompt)
                if role == "reviewer":
                    return "Use the requested title Important Meeting."
                title = "Important Meeting" if "Independent review:" in prompt else "Wrong Meeting"
                return json.dumps({"content": None, "tool_calls": [{"name": "create_task", "arguments": {
                    "user_id": "user_1", "title": title}}]})
        class Gate:
            def decide(self, arm, draft, previous, round_index, max_rounds, question, passages, **kwargs):
                return {"stop": "Important Meeting" in draft, "distance": 1.0,
                        "p_stop": .9, "p_approve": .9 if round_index == 2 else .1}
        agents = Agents()
        agent = make_agent(agents, Gate(), self.env.get_tools(), self.env.get_policy(), "combined")
        message, state = agent.generate_next_message(
            UserMessage.text("Create Important Meeting for user_1."), agent.get_init_state())
        self.assertEqual(message.tool_calls[0].arguments["title"], "Important Meeting")
        self.assertEqual([e["route"] for e in agent.events], ["review", "execute"])
        self.assertNotIn("Important Meeting", str(self.env.tools.db.model_dump()))
        self.assertFalse(self.env.get_response(message.tool_calls[0]).error)
        self.assertIn("Important Meeting", str(self.env.tools.db.model_dump()))
        self.assertNotIn("Wrong Meeting", str(self.env.tools.db.model_dump()))
        self.assertEqual(len(state.messages), 2)

    def test_cap_rejection_never_returns_executable_tool(self):
        from tau2.agent.base_agent import AgentError
        from tau2.data_model.message import UserMessage
        class Agents:
            def generate(self, role, system, prompt):
                return json.dumps({"content": None, "tool_calls": [{"name": "create_task", "arguments": {
                    "user_id": "user_1", "title": "Wrong Meeting"}}]})
        class Gate:
            def decide(self, *args, **kwargs): return {"stop": False, "distance": 1.0}
        agent = make_agent(Agents(), Gate(), self.env.get_tools(), self.env.get_policy(), "combined", max_rounds=1)
        before = self.env.tools.db.model_dump()
        with self.assertRaises(AgentError):
            agent.generate_next_message(UserMessage.text("Make Important Meeting."), agent.get_init_state())
        self.assertEqual(before, self.env.tools.db.model_dump())
        self.assertEqual(agent.events[-1]["route"], "rejected_at_cap")

    def test_local_judges_restore_globals_and_never_select_hosted_provider(self):
        from tau2.user import user_simulator
        from tau2.evaluator import evaluator_nl_assertions
        from tau2.data_model.message import AssistantMessage
        original_model = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS
        original_args = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS
        seen = []
        def fake(**kwargs):
            seen.append(kwargs)
            return AssistantMessage(role="assistant", content="###STOP###",
                                    usage={"prompt_tokens": 1, "completion_tokens": 1})
        with patch.object(user_simulator, "generate", fake):
            with local_judges() as calls:
                user_simulator.generate(model="openai/Qwen/Qwen2.5-1.5B-Instruct", messages=[])
                with self.assertRaises(ValueError):
                    user_simulator.generate(model="openai/hosted-model", messages=[])
            self.assertIs(user_simulator.generate, fake)
        self.assertEqual(seen[0]["api_base"], "http://127.0.0.1:8001/v1")
        self.assertEqual(seen[0]["response_validation_retries"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["hosted_api_calls"], 0)
        self.assertEqual(evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS, original_model)
        self.assertIs(evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS, original_args)

    def test_missing_nl_judge_results_fail_instead_of_vacuously_passing(self):
        from tau2.evaluator import evaluator_nl_assertions
        from tau2.data_model.message import AssistantMessage, UserMessage
        def fake(**kwargs):
            return AssistantMessage(role="assistant", content='{"results": []}')
        with patch.object(evaluator_nl_assertions, "generate", fake):
            with local_judges() as calls:
                with self.assertRaisesRegex(ValueError, "every expected outcome"):
                    evaluator_nl_assertions.generate(model="openai/Qwen/Qwen2.5-1.5B-Instruct",
                        messages=[UserMessage.text("expectedOutcomes:\n['Agent confirmed success']")])
        self.assertEqual(calls[0]["status"], "failed")


@unittest.skipUnless(HAS_TAU and os.environ.get("HALTIQ_TAU_UPSTREAM"),
                     "optional actual pinned checkout; no live model calls")
class OfficialRunnerTests(unittest.TestCase):
    def test_native_retail_simulation_and_failure_denominator_with_scripted_io(self):
        from tau2.data_model.message import AssistantMessage
        from tau2.user import user_simulator
        class Agents:
            usage = []
            def generate(self, role, system, prompt):
                self.usage.append({"role": role, "scripted_test_only": True})
                return json.dumps({"content": "Please provide the order ID.", "tool_calls": []})
        class Gate:
            def decide(self, *args, **kwargs):
                return {"stop": True, "distance": 1.0, "p_stop": .8, "p_approve": .9}
        calls = []
        def user(**kwargs):
            calls.append(kwargs)
            return AssistantMessage(role="assistant", content="Please help me.",
                cost=0.0, usage={"prompt_tokens": 0, "completion_tokens": 0})
        with tempfile.TemporaryDirectory() as temp, patch.object(user_simulator, "generate", user):
            (Path(temp) / "run" / "runtime").mkdir(parents=True)
            summary = run(Agents(), Gate(), Path(temp) / "run", limit=1, max_steps=2,
                          upstream_root=os.environ["HALTIQ_TAU_UPSTREAM"])
            self.assertEqual(summary["planned_episodes"], 4)
            self.assertEqual(summary["reported_episodes"], 4)
            self.assertTrue(all(arm["success_failure_inclusive"] == 0 for arm in summary["arms"].values()))
            rows = [json.loads(line) for line in (Path(temp) / "run/outcomes.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["termination_reason"] == "max_steps" for row in rows))
            self.assertTrue(all(row["status"] == "completed" for row in rows))
            self.assertFalse(summary["leaderboard_comparable"])
            self.assertEqual(len(calls), 4)
            with self.assertRaisesRegex(ValueError, "already contains study artifacts"):
                run(Agents(), Gate(), Path(temp) / "run", limit=1, max_steps=2,
                    upstream_root=os.environ["HALTIQ_TAU_UPSTREAM"])
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
