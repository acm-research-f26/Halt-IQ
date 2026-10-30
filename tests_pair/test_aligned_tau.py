"""Bounded format correction, fresh tasks and native-tool isolation controls."""
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from haltiq_pair.aligned_tau import (ARMS, FORMAT_POLICY, MODEL, ORIGINAL_TASK_IDS,
                                     make_agent, propose_valid, run,
                                     select_task_ids, summarize)


class ScriptedAgents:
    model_id = MODEL
    provenance = {"scripted_test_only": True, "model_id": MODEL}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.usage, self.prompts = [], []

    def generate(self, role, system, prompt):
        self.prompts.append((role, system, prompt))
        self.usage.append({"role": role, "usage": {"totalTokens": 3}, "scripted_test_only": True})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class ProposalAndSelectionTests(unittest.TestCase):
    def test_one_format_correction_retains_both_raw_outputs_and_usage(self):
        valid = json.dumps({"content": "Please provide your order ID.", "tool_calls": []})
        agents, events = ScriptedAgents(["{}", valid]), []
        proposal = propose_valid(agents, "public policy", "public observation", set(),
                                 round_index=1, trace=events.append)
        self.assertEqual(proposal["content"], "Please provide your order ID.")
        self.assertEqual(len(agents.usage), 2)
        self.assertIn("{}", agents.prompts[1][2])
        self.assertIn("public observation", agents.prompts[1][2])
        self.assertIn("exactly content and tool_calls", agents.prompts[1][2])
        self.assertEqual([e["raw"] for e in events if e["event"] == "writer_output"], ["{}", valid])
        self.assertEqual(sum(len(e["call_usage"]) for e in events if e["event"] == "writer_output"), 2)
        self.assertEqual(sum(e["event"] == "format_correction_requested" for e in events), 1)

    def test_second_invalid_response_fails_without_third_call(self):
        agents, events = ScriptedAgents(["{}", "[]", "must not be used"]), []
        with self.assertRaisesRegex(ValueError, "one declared format correction"):
            propose_valid(agents, "policy", "observed", set(), round_index=1, trace=events.append)
        self.assertEqual(len(agents.usage), 2)
        self.assertEqual(sum(e["event"] == "invalid_proposal" for e in events), 2)

    def test_provider_failure_does_not_trigger_format_correction(self):
        agents, events = ScriptedAgents([RuntimeError("GPU unavailable"), "unused"]), []
        with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
            propose_valid(agents, "policy", "observed", set(), round_index=1, trace=events.append)
        self.assertEqual(len(agents.usage), 1)
        self.assertFalse(any(e["event"] == "format_correction_requested" for e in events))

    def test_valid_text_is_not_repaired_using_correctness_or_gold(self):
        raw = json.dumps({"content": "A potentially incorrect but contract-valid answer.", "tool_calls": []})
        agents = ScriptedAgents([raw, "unused"])
        self.assertEqual(propose_valid(agents, "policy", "public observation", set(),
            round_index=1, trace=lambda event: None), json.loads(raw))
        self.assertEqual(len(agents.usage), 1)
        self.assertFalse(FORMAT_POLICY["gold_or_reward_based_repairs"])

    def test_evaluation_excludes_old_ten_and_new_two_smokes_deterministically(self):
        ids = list(ORIGINAL_TASK_IDS) + [str(i) for i in range(150, 180)]
        selected, development = select_task_ids(ids)
        reverse, reverse_development = select_task_ids(list(reversed(ids)))
        self.assertEqual((selected, development), (reverse, reverse_development))
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(development), 2)
        self.assertFalse(set(selected) & (ORIGINAL_TASK_IDS | set(development)))
        smoke, _ = select_task_ids(ids, phase="smoke", limit=2)
        self.assertEqual(smoke, development)
        self.assertEqual(select_task_ids(ids, manifest={"task_ids": selected})[0], selected)
        with self.assertRaisesRegex(ValueError, "frozen public-hash"):
            select_task_ids(ids, manifest={"task_ids": selected[::-1]})
        with self.assertRaisesRegex(ValueError, "data only"):
            select_task_ids(ids, manifest={"task_ids": selected, "code": "not permitted"})

    def test_failed_and_unscored_attempts_remain_in_denominator(self):
        rows = [{"arm": arm, "status": "completed" if arm == "first_valid" else "failed",
                 "reward": 0.0, "route_events": [{"route": "rejected_at_cap"}],
                 "format_corrections_requested": 1, "writer_reviewer_usage": [{}, {}],
                 "official_calls": [{}]} for arm in ARMS]
        summary = summarize(rows, 2, ARMS)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["planned_episodes"], 8)
        self.assertTrue(all(a["success_failure_inclusive"] == 0 for a in summary["arms"].values()))
        self.assertEqual(summary["arms"]["retrained_AND"]["controller_cap_refusals"], 1)
        self.assertEqual(summary["arms"]["retrained_AND"]["writer_reviewer_invocations"], 2)

    def test_interrupting_the_last_planned_attempt_does_not_complete_the_study(self):
        row = {"arm": "first_valid", "status": "interrupted", "reward": 0.0,
               "route_events": [], "format_corrections_requested": 0,
               "writer_reviewer_usage": [], "official_calls": []}
        summary = summarize([row], 1, ("first_valid",))
        self.assertEqual(summary["reported_episodes"], summary["planned_episodes"])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["arms"]["first_valid"]["interrupted"], 1)


HAS_TAU = importlib.util.find_spec("tau2") is not None


@unittest.skipUnless(HAS_TAU, "optional pinned official Tau dependency")
class NativeAgentTests(unittest.TestCase):
    def setUp(self):
        from tau2.domains.mock.environment import get_environment
        self.env = get_environment()
        self.good = json.dumps({"content": None, "tool_calls": [{"name": "create_task", "arguments": {
            "user_id": "user_1", "title": "Important Meeting"}}]})

    def test_only_contract_valid_returned_tool_reaches_the_official_executor(self):
        from tau2.data_model.message import UserMessage
        agents, events = ScriptedAgents(["{}", self.good]), []
        agent = make_agent(agents, None, self.env.get_tools(), self.env.get_policy(),
                           "first_valid", trace=events.append)
        before = self.env.tools.db.model_dump()
        response, state = agent.generate_next_message(UserMessage.text("Create Important Meeting for user_1."),
                                                       agent.get_init_state())
        self.assertEqual(before, self.env.tools.db.model_dump())
        self.assertEqual(len(agents.usage), 2)
        self.assertEqual(agent.events[-1]["route"], "execute")
        self.assertEqual(len(state.messages), 2)
        self.assertFalse(self.env.get_response(response.tool_calls[0]).error)
        self.assertIn("Important Meeting", str(self.env.tools.db.model_dump()))

    def test_and_cap_refusal_never_returns_or_executes_a_tool(self):
        from tau2.agent.base_agent import AgentError
        from tau2.data_model.message import UserMessage
        class Gate:
            def decide(self, arm, *args, **kwargs):
                self.arm = arm
                return {"stop": False, "distance": 1.0, "p_stop": .99, "p_approve": .1}
        gate = Gate()
        agent = make_agent(ScriptedAgents([self.good]), gate, self.env.get_tools(),
                           self.env.get_policy(), "retrained_AND", max_rounds=1)
        before = self.env.tools.db.model_dump()
        with self.assertRaisesRegex(AgentError, "revision cap"):
            agent.generate_next_message(UserMessage.text("Create the task."), agent.get_init_state())
        self.assertEqual(before, self.env.tools.db.model_dump())
        self.assertEqual(gate.arm, "retrained_AND")
        self.assertEqual(agent.events[-1]["route"], "rejected_at_cap")

    def test_twice_invalid_never_reaches_a_gate_or_tool(self):
        from tau2.agent.base_agent import AgentError
        from tau2.data_model.message import UserMessage
        agent = make_agent(ScriptedAgents(["{}", "[]"]), None, self.env.get_tools(),
                           self.env.get_policy(), "first_valid")
        before = self.env.tools.db.model_dump()
        with self.assertRaises(AgentError):
            agent.generate_next_message(UserMessage.text("Create a task."), agent.get_init_state())
        self.assertEqual(before, self.env.tools.db.model_dump())
        self.assertEqual(agent.events[-1]["route"], "technical_failure")

    def test_actual_official_client_strips_provider_prefix_before_loopback_http(self):
        from haltiq_pair.tau import local_judges
        from tau2.data_model.message import UserMessage
        from tau2.user import user_simulator
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                payload = json.dumps({"id": "chatcmpl-offline-test", "object": "chat.completion",
                    "created": 1, "model": MODEL,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
            with local_judges(endpoint, MODEL, seed=43) as calls:
                response = user_simulator.generate(model="openai/" + MODEL,
                                                    messages=[UserMessage.text("Public observation only.")])
            self.assertEqual(response.content, "Hello.")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["status"], "completed")
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]["model"], MODEL)
            self.assertEqual(seen[0]["temperature"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


@unittest.skipUnless(HAS_TAU and os.environ.get("HALTIQ_TAU_UPSTREAM"), "pinned checkout required")
class NativeRunnerTests(unittest.TestCase):
    def test_official_episode_local7b_route_plan_and_failure_denominator(self):
        from tau2.data_model.message import AssistantMessage
        from tau2.user import user_simulator
        agents = ScriptedAgents([json.dumps({"content": "Please provide the order ID.", "tool_calls": []})] * 20)
        class Gate:
            provenance = {"scripted_test_only": True}
            def decide(self, *args, **kwargs):
                return {"stop": True, "distance": 0.0}
        seen = []
        def user(**kwargs):
            seen.append(kwargs)
            return AssistantMessage(role="assistant", content="Please help me.",
                cost=0.0, usage={"prompt_tokens": 0, "completion_tokens": 0})
        with tempfile.TemporaryDirectory() as temporary, patch.object(user_simulator, "generate", user):
            out = Path(temporary) / "study"
            summary = run(agents, Gate(), out, limit=1, max_steps=2,
                          upstream_root=os.environ["HALTIQ_TAU_UPSTREAM"])
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(summary["reported_episodes"], 4)
            self.assertTrue(summary["all_simulations_executed"])
            self.assertTrue(all(a["success_failure_inclusive"] == 0 for a in summary["arms"].values()))
            self.assertEqual(plan["simulator_and_judge_model"], MODEL)
            self.assertEqual(plan["arms"], list(ARMS))
            self.assertEqual(plan["format_policy"]["max_writer_attempts_per_proposal"], 2)
            self.assertFalse(set(plan["task_ids"]) & (ORIGINAL_TASK_IDS | set(plan["development_smoke_task_ids"])))
            self.assertEqual(len(seen), 4)
            self.assertTrue(all(call["model"] == "openai/" + MODEL for call in seen))
            self.assertTrue(all(call["api_base"] == "http://127.0.0.1:8001/v1" for call in seen))
            self.assertTrue(all(call["num_retries"] == 0 for call in seen))
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["termination_reason"] == "max_steps" for row in rows))
            with self.assertRaisesRegex(ValueError, "already contains"):
                run(agents, Gate(), out, limit=1, max_steps=2,
                    upstream_root=os.environ["HALTIQ_TAU_UPSTREAM"])


if __name__ == "__main__":
    unittest.main()
