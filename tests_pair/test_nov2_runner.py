"""Synthetic end-to-end runner, branch accounting and interruption controls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from haltiq_pair import hotpot, nov2_protocol, nov2_runner as runner
from haltiq_pair.nov2_policy import PolicyConfig


def row(index, answer="Paris"):
    return {"id": f"{index:024x}", "question": f"Where was author {index} born?",
        "type": "bridge", "level": "hard", "answer": answer,
        "context": [["Biography", ["The author was born in Paris."]], ["Other", ["Unrelated evidence."]]],
        "supporting_facts": [["Biography", 0]]}


class FakeAgents:
    provenance = {"execution": "injected synthetic double", "roles": ["writer", "reviewer", "challenger"]}

    def __init__(self, *, receipt_root=None, fail_at=None, bad_first=False):
        self.usage, self.prompts = [], []
        self.receipt_root = receipt_root
        self.fail_at, self.bad_first = fail_at, bad_first
        self.test_questions, self.output = set(), None
        self.frozen_before_test = []

    def generate(self, role, system, prompt):
        self.prompts.append({"role": role, "system": system, "prompt": prompt})
        number = len(self.prompts)
        if any(question in prompt for question in self.test_questions):
            freeze = json.loads((self.output / "freeze.json").read_text())
            self.frozen_before_test.append(freeze)
            for head in freeze["checkpoints"]:
                assert runner._sha(self.output / head["path"]) == head["sha256"]
        call_id, response_id = f"{number:032x}", f"chatcmpl-{number:032x}"
        usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        if self.receipt_root is not None:
            self.receipt_root.mkdir(parents=True, exist_ok=True)
            primary = self.receipt_root.parent / "requests" / (response_id + ".json")
            primary.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"yash_call_id": call_id, "yash_role": role}
            primary.write_text(json.dumps({"id": response_id, "usage": usage,
                "finish_reason": "stop", "request": {"metadata": metadata}}), encoding="utf-8")
            (self.receipt_root / (call_id + ".json")).write_text(json.dumps({
                "call_id": call_id, "role": role, "response_id": response_id,
                "primary_receipt": "requests/" + primary.name,
                "primary_sha256": runner._sha(primary), "finish_reason": "stop",
                "metadata": metadata, "usage": usage}), encoding="utf-8")
        failed = number == self.fail_at
        self.usage.append({"role": role, "call_id": call_id,
                           "status": "failed" if failed else "completed", "usage": {} if failed else usage})
        if failed:
            raise RuntimeError("Synthetic transport failure after a completed model response")
        if self.bad_first and number == 1:
            return "invalid JSON"
        if role == "writer":
            return json.dumps({"answer": "Paris", "supporting_facts": [["Biography", 0]]})
        if role == "reviewer":
            return json.dumps({"approved": True, "feedback": "Supported by the public biography."})
        return json.dumps({"alternative": "Paris", "evidence": [["Biography", 0]],
                           "feedback": "Independent evidence supports Paris; verify the cited sentence."})


class FakeCritic:
    provenance = {"execution": "synthetic verifier double"}

    def __init__(self, score=.95):
        self.value, self.calls, self.last_record = score, [], None
        self.context, self.contexts = {}, []

    def score(self, question, passages, answer):
        self.calls.append((question, passages, answer))
        self.contexts.append(dict(self.context))
        self.last_record = {"physical_critic_calls": 1, "physical_reranker_forwards": 1, "p_true": self.value}
        return self.value


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(index) for index in range(20)]
        self.plan = nov2_protocol.build_plan(self.rows, (), train_count=2, dev_count=1, test_count=2)
        self.config = PolicyConfig(epochs=4)

    def run_fake(self, output, agents=None, critic=None, **kwargs):
        agents, critic = agents or FakeAgents(), critic or FakeCritic()
        agents.output = Path(output)
        test = nov2_protocol.resolve_plan(self.plan, self.rows)["test"]
        agents.test_questions = {public.question for public, _ in test}
        summary = runner.run_study(agents, critic, self.plan, self.rows, output,
            role_agents=agents, policy_config=self.config, **kwargs)
        return summary, agents, critic

    def test_full_fake_study_freezes_five_heads_before_same_questions_all_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            summary, agents, critic = self.run_fake(output)
            self.assertEqual(summary["study_status"], "completed")
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["all_episodes_succeeded"])
            self.assertEqual(summary["deployment_seed"], 0)
            self.assertEqual({item["seed"] for item in summary["fitted_checkpoints"]}, set(range(5)))
            self.assertTrue(agents.frozen_before_test)
            self.assertTrue(all(value["test_calls_at_freeze"] == 0 for value in agents.frozen_before_test))
            episodes = runner._read(output / "episodes.jsonl")
            expected_ids = set(self.plan["partitions"]["test"]["question_ids"])
            self.assertEqual(len(episodes), 14)
            for arm in runner.ARMS:
                rows = [item for item in episodes if item["arm"] == arm]
                self.assertEqual({item["question_id"] for item in rows}, expected_ids)
                self.assertEqual(summary["arms"][arm]["metrics"]["joint_f1"], 1.)
            for name in ("freeze.json", "summary.json", "results.csv", "Yash-Nov2-Study-Report.txt",
                         "Yash-Nov2-Answer-F1.png", "status.json"):
                self.assertTrue((output / name).is_file(), name)
            from PIL import Image
            with Image.open(output / "Yash-Nov2-Answer-F1.png") as chart:
                self.assertEqual(chart.size, (1300, 720))
            self.assertEqual(json.loads((output / "status.json").read_text())["recorded_test_attempts"], 14)
            self.assertFalse(summary["best_prefix_is_primary"])
            analysis = summary["policy_analysis"]
            self.assertEqual(len(analysis["seed_dev_checkpoint_selection"]), 5)
            self.assertEqual(analysis["training_label_distribution"]["train"]["states"], 6)
            self.assertEqual(analysis["inherited_head_vs_strands"]["AND_vs_strands_disagreements"], 0)
            self.assertTrue(all(value["both_panels_fully_attempted"] for value in analysis["learned_vs_matched_controls"].values()))
            self.assertTrue(all(value["stage"] in {"train", "dev", "test"} and value["question_id"] for value in critic.contexts))
            self.assertTrue(all("arm" in value for value in critic.contexts if value["stage"] == "test"))
            fixed1 = [item for item in episodes if item["arm"] == "fixed1"]
            self.assertTrue(all(item["attempted_agent_calls"] == 1 and item["critic_calls"] == 0 for item in fixed1))
            self.assertTrue(all(not item["gate_approved"] for item in fixed1))
            self.assertEqual(sum(item["critic_calls"] for item in episodes) + summary["collection"]["physical_critic_calls"], len(critic.calls))

    def test_shared_prefixes_are_not_charged_twice_as_physical_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            summary, agents, _ = self.run_fake(output)
            self.assertEqual(summary["collection"]["physical_agent_invocations"], 3 * 14)
            self.assertEqual(summary["collection"]["branch_outcomes"], 3 * 8)
            branches = runner._read(output / "training-branches.jsonl")
            self.assertEqual(sum(item["attempted_agent_calls"] for item in branches), 3 * 12)
            self.assertTrue(all(item["attempted_agent_calls"] == 0 for item in branches if item["action"] == "STOP"))
            revisions = [item for item in branches if item["state_id"] == "initial/REVISE" and item["action"] == "REVISE"]
            self.assertTrue(all(item["child_path_calls"] == 6 and item["attempted_agent_calls"] == 2 for item in revisions))
            for item in branches:
                self.assertAlmostEqual(item["utility"], item["metrics"]["f1"] - .02 * item["attempted_agent_calls"] / 10)
            episodes = runner._read(output / "episodes.jsonl")
            self.assertEqual(len(agents.prompts), summary["collection"]["physical_agent_invocations"] + sum(item["attempted_agent_calls"] for item in episodes))
            states = runner._read(output / "training-examples.jsonl")
            self.assertEqual(len(states), 9)
            for state in states:
                self.assertEqual(set(state["utilities"]), set(state["available_actions"]))
            test_ids = set(self.plan["partitions"]["test"]["question_ids"])
            self.assertFalse(test_ids & {item["question_id"] for item in states})

    def test_public_agents_never_receive_gold_and_budget_caps_never_approve(self):
        self.rows = [dict(item, answer="SECRET GOLD ONLY") for item in self.rows]
        self.plan = nov2_protocol.build_plan(self.rows, (), train_count=2, dev_count=1, test_count=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            summary, agents, critic = self.run_fake(output, critic=FakeCritic(score=.1))
            self.assertTrue(all("SECRET GOLD ONLY" not in item["prompt"] + item["system"] for item in agents.prompts))
            self.assertNotIn("SECRET GOLD ONLY", json.dumps(critic.calls))
            self.assertEqual(summary["arms"]["learned3"]["metrics"]["f1"], 0)
            episodes = runner._read(output / "episodes.jsonl")
            caps = [item for item in episodes if item["termination"] == "budget_cap"]
            self.assertTrue(caps)
            self.assertTrue(all(not item["gate_approved"] for item in caps))
            revised = [item for item in episodes if item["arm"] == "always_revise3"]
            self.assertTrue(all(item["returned_draft_number"] == 3 and item["iterations"] == 3 for item in revised))
            invoked = [item for item in episodes if item["arm"] == "always_invoke"]
            self.assertTrue(all(item["agent_calls"]["challenger"] == 1 for item in invoked))

    def test_failed_initial_state_kept_in_collection_coverage_without_imputation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            summary, _, _ = self.run_fake(output, agents=FakeAgents(fail_at=1))
            coverage = summary["collection"]["initial_question_coverage"]["train"]
            self.assertEqual((coverage["planned"], coverage["attempted"], coverage["valid_initial_states"], coverage["failed"]), (2, 2, 1, 1))
            branches = runner._read(output / "training-branches.jsonl")
            failed = [item for item in branches if item["action"] == "INITIAL"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(set(failed[0]["metrics"].values()), {0.})
            self.assertIsNone(summary["collection"]["physical_tokens"])

    def test_failed_sdk_call_can_have_known_physical_response_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            agents = FakeAgents(receipt_root=output / "runtime/writer/by-client", fail_at=1)
            summary, agents, _ = self.run_fake(output, agents=agents, data_mode="pinned_dataset_live")
            calls = runner._read(output / "physical-calls.jsonl")
            self.assertEqual(calls[0]["status"], "failed")
            self.assertIsNone(calls[0]["sdk_reported_tokens"])
            self.assertEqual(calls[0]["total_tokens"], 13)
            self.assertEqual(calls[0]["response_receipt"]["status"], "reconciled")
            self.assertEqual(summary["journal_integrity"]["unknown_response_token_receipts"], 0)
            self.assertTrue(summary["journal_integrity"]["all_response_tokens_reconciled"])
            self.assertEqual(summary["collection"]["physical_tokens"], 13 * summary["collection"]["physical_agent_invocations"])

    def test_timeout_keeps_all_unattempted_tasks_pending_and_claim_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            # Force the boundary deterministically rather than racing a tiny wall-clock duration.
            with patch.object(runner._Ledger, "check", side_effect=runner.StudyTimeout("synthetic deadline")):
                summary, agents, _ = self.run_fake(output)
            self.assertEqual(summary["study_status"], "interrupted")
            self.assertFalse(summary["complete"])
            self.assertTrue(all(value["answer_f1_delta_learned_minus_control"] is None
                for value in summary["policy_analysis"]["learned_vs_matched_controls"].values()))
            self.assertEqual(len(agents.prompts), 0)
            self.assertTrue(all(value["pending"] == 2 and value["total_tokens"] is None for value in summary["arms"].values()))
            self.assertEqual(runner._read(output / "episodes.jsonl"), [])
            with self.assertRaises(FileExistsError):
                self.run_fake(output, agents=agents)
            self.assertEqual(len(agents.prompts), 0)

    def test_expired_monotonic_budget_refuses_next_call(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = runner._Ledger(Path(directory), runner.time.monotonic() - 1., "injected_rows_offline_verification")
            with self.assertRaises(runner.StudyTimeout):
                ledger.check()

    def test_plan_drift_and_invalid_seed_refused_before_any_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            bad = deepcopy(self.plan)
            bad["partitions"]["test"]["public_inputs_sha256"] = "changed"
            agents = FakeAgents()
            with self.assertRaises(ValueError):
                runner.run_study(agents, FakeCritic(), bad, self.rows, output, role_agents=agents)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                self.run_fake(output, seed=4)
            self.assertEqual(agents.prompts, [])

    def test_primary_reader_rejects_malformed_json_not_silent_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_bytes(b'{"status":"completed"}\nnot-json\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                runner._read(path)


class CorrelatedReceiptTests(unittest.TestCase):
    def test_missing_mismatched_and_duplicate_receipts_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = runner._Ledger(root, float("inf"), "pinned_dataset_live")
            agents = FakeAgents(receipt_root=root / "by-client")
            entry = {"role": "writer", "status": "completed", "usage": {"total_tokens": 999}, "call_id": "missing"}
            self.assertIsNone(ledger.receipt(entry, agents)["tokens"])
            agents.generate("writer", "", "synthetic")
            actual = agents.usage[-1]
            value = ledger.receipt(actual, agents)
            self.assertEqual(value["tokens"], 13)
            self.assertIsNone(ledger.receipt(actual, agents)["tokens"])
            # A mismatched primary response cannot be replaced by SDK estimates.
            agents.generate("writer", "", "synthetic")
            actual = agents.usage[-1]
            primary = root / ("requests/chatcmpl-" + f"{2:032x}" + ".json")
            primary.write_text(json.dumps({"id": "wrong", "usage": {"total_tokens": 13}}))
            self.assertIsNone(ledger.receipt(actual, agents)["tokens"])

    def test_correlated_receipt_rejects_role_metadata_sha_usage_and_finish_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = runner._Ledger(root, float("inf"), "pinned_dataset_live")
            agents = FakeAgents(receipt_root=root / "by-client")
            for mutation in ("client_role", "client_metadata", "raw_metadata", "primary_sha",
                             "client_finish", "raw_finish", "client_usage", "raw_usage"):
                agents.generate("writer", "", "synthetic")
                entry = agents.usage[-1]
                client_path = agents.receipt_root / (entry["call_id"] + ".json")
                client = json.loads(client_path.read_text())
                primary = root / client["primary_receipt"]
                physical = json.loads(primary.read_text())
                if mutation == "client_role":
                    client["role"] = "reviewer"
                elif mutation == "client_metadata":
                    client["metadata"]["yash_role"] = "reviewer"
                elif mutation == "raw_metadata":
                    physical["request"]["metadata"]["yash_call_id"] = "f" * 32
                elif mutation == "primary_sha":
                    client["primary_sha256"] = "0" * 64
                elif mutation == "client_finish":
                    client["finish_reason"] = "length"
                elif mutation == "raw_finish":
                    physical["finish_reason"] = "length"
                elif mutation == "client_usage":
                    client["usage"].update(prompt_tokens=9, completion_tokens=4)
                else:
                    physical["usage"].update(prompt_tokens=9, completion_tokens=4)
                if mutation.startswith("raw_"):
                    primary.write_text(json.dumps(physical), encoding="utf-8")
                    client["primary_sha256"] = runner._sha(primary)
                client_path.write_text(json.dumps(client), encoding="utf-8")
                with self.subTest(mutation=mutation):
                    self.assertIsNone(ledger.receipt(entry, agents)["tokens"])


if __name__ == "__main__":
    unittest.main()
