"""Mock HTTP protocol/accounting tests, never hosted-model benchmark results."""
import json
import getpass
from pathlib import Path
import tempfile
import time
import unittest
import warnings
from unittest.mock import patch

from haltiq.budget import BudgetLedger
from haltiq.providers import JevClient, ProviderError
from pc_strands import evaluate as core
from pc_strands.data import Task, fingerprint
from pc_strands.decider import CORRECTNESS_CRITERIA, CORRECTNESS_INSTRUCTIONS, sha256_file
from pc_strands.jev_replay import HostedJevDecider, hidden_key_entry, read_existing_ledger, run_matched_jev


class NoWaitGuard:
    idle_seconds = 0
    deadline = time.time() + 60
    def before(self): pass
    def after(self, _): pass


def fixture(root):
    tasks = [Task("hotpotqa:q1", "Who?", ["public evidence"], ["Alpha"], "toy", "confirmation"),
             Task("hotpotqa:q2", "Which?", ["more public evidence"], ["Beta"], "toy", "confirmation")]
    source, reference = root / "generation", root / "reference"
    source.mkdir(); reference.mkdir()
    dataset = root / "tasks.json"
    serialized = [{**t.__dict__, "local_partition": "confirmation"} for t in tasks]
    dataset.write_text(json.dumps(serialized))
    snapshots = [dict(t.__dict__) for t in tasks]
    (source / "tasks-evaluator-only.json").write_text(json.dumps(snapshots))
    trajectories = []
    for index, task in enumerate(tasks):
        rounds = []
        for number in range(1, 11):
            answer = "wrong" if index == 0 and number == 1 else task.answers[0]
            call = {"model": "qwen3:8b", "input_tokens": 100, "output_tokens": 4,
                    "elapsed_seconds": .1, "cost_usd": 0., "response": {"model": "qwen3:8b",
                    "done": True, "done_reason": "stop", "message": {"content": json.dumps({"answer": answer})}}}
            rounds.append({"round": number, "status": "completed", "draft": answer,
                           "call": call, **core.evaluate(answer, task.answers)})
        trajectories.append({"task_id": task.id, "status": "completed", "rounds": rounds})
    (source / "trajectories.json").write_text(json.dumps(trajectories))
    (source / "manifest.json").write_text(json.dumps({"status": "completed", "writer_unload_verified": True}))
    config = {"phase": "confirmation", "mode": "head-replay", "policy_name": "strands-tuned",
              "threshold": .8, "max_drafts": 10, "temperature": None, "seed": 42,
              "feedback": core.GENERIC_FEEDBACK, "trajectory_sha256": sha256_file(source / "trajectories.json"),
              "dataset_sha256": sha256_file(dataset)}
    (reference / "manifest.json").write_text(json.dumps({"status": "completed", "frozen": {
        "config": config, "task_ids": [t.id for t in tasks], "dataset_state_sha256": fingerprint(snapshots)}}))
    return reference, source, dataset


def response(p):
    return {"model": "jev-1.13.0", "answers": {"correct": {"type": "noul", "noul": p}},
            "usage": {"input_tokens": 100, "output_tokens": 2}}


class JevReplayTests(unittest.TestCase):
    def test_getpass_echo_fallback_stops_before_key_read_or_network(self):
        key_read = []
        def unsafe_fallback(_prompt):
            warnings.warn("mock noninteractive fallback", getpass.GetPassWarning)
            key_read.append(True)  # Real getpass reads stdin only after this warning.
            return "never-read-mock-value"
        with patch("getpass.getpass", side_effect=unsafe_fallback), patch("haltiq.providers._http_json") as http:
            with self.assertRaisesRegex(RuntimeError, "No key was read or sent"):
                hidden_key_entry()
            self.assertEqual(key_read, [])
            http.assert_not_called()

    def test_adapter_uses_exact_shared_single_question_and_rejects_private_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.sqlite", limit_usd=4.5)
            adapter = HostedJevDecider(JevClient(api_key="mock-only-key", budget=ledger))
            public = {"question": "Who?", "evidence": ["unchanged"], "draft": "Alpha"}
            with patch("haltiq.providers._http_json", return_value=response(.9)) as http:
                result = adapter.score(public)
                payload = json.loads(http.call_args.args[1])
                self.assertEqual(payload["state"], public)
                self.assertEqual(list(payload["questions"]), ["correct"])
                self.assertEqual(payload["questions"]["correct"]["instructions"], CORRECTNESS_INSTRUCTIONS)
                self.assertEqual(payload["questions"]["correct"]["criteria"], CORRECTNESS_CRITERIA)
                self.assertIsNone(result["logits"])
                with self.assertRaises(ValueError): adapter.score({**public, "gold": "private"})
                self.assertEqual(http.call_count, 1)

    def test_shared_ledger_no_reset_no_log_credentials_and_all_role_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference, source, dataset = fixture(root)
            ledger = BudgetLedger(root / "ledger.sqlite", limit_usd=4.5)
            prior = ledger.reserve(); ledger.settle(prior, 1000)
            before = read_existing_ledger(ledger.path)
            client = JevClient(api_key="mock-only-key", budget=ledger)
            output = root / "optional"
            with patch("haltiq.providers._http_json", side_effect=[response(.2), response(.9), response(.9)]) as http:
                run_matched_jev(reference, source, dataset, output, client, NoWaitGuard())
                self.assertEqual(http.call_count, 3)
            after = read_existing_ledger(ledger.path)
            self.assertEqual(after["attempts"], before["attempts"] + 3)
            self.assertEqual(after["unsettled"], 0)
            self.assertAlmostEqual(after["used_usd"] - before["used_usd"], 3 * .0000042)
            rows = json.loads((output / "episodes.json").read_text())
            primary = [r for r in rows if r["policy"] == "jev-1.13-single-correctness"]
            self.assertEqual(sum(r["decision_output_tokens"] for r in primary), 6)
            self.assertAlmostEqual(sum(r["cost_usd"] for r in primary), 3 * .0000042)
            self.assertTrue(any(r["policy"].endswith("-cap3") for r in rows))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["actual_evaluation_usage"]["reported_output_tokens"], 6)
            self.assertAlmostEqual(manifest["actual_evaluation_usage"]["hosted_cost_usd"], 3 * .0000042)
            summary = json.loads((output / "summary.json").read_text())
            p = summary["jev-1.13-single-correctness"]
            self.assertEqual(p["mean_raw_reported_tokens"], (3 * 104 + 3 * 102) / 2)
            text = ''.join(p.read_text() for p in output.rglob('*.json*'))
            self.assertNotIn("mock-only-key", text)
            self.assertNotIn("Authorization", text)
            self.assertIn('"logits": null', text)

    def test_uncertain_failure_keeps_reservation_and_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference, source, dataset = fixture(root)
            ledger = BudgetLedger(root / "ledger.sqlite", limit_usd=4.5)
            client = JevClient(api_key="mock-only-key", budget=ledger)
            output = root / "optional"
            with patch("haltiq.providers._http_json", side_effect=ProviderError("mock timeout")) as http:
                with self.assertRaises(core.UncertainRequest):
                    run_matched_jev(reference, source, dataset, output, client, NoWaitGuard())
                self.assertEqual(http.call_count, 1)
                self.assertEqual(ledger.snapshot()["unsettled_requests"], 1)
                with self.assertRaises(RuntimeError):
                    run_matched_jev(reference, source, dataset, output, client, NoWaitGuard(), resume=True)
                self.assertEqual(http.call_count, 1)

    def test_tiny_remaining_budget_stops_before_http_or_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference, source, dataset = fixture(root)
            ledger = BudgetLedger(root / "ledger.sqlite", limit_usd=.001)
            client = JevClient(api_key="mock-only-key", budget=ledger)
            with patch("haltiq.providers._http_json") as http:
                with self.assertRaises(core.UncertainRequest):
                    run_matched_jev(reference, source, dataset, root / "optional", client, NoWaitGuard())
                http.assert_not_called()
            self.assertEqual(ledger.snapshot()["attempts"], 0)

    def test_replay_existing_success_uses_journal_without_paying_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference, source, dataset = fixture(root)
            ledger = BudgetLedger(root / "ledger.sqlite", limit_usd=4.5)
            client = JevClient(api_key="mock-only-key", budget=ledger)
            output = root / "optional"
            with patch("haltiq.providers._http_json", side_effect=[response(.9), response(.9)]) as http:
                run_matched_jev(reference, source, dataset, output, client, NoWaitGuard())
                first = (output / "episodes.json").read_bytes()
                old_files = {str(p): p.read_bytes() for p in output.rglob('*') if p.is_file()}
                with self.assertRaises(ValueError):
                    run_matched_jev(reference, source, dataset, output, client, NoWaitGuard())
                self.assertEqual(old_files, {str(p): p.read_bytes() for p in output.rglob('*') if p.is_file()})
                run_matched_jev(reference, source, dataset, output, client, NoWaitGuard(), resume=True)
                self.assertEqual(http.call_count, 2)
                self.assertEqual(first, (output / "episodes.json").read_bytes())
                self.assertEqual(ledger.snapshot()["attempts"], 2)

    def test_missing_ledger_is_not_created_and_mismatched_dataset_makes_no_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError): read_existing_ledger(root / "missing.sqlite")
            self.assertFalse((root / "missing.sqlite").exists())
            reference, source, dataset = fixture(root)
            data = json.loads(dataset.read_text()); data[0]["evidence"] = ["changed"]; dataset.write_text(json.dumps(data))
            ledger = BudgetLedger(root / "ledger.sqlite", limit_usd=4.5)
            client = JevClient(api_key="mock-only-key", budget=ledger)
            with patch("haltiq.providers._http_json") as http, self.assertRaises(ValueError):
                run_matched_jev(reference, source, dataset, root / "optional", client, NoWaitGuard())
            http.assert_not_called()
            self.assertEqual(ledger.snapshot()["attempts"], 0)

    def test_expired_deadline_stops_before_any_billable_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference, source, dataset = fixture(root)
            ledger = BudgetLedger(root / "ledger.sqlite", limit_usd=4.5)
            client = JevClient(api_key="mock-only-key", budget=ledger)
            guard = NoWaitGuard(); guard.deadline = time.time() - 1
            with patch("haltiq.providers._http_json") as http, self.assertRaises(core.UncertainRequest):
                run_matched_jev(reference, source, dataset, root / "optional", client, guard)
            http.assert_not_called()
            self.assertEqual(ledger.snapshot()["attempts"], 0)


if __name__ == "__main__":
    unittest.main()
