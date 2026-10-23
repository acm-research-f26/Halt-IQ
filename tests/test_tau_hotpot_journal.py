import json
from pathlib import Path
import tempfile
import unittest

from haltiq_tau_hotpot.journal import CallJournal, UncertainCall


class PhysicalJournalTests(unittest.TestCase):
    def test_pending_is_durable_before_dispatch_and_completion_binds_result(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = CallJournal(temp)
            def operation():
                pending = json.loads(journal.path.read_text().splitlines()[0])
                self.assertEqual(pending["event"], "pending")
                return {"actual": 1}
            self.assertEqual(journal.call("writer", {"public": "question"}, operation), {"actual": 1})
            self.assertFalse(journal.verify()["pending_or_uncertain"])
            record = journal.records[-1]
            (Path(temp) / record["response"]).write_text("{}")
            with self.assertRaises(ValueError):
                journal.verify()

    def test_uncertain_call_stops_and_is_never_implicitly_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = CallJournal(temp)
            calls = []
            def operation():
                calls.append(1)
                raise TimeoutError("no terminal response")
            with self.assertRaises(UncertainCall):
                journal.call("writer", {"x": 1}, operation)
            self.assertEqual(calls, [1])
            with self.assertRaises(UncertainCall):
                CallJournal(temp)

    def test_truncated_journal_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "calls.jsonl").write_text('{"event":')
            with self.assertRaises(ValueError):
                CallJournal(temp)

    def test_exact_reuse_is_bound_to_completed_request_and_counts_zero_new_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = CallJournal(temp)
            request = {"payload": "unchanged", "seed": 42}
            original = journal.call("writer", request, lambda: {"answer": "fixture"})
            completed = dict(journal.records[-1])
            self.assertEqual(journal.reuse(completed, request), original)
            self.assertEqual(journal.verify()["by_role"], {"writer": 1})
            self.assertEqual(journal.summary()["exact_response_reuses"], 1)
            reopened = CallJournal(temp)
            self.assertEqual(reopened.verify(), journal.verify())
            with self.assertRaisesRegex(ValueError, "changed"):
                journal.reuse(completed, {**request, "seed": 43})
            self.assertEqual(journal.summary()["exact_response_reuses"], 1)

    def test_forged_cached_event_and_tampered_response_are_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = CallJournal(temp)
            journal.call("reviewer", {"x": 1}, lambda: {"feedback": "fixture"})
            event = dict(journal.records[-1])
            with self.assertRaises(UncertainCall):
                journal.reuse({**event, "role": "writer"}, {"x": 1})
            (Path(temp) / event["response"]).write_text("{}")
            with self.assertRaises(ValueError):
                journal.reuse(event, {"x": 1})
            with self.assertRaises(ValueError):
                CallJournal(temp)

    def test_caught_uncertain_exception_cannot_enable_another_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = CallJournal(temp)
            def failure():
                raise TimeoutError("response unknown")
            with self.assertRaises(UncertainCall):
                journal.call("writer", {}, failure)
            calls = []
            with self.assertRaises(UncertainCall):
                journal.call("writer", {}, lambda: calls.append(1))
            self.assertFalse(calls)

    def test_unbound_reuse_event_is_rejected_on_read(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = CallJournal(temp)
            journal.call("writer", {}, lambda: {"fixture": 1})
            event = journal.records[-1]
            journal._append({"event": "exact_response_reuse", "id": event["id"], "role": "writer",
                "request_sha256": "0" * 64, "response_sha256": event["response_sha256"],
                "new_physical_calls": 0})
            with self.assertRaisesRegex(ValueError, "Unbound"):
                CallJournal(temp)


if __name__ == "__main__":
    unittest.main()
