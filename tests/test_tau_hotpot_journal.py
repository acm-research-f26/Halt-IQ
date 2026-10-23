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


if __name__ == "__main__":
    unittest.main()
