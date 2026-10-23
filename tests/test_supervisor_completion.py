"""Completion proof and phase-only intervention boundaries, without model calls."""
from pathlib import Path
import unittest
from unittest.mock import patch

from haltiq_supervisor.engine import run, verify_completion, wait_owned_tree_clear
from pc_strands.campaign import load, save
from tests import test_supervisor_engine as fixtures


class CompletionTests(unittest.TestCase):
    def test_all_planned_failures_count_but_unknown_work_or_wrong_owner_never_passes(self):
        stage = {"id": "evaluate", "planned_units": 20}
        state = {"run_id": "current", "plan_sha256": "expected"}
        receipt = {"run_id": "current", "plan_sha256": "expected", "phase": "evaluate",
            "terminal": True, "pending_or_uncertain": False, "cleanup_verified": True,
            "planned_units_accounted": True, "planned_units": 20,
            "completed_units": 17, "failed_units": 2, "not_run_units": 1}
        verify_completion(stage, state, receipt)
        for changes in ({"run_id": "old"}, {"plan_sha256": "old"}, {"phase": "train"},
                        {"pending_or_uncertain": True}, {"cleanup_verified": False}, {"terminal": 1},
                        {"planned_units_accounted": False}, {"completed_units": 19},
                        {"failed_units": -1}, {"not_run_units": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verify_completion(stage, state, {**receipt, **changes})

    def test_windows_exit_accounting_may_drain_but_true_descendants_remain_blocked(self):
        clock = {"now": 0}
        def sleep(seconds):
            clock["now"] += seconds
        with patch("haltiq_supervisor.engine.os.name", "nt"), \
             patch("haltiq_supervisor.engine.time.monotonic", side_effect=lambda: clock["now"]), \
             patch("haltiq_supervisor.engine.descendants_remain", side_effect=[True, True, False]):
            self.assertTrue(wait_owned_tree_clear(None, None, hard_monotonic=100, sleep=sleep))
        self.assertGreater(clock["now"], 0)
        self.assertLess(clock["now"], 2)
        for hard, stop, expected in ((100, None, 2), (.3, None, .3), (100, .15, .15)):
            clock["now"] = 0
            with self.subTest(hard=hard, stop=stop), patch("haltiq_supervisor.engine.os.name", "nt"), \
                 patch("haltiq_supervisor.engine.time.monotonic", side_effect=lambda: clock["now"]), \
                 patch("haltiq_supervisor.engine.descendants_remain", return_value=True):
                self.assertFalse(wait_owned_tree_clear(None, None, hard_monotonic=hard,
                                                      stop_deadline=stop, sleep=sleep))
            self.assertAlmostEqual(clock["now"], expected)


class PhaseCallbackTests(unittest.TestCase):
    def setUp(self):
        # Reuse the real frozen fixture's setup helpers, without inheriting/rerunning its suite.
        self.fixture = fixtures.FrozenEngineTests(methodName="test_two_phase_autochain_uses_frozen_source_and_never_starts_codex")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def controller(self, callback):
        fixture = self.fixture
        with patch("haltiq_supervisor.engine.__file__", str(fixture.directory / "source/haltiq_supervisor/engine.py")), \
             patch("haltiq_supervisor.engine.emit_handoff", side_effect=callback):
            return run(fixture.directory, poll_seconds=.01)

    def test_one_completed_review_can_approve_next_registered_phase_without_polling_codex(self):
        fixture = self.fixture
        fixture.plan["stages"] = [fixture.step("first", review=True), fixture.step("second")]
        fixture.adopt()
        calls = []
        def callback(directory, **event):
            calls.append(event["kind"])
            if event["kind"] == "action_required":
                state = load(event["state_path"])
                request = state["review_request"]
                Path(request["approval_path"]).parent.mkdir()
                save(Path(request["approval_path"]), request["expected_payload"])
            return {"status": "completed"}
        state = self.controller(callback)
        self.assertEqual(state["status"], "completed")
        self.assertTrue((fixture.directory / "second.json").is_file())
        self.assertEqual(calls.count("action_required"), 1)
        self.assertEqual(calls.count("final_completed"), 1)
        self.assertNotIn("review_request", state)

    def test_notification_failure_preserves_completed_science_and_records_separate_error(self):
        fixture = self.fixture
        fixture.plan["stages"] = [fixture.step("first")]
        fixture.adopt()
        state = self.controller(lambda *args, **kwargs: (_ for _ in ()).throw(OSError("notification unavailable")))
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["stages"]["first"]["status"], "completed")
        self.assertEqual(len(state["notification_errors"]), 2)


if __name__ == "__main__":
    unittest.main()
