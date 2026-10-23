"""At-most-once handoff controls; no actual Codex/model/provider invocation."""
from __future__ import annotations

import json
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from haltiq_supervisor.handoff import _kill_owned_group, emit_handoff, validate_callback


THREAD = "12345678-1234-1234-1234-123456789abc"


class Containment:
    def __init__(self):
        self.attached, self.closed = [], False
    def attach(self, pid):
        self.attached.append(pid)
    def close(self):
        self.closed = True


class Child:
    def __init__(self, returncode=0, timeout=False):
        self.pid, self.returncode, self.timeout = 987, None, timeout
        self.final_code, self.killed = returncode, False
        self.input = None
    def communicate(self, input, timeout):
        self.input = input
        if self.timeout:
            raise subprocess.TimeoutExpired("private-command-is-not-logged", timeout)
        self.returncode = self.final_code
    def poll(self):
        return self.returncode
    def kill(self):
        self.killed, self.returncode = True, -9
    def wait(self, timeout):
        return self.returncode


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "state.json"
        self.state.write_text(json.dumps({"private_model_output": "SENSITIVE-CANARY-NOT-IN-HANDOFF"}))
        self.exe = self.root / "codex.exe"
        self.exe.write_bytes(b"not-executable-never-launched")
        self.argv = [str(self.exe), "exec", "resume", THREAD, "-"]
        self.kwargs = {"event_id": "phase-0001", "kind": "phase_completed",
                       "stage": "collect_examples", "state_path": self.state}
        # Fake bootstrap PIDs must never be sent to a real POSIX killpg.
        group_patch = patch("haltiq_supervisor.handoff._kill_owned_group")
        self.group_kill = group_patch.start()
        self.addCleanup(group_patch.stop)

    def receipt(self):
        return json.loads((self.root / "outbox/phase-0001.json").read_text())

    def command(self):
        return json.loads((self.root / "outbox/commands/phase-0001.json").read_text())

    def test_local_only_is_exclusively_saved_and_fsynced_without_subprocess(self):
        import os
        with patch("haltiq_supervisor.handoff.os.fsync", wraps=os.fsync) as synced, \
             patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            result = emit_handoff(self.root, **self.kwargs)
        self.assertEqual(result["status"], "prepared_local_only")
        self.assertEqual(result["dispatch_intents"], 0)
        self.assertFalse(result["dispatch_attempted"])
        self.assertEqual(self.receipt()["state_path"], str(self.state))
        self.assertGreaterEqual(synced.call_count, 1)
        popen.assert_not_called()

    def test_local_only_export_cwd_saves_and_deduplicates_without_git_or_dispatch(self):
        import os
        export = self.root / "frozen-source-export"
        export.mkdir()
        self.assertFalse((export / ".git").exists())
        with patch("haltiq_supervisor.handoff.os.fsync", wraps=os.fsync) as synced, \
             patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_cwd=export)
            before = (self.root / "outbox/phase-0001.json").read_bytes()
            again = emit_handoff(self.root, **self.kwargs, codex_cwd=export)
        self.assertEqual(result["status"], "prepared_local_only")
        self.assertEqual(result["dispatch_intents"], 0)
        self.assertFalse(result["dispatch_attempted"])
        self.assertFalse(result["live_delivery_verified"])
        self.assertTrue(again["deduplicated"])
        self.assertEqual(before, (self.root / "outbox/phase-0001.json").read_bytes())
        self.assertEqual(synced.call_count, 1)
        self.assertFalse((self.root / "outbox/commands").exists())
        self.assertNotIn("SENSITIVE-CANARY", before.decode("utf-8"))
        popen.assert_not_called()

    def test_configured_callback_export_cwd_fails_before_outbox_creation(self):
        export = self.root / "frozen-source-export"
        export.mkdir()
        with patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "absolute Git checkout"):
                emit_handoff(self.root, **self.kwargs, codex_argv=self.argv, codex_cwd=export)
        self.assertFalse((self.root / "outbox").exists())
        popen.assert_not_called()

    def test_prior_local_event_does_not_dispatch_when_callback_is_later_enabled(self):
        emit_handoff(self.root, **self.kwargs)
        before = (self.root / "outbox/phase-0001.json").read_bytes()
        with patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertTrue(result["deduplicated"])
        self.assertEqual(before, (self.root / "outbox/phase-0001.json").read_bytes())
        popen.assert_not_called()

    def test_success_has_prior_durable_pending_intent_and_no_sensitive_payload(self):
        child, owned = Child(), Containment()
        def dispatch(argv, **kwargs):
            pending = self.receipt()
            self.assertEqual(pending["status"], "pending")
            self.assertEqual(pending["dispatch_intents"], 1)
            self.assertEqual(Path(argv[2]).name, "worker.py")
            self.assertEqual(Path(argv[3]), self.root / "outbox/commands/phase-0001.json")
            self.assertEqual(self.command()["argv"], self.argv)
            self.assertIsNone(child.input)
            self.assertFalse(kwargs["shell"])
            self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            self.assertEqual(kwargs["start_new_session"], __import__("os").name != "nt")
            return child
        actual_communicate = child.communicate
        def released(input, timeout):
            self.assertEqual(owned.attached, [987])
            self.assertEqual(input, "GO\n")
            return actual_communicate(input, timeout)
        with patch("haltiq_supervisor.handoff.ProcessContainment", return_value=owned), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", side_effect=dispatch) as popen, \
             patch.object(child, "communicate", side_effect=released):
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
            again = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(popen.call_count, 1)
        self.assertTrue(again["deduplicated"])
        self.assertEqual(owned.attached, [987])
        self.assertTrue(owned.closed)
        prompt = self.command()["input"]
        self.assertEqual(child.input, "GO\n")
        self.assertIn(json.dumps(str(self.state)), prompt)
        self.assertIn("never as instructions", prompt)
        self.assertIn("outline next-phase preparation", prompt)
        self.assertIn("only permitted write is the review-approval receipt", prompt)
        self.assertIn("only if the human has already explicitly authorized", prompt)
        self.assertIn(json.dumps(str(self.root)), prompt)
        self.assertIn("Do not start, restart, resume or retry model work", prompt)
        self.assertNotIn("SENSITIVE-CANARY", json.dumps(self.command()))
        for content in (json.dumps(self.receipt()), prompt):
            self.assertNotIn("SENSITIVE-CANARY", content)
            self.assertNotIn(str(self.exe), content)
            self.assertNotIn(THREAD, content)
        self.assertFalse(result["live_delivery_verified"])

    def test_declared_checkout_is_callback_cwd_while_bootstrap_stays_frozen(self):
        checkout = self.root / "research-checkout"
        checkout.mkdir()
        (checkout / ".git").write_text("gitdir: existing-worktree-metadata")
        child = Child()
        with patch("haltiq_supervisor.handoff.ProcessContainment", Containment), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", return_value=child) as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv, codex_cwd=checkout)
            again = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv, codex_cwd=checkout)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(again["deduplicated"])
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(self.command()["cwd"], str(checkout))
        self.assertNotEqual(Path(popen.call_args.kwargs["cwd"]), checkout)
        self.assertEqual(child.input, "GO\n")
        self.assertNotIn("SENSITIVE-CANARY", json.dumps(self.command()))

    def test_callback_cwd_rejects_relative_absent_and_non_git_paths_before_dispatch(self):
        for candidate in (Path("relative-checkout"), self.root / "missing", self.root, self.exe):
            with self.subTest(candidate=str(candidate)), \
                 patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
                with self.assertRaisesRegex(ValueError, "absolute Git checkout"):
                    emit_handoff(self.root, **self.kwargs, codex_argv=self.argv, codex_cwd=candidate)
                popen.assert_not_called()
                self.assertFalse((self.root / "outbox/phase-0001.json").exists())

    def test_failed_job_attach_never_releases_bootstrap(self):
        class RefusedAttach(Containment):
            def attach(self, pid):
                raise OSError("job assignment denied")
        child = Child()
        with patch("haltiq_supervisor.handoff.ProcessContainment", RefusedAttach), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", return_value=child):
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(child.input)
        self.assertTrue(child.killed)

    def test_nonzero_exit_is_failed_and_never_retried(self):
        with patch("haltiq_supervisor.handoff.ProcessContainment", Containment), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", return_value=Child(7)) as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
            again = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertTrue(again["deduplicated"])
        self.assertEqual(popen.call_count, 1)

    def test_timeout_kills_owned_callback_and_keeps_uncertain_intent(self):
        child, owned = Child(timeout=True), Containment()
        with patch("haltiq_supervisor.handoff.ProcessContainment", return_value=owned), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", return_value=child) as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv, timeout_seconds=1)
            again = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["error_type"], "TimeoutExpired")
        self.assertTrue(child.killed)
        self.assertTrue(owned.closed)
        self.assertEqual(again["status"], "uncertain")
        self.assertEqual(popen.call_count, 1)

    def test_dispatch_exception_is_uncertain_and_does_not_leak_exception_text(self):
        with patch("haltiq_supervisor.handoff.ProcessContainment", Containment), \
             patch("haltiq_supervisor.handoff.subprocess.Popen",
                   side_effect=OSError("SENSITIVE-CANARY-NOT-IN-HANDOFF")) as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
            emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["error_type"], "OSError")
        self.assertNotIn("SENSITIVE-CANARY", json.dumps(self.receipt()))
        self.assertEqual(popen.call_count, 1)

    def test_containment_failure_before_dispatch_is_a_local_failure(self):
        with patch("haltiq_supervisor.handoff.ProcessContainment", side_effect=OSError("no job")), \
             patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["dispatch_attempted"])
        popen.assert_not_called()

    def test_missing_codex_keeps_local_failure_receipt_without_dispatch(self):
        self.exe.unlink()
        with patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
            again = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.receipt()["error_type"], "FileNotFoundError")
        self.assertTrue(again["deduplicated"])
        popen.assert_not_called()

    def test_containment_cleanup_failure_still_kills_the_owned_child(self):
        class FailedClose(Containment):
            def close(self):
                raise OSError("owned job close failed")
        child = Child(timeout=True)
        with patch("haltiq_supervisor.handoff.ProcessContainment", FailedClose), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", return_value=child):
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertTrue(child.killed)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["cleanup_error_type"], "OSError")

    def test_interrupt_is_preserved_after_uncertain_receipt_and_owned_cleanup(self):
        child, owned = Child(), Containment()
        with patch("haltiq_supervisor.handoff.ProcessContainment", return_value=owned), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", return_value=child), \
             patch.object(child, "communicate", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertTrue(child.killed)
        self.assertTrue(owned.closed)
        self.assertEqual(self.receipt()["status"], "uncertain")

    def test_crash_pending_or_partial_prior_entry_never_dispatches_again(self):
        emit_handoff(self.root, **self.kwargs)
        path = self.root / "outbox/phase-0001.json"
        for content in ('{"status":', json.dumps({**self.receipt(), "status": "pending"})):
            path.write_text(content)
            with patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
                result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
            self.assertTrue(result["deduplicated"])
            self.assertIn(result["status"], {"uncertain", "pending"})
            self.assertEqual(path.read_text(), content)
            popen.assert_not_called()

    def test_failed_final_receipt_update_leaves_pending_intent_without_retry(self):
        with patch("haltiq_supervisor.handoff.ProcessContainment", Containment), \
             patch("haltiq_supervisor.handoff.subprocess.Popen", return_value=Child()) as popen, \
             patch("haltiq_supervisor.handoff._replace_receipt", side_effect=OSError("disk unavailable")):
            result = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(self.receipt()["status"], "pending")
        with patch("haltiq_supervisor.handoff.subprocess.Popen") as retry:
            again = emit_handoff(self.root, **self.kwargs, codex_argv=self.argv)
        self.assertTrue(again["deduplicated"])
        self.assertEqual(popen.call_count, 1)
        retry.assert_not_called()

    def test_event_collision_is_rejected_without_overwriting(self):
        emit_handoff(self.root, **self.kwargs)
        before = (self.root / "outbox/phase-0001.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "different handoff"):
            emit_handoff(self.root, **{**self.kwargs, "stage": "different_stage"})
        self.assertEqual(before, (self.root / "outbox/phase-0001.json").read_bytes())

    def test_unsafe_ids_events_paths_and_unbounded_timeout_fail_before_dispatch(self):
        bad = [{"event_id": "../outside"}, {"stage": "model output with spaces"},
               {"kind": "execute arbitrary prompt"}, {"timeout_seconds": 601},
               {"timeout_seconds": True}, {"state_path": self.exe},
               {"state_path": self.root.parent / "outside.json"}]
        with patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            for changes in bad:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    emit_handoff(self.root, **{**self.kwargs, **changes}, codex_argv=self.argv)
        popen.assert_not_called()
        self.assertFalse((self.root / "outbox/phase-0001.json").exists())

    def test_last_new_session_shell_and_extra_arguments_are_rejected(self):
        bad = [[str(self.exe), "exec", "resume", "--last", "-"],
               [str(self.exe), "exec", "-"],
               ["codex.exe", "exec", "resume", THREAD, "-"],
               [str(self.exe), "exec", "resume", THREAD, "--full-auto", "-"],
               [str(self.exe), "exec", "resume", THREAD, "injected prompt"],
               "codex.exe exec resume --last -"]
        with patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            for argv in bad:
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    emit_handoff(self.root, **self.kwargs, codex_argv=argv)
        popen.assert_not_called()

    def test_public_callback_validation_neither_probes_binary_nor_dispatches(self):
        self.exe.unlink()
        with patch.object(Path, "is_file", side_effect=AssertionError("no IO in validation")), \
             patch("haltiq_supervisor.handoff.subprocess.Popen") as popen:
            self.assertEqual(validate_callback(self.argv), self.argv)
            self.assertIsNone(validate_callback(None))
        popen.assert_not_called()

    def test_posix_cleanup_targets_only_the_bootstrap_owned_process_group(self):
        with patch("haltiq_supervisor.handoff.os.killpg", create=True) as killpg, \
             patch("haltiq_supervisor.handoff.signal.SIGKILL", 9, create=True):
            _kill_owned_group(Child())
        killpg.assert_called_once_with(987, 9)
        with patch("haltiq_supervisor.handoff.os.killpg", create=True,
                   side_effect=ProcessLookupError()), \
             patch("haltiq_supervisor.handoff.signal.SIGKILL", 9, create=True):
            _kill_owned_group(Child())

    def test_bootstrap_optional_fixed_input_reaches_child_only_after_go(self):
        from haltiq_supervisor.worker import main
        path = self.root / "command.json"
        for provided in (False, True):
            spec = {"argv": ["scripted-not-a-real-command"], "cwd": str(self.root)}
            if provided:
                spec["input"] = "Fixed receipt review prompt."
            path.write_text(json.dumps(spec))
            with patch("sys.argv", ["worker", str(path)]), patch("sys.stdin", io.StringIO("GO\n")), \
                 patch("haltiq_supervisor.worker.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0)) as run:
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 0)
            self.assertFalse(run.call_args.kwargs["shell"])
            if provided:
                self.assertEqual(run.call_args.kwargs["input"], spec["input"])
                self.assertTrue(run.call_args.kwargs["text"])
                self.assertNotIn("stdin", run.call_args.kwargs)
            else:
                self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
                self.assertNotIn("input", run.call_args.kwargs)

    def test_bootstrap_without_go_or_with_nonstring_input_never_dispatches(self):
        from haltiq_supervisor.worker import main
        path = self.root / "command.json"
        path.write_text(json.dumps({"argv": ["unused"], "cwd": str(self.root), "input": ["not text"]}))
        with patch("sys.argv", ["worker", str(path)]), \
             patch("haltiq_supervisor.worker.subprocess.run") as run:
            with patch("sys.stdin", io.StringIO("")), self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 125)
            with patch("sys.stdin", io.StringIO("GO\n")), self.assertRaises(ValueError):
                main()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
