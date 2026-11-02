"""No-GPU runtime contracts: durable correlation, failures and owned cleanup."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from haltiq_pair import nov2_runtime as runtime


CALL = "a" * 32
RESPONSE = "chatcmpl-" + "b" * 32


def physical(root, call=CALL, role="writer"):
    metadata = {"yash_call_id": call, "yash_role": role}
    body = {"metadata": metadata, "messages": [{"role": "user", "content": "public"}]}
    response = {"id": RESPONSE, "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
                "choices": [{"finish_reason": "stop"}]}
    receipt = {"id": RESPONSE, "request": body, "usage": response["usage"],
               "finish_reason": "stop", "response_text": "answer"}
    path = Path(root) / "requests" / (RESPONSE + ".json")
    path.parent.mkdir()
    raw = (json.dumps(receipt, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return body, response, path, raw


class CorrelationControls(unittest.TestCase):
    def test_uuid_raw_receipt_hash_and_tokens_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            body, response, path, raw = physical(tmp)
            record = runtime.correlate_response(tmp, body, response)
            saved = json.loads((Path(tmp) / "by-client" / (CALL + ".json")).read_text())
            self.assertEqual(record, saved)
            self.assertEqual(saved["primary_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(saved["usage"], response["usage"])
            self.assertEqual(saved["primary_receipt"], "requests/" + path.name)
            self.assertEqual(path.read_bytes(), raw)

    def test_usage_and_metadata_mismatch_do_not_publish_client_receipt(self):
        for mutation in ("usage", "metadata"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                body, response, _, _ = physical(tmp)
                if mutation == "usage":
                    response["usage"] = dict(response["usage"], total_tokens=11)
                else:
                    body = deepcopy(body)
                    body["metadata"]["yash_role"] = "challenger"
                with self.assertRaises(ValueError):
                    runtime.correlate_response(tmp, body, response)
                self.assertFalse((Path(tmp) / "by-client" / (CALL + ".json")).exists())

    def test_duplicate_client_id_preserves_first_correlation(self):
        with tempfile.TemporaryDirectory() as tmp:
            body, response, _, _ = physical(tmp)
            runtime.correlate_response(tmp, body, response)
            target = Path(tmp) / "by-client" / (CALL + ".json")
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                runtime.correlate_response(tmp, body, response)
            self.assertEqual(target.read_bytes(), before)

    def test_invalid_metadata_and_response_identity_are_rejected(self):
        for call, role in (("../escape", "writer"), (CALL, "undeclared")):
            with self.subTest(call=call, role=role), tempfile.TemporaryDirectory() as tmp:
                body, response, _, _ = physical(tmp, call, role)
                with self.assertRaises(ValueError):
                    runtime.correlate_response(tmp, body, response)
        with tempfile.TemporaryDirectory() as tmp:
            body, response, _, _ = physical(tmp)
            response["id"] = "../../escape"
            with self.assertRaises(ValueError):
                runtime.correlate_response(tmp, body, response)
            self.assertIsNone(runtime.correlate_response(tmp, {}, response))

    def test_correlated_writer_rejects_duplicate_before_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "by-client").mkdir()
            (output / "by-client" / (CALL + ".json")).write_text("first immutable correlation")
            writer = object.__new__(runtime.CorrelatedWriter)
            writer.output = output
            with patch.object(runtime.Writer, "complete") as generate:
                with self.assertRaises(ValueError):
                    writer.complete({"metadata": {"yash_call_id": CALL, "yash_role": "writer"}})
                generate.assert_not_called()

    def test_correlated_writer_rejects_invalid_role_before_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = object.__new__(runtime.CorrelatedWriter)
            writer.output = Path(tmp)
            with patch.object(runtime.Writer, "complete") as generate:
                with self.assertRaises(ValueError):
                    writer.complete({"metadata": {"yash_call_id": CALL, "yash_role": "not-a-role"}})
                generate.assert_not_called()


class RecordingControls(unittest.TestCase):
    def test_success_is_durable_with_input_hash_context_and_provenance(self):
        critic = Mock(provenance={"critic_revision": "pinned"}, last_record={"usage": {"total_tokens": 7}})
        critic.score.return_value = .75
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = runtime.RecordedCritic(critic, tmp)
            wrapper.context = {"phase": "train", "question_id": "public-id"}
            self.assertEqual(wrapper.score("question", ["public evidence"], "draft"), .75)
            record = wrapper.usage[0]
            saved = json.loads((Path(tmp) / (record["call_id"] + ".json")).read_text())
            self.assertEqual(record, saved)
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["context"], wrapper.context)
            self.assertEqual(saved["public_input_sha256"], hashlib.sha256(json.dumps(
                ["question", ["public evidence"], "draft"], ensure_ascii=False).encode()).hexdigest())
            self.assertEqual(saved["critic"]["usage"]["total_tokens"], 7)
            self.assertEqual(wrapper.last_record["call_id"], saved["call_id"])
            self.assertEqual(wrapper.provenance["critic_revision"], "pinned")
            self.assertGreaterEqual(saved["seconds"], 0)

    def test_failure_and_invalid_scores_each_remain_durable(self):
        for outcome in (RuntimeError("synthetic transport failure"), float("nan"), True, 1.1):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                critic = Mock(provenance={}, last_record=None)
                if isinstance(outcome, Exception):
                    critic.score.side_effect = outcome
                else:
                    critic.score.return_value = outcome
                wrapper = runtime.RecordedCritic(critic, tmp)
                with self.assertRaises((RuntimeError, ValueError)):
                    wrapper.score("question", ["public passage"], "draft")
                self.assertEqual(len(wrapper.usage), 1)
                record = wrapper.usage[0]
                self.assertEqual(json.loads((Path(tmp) / (record["call_id"] + ".json")).read_text()), record)
                self.assertEqual(record["status"], "failed")
                self.assertIsNone(record["score"])
                self.assertIsNone(wrapper.last_record)
                self.assertIn("error", record)

    def test_uuid_collision_refused_before_verifier_and_first_receipt_unchanged(self):
        critic = Mock(provenance={}, last_record={})
        critic.score.return_value = .6
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime.uuid, "uuid4", return_value=SimpleNamespace(hex=CALL)):
            wrapper = runtime.RecordedCritic(critic, tmp)
            wrapper.score("first", ["public"], "draft")
            path = Path(tmp) / (CALL + ".json")
            before = path.read_bytes()
            with self.assertRaises((ValueError, FileExistsError)):
                wrapper.score("second", ["public"], "draft")
            self.assertEqual(critic.score.call_count, 1)
            self.assertEqual(path.read_bytes(), before)

    def test_user_interrupt_still_publishes_failed_verifier_attempt(self):
        critic = Mock(provenance={}, last_record=None)
        critic.score.side_effect = KeyboardInterrupt("synthetic notebook interruption")
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = runtime.RecordedCritic(critic, tmp)
            with self.assertRaises(KeyboardInterrupt):
                wrapper.score("question", ["public"], "draft")
            self.assertEqual(len(wrapper.usage), 1)
            record = wrapper.usage[0]
            self.assertEqual(record["error"]["type"], "KeyboardInterrupt")
            self.assertEqual(json.loads((Path(tmp) / (record["call_id"] + ".json")).read_text()), record)


class OwnershipControls(unittest.TestCase):
    def test_occupied_port_refuses_before_starting_any_owned_service(self):
        probe = Mock()
        probe.__enter__ = Mock(return_value=probe)
        probe.__exit__ = Mock(return_value=False)
        probe.connect_ex.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, \
                patch("torch.cuda.is_available", return_value=True), \
                patch.object(runtime.socket, "socket", return_value=probe), \
                patch("haltiq_pair.critic.start_critic") as start_critic, \
                patch.object(runtime.subprocess, "Popen") as start_writer:
            manager = runtime.Runtime(Path(tmp) / "new", qualify=False)
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                with manager:
                    self.fail("Must not yield a runtime")
            start_critic.assert_not_called()
            start_writer.assert_not_called()
            self.assertIsNone(manager.writer)
            self.assertIsNone(manager.critic_server)

    def test_close_only_terminates_owned_handles_and_escalates_owned_timeout(self):
        manager = runtime.Runtime("not-used", qualify=False)
        writer, critic, log = Mock(), Mock(), Mock()
        writer.poll.return_value = None
        writer.wait.side_effect = [subprocess.TimeoutExpired("owned-writer", 20), 0]
        manager.writer, manager.critic_server, manager.log = writer, critic, log
        manager.close()
        writer.terminate.assert_called_once()
        writer.kill.assert_called_once()
        self.assertEqual(writer.wait.call_count, 2)
        critic.close.assert_called_once()
        log.close.assert_called_once()

    def test_partial_startup_failure_closes_only_services_created_by_this_context(self):
        manager = runtime.Runtime("not-used", qualify=False)
        critic = Mock()

        def failed_start():
            manager.critic_server = critic
            raise RuntimeError("synthetic writer startup failure")

        with patch.object(manager, "start", side_effect=failed_start):
            with self.assertRaises(RuntimeError):
                with manager:
                    self.fail("Must not yield after failed startup")
        critic.close.assert_called_once()
        self.assertIsNone(manager.writer)

    def test_writer_cleanup_failure_does_not_skip_other_owned_resources(self):
        manager = runtime.Runtime("not-used", qualify=False)
        writer, critic, log = Mock(), Mock(), Mock()
        writer.poll.return_value = None
        writer.terminate.side_effect = OSError("synthetic already-exited process race")
        manager.writer, manager.critic_server, manager.log = writer, critic, log
        with self.assertRaises(OSError):
            manager.close()
        log.close.assert_called_once()
        critic.close.assert_called_once()

    def test_mocked_full_runtime_yields_matching_receipt_roots_and_closes_owned_services(self):
        health = {"model": runtime.MODEL, "revision": runtime.REVISION, "dtype": "torch.float16",
                  "client_correlation": "by-client/<call_id>.json"}
        probe = Mock()
        probe.__enter__, probe.__exit__ = Mock(return_value=probe), Mock(return_value=False)
        probe.connect_ex.return_value = 1
        server = Mock(endpoint="http://127.0.0.1:8000", provenance={"serving_dtype": "float16"})
        writer = Mock()
        writer.poll.return_value = None
        fake_critic = Mock(provenance={"critic_revision": "pinned"})
        with tempfile.TemporaryDirectory() as tmp, \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_name", return_value="FAKE GPU - not queried"), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=123)), \
                patch("importlib.metadata.version", return_value="fake-test-version"), \
                patch.object(runtime.socket, "socket", return_value=probe), \
                patch("haltiq_pair.critic.start_critic", return_value=server), \
                patch("haltiq_pair.critic.Critic", return_value=fake_critic), \
                patch.object(runtime.subprocess, "Popen", return_value=writer), \
                patch.object(runtime.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(health))):
            manager = runtime.Runtime(tmp, qualify=False)
            with manager as (agents, critic):
                self.assertEqual(agents.max_tokens, 512)
                self.assertEqual(agents.receipt_root, Path(tmp) / "writer/by-client")
                self.assertEqual(agents.request_receipt_root, Path(tmp) / "writer/requests")
                self.assertIsInstance(critic, runtime.RecordedCritic)
                self.assertFalse(manager.provenance["qualification_is_benchmark_evidence"])
                self.assertTrue((Path(tmp) / "runtime.json").exists())
            writer.terminate.assert_called_once()
            server.close.assert_called_once()
            self.assertTrue(manager.log.closed)


if __name__ == "__main__":
    unittest.main()
