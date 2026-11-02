"""No-GPU runtime contracts: durable correlation, failures and owned cleanup."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import string
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


class WriterHTTPControls(unittest.TestCase):
    def response(self):
        return {"id": RESPONSE, "object": "chat.completion", "created": 1, "model": runtime.MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}

    def test_actual_post_binds_json_body_without_request_query_parameter(self):
        from fastapi.testclient import TestClient
        fake = SimpleNamespace(metadata={"model": runtime.MODEL, "revision": runtime.REVISION},
                               complete=Mock(return_value=self.response()))
        body = {"model": runtime.MODEL, "messages": [{"role": "user", "content": "public question"}]}
        with TestClient(runtime.create_writer_app(fake)) as client:
            response = client.post("/v1/chat/completions", json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), self.response())
            fake.complete.assert_called_once_with(body)
            self.assertEqual(client.get("/health").json()["client_correlation"], "by-client/<call_id>.json")
            self.assertEqual(client.get("/v1/models").json()["data"][0]["id"], runtime.MODEL)
            operation = client.get("/openapi.json").json()["paths"]["/v1/chat/completions"]["post"]
            self.assertNotIn("request", [item["name"] for item in operation.get("parameters", [])])

    def test_actual_streaming_post_publishes_correlated_primary_tokens(self):
        from fastapi.testclient import TestClient
        from haltiq_pair.server import persist_request
        response = self.response()
        raw_output = '{"answer":"Paris","supporting_facts":[]}'
        response["choices"][0]["message"]["content"] = raw_output
        with tempfile.TemporaryDirectory() as tmp:
            writer = object.__new__(runtime.CorrelatedWriter)
            writer.output = Path(tmp)
            writer.model = object()
            writer.metadata = {"model": runtime.MODEL, "revision": runtime.REVISION}

            def generated_without_model(instance, body):
                persist_request(instance.output, {"id": RESPONSE, "request": body,
                    "usage": response["usage"], "finish_reason": "stop", "response_text": raw_output})
                return response

            body = {"model": runtime.MODEL, "stream": True,
                    "metadata": {"yash_call_id": CALL, "yash_role": "writer"},
                    "messages": [{"role": "user", "content": "public question"}]}
            with patch.object(runtime.CorrelatedWriter, "_prefix_for", return_value=lambda *args: [1]), \
                    patch.object(runtime.Writer, "complete", new=generated_without_model), \
                    TestClient(runtime.create_writer_app(writer)) as client:
                with client.stream("POST", "/v1/chat/completions", json=body) as streamed:
                    self.assertEqual(streamed.status_code, 200)
                    self.assertTrue(streamed.headers["content-type"].startswith("text/event-stream"))
                    text = streamed.read().decode()
                payloads = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
                self.assertEqual(payloads[-1], "[DONE]")
                chunks = [json.loads(value) for value in payloads[:-1]]
                self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], raw_output)
                self.assertEqual(chunks[-1]["usage"], response["usage"])
            correlation = json.loads((Path(tmp) / "by-client" / (CALL + ".json")).read_bytes())
            primary = (Path(tmp) / "requests" / (RESPONSE + ".json")).read_bytes()
            self.assertEqual(correlation["call_id"], CALL)
            self.assertEqual(correlation["response_id"], RESPONSE)
            self.assertEqual(correlation["usage"], response["usage"])
            self.assertEqual(correlation["primary_sha256"], hashlib.sha256(primary).hexdigest())
            self.assertEqual(json.loads(primary)["request"], body)

    def test_actual_post_returns_structured_writer_failure(self):
        from fastapi.testclient import TestClient
        fake = SimpleNamespace(metadata={}, complete=Mock(side_effect=ValueError("synthetic refused request")))
        with TestClient(runtime.create_writer_app(fake)) as client:
            response = client.post("/v1/chat/completions", json={"messages": []})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"]["type"], "ValueError")
            fake.complete.assert_called_once_with({"messages": []})


class SchemaConstraintControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        cls.torch = torch
        alphabet = sorted(set(string.printable + 'éSão東京'))
        vocab = {token: index for index, token in enumerate(["<eos>", "<pad>", "<unk>"] + alphabet)}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        backend.decoder = decoders.Fuse()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                               pad_token="<pad>", unk_token="<unk>")
        cls.tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant: "
        cls.data = runtime.build_constraint_tokenizer_data(cls.tokenizer)

    def allowed_output(self, role, text):
        prefix = runtime.build_constraint_prefix(self.data, role)
        tokens = self.tokenizer.encode("public prompt", add_special_tokens=False)
        for token in self.tokenizer.encode(text, add_special_tokens=False):
            allowed = prefix(0, self.torch.tensor(tokens))
            if token not in allowed:
                return False
            tokens.append(token)
        return self.tokenizer.eos_token_id in prefix(0, self.torch.tensor(tokens))

    def test_real_tokenizer_constraint_accepts_all_role_structures_and_json_escapes(self):
        examples = {
            "writer": '{"answer":"São \\"東京\\"","supporting_facts":[{"title":"évidence","sentence_index":12},{"title":"Other","sentence_index":0}]}',
            "reviewer": '{"approved":true,"feedback":""}',
            "challenger": '{"alternative":"Other","evidence":[],"feedback":"Check evidence"}'}
        # Generate valid escaped text using JSON serialization, without answer labels.
        examples["writer"] = json.dumps({"answer": 'São "東京"', "supporting_facts": [
            {"title": "évidence", "sentence_index": 12}, {"title": "Other", "sentence_index": 0}]}, ensure_ascii=False)
        for role, text in examples.items():
            with self.subTest(role=role):
                self.assertTrue(self.allowed_output(role, text))
                self.assertEqual(runtime._validate_role_json(text, role), json.loads(text))

    def test_constraint_blocks_numeric_title_wrong_keys_boolean_strings_and_trailing_quote(self):
        bad = [("writer", '{"answer":"A","supporting_facts":[{"title":2,"sentence_index":0}]}'),
               ("writer", '{"answer":"A","supporting_facts":[{"title":"T","index":0}]}'),
               ("writer", '{"answer":"A","supporting_facts":[]} "'),
               ("reviewer", '{"approved":"true","feedback":"x"}')]
        for role, text in bad:
            with self.subTest(text=text):
                self.assertFalse(self.allowed_output(role, text))

    def test_numeric_minimum_is_explicitly_postvalidated_without_coercion(self):
        text = '{"answer":"A","supporting_facts":[{"title":"T","sentence_index":-1}]}'
        self.assertTrue(self.allowed_output("writer", text))  # Document the pinned library's limitation.
        with self.assertRaises(ValueError):
            runtime._validate_role_json(text, "writer")
        self.assertIn("not token-enforced", runtime.constraint_provenance()["limitation"])
        for raw in ('{"approved":1,"feedback":"x"}', '{"approved":true,"approved":false,"feedback":"x"}'):
            with self.assertRaises(ValueError):
                runtime._validate_role_json(raw, "reviewer")

    def test_fresh_parser_and_effective_config_ignore_environment_overrides(self):
        with patch.dict("os.environ", {"LMFE_MAX_CONSECUTIVE_WHITESPACES": "1", "LMFE_MAX_JSON_ARRAY_LENGTH": "1"}):
            a = runtime.build_constraint_prefix(self.data, "writer")
            b = runtime.build_constraint_prefix(self.data, "writer")
        self.assertIsNot(a.token_enforcer, b.token_enforcer)
        self.assertIsNot(a.token_enforcer.root_parser, b.token_enforcer.root_parser)
        self.assertEqual(a.token_enforcer.prefix_states, {})
        config = a.token_enforcer.root_parser.config
        self.assertEqual({key: getattr(config, key) for key in runtime.LMFE_EFFECTIVE_CONFIG}, runtime.LMFE_EFFECTIVE_CONFIG)

    def test_tokenizer_tree_is_cached_but_request_state_is_not_and_missing_pin_fails_closed(self):
        writer = object.__new__(runtime.CorrelatedWriter)
        writer.tokenizer = self.tokenizer
        with patch.object(runtime, "build_constraint_tokenizer_data", return_value=self.data) as build:
            a, b = writer._prefix_for("writer"), writer._prefix_for("reviewer")
        build.assert_called_once_with(self.tokenizer)
        self.assertIsNot(a.token_enforcer, b.token_enforcer)
        with patch("importlib.metadata.version", return_value="wrong-version"):
            with self.assertRaises(RuntimeError):
                runtime.constraint_dependencies()
        with self.assertRaises(ValueError):
            runtime.build_constraint_prefix(self.data, "undeclared")

    def test_generate_proxy_preserves_kwargs_attributes_and_shared_model(self):
        model = Mock()
        model.config = object()
        model.generate.return_value = "physical result"
        prefix = lambda batch, sent: [1]
        proxy = runtime._ConstrainedModel(model, prefix)
        kwargs = dict(max_new_tokens=512, do_sample=False, prefill_chunk_size=512,
                      logits_to_keep=1, use_cache=True, pad_token_id=3)
        self.assertEqual(proxy.generate("inputs", **kwargs), "physical result")
        self.assertIs(proxy.config, model.config)
        self.assertEqual(model.generate.call_args.args, ("inputs",))
        self.assertEqual(model.generate.call_args.kwargs, dict(kwargs, prefix_allowed_tokens_fn=prefix))
        with self.assertRaises(ValueError):
            proxy.generate(prefix_allowed_tokens_fn=lambda *args: [])

    def test_real_http_uses_constraint_callback_and_preserves_failed_output_token_receipts(self):
        from fastapi.testclient import TestClient
        tokenizer, torch = self.tokenizer, self.torch
        valid = '{"answer":"A","supporting_facts":[{"title":"T","sentence_index":12}]}'
        invalid = '{"answer":"A","supporting_facts":[{"title":2,"sentence_index":0}]}"'
        captured = []

        class ReplayModel:
            def __init__(self, text, check_tokens):
                self.text, self.check_tokens = text, check_tokens

            def generate(self, **kwargs):
                captured.append(kwargs)
                ids = kwargs["input_ids"][0].tolist()
                prefix = kwargs["prefix_allowed_tokens_fn"]
                for token in tokenizer.encode(self.text, add_special_tokens=False):
                    if self.check_tokens and token not in prefix(0, torch.tensor(ids)):
                        raise AssertionError("The replay attempted a forbidden token")
                    ids.append(token)
                if self.check_tokens:
                    self.assert_end = tokenizer.eos_token_id in prefix(0, torch.tensor(ids))
                    if not self.assert_end:
                        raise AssertionError("EOS must be blocked until the JSON is complete")
                return torch.tensor([ids])

        for text, enforce, expected, budget in ((valid, True, 200, 512), (invalid, False, 400, 512),
                (valid, True, 400, len(tokenizer.encode(valid, add_special_tokens=False)))):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                writer = object.__new__(runtime.CorrelatedWriter)
                writer.output, writer.tokenizer, writer.device = Path(tmp), tokenizer, "cpu"
                writer.model, writer.max_context = ReplayModel(text, enforce), 16384
                writer._constraint_tokenizer_data = self.data
                writer.metadata = {"format_constraints": runtime.constraint_provenance()}
                body = {"model": runtime.MODEL, "messages": [{"role": "user", "content": "public"}],
                        "metadata": {"yash_call_id": CALL, "yash_role": "writer"}, "max_tokens": budget, "stream": True}
                original_model = writer.model
                with TestClient(runtime.create_writer_app(writer)) as client:
                    response = client.post("/v1/chat/completions", json=body)
                self.assertEqual(response.status_code, expected, response.text)
                self.assertIs(writer.model, original_model)
                self.assertNotIn("generate", vars(original_model))  # No method monkeypatch.
                correlation = json.loads((Path(tmp) / "by-client" / (CALL + ".json")).read_bytes())
                primary = json.loads((Path(tmp) / correlation["primary_receipt"]).read_bytes())
                self.assertEqual(primary["response_text"], text)
                self.assertGreater(correlation["usage"]["total_tokens"], 0)
                self.assertEqual(correlation["usage"], primary["usage"])
                check = json.loads((Path(tmp) / "format-checks" / (CALL + ".json")).read_bytes())
                self.assertEqual(check["status"], "incomplete_length" if budget != 512 else
                                 "validated" if expected == 200 else "failed")
                self.assertEqual(captured[-1]["max_new_tokens"], budget)
                self.assertEqual(captured[-1]["prefill_chunk_size"], 512)


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
    def test_failed_qualification_keeps_role_trace_receipt_and_clears_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = runtime.Runtime(tmp)
            manager.agents = object()
            manager.critic = SimpleNamespace(context={}, usage=[], score=Mock())
            agents = SimpleNamespace(usage=[])

            def fail(role, system, prompt):
                agents.usage.append({"role": role, "status": "failed", "call_id": CALL,
                                     "usage": {"total_tokens": 13}})
                raise ValueError("synthetic schema failure")
            agents.generate = fail
            with patch("haltiq_pair.nov2_orchestration.RoleAgents.from_agents", return_value=agents):
                with self.assertRaises(RuntimeError):
                    manager.qualify()
            receipt = json.loads((Path(tmp) / "qualification.json").read_bytes())
            self.assertEqual(receipt["status"], "FAILED")
            self.assertEqual(receipt["role_usage"][0]["usage"]["total_tokens"], 13)
            self.assertEqual(manager.critic.context, {})
            manager.critic.score.assert_not_called()

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
                  "client_correlation": "by-client/<call_id>.json", "format_constraints": runtime.constraint_provenance()}
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
