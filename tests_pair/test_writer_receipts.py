"""Response receipts remain complete when the optional JSONL mirror fails."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from haltiq_pair import server


class WriterReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import BatchEncoding
        except ImportError:
            raise unittest.SkipTest("optional CPU writer transport dependencies")

    def fixture_writer(self, directory, text="fixture response"):
        import torch
        from transformers import BatchEncoding
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, *args, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})
            def decode(self, *args, **kwargs):
                return text
        class Model:
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3]])
        writer = server.Writer.__new__(server.Writer)
        writer.output, writer.device, writer.max_context = Path(directory), "cpu", 16384
        writer.tokenizer, writer.model = Tokenizer(), Model()
        return writer

    def test_unique_atomic_receipts_round_trip_unicode_and_control_payloads(self):
        text = "Unicode \u03b1\u2028\u2029\U0001f642; controls\n\r\t\u0000\"\\"
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": 3}
        original = deepcopy(body)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server.os, "fsync", wraps=os.fsync) as fsync, \
                patch.object(server.os, "replace", wraps=os.replace) as replace:
            writer = self.fixture_writer(directory, text)
            replies = [writer.complete(body), writer.complete(body)]
            self.assertEqual(len({reply["id"] for reply in replies}), 2)
            receipts = sorted((Path(directory) / "requests").glob("*.json"))
            self.assertEqual(len(receipts), 2)
            self.assertEqual(fsync.call_count, 2)
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(list((Path(directory) / "requests").glob("*.tmp")), [])
            for receipt in receipts:
                record = json.loads(receipt.read_bytes())
                self.assertEqual(receipt.name, record["id"] + ".json")
                self.assertEqual(record["request"], original)
                self.assertEqual(record["response_text"], text)
                self.assertEqual(record["logging"]["primary_receipt"], "requests/" + receipt.name)
            mirror = [json.loads(line) for line in (Path(directory) / "model_requests.jsonl").read_bytes().split(b"\n") if line]
            self.assertEqual(len(mirror), 2)
            self.assertEqual({record["id"] for record in mirror}, {reply["id"] for reply in replies})
            self.assertEqual(body, original)

    def test_mirror_io_failures_keep_success_and_the_already_published_primary(self):
        original_open = Path.open
        for fault_stage in ("open", "write", "flush", "close"):
            with self.subTest(fault_stage=fault_stage), tempfile.TemporaryDirectory() as directory:
                writer = self.fixture_writer(directory)
                class BrokenAppend:
                    def __init__(self, path, *args, **kwargs):
                        self.stream = original_open(path, *args, **kwargs)
                    def __enter__(self):
                        return self
                    def write(self, text):
                        if fault_stage == "write":
                            self.stream.write(text[:17])
                            self.stream.flush()
                            raise OSError("synthetic mirror write failure")
                        return self.stream.write(text)
                    def flush(self):
                        self.stream.flush()
                        if fault_stage == "flush":
                            raise OSError("synthetic mirror flush failure")
                    def __exit__(self, *args):
                        self.stream.close()
                        if fault_stage == "close":
                            raise OSError("synthetic mirror close failure")
                def open_with_fault(path, *args, **kwargs):
                    if path.name == "model_requests.jsonl" and args[0] == "a":
                        primaries = list((Path(directory) / "requests").glob("*.json"))
                        self.assertEqual(len(primaries), 1)
                        self.assertEqual(json.loads(primaries[0].read_bytes())["response_text"], "fixture response")
                        if fault_stage == "open":
                            raise OSError("synthetic mirror open failure")
                        return BrokenAppend(path, *args, **kwargs)
                    return original_open(path, *args, **kwargs)
                with patch.object(Path, "open", new=open_with_fault), self.assertLogs(server.__name__, "WARNING"):
                    reply = writer.complete({"messages": [{"role": "user", "content": "fixture"}]})
                self.assertEqual(reply["choices"][0]["message"]["content"], "fixture response")
                self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
                record = json.loads((Path(directory) / "requests" / (reply["id"] + ".json")).read_bytes())
                self.assertEqual(record["logging"]["compatibility_append_error"]["type"], "OSError")
                self.assertIn("synthetic mirror", record["logging"]["compatibility_append_error"]["message"])
                self.assertEqual(record["usage"], reply["usage"])

    def test_failed_error_annotation_still_preserves_the_initial_primary_receipt(self):
        original_open, atomic = Path.open, server._atomic_request_receipt
        def no_mirror(path, *args, **kwargs):
            if path.name == "model_requests.jsonl":
                raise OSError("synthetic mirror failure")
            return original_open(path, *args, **kwargs)
        for operation in ("fsync", "replace"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                initial = []
                def fail_annotation(path, record):
                    if initial:
                        with patch.object(server.os, operation, side_effect=OSError("synthetic annotation failure")):
                            return atomic(path, record)
                    atomic(path, record)
                    initial.append(path.read_bytes())
                with patch.object(Path, "open", new=no_mirror), \
                        patch.object(server, "_atomic_request_receipt", side_effect=fail_annotation), \
                        self.assertLogs(server.__name__, "WARNING") as warnings:
                    reply = self.fixture_writer(directory).complete({"messages": [{"role": "user", "content": "fixture"}]})
                primary = Path(directory) / "requests" / (reply["id"] + ".json")
                self.assertEqual(primary.read_bytes(), initial[0])
                self.assertEqual(json.loads(primary.read_bytes())["response_text"], "fixture response")
                self.assertEqual(len(warnings.output), 2)
                self.assertEqual(list(primary.parent.glob("*.tmp")), [])

    def test_failed_initial_publication_does_not_expose_a_partial_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "requests" / "fixture.json"
            with patch.object(server.os, "replace", side_effect=OSError("synthetic publish failure")):
                with self.assertRaisesRegex(OSError, "publish failure"):
                    server._atomic_request_receipt(primary, {"id": "fixture", "response_text": "complete payload"})
            self.assertFalse(primary.exists())
            self.assertEqual(list(primary.parent.iterdir()), [])

    def test_writer_provenance_names_atomic_files_as_the_primary_receipts(self):
        model = Mock()
        model.to.return_value, model.eval.return_value = model, model
        model.config = SimpleNamespace(_attn_implementation="sdpa")
        with tempfile.TemporaryDirectory() as directory, \
                patch("transformers.AutoTokenizer.from_pretrained"), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model):
            writer = server.Writer(directory, device="cpu")
            receipt = json.loads((Path(directory) / "writer.json").read_bytes())
            self.assertEqual(receipt, writer.metadata)
            self.assertEqual(receipt["request_receipt_primary"], "requests/<response_id>.json")
            self.assertEqual(receipt["request_receipt_policy"], "exclusive_temp_flush_fsync_replace")
            self.assertIn("best-effort mirror", receipt["request_receipt_compatibility"])


if __name__ == "__main__":
    unittest.main()
