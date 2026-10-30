"""No downloads or GPU calls: 7B identity, NF4 load contract and HTTP metadata."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from haltiq_pair import server
from haltiq_pair.strong_writer import (BITSANDBYTES_VERSION, MODEL, REVISION,
                                      StrongRuntime, StrongWriter, create_app,
                                      validate_health)


class StrongWriterTests(unittest.TestCase):
    def valid_metadata(self):
        return {"model": MODEL, "revision": REVISION, "quantization": {
            "weight_bits": 4, "type": "nf4", "compute_dtype": "torch.float16",
            "version": BITSANDBYTES_VERSION}}

    def test_frozen_variant_does_not_change_original_model(self):
        self.assertEqual(server.MODEL, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(MODEL, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(REVISION, "a09a35458c702b33eeacc393d103063234e8bc28")
        self.assertEqual(BITSANDBYTES_VERSION, "0.50.2")

    def test_response_and_sse_never_report_the_original_model(self):
        writer = StrongWriter.__new__(StrongWriter)
        original = {"id": "test", "created": 1, "model": server.MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}], "usage": {"total_tokens": 2}}
        with patch.object(server.Writer, "complete", return_value=original):
            response = writer.complete({"model": MODEL, "messages": []})
        self.assertEqual(response["model"], MODEL)
        chunks = "".join(server.stream_response(response))
        self.assertIn(MODEL, chunks)
        self.assertNotIn(server.MODEL, chunks)

    def test_wrong_request_model_rejected_before_generation(self):
        writer = StrongWriter.__new__(StrongWriter)
        with patch.object(server.Writer, "complete") as complete:
            with self.assertRaisesRegex(ValueError, "pinned 7B"):
                writer.complete({"model": server.MODEL})
        complete.assert_not_called()

    def test_quantized_load_pins_fp16_and_preserves_context(self):
        torch = SimpleNamespace(float16="fake_fp16", cuda=SimpleNamespace(is_available=lambda: True),
                                __version__="test-torch")
        model = MagicMock()
        model.eval.return_value = model
        model.config._attn_implementation = "sdpa"
        factory = MagicMock(return_value=model)
        quantizer = MagicMock(return_value="fake_nf4")
        transformers = SimpleNamespace(__version__="test-transformers",
            AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value="test-tokenizer")),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory),
            BitsAndBytesConfig=quantizer)
        with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", {
                "torch": torch, "transformers": transformers}), patch(
                "haltiq_pair.strong_writer.version", return_value=BITSANDBYTES_VERSION):
            writer = StrongWriter(temporary)
            metadata = json.loads((Path(temporary) / "writer.json").read_text())
        quantizer.assert_called_once_with(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="fake_fp16", bnb_4bit_use_double_quant=True)
        self.assertEqual(factory.call_args.args, (MODEL,))
        self.assertEqual(factory.call_args.kwargs["revision"], REVISION)
        self.assertEqual(factory.call_args.kwargs["device_map"], {"": 0})
        self.assertEqual(factory.call_args.kwargs["quantization_config"], "fake_nf4")
        model.to.assert_not_called()
        self.assertEqual(writer.max_context, 16384)
        self.assertEqual(metadata["prefill_chunk_size"], 512)
        self.assertEqual(metadata["context_policy"], "reject_overflow_without_truncation")
        self.assertFalse(metadata["quantization"]["cpu_offload"])
        validate_health(metadata)

    def test_health_refuses_wrong_revision_or_quantizer(self):
        validate_health(self.valid_metadata())
        for key, value in (("revision", "main"), ("model", server.MODEL)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_health({**self.valid_metadata(), key: value})
        metadata = self.valid_metadata()
        metadata["quantization"]["version"] = "unfrozen"
        with self.assertRaises(RuntimeError):
            validate_health(metadata)

    def test_http_surface_is_7b_and_post_request_is_not_a_query_parameter(self):
        from fastapi.testclient import TestClient
        fake = SimpleNamespace(metadata=self.valid_metadata(), complete=lambda body: {
            "model": MODEL, "request": body})
        with TestClient(create_app(fake)) as client:
            self.assertEqual(client.get("/health").json()["revision"], REVISION)
            self.assertEqual(client.get("/v1/models").json()["data"][0]["id"], MODEL)
            response = client.post("/v1/chat/completions", json={"model": MODEL})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["model"], MODEL)

    def test_deployment_seed_is_not_selected_using_test_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "seed"):
                StrongRuntime(temporary, SimpleNamespace(seed=4))


if __name__ == "__main__":
    unittest.main()
