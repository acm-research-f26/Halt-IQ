"""OpenAI/Strands request compatibility at the local Qwen transport boundary."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

from haltiq_pair.server import MODEL, REVISION, Writer, normalize_messages


STRANDS_MESSAGES = [
    {"role": "system", "content": [{"type": "text", "text": "Return a short answer."}]},
    {"role": "user", "content": [{"type": "text", "text": "First part. "},
                                  {"type": "text", "text": "Second part."}]},
]


class MessageNormalizationTests(unittest.TestCase):
    def test_standard_strands_text_blocks_and_string_content(self):
        request = deepcopy(STRANDS_MESSAGES)
        request.append({"role": "assistant", "content": "plain reply"})
        original = deepcopy(request)
        normalized = normalize_messages(request)
        self.assertEqual([m["content"] for m in normalized],
                         ["Return a short answer.", "First part. Second part.", "plain reply"])
        self.assertEqual(request, original)

    def test_empty_and_null_content_keep_tool_calls_and_decode_arguments(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "lookup", "arguments": '{"id":"test"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": []}]
        original = deepcopy(messages)
        result = normalize_messages(messages)
        self.assertEqual(result[0]["content"], "")
        self.assertEqual(result[1]["content"], "")
        self.assertEqual(result[0]["tool_calls"][0]["function"]["arguments"], {"id": "test"})
        self.assertEqual(result[1]["tool_call_id"], "call_1")
        self.assertEqual(messages, original)

    def test_unsupported_multimodal_or_malformed_blocks_fail_explicitly(self):
        for block in ({"type": "image_url", "image_url": {"url": "fixture"}},
                      {"type": "input_audio", "input_audio": {}},
                      {"type": "text", "text": 3}, "not a content block"):
            with self.subTest(block=block), self.assertRaises(ValueError):
                normalize_messages([{"role": "user", "content": [block]}])

    def test_invalid_tool_argument_json_and_unsupported_roles_fail(self):
        for arguments in ("not JSON", "[]", [], None):
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                normalize_messages([{"role": "assistant", "content": None,
                                     "tool_calls": [{"function": {"name": "lookup", "arguments": arguments}}]}])
        with self.assertRaises(ValueError):
            normalize_messages([{"role": "unrecognized", "content": "text"}])

    def test_writer_complete_normalizes_the_actual_request_before_template(self):
        try:
            import torch
            from transformers import BatchEncoding
        except ImportError:
            self.skipTest("optional local model transport dependencies")
        captured = []
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                captured.extend(messages)
                if any(not isinstance(m["content"], str) for m in messages):
                    raise TypeError("Qwen content must be strings")
                return BatchEncoding({"input_ids": torch.tensor([[1]])})
            def decode(self, ids, **kwargs):
                return "verified"
        class Model:
            def generate(self, **kwargs):
                return torch.tensor([[1, 2]])
        body = {"messages": deepcopy(STRANDS_MESSAGES), "max_tokens": 3}
        original = deepcopy(body)
        with tempfile.TemporaryDirectory() as directory:
            writer = Writer.__new__(Writer)
            writer.tokenizer, writer.model, writer.device = Tokenizer(), Model(), "cpu"
            writer.output, writer.max_context = Path(directory), 100
            reply = writer.complete(body)
            self.assertEqual(reply["choices"][0]["message"]["content"], "verified")
            self.assertEqual(captured, normalize_messages(STRANDS_MESSAGES))
            receipt = json.loads((Path(directory)/"model_requests.jsonl").read_text())
            self.assertEqual(receipt["request"], original)
        self.assertEqual(body, original)


@unittest.skipUnless(os.environ.get("HALTIQ_TEST_QWEN_TOKENIZER") == "1",
                     "optional pinned public tokenizer download, no model weights")
class PinnedQwenTemplateTests(unittest.TestCase):
    def test_real_template_accepts_sdk_blocks_and_object_tool_arguments(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        messages = deepcopy(STRANDS_MESSAGES) + [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "lookup", "arguments": '{"id":"test"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "Observed fixture result."}]},
        ]
        normalized = normalize_messages(messages)
        rendered = tokenizer.apply_chat_template(normalized, tokenize=False,
                                                  add_generation_prompt=True)
        self.assertIn("First part. Second part.", rendered)
        self.assertIn('"arguments": {"id": "test"}', rendered)
        self.assertIn("Observed fixture result.", rendered)
        self.assertNotIn('"arguments": "{', rendered)


if __name__ == "__main__":
    unittest.main()
