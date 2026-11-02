"""Real installed Strands/OpenAI SDK transport, with no model or network calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import openai
from strands.types.exceptions import MaxTokensReachedException

from haltiq_pair import hotpot
from haltiq_pair.nov2_orchestration import (ROLE_FORMAT_EXAMPLES, ROLE_FORMAT_GUIDANCE,
                                           ROLE_FORMAT_VERSION, RoleAgents)
from haltiq_pair.server import stream_response


class SDKTransportControls(unittest.TestCase):
    def invoke(self, role, finish_reason="stop", content="public response"):
        requests = []
        original_client = openai.AsyncOpenAI
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        receipt_root = Path(temporary.name)

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            response = {"id": "chatcmpl-" + "b" * 32, "object": "chat.completion",
                        "created": 1, "model": body["model"],
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                     "finish_reason": finish_reason}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
            if finish_reason == "length":
                metadata = body["metadata"]
                (receipt_root / (metadata["yash_call_id"] + ".json")).write_text(json.dumps({
                    "call_id": metadata["yash_call_id"], "role": role, "usage": response["usage"]}))
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content="".join(stream_response(response)).encode())

        def client_factory(**kwargs):
            return original_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

        agents = RoleAgents(max_tokens=512, receipt_root=receipt_root)
        with patch("openai.AsyncOpenAI", side_effect=client_factory):
            if finish_reason == "length":
                with self.assertRaises(MaxTokensReachedException):
                    agents.generate(role, "Independent role system prompt", "Public evidence only")
                result = None
            else:
                result = agents.generate(role, "Independent role system prompt", "Public evidence only")
        return result, agents.usage, requests

    def test_three_roles_send_correlation_metadata_and_role_caps_through_real_sdk(self):
        for role, cap in (("writer", 512), ("reviewer", 256), ("challenger", 256)):
            with self.subTest(role=role):
                result, usage, requests = self.invoke(role)
                self.assertEqual(result, "public response")
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["max_tokens"], cap)
                self.assertEqual(requests[0]["temperature"], 0)
                self.assertEqual(requests[0]["metadata"], {"yash_call_id": usage[0]["call_id"], "yash_role": role})
                self.assertEqual(usage[0]["status"], "completed")
                self.assertEqual(usage[0]["usage"]["totalTokens"], 7)

    def test_real_sdk_sends_exact_shared_structural_examples_for_every_role(self):
        for role in ("writer", "reviewer", "challenger"):
            with self.subTest(role=role):
                _, usage, requests = self.invoke(role)
                systems = [message["content"] for message in requests[0]["messages"] if message["role"] == "system"]
                texts = [value if isinstance(value, str) else "".join(block["text"] for block in value)
                         for value in systems]
                system = "\n".join(texts)
                self.assertIn("Independent role system prompt", system)
                self.assertIn(ROLE_FORMAT_GUIDANCE[role], system)
                self.assertIn(ROLE_FORMAT_EXAMPLES[role], system)
                self.assertNotIn('"answer":"Paris"', system)
                self.assertEqual(usage[0]["format_guidance_version"], ROLE_FORMAT_VERSION)
        provenance = RoleAgents(max_tokens=512).provenance["shared_format_guidance"]
        self.assertEqual(provenance["version"], ROLE_FORMAT_VERSION)
        self.assertEqual(provenance["examples"], ROLE_FORMAT_EXAMPLES)
        self.assertEqual(provenance["text_by_role"], ROLE_FORMAT_GUIDANCE)
        self.assertEqual(hotpot.parse_draft(ROLE_FORMAT_EXAMPLES["writer"])["supporting_facts"],
                         [["EXACT_PUBLIC_TITLE", 0]])
        self.assertIs(hotpot.parse_review(ROLE_FORMAT_EXAMPLES["reviewer"])["approved"], False)

    def test_real_sdk_preserves_raw_alias_output_instead_of_repairing_it(self):
        raw = '```json\n{"answer":"PUBLIC_ANSWER","supporting_facts":[{"exact passage title":"TITLE","zero-based sentence index":0}]}\n```'
        result, usage, _ = self.invoke("writer", content=raw)
        self.assertEqual(result, raw)
        self.assertEqual(usage[0]["status"], "completed")
        with self.assertRaises(ValueError):
            hotpot.parse_draft(result)

    def test_real_sdk_length_termination_retains_usage_but_fails_prediction(self):
        _, usage, requests = self.invoke("writer", finish_reason="length")
        self.assertEqual(len(requests), 1)
        self.assertEqual(usage[0]["status"], "failed")
        self.assertEqual(usage[0]["correlated_usage"]["total_tokens"], 7)
        self.assertTrue(usage[0]["token_accounting_known"])


if __name__ == "__main__":
    unittest.main()
