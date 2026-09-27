from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from haltiq.budget import BudgetError, BudgetExceeded, BudgetLedger, JEV_MAX_INPUT_TOKENS
from haltiq.providers import JevClient, OllamaClient, ProviderError, _NoRedirect


QUESTIONS = {"supported": {"type": "noul", "instructions": "Is the answer supported by evidence?"}}
JEV_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {"supported": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 1000, "output_tokens": 10},
}
OLLAMA_RESPONSE = {
    "model": "qwen3:8b", "done": True, "done_reason": "stop",
    "message": {"content": '{"answer":"B"}'}, "prompt_eval_count": 50, "eval_count": 8,
}
OLLAMA_TAGS = {"models": [{"name": "qwen3:8b", "model": "qwen3:8b", "size": 5_000_000_000}]}


class FakeResponse(io.BytesIO):
    pass


def fake_response(data):
    return FakeResponse(json.dumps(data).encode())


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.sqlite"

    def test_reserve_settle_and_reopen_preserve_accounting(self):
        ledger = BudgetLedger(self.path)
        reservation = ledger.reserve()
        self.assertAlmostEqual(ledger.snapshot()["reserved_usd"], JEV_MAX_INPUT_TOKENS * 0.042 / 1_000_000)
        self.assertAlmostEqual(ledger.settle(reservation, 1000), 0.000042)
        reopened = BudgetLedger(self.path, limit_usd=10)
        self.assertAlmostEqual(reopened.snapshot()["spent_usd"], 0.000042)
        self.assertEqual(reopened.snapshot()["reserved_usd"], 0)
        self.assertEqual(reopened.snapshot()["attempts"], 1)

    def test_concurrent_reservations_never_exceed_limit(self):
        ledger = BudgetLedger(self.path, limit_usd=0.03)
        def reserve(_):
            try:
                return ledger.reserve()
            except BudgetExceeded:
                return None
        with ThreadPoolExecutor(max_workers=16) as executor:
            accepted = list(executor.map(reserve, range(50)))
        self.assertEqual(sum(item is not None for item in accepted), 10)
        snapshot = ledger.snapshot()
        self.assertLessEqual(snapshot["used_usd"], 0.03)
        self.assertEqual(snapshot["attempts"], 10)

    def test_invalid_limits_and_rates(self):
        for value in (0, -1, 31, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(BudgetError):
                BudgetLedger(self.path, limit_usd=value)
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(BudgetError):
                BudgetLedger(self.path, rate_per_million=rate)

    def test_changed_rate_cannot_reset_ledger(self):
        ledger = BudgetLedger(self.path)
        ledger.reserve()
        with self.assertRaises(BudgetError):
            BudgetLedger(self.path, rate_per_million=0.0042)
        self.assertEqual(ledger.snapshot()["attempts"], 1)

    def test_oversized_usage_and_double_settlement_cannot_release_reservation(self):
        ledger = BudgetLedger(self.path)
        reservation = ledger.reserve()
        with self.assertRaises(BudgetError):
            ledger.settle(reservation, JEV_MAX_INPUT_TOKENS + 1)
        self.assertEqual(ledger.snapshot()["unsettled_requests"], 1)
        ledger.settle(reservation, 100)
        ledger.settle(reservation, 100)
        with self.assertRaises(BudgetError):
            ledger.settle(reservation, 0)
        self.assertEqual(ledger.snapshot()["input_tokens"], 100)

    def test_lowered_limit_still_counts_earlier_spending(self):
        ledger = BudgetLedger(self.path, limit_usd=1)
        ledger.reserve()
        smaller = BudgetLedger(self.path, limit_usd=0.001)
        with self.assertRaises(BudgetExceeded):
            smaller.reserve()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = BudgetLedger(Path(self.temp.name) / "ledger.sqlite")

    def client(self):
        return JevClient(api_key="test-key", budget=self.ledger)

    @patch("haltiq.providers.build_opener")
    def test_jev_request_and_response(self, factory):
        factory.return_value.open.return_value = fake_response(JEV_RESPONSE)
        result = self.client().judge({"evidence": "B"}, QUESTIONS)
        request = factory.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(json.loads(request.data)["questions"], QUESTIONS)
        self.assertEqual(result.data, JEV_RESPONSE)
        self.assertAlmostEqual(result.cost_usd, 0.000042)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(self.ledger.snapshot()["successful_requests"], 1)

    @patch("haltiq.providers.build_opener")
    def test_timeout_is_one_attempt_and_keeps_reservation(self, factory):
        factory.return_value.open.side_effect = URLError("sensitive-server-detail")
        with self.assertRaises(ProviderError) as error:
            self.client().judge({}, QUESTIONS)
        self.assertNotIn("sensitive", str(error.exception))
        self.assertEqual(factory.return_value.open.call_count, 1)
        self.assertEqual(self.ledger.snapshot()["unsettled_requests"], 1)

    @patch("haltiq.providers.build_opener")
    def test_http_rate_limit_is_not_retried(self, factory):
        factory.return_value.open.side_effect = HTTPError("unused", 429, "rate limit", {}, io.BytesIO(b"secret"))
        with self.assertRaisesRegex(ProviderError, "HTTP 429"):
            self.client().judge({}, QUESTIONS)
        self.assertEqual(factory.return_value.open.call_count, 1)
        self.assertEqual(self.ledger.snapshot()["unsettled_requests"], 1)

    @patch("haltiq.providers.build_opener")
    def test_malformed_jev_responses_keep_reservation(self, factory):
        malformed = [
            {**JEV_RESPONSE, "answers": {}},
            {**JEV_RESPONSE, "usage": {}},
            {**JEV_RESPONSE, "usage": {"input_tokens": True, "output_tokens": 0}},
            {**JEV_RESPONSE, "usage": {"input_tokens": -1, "output_tokens": 0}},
            {**JEV_RESPONSE, "model": "unexpected"},
        ]
        for probability in (True, "0.9", -0.1, 1.1, float("nan"), float("inf")):
            malformed.append({**JEV_RESPONSE, "answers": {"supported": {"type": "noul", "noul": probability}}})
        for data in malformed:
            with self.subTest(data=data):
                factory.return_value.open.return_value = fake_response(data)
                with self.assertRaises(ProviderError):
                    self.client().judge({}, QUESTIONS)
        self.assertEqual(self.ledger.snapshot()["unsettled_requests"], len(malformed))
        self.assertEqual(self.ledger.snapshot()["spent_usd"], 0)

    @patch("haltiq.providers.build_opener")
    def test_budget_denial_makes_no_http_call(self, factory):
        ledger = BudgetLedger(Path(self.temp.name) / "tiny.sqlite", limit_usd=0.001)
        with self.assertRaises(BudgetExceeded):
            JevClient(api_key="test-key", budget=ledger).judge({}, QUESTIONS)
        factory.assert_not_called()

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "cloud-secret"})
    @patch("haltiq.providers.build_opener")
    def test_local_openjev_never_forwards_cloud_key_or_charges(self, factory):
        factory.return_value.open.return_value = fake_response({**JEV_RESPONSE, "model": "openjev-0.1"})
        result = JevClient(base_url="http://127.0.0.1:8080", model="openjev-latest").judge({}, QUESTIONS)
        request = factory.return_value.open.call_args.args[0]
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(result.cost_usd, 0)
        self.assertEqual(self.ledger.snapshot()["attempts"], 0)

    def test_reject_remote_credential_endpoints(self):
        for url in ("https://example.com", "https://api.typesafe.ai.evil.test", "http://api.typesafe.ai", "https://api.typesafe.ai:444", "https://secret@api.typesafe.ai", "http://localhost:8080/v1", "http://localhost:8080?key=secret"):
            with self.subTest(url=url), self.assertRaises(ProviderError):
                JevClient(base_url=url, api_key="test-key", budget=self.ledger)
        with self.assertRaises(ProviderError):
            OllamaClient(base_url="https://api.typesafe.ai")
        with self.assertRaises(ProviderError):
            OllamaClient(model="gemma4:cloud")

    def test_paid_calls_require_key_explicit_budget_and_pinned_model(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ProviderError):
                JevClient(budget=self.ledger)
        with self.assertRaises(ProviderError):
            JevClient(api_key="test-key")
        with self.assertRaises(ProviderError):
            JevClient(api_key="test-key", budget=self.ledger, model="jev-latest")
        with self.assertRaises(ProviderError):
            JevClient(api_key="test-key\n", budget=self.ledger)

    @patch("haltiq.providers.build_opener")
    def test_ollama_uses_native_structured_local_request(self, factory):
        factory.return_value.open.side_effect = [fake_response(OLLAMA_TAGS), fake_response(OLLAMA_RESPONSE)]
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = OllamaClient(seed=123).chat([{"role": "user", "content": "Question?"}], schema)
        request = factory.return_value.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["seed"], 123)
        self.assertEqual(payload["options"]["num_predict"], 512)
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["format"], schema)
        self.assertEqual(result.data, {"answer": "B"})
        self.assertEqual(result.cost_usd, 0)

    @patch("haltiq.providers._http_json")
    def test_ollama_large_context_preserves_evidence_and_records_estimate(self, http):
        http.side_effect = [OLLAMA_TAGS, OLLAMA_RESPONSE]
        messages = [{"role": "user", "content": "Evidence. " * 1400}]
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = OllamaClient(num_ctx=32768).chat(messages, schema)
        payload = json.loads(http.call_args.args[1])
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["format"], schema)
        self.assertEqual(payload["options"]["num_ctx"], 32768)
        self.assertEqual(result.context_tokens, 32768)
        self.assertGreater(result.context_budget_estimate_tokens, 14000)
        self.assertLess(result.context_budget_estimate_tokens, 32768)
        self.assertEqual(result.input_tokens, 50)

    @patch("haltiq.providers._http_json")
    def test_ollama_context_guard_rejects_before_any_network_call(self, http):
        oversized = [
            ([{"role": "user", "content": "x" * 8192}], None),
            ([{"role": "user", "content": "hi"}], {"description": "x" * 8192}),
            # UTF-8 bytes matter: these 2000 characters occupy 8000 bytes.
            ([{"role": "user", "content": "\U0001f30d" * 2000}], None),
        ]
        for messages, schema in oversized:
            with self.subTest(schema=schema is not None, content_length=len(messages[0]["content"])):
                with self.assertRaisesRegex(ProviderError, "--context-tokens"):
                    OllamaClient().chat(messages, schema)
        http.assert_not_called()

    def test_ollama_context_size_validation(self):
        for value in (True, False, 8192.0, "8192", None, 2047, 65537, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ProviderError, "num_ctx"):
                OllamaClient(num_ctx=value)
        self.assertEqual(OllamaClient(num_ctx=2048).num_ctx, 2048)
        self.assertEqual(OllamaClient(num_ctx=65536).num_ctx, 65536)

    @patch("haltiq.providers.build_opener")
    def test_ollama_rejects_truncated_empty_or_non_json(self, factory):
        for change in (
            {"done": False}, {"done_reason": "length"}, {"message": {"content": ""}},
            {"message": {"content": "not json"}}, {"message": {"content": "[]"}},
            {"prompt_eval_count": -1}, {"eval_count": True},
        ):
            with self.subTest(change=change):
                factory.return_value.open.side_effect = [fake_response(OLLAMA_TAGS), fake_response({**OLLAMA_RESPONSE, **change})]
                with self.assertRaises(ProviderError):
                    OllamaClient().chat([{"role": "user", "content": "hi"}], {})

    @patch("haltiq.providers.build_opener")
    def test_ollama_remote_alias_is_blocked_before_generation(self, factory):
        factory.return_value.open.return_value = fake_response({"models": [
            {"name": "qwen3:8b", "size": 1024, "remote_host": "https://ollama.com", "remote_model": "qwen3"}
        ]})
        with self.assertRaisesRegex(ProviderError, "remote alias"):
            OllamaClient().chat([{"role": "user", "content": "hi"}])
        self.assertEqual(factory.return_value.open.call_count, 1)
        self.assertTrue(factory.return_value.open.call_args.args[0].full_url.endswith("/api/tags"))

    @patch("haltiq.providers.build_opener")
    def test_ollama_missing_local_weights_blocks_generation(self, factory):
        for models in ([], [{"name": "qwen3:8b", "size": 0}]):
            with self.subTest(models=models):
                factory.return_value.open.return_value = fake_response({"models": models})
                with self.assertRaises(ProviderError):
                    OllamaClient().chat([{"role": "user", "content": "hi"}])

    def test_no_redirect_handler_rejects_redirect(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com"))


if __name__ == "__main__":
    unittest.main()
