import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from haltiq_pair.core import Agents, Gate
from haltiq_pair.linear import LinearHead
from haltiq_pair.server import parse_tool_calls, stream_response


class CoreTests(unittest.TestCase):
    def test_strands_vetoes_even_when_linear_wants_to_stop(self):
        class Critic:
            def score(self, *_): return 0.1
        gate = Gate(Critic())
        result = gate.decide("combined", "a", None, 1, 6, "question", ["passage"])
        self.assertGreater(result["p_stop"], 0.5)
        self.assertFalse(result["stop"])
        self.assertTrue(gate.decide("linear", "a", None, 1, 6, "q", ["p"])["stop"])

    def test_head_can_veto_the_combined_gate(self):
        class Critic:
            def score(self, *_): return 0.9
        gate = Gate(Critic(), LinearHead((0, 0, 0, 0, 0), -10))
        self.assertFalse(gate.decide("combined", "a", None, 1, 6, "q", ["p"])["stop"])
        self.assertTrue(gate.decide("strands", "a", None, 1, 6, "q", ["p"])["stop"])

    def test_no_critic_call_for_fixed_arms(self):
        class Critic:
            def score(self, *_): raise AssertionError("unexpected physical critic call")
        gate = Gate(Critic())
        self.assertTrue(gate.decide("fixed1", "a", None, 1, 6, "q", ["p"])["stop"])
        self.assertFalse(gate.decide("fixed6", "a", None, 1, 6, "q", ["p"])["stop"])
        self.assertTrue(gate.decide("fixed6", "a", "b", 6, 6, "q", ["p"])["stop"])

    def test_nonfinite_critic_is_a_failure(self):
        class Critic:
            def score(self, *_): return float("nan")
        with self.assertRaises(ValueError):
            Gate(Critic()).decide("combined", "a", None, 1, 6, "q", ["p"])

    def test_tool_markup_is_parsed_without_execution(self):
        text, calls = parse_tool_calls('<tool_call>{"name":"lookup","arguments":{"id":"1"}}</tool_call>')
        self.assertIsNone(text)
        self.assertEqual(calls[0]["function"]["name"], "lookup")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"id": "1"})

    def test_live_strands_sdk_uses_loopback_stream_and_records_tokens(self):
        try:
            import strands.models.openai
        except ImportError:
            self.skipTest("optional Strands SDK is not installed")
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                response = {"id": "test", "created": 1, "model": "fixture-model",
                            "choices": [{"message": {"role": "assistant", "content": "verified"},
                                         "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write("".join(stream_response(response)).encode())
        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            agents = Agents(f"http://127.0.0.1:{httpd.server_port}/v1", "fixture-model")
            self.assertEqual(agents.generate("writer", "system", "question"), "verified")
            self.assertEqual(agents.usage[0]["usage"]["inputTokens"], 3)
            self.assertTrue(bodies[0]["stream"])
            self.assertEqual(len(bodies), 1)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
