"""Real loopback transport against a deterministic stub, not model inference."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from queue import Queue
from threading import Thread
import unittest
from unittest.mock import patch

from haltiq.critics import HINTS, QUESTIONS, JevCritic
from haltiq.providers import JevClient


class JevLoopbackHTTPTests(unittest.TestCase):
    def test_jev_critic_rejects_unsupported_draft_over_actual_http(self):
        records = Queue()
        response = {
            "model": "stub-openjev-for-transport-test",
            "answers": {
                "supported": {"type": "noul", "noul": 0.15},
                "complete": {"type": "noul", "noul": 0.95},
                "relevant": {"type": "noul", "noul": 0.90},
            },
            "usage": {"input_tokens": 187, "output_tokens": 12},
        }

        class StubHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                records.put((self.path, dict(self.headers.items()), json.loads(body)))
                if self.path != "/v1/systemone":
                    self.send_error(404)
                    return
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
        thread.start()
        try:
            client = JevClient(
                base_url=f"http://127.0.0.1:{server.server_address[1]}",
                model="openjev-latest",
                timeout=2,
            )
            state = {
                "question": "Who manages Cedar Library?",
                "evidence": ["Mira manages Cedar Library."],
                "draft": "Noah",
                "answers": ["Mira"],
            }
            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-only-cloud-secret"}):
                # Construct inside the patched environment to exercise the
                # credential decision as well as the actual HTTP transport.
                client = JevClient(base_url=client.base_url, model="openjev-latest", timeout=2)
                review = JevCritic(client, threshold=0.8).review(state)

            path, headers, body = records.get(timeout=1)
            self.assertEqual(path, "/v1/systemone")
            self.assertEqual(headers["Content-Type"], "application/json")
            self.assertNotIn("authorization", {key.lower() for key in headers})
            self.assertEqual(body["model"], "openjev-latest")
            self.assertEqual(body["questions"], QUESTIONS)
            self.assertEqual(body["state"], {key: state[key] for key in ("question", "evidence", "draft")})
            self.assertNotIn("answers", body["state"])
            self.assertFalse(review.approved)
            self.assertEqual(review.feedback, HINTS["supported"])
            self.assertEqual(review.scores, {"supported": 0.15, "complete": 0.95, "relevant": 0.90})
            self.assertEqual(review.call.data, response)
            self.assertEqual(review.call.input_tokens, 187)
            self.assertEqual(review.call.output_tokens, 12)
            self.assertEqual(review.call.cost_usd, 0.0)
            self.assertGreater(review.call.latency_ms, 0)
            self.assertEqual(review.call.model, "stub-openjev-for-transport-test")
            self.assertTrue(records.empty())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
