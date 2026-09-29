"""Wrapper tests with a fake scorer; run in the separate OpenJev environment."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient

from scripts.serve_openjev import create_app


RUNTIME = {"implementation": "test-only", "upstream_revision": "test-commit", "model_id": "test-gemma",
           "model_repository": "test-only", "model_revision": "test-revision", "backend": "mlx",
           "max_prompt_tokens": 500, "files": {}, "packages": {}, "model_path": "unused", "device": "auto"}
REQUEST = {"model": "openjev-latest", "state": {"draft": "a short answer"},
           "questions": {"supported": {"type": "noul", "instructions": "Is it supported?"}}}


class FakeScorer:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.active = 0
        self.overlapped = False
        self.last_timing = {}

    def context_ids(self, context):
        return list(context)

    def option_ids(self, option):
        return [1]

    def score(self, context, options, **kwargs):
        self.active += 1
        if self.active > 1:
            self.overlapped = True
        time.sleep(0.005)
        self.calls.append(context)
        self.last_timing = {"context_tokens": len(context), "option_tokens": len(options)}
        self.active -= 1
        return [SimpleNamespace(probability=0.9), SimpleNamespace(probability=0.1)]


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.scorer = FakeScorer()
        self.client = self.enterContext(TestClient(create_app(deepcopy(RUNTIME), lambda *args, **kwargs: self.scorer)))

    def test_pinned_model_and_metadata_are_returned_instead_of_alias(self):
        response = self.client.post("/v1/systemone", json=REQUEST)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "test-gemma")
        self.assertEqual(payload["answers"]["supported"], {"type": "noul", "noul": 0.9})
        self.assertEqual(payload["usage"]["output_tokens"], 2)
        self.assertEqual(payload["openjev_metadata"]["upstream_revision"], "test-commit")
        self.assertTrue(self.client.get("/health").json()["ready"])

    def test_unknown_model_is_rejected_before_scoring(self):
        response = self.client.post("/v1/systemone", json={**REQUEST, "model": "jev-1.13.0"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.scorer.calls), 1)  # startup warm-up only

    def test_long_input_is_rejected_before_scoring_and_not_truncated(self):
        response = self.client.post("/v1/systemone", json={**REQUEST, "state": {"evidence": "x" * 1000}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("not truncated", response.json()["detail"])
        self.assertEqual(len(self.scorer.calls), 1)

    def test_unsupported_or_excess_questions_are_rejected(self):
        for questions in ({"q": {"type": "choice", "criteria": {"yes": "", "no": ""}}},
                          {str(i): {"type": "noul"} for i in range(4)}):
            with self.subTest(questions=questions):
                response = self.client.post("/v1/systemone", json={**REQUEST, "questions": questions})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.scorer.calls), 1)

    def test_concurrent_requests_cannot_overlap_shared_scorer(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _: self.client.post("/v1/systemone", json=REQUEST), range(8)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertFalse(self.scorer.overlapped)


if __name__ == "__main__":
    unittest.main()
