"""Request/profile software controls only; no models or provider requests."""
from copy import deepcopy
import json
import subprocess
import sys
import unittest

from haltiq_tau_hotpot.generation import (MAIN_PROFILE, LEGACY_PROFILE, REVIEW_SEQUENCE,
    WRITER_SYSTEM, REVIEWER_SYSTEM, ANSWER_SCHEMA, generation_payload, generation_profile,
    parse_response, response_metadata, synthetic_reviews)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"purpose": "main_study", "generation_profile": deepcopy(MAIN_PROFILE)}
        self.state = {"question": "Synthetic?", "evidence": "invented"}

    def test_legacy_requests_preserve_original_prompts_schema_and384_token_budget(self):
        for purpose in (None, "qualification_pilot", "qualification_recovery"):
            config = {"purpose": purpose}
            for review in (False, True):
                payload = generation_payload(config, self.state, "feedback", review=review)
                self.assertEqual(payload["messages"], [{"role": "system", "content": REVIEWER_SYSTEM if review else WRITER_SYSTEM},
                    {"role": "user", "content": json.dumps({"state": self.state, "feedback": "feedback"}, ensure_ascii=False)}])
                self.assertEqual(payload["options"], {"temperature": 0, "seed": 42, "num_predict": 384,
                    "num_ctx": 16384, "num_thread": 4, "num_gpu": 99})
                self.assertEqual("format" in payload, not review)
                if not review: self.assertEqual(payload["format"], ANSWER_SCHEMA)

    def test_new_profile_changes_only_reviewer_instruction_and_role_budget(self):
        old = generation_payload({}, self.state)
        self.assertEqual(generation_payload(self.config, self.state), old)
        new = generation_payload(self.config, self.state, review=True)
        self.assertEqual(new["options"]["num_predict"], 768)
        self.assertEqual(new["messages"][0]["content"], REVIEWER_SYSTEM + " " + MAIN_PROFILE["reviewer_instruction"])
        self.assertEqual(new["messages"][1], generation_payload({}, self.state, review=True)["messages"][1])
        self.assertEqual(self.config["generation_profile"], MAIN_PROFILE)

    def test_legacy_recovery_cannot_silently_enable_future_profile(self):
        for purpose in (None, "qualification_pilot", "qualification_recovery"):
            with self.subTest(purpose=purpose), self.assertRaisesRegex(ValueError, "legacy"):
                generation_payload({"purpose": purpose, "generation_profile": MAIN_PROFILE}, self.state)
        for purpose in ("main_study", "runtime_probe"):
            with self.assertRaises(ValueError): generation_payload({"purpose": purpose}, self.state)
        changed = deepcopy(self.config)
        changed["generation_profile"]["reviewer_max_tokens"] = 1536
        with self.assertRaises(ValueError): generation_payload(changed, self.state)

    def test_only_synthetic_reviewer_ab_can_override_the_profile(self):
        config = {**self.config, "purpose": "runtime_probe"}
        self.assertEqual(generation_payload(config, self.state, review=True, profile="legacy"),
                         generation_payload({}, self.state, review=True))
        self.assertEqual(generation_payload(config, self.state, review=True, profile="main"),
                         generation_payload(self.config, self.state, review=True))
        for bad_config, review, profile in ((self.config, True, "legacy"), (config, False, "legacy"),
                                            (config, True, "unknown"), ({}, True, "main")):
            with self.assertRaises(ValueError): generation_payload(bad_config, self.state, review=review, profile=profile)

    def test_shared_parser_preserves_truncation_and_nonempty_schema_failures(self):
        raw = {"done": True, "done_reason": "length", "message": {"content": "unfinished"}}
        with self.assertRaisesRegex(ValueError, "output budget"): parse_response(raw, review=True)
        raw.update(done_reason="stop", message={"content": ""})
        with self.assertRaisesRegex(ValueError, "empty"): parse_response(raw, review=True)
        raw["message"]["content"] = "Retain."
        self.assertEqual(parse_response(raw, review=True), "Retain.")
        raw.update(prompt_eval_count=16000, eval_count=384)
        with self.assertRaisesRegex(ValueError, "truncation"): parse_response(raw, review=True)

    def test_metadata_is_unicode_text_not_a_quality_or_native_score(self):
        metadata = response_metadata({"done": True, "done_reason": "length", "prompt_eval_count": 10,
            "eval_count": 384, "message": {"content": "Synthetic café feedback."}})
        self.assertEqual(metadata["characters"], 24)
        self.assertEqual(metadata["words"], 3)
        self.assertNotIn("correct", metadata)
        self.assertEqual(REVIEW_SEQUENCE, (("retain", "legacy"), ("retain", "main"), ("two-hop", "main"), ("two-hop", "legacy")))
        self.assertEqual(set(synthetic_reviews()), {"retain", "two-hop"})
        self.assertTrue(all(set(state) == {"question", "evidence", "draft"} for state in synthetic_reviews().values()))

    def test_import_performs_no_model_or_tau_initialization(self):
        code = "import sys;import haltiq_tau_hotpot.generation;assert not any(x in sys.modules for x in ('torch','transformers','tau2'))"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__": unittest.main()
