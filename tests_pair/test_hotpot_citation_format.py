"""Unambiguous citation encodings share one gold-blind canonical form."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from haltiq_pair.hotpot import CITATION_FORMAT_POLICY, parse_draft, run_hotpot


class CitationFormatTests(unittest.TestCase):
    def parse_facts(self, facts):
        return parse_draft(json.dumps({"answer": "Unchanged answer", "supporting_facts": facts}))

    def test_exact_object_encodings_equal_the_existing_list_pairs(self):
        expected = self.parse_facts([["Exact Passage Title", 0], ["Other title", 3]])
        for title_key in ("exact_passage_title", "title"):
            with self.subTest(title_key=title_key):
                actual = self.parse_facts([{title_key: "Exact Passage Title", "sentence_index": 0},
                                           {"sentence_index": 3, title_key: "Other title"}])
                self.assertEqual(actual, expected)
                self.assertTrue(all(type(fact) is list for fact in actual["supporting_facts"]))
        mixed = self.parse_facts([["Exact Passage Title", 0],
                                  {"title": "Other title", "sentence_index": 3}])
        self.assertEqual(mixed, expected)
        mixed_all = self.parse_facts([["Exact Passage Title", 0],
                                      {"title": "Other title", "sentence_index": 3},
                                      {"exact_passage_title": "Exact Passage Title", "sentence_index": 0}])
        self.assertEqual(mixed_all["supporting_facts"],
                         [["Exact Passage Title", 0], ["Other title", 3], ["Exact Passage Title", 0]])

    def test_preserves_titles_and_indices_without_matching_or_coercion(self):
        title = "  Unmatched Title, Exact Case! \u03b1  "
        result = self.parse_facts([{"exact_passage_title": title, "sentence_index": 999999}])
        self.assertEqual(result, {"answer": "Unchanged answer", "supporting_facts": [[title, 999999]]})
        self.assertEqual(self.parse_facts([])["supporting_facts"], [])

    def test_rejects_ambiguous_extra_missing_or_unknown_object_fields(self):
        invalid = [
            {}, {"title": "A"}, {"sentence_index": 0},
            {"exact_passage_title": "A", "title": "A", "sentence_index": 0},
            {"title": "A", "sentence_index": 0, "confidence": 1},
            {"exact_passage_title": "A", "sentence_index": 0, "extra": None},
            {"passage_title": "A", "sentence_index": 0},
            {"title": "A", "index": 0},
        ]
        for fact in invalid:
            with self.subTest(fact=fact), self.assertRaises(ValueError):
                self.parse_facts([fact])

    def test_rejects_bad_types_and_shapes_for_every_encoding(self):
        for title_key in (None, "title", "exact_passage_title"):
            for index in (True, False, "0", 0.0, -1, None, [], {}):
                fact = ["A", index] if title_key is None else {title_key: "A", "sentence_index": index}
                with self.subTest(title_key=title_key, index=index), self.assertRaises(ValueError):
                    self.parse_facts([fact])
            for title in (None, 4, True, [], {}):
                fact = [title, 0] if title_key is None else {title_key: title, "sentence_index": 0}
                with self.subTest(title_key=title_key, title=title), self.assertRaises(ValueError):
                    self.parse_facts([fact])
        for fact in (None, "A", 0, [], ["A"], ["A", 0, "extra"]):
            with self.subTest(fact=fact), self.assertRaises(ValueError):
                self.parse_facts([fact])

    def test_manifest_freezes_the_policy_and_refuses_old_policy_resume(self):
        class Agents:
            model_id = "offline-citation-format-fixture"
            def __init__(self):
                self.usage, self.calls = [], 0
            def generate(self, *args):
                self.calls += 1
                return json.dumps({"answer": "Unchanged answer", "supporting_facts": [
                    {"exact_passage_title": "Candidate", "sentence_index": 0}]})
        class Gate:
            provenance = {"identity": "offline-citation-format-fixture"}
            def decide(self, **kwargs):
                return {"stop": True}
        fixture = {"id": "synthetic-citation-format", "question": "Fixture question?",
                   "answer": "Unchanged answer", "type": "bridge", "level": "hard",
                   "context": [["Candidate", ["Fixture candidate sentence."]]],
                   "supporting_facts": [["Candidate", 0]]}
        agents = Agents()
        kwargs = {"limit": 1, "rows": [fixture], "arms": ("fixed1",), "excluded_ids": ()}
        with tempfile.TemporaryDirectory() as directory:
            summary = run_hotpot(agents, Gate(), directory, **kwargs)
            self.assertEqual(summary["arms"]["fixed1"]["completed"], 1)
            manifest_path = Path(directory) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["config"]["citation_format_policy"], CITATION_FORMAT_POLICY)
            calls = agents.calls
            run_hotpot(agents, Gate(), directory, resume=True, **kwargs)
            self.assertEqual(agents.calls, calls)
            for old_policy in (None, dict(CITATION_FORMAT_POLICY, version=1)):
                old_manifest = deepcopy(manifest)
                if old_policy is None:
                    old_manifest["config"].pop("citation_format_policy")
                else:
                    old_manifest["config"]["citation_format_policy"] = old_policy
                manifest_path.write_text(json.dumps(old_manifest))
                with self.subTest(old_policy=old_policy), self.assertRaisesRegex(ValueError, "configuration changed"):
                    run_hotpot(agents, Gate(), directory, resume=True, **kwargs)
                self.assertEqual(agents.calls, calls)


if __name__ == "__main__":
    unittest.main()
