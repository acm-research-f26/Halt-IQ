"""Offline HotpotQA metric parity and benchmark-label isolation checks.

Expected values follow the official hotpot_evaluate_v1.py answer scorer:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
Alias aggregation is HaltIQ's extension; HotpotQA itself has one gold answer.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from haltiq.metrics import hotpot_evaluate, hotpot_normalize
from haltiq.tasks import Task, evaluate, load_tasks


def benchmark_task(**changes):
    task = Task(
        id="hotpot-unit-test",
        split="dev",
        category="bridge",
        question="Which country is the destination in?",
        evidence=["Destination [0]: The destination is in Exampleland."],
        answers=["PRIVATE_REFERENCE_ANSWER"],
        demo_drafts=[],
        demo_scores=[],
        benchmark="hotpotqa_distractor",
        metric="hotpotqa",
        metadata={
            "supporting_facts": [["PRIVATE_SUPPORT_TITLE", 7]],
            "level": "PRIVATE_DIFFICULTY",
            "source_split": "validation",
        },
    )
    return replace(task, **changes)


class HotpotMetricTests(unittest.TestCase):
    def assert_score(self, answer, gold, exact, f1):
        result = hotpot_evaluate(answer, gold)
        self.assertEqual(result["exact_match"], exact)
        self.assertAlmostEqual(result["token_f1"], f1)

    def test_ascii_punctuation_is_deleted_not_replaced_with_spaces(self):
        self.assert_score("U.S.", ["US"], True, 1.0)
        self.assert_score("New-York", ["New York"], False, 0.0)
        self.assertEqual(hotpot_normalize("New-York"), "newyork")

    def test_articles_lowercase_and_whitespace_follow_official_normalizer(self):
        self.assert_score("  THE\tUnited\n States! ", ["United States"], True, 1.0)
        self.assert_score("An apple and a pear", ["apple and pear"], True, 1.0)
        # Remove complete articles, not the same letters inside another word.
        self.assertEqual(hotpot_normalize("The theatre"), "theatre")

    def test_special_answer_mismatch_disallows_partial_overlap_both_directions(self):
        for special in ("yes", "no", "noanswer"):
            for answer, label in ((special, special + " indeed"), (special + " indeed", special)):
                with self.subTest(answer=answer, label=label):
                    self.assert_score(answer, [label], False, 0.0)
            with self.subTest(special=special):
                self.assert_score(special.upper() + "!", [special], True, 1.0)
        self.assert_score("yes", ["no"], False, 0.0)

    def test_empty_normalized_answers_match_but_have_zero_f1(self):
        self.assert_score("the", ["an"], True, 0.0)
        self.assert_score("", [""], True, 0.0)
        self.assert_score("!!!", ["answer"], False, 0.0)

    def test_unicode_is_lowercased_without_casefold_or_compatibility_normalization(self):
        self.assert_score("Straße", ["STRASSE"], False, 0.0)
        self.assert_score("Ｓ", ["S"], False, 0.0)
        # Official punctuation removal is ASCII-only.
        self.assert_score("New—York", ["NewYork"], False, 0.0)

    def test_aliases_take_best_result_without_special_answer_short_circuit(self):
        self.assert_score("yes", ["no", "YES"], True, 1.0)
        self.assert_score("red blue", ["green", "blue orange"], False, 0.5)
        self.assert_score("United States", ["USA", "the United States"], True, 1.0)

    def test_token_overlap_counts_repeated_tokens(self):
        self.assert_score("red red", ["red"], False, 2 / 3)
        self.assert_score("red blue red", ["red red green"], False, 2 / 3)


class BenchmarkTaskTests(unittest.TestCase):
    def test_task_uses_selected_benchmark_metric_and_preserves_diagnostic_numbers(self):
        hotpot = benchmark_task(answers=["3"])
        diagnostic = replace(hotpot, benchmark="diagnostic", metric="diagnostic")
        self.assertTrue(hotpot.evaluate("-3")["exact_match"])
        self.assertFalse(diagnostic.evaluate("-3")["exact_match"])
        self.assertFalse(evaluate("1.5", ["15"])["exact_match"])
        self.assertFalse(evaluate("\N{MINUS SIGN}3", ["3"])["exact_match"])
        self.assertTrue(evaluate("\N{MINUS SIGN}3", ["-3"])["exact_match"])

    def test_task_state_excludes_reference_answer_and_supporting_fact_metadata(self):
        task = benchmark_task()
        for draft in (None, "Exampleland"):
            with self.subTest(draft=draft):
                state = task.state(draft)
                expected = {"question": task.question, "evidence": task.evidence}
                if draft is not None:
                    expected["draft"] = draft
                self.assertEqual(state, expected)
                serialized = json.dumps(state)
                for private in ("PRIVATE_REFERENCE_ANSWER", "PRIVATE_SUPPORT_TITLE", "PRIVATE_DIFFICULTY"):
                    self.assertNotIn(private, serialized)

    def test_jsonl_loader_retains_metric_selection_and_private_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text(json.dumps(asdict(benchmark_task(answers=["yes"]))) + "\n", encoding="utf-8")
            task = load_tasks(path)[0]
        self.assertEqual(task.benchmark, "hotpotqa_distractor")
        self.assertEqual(task.metadata["supporting_facts"], [["PRIVATE_SUPPORT_TITLE", 7]])
        self.assertEqual(task.evaluate("yes indeed"), {"exact_match": False, "token_f1": 0.0})
        self.assertNotIn("metadata", task.state())


if __name__ == "__main__":
    unittest.main()
