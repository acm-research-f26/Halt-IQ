"""HotpotQA data preparation checks using fictional, official-shaped fixtures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from haltiq.hotpot import normalize_question, prepare_hotpot


def fixture(size=40):
    return [
        {
            "_id": f"fictional-{index:03}",
            "question": f"Which archive houses fictional item {index}?",
            "answer": f"Archive {index}",
            "type": "comparison" if index % 4 == 0 else "bridge",
            "level": "hard",
            "context": [
                [f"Title {index}-{paragraph}", [f"Original sentence {index}/{paragraph}/0. ",
                                                f" Original sentence {index}/{paragraph}/1."]]
                for paragraph in range(10)
            ],
            "supporting_facts": [[f"Title {index}-2", 0], [f"Title {index}-8", 1]],
        }
        for index in range(size)
    ]


class HotpotPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, rows=None, *, name="prepared", **kwargs):
        source = self.root / f"{name}-source.json"
        source.write_text(json.dumps(fixture() if rows is None else rows), encoding="utf-8")
        directory = self.root / name
        manifest = prepare_hotpot(source, directory, **{"dev_size": 8, "test_size": 16, **kwargs})
        tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text().splitlines()]
        return source, directory, manifest, tasks

    def test_stratified_disjoint_subsets_and_manifest_provenance(self):
        source, directory, manifest, tasks = self.prepare()
        dev = [task for task in tasks if task["split"] == "dev"]
        test = [task for task in tasks if task["split"] == "test"]
        self.assertEqual((len(dev), len(test)), (8, 16))
        self.assertEqual(manifest["splits"]["dev"]["type_counts"], {"bridge": 6, "comparison": 2})
        self.assertEqual(manifest["splits"]["test"]["type_counts"], {"bridge": 12, "comparison": 4})
        self.assertFalse({t["id"] for t in dev} & {t["id"] for t in test})
        self.assertFalse({normalize_question(t["question"]) for t in dev}
                         & {normalize_question(t["question"]) for t in test})
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(manifest["output"]["sha256"], hashlib.sha256((directory / "tasks.jsonl").read_bytes()).hexdigest())
        self.assertEqual(manifest["source"]["split"], "official_dev")
        self.assertEqual(manifest["source"]["context_paragraph_count_distribution"], {"10": 40})
        self.assertEqual(manifest["splits"]["dev"]["context_paragraph_count_distribution"], {"10": 8})
        self.assertEqual(manifest["derived_data_license"], "CC-BY-SA-4.0")
        self.assertFalse(manifest["evaluation"]["test_split_is_official_test"])
        self.assertEqual(json.loads((directory / "manifest.json").read_text()), manifest)
        self.assertIn("not the official hidden test set", (directory / "README.md").read_text())
        self.assertIn("yang2018hotpotqa", (directory / "CITATION.bib").read_text())

    def test_source_order_does_not_change_selection_or_output(self):
        rows = fixture()
        _, first_dir, first_manifest, first = self.prepare(rows, name="first")
        _, second_dir, second_manifest, second = self.prepare(list(reversed(rows)), name="second")
        self.assertEqual(first, second)
        self.assertEqual((first_dir / "tasks.jsonl").read_bytes(), (second_dir / "tasks.jsonl").read_bytes())
        self.assertEqual(first_manifest["splits"], second_manifest["splits"])
        self.assertNotEqual(first_manifest["source"]["sha256"], second_manifest["source"]["sha256"])

    def test_duplicate_questions_excluded_stably_and_counted(self):
        rows = fixture()
        duplicate = deepcopy(rows[0])
        duplicate["_id"] = "zz-duplicate"
        duplicate["question"] = "  WHICH ARCHIVE houses fictional item 0!!! "
        rows.append(duplicate)
        _, _, manifest, tasks = self.prepare(rows, dev_size=20, test_size=20)
        self.assertEqual(len(tasks), 40)
        self.assertEqual(len({normalize_question(task["question"]) for task in tasks}), 40)
        self.assertEqual(manifest["sampling"]["excluded_duplicate_question_count"], 1)
        self.assertEqual(manifest["sampling"]["excluded_duplicate_questions"],
                         [{"source_id": "zz-duplicate", "kept_source_id": "fictional-000"}])
        _, _, reversed_manifest, reversed_tasks = self.prepare(list(reversed(rows)), name="reversed", dev_size=20, test_size=20)
        self.assertEqual(tasks, reversed_tasks)
        self.assertEqual(manifest["sampling"], reversed_manifest["sampling"])

    def test_complete_context_is_preserved_without_oracle_filter_or_scripts(self):
        rows = fixture()
        _, _, _, tasks = self.prepare(rows)
        originals = {row["_id"]: row for row in rows}
        for task in tasks:
            row = originals[task["metadata"]["source_id"]]
            self.assertEqual(task["id"], "hotpotqa:" + row["_id"])
            self.assertEqual(task["benchmark"], "hotpotqa-distractor")
            self.assertEqual(task["metric"], "hotpotqa")
            self.assertEqual(task["answers"], [row["answer"]])
            self.assertEqual(len(task["evidence"]), 10)
            self.assertEqual(task["metadata"]["context_paragraph_count"], 10)
            self.assertEqual(task["metadata"]["supporting_facts"], row["supporting_facts"])
            self.assertNotIn("demo_drafts", task)
            self.assertNotIn("demo_scores", task)
            for paragraph, (title, sentences) in zip(task["evidence"], row["context"]):
                self.assertEqual(paragraph, title + "\n[0] " + sentences[0] + "\n[1] " + sentences[1])

    def test_labels_and_supporting_annotations_do_not_affect_selection(self):
        rows = fixture()
        _, _, manifest, _ = self.prepare(rows)
        for row in rows:
            row["answer"] = "An altered gold label"
            row["supporting_facts"] = [[row["context"][0][0], 1]]
        _, _, changed, _ = self.prepare(rows, name="changed")
        self.assertEqual(manifest["splits"], changed["splits"])

    def test_short_contexts_blank_sentences_and_annotation_errors_remain_eligible(self):
        rows = fixture()
        rows[0]["context"] = rows[0]["context"][:2]
        rows[0]["context"][0][1] = ["", "  ", "Final preserved sentence."]
        rows[1]["supporting_facts"][0][1] = 100
        rows[2]["supporting_facts"] = []
        rows[3]["context"][0][0] = rows[3]["context"][1][0]
        _, _, manifest, tasks = self.prepare(rows, dev_size=20, test_size=20)
        self.assertEqual(len(tasks), 40)
        self.assertEqual(manifest["source"]["context_paragraph_count_distribution"], {"2": 1, "10": 39})
        self.assertEqual(manifest["source"]["annotation_warnings"]["task_count"], 4)
        self.assertEqual(manifest["source"]["annotation_warnings"]["code_counts"], {
            "duplicate_context_title": 1,
            "missing_or_invalid_supporting_facts": 1,
            "supporting_sentence_out_of_range": 1,
            "supporting_title_not_in_context": 2,
        })
        by_id = {task["metadata"]["source_id"]: task for task in tasks}
        self.assertEqual(len(by_id["fictional-000"]["evidence"]), 2)
        self.assertEqual(by_id["fictional-000"]["evidence"][0],
                         "Title 0-0\n[0] \n[1]   \n[2] Final preserved sentence.")
        self.assertEqual(by_id["fictional-001"]["metadata"]["supporting_facts"], rows[1]["supporting_facts"])
        self.assertEqual(by_id["fictional-001"]["metadata"]["annotation_warnings"],
                         [{"code": "supporting_sentence_out_of_range", "fact_index": 0}])
        _, _, original, _ = self.prepare(fixture(), name="before_warnings", dev_size=20, test_size=20)
        for split in ("dev", "test"):
            self.assertEqual(manifest["splits"][split]["source_ids"], original["splits"][split]["source_ids"])

    def test_existing_directory_is_never_overwritten(self):
        source, directory, _, _ = self.prepare()
        sentinel = directory / "sentinel.txt"
        sentinel.write_text("Keep this")
        with self.assertRaises(FileExistsError):
            prepare_hotpot(source, directory)
        self.assertEqual(sentinel.read_text(), "Keep this")

    def test_insufficient_unique_questions_and_invalid_parameters_fail_before_write(self):
        for name, params in (
            ("too_large", {"dev_size": 30, "test_size": 30}),
            ("zero", {"dev_size": 0}),
            ("negative", {"test_size": -1}),
            ("bool", {"dev_size": True}),
            ("fractional", {"test_size": 1.5}),
            ("seed", {"seed": "42"}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.prepare(name=name, **params)
            self.assertFalse((self.root / name).exists())

    def test_entire_source_validated_even_when_sampling_fewer_rows(self):
        invalid_cases = [
            ("id", lambda rows: rows[-1].update(_id=rows[0]["_id"])),
            ("answer", lambda rows: rows[-1].update(answer="  ")),
            ("question", lambda rows: rows[-1].update(question="")),
            ("punctuation_question", lambda rows: rows[-1].update(question="?!?")),
            ("type", lambda rows: rows[-1].update(type="other")),
            ("short_context", lambda rows: rows[-1].update(context=rows[-1]["context"][:1])),
            ("long_context", lambda rows: rows[-1]["context"].append(deepcopy(rows[-1]["context"][0]))),
            ("sentence", lambda rows: rows[-1]["context"][0][1].append(42)),
        ]
        for name, mutate in invalid_cases:
            rows = fixture()
            mutate(rows)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.prepare(rows, name=name, dev_size=1, test_size=1)
            self.assertFalse((self.root / name).exists())
        for name, rows in (("empty", []), ("object", {}), ("not_rows", [None])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.prepare(rows, name=name)
            self.assertFalse((self.root / name).exists())


if __name__ == "__main__":
    unittest.main()
