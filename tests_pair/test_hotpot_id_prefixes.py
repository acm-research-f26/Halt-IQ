"""Teammate ID prefixes must match the public dataset's bare identifiers."""
from pathlib import Path
import tempfile
import unittest

from haltiq_pair.hotpot import (_bare_id, select_questions, split_public_gold,
                                teammate_exclusions)


class HotpotIdPrefixTests(unittest.TestCase):
    def test_both_known_prefixes_canonicalize_to_the_same_public_id(self):
        qid = "5a83b1e75542990548d0b220"
        for spelling in (qid, "hotpotqa:" + qid, "hotpot_" + qid):
            with self.subTest(spelling=spelling):
                self.assertEqual(_bare_id(spelling), qid)
        self.assertEqual(_bare_id("unrecognized_" + qid), "unrecognized_" + qid)

    def test_all_three_teammate_sources_merge_duplicate_prefix_spellings(self):
        first, second, third = (str(index) * 24 for index in (1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "strands").mkdir()
            (root / "data" / "haltiq_sri_trajectory_strands.csv").write_text(
                f"id,unused\nhotpot_{first},fixture\nhotpotqa:{second},fixture\n")
            (root / "data" / "sri_rounds_with_strands.csv").write_text(
                f"id,unused\n{first},fixture\nhotpot_{second},fixture\n")
            (root / "strands" / "dev20_ids.txt").write_text(f"hotpotqa:{third}\nhotpot_{third}\n")
            self.assertEqual(teammate_exclusions(root), {first, second, third})

    def test_prefixed_exclusions_actually_remove_bare_dataset_rows(self):
        ids = [str(index) * 24 for index in (1, 2, 3)]
        rows = [{"id": qid, "type": "bridge", "level": "hard"} for qid in ids]
        selected = select_questions(rows, 1, 42, ["hotpot_" + ids[0], "hotpotqa:" + ids[1]])
        self.assertEqual([item["id"] for item in selected], [ids[2]])
        self.assertEqual(selected, select_questions(rows, 1, 42, ids[:2]))

    def test_prefixed_dataset_ids_use_the_same_public_identity(self):
        qid = "5a83b1e75542990548d0b220"
        fixture = {"id": "hotpot_" + qid, "question": "Synthetic question?", "answer": "Fixture",
                   "context": [["Candidate", ["Synthetic sentence."]]],
                   "supporting_facts": [["Candidate", 0]]}
        public, _ = split_public_gold(fixture)
        self.assertEqual(public.id, qid)


if __name__ == "__main__":
    unittest.main()
