"""Offline study-integrity controls; no model or GPU calls."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from haltiq_pair import hotpot, nov2_protocol as protocol


def row(index, *, kind="bridge", level="hard", answer="Paris"):
    return {"id": f"{index:024x}", "question": f"Where was author {index} born?",
            "type": kind, "level": level, "answer": answer,
            "context": {"title": ["Biography", "Distractor"],
                        "sentences": [["The author was born in Paris."], ["Other evidence."]]},
            "supporting_facts": {"title": ["Biography"], "sent_id": [0]}}


def record(qid, arm="combined", status="completed", *, tokens=30, approved=True):
    return {"question_id": qid, "arm": arm, "status": status,
            "prediction": {"answer": "Paris", "supporting_facts": [["Biography", 0]]},
            "metrics": {name: 1.0 for name in hotpot.METRICS},
            "iterations": 2, "agent_calls": {"writer": 2, "reviewer": 1, "challenger": 1},
            "attempted_agent_calls": 4, "total_tokens": tokens, "elapsed_seconds": 1.5,
            "actions": ["INVOKE", "REVISE", "STOP"], "termination": "approved" if approved else "budget_cap",
            "gate_approved": approved}


class FrozenPlanTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(i, kind="comparison" if i < 20 else "bridge") for i in range(100)]

    def test_deterministic_stratified_whole_question_disjoint_partitions(self):
        plan = protocol.build_plan(self.rows, (), seed=7)
        self.assertEqual(plan, protocol.build_plan(reversed(self.rows), (), seed=7))
        resolved = protocol.resolve_plan(plan, reversed(self.rows))
        groups = [plan["partitions"][name]["question_ids"] for name in protocol.PARTITIONS]
        self.assertEqual(list(map(len, groups)), [20, 10, 20])
        self.assertEqual(len(set().union(*map(set, groups))), 50)
        self.assertEqual(plan["partitions"]["train"]["strata"], [
            {"type": "bridge", "level": "hard", "count": 16},
            {"type": "comparison", "level": "hard", "count": 4}])
        for name in protocol.PARTITIONS:
            self.assertEqual([public.id for public, _ in resolved[name]], plan["partitions"][name]["question_ids"])
            self.assertNotIn("answer", asdict(resolved[name][0][0]))
        changed_seed = protocol.build_plan(self.rows, (), seed=8)
        self.assertNotEqual(groups[0], changed_seed["partitions"]["train"]["question_ids"])

    def test_both_prefixes_exclude_canonical_ids_and_all_input_duplicates_rejected(self):
        excluded = ["hotpot_" + self.rows[0]["id"], "hotpotqa:" + self.rows[1]["id"]]
        plan = protocol.build_plan(self.rows, excluded)
        self.assertEqual(plan["excluded_ids"], [self.rows[0]["id"], self.rows[1]["id"]])
        for partition in plan["partitions"].values():
            self.assertFalse(set(partition["question_ids"]) & set(plan["excluded_ids"]))
        duplicate = dict(self.rows[0], id="hotpot_" + self.rows[0]["id"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            protocol.build_plan(self.rows + [duplicate], excluded)
        decorated = [dict(item, id=" hotpot_" + item["id"] + " ") for item in self.rows]
        self.assertEqual(protocol.build_plan(decorated, excluded), plan)

    def test_gold_changes_only_label_digest_not_ids_or_public_hash(self):
        first = protocol.build_plan(self.rows, ())
        changed = [dict(item, answer="Different hidden answer", supporting_facts=[["Distractor", 0]])
                   for item in self.rows]
        second = protocol.build_plan(changed, ())
        for name in protocol.PARTITIONS:
            old, new = first["partitions"][name], second["partitions"][name]
            self.assertEqual(old["question_ids"], new["question_ids"])
            self.assertEqual(old["public_inputs_sha256"], new["public_inputs_sha256"])
            self.assertEqual(old["strata"], new["strata"])
            self.assertNotEqual(old["label_digest"], new["label_digest"])

    def test_all_partition_selection_finishes_before_gold_is_read(self):
        select = hotpot.select_questions
        split = hotpot.split_public_gold
        selected = []

        def observed_select(*args, **kwargs):
            value = select(*args, **kwargs)
            selected.append(value)
            return value

        def guarded_split(value):
            self.assertEqual(len(selected), 3)
            return split(value)

        with patch.object(hotpot, "select_questions", side_effect=observed_select), \
                patch.object(hotpot, "split_public_gold", side_effect=guarded_split):
            protocol.build_plan(self.rows, ())

    def test_resolve_rejects_pin_public_label_stratum_and_id_drift(self):
        plan = protocol.build_plan(self.rows, ())
        qid = plan["partitions"]["test"]["question_ids"][0]
        for field, new_value in [("answer", "changed"), ("question", "changed"), ("type", "changed")]:
            with self.subTest(field=field):
                rows = [dict(item, **{field: new_value}) if item["id"] == qid else item for item in self.rows]
                with self.assertRaisesRegex(ValueError, "changed"):
                    protocol.resolve_plan(plan, rows)
        changed = deepcopy(plan)
        changed["dataset"]["revision"] = "unpinned"
        with self.assertRaisesRegex(ValueError, "pin"):
            protocol.resolve_plan(changed, self.rows)
        with self.assertRaisesRegex(ValueError, "missing"):
            protocol.resolve_plan(plan, [item for item in self.rows if item["id"] != qid])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            protocol.resolve_plan(plan, self.rows + [self.rows[0]])
        for ids in [[qid, qid], [plan["partitions"]["train"]["question_ids"][0]], ["hotpot_" + qid]]:
            changed = deepcopy(plan)
            changed["partitions"]["test"]["question_ids"] = ids
            with self.assertRaisesRegex(ValueError, "canonical"):
                protocol.resolve_plan(changed, self.rows)

    def test_invalid_counts_insufficient_rows_and_empty_unused_partition(self):
        for kwargs in ({"train_count": True}, {"dev_count": -1}, {"test_count": 1.5}, {"seed": False},
                       {"train_count": 0, "dev_count": 0, "test_count": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                protocol.build_plan(self.rows, (), **kwargs)
        with self.assertRaisesRegex(ValueError, "eligible"):
            protocol.build_plan(self.rows[:49], ())
        plan = protocol.build_plan(self.rows, (), train_count=0, dev_count=0, test_count=2)
        self.assertEqual(protocol.resolve_plan(plan, self.rows)["train"], [])
        json.dumps(plan, allow_nan=False)


class ExclusionTests(unittest.TestCase):
    def test_only_teammate_id_columns_and_explicit_nested_json_ids_are_used(self):
        ids = [f"{index:024x}" for index in range(1, 9)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "strands").mkdir()
            (root / "data/haltiq_sri_trajectory_strands.csv").write_text(
                f"id,answer,strands\nhotpot_{ids[0]},{ids[7]},not-a-score\n", encoding="utf-8")
            (root / "data/sri_rounds_with_strands.csv").write_text(
                f"id,f1\nhotpotqa:{ids[0]},not-a-number\n{ids[1]},bad\n", encoding="utf-8")
            (root / "strands/dev20_ids.txt").write_text(ids[2] + "\n", encoding="utf-8")
            path = root / "manifest.json"
            path.write_text(json.dumps({"config": {"question_ids": [ids[3], "hotpot_" + ids[4]]},
                "records": [{"question_id": ids[5]}, {"nested": {"ids": [ids[6]], "id": "not-a-hotpot-id"}}, {"id": ""}],
                "answer": ids[7], "unrelated": [ids[7]], "metrics": {ids[7]: 1.0}}), encoding="utf-8")
            journal = root / "explicit.jsonl"
            journal.write_text(json.dumps({"id": ids[0]}) + "\n", encoding="utf-8")
            self.assertEqual(protocol.collect_exclusions(root), set(ids[:3]))
            self.assertEqual(protocol.collect_exclusions(root, [path, journal]), set(ids[:7]))

    def test_explicit_bad_or_missing_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            with self.assertRaises(FileNotFoundError):
                protocol.collect_exclusions(root, [missing])
            broken = root / "broken.json"
            broken.write_text("not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                protocol.collect_exclusions(root, [broken])
            with self.assertRaises(ValueError):
                protocol.collect_exclusions(root, [root / "old-model.pt"])


class FailureInclusiveSummaryTests(unittest.TestCase):
    def test_failures_interrupted_and_pending_each_stay_in_denominator(self):
        records = [record(row(i)["id"], status=status) for i, status in
                   enumerate(("completed", "failed", "interrupted"))]
        summary = protocol.summarize(records, ("combined",), 4)
        arm = summary["arms"]["combined"]
        self.assertEqual((arm["completed"], arm["failed"], arm["interrupted"], arm["pending"]), (1, 1, 1, 1))
        # Failed rows deliberately contain 1.0 metrics: only completed outputs score.
        self.assertEqual(set(arm["metrics"].values()), {0.25})
        self.assertEqual(arm["mean_writer_iterations"], 1.5)
        self.assertEqual(arm["mean_role_calls"], {"challenger": 0.75, "reviewer": 0.75, "writer": 1.5})
        self.assertEqual(arm["total_calls"], 12)
        self.assertEqual(arm["mean_total_calls"], 3)
        self.assertEqual(arm["challenger_calls"], 3)
        self.assertEqual(arm["mean_elapsed_seconds"], 1.125)
        self.assertEqual(arm["action_distribution"], {"INVOKE": 3, "REVISE": 3, "STOP": 3})
        self.assertIsNone(arm["total_tokens"])
        self.assertIsNone(arm["mean_total_tokens"])
        self.assertFalse(summary["complete"])

    def test_unknown_token_or_incomplete_journal_never_becomes_zero_or_full_total(self):
        for altered in ({"total_tokens": None}, {"journal_complete": False}):
            records = [record(row(i)["id"]) for i in range(2)]
            records[1].update(altered)
            arm = protocol.summarize(records, ("combined",), 2)["arms"]["combined"]
            self.assertIsNone(arm["total_tokens"])
            self.assertIsNone(arm["mean_total_tokens"])
            self.assertFalse(arm["journal_complete"])
            self.assertEqual(arm["known_recorded_tokens"], 30 if "total_tokens" in altered else 60)
        records = [record(row(i)["id"], tokens=0) for i in range(2)]
        arm = protocol.summarize(records, ("combined",), 2)["arms"]["combined"]
        self.assertEqual((arm["total_tokens"], arm["mean_total_tokens"]), (0, 0))
        self.assertTrue(arm["journal_complete"])

    def test_budget_termination_is_not_approval_and_coverage_is_not_success(self):
        rows = [record(row(0)["id"], approved=False), record(row(1)["id"], status="failed", approved=False)]
        rows[1]["attempted_agent_calls"] += 1
        summary = protocol.summarize(rows, ("combined", "strands"), 2)
        arm = summary["arms"]["combined"]
        self.assertEqual(arm["gate_approved"], 0)
        self.assertEqual(arm["termination_counts"], {"budget_cap": 2})
        self.assertEqual(arm["attempted_agent_calls"], 9)
        self.assertEqual(arm["total_calls"], 8)
        self.assertEqual(summary["arms"]["strands"]["pending"], 2)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["all_episodes_succeeded"])
        final = protocol.summarize(rows, ("combined",), 2)
        self.assertTrue(final["complete"])
        self.assertFalse(final["all_episodes_succeeded"])

    def test_rejects_duplicates_unknown_arms_overcoverage_and_invalid_bounds(self):
        good = record(row(0)["id"])
        for rows, arms, count in [([good, deepcopy(good)], ("combined",), 2),
                                  ([good], ("other",), 1),
                                  ([good, record(row(1)["id"])], ("combined",), 1),
                                  ([good], ("combined", "combined"), 1)]:
            with self.subTest(rows=rows, arms=arms), self.assertRaises(ValueError):
                protocol.summarize(rows, arms, count)
        for key, value in [("iterations", -1), ("iterations", True), ("total_tokens", -1),
                           ("elapsed_seconds", float("nan")), ("elapsed_seconds", -0.1),
                           ("attempted_agent_calls", 1), ("agent_calls", {"writer": 0.5}),
                           ("status", "pending"), ("status", []), ("arm", []),
                           ("actions", ["APPROVE"]), ("actions", [[]]), ("gate_approved", "yes"),
                           ("prediction", None), ("question_id", "hotpot_" + row(0)["id"])]:
            changed = dict(good, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                protocol.summarize([changed], ("combined",), 1)
        for invalid in (1.1, float("inf"), True):
            changed = deepcopy(good)
            changed["metrics"]["f1"] = invalid
            with self.assertRaises(ValueError):
                protocol.summarize([changed], ("combined",), 1)
        incomplete = deepcopy(good)
        del incomplete["metrics"]["sp_f1"]
        with self.assertRaises(ValueError):
            protocol.summarize([incomplete], ("combined",), 1)


if __name__ == "__main__":
    unittest.main()
