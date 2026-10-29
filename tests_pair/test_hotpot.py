"""Offline structural tests; none of these are benchmark result evidence."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from haltiq_pair.hotpot import (
    ARMS, Gold, METRICS, normalize_answer, parse_draft, parse_review,
    run_episode, run_hotpot, score_prediction, select_questions,
    split_public_gold, summarize, teammate_exclusions,
)


def row(qid="q1", answer="SECRET GOLD", kind="bridge", level="hard"):
    return {"id": qid, "question": "Where was the author born?", "answer": answer,
            "type": kind, "level": level,
            "context": {"title": ["Biography", "Distractor"],
                        "sentences": [["The author was born in Paris."], ["A different author was born in Rome."]]},
            "supporting_facts": {"title": ["Biography"], "sent_id": [0]}}


class FakeAgents:
    model_id = "offline-test-double"
    base_url = "http://127.0.0.1:8001/v1"

    def __init__(self, fail_on=None):
        self.usage = []
        self.calls = []
        self.fail_on = fail_on

    def generate(self, role, system, prompt):
        self.calls.append({"role": role, "system": system, "prompt": prompt})
        if len(self.calls) == self.fail_on:
            raise RuntimeError("Synthetic inference failure")
        self.usage.append({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        return json.dumps({"answer": "Paris", "supporting_facts": [["Biography", 0]]}) if role == "writer" else json.dumps({"approved": True, "feedback": "Supported by the biography."})


class FakeGate:
    provenance = {"identity": "offline-test-double", "threshold": 0.5}

    def __init__(self):
        self.calls = []

    def decide(self, **kwargs):
        self.calls.append(kwargs)
        assert all(isinstance(passage, str) for passage in kwargs["passages"])
        terminal = kwargs["arm"] == "fixed1" or kwargs["round_index"] == kwargs["max_rounds"]
        adaptive_stop = kwargs["arm"] not in {"fixed1", "fixed6"} and kwargs["round_index"] == 2
        return {"stop": terminal or adaptive_stop, "p_stop": 0.8, "p_approve": 0.9,
                "features": [0.0, kwargs["round_index"] / kwargs["max_rounds"]],
                "reason": "test gate", "distance": 0.0}


class HotpotMetricsTests(unittest.TestCase):
    def test_official_normalization_and_special_answers(self):
        self.assertEqual(normalize_answer("The (United) States!"), "united states")
        self.assertEqual(score_prediction({"answer": "yes indeed", "supporting_facts": []}, Gold("yes", ())) ["f1"], 0)
        self.assertEqual(score_prediction({"answer": "an answer", "supporting_facts": []}, Gold("The answer", ())) ["em"], 1)

    def test_answer_support_and_joint_precision_recall(self):
        metrics = score_prediction({"answer": "Paris France", "supporting_facts": [["A", 0], ["A", 0], ["C", 0]]},
                                   Gold("Paris", (("A", 0), ("B", 1))))
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(metrics["sp_f1"], 0.5)
        self.assertEqual(metrics["joint_prec"], 0.25)
        self.assertEqual(metrics["joint_recall"], 0.5)
        self.assertAlmostEqual(metrics["joint_f1"], 1 / 3)
        self.assertEqual(metrics["joint_em"], 0)

    def test_wrong_support_sentence_is_false_positive(self):
        metrics = score_prediction({"answer": "Paris", "supporting_facts": [["Biography", 99]]}, Gold("Paris", (("Biography", 0),)))
        self.assertEqual(metrics["em"], 1)
        self.assertEqual(metrics["sp_f1"], 0)
        self.assertEqual(metrics["joint_f1"], 0)

    def test_strict_draft_and_review_schemas(self):
        self.assertEqual(parse_draft('```json\n{"answer":"Paris","supporting_facts":[]}\n```')["answer"], "Paris")
        for bad in ('{"answer":"Paris"}', '{"answer":"Paris","supporting_facts":[["A",true]]}', '{"answer":"Paris","supporting_facts":[["A",-1]]}', 'not JSON'):
            with self.assertRaises(ValueError):
                parse_draft(bad)
        with self.assertRaises(ValueError):
            parse_review('{"approved":"yes","feedback":"ok"}')


class SelectionTests(unittest.TestCase):
    def test_deterministic_stratified_selection_excludes_cached_ids(self):
        rows = [row(str(index), kind="comparison" if index < 4 else "bridge") for index in range(20)]
        first = select_questions(rows, 10, 42, {"hotpotqa:0"})
        second = select_questions(reversed(rows), 10, 42, {"0"})
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])
        self.assertNotIn("0", {r["id"] for r in first})
        self.assertEqual(sum(r["type"] == "comparison" for r in first), 2)

    def test_selection_independent_of_gold_and_fails_short_or_duplicate(self):
        rows = [row(str(index)) for index in range(10)]
        changed = [dict(item, answer="DIFFERENT GOLD", supporting_facts={}) for item in rows]
        self.assertEqual([r["id"] for r in select_questions(rows, 5, 9)], [r["id"] for r in select_questions(changed, 5, 9)])
        with self.assertRaises(ValueError):
            select_questions(rows, 11, 9)
        with self.assertRaises(ValueError):
            select_questions(rows + [rows[0]], 5, 9)
        with self.assertRaises(ValueError):
            select_questions(rows, 1.5, 9)

    def test_public_boundary_omits_gold_and_keeps_every_distractor(self):
        public, gold = split_public_gold(row())
        self.assertEqual(gold.answer, "SECRET GOLD")
        self.assertNotIn("SECRET GOLD", json.dumps(asdict(public)))
        self.assertEqual([title for title, _ in public.passages], ["Biography", "Distractor"])

    def test_teammate_exclusions_only_read_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "strands").mkdir()
            (root / "data" / "sri_rounds_with_strands.csv").write_text("id,em\nhotpotqa:one,1\none,0\ntwo,1\n")
            (root / "strands" / "dev20_ids.txt").write_text("hotpotqa:three\n")
            self.assertEqual(teammate_exclusions(root), {"one", "two", "three"})


class LiveLoopTests(unittest.TestCase):
    def test_all_arms_make_fresh_calls_and_gold_never_reaches_either_agent_or_gate(self):
        agents, gate = FakeAgents(), FakeGate()
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_hotpot(agents, gate, tmp, limit=1, rows=[row(answer="Paris")], excluded_ids=())
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            self.assertEqual(manifest["config"]["data_mode"], "injected_rows_offline_verification")
            self.assertEqual(manifest["config"]["question_ids"], ["q1"])
            self.assertTrue(summary["all_episodes_succeeded"])
            expected_calls = {"fixed1": 1, "fixed6": 11, "linear": 4, "strands": 4, "combined": 4}
            for arm, count in expected_calls.items():
                self.assertEqual(summary["arms"][arm]["model_calls"], count)
                self.assertEqual(summary["arms"][arm]["metrics"]["joint_em"], 1)
            self.assertEqual(len(agents.calls), sum(expected_calls.values()))
            # Live agent and gate inputs have the two candidate passages, never a gold object.
            for call in agents.calls:
                self.assertIn("Distractor", call["prompt"])
                self.assertNotIn('"gold"', call["prompt"])
            for call in gate.calls:
                self.assertEqual(set(call), {"arm", "draft", "previous", "round_index", "max_rounds", "question", "passages", "previous_distance", "critic_approved"})
                self.assertGreaterEqual(call["round_index"], 1)

    def test_gold_mutation_changes_only_evaluation(self):
        first_agents, second_agents = FakeAgents(), FakeAgents()
        first_gate, second_gate = FakeGate(), FakeGate()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            right = run_hotpot(first_agents, first_gate, first, limit=1, rows=[row(answer="Paris")], arms=("combined",), excluded_ids=())
            wrong = run_hotpot(second_agents, second_gate, second, limit=1, rows=[row(answer="SECRET GOLD")], arms=("combined",), excluded_ids=())
        self.assertEqual(first_agents.calls, second_agents.calls)
        self.assertEqual(first_gate.calls, second_gate.calls)
        self.assertEqual(right["arms"]["combined"]["metrics"]["em"], 1)
        self.assertEqual(wrong["arms"]["combined"]["metrics"]["em"], 0)

    def test_failures_remain_in_planned_denominator(self):
        agents = FakeAgents(fail_on=1)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_hotpot(agents, FakeGate(), tmp, limit=2, rows=[row("a", answer="Paris"), row("b", answer="Paris")], arms=("fixed1",), excluded_ids=())
        arm = summary["arms"]["fixed1"]
        self.assertEqual((arm["completed"], arm["failed"], arm["planned"]), (1, 1, 2))
        self.assertEqual(arm["metrics"]["em"], 0.5)
        self.assertEqual(arm["model_calls"], 2)
        self.assertFalse(summary["all_episodes_succeeded"])

    def test_forced_budget_is_recorded(self):
        class NeverStop(FakeGate):
            def decide(self, **kwargs):
                return {"stop": False, "distance": 0}
        result = run_episode(FakeAgents(), NeverStop(), split_public_gold(row())[0], "combined", max_rounds=2)
        self.assertTrue(result["forced_budget_stop"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["model_calls"], 3)

    def test_nonfinite_gate_output_is_a_durable_failed_episode(self):
        class InvalidGate(FakeGate):
            def decide(self, **kwargs):
                return {"stop": True, "distance": float("nan")}
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_hotpot(FakeAgents(), InvalidGate(), tmp, limit=1, rows=[row()], arms=("fixed1",), excluded_ids=())
            self.assertEqual(summary["arms"]["fixed1"]["failed"], 1)
            result = json.loads((Path(tmp) / "episodes.jsonl").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["gate_decision"])

    def test_resume_skips_completed_refuses_changed_configuration(self):
        agents = FakeAgents()
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(limit=1, rows=[row(answer="Paris")], arms=("fixed1",), excluded_ids=())
            run_hotpot(agents, FakeGate(), tmp, **kwargs)
            count = len(agents.calls)
            run_hotpot(agents, FakeGate(), tmp, resume=True, **kwargs)
            self.assertEqual(len(agents.calls), count)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run_hotpot(agents, FakeGate(), tmp, seed=99, resume=True, **kwargs)
            with self.assertRaisesRegex(ValueError, "already contains"):
                run_hotpot(agents, FakeGate(), tmp, **kwargs)

    def test_interrupted_attempt_has_zero_quality_and_unknown_cost_without_retry(self):
        agents = FakeAgents()
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(limit=1, rows=[row(answer="Paris")], arms=("fixed1",), excluded_ids=())
            run_hotpot(agents, FakeGate(), tmp, **kwargs)
            (Path(tmp) / "episodes.jsonl").write_text("")
            (Path(tmp) / "traces.jsonl").write_text(json.dumps({"event": "episode_started", "question_id": "q1", "arm": "fixed1"}) + "\n")
            count = len(agents.calls)
            summary = run_hotpot(agents, FakeGate(), tmp, resume=True, **kwargs)
            self.assertEqual(len(agents.calls), count)
            self.assertEqual(summary["arms"]["fixed1"]["metrics"]["em"], 0)
            self.assertEqual(summary["arms"]["fixed1"]["unaccounted_interrupted_attempts"], 1)

    def test_summary_pending_episodes_are_not_removed_from_denominator(self):
        record = {"arm": "combined", "status": "completed", "metrics": {key: 1.0 for key in METRICS}}
        summary = summarize([record], ("combined",), 2)["arms"]["combined"]
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["metrics"]["f1"], 0.5)

    def test_summary_understands_strands_sdk_usage_and_unknown_cost(self):
        record = {"arm": "combined", "status": "completed", "usage": [{"role": "writer", "usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15}}]}
        summary = summarize([record], ("combined",), 1)["arms"]["combined"]
        self.assertEqual(summary["reported_agent_tokens"], {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})
        record["cost_unknown"] = True
        unknown = summarize([record], ("combined",), 1)["arms"]["combined"]
        self.assertTrue(unknown["recorded_costs_are_lower_bounds"])
        self.assertEqual(set(unknown["reported_agent_tokens"].values()), {None})

    def test_fixed6_cannot_be_accidentally_relabelled_short_budget(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "fixed6 requires"):
            run_hotpot(FakeAgents(), FakeGate(), tmp, limit=1, rows=[row()], arms=("fixed6",), max_rounds=2)


if __name__ == "__main__":
    unittest.main()
