"""Trajectory audit controls; synthetic harness results are not model evidence."""
from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from pc_strands.data import Task
from pc_strands.evaluate import (GENERIC_FEEDBACK, OllamaWriter,
                                  generate_trajectories, replay_trajectories)
from pc_strands.metrics import evaluate
from scripts.audit_trajectory_signal import REPO, analyze_trajectories, audit_bundle


def saved(task, drafts):
    return [{"task_id": task.id, "status": "completed", "rounds": [
        {"round": i + 1, "status": "completed", "draft": draft, **evaluate(draft, task.answers)}
        for i, draft in enumerate(drafts)]}]


class NoWait:
    idle_seconds = 0
    def before(self):
        pass
    def after(self, elapsed):
        pass


class ScriptedWriter:
    """Synthetic wrong-then-correct sequence, never a real writer benchmark."""
    def payload(self, state, previous=None, feedback=None):
        return {"previous": previous}
    def generate(self, payload):
        answer = "London" if payload["previous"] is None else "Paris"
        return {"response": {"done": True, "done_reason": "stop", "message": {
            "content": json.dumps({"answer": answer})}}, "input_tokens": 10,
            "output_tokens": 1, "elapsed_seconds": .01, "cost_usd": 0}
    answer = staticmethod(OllamaWriter.answer)
    def unload(self):
        pass


class ScriptedDecision:
    def __init__(self, probability):
        self.probability = probability
    def score(self, state, temperature=None):
        return {"p_true": self.probability, "logits": [0, math.log(self.probability / (1 - self.probability))],
                "input_tokens": 20, "elapsed_seconds": .01}
    def offload(self):
        pass


class TrajectorySignalTests(unittest.TestCase):
    def setUp(self):
        self.task = Task("hotpotqa:synthetic-control", "Capital of France?", ["France capital: Paris"], ["Paris"])

    def test_wording_change_is_distinct_from_quality_signal(self):
        _, result = analyze_trajectories([self.task], saved(self.task, ["London", "The London"]), expected_drafts=2)
        self.assertEqual(result["raw_answer_variants_per_question"], {"2": 1})
        self.assertEqual(result["questions_with_changing_f1"], 0)
        self.assertEqual(result["oracle_f1_gain_over_first"], 0)

    def test_improving_candidate_creates_measurable_opportunity(self):
        _, result = analyze_trajectories([self.task], saved(self.task, ["London", "Paris"]), expected_drafts=2)
        self.assertEqual(result["questions_with_changing_em"], 1)
        self.assertEqual(result["oracle_f1_gain_over_first"], 1)

    def test_stale_stored_score_is_not_accepted(self):
        rows = saved(self.task, ["London", "Paris"])
        rows[0]["rounds"][0]["token_f1"] = 1
        with self.assertRaisesRegex(ValueError, "stored F1"):
            analyze_trajectories([self.task], rows, expected_drafts=2)

    def test_missing_round_and_duplicate_cohort_are_rejected(self):
        rows = saved(self.task, ["London", "Paris"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze_trajectories([self.task], rows, expected_drafts=3)
        with self.assertRaisesRegex(ValueError, "cohort/order"):
            analyze_trajectories([self.task], rows + copy.deepcopy(rows), expected_drafts=2)

    def test_synthetic_actual_harness_can_produce_different_f1(self):
        """Mock intervention proves sensitivity, not real-model improvement."""
        config = {"max_drafts": 3, "feedback": GENERIC_FEEDBACK,
                  "model_fingerprint": "synthetic-harness-control", "threshold": .8}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trajectories = generate_trajectories([self.task], ScriptedWriter(), root / "writer", config, NoWait())
            selected = []
            for name, probability in (("early", .9), ("continue", .1)):
                episodes = replay_trajectories([self.task], trajectories, ScriptedDecision(probability),
                                              root / name, config, NoWait())
                selected.append(next(row for row in episodes if row["policy"] == "strands-untuned"))
        self.assertEqual([row["final_answer"] for row in selected], ["London", "Paris"])
        self.assertEqual([row["token_f1"] for row in selected], [0, 1])
        self.assertEqual([row["rounds"] for row in selected], [1, 3])

    def test_completed_panel_rescores_and_distinguishes_critic_from_writer(self):
        result = audit_bundle(REPO / "results/strands-20261018")
        signal = result["trajectory_signal"]
        self.assertEqual((signal["questions"], signal["saved_draft_states"]), (20, 200))
        self.assertEqual(signal["questions_with_changing_f1"], 0)
        self.assertEqual(signal["raw_answer_variants_per_question"], {"1": 18, "2": 2})
        self.assertEqual(result["head_change"]["first_logit_pairs_changed"], 20)
        self.assertEqual(result["head_change"]["first_approval_disagreements"], 0)
        self.assertAlmostEqual(result["first_decision_metrics"]["untuned"]["auroc"], .38)
        self.assertAlmostEqual(result["first_decision_metrics"]["jev"]["auroc"], .915)

    def test_hosted_probability_and_cohort_are_recomputed_not_copied(self):
        source = REPO / "results/strands-20261018"
        inputs = audit_bundle(source)["input_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in inputs:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            path = root / "audits/optional-jev-independent-audit.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            changed = copy.deepcopy(original)
            changed["data"]["per_question"][0]["jev_first_p"] = .1
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diagnostic mismatch"):
                audit_bundle(root)
            changed = copy.deepcopy(original)
            changed["data"]["per_question"][0]["question_id"] = "foreign-question"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cohort/order mismatch"):
                audit_bundle(root)


if __name__ == "__main__":
    unittest.main()
