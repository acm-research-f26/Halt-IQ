"""Native Tau2 stopping-policy controls with invented deterministic callbacks.

These are software tests, not model-quality evidence. No models, GPU probes,
provider/network requests, or actual training are used.
"""
from copy import deepcopy
import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from haltiq_tau_hotpot.journal import UncertainCall
from haltiq_tau_hotpot.qualification import native_episode


@unittest.skipUnless(importlib.util.find_spec("tau2"), "Requires installed official Tau2 runtime; callbacks stay synthetic")
class FallbackTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from haltiq_tau_hotpot import agent, domain
        self.agent, self.domain = agent, domain
        self.case = {"public": {"id": "synthetic-cap-fallback", "question": "Which invented code is requested?",
            "type": "bridge", "level": "software_control", "context": [["Invented note", ["The requested code is cedar."]]]},
            "evaluator": {"answer": "cedar", "supporting_facts": [["Invented note", 0]]}}
        self.answers = ["ash", "birch", "cedar"]

    def draft(self, answer):
        return {"answer": answer, "supporting_facts": [["Invented note", 0]]}

    def episode(self, *, policy=None, answers=None, gate=None, reviewer=None, caps=None, arm="released", writer=None):
        drafts = iter(answers or self.answers)
        return self.agent.run_episode(deepcopy(self.case), writer or (lambda *_: self.draft(next(drafts))),
            reviewer or (lambda *_: "Recheck the invented note."), gate or (lambda *_: {"p_true": .1}),
            arm=arm, max_drafts=3, threshold=.8, passage_indices=[0], seed=42,
            timeout_seconds=1800, caps=caps, cap_fallback=policy)

    def test_final_valid_rejected_draft_is_submitted_only_under_explicit_new_policy(self):
        legacy = self.episode()
        future = self.episode(policy="last_valid")
        self.assertIsNone(legacy["submitted"])
        self.assertEqual(legacy["failure_category"], "draft_cap_no_accepted_submission")
        self.assertEqual(legacy["metrics"]["em"], 0.)
        self.assertNotIn("cap_fallback", legacy)
        self.assertNotIn("submission_reason", legacy)
        self.assertEqual(future["submitted"], self.draft("cedar"))
        self.assertEqual(future["status"], "completed")
        self.assertEqual(future["metrics"]["em"], 1.)
        self.assertEqual(future["counts"]["answer_drafts"], 3)
        self.assertEqual(future["counts"]["gate_calls"], 3)
        self.assertEqual(future["counts"]["reviewer_calls"], 2)
        self.assertEqual(future["critic_threshold"], .8)
        self.assertTrue(future["budget_fallback_used"])
        self.assertFalse(future["gate_approved"])
        self.assertEqual(future["submission_reason"], "budget_fallback")
        fallback = next(e for e in future["trace"] if e["event"] == "budget_fallback")
        self.assertEqual(fallback["selected_draft_number"], 3)
        calls = [m for m in future["native_simulation"]["messages"] if m.get("tool_calls")]
        self.assertEqual(calls[-1]["tool_calls"][0]["name"], "submit_answer")
        self.assertEqual(calls[-1]["tool_calls"][0]["arguments"], self.draft("cedar"))

    def test_fallback_selects_latest_visited_draft_and_can_make_the_answer_wrong(self):
        row = self.episode(policy="last_valid", answers=["cedar", "cedar", "wrong"])
        self.assertEqual(row["submitted"], self.draft("wrong"))
        self.assertEqual(row["metrics"]["em"], 0.)
        self.assertTrue(row["budget_fallback_used"])

    def test_early_gate_approval_still_stops_before_revision_or_fallback(self):
        reviewer = Mock(side_effect=AssertionError("No review after approval"))
        row = self.episode(policy="last_valid", gate=lambda *_: {"p_true": .9}, reviewer=reviewer)
        self.assertEqual(row["submitted"], self.draft("ash"))
        self.assertEqual(row["counts"]["answer_drafts"], 1)
        self.assertTrue(row["gate_approved"])
        self.assertFalse(row["budget_fallback_used"])
        self.assertEqual(row["submission_reason"], "gate_approved")
        reviewer.assert_not_called()

    def test_known_reviewer_or_gate_failure_does_not_rescue_an_earlier_valid_draft(self):
        for role in ("reviewer", "gate"):
            fail = Mock(side_effect=ValueError("Known synthetic technical failure"))
            row = self.episode(policy="last_valid", **{role: fail})
            with self.subTest(role=role):
                self.assertEqual(row["status"], "completed_with_failure")
                self.assertIsNone(row["submitted"])
                self.assertEqual(row["failure_category"], role + "_value_error")
                self.assertFalse(row["budget_fallback_used"])

    def test_invalid_final_draft_never_falls_back_to_the_previous_valid_one(self):
        drafts = iter([self.draft("cedar"), self.draft("cedar"), {"answer": "cedar", "supporting_facts": [["Missing title", 0]]}])
        row = self.episode(policy="last_valid", writer=lambda *_: next(drafts))
        self.assertEqual(row["failure_category"], "invalid_submission")
        self.assertIsNone(row["submitted"])
        self.assertEqual(row["counts"]["gate_calls"], 2)
        self.assertFalse(row["budget_fallback_used"])

    def test_uncertain_provider_work_aborts_without_any_submission(self):
        writer = Mock(side_effect=UncertainCall("Synthetic unresolved request"))
        with self.assertRaises(UncertainCall) as caught:
            self.episode(policy="last_valid", writer=writer)
        self.assertFalse(any(e.get("name") == "submit_answer" for e in caught.exception.haltiq_episode_partial["trace"]))
        writer.assert_called_once()

    def test_wall_tool_and_gate_caps_cannot_be_bypassed_by_budget_fallback(self):
        for caps in ({"tool_calls": 2}, {"gate_calls": 2}, {"reviewer_calls": 1}):
            row = self.episode(policy="last_valid", caps=caps)
            with self.subTest(caps=caps):
                self.assertIsNone(row["submitted"])
                self.assertFalse(row["budget_fallback_used"])
                self.assertEqual(row["status"], "completed_with_failure")
        base = self.agent.HotpotEpisodeAgent
        class StopAtSubmit(base):
            def _call_tool(self, state, name, arguments):
                if name == "submit_answer": self.wall_deadline = -1
                return super()._call_tool(state, name, arguments)
        with patch.object(self.agent, "HotpotEpisodeAgent", StopAtSubmit):
            row = self.episode(policy="last_valid")
        self.assertIsNone(row["submitted"])
        self.assertEqual(row["failure_category"], "episode_wall_timeout")
        self.assertFalse(row["budget_fallback_used"])

    def test_decomposed_low_evidence_is_not_relabelled_as_a_gate_approval(self):
        row = self.episode(policy="last_valid", arm="decomposed",
            gate=lambda _, __, criterion: {"p_true": .9 if criterion == "answer" else .1})
        self.assertEqual(row["counts"]["gate_calls"], 6)
        self.assertTrue(row["budget_fallback_used"])
        self.assertFalse(row["gate_approved"])
        self.assertEqual(row["submitted"], self.draft("cedar"))

    def test_native_audit_replays_explicit_fallback_and_legacy_omits_new_keyword(self):
        for policy in (None, "last_valid"):
            row = self.episode(policy=policy)
            run = Mock(wraps=self.agent.run_episode)
            helpers = {"haltiq_tau_hotpot.agent": SimpleNamespace(run_episode=run),
                       "haltiq_tau_hotpot.domain": self.domain}
            native_episode(row, self.case, [0], helpers, arm="released", max_drafts=3)
            if policy is None:
                self.assertNotIn("cap_fallback", run.call_args.kwargs)
            else:
                self.assertEqual(run.call_args.kwargs["cap_fallback"], "last_valid")
                changed = deepcopy(row)
                changed["gate_approved"] = True
                with self.assertRaises(ValueError): native_episode(changed, self.case, [0], helpers, arm="released", max_drafts=3)

    def test_fixed_target_and_invalid_policy_have_explicit_unchanged_semantics(self):
        row = self.episode(policy="last_valid", arm="fixed1")
        self.assertEqual(row["submitted"], self.draft("ash"))
        self.assertEqual(row["submission_reason"], "fixed_target")
        self.assertFalse(row["budget_fallback_used"])
        self.assertEqual(row["counts"]["gate_calls"], 0)
        with self.assertRaises(ValueError): self.episode(policy="choose_correct_answer")


if __name__ == "__main__": unittest.main()
