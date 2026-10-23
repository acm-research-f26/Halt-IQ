"""Check the actual public scientific data and reject misleading mutations."""
from copy import deepcopy
from pathlib import Path
import unittest

from scripts.verify_partner_method import read, verify

ARTIFACT = Path(__file__).resolve().parents[1] / "experiments/hotpotqa/partner-method-20.json"


class PartnerArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observed = read(ARTIFACT)

    def changed(self):
        return deepcopy(self.observed)

    def test_actual_data_recomputes_and_original_git_blobs_survive_future_cleanup_changes(self):
        result = verify(self.observed)
        self.assertEqual(result["actual_native_forwards"], 40)
        self.assertEqual(result["same20_top3_decision_mismatches"], 0)
        self.assertFalse(self.observed["heldout_confirmation"])
        self.assertFalse(self.observed["quality_gain_claim"])

    def test_changed_evaluator_label_is_rejected_against_saved_draft_and_reference(self):
        changed = self.changed()
        changed["cases"][0]["label"] = not changed["cases"][0]["label"]
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            verify(changed, source_blobs=False)

    def test_probability_cannot_be_reused_after_raw_logit_change(self):
        changed = self.changed()
        changed["cases"][0]["scores"]["baseline_3000"]["raw_logits_false_true"][1] += 1
        with self.assertRaisesRegex(ValueError, "raw logits"):
            verify(changed, source_blobs=False)

    def test_top3_ranking_must_follow_all_actual_raw_scores(self):
        changed = self.changed()
        changed["cases"][0]["retrieval"]["selected_paragraph_indices"].reverse()
        with self.assertRaisesRegex(ValueError, "ordering"):
            verify(changed, source_blobs=False)

    def test_accounting_cannot_drop_physical_tokens_or_add_overlapping_startup_time(self):
        changed = self.changed()
        changed["physical_work"]["encoder_input_tokens"] -= 1
        with self.assertRaisesRegex(ValueError, "token accounting"):
            verify(changed, source_blobs=False)
        changed = self.changed()
        changed["timing_seconds"]["intentional_idle"] += 5
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            verify(changed, source_blobs=False)

    def test_postexit_qualification_cannot_erase_original_cleanup_failure(self):
        changed = self.changed()
        changed["lifecycle"]["original_owned_models_released"] = True
        with self.assertRaisesRegex(ValueError, "cleanup failure"):
            verify(changed, source_blobs=False)

    def test_missing_case_cannot_improve_denominator_or_be_replaced(self):
        changed = self.changed()
        changed["cases"].pop()
        with self.assertRaisesRegex(ValueError, "Exactly20"):
            verify(changed, source_blobs=False)
        changed = self.changed()
        changed["cases"][1]["id"] = changed["cases"][0]["id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            verify(changed, source_blobs=False)

    def test_cached_partner_approval_agreement_is_recomputed_not_asserted(self):
        changed = self.changed()
        first = changed["cases"][0]
        first["partner_cached_scores"]["minilm_top3"] = 0 if first["scores"]["minilm_top3"]["p_true"] >= .8 else 1
        with self.assertRaisesRegex(ValueError, "Cached comparison"):
            verify(changed, source_blobs=False)


if __name__ == "__main__":
    unittest.main()
