"""Offline controls for the pilot; no downloaded weights or native calls."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pc_strands.control import GpuLease, ResourcePressure
from haltiq_hotpot import metrics, protocol, runner


def case(index):
    return {"id": str(index), "type": "bridge" if index < 10 else "comparison",
        "public": {"question": f"Question {index}?", "draft": "Paris" if index % 2 else "London",
            "paragraphs": [{"title": "Page0", "sentences": ["Paris is in France."]},
                           {"title": "Page1", "sentences": ["Other public text."]}]},
        "evaluator": {"answer": "Paris", "supporting_titles": ["Page0"]}}


def cohort():
    return {"schema": "haltiq-hotpot-partner-pilot-v1", "scope": protocol.SCOPE,
        "partner_revision": protocol.PARTNER_REVISION, "partner_csv_sha256": protocol.PARTNER_CSV_SHA256,
        "dataset_revision": protocol.DATASET_REVISION, "dataset_sha256": protocol.DATASET_SHA256,
        "selection": {"answer_blind": True, "seed": 42, "rank": "sha256('42:'+official_id)",
            "per_type": 10, "source_pool_counts": {"bridge": 176, "comparison": 44}, "model_scoring_started": False},
        "instructions": protocol.INSTRUCTIONS, "criteria": None, "threshold": .8,
        "arms": list(protocol.ARMS), "cases": [case(i) for i in range(20)]}


class SyntheticControl:
    def __init__(self, directory, deadline, **limits):
        self.limits = limits
        self.idle_seconds = 0
        self.last_resources = {"available_ram_bytes": 10 * 1024**3, "gpu_free_mib": 12 * 1024}
    def before_work(self, require_gpu=False):
        pass
    def _sleep(self, seconds):
        self.idle_seconds += seconds
    def rest_after_block(self, seconds):
        self.idle_seconds += max(5, seconds)


class ProtocolTests(unittest.TestCase):
    def test_official_join_rebuilds_all_context_ignoring_partner_oracle(self):
        source = {"_id": "id", "question": "What city?", "type": "bridge", "answer": "Paris",
            "context": [["Other", ["Public distractor."]], ["Paris", ["Public evidence."]]],
            "supporting_facts": [["Paris", 0]]}
        partner = {"id": "id", "question": "What city?", "draft": "Paris",
            "context": "GOLD_ONLY_ORACLE_MUST_NOT_LEAK", "em": "999", "strands_top3": "1"}
        row = protocol.reconstruct_rows([partner], [source])[0]
        self.assertEqual(len(row["public"]["paragraphs"]), 2)
        state = protocol.state_text(row["public"], protocol.baseline_evidence(row["public"]))
        self.assertIn("Public distractor.", state)
        self.assertNotIn("ORACLE", state)
        self.assertNotIn("supporting_titles", state)
        changed = dict(partner, id="different")
        with self.assertRaises(ValueError):
            protocol.reconstruct_rows([changed], [source])

    def test_selection_is_balanced_and_invariant_to_labels_scores_and_drafts(self):
        rows = [case(i) for i in range(40)]
        for i, row in enumerate(rows):
            row["type"] = "bridge" if i % 2 else "comparison"
        chosen = protocol.select_balanced(rows)
        modified = deepcopy(rows)
        for row in modified:
            row["public"]["draft"] = "changed"
            row["evaluator"]["answer"] = "changed reference"
            row["evaluator"]["future_score"] = .99
        self.assertEqual([r["id"] for r in chosen], [r["id"] for r in protocol.select_balanced(modified)])
        self.assertEqual(sum(r["type"] == "bridge" for r in chosen), 10)
        with self.assertRaises(ValueError):
            protocol.select_balanced(rows, per_type=5)

    def test_exact_partner_formats_query_titles_cutoff_and_stable_rank(self):
        public = case(1)["public"]
        public["paragraphs"] = [{"title": f"Title{i}", "sentences": ["x" * 1000, str(i)]} for i in range(4)]
        self.assertEqual(len(protocol.baseline_evidence(public)), 3000)
        self.assertNotIn("Title", protocol.baseline_evidence(public))
        pairs = protocol.retrieval_pairs(public)
        self.assertEqual(pairs[0][0], "Question 1? Paris")
        self.assertTrue(pairs[0][1].startswith("Title0: "))
        evidence, indices = protocol.ranked_evidence(public, [3, 3, 2, 1])
        self.assertEqual(indices, [0, 1, 2])
        self.assertGreater(len(evidence), 3000)  # no undocumented post-retrieval character cut
        with self.assertRaises(ValueError):
            protocol.ranked_evidence(public, [float("nan")] * 4)

    def test_protocol_and_public_payload_tampering_is_rejected(self):
        valid = cohort()
        protocol.validate_cohort(valid)
        changed = deepcopy(valid)
        changed["cases"][0]["public"]["answer"] = "evaluator leakage"
        with self.assertRaises(ValueError):
            protocol.validate_cohort(changed)
        changed = deepcopy(valid)
        changed["threshold"] = .5
        with self.assertRaises(ValueError):
            protocol.validate_cohort(changed)

    def test_module_imports_do_not_import_torch_or_transformers(self):
        result = subprocess.run([sys.executable, "-c", "import haltiq_hotpot.runner,sys; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class MetricTests(unittest.TestCase):
    def rows(self, data):
        return [{"id": c["id"], "status": "completed", "p_true": .9 if i % 2 else .1,
                 "raw_logits_false_true": [0, math.log(9) if i % 2 else -math.log(9)],
                 "temperature": 1} for i, c in enumerate(data["cases"])]

    def test_critic_accuracy_can_differ_while_answer_em_is_unchanged(self):
        data = cohort()
        good = self.rows(data)
        bad = deepcopy(good)
        for row in bad:
            row["p_true"] = 1 - row["p_true"]
        self.assertEqual(metrics.classifier(data["cases"], good)["critic_accuracy_observed"], 1)
        self.assertEqual(metrics.classifier(data["cases"], bad)["critic_accuracy_observed"], 0)
        self.assertEqual(sum(protocol.writer_metrics(c)["exact_match"] for c in data["cases"]) / 20, .5)

    def test_missing_scores_keep_planned_denominator_and_do_not_become_correct_rejections(self):
        data = cohort()
        rows = self.rows(data)
        rows[0] = {"id": rows[0]["id"], "status": "context_window_failure"}
        measured = metrics.classifier(data["cases"], rows)
        self.assertEqual(measured["critic_accuracy_failure_inclusive"], .95)
        self.assertEqual(measured["critic_accuracy_observed"], 1)
        self.assertEqual(measured["score_coverage"], .95)
        self.assertEqual(measured["confusion_observed"]["tn"], 9)
        with self.assertRaises(ValueError):
            metrics.classifier(data["cases"], rows[:-1])

    def test_extreme_logit_likelihood_and_undefined_bootstrap_are_explicit(self):
        data = cohort()
        rows = self.rows(data)
        for row in rows:
            row["raw_logits_false_true"] = [0, 1000]
        value = metrics.classifier(data["cases"], rows)["nll_raw_logits_observed"]
        self.assertTrue(math.isfinite(value))
        single = data["cases"][:1]
        result = metrics.paired(single, rows[:1], rows[:1], draws=10)
        self.assertEqual(result["undefined_auroc_draws"], 10)
        self.assertIsNone(result["auroc_bootstrap95_complete_pairs"])
        self.assertEqual(result["accuracy_bootstrap95"], [0, 0])


class RunnerTests(unittest.TestCase):
    def arguments(self, root):
        path = root / "cohort.json"
        path.write_text(json.dumps(cohort()), encoding="utf-8")
        (root / "shared.lock").write_bytes(b"0")
        for name in ("checkpoint", "base", "ranker"):
            (root / name).mkdir()
        return argparse.Namespace(cohort=path, cohort_sha256=protocol.digest(path),
            partner_csv=root / "partner.csv", dataset=root / "dataset.json", checkpoint_dir=root / "checkpoint",
            base_dir=root / "base", reranker_dir=root / "ranker", current_checkpoint=None,
            output=root / "output", lease_path=root / "shared.lock", deadline=None, allow_model_calls=True)

    def factories(self, args, calls, fail_at=None, overflow_at=None, nonfinite_ranker=False):
        test = self
        def assert_owned():
            with test.assertRaises(ResourcePressure):
                with GpuLease(args.lease_path, "synthetic-conflict"):
                    pass
        class Ranker:
            def __init__(self, path):
                assert_owned()
                calls.append("ranker_loaded")
            def rank(self, public):
                assert_owned()
                test.assertEqual(set(public), {"question", "draft", "paragraphs"})
                calls.append("ranked")
                return {"scores": [float("nan"), 0] if nonfinite_ranker else [1, 0], "pair_count": 2, "input_tokens": 12,
                        "padded_tokens": 12, "seconds": .001, "physical_forwards": 1}
            def close(self):
                assert_owned()
                calls.append("ranker_closed")
        class Decider:
            metadata = {"synthetic_control_only": True}
            def __init__(self, checkpoint, base, *, kind):
                assert_owned()
                calls.append("decider_loaded")
                self.preflights = 0
                self.forwards = 0
            def activate(self):
                calls.append("activated")
            def prepare(self, state):
                self.preflights += 1
                if self.preflights == overflow_at:
                    raise ValueError("prompt exceeds the context window")
                test.assertTrue(state.startswith("Question: "))
                test.assertNotIn("evaluator", state)
                return state
            def score(self, state):
                assert_owned()
                self.forwards += 1
                calls.append("forward")
                if self.forwards == fail_at:
                    raise RuntimeError("synthetic uncertain native forward")
                return {"p_true": .9, "p_true_unrounded": .9, "raw_logits_false_true": [0, math.log(9)],
                        "temperature": 1, "input_tokens": 100, "seconds": .001,
                        "physical_forwards": 1, "head_scores": 1}
            def close(self):
                assert_owned()
                calls.append("decider_closed")
        return Ranker, Decider

    def execute(self, args, calls, *, control_class=SyntheticControl, **options):
        ranker, decider = self.factories(args, calls, **options)
        def freeze(args):
            return {"cohort": args.cohort}, {"cohort": protocol.digest(args.cohort)}
        with patch("haltiq_hotpot.runner.prepare_cohort", return_value=cohort()), \
             patch("haltiq_hotpot.runner.ensure_port_closed"), \
             patch("haltiq_hotpot.runner.importlib.metadata.version", return_value="synthetic"):
            return runner.run(args, ranker_factory=ranker, decider_factory=decider,
                              control_class=control_class, freeze=freeze)

    def test_completed_paired_run_accounts_for_retrieval_and_native_work_and_preserves_null_gain(self):
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            result = self.execute(args, calls)
            self.assertEqual(result["physical_encoder_forwards"], 40)
            self.assertEqual(result["physical_reranker_forwards"], 20)
            self.assertEqual(result["physical_encoder_tokens"], 4000)
            self.assertTrue(result["frozen_files_unchanged"])
            self.assertTrue(result["gpu_lease_released"])
            self.assertEqual(calls[-1], "decider_closed")
            self.assertEqual([a["model"] for a in result["admissions"]], ["minilm_cpu", "hobson"])
            self.assertEqual(len(result["admissions"][1]["samples"]), 4)
            self.assertEqual(result["metrics"]["models"]["hobson"]["paired_top3_minus_baseline"]["candidate_minus_baseline_accuracy"], 0)
            with self.assertRaises(FileExistsError):
                self.execute(args, [])

    def test_uncertain_second_forward_is_not_retried_and_completed_first_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            with self.assertRaisesRegex(RuntimeError, "uncertain"):
                self.execute(args, calls, fail_at=2)
            receipt = json.loads((args.output / "pilot.json").read_text())
            self.assertEqual(calls.count("forward"), 2)
            self.assertEqual(receipt["physical_encoder_forwards"], 1)
            self.assertTrue(receipt["native_pending_or_uncertain"])
            self.assertTrue(receipt["owned_models_released"])
            self.assertTrue(receipt["gpu_lease_released"])
            self.assertEqual(receipt["records"][0]["status"], "completed")

    def test_window_failure_is_counted_without_native_inference_or_shortening(self):
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            receipt = self.execute(args, calls, overflow_at=2)
            self.assertEqual(receipt["status"], "completed_with_failures")
            self.assertEqual(receipt["physical_encoder_forwards"], 39)
            self.assertEqual(receipt["records"][1]["status"], "context_window_failure")
            self.assertEqual(receipt["metrics"]["models"]["hobson"]["minilm_top3"]["score_coverage"], .95)

    def test_dry_run_never_constructs_model_or_writes_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            args.allow_model_calls = False
            result = self.execute(args, calls)
            self.assertEqual(result["status"], "dry_run_no_model_import_or_calls")
            self.assertEqual(calls, [])
            self.assertFalse(args.output.exists())

    def test_failed_admission_never_imports_backend_and_placement_precedes_runtime_guards(self):
        class Denied(SyntheticControl):
            def before_work(self, require_gpu=False):
                raise ResourcePressure("synthetic startup memory pressure")
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            with self.assertRaises(ResourcePressure):
                self.execute(args, calls, control_class=Denied)
            self.assertEqual(calls, [])
            result = json.loads((args.output / "pilot.json").read_text())
            self.assertEqual(result["physical_encoder_forwards"], 0)
            self.assertTrue(result["gpu_lease_released"])
        test = self
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            class PlacementObserved(SyntheticControl):
                def before_work(self, require_gpu=False):
                    if require_gpu and self.limits["minimum_ram_gib"] == 7:
                        test.assertIn("activated", calls)
            self.execute(args, calls, control_class=PlacementObserved)

    def test_nonfinite_retrieval_retains_physical_work_and_unresolved_journal_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                self.execute(args, calls, nonfinite_ranker=True)
            result = json.loads((args.output / "pilot.json").read_text())
            events = [json.loads(line) for line in (args.output / "physical-work.jsonl").read_text().splitlines()]
            self.assertEqual(result["physical_reranker_forwards"], 1)
            self.assertTrue(result["retrieval_pending_or_uncertain"])
            self.assertEqual([event["status"] for event in events], ["pending"])
            self.assertEqual(calls.count("ranked"), 1)
            self.assertNotIn("decider_loaded", calls)
            self.assertTrue(result["gpu_lease_released"])

    def test_native_journal_failure_does_not_clear_pending_flag_or_repeat_completed_forward(self):
        actual_journal = runner.journal
        def failed_completed_journal(path, event):
            if event["operation"] == "native" and event["status"] == "completed":
                raise OSError("synthetic native completion fsync failure")
            return actual_journal(path, event)
        with tempfile.TemporaryDirectory() as folder:
            args, calls = self.arguments(Path(folder)), []
            with patch("haltiq_hotpot.runner.journal", side_effect=failed_completed_journal):
                with self.assertRaisesRegex(OSError, "fsync"):
                    self.execute(args, calls)
            result = json.loads((args.output / "pilot.json").read_text())
            events = [json.loads(line) for line in (args.output / "physical-work.jsonl").read_text().splitlines()]
            native = [event for event in events if event["operation"] == "native"]
            self.assertEqual(result["physical_encoder_forwards"], 1)
            self.assertEqual(calls.count("forward"), 1)
            self.assertTrue(result["native_pending_or_uncertain"])
            self.assertEqual([event["status"] for event in native], ["pending"])
            self.assertEqual(result["records"][0]["status"], "native_pending")
            self.assertTrue(result["owned_models_released"])


if __name__ == "__main__":
    unittest.main()
