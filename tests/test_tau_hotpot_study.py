"""Offline main-study contracts and synthetic controls, never research outcomes."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from haltiq_tau_hotpot import protocol
from haltiq_tau_hotpot.journal import CallJournal, fingerprint
from haltiq_tau_hotpot.main_worker import analyse, valid_learning_rows, verify_feature_rows
from haltiq_tau_hotpot.hobson import file_hash, stable_hash
from haltiq_tau_hotpot.scoring import score


def case(identifier, kind, partition):
    return {"partition": partition, "public": {"id": identifier, "question": "Question " + identifier + "?",
        "type": kind, "level": "synthetic fixture", "context": [["Public", ["A fact."]]]},
        "evaluator": {"answer": "correct", "supporting_facts": [["Public", 0]]}}


class MainProtocolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.research = deepcopy(protocol.MAIN_SETTINGS)
        self.freeze = {"schema": "tau2-hotpot-main-v1", "main": deepcopy(self.research), "selection": {"answer_blind": True,
            "quality_filtering": False, "length_filtering": False, "replacement_allowed": False}, "files": {}}
        self.rows = {}
        for name, n, partition in (("main_development", 20, "development"), ("main_fit", 400, "fit"),
                ("main_tune", 100, "tune"), ("main_calibration", 100, "calibration")):
            self.rows[name] = [case(name + str(i), "bridge" if i % 2 == 0 else "comparison", partition) for i in range(n)]
        fresh = [case("fresh" + str(i), "bridge" if i % 2 == 0 else "comparison", "fresh") for i in range(60)]
        for n in (20, 40, 60): self.rows["main_fresh" + str(n)] = fresh[:n]
        for name, rows in self.rows.items():
            self.freeze["files"][name] = self.save(name, rows) | {"count": len(rows),
                "type_counts": {"bridge": len(rows) // 2, "comparison": len(rows) // 2}}
        checkpoint, base = self.root / "checkpoint", self.root / "base"
        (checkpoint / "lora").mkdir(parents=True)
        base.mkdir()
        model_files = {"released_head_sha256": checkpoint / "head.safetensors",
            "adapter_sha256": checkpoint / "lora/adapter_model.safetensors",
            "base_weights_sha256": base / "model.safetensors-00001-of-00001.safetensors"}
        for path in model_files.values(): path.write_bytes(b"SYNTHETIC-PROTOCOL-BINDING-NOT-MODEL")
        self.pilot = {"schema": "haltiq-tau-hotpot-qualified-pilot-v1", "status": "PASS_QUALIFIED_ENGINEERING_PILOT",
            "qualified": True, "terminal": True, "cleanup_verified": True, "pending_or_uncertain": False,
            "scope": "synthetic contract fixture", "source_sha256": {"synthetic-source": "0" * 64},
            "model_binding": {**{key: file_hash(path) for key, path in model_files.items()}, "head_parameters": 1053184,
                "upstream_sha256": {name: protocol.PINS["upstream/" + name]
                    for name in ("modeling.py", "infer.py", "prompting.py", "schema.py")}}}
        self.config = {"purpose": "main_study", "checkpoint": str(checkpoint), "base": str(base),
                       "native_feature_cache": True, "response_cache": True}
        for key, value in (("protocol", self.research), ("data_freeze", self.freeze),
                ("combined_pilot_proof", self.pilot)):
            spec = self.save(key, value)
            self.config[key], self.config[key + "_sha256"] = spec["path"], spec["sha256"]
        throughput = self.save("throughput", {"answer_blind": True, "whole_route_fits": True,
                                               "scope": "synthetic contract fixture, not measured throughput"})
        self.selected = {"n": 20, "selected_n": 20, "answer_blind": True, "fresh_scoring_started": False,
            "basis": "measured_throughput_and_resources", "data_freeze_sha256": self.config["data_freeze_sha256"],
            "protocol_sha256": self.config["protocol_sha256"],
            "dataset_sha256": self.freeze["files"]["main_fresh20"]["sha256"],
            "throughput_proof": throughput["path"], "throughput_proof_sha256": throughput["sha256"],
            "frozen_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
        self.update_selected()

    def save(self, name, value):
        path = self.root / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        return {"path": str(path), "sha256": file_hash(path)}

    def update_selected(self):
        spec = self.save("fresh_n_freeze", self.selected)
        self.config.update(fresh_n_freeze=spec["path"], fresh_n_freeze_sha256=spec["sha256"])

    def update_freeze(self):
        spec = self.save("data_freeze", self.freeze)
        self.config["data_freeze_sha256"] = spec["sha256"]
        self.selected["data_freeze_sha256"] = spec["sha256"]
        self.update_selected()

    def test_import_has_no_torch_or_tau_model_side_effect(self):
        script = ("import sys;import haltiq_tau_hotpot.protocol,haltiq_tau_hotpot.main_worker;"
                  "assert not any(k in sys.modules for k in ('torch','tau2','transformers'));print('safe')")
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_valid_blind_nested_plan_returns_exact620_prep_and_selected20_without_dispatch(self):
        with patch("subprocess.Popen") as spawn:
            loaded = protocol.load_main(self.config)
        self.assertEqual(sum(len(loaded["cohorts"]["main_" + p]) for p in ("development", "fit", "tune", "calibration")), 620)
        self.assertEqual(loaded["fresh_n"], 20)
        self.assertEqual(len(loaded["cohorts"]["main_fresh"]), 20)
        spawn.assert_not_called()

    def test_no_static_pilot_success_or_pending_cleanup_can_unlock_main(self):
        for change in ({"qualified": False}, {"pending_or_uncertain": True}, {"cleanup_verified": False}, {"terminal": 1}):
            proof = deepcopy(self.pilot) | change
            spec = self.save("combined_pilot_proof", proof)
            config = {**self.config, "combined_pilot_proof_sha256": spec["sha256"]}
            with self.subTest(change=change), self.assertRaises(ValueError): protocol.load_main(config)

    def test_main_requires_declared_caches_and_exact_model_identity_from_real_pilot(self):
        for key in ("native_feature_cache", "response_cache"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cache"):
                protocol.load_main({**self.config, key: False})
        proof = deepcopy(self.pilot)
        proof["model_binding"]["released_head_sha256"] = "0" * 64
        spec = self.save("combined_pilot_proof", proof)
        with self.assertRaisesRegex(ValueError, "qualified"):
            protocol.load_main({**self.config, "combined_pilot_proof_sha256": spec["sha256"]})

    def test_model_cutoff_reserves_final_hour_without_extending_window_on_late_dispatch(self):
        start = datetime.now(timezone.utc) - timedelta(minutes=5)
        config = {"start_time": start.isoformat(), "model_deadline": (start + timedelta(hours=13)).isoformat()}
        deadline = (start + timedelta(hours=14)).isoformat()
        self.assertEqual(protocol.execution_cutoff(config, deadline, "features"), config["model_deadline"])
        self.assertEqual(protocol.execution_cutoff(config, deadline, "audit"), deadline)
        with self.assertRaises(ValueError):
            protocol.execution_cutoff(config, (start + timedelta(hours=15)).isoformat(), "compare")

    def test_fresh_outcome_clock_and_quality_selection_mutations_are_rejected(self):
        for change in ({"fresh_scoring_started": True}, {"answer_blind": False}, {"n": 10},
                {"basis": "quality looked better"}, {"frozen_at": "2026-10-23T01:00:00"},
                {"selected_n": 60}, {"protocol_sha256": "0" * 64}):
            original = deepcopy(self.selected)
            self.selected.update(change)
            self.update_selected()
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)): protocol.load_main(self.config)
            self.selected = original
            self.update_selected()

    def test_changed_cohort_and_non_nested_or_overlapping_fresh_cases_rejected(self):
        Path(self.freeze["files"]["main_fit"]["path"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "hash"): protocol.load_main(self.config)
        spec = self.save("main_fit", self.rows["main_fit"])
        self.freeze["files"]["main_fit"].update(spec)
        altered = deepcopy(self.rows["main_fresh40"])
        altered[0]["public"]["question"] = "Different unselected question?"
        spec = self.save("main_fresh40", altered)
        self.freeze["files"]["main_fresh40"].update(spec)
        self.update_freeze()
        with self.assertRaisesRegex(ValueError, "nested"): protocol.load_main(self.config)

    def test_frozen_minima_cannot_silently_drop_failed_cases_or_filter_wrong_labels(self):
        rows = [{"question_id": str(i), "criterion": "answer", "partition": "fit", "status": "completed",
                 "label": 0} for i in range(320)] + [{"question_id": str(i), "criterion": "answer",
                    "partition": "fit", "status": "failed"} for i in range(320, 400)]
        valid, counts = valid_learning_rows(rows, "answer", "fit", 320)
        self.assertEqual(len(valid), 320)
        self.assertEqual(counts, {"planned": 400, "valid": 320, "failed": 80})
        rows[0]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "minima"): valid_learning_rows(rows, "answer", "fit", 320)

    def test_current_dispatch_rejects_changed_actual_config_before_frozen_source_io(self):
        root = self.root / "run"
        args = SimpleNamespace(previous=root, config=self.root / "config.json", output=root / "collect",
            control=root / "workers/collect", phase="collect", deadline="2026-10-23T23:00:00+00:00")
        plan = {"purpose": "main_study", "stages": [{"id": name} for name in protocol.PHASES],
                "main_protocol": {"path": str(args.config)}, "deadline": args.deadline}
        state = {"run_id": "fixture", "plan_sha256": "0" * 64, "stages": {"collect": {"status": "started"}}}
        with patch("haltiq_tau_hotpot.protocol.__file__", str(root / "source/haltiq_tau_hotpot/protocol.py")), \
             patch.dict(os.environ, HALTIQ_RUN_ID="fixture", HALTIQ_PLAN_SHA256="0" * 64, HALTIQ_PHASE="collect"), \
             patch("haltiq_supervisor.plan.verify_frozen") as frozen:
            protocol.verify_dispatch(args, plan, state)
            frozen.assert_called_once()
            args.config = self.root / "different.json"
            with self.assertRaisesRegex(ValueError, "current phase"): protocol.verify_dispatch(args, plan, state)


class MainEvidenceTests(unittest.TestCase):
    def test_native_state_label_and_tensor_hash_join_catches_mislabeling_without_any_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(previous=root)
            item = case("one", "bridge", "development")
            context = {"cohorts": {"main_development": [item], "main_fit": [], "main_tune": [], "main_calibration": []}}
            draft = {"answer": "correct", "supporting_facts": [["Public", 0]]}
            collect = root / "collect"
            collect.mkdir()
            (collect / "episodes.json").write_text(json.dumps([{"question_id": "one", "episode": {"submitted": draft}}]))
            journal = CallJournal(root / "features/physical")
            rows = []
            for criterion in ("answer", "evidence"):
                tensor = root / (criterion + ".safetensors")
                tensor.write_bytes(b"MOCK-CACHE-NO-REAL-TENSORS")
                state = "Question: Question one?\nEvidence: Public: A fact.\nDraft answer: correct"
                if criterion == "evidence": state += '\nCitations: [["Public", 0]]'
                observed = {"physical_forwards": 1, "retrieval_indices": [0], "feature": {"criterion": criterion,
                    "provenance": {"state_sha256": stable_hash(state)}, "path": str(tensor), "sha256": file_hash(tensor)}}
                journal.call("native_" + criterion, {"scope": "mock-only synthetic fixture"}, lambda observed=observed: observed)
                rows.append({"question_id": "one", "partition": "development", "criterion": criterion,
                             "status": "completed", "label": 1, "observed": observed})
            baseline = {"rows_sha256": fingerprint(rows[:1])}
            verify_feature_rows(args, context, rows, baseline)
            rows[1]["label"] = 0
            with self.assertRaisesRegex(ValueError, "join failed"): verify_feature_rows(args, context, rows, baseline)

    def test_wrong_to_correct_control_proves_harness_can_show_real_answer_difference(self):
        cases = [case("b", "bridge", "fresh"), case("c", "comparison", "fresh")]
        initial, episodes = {}, []
        for item in cases:
            qid = item["public"]["id"]
            wrong = {"answer": "wrong", "supporting_facts": [["Public", 0]]}
            initial[qid] = wrong
            for arm in protocol.ARMS:
                output = deepcopy(wrong)
                if arm in {"trained", "fixed3"}: output["answer"] = "correct"
                trace = [{"event": "draft", "status": "completed", "draft": deepcopy(wrong)}]
                if arm in {"released", "trained", "decomposed"}:
                    for criterion in (("answer", "evidence") if arm == "decomposed" else ("answer",)):
                        trace.append({"event": "gate", "status": "completed", "criterion": criterion,
                            "score": {"p_true_unrounded": .5, "p_true": .5, "raw_logits_false_true": [0., 0.], "temperature": 1.}})
                episodes.append({"question_id": qid, "arm": arm, "status": "completed", "submitted": output,
                    "metrics": score(output, item["evaluator"]), "trace": trace, "counts": {"answer_drafts": 3 if arm == "fixed3" else 1}})
        result = analyse(episodes, {"cohorts": {"main_fresh": cases}, "fresh_freeze": {"scope": "synthetic fixture only"}}, initial)
        self.assertEqual(result["policies"]["released"]["metrics"]["em"], 0)
        self.assertEqual(result["policies"]["trained"]["metrics"]["em"], 1)
        self.assertEqual(result["paired"]["trained_minus_released"]["em"]["bootstrap95"], [1., 1.])
        self.assertEqual(result["policies"]["trained"]["repairs"], 2)
        self.assertEqual(result["evidence_first_draft_decision"]["scored_questions"], 2)
        self.assertEqual(len(result["policies"]), 8)


if __name__ == "__main__":
    unittest.main()
