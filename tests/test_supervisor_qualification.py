"""Bounded bootstrap qualification cannot waive any other live-study gate."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from haltiq_supervisor.plan import PACKAGES, inspect, prepare, source_files, validate, verify_frozen
from pc_strands.campaign import digest


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "repo"
        self.source.mkdir()
        for package in PACKAGES:
            (self.source / package).mkdir()
            (self.source / package / "__init__.py").write_text("# fixture only\n")
        for name in ("haltiq_supervisor/engine.py", "pc_strands/control.py", "haltiq_tau_hotpot/worker.py"):
            (self.source / name).write_text("# fixture only, never executed\n")
        (self.source / "pyproject.toml").write_text("# fixture only\n")
        self.freeze_path, self.config_path = self.root / "data-freeze.json", self.root / "worker-config.json"
        self.freeze = {"schema": "tau2-hotpot-qualification-v1",
            "selection": {"answer_blind": True, "quality_filtering": False,
                          "length_filtering": False, "replacement_allowed": False},
            "pilot": {"training": {"fit": 40, "tune": 10, "calibration": 10}, "heldout": 4,
                "max_drafts": 3, "arms": ["fixed1", "fixed3", "released", "trained"],
                "criteria": ["answer"], "threshold": .8, "engineering_only": True,
                "baseline_before_weight_updates": True}, "files": {}}
        for part, count in (("fit", 40), ("tune", 10), ("calibration", 10), ("heldout", 4)):
            cases = [{"partition": "engineering_heldout" if part == "heldout" else part,
                      "public": {"id": f"{part}-{index}", "question": f"Unique {part} question {index}?",
                                 "type": "bridge" if index % 2 else "comparison"}}
                     for index in range(count)]
            path = self.root / ("pilot_" + part + ".json")
            self.write(path, cases)
            self.freeze["files"]["pilot_" + part] = {**self.pin(path), "count": count,
                "type_counts": {"bridge": count // 2, "comparison": count // 2}}
        self.config = {"purpose": "qualification_pilot", "data_freeze": str(self.freeze_path)}
        proof = self.root / "implementation-proof.json"
        self.write(proof, {"verified": True})
        self.plan = {"version": 1, "scope": "custom_tau2_hotpot", "purpose": "qualification_pilot",
            "status": "ready", "source_root": str(self.source), "python": sys.executable,
            "source_paths": [*PACKAGES, "pyproject.toml"], "inputs": [],
            "readiness": {key: [{**self.pin(proof), "json_equals": {"verified": True}}]
                          for key in ("custom_domain", "durable_call_journals", "hobson_training", "gpu_coordinator")},
            "deadline": (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat(),
            "hard_deadline": (datetime.now(timezone.utc) + timedelta(minutes=4)).isoformat(),
            "gpu_lease": str(self.root / "gpu.lock"), "study_lease": str(self.root / "study.lock"),
            "private_port": 11435, "resources": {"ram_gib": 7, "vram_gib": 3,
                "startup_ram_gib": 8.5, "startup_vram_gib": 9, "stability_seconds": 30,
                "minimum_rest_seconds": 5, "cpu_threads": 4, "gpu_owners": 1, "duty_cycle": .5},
            "stages": []}
        for phase, count in (("collect", 64), ("features", 64), ("train", 1), ("compare", 16)):
            receipt = "{run}/" + phase + "/completion.json"
            self.plan["stages"].append({"id": phase, "model_work": True, "gpu": phase != "train",
                "planned_units": count, "completion_receipt": receipt, "outputs": [{"path": receipt}],
                "argv": [sys.executable, "-B", "-m", "haltiq_tau_hotpot.worker", phase,
                    "--config", str(self.config_path), "--output", "{run}/" + phase,
                    "--previous", "{run}", "--deadline", "{deadline}", "--control", "{control}",
                    "--allow-model-calls"]})
        self.repin()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def pin(path):
        return {"path": str(path), "sha256": digest(path)}

    def repin(self):
        self.write(self.freeze_path, self.freeze)
        self.config["data_freeze_sha256"] = digest(self.freeze_path)
        self.write(self.config_path, self.config)
        self.plan["qualification_protocol"] = self.pin(self.config_path)
        self.plan["inputs"] = [self.pin(self.config_path), self.pin(self.freeze_path),
            *[{"path": spec["path"], "sha256": spec["sha256"]} for spec in self.freeze["files"].values()]]
        self.plan["reviewed_source_sha256"] = {name: digest(path) for name, path in source_files(self.plan).items()}

    def test_pilot_omits_only_combined_proof_and_preparation_never_dispatches(self):
        self.assertEqual(inspect(self.plan)["status"], "ready_not_started")
        path = self.root / "supervision.json"
        self.write(path, self.plan)
        with patch("subprocess.Popen") as spawn:
            state = prepare(path, self.root / "run", allow_model_calls=True)
        self.assertEqual(state["status"], "prepared_not_started")
        self.assertIn("haltiq_tau_hotpot/worker.py", state["source_sha256"])
        spawn.assert_not_called()

    def test_all_other_implementation_proofs_remain_required(self):
        for key in tuple(self.plan["readiness"]):
            candidate = deepcopy(self.plan)
            candidate["readiness"].pop(key)
            self.assertIn("Missing verified readiness: " + key, inspect(candidate)["blockers"])

    def test_main_default_keeps_combined_pilot_requirement(self):
        for purpose in (None, "main_study"):
            candidate = deepcopy(self.plan)
            if purpose is None:
                candidate.pop("purpose")
            else:
                candidate["purpose"] = purpose
            self.assertIn("Missing verified readiness: combined_pilot", inspect(candidate)["blockers"])

    def test_unknown_purpose_and_offline_pilot_are_rejected(self):
        for changes in ({"purpose": "qualification"}, {"purpose": None},
                        {"scope": "offline_supervision_fixture"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "purpose"):
                validate({**self.plan, **changes})

    def test_scope_caps_types_arms_criteria_and_baseline_are_immutable(self):
        original = deepcopy(self.freeze)
        changes = [("heldout", 10), ("heldout", True), ("heldout", 0), ("max_drafts", 10),
            ("max_drafts", True), ("arms", ["fixed1", "released"]), ("criteria", ["answer", "evidence"]),
            ("threshold", .7), ("engineering_only", False), ("baseline_before_weight_updates", False),
            ("training", {"fit": 42, "tune": 10, "calibration": 10}),
            ("training", {"fit": 40, "tune": 10, "calibration": True}),
            ("training", {"fit": 40, "tune": 10, "calibration": 0})]
        for key, value in changes:
            self.freeze = deepcopy(original)
            self.freeze["pilot"][key] = value
            self.repin()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(self.plan)

    def test_filtering_and_nonblind_selection_cannot_be_rebound_into_pilot(self):
        for key in self.freeze["selection"]:
            candidate = deepcopy(self.freeze)
            candidate["selection"][key] = not candidate["selection"][key]
            self.write(self.freeze_path, candidate)
            self.config["data_freeze_sha256"] = digest(self.freeze_path)
            self.write(self.config_path, self.config)
            self.plan["qualification_protocol"] = self.pin(self.config_path)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "answer-blind"):
                validate(self.plan)

    def test_all_config_data_and_cohort_inputs_are_required_and_hash_bound(self):
        for index in range(len(self.plan["inputs"])):
            candidate = deepcopy(self.plan)
            candidate["inputs"].pop(index)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "pinned inputs"):
                validate(candidate)
        self.config_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            validate(self.plan)

    def test_changed_cohort_count_type_or_question_overlap_is_rejected(self):
        spec = self.freeze["files"]["pilot_heldout"]
        path = Path(spec["path"])
        original = json.loads(path.read_text())
        for mutation in ("count", "type", "overlap_id", "overlap_question"):
            rows = deepcopy(original)
            if mutation == "count":
                rows.pop()
            elif mutation == "type":
                rows[0]["public"]["type"] = "bridge"
            elif mutation == "overlap_id":
                rows[0]["public"]["id"] = "fit-0"
            else:
                rows[0]["public"]["question"] = " UNIQUE fit question 0? "
            self.write(path, rows)
            spec["sha256"] = digest(path)
            self.repin()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(self.plan)

    def test_no_extra_reordered_phase_or_mislabeled_unit_count(self):
        for mutation in ("extra", "reorder", "count", "nonmodel"):
            candidate = deepcopy(self.plan)
            if mutation == "extra":
                candidate["stages"].append({**candidate["stages"][-1], "id": "main"})
            elif mutation == "reorder":
                candidate["stages"][0], candidate["stages"][1] = candidate["stages"][1], candidate["stages"][0]
            elif mutation == "count":
                candidate["stages"][1]["planned_units"] = 124
            else:
                candidate["stages"][2]["model_work"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(candidate)

    def test_entrypoint_phase_config_paths_and_extra_flags_cannot_escape_scope(self):
        for mutation in ("module", "phase", "config", "previous", "output", "extra_flag", "duplicate_flag"):
            candidate = deepcopy(self.plan)
            argv = candidate["stages"][0]["argv"]
            if mutation == "module":
                argv[3] = "haltiq_hotpot.runner"
                (self.source / "haltiq_hotpot/runner.py").write_text("# fixture only\n")
            elif mutation == "phase":
                argv[4] = "compare"
            elif mutation in {"config", "previous", "output"}:
                argv[argv.index("--" + mutation) + 1] = "unbound"
            elif mutation == "extra_flag":
                argv.extend(["--long-study", "14h"])
            else:
                argv.extend(["--config", str(self.config_path)])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(candidate)

    def test_package_and_reviewed_source_freeze_remain_mandatory(self):
        candidate = deepcopy(self.plan)
        candidate["source_paths"].remove("haltiq_tau_hotpot")
        with self.assertRaisesRegex(ValueError, "Freeze every study package"):
            validate(candidate)
        candidate = deepcopy(self.plan)
        candidate.pop("reviewed_source_sha256")
        self.assertIn("Reviewed complete source manifest is not bound", inspect(candidate)["blockers"])

    def test_explicit_model_optin_and_frozen_cohort_recheck_are_preserved(self):
        path = self.root / "supervision.json"
        self.write(path, self.plan)
        with self.assertRaisesRegex(ValueError, "allow-model-calls"):
            prepare(path, self.root / "no-optin")
        directory = self.root / "run"
        state = prepare(path, directory, allow_model_calls=True)
        Path(self.freeze["files"]["pilot_fit"]["path"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "Pinned file changed"):
            verify_frozen(self.plan, state, directory)


if __name__ == "__main__":
    unittest.main()
