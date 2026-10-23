"""Recovery ownership and real Tau2 routing with synthetic provider responses.

Fixtures below are software controls, never scientific/model evidence. No HTTP,
GPU telemetry, Torch loading, actual model operation or process job is started.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from haltiq_tau_hotpot import recovery as r
from haltiq_tau_hotpot.journal import CallJournal, durable_write, fingerprint
from haltiq_tau_hotpot.recovery_worker import QuestionRuntime, assert_envelope
from haltiq_tau_hotpot.runtime import LocalRuntime


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    return path


class Control:
    def __init__(self, directory):
        self.directory = directory
        self.deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        self.active_seconds = self.idle_seconds = 0.
    def _check_stop(self): pass
    def before_work(self, **_): pass
    def rest_after_block(self, _): pass


class RecoveryContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.old = self.root / "old-failed-owner"
        self.old.mkdir()
        self.original = {key: "fixture-" + key for key in (
            "checkpoint", "base", "reranker", "ollama", "ollama_models", "writer_digest", "tau_revision")}
        self.original.update(gpu_lease=str(self.root / "gpu.lock"), study_lease=str(self.root / "study.lock"),
                             model_inputs=[], runtime_modules={})
        self.deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.old_plan = {"python": sys.executable, "deadline": self.deadline,
                         "hard_deadline": (datetime.now(timezone.utc) + timedelta(hours=1, minutes=5)).isoformat()}
        write(self.old / "plan.json", self.old_plan)
        original_path = write(self.root / "original.json", self.original)
        old_input = self.old / "immutable-input.txt"
        old_input.write_text("synthetic fixed input", encoding="utf-8")
        self.prep = {"schema": "haltiq-reusable-pilot-preparation-v1", "status": "PASS_REUSABLE_PREPARATION",
            "source_run": str(self.old), "source_overall_status": "failed", "combined_pilot_qualified": False,
            "original_config": r.spec(original_path), "arms": r.ARMS, "max_drafts": 3, "threshold": .8,
            "question_ids": ["fixture-q" + str(n) for n in range(4)],
            "input_sha256": {str(old_input): r.q.sha(old_input)}}
        self.prep_path = write(self.root / "preparation.json", self.prep)

    def prepared(self):
        with patch.object(r, "preparation_binding", return_value=(self.prep, self.original, {}, {})):
            return r.prepare_recovery(self.prep_path, self.repo, sys.executable, self.root / "prepared")

    def test_import_is_model_free_and_does_not_initialize_tau(self):
        command = "import sys;import haltiq_tau_hotpot.recovery,haltiq_tau_hotpot.recovery_worker;assert not any(x in sys.modules for x in ('torch','transformers','tau2'))"
        result = subprocess.run([sys.executable, "-B", "-c", command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prepare_is_blocked_four_separate_owners_with_original_cutoffs(self):
        before = {str(p): p.read_bytes() for p in self.old.rglob("*") if p.is_file()}
        with patch("subprocess.Popen") as spawned:
            plan = self.prepared()
        spawned.assert_not_called()
        self.assertEqual(plan["status"], "blocked_preparation_only")
        self.assertEqual(plan["reviewed_source_sha256"], {})
        self.assertEqual(plan["readiness"], {})
        self.assertIsNone(plan["codex_argv"])
        self.assertEqual([s["id"] for s in plan["stages"]], list(r.PHASES))
        self.assertEqual([s["planned_units"] for s in plan["stages"]], [4] * 4)
        self.assertEqual(plan["deadline"], self.old_plan["deadline"])
        self.assertEqual(plan["hard_deadline"], self.old_plan["hard_deadline"])
        self.assertEqual(plan["resources"]["startup_ram_gib"], 11)
        self.assertEqual(plan["resources"]["ram_gib"], 7)
        self.assertEqual(r.validate_plan(plan)["question_ids"], self.prep["question_ids"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.old.rglob("*") if p.is_file()})

    def test_prepare_rejects_original_run_and_source_preservation_boundaries(self):
        for output in (self.old / "new-template", self.old, self.root, self.repo / "new-template"):
            with self.subTest(output=output), patch.object(r, "preparation_binding", return_value=(self.prep, self.original, {}, {})):
                with self.assertRaises(ValueError):
                    r.prepare_recovery(self.prep_path, self.repo, sys.executable, output)
        with self.assertRaises(ValueError):
            r._external_output(self.old / "new-proof.json", self.old)
        with self.assertRaises(ValueError):
            r._external_output(self.prep_path, self.old)

    def test_method_environment_and_lease_changes_are_rejected_before_dispatch(self):
        plan = self.prepared()
        original_config = json.loads(Path(plan["recovery_protocol"]["path"]).read_text())
        for key in ("writer_digest", "base", "gpu_lease", "study_lease", "tau_revision", "runtime_modules"):
            with self.subTest(key=key):
                changed = deepcopy(original_config)
                changed[key] = {"foreign": "module"} if key == "runtime_modules" else "foreign"
                write(plan["recovery_protocol"]["path"], changed)
                edited = deepcopy(plan)
                edited["recovery_protocol"] = r.spec(plan["recovery_protocol"]["path"])
                edited["inputs"][0] = edited["recovery_protocol"]
                with self.assertRaisesRegex(ValueError, "preserve original"):
                    r.validate_plan(edited)
        write(plan["recovery_protocol"]["path"], original_config)
        plan["recovery_protocol"] = r.spec(plan["recovery_protocol"]["path"])
        for mutator in (lambda p: p.update(gpu_lease="foreign"),
                        lambda p: p["resources"].update(startup_ram_gib=8.5),
                        lambda p: p["stages"][0].update(planned_units=3),
                        lambda p: p["stages"].reverse(),
                        lambda p: p.update(deadline="2099-01-01T00:00:00+00:00"),
                        lambda p: p["inputs"].pop()):
            candidate = deepcopy(plan)
            candidate["inputs"][0] = candidate["recovery_protocol"]
            mutator(candidate)
            with self.assertRaises(ValueError): r.validate_plan(candidate)

    def owner_fixture(self):
        run = self.root / "synthetic-owner"
        source = run / "source"
        source.mkdir(parents=True)
        (source / "fixture.py").write_text("# Offline ownership contract only\n", encoding="utf-8")
        source_sha = {"fixture.py": r.q.sha(source / "fixture.py")}
        plan = {"purpose": "qualification_pilot", "python": sys.executable,
            "reviewed_source_sha256": source_sha, "inputs": [], "readiness": {},
            "stages": [{"id": phase, "planned_units": count,
                "outputs": [{"path": "{run}/" + phase + "/completion.json"}]} for phase, count in zip(r.PREP, (64, 64, 1))]}
        write(run / "plan.json", plan)
        state = {"status": "failed", "run_id": "SYNTHETIC-CONTRACT-NOT-MODEL-EVIDENCE",
            "plan_sha256": r.q.sha(run / "plan.json"), "source_sha256": source_sha,
            "python_sha256": r.q.sha(sys.executable), "stages": {}}
        for phase, count in zip(r.PREP, (64, 64, 1)):
            completion = {"run_id": state["run_id"], "plan_sha256": state["plan_sha256"], "phase": phase,
                "terminal": True, "pending_or_uncertain": False, "cleanup_verified": True,
                "planned_units_accounted": True, "planned_units": count, "completed_units": count,
                "failed_units": 0, "not_run_units": 0}
            path = write(run / phase / "completion.json", completion)
            state["stages"][phase] = {"status": "completed", "exit_code": 0,
                "owned_tree_clear": True, "os_handoff_verified": True, "output_sha256": {str(path): r.q.sha(path)}}
        write(run / "supervisor.json", state)
        return run, plan, state

    def test_import_requires_actual_completed_prep_and_keeps_original_failed(self):
        run, plan, state = self.owner_fixture()
        before = (run / "supervisor.json").read_bytes()
        r._owner(run, purpose="qualification_pilot", phases=r.PREP, terminal_status="failed")
        self.assertEqual(before, (run / "supervisor.json").read_bytes())
        self.assertFalse((run / "compare/completion.json").exists())
        for key, changed in (("status", "started"), ("exit_code", 1), ("owned_tree_clear", False), ("os_handoff_verified", False)):
            edited = deepcopy(state)
            edited["stages"]["features"][key] = changed
            write(run / "supervisor.json", edited)
            with self.assertRaises(ValueError):
                r._owner(run, purpose="qualification_pilot", phases=r.PREP, terminal_status="failed")
        write(run / "supervisor.json", state)
        (run / "source/fixture.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source/interpreter"):
            r._owner(run, purpose="qualification_pilot", phases=r.PREP, terminal_status="failed")

    def measurement_fixture(self):
        run = self.root / "measurement"
        started = datetime.now(timezone.utc) - timedelta(seconds=5)
        journal = CallJournal(run / "compare_q1/compare/physical")
        journal.call("synthetic_offline_operation", {"fixture": True}, lambda: {"no_models": True})
        finished = datetime.now(timezone.utc) + timedelta(seconds=5)
        write(run / "supervisor.json", {"plan_sha256": "fixture", "stages": {"compare_q1": {
            "started_at": started.isoformat(), "finished_at": finished.isoformat()}}})
        timing = write(run / "compare_q1/phase-timing.json", {"wall_seconds": 5., "active_seconds": 1., "intentional_idle_seconds": 4.})
        terminal = write(run / "compare_q1/compare/runtime-terminal.json", {"timings": {"startup_seconds": .2, "switch_seconds": .4}})
        return run, timing, terminal

    def test_measurements_are_real_owner_paths_and_nonoverlapping_clock_bound(self):
        run, timing, terminal = self.measurement_fixture()
        source, physical, measured = r._measurement(run, "compare_q1", recovery=True)
        self.assertEqual(Path(source["physical_dir"]), run / "compare_q1/compare/physical")
        self.assertEqual(physical["completed"], 1)
        self.assertGreaterEqual(measured["supervisor_stage_wall_seconds"], 10.)
        self.assertEqual(measured["receipt_sha256"], r.q.sha(timing))
        for bad in (float("inf"), float("nan"), -1., 30.):
            write(timing, {"wall_seconds": bad, "active_seconds": 1., "intentional_idle_seconds": 4.})
            with self.assertRaises(ValueError): r._measurement(run, "compare_q1", recovery=True)
        write(timing, {"wall_seconds": 5., "active_seconds": 1., "intentional_idle_seconds": 4.})
        write(terminal, {"timings": {"startup_seconds": float("inf")}})
        with self.assertRaises(ValueError): r._measurement(run, "compare_q1", recovery=True)

    def test_current_dispatch_rejects_foreign_or_unfrozen_worker_before_model_import(self):
        from haltiq_tau_hotpot.recovery_worker import verify_dispatch
        args = SimpleNamespace(previous=self.root / "not-this-source", phase="compare_q1")
        with patch("haltiq_supervisor.plan.verify_frozen") as frozen, self.assertRaises(ValueError):
            verify_dispatch(args, {"purpose": "qualification_recovery"}, {})
        frozen.assert_not_called()


class RecoveryQualificationIntegrity(unittest.TestCase):
    """Exercise the dependency reconstruction layer; semantic auditors are mocked."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "recovery"
        self.run.mkdir()
        self.old = self.root / "old"
        self.old.mkdir()
        self.imported = {"source_run": str(self.old), "input_sha256": {}, "model_binding": {"fixture": "bound"},
            "trained_head": r.spec(write(self.old / "head.json", {"fixture": True})),
            "baseline": r.spec(write(self.old / "baseline.json", {"fixture": True})),
            "learning": r.spec(write(self.old / "learning.json", {"fixture": True})),
            "source_plan_sha256": "old-plan", "source_supervisor_sha256": "old-supervisor",
            "source_sha256": {}, "completed_import_stages": {"fixture": True},
            "historical_failed_compare": {"promoted_to_recovery_episodes": False}}
        prep = write(self.root / "prep.json", {"synthetic_contract": True})
        self.prep_spec = r.spec(prep)
        self.plan = {"inputs": [self.prep_spec], "readiness": {}, "stages": []}
        self.state = {"run_id": "synthetic-owner", "plan_sha256": "synthetic-plan", "source_sha256": {}}
        write(self.run / "supervisor.json", self.state)
        write(self.run / "plan.json", self.plan)
        for phase in r.PHASES:
            write(self.run / phase / "retained.json", {"phase": phase, "fixture": True})
        self.measurements = {}
        self.physical = {}
        self.timings = {}
        for phase in r.PREP + r.PHASES:
            self.measurements[phase] = {"run": str(self.old if phase in r.PREP else self.run), "phase": phase}
            self.physical[phase] = {"completed": 0, "pending_or_uncertain": False}
            self.timings[phase] = {"wall_seconds": 0., "supervisor_stage_wall_seconds": 0.}
        self.value = {"schema": "haltiq-tau-hotpot-qualified-pilot-v2", "status": "PASS_QUALIFIED_ENGINEERING_PILOT",
            "planned_units": [64, 64, 1, 16], "qualified": True, "terminal": True, "cleanup_verified": True,
            "actual_os_handoff_verified": True, "pending_or_uncertain": False,
            "preparation_owner": {"proof": self.prep_spec, "run": str(self.old),
                "plan_sha256": "old-plan", "supervisor_sha256": "old-supervisor", "source_sha256": {},
                "completed_import_stages": {"fixture": True}, "baseline_path": self.imported["baseline"]["path"],
                "learning_path": self.imported["learning"]["path"]},
            "historical_failed_compare": self.imported["historical_failed_compare"],
            "recovery_run": str(self.run), "run": str(self.run),
            "run_id": self.state["run_id"], "plan_sha256": self.state["plan_sha256"],
            "supervisor_sha256": r.q.sha(self.run / "supervisor.json"), "source_sha256": {},
            "model_binding": self.imported["model_binding"], "trained_head_path": self.imported["trained_head"]["path"],
            "trained_head_sha256": self.imported["trained_head"]["sha256"], "untouched_baseline_sha256": self.imported["baseline"]["sha256"],
            "measurement_sources": self.measurements, "physical_calls": self.physical, "timing": {"phases": self.timings},
            "pinned_inputs_sha256": {self.prep_spec["path"]: self.prep_spec["sha256"], **{
                str(p): r.q.sha(p) for p in self.run.rglob("*") if p.is_file()}},
            "output_sha256": {str(p.relative_to(self.run)): r.q.sha(p) for phase in r.PHASES
                for p in (self.run / phase).rglob("*") if p.is_file()}}

    def verify(self, value):
        def measurement(run, phase, **_):
            return self.measurements[phase], self.physical[phase], self.timings[phase]
        with patch.object(r, "preparation_binding", return_value=(self.imported, {}, {}, {})), \
             patch.object(r, "_owner", return_value=(self.plan, self.state)), \
             patch.object(r, "validate_plan", return_value={"preparation_proof": self.prep_spec}), \
             patch.object(r, "_measurement", side_effect=measurement), \
             patch("pc_strands.control.GpuLease") as lease:
            try: return r.verify_qualified_pilot(value)
            finally: lease.assert_not_called()

    def test_full_pin_union_and_measurements_are_reconstructed_not_trusted(self):
        value, trained, owner = self.verify(self.value)
        self.assertEqual(trained, Path(self.imported["trained_head"]["path"]))
        self.assertEqual(len(owner["verification_inputs"]), len(value["pinned_inputs_sha256"]))
        for changed in ("missing_pin", "redirected_measurement", "invented_timing", "missing_output"):
            edited = deepcopy(self.value)
            if changed == "missing_pin": edited["pinned_inputs_sha256"].pop(next(iter(edited["pinned_inputs_sha256"])))
            if changed == "redirected_measurement": edited["measurement_sources"]["compare_q2"]["run"] = str(self.old)
            if changed == "invented_timing": edited["timing"]["phases"]["train"]["wall_seconds"] = 999.
            if changed == "missing_output": edited["output_sha256"].pop(next(iter(edited["output_sha256"])))
            with self.subTest(changed=changed), self.assertRaises(ValueError): self.verify(edited)


class PlacementEnvelope(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = {"writer_digest": "synthetic-writer", "response_cache": True,
            "gpu_lease": str(self.root / "gpu.lock"), "checkpoint": "never-load", "base": "never-load"}
        self.runtime = QuestionRuntime(self.config, self.root / "runtime", Control(self.root / "control"))

    def test_eviction_and_second_native_placement_are_rejected_before_any_dispatch(self):
        self.runtime.journal.call("writer_load", {}, lambda: {})
        self.runtime.writer_loaded = True
        self.runtime._http = Mock(return_value={"models": []})
        with patch.object(LocalRuntime, "_writer_ready") as reload, self.assertRaisesRegex(RuntimeError, "evicted"):
            self.runtime._writer_ready()
        reload.assert_not_called()
        self.assertEqual(self.runtime.journal.summary()["dispatched"], 1)
        self.runtime.journal.call("native_startup", {}, lambda: {})
        with patch.object(LocalRuntime, "_native_ready") as reload, self.assertRaisesRegex(RuntimeError, "reload"):
            self.runtime._native_ready()
        reload.assert_not_called()
        with self.assertRaises(RuntimeError): self.runtime._writer_ready()

    def test_envelope_rejects_duplicate_or_reverse_placements(self):
        for sequence in (("writer_load", "writer_load"), ("native_startup", "native_startup"), ("native_startup", "writer_load")):
            runtime = SimpleNamespace(journal=SimpleNamespace(records=[{"event": "pending", "role": role} for role in sequence]))
            with self.subTest(sequence=sequence), self.assertRaises(RuntimeError): assert_envelope(runtime)


@unittest.skipUnless(importlib.util.find_spec("tau2"), "Requires installed official Tau2 runtime; providers stay mocked")
class NativeTauPrefixControls(unittest.TestCase):
    """Real native episode/prompt/cache/placement control with no real models."""
    def run_control(self, invalid=False):
        from haltiq_tau_hotpot.agent import run_episode
        from haltiq_tau_hotpot.recovery_worker import run_question
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = {"id": "software-control", "question": "What letter is requested?", "type": "bridge", "level": "hard",
                "context": [["Public passage", ["The requested answer is C."]]]}
            case = {"public": public, "evaluator": {"answer": "C", "supporting_facts": [["Public passage", 0]]}, "partition": "heldout"}
            first = {"answer": "A", "supporting_facts": [["Public passage", 0]]}
            source = root / "original"
            write(source / "collect/episodes.json", [{"question_id": public["id"],
                "episode": {"submitted": first}, "initial_retrieval": {"indices": [0]}}])
            write(source / "features/features.json", [{"question_id": public["id"], "observed": {"feature": {"provenance_sha256": "software-only"}}}])
            learning = write(source / "train/learning.json", {"criteria": {"answer": {
                "released_calibration": {"temperature": 1.1}, "trained_calibration": {"temperature": 1.2}}}})
            trained = write(source / "trained.json", {"not_model_weights": True})
            prep_path = write(root / "prep.json", {"not_a_qualification": True})
            prep = {"source_run": str(source), "learning": r.spec(learning), "trained_head": r.spec(trained)}
            config = {"preparation_proof": r.spec(prep_path), "question_ids": [public["id"]] * 4,
                "checkpoint": str(root / "released"), "base": "not-loaded", "writer_digest": "synthetic-writer",
                "response_cache": True, "native_feature_cache": True, "gpu_lease": str(root / "never.lock")}
            (root / "released").mkdir()
            (root / "released/head.safetensors").write_bytes(b"NOT-WEIGHTS")
            args = SimpleNamespace(phase="compare_q1", output=root / "new-question")
            args.output.mkdir()
            calls, instances = [], []
            class Tensor:
                def contiguous(self): return self
            class Backend:
                metadata = {"synthetic_fixture": True}
                def __init__(self, *_): pass
                def activate(self): calls.append("native_activate")
                def prepare_state(self, state, **_): return {"state": state}
                def observe(self, state, **kwargs):
                    answer = state.rsplit("Draft answer: ", 1)[1]
                    return {"p_true": .9 if answer == "C" else .1, "physical_forwards": 1, "head_scores": 2,
                        "temperature": kwargs["temperature"], "feature": {"decide": Tensor(), "options": Tensor(),
                            "provenance_sha256": fingerprint(state), "provenance": {"state_sha256": fingerprint(state)}}}
            class MockRuntime(QuestionRuntime):
                def __enter__(self):
                    instances.append(self)
                    self.resident = False
                    self.ranker = SimpleNamespace(rank=lambda _: {"scores": [.5], "physical_forwards": 1})
                    return self
                def __exit__(self, *_):
                    self.cleanup = {"verified": True, "private_port_closed": True, "gpu_lease_released": True}
                def _startup_guard(self): pass
                def _close_native(self):
                    if self.backend is not None: raise AssertionError("No native-to-writer switch is allowed in this route")
                def _http(self, path, data=None, **_):
                    if path == "/api/ps": return {"models": [{"digest": config["writer_digest"]}] if self.resident else []}
                    if path == "/api/generate":
                        self.resident = data["keep_alive"] != 0
                        return {"done": True}
                    if path != "/api/chat": raise AssertionError(path)
                    payload = json.loads(data["messages"][1]["content"])
                    is_writer = "format" in data
                    calls.append("writer" if is_writer else "reviewer")
                    if not is_writer: text = "Review: " + payload["state"]["draft"]["answer"]
                    elif invalid: text = "not json"
                    else:
                        previous = payload["state"]["previous_draft"]["answer"]
                        text = json.dumps({"answer": {"A": "B", "B": "C"}[previous], "supporting_facts": [["Public passage", 0]]})
                    return {"done": True, "done_reason": "stop", "message": {"content": text}}
            module = ModuleType("safetensors.torch")
            module.save_file = lambda tensors, path: Path(path).write_bytes(b"SOFTWARE-CACHE-NOT-TENSORS")
            module.load_file = lambda *_, **__: {"decide": Tensor(), "options": Tensor()}
            def score_head(head, feature, *, temperature):
                # This deterministic probability is a routing control, not a learned result.
                return {"p_true": .1, "physical_forwards": 0, "head_scores": 1, "temperature": temperature}
            environment = {"HALTIQ_RUN_ID": "software-control", "HALTIQ_PLAN_SHA256": "software-control", "HALTIQ_PHASE": args.phase}
            with patch.dict("os.environ", environment), \
                 patch.dict(sys.modules, {"safetensors.torch": module}), \
                 patch("haltiq_tau_hotpot.recovery_worker.preparation_binding", return_value=(prep, {}, {}, {"pilot_heldout": [case] * 4})), \
                 patch("haltiq_tau_hotpot.recovery_worker.QuestionRuntime", MockRuntime), \
                 patch("haltiq_tau_hotpot.hobson.HobsonBackend", Backend), \
                 patch("haltiq_tau_hotpot.hobson.load_head", return_value="SYNTHETIC-HEAD"), \
                 patch("haltiq_tau_hotpot.hobson.score_head", side_effect=score_head), \
                 patch("haltiq_tau_hotpot.worker.hydrate", return_value={"feature": {"provenance_sha256": "software-only"}}):
                completion = run_question(args, config, Control(root / "control"))
            runtime = instances[0]
            assert_envelope(runtime)
            self.assertFalse(runtime.journal.verify()["pending_or_uncertain"])
            self.assertEqual(runtime.journal.summary()["by_role"].get("writer_load"), 1)
            self.assertLessEqual(runtime.journal.summary()["by_role"].get("native_startup", 0), 1)
            rows = json.loads((args.output / "compare/episodes.json").read_text())
            self.assertEqual([row["arm"] for row in rows], r.ARMS)
            self.assertTrue(all(row["question_id"] == public["id"] for row in rows))
            self.assertGreater(runtime.journal.summary()["exact_response_reuses"], 0)
            return deepcopy(rows), dict(runtime.journal.summary()), completion, list(calls)

    def test_changed_prefix_answers_use_one_writer_then_one_native_placement(self):
        rows, physical, completion, calls = self.run_control()
        self.assertEqual(rows[0]["submitted"]["answer"], "A")
        self.assertEqual(rows[1]["submitted"]["answer"], "C")
        self.assertEqual(physical["by_role"]["writer"], 2)
        self.assertEqual(physical["by_role"]["reviewer"], 2)
        self.assertEqual(physical["by_role"]["native_startup"], 1)
        self.assertEqual(calls[:4], ["reviewer", "writer", "reviewer", "writer"])
        self.assertEqual(completion["planned_units"], 4)

    def test_known_invalid_response_is_paired_failure_without_writer_reload(self):
        rows, physical, completion, _ = self.run_control(invalid=True)
        self.assertEqual(physical["by_role"]["writer"], 1)
        self.assertEqual(physical["by_role"]["reviewer"], 1)
        self.assertEqual(completion["failed_units"], 3)
        self.assertEqual([row["status"] for row in rows], ["completed"] + ["completed_with_failure"] * 3)
        self.assertTrue(all(row["failure_category"] == "writer_value_error" for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
