"""Invented, byte-bound audit fixtures; no model inference or live OS inspection."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from haltiq_supervisor import engine, plan as supervisor
from haltiq_tau_hotpot import generation, hobson, probe_audit as audit, qualification as q, runtime, runtime_probe
from haltiq_tau_hotpot.journal import CallJournal, canonical
from tests.test_tau_hotpot_qualification import dump, tensors


class SyntheticProbe:
    """Real files/journal serialization; model operations are explicit fixture lambdas."""
    def __init__(self, root):
        self.root, self.run = root, root / "run"
        self.output = self.run / "probe"
        source = self.run / "source"
        source.mkdir(parents=True)
        for name in ("runtime", "runtime_probe", "journal", "hobson", "generation"):
            dump(source / "haltiq_tau_hotpot" / (name + ".py"), {"synthetic_not_executable": name})
        python = root / "fixture-python.exe"
        python.write_bytes(b"SYNTHETIC-NEVER-EXECUTED")
        checkpoint, base, prior = root / "checkpoint", root / "base", root / "prior-pilot"
        checkpoint.mkdir(); base.mkdir(); (checkpoint / "lora").mkdir()
        (checkpoint / "lora/adapter_model.safetensors").write_bytes(b"SYNTHETIC-ADAPTER-NOT-LOADED")
        (base / "model.safetensors-00001-of-00001.safetensors").write_bytes(b"SYNTHETIC-BASE-NOT-LOADED")
        weights = {name: (shape, [0.] * math.prod(shape)) for name, shape in q.HEAD_SHAPES.items()}
        tensors(checkpoint / "head.safetensors", weights)
        self.trained = prior / "train/learned/answer-head.safetensors"
        self.trained.parent.mkdir(parents=True)
        weights["q.bias"][1][0] = .001
        tensors(self.trained, weights)
        self.encoder = {"base/model.safetensors-00001-of-00001.safetensors": q.sha(base / "model.safetensors-00001-of-00001.safetensors"),
            "checkpoint/lora/adapter_model.safetensors": q.sha(checkpoint / "lora/adapter_model.safetensors")}
        self.encoder.update({"upstream/" + n: str(i) * 64 for i, n in enumerate(("modeling.py", "infer.py", "prompting.py", "schema.py"), 1)})
        (prior / "source").mkdir()
        (prior / "source/fixture.py").write_text("# synthetic former pilot, never run\n")
        old = {"purpose": "qualification_pilot", "scope": "custom_tau2_hotpot", "python": str(python),
            "stages": [{"id": p, "outputs": [{"path": "{run}/" + p + "/receipt.json"}]} for p in q.PHASES],
            "reviewed_source_sha256": {"fixture.py": q.sha(prior / "source/fixture.py")}}
        dump(prior / "plan.json", old)
        state = {"status": "completed", "run_id": "synthetic-prior", "plan_sha256": q.sha(prior / "plan.json"),
            "source_sha256": old["reviewed_source_sha256"], "python_sha256": q.sha(python), "stages": {}}
        for phase in q.PHASES:
            path = prior / phase / "receipt.json"; dump(path, {"synthetic_phase": phase})
            state["stages"][phase] = {"status": "completed", "exit_code": 0, "owned_tree_clear": True,
                "os_handoff_verified": True, "output_sha256": {str(path): q.sha(path)}}
        dump(prior / "supervisor.json", state)
        self.config = {"purpose": "runtime_probe", "targets": [512, 2048, 4096], "criteria": ["answer", "evidence"],
            "native_feature_cache": True, "response_cache": True, "new_learning": False, "benchmark_questions": 0,
            "generation_profile": deepcopy(generation.MAIN_PROFILE), "checkpoint": str(checkpoint), "base": str(base), "combined_pilot_proof": str(root / "pilot-proof.json")}
        self.binding = q.model_binding(self.config, {"encoder_sha256": self.encoder,
            "released_head_sha256": q.sha(checkpoint / "head.safetensors")})
        proof = {"schema": "haltiq-tau-hotpot-qualified-pilot-v1", "status": "PASS_QUALIFIED_ENGINEERING_PILOT",
            "qualified": True, "terminal": True, "pending_or_uncertain": False, "cleanup_verified": True,
            "actual_os_handoff_verified": True, "run": str(prior), "plan_sha256": state["plan_sha256"],
            "supervisor_sha256": q.sha(prior / "supervisor.json"), "run_id": state["run_id"],
            "trained_head_sha256": q.sha(self.trained), "model_binding": self.binding}
        dump(root / "pilot-proof.json", proof)
        self.config["combined_pilot_proof_sha256"] = q.sha(root / "pilot-proof.json")
        dump(root / "config.json", self.config)
        spec = {"path": str(root / "config.json"), "sha256": q.sha(root / "config.json")}
        pilot_spec = {"path": str(root / "pilot-proof.json"), "sha256": q.sha(root / "pilot-proof.json")}
        ready = root / "ready.json"; dump(ready, {"ready": True, "synthetic": True})
        start = datetime.now(timezone.utc) - timedelta(minutes=5)
        self.plan = {"version": 1, "scope": "custom_tau2_hotpot", "purpose": "runtime_probe", "status": "ready",
            "source_root": str(Path(audit.__file__).parents[1]), "python": str(python),
            "source_paths": [*supervisor.PACKAGES, "pyproject.toml"], "runtime_probe": spec, "inputs": [spec, pilot_spec],
            "readiness": {k: [{"path": str(ready), "sha256": q.sha(ready), "json_equals": {"ready": True}}] for k in supervisor.READINESS},
            "start_time": start.isoformat(), "deadline": (start + timedelta(minutes=25)).isoformat(),
            "hard_deadline": (start + timedelta(minutes=30)).isoformat(), "private_port": 11435,
            "gpu_lease": str(root / "gpu.lock"), "study_lease": str(root / "study.lock"),
            "resources": {"ram_gib": 7, "vram_gib": 3, "startup_ram_gib": 8.5, "startup_vram_gib": 9,
                "stability_seconds": 30, "minimum_rest_seconds": 5, "cpu_threads": 4, "gpu_owners": 1, "duty_cycle": .5},
            "stages": [{"id": "probe", "model_work": True, "gpu": True, "planned_units": 25,
                "argv": [str(python), "-B", "-m", "haltiq_tau_hotpot.runtime_probe", "probe", "--config", str(root / "config.json"),
                    "--output", "{run}/probe", "--previous", "{run}", "--deadline", "{deadline}", "--control", "{control}", "--allow-model-calls"],
                "completion_receipt": "{run}/probe/completion.json", "outputs": [{"path": "{run}/probe/" + n}
                    for n in ("probe.json", "completion.json", "runtime-terminal.json", "phase-timing.json")]}]}
        self.plan["reviewed_source_sha256"] = {p.relative_to(source).as_posix(): q.sha(p) for p in source.rglob("*") if p.is_file()}
        dump(self.run / "plan.json", self.plan)
        self.state = {"status": "completed", "run_id": "synthetic-probe-not-a-real-study", "plan_sha256": q.sha(self.run / "plan.json"),
            "source_sha256": self.plan["reviewed_source_sha256"], "python_sha256": q.sha(python), "supervisor_wall_seconds": 1500.,
            "stages": {"probe": {"status": "completed", "exit_code": 0, "owned_tree_clear": True, "os_handoff_verified": True,
                "started_at": start.isoformat(), "finished_at": (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()}}}
        self.journal, self.values = CallJournal(self.output / "physical"), {}
        self.call("writer_load", {"model": "qwen3:8b"}, {"done": True})
        state = {"question": "What is the synthetic key?", "evidence": [{"title": "Synthetic runtime note",
            "sentences": [{"index": 0, "text": "The synthetic key is control."}]}], "previous_draft": None}
        answer = {"answer": "control", "supporting_facts": [["Synthetic runtime note", 0]]}
        payload = generation.generation_payload(self.config, state)
        self.call("writer", payload, {"done": True, "done_reason": "stop", "prompt_eval_count": 50,
            "eval_count": 10, "message": {"content": json.dumps(answer)}})
        self.journal.reuse(deepcopy(self.journal.records[-1]), payload)
        review_rows, review_states, main_event = [], generation.synthetic_reviews(), None
        for index, (case_id, profile) in enumerate(generation.REVIEW_SEQUENCE):
            payload = generation.generation_payload(self.config, review_states[case_id], review=True, profile=profile)
            raw = {"done": True, "done_reason": "length" if profile == "legacy" else "stop", "prompt_eval_count": 50,
                "eval_count": 384 if profile == "legacy" else 12, "message": {"content": "Retain. Synthetic technical fixture."}}
            self.call("reviewer", payload, raw)
            event = deepcopy(self.journal.records[-1])
            if index == 1: main_event, main_payload = event, payload
            review_rows.append({"case_id": case_id, "profile": profile, "state": review_states[case_id],
                "status": "known_terminal_failure" if profile == "legacy" else "completed",
                "feedback": None if profile == "legacy" else raw["message"]["content"],
                "failure": "Writer/reviewer did not complete within its output budget" if profile == "legacy" else None,
                "physical_call_id": event["id"], "metadata": generation.response_metadata(raw),
                "request_sha256": q.fingerprint(payload)})
        self.journal.reuse(main_event, main_payload)
        self.call("writer_unload", {"model": "qwen3:8b"}, {"done": True})
        self.call("native_startup", {"checkpoint": str(checkpoint)}, {"metadata": {
            "encoder_sha256": self.encoder, "released_head_sha256": self.binding["released_head_sha256"], "head_parameters": 1053184}})
        rows = []
        for target in (512, 2048, 4096):
            for criterion in ("answer", "evidence"):
                public, draft, n = runtime_probe.synthetic_input(SimpleNamespace(prepare_state=lambda state, **_: {"input_tokens": 71 + state.count(" filler")}), target, criterion)
                public_state = runtime.decision_state(public, draft, [0], criterion)
                path = self.output / "features" / (q.fingerprint({"state": public_state, "criterion": criterion}) + ".safetensors")
                path.parent.mkdir(exist_ok=True)
                tensors(path, {"decide": ([2048], [0.] * 2048), "options": ([2, 2048], [0.] * 4096)})
                provenance = {"criterion": criterion, "state_sha256": q.fingerprint(public_state),
                    "instructions_sha256": q.fingerprint(hobson.INSTRUCTIONS_BY_CRITERION[criterion]),
                    "source_sha256": q.sha(source / "haltiq_tau_hotpot/hobson.py"), "option_order": ["false", "true"],
                    "token_ids_sha256": "a" * 64, "rendered_question_sha256": "b" * 64}
                feature = {"schema": hobson.FEATURE_SCHEMA, "criterion": criterion, "option_order": ["false", "true"],
                    "encoder_sha256": self.encoder, "provenance": provenance, "provenance_sha256": q.fingerprint(provenance),
                    "path": str(path), "sha256": q.sha(path)}
                actual = {"raw_logits_false_true": [0., 0.], "temperature": 1., "p_true_unrounded": .5, "p_true": .5,
                    "physical_forwards": 1, "head_scores": 2, "native_logits_false_true": [0., 0.],
                    "cached_replay_max_logit_error": 0., "input_tokens": n, "seconds": 0., "retrieval_indices": [0], "feature": feature}
                self.call("native_" + criterion, {"state": public_state, "criterion": criterion, "head": "released", "temperature": 1.}, actual)
                event = deepcopy(self.journal.records[-1])
                cached = []
                for head in (checkpoint / "head.safetensors", self.trained):
                    value = {k: actual[k] for k in ("raw_logits_false_true", "temperature", "p_true_unrounded", "p_true", "feature", "retrieval_indices")}
                    value.update(physical_forwards=0, head_scores=1, cache_reused=True,
                        original_native_call_id=event["id"], original_native_response_sha256=event["response_sha256"])
                    self.call("cached_head_" + criterion, {"original_native_call_id": event["id"], "feature_sha256": feature["sha256"],
                        "head_sha256": q.sha(head), "criterion": criterion, "temperature": 1.}, value)
                    cached.append(value)
                rows.append({"criterion": criterion, "target_tokens": target, "input_tokens": n, "public": public, "draft": draft,
                    "actual": actual, "cached_released": cached[0], "cached_trained": cached[1], "cold_first_forward_after_placement": not rows})
        self.probe = {"schema": "haltiq-runtime-probe-v2", "status": "PASS_RUNTIME_PROBE", "engineering_only": True,
            "benchmark_quality_claim": False, "new_learning": False, "long_study_started": False, "points": rows,
            "generation_results": [answer], "reviewer_ab": review_rows, "generation_profile": deepcopy(generation.MAIN_PROFILE), "calls": self.journal.verify(), "pilot_proof_sha256": self.config["combined_pilot_proof_sha256"],
            "pilot_run_id": proof["run_id"], "released_head_sha256": self.binding["released_head_sha256"], "trained_head_sha256": q.sha(self.trained),
            "wrapper_plan_sha256": self.state["plan_sha256"], "source_sha256": self.state["source_sha256"]}
        self.terminal = {"cleanup": {"verified": True, "private_port_closed": True, "owned_server_exited": True,
            "gpu_lease_released": True, "errors": [], "native_unloads": [{"allocated_bytes": 0, "reserved_bytes": 0}]},
            "calls": self.journal.verify(), "timings": {"startup_seconds": 0., "switch_seconds": 0.}, "active_seconds": 100., "intentional_idle_seconds": 100.}
        self.completion = {"run_id": self.state["run_id"], "plan_sha256": self.state["plan_sha256"], "phase": "probe", "terminal": True,
            "pending_or_uncertain": False, "cleanup_verified": True, "planned_units_accounted": True, "planned_units": 25,
            "completed_units": 25, "failed_units": 0, "not_run_units": 0, "finished_at": datetime.now(timezone.utc).isoformat()}
        self.timing = {"wall_seconds": 250., "active_seconds": 100., "intentional_idle_seconds": 100.}
        self.handoff = Mock()
        self.helpers = {"haltiq_supervisor.plan": supervisor, "haltiq_supervisor.engine": SimpleNamespace(verify_completion=engine.verify_completion, check_handoff=self.handoff),
            "pc_strands.control": SimpleNamespace(GpuLease=lambda *_: nullcontext()), "haltiq_tau_hotpot.hobson": SimpleNamespace(
                __file__=str(source / "haltiq_tau_hotpot/hobson.py"), FEATURE_SCHEMA=hobson.FEATURE_SCHEMA,
                INSTRUCTIONS_BY_CRITERION=hobson.INSTRUCTIONS_BY_CRITERION, encoder_binding=lambda *_: self.encoder)}
        self.modules = {"haltiq_tau_hotpot.runtime": SimpleNamespace(__file__=str(source / "haltiq_tau_hotpot/runtime.py"),
            **{name: getattr(runtime, name) for name in ("decision_state", "WRITER_SYSTEM", "REVIEWER_SYSTEM", "ANSWER_SCHEMA")}),
            "haltiq_tau_hotpot.generation": SimpleNamespace(__file__=str(source / "haltiq_tau_hotpot/generation.py"),
                **{name: getattr(generation, name) for name in ("MAIN_PROFILE", "REVIEW_SEQUENCE", "synthetic_reviews", "generation_payload", "parse_response", "response_metadata")}),
            "haltiq_tau_hotpot.runtime_probe": SimpleNamespace(__file__=str(source / "haltiq_tau_hotpot/runtime_probe.py"), pilot_binding=runtime_probe.pilot_binding),
            "haltiq_tau_hotpot.journal": SimpleNamespace(__file__=str(source / "haltiq_tau_hotpot/journal.py"), CallJournal=CallJournal)}
        for path in (Path(self.plan["gpu_lease"]), Path(self.plan["study_lease"]), self.run / "supervisor.lock"): path.write_bytes(b"fixture")
        self.sync()

    def call(self, role, request, value):
        result = self.journal.call(role, request, lambda: value)
        self.values[self.journal.records[-1]["id"]] = value
        return result

    def sync(self):
        """Repin owned artifacts after deliberate mutation; keep original request ancestry."""
        for event in self.journal.records:
            if event["event"] == "completed":
                path = self.journal.directory / event["response"]
                path.write_bytes(canonical(self.values[event["id"]]) + b"\n"); event["response_sha256"] = q.sha(path)
        for index, row in enumerate(self.probe["reviewer_ab"]):
            dump(self.output / "reviewer-ab" / f"{index}.json", row)
        for row in self.probe["points"]:
            dump(self.output / "points" / f"{row['criterion']}-{row['target_tokens']}.json", row)
        self.journal.path.write_text("".join(json.dumps(e) + "\n" for e in self.journal.records))
        for name, value in (("probe.json", self.probe), ("runtime-terminal.json", self.terminal), ("completion.json", self.completion), ("phase-timing.json", self.timing)):
            dump(self.output / name, value)
        self.state["stages"]["probe"]["output_sha256"] = {str(self.output / name): q.sha(self.output / name)
            for name in ("probe.json", "runtime-terminal.json", "completion.json", "phase-timing.json")}
        self.state["stages"]["probe"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        dump(self.run / "supervisor.json", self.state)

    def audit(self):
        @contextmanager
        def helpers(_): yield self.helpers
        with patch.object(q, "frozen_helpers", helpers), patch.object(audit.importlib, "import_module", side_effect=self.modules.__getitem__):
            return audit.audit_run(self.run, expected_plan_sha256=self.state["plan_sha256"])


class ProbeAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = SyntheticProbe(self.root)

    def test_bound_fixture_checks_all_actual_bytes_without_new_model_or_head_execution(self):
        result = self.fixture.audit()
        self.assertTrue(result["qualified"])
        self.assertEqual(result["physical_calls"]["completed"], 26)
        self.assertEqual(result["physical_calls"]["exact_response_reuses"], 2)
        self.assertEqual(result["new_model_forwards"], 0)
        self.assertEqual(result["new_head_scores"], 0)
        self.assertEqual(set(result["pinned_inputs_sha256"]), {str(self.root / name) for name in ("config.json", "pilot-proof.json", "ready.json")})
        self.assertEqual(len([c for c in result["timing"]["physical_calls"] if "input_tokens" in c]), 6)
        self.fixture.handoff.assert_called_once()

    def test_running_owner_is_rejected_before_any_outcomes_or_live_cleanup_check(self):
        self.fixture.state["status"] = "running"; self.fixture.sync()
        (self.fixture.output / "probe.json").unlink()
        with self.assertRaisesRegex(q.QualificationError, "genuinely terminal"): self.fixture.audit()
        self.fixture.handoff.assert_not_called()

    def test_frozen_source_interpreter_and_owned_output_bytes_cannot_change(self):
        for path in (self.fixture.run / "source/haltiq_tau_hotpot/runtime.py", Path(self.fixture.plan["python"]), self.fixture.output / "completion.json"):
            old = path.read_bytes(); path.write_bytes(old + b"changed")
            try:
                with self.subTest(path=path.name), self.assertRaises(q.QualificationError): self.fixture.audit()
            finally: path.write_bytes(old)

    def test_pinned_input_and_readiness_changes_fail_before_native_evidence(self):
        for path in (self.root / "config.json", self.root / "ready.json"):
            old = path.read_bytes(); dump(path, {"changed": True})
            try:
                with self.subTest(path=path.name), self.assertRaises(ValueError): self.fixture.audit()
            finally: path.write_bytes(old)

    def test_rebound_point_artifact_cannot_smuggle_real_questions_or_changed_public_state(self):
        row = self.fixture.probe["points"][0]
        row["public"]["question"] = "A real benchmark question is forbidden"
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "Non-synthetic"): self.fixture.audit()

    def test_criterion_length_and_cold_claims_cannot_be_changed_after_hash_rebinding(self):
        row = self.fixture.probe["points"][0]
        for field, value in (("criterion", "evidence"), ("input_tokens", 490), ("cold_first_forward_after_placement", False)):
            old = row[field]; row[field] = value; self.fixture.sync()
            try:
                with self.subTest(field=field), self.assertRaises(q.QualificationError): self.fixture.audit()
            finally: row[field] = old; self.fixture.sync()

    def test_feature_provenance_and_finite_tensor_bytes_are_inspected(self):
        feature = self.fixture.probe["points"][0]["actual"]["feature"]
        original = deepcopy(feature["provenance"])
        feature["provenance"]["state_sha256"] = "f" * 64
        feature["provenance_sha256"] = q.fingerprint(feature["provenance"])
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "provenance"): self.fixture.audit()
        feature["provenance"], feature["provenance_sha256"] = original, q.fingerprint(original)
        path = Path(feature["path"])
        tensors(path, {"decide": ([2048], [float("nan")] + [0.] * 2047), "options": ([2, 2048], [0.] * 4096)})
        feature["sha256"] = q.sha(path); self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "Nonfinite"): self.fixture.audit()

    def test_changed_allowed_synthetic_text_still_requires_original_pending_request(self):
        self.fixture.probe["points"][0]["public"]["context"][0][1][0] += " filler"
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "Unbound"): self.fixture.audit()

    def test_empty_readiness_gate_is_not_sufficient_even_with_a_rebound_owner(self):
        self.fixture.plan["readiness"]["combined_pilot"] = []
        dump(self.fixture.run / "plan.json", self.fixture.plan)
        hashed = q.sha(self.fixture.run / "plan.json")
        self.fixture.state["plan_sha256"] = self.fixture.probe["wrapper_plan_sha256"] = self.fixture.completion["plan_sha256"] = hashed
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "readiness"): self.fixture.audit()

    def test_cached_response_requires_exact_native_ancestry_and_zero_new_encoder_count(self):
        row = self.fixture.probe["points"][0]["cached_trained"]
        for field, value in (("original_native_call_id", "other-native"), ("physical_forwards", 1), ("temperature", .9)):
            old = row[field]; row[field] = value; self.fixture.sync()
            try:
                with self.subTest(field=field), self.assertRaises(q.QualificationError): self.fixture.audit()
            finally: row[field] = old; self.fixture.sync()

    def test_released_native_cached_raw_logits_and_probabilities_are_checked(self):
        row = self.fixture.probe["points"][0]["cached_released"]
        row.update(raw_logits_false_true=[1., 1.], p_true=.5, p_true_unrounded=.5)
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "cached logits"): self.fixture.audit()

    def test_changed_trained_head_or_qualified_prior_owner_blocks_reuse(self):
        for path in (self.fixture.trained, self.root / "prior-pilot/supervisor.json"):
            old = path.read_bytes(); path.write_bytes(old + b"changed")
            try:
                with self.subTest(path=path.name), self.assertRaises(ValueError): self.fixture.audit()
            finally: path.write_bytes(old)

    def test_extra_reuse_wrong_role_or_missing_completion_is_not_hidden_by_summary(self):
        original = deepcopy(self.fixture.journal.records)
        for kind in ("extra", "role", "unresolved"):
            self.fixture.journal.records = deepcopy(original)
            if kind == "extra": self.fixture.journal.records.append(deepcopy(original[-3]))
            elif kind == "role": next(e for e in self.fixture.journal.records if e["event"] == "exact_response_reuse")["role"] = "native_answer"
            else: self.fixture.journal.records.pop()
            self.fixture.sync()
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.fixture.audit()

    def test_invented_generation_result_cannot_replace_the_committed_writer_response(self):
        self.fixture.probe["generation_results"][0] = {"answer": "forged", "supporting_facts": []}
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "committed response"): self.fixture.audit()

    def test_ab_profile_order_metadata_and_main_completion_are_checked_against_raw_calls(self):
        rows = self.fixture.probe["reviewer_ab"]
        for field, value in (("profile", "legacy"), ("metadata", {"invented": True}), ("feedback", "forged feedback")):
            original = deepcopy(rows[1][field])
            rows[1][field] = value
            self.fixture.sync()
            try:
                with self.subTest(field=field), self.assertRaises(q.QualificationError): self.fixture.audit()
            finally:
                rows[1][field] = original
                self.fixture.sync()
        self.fixture.values[rows[1]["physical_call_id"]]["done_reason"] = "length"
        self.fixture.sync()
        with self.assertRaises(q.QualificationError): self.fixture.audit()

    def test_review_reuse_cannot_redirect_to_legacy_even_with_rebound_summary(self):
        rows = self.fixture.probe["reviewer_ab"]
        reuse = next(e for e in self.fixture.journal.records if e["event"] == "exact_response_reuse" and e["role"] == "reviewer")
        legacy = next(e for e in self.fixture.journal.records if e["event"] == "completed" and e["id"] == rows[0]["physical_call_id"])
        pending = next(e for e in self.fixture.journal.records if e["event"] == "pending" and e["id"] == legacy["id"])
        reuse.update(id=legacy["id"], request_sha256=pending["request_sha256"], response_sha256=legacy["response_sha256"])
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "first main-profile"): self.fixture.audit()

    def test_cleanup_failure_bad_phase_time_and_live_handoff_failure_suppress_qualification(self):
        self.fixture.terminal["cleanup"]["native_unloads"][0]["allocated_bytes"] = 1
        self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "cleanup failed"): self.fixture.audit()
        self.fixture.terminal["cleanup"]["native_unloads"][0]["allocated_bytes"] = 0
        self.fixture.timing["active_seconds"] = 1000.; self.fixture.sync()
        with self.assertRaisesRegex(q.QualificationError, "phase timing"): self.fixture.audit()
        self.fixture.timing["active_seconds"] = 100.; self.fixture.sync()
        self.fixture.handoff.side_effect = RuntimeError("fixture GPU lease busy")
        with self.assertRaisesRegex(RuntimeError, "GPU lease busy"): self.fixture.audit()

    def test_new_private_receipt_fails_closed_and_cannot_overwrite_or_mutate_evidence(self):
        output = self.root / "qualification.json"
        with patch.object(audit, "audit_run", side_effect=q.QualificationError("synthetic missing evidence")):
            result = audit.qualify(self.fixture.run, output, expected_plan_sha256="0" * 64)
        self.assertEqual(result["schema"], audit.SCHEMA)
        self.assertFalse(result["qualified"])
        self.assertNotIn("ready", result)
        with self.assertRaises(q.QualificationError): audit.qualify(self.fixture.run, output, expected_plan_sha256="0" * 64)
        with self.assertRaises(q.QualificationError): audit.qualify(self.fixture.run, self.fixture.run / "probe/new.json", expected_plan_sha256="0" * 64)

    def test_auditor_import_does_not_initialize_torch_transformers_or_tau(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import sys;import haltiq_tau_hotpot.probe_audit;assert not any(x in sys.modules for x in ('torch','transformers','tau2'))"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
