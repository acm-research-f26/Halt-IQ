"""Synthetic scheduling/ownership controls only; no model-quality evidence."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from haltiq_supervisor.plan import READINESS, inspect, source_files
from haltiq_tau_hotpot import forecasting, planning, protocol
from haltiq_tau_hotpot.hobson import file_hash
from haltiq_tau_hotpot.journal import durable_write, fingerprint
from haltiq_tau_hotpot.qualification import journal


def case(name, index, partition):
    identifier = name + str(index)
    return {"partition": partition, "public": {"id": identifier, "question": identifier + "?",
        "type": "bridge" if index % 2 == 0 else "comparison", "level": "synthetic fixture",
        "context": [["Public", ["A public fact."]]]},
        "evaluator": {"answer": "correct", "supporting_facts": [["Public", 0]]}}


class PlanningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.source = Path(temporary.name), Path(__file__).resolve().parents[1]
        self.now = datetime.now(timezone.utc)
        self.freeze = {"schema": "tau2-hotpot-main-v1", "main": deepcopy(protocol.MAIN_SETTINGS),
            "selection": {"answer_blind": True, "quality_filtering": False, "length_filtering": False,
                          "replacement_allowed": False}, "files": {}}
        fresh = [case("fresh", index, "fresh") for index in range(60)]
        for name, n, partition in planning.COHORTS:
            rows = fresh[:n] if partition == "fresh" else [case(name, index, partition) for index in range(n)]
            self.freeze["files"][name] = self.save(name, rows) | {"count": n,
                "type_counts": {"bridge": n // 2, "comparison": n // 2}}
        self.freeze_spec = self.save("freeze", self.freeze)
        self.checkpoint, self.base = self.root / "checkpoint", self.root / "base"
        (self.checkpoint / "lora").mkdir(parents=True)
        self.base.mkdir()
        models = {"released_head_sha256": self.checkpoint / "head.safetensors",
            "adapter_sha256": self.checkpoint / "lora/adapter_model.safetensors",
            "base_weights_sha256": self.base / "model.safetensors-00001-of-00001.safetensors"}
        self.model_inputs = []
        for path in list(models.values()) + [self.root / "tau2.py", self.root / "strands_decider.py"]:
            path.write_bytes(b"SYNTHETIC-BINDING-NOT-A-MODEL\n")
            self.model_inputs.append(planning._spec(path))
        self.model_binding = {key: file_hash(path) for key, path in models.items()} | {
            "head_parameters": 1053184, "upstream_sha256": {name: protocol.PINS["upstream/" + name]
                for name in ("modeling.py", "infer.py", "prompting.py", "schema.py")}}
        self.runtime = {"hosted_services": False, "checkpoint": str(self.checkpoint), "base": str(self.base),
            "reranker": str(self.root), "ollama": str(self.root / "not-executed.exe"),
            "ollama_models": str(self.root), "writer_digest": "0" * 64,
            "gpu_lease": str(self.root / "gpu.lock"), "study_lease": str(self.root / "study.lock"),
            "tau_revision": "synthetic fixture", "model_inputs": self.model_inputs,
            "runtime_modules": {"tau2": str(self.root / "tau2.py"), "strands_decider": str(self.root / "strands_decider.py")},
            "secret_unrecognized_field": "MUST-NOT-BE-COPIED"}
        self.runtime_spec = self.save("runtime", self.runtime)
        self.template = self.root / "template"
        planning.prepare_template(self.runtime_spec["path"], self.freeze_spec["path"], self.source,
                                  sys.executable, self.template)
        self.resource = {"schema": "haltiq-main-resource-admission-v1", "status": "PASS_RESOURCE_ADMISSION",
            "os_gpu_lease_free": True, "study_lease_free": True, "private_port_closed": True,
            "config": self.runtime_spec,
            "samples": [{"at": (self.now - timedelta(seconds=31 - index * 10)).isoformat(),
                "available_ram_gib": 11.1, "free_vram_gib": 9.2, "gpu_temperature_c": 45} for index in range(4)]}
        self.resource_spec = self.save("resources", self.resource)
        self.pilot = self.make_pilot()
        self.pilot_spec = self.save("pilot", self.pilot)
        self.review = self.make_review()
        self.review_spec = self.save("review", self.review)

    def save(self, name, value):
        path = self.root / (name + ".json")
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        return planning._spec(path)

    def make_review(self):
        template_plan = planning.read(self.template / "plan-template.json")
        manifest = {name: file_hash(path) for name, path in source_files(template_plan).items()}
        ready = self.save("readiness", {"scope": "synthetic test fixture", "ready": True})
        readiness = {key: [{**ready, "json_equals": {"ready": True}}] for key in READINESS}
        readiness["combined_pilot"] = [{**self.pilot_spec, "json_equals": {"qualified": True,
            "terminal": True, "pending_or_uncertain": False, "cleanup_verified": True}}]
        return {"schema": "haltiq-main-source-review-v1", "status": "PASS_REVIEWED_SOURCE",
            "source_root": str(self.source), "source_sha256": manifest,
            "exact_prefix_cache_validated": True, "readiness": readiness}

    def make_pilot(self):
        run = self.root / "synthetic-pilot"
        (run / "source").mkdir(parents=True)
        (run / "source/fixture.py").write_text("# Synthetic owner, not model execution.\n", encoding="utf-8")
        manifest = {"fixture.py": file_hash(run / "source/fixture.py")}
        state = {"run_id": "SYNTHETIC-NOT-A-REAL-RUN", "status": "completed", "stages": {},
                 "source_sha256": manifest, "python_sha256": file_hash(sys.executable)}
        plan = {"purpose": "qualification_pilot", "scope": "custom_tau2_hotpot", "stages": [],
                "reviewed_source_sha256": manifest, "python": sys.executable}
        physical, timing, outputs = {}, {}, {}
        roles = {"collect": ["writer_load"] + ["writer"] * 64 + ["retrieval", "writer_unload"],
            "features": ["native_startup"] + ["native_answer"] * 64,
            "train": ["head_training_and_calibration"],
            "compare": ["reviewer", "cpu_head", "writer", "writer_load", "native_startup", "writer_unload"]}
        for phase in ("collect", "features", "train", "compare"):
            directory = run / phase
            (directory / "physical/responses").mkdir(parents=True)
            records = []
            for index, role in enumerate(roles[phase]):
                response = directory / "physical/responses" / (str(index) + ".json")
                durable_write(response, {"input_tokens": 4096, "scope": "synthetic timing fixture"})
                identity, at = str(index), (self.now - timedelta(seconds=10)).isoformat()
                records += [{"event": "pending", "id": identity, "role": role, "at": at,
                    "request_sha256": "0" * 64}, {"event": "completed", "id": identity, "role": role,
                    "at": at, "elapsed_seconds": .01, "response": "responses/" + response.name,
                    "response_sha256": file_hash(response)}]
            journal_path = directory / "physical/calls.jsonl"
            journal_path.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
            values = journal(directory / "physical")
            physical[phase] = {key: value for key, value in values.items() if key != "responses"}
            phase_timing = {"wall_seconds": 2., "active_seconds": 1., "intentional_idle_seconds": 1.}
            durable_write(directory / "phase-timing.json", phase_timing)
            timing[phase] = {**phase_timing, "receipt_sha256": file_hash(directory / "phase-timing.json")}
            relative = ["physical/calls.jsonl", "phase-timing.json"]
            if phase != "train":
                runtime = {"startup_seconds": .01, "switch_seconds": .01}
                durable_write(directory / "runtime-terminal.json", {"timings": runtime})
                timing[phase]["runtime_startup_and_switch"] = runtime
                relative.append("runtime-terminal.json")
            stage_outputs = [{"path": "{run}/" + phase + "/" + value} for value in relative]
            plan["stages"].append({"id": phase, "outputs": stage_outputs})
            hashes = {str((directory / value).resolve()): file_hash(directory / value) for value in relative}
            state["stages"][phase] = {"status": "completed", "exit_code": 0,
                "owned_tree_clear": True, "os_handoff_verified": True, "output_sha256": hashes}
            outputs.update({str(Path(phase) / value).replace("\\", "/"): file_hash(directory / value) for value in relative})
        durable_write(run / "plan.json", plan)
        state["plan_sha256"] = file_hash(run / "plan.json")
        durable_write(run / "supervisor.json", state)
        return {"schema": "haltiq-tau-hotpot-qualified-pilot-v1", "status": "PASS_QUALIFIED_ENGINEERING_PILOT",
            "scope": "SYNTHETIC UNIT TEST FIXTURE, NEVER SCIENTIFIC EVIDENCE", "qualified": True,
            "terminal": True, "cleanup_verified": True, "actual_os_handoff_verified": True,
            "pending_or_uncertain": False, "planned_units": [64, 64, 1, 16],
            "run": str(run), "run_id": state["run_id"], "plan_sha256": state["plan_sha256"],
            "supervisor_sha256": file_hash(run / "supervisor.json"), "source_sha256": manifest,
            "model_binding": self.model_binding, "pinned_inputs_sha256": {}, "output_sha256": outputs,
            "trained_head_sha256": "e" * 64,
            "physical_calls": physical, "timing": {"phases": timing}}

    def compile(self, name="compiled"):
        # Isolate compilation mechanics from the separate V2 lineage adapter.
        # These are invented unit-test observations, never live readiness.
        actual_forecast = forecasting.forecast_route
        def synthetic_complete_forecast(measured, **kwargs):
            measured = {**measured, "per_question_validation_overhead_max_seconds": 2., "runtime_instances_measured": 6}
            return actual_forecast(measured, **kwargs)
        with patch.object(planning, "forecast_route", side_effect=synthetic_complete_forecast):
            return planning.compile_main(self.template, self.root / name, start_time=self.now.isoformat(),
                review_receipt=self.review_spec["path"], pilot_proof=self.pilot_spec["path"],
                resource_receipt=self.resource_spec["path"], now=self.now)

    def make_probe(self, version=1):
        run = self.root / "synthetic-runtime-probe"
        (run / "source").mkdir(parents=True)
        (run / "source/fixture.py").write_text("# No models; synthetic scheduling fixture.\n", encoding="utf-8")
        source = {"fixture.py": file_hash(run / "source/fixture.py")}
        directory = run / "probe/physical"
        (directory / "responses").mkdir(parents=True)
        records, points = [], []
        def append(role, value, seconds=.01, request=None):
            identity = "call-" + str(sum(r["event"] == "pending" for r in records))
            response = directory / "responses" / (identity + ".json")
            durable_write(response, value)
            at = self.now.isoformat()
            pending = {"event": "pending", "id": identity, "role": role,
                "request_sha256": fingerprint(request) if request is not None else "0" * 64, "at": at}
            event = {"event": "completed", "id": identity, "role": role, "at": at,
                "elapsed_seconds": seconds, "response": "responses/" + response.name, "response_sha256": file_hash(response)}
            records.extend((pending, event))
            return {"event": "exact_response_reuse", "id": identity, "role": role, "at": at,
                "request_sha256": pending["request_sha256"], "response_sha256": event["response_sha256"], "new_physical_calls": 0}
        append("native_startup", {"scope": "synthetic fixture"})
        for target in (512, 2048, 4096):
            for criterion in ("answer", "evidence"):
                actual = {"input_tokens": target, "physical_forwards": 1, "head_scores": 2,
                          "raw_logits_false_true": [-1., 1.], "temperature": 1., "unique_fixture": criterion + str(target)}
                same = {**actual, "physical_forwards": 0, "head_scores": 1}
                trained = {**same, "raw_logits_false_true": [-2., 2.]}
                append("native_" + criterion, actual)
                append("cached_head_" + criterion, same)
                append("cached_head_" + criterion, trained)
                points.append({"criterion": criterion, "target_tokens": target, "input_tokens": target,
                    "actual": actual, "cached_released": same, "cached_trained": trained,
                    "cold_first_forward_after_placement": not points, "public": {"type": "synthetic_control"}})
        append("writer_load", {"scope": "synthetic fixture"})
        ab, compact = [], []
        if version == 1:
            records.append(append("writer", {"scope": "synthetic generation fixture"}, seconds=24.))
            records.append(append("reviewer", {"scope": "synthetic review fixture"}))
        else:
            from haltiq_tau_hotpot.generation import (MAIN_PROFILE, REVIEW_SEQUENCE, generation_payload,
                response_metadata, synthetic_reviews)
            config = {"purpose": "runtime_probe", "generation_profile": deepcopy(MAIN_PROFILE)}
            config_spec = self.save("probe-config", config)
            state = {"question": "What is the synthetic key?", "evidence": [{"title": "Synthetic runtime note",
                "sentences": [{"index": 0, "text": "The synthetic key is control."}]}], "previous_draft": None}
            answer = {"answer": "control", "supporting_facts": [["Synthetic runtime note", 0]]}
            raw = {"done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 12,
                   "message": {"content": json.dumps(answer)}}
            records.append(append("writer", raw, seconds=24., request=generation_payload(config, state)))
            reuse = None
            for index, (case_id, profile) in enumerate(REVIEW_SEQUENCE):
                state = synthetic_reviews()[case_id]
                # Legacy truncation is retained as a technical failure; main
                # cells are two predeclared successful synthetic responses.
                raw = {"done": True, "done_reason": 'length' if index == 0 else 'stop',
                    "prompt_eval_count": 120, "eval_count": 12, "message": {"content": "Retain: public synthetic note supports this."}}
                request = generation_payload(config, state, review=True, profile=profile)
                event = append("reviewer", raw, seconds=[4., 7., 11., 5.][index], request=request)
                failure = 'Writer/reviewer did not complete within its output budget' if index == 0 else None
                row = {"case_id": case_id, "profile": profile, "state": state,
                    "status": 'known_terminal_failure' if failure else 'completed', "failure": failure,
                    "feedback": None if failure else raw['message']['content'], "physical_call_id": event['id'],
                    "metadata": response_metadata(raw), "request_sha256": fingerprint(request)}
                durable_write(run / 'probe/reviewer-ab' / f'{index}.json', row)
                ab.append(row)
                compact.append({"id": event['id'], "case_id": case_id, "profile": profile,
                    "state_sha256": fingerprint(state), "request_sha256": fingerprint(request),
                    "seconds": [4., 7., 11., 5.][index], "metadata": response_metadata(raw), "status": row['status']})
                if index == 1: reuse = event
            records.append(reuse)
        append("writer_unload", {"scope": "synthetic fixture"})
        if version == 2:
            generation_roles = {'writer_load', 'writer', 'reviewer', 'writer_unload'}
            records = [r for r in records if r['role'] in generation_roles] + [r for r in records if r['role'] not in generation_roles]
        (directory / "calls.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
        from haltiq_tau_hotpot.journal import CallJournal
        stats = CallJournal(directory).verify()
        outputs = ["probe.json", "runtime-terminal.json", "physical/calls.jsonl", "completion.json"]
        readiness = self.save("probe-readiness", {"ready": True, "scope": "synthetic fixture"})
        plan = {"purpose": "runtime_probe", "python": sys.executable, "reviewed_source_sha256": source,
            "inputs": [self.runtime_spec, self.pilot_spec],
            "readiness": {key: [{**readiness, "json_equals": {"ready": True}}] for key in READINESS},
            "stages": [{"id": "probe", "planned_units": 25 if version == 2 else 22,
                        "outputs": [{"path": "{run}/probe/" + name} for name in outputs]}]}
        if version == 2:
            plan['runtime_probe'] = config_spec
            plan['inputs'].append(config_spec)
        durable_write(run / "plan.json", plan)
        plan_sha, run_id = file_hash(run / "plan.json"), "SYNTHETIC-RUNTIME-PROBE"
        value = {"schema": "haltiq-runtime-probe-v" + str(version), "status": "PASS_RUNTIME_PROBE", "engineering_only": True,
            "benchmark_quality_claim": False, "new_learning": False, "long_study_started": False,
            "wrapper_plan_sha256": plan_sha, "source_sha256": source, "pilot_proof_sha256": self.pilot_spec["sha256"],
            "released_head_sha256": self.model_binding["released_head_sha256"], "trained_head_sha256": "e" * 64,
            "points": points, "calls": stats}
        if version == 2:
            value.update(generation_profile=deepcopy(MAIN_PROFILE), reviewer_ab=ab, generation_results=[answer])
        durable_write(run / "probe/probe.json", value)
        durable_write(run / "probe/runtime-terminal.json", {"cleanup": {"verified": True}, "calls": stats,
            "timings": {"startup_seconds": .01, "switch_seconds": .01}})
        durable_write(run / "probe/completion.json", {"run_id": run_id, "plan_sha256": plan_sha,
            "phase": "probe", "terminal": True, "pending_or_uncertain": False, "cleanup_verified": True,
            "planned_units_accounted": True, "planned_units": 25 if version == 2 else 22,
            "completed_units": 25 if version == 2 else 22, "failed_units": 0, "not_run_units": 0})
        hashes = {str(run / "probe" / name): file_hash(run / "probe" / name) for name in outputs}
        state = {"run_id": run_id, "plan_sha256": plan_sha, "source_sha256": source, "status": "completed",
            "python_sha256": file_hash(sys.executable), "stages": {"probe": {"status": "completed", "exit_code": 0,
                "owned_tree_clear": True, "os_handoff_verified": True, "output_sha256": hashes}}}
        durable_write(run / "supervisor.json", state)
        proof = {"schema": "haltiq-qualified-runtime-probe-v" + str(version), "status": "PASS_QUALIFIED_RUNTIME_PROBE",
            "qualified": True, "terminal": True, "cleanup_verified": True, "pending_or_uncertain": False,
            "pilot_proof_sha256": self.pilot_spec["sha256"], "model_binding": self.model_binding,
            "run": str(run), "plan_sha256": plan_sha, "supervisor_sha256": file_hash(run / "supervisor.json"),
            "source_sha256": source, "output_sha256": {"probe/" + name: file_hash(run / "probe" / name) for name in outputs}}
        if version == 2:
            proof.update(generation_profile=deepcopy(MAIN_PROFILE), reviewer_ab=compact)
            proof['output_sha256'].update({'probe/reviewer-ab/' + str(i) + '.json': file_hash(run / 'probe/reviewer-ab' / f'{i}.json') for i in range(4)})
        proof["pinned_inputs_sha256"] = {spec["path"]: spec["sha256"] for spec in plan["inputs"]
            + [item for group in plan["readiness"].values() for item in group]}
        return self.save("probe-proof", proof)

    def recovered_measurement_fixture(self):
        """Scheduling adapter fixture, not a real qualified recovery run."""
        from haltiq_tau_hotpot.journal import CallJournal
        old, new = Path(self.pilot["run"]), self.root / "synthetic-recovery"
        sources, timings, physical = {}, {}, {}
        for phase in ("collect", "features", "train"):
            directory = old / phase
            sources[phase] = {"run": str(old), "phase": phase,
                "physical_dir": str(directory / "physical"), "timing_path": str(directory / "phase-timing.json"),
                "runtime_terminal_path": str(directory / "runtime-terminal.json") if phase != "train" else None}
            timings[phase] = deepcopy(self.pilot["timing"]["phases"][phase])
            physical[phase] = CallJournal(directory / "physical").verify()
        for index in range(1, 5):
            phase = "compare_q" + str(index)
            directory = new / phase
            shutil.copytree(old / "compare/physical", directory / "compare/physical")
            for filename, target in (("phase-timing.json", directory / "phase-timing.json"),
                                     ("runtime-terminal.json", directory / "compare/runtime-terminal.json")):
                shutil.copyfile(old / "compare" / filename, target)
            sources[phase] = {"run": str(new), "phase": phase,
                "physical_dir": str(directory / "compare/physical"), "timing_path": str(directory / "phase-timing.json"),
                "runtime_terminal_path": str(directory / "compare/runtime-terminal.json")}
            timings[phase] = deepcopy(self.pilot["timing"]["phases"]["compare"])
            timings[phase]["supervisor_stage_wall_seconds"] = 4. + index
            physical[phase] = CallJournal(directory / "compare/physical").verify()
        proof = {**deepcopy(self.pilot), "schema": "haltiq-tau-hotpot-qualified-pilot-v2",
            "run": str(new), "measurement_sources": sources, "physical_calls": physical,
            "timing": {"phases": timings}, "preparation_owner": {"source_sha256": self.pilot["source_sha256"]},
            "source_sha256": {"recovery_fixture.py": "f" * 64},
            "historical_failed_compare": {"scope": "synthetic prior failed expense, not new outcomes", "completed": 6}}
        inputs = [planning._spec(path) for origin in sources.values() for path in
                  (Path(origin["physical_dir"]) / "calls.jsonl", Path(origin["timing_path"]))]
        inputs += [self.model_inputs[-1]]
        return self.save("recovered-pilot", proof), proof, {"measurement_sources": sources, "verification_inputs": inputs}

    def test_import_and_prepare_have_no_model_process_or_clock_adoption(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys;import haltiq_tau_hotpot.planning;assert not any(x in sys.modules for x in ('torch','tau2','transformers'))"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = planning.read(self.template / "main-config-template.json")
        self.assertIsNone(config["start_time"])
        self.assertIsNone(config["model_deadline"])
        self.assertNotIn("secret_unrecognized_field", config)
        self.assertEqual(inspect(planning.read(self.template / "plan-template.json"))["status"], "blocked_no_launch")
        self.assertFalse((self.template / "supervisor.json").exists())

    def test_compile_complete_five_phase_plan_without_adoption_or_model_work(self):
        with patch("subprocess.Popen") as process:
            result = self.compile()
        process.assert_not_called()
        self.assertEqual(result["status"], "compiled_not_adopted")
        directory = self.root / "compiled"
        plan = planning.read(directory / "plan.json")
        self.assertEqual([stage["planned_units"] for stage in plan["stages"]], [620, 1240, 1, result["selected_n"] * 8, 1])
        self.assertEqual(inspect(plan)["status"], "ready_not_started")
        self.assertEqual(plan["resources"]["startup_ram_gib"], 11)
        self.assertIsNone(plan["codex_argv"])
        self.assertFalse((directory / "supervisor.json").exists())
        self.assertEqual((planning._aware(plan["deadline"]) - self.now).total_seconds(), 14 * 3600)
        self.assertEqual((planning._aware(plan["model_deadline"]) - self.now).total_seconds(), 13 * 3600)
        self.assertEqual((planning._aware(plan["hard_deadline"]) - self.now).total_seconds(), 14 * 3600 + 300)
        context = protocol.load_main(planning.read(directory / "main-config.json"))
        self.assertEqual(context["fresh_n"], result["selected_n"])
        self.assertFalse(context["fresh_freeze"]["fresh_scoring_started"])

    def test_full_route_counts_rest_proxy_and_no_quality_based_selection(self):
        measured = planning.pilot_measurements(self.pilot_spec["path"])
        value = planning.forecast_route(measured)
        first = value["candidates"][0]
        self.assertEqual(first["counts"]["native_answer"], 820)
        self.assertEqual(first["counts"]["native_evidence"], 820)
        self.assertEqual(first["counts"]["writer"], 820)
        self.assertEqual(first["counts"]["reviewer"], 180)
        self.assertGreaterEqual(first["components_seconds"]["cpu_head"], 400 * 5)
        self.assertGreater(first["components_seconds"]["native_evidence"], first["components_seconds"]["native_answer"])
        self.assertEqual(value["finalization_reserve_seconds"], 3600)
        self.pilot["comparison"] = {"quality": "MUST NEVER AFFECT N"}
        self.pilot_spec = self.save("pilot", self.pilot)
        again = planning.forecast_route(planning.pilot_measurements(self.pilot_spec["path"]))
        self.assertEqual(again["selected_n"], value["selected_n"])
        self.assertEqual(again["candidates"], value["candidates"])

    def test_main_profile_is_explicit_and_legacy_review_cost_is_widened_before_new_measurement(self):
        from haltiq_tau_hotpot.generation import MAIN_PROFILE
        config = planning.read(self.template / 'main-config-template.json')
        self.assertEqual(config['generation_profile'], MAIN_PROFILE)
        measured = planning.pilot_measurements(self.pilot_spec['path'])
        measured['steady_role_max_seconds']['reviewer'] = 8.
        route = forecasting.forecast_route(measured)
        self.assertTrue(route['reviewer_cost_is_unmeasured_proxy'])
        self.assertEqual(route['main_reviewer_seconds_per_call'], 16.)
        self.assertEqual(route['candidates'][0]['components_seconds']['reviewer'], 180 * 32.)
        self.assertIn('768-token', route['reviewer_timing_basis'])

    def test_prefix_forecast_requires_exact_known_failure_memoization(self):
        from haltiq_tau_hotpot.runtime import LocalRuntime
        control = SimpleNamespace(before_work=lambda **_: None, rest_after_block=lambda _: None)
        runtime = LocalRuntime({"response_cache": True, "writer_digest": "synthetic-fixture"},
                               self.root / "known-failure-runtime", control)
        raw = {"done": True, "done_reason": "stop", "message": {"content": "[]"}}
        with patch.object(runtime, "_writer_ready") as placement, patch.object(runtime, "_http", return_value=raw) as request:
            for _ in range(7):  # Repeated logical arms, never seven model calls.
                with self.assertRaisesRegex(ValueError, "schema"):
                    runtime.generate({"question": "Synthetic?", "evidence": "Public fixture"})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(placement.call_count, 1)
        self.assertEqual(runtime.journal.verify()["exact_response_reuses"], 6)
        self.assertEqual(runtime.journal.summary()["by_role"], {"writer": 1})
        research = planning.read(self.template / "research-protocol.json")
        self.assertIs(research["forecast_policy"]["known_terminal_failure_cache"], True)
        policy = deepcopy(forecasting.FORECAST_POLICY)
        policy["known_terminal_failure_cache"] = False
        with self.assertRaisesRegex(ValueError, "canonical protocol"):
            forecasting.forecast_route(planning.pilot_measurements(self.pilot_spec["path"]), policy=policy)

    def test_recovery_measurements_use_seven_real_directories_and_separate_failed_history(self):
        spec, proof, owners = self.recovered_measurement_fixture()
        with patch("haltiq_tau_hotpot.recovery.verify_qualified_pilot", create=True,
                   return_value=(proof, self.checkpoint / "head.safetensors", owners)) as verified:
            measured = forecasting.pilot_measurements(spec["path"])
        verified.assert_called_once_with(spec["path"])
        self.assertEqual(set(measured["phase_wall_seconds"]), set(proof["measurement_sources"]))
        self.assertNotIn("compare", measured["phase_wall_seconds"])
        self.assertEqual(measured["runtime_instances_measured"], 6)
        self.assertEqual(measured["per_question_validation_overhead_max_seconds"], 6.)
        self.assertEqual(measured["historical_failed_compare"], proof["historical_failed_compare"])
        self.assertFalse(measured["historical_failed_expense_in_forecast_rates"])
        self.assertEqual(measured["physical_calls"], sum(v["completed"] for v in proof["physical_calls"].values()))
        self.assertEqual(measured["qualification"], spec)

    def test_recovery_missing_child_or_changed_import_is_blocked_without_legacy_fallback(self):
        spec, proof, owners = self.recovered_measurement_fixture()
        with patch("haltiq_tau_hotpot.recovery.verify_qualified_pilot", create=True,
                   return_value=(proof, self.checkpoint / "head.safetensors", owners)):
            saved = owners["measurement_sources"].pop("compare_q4")
            with self.assertRaisesRegex(ValueError, "all genuine owners"):
                forecasting.pilot_measurements(spec["path"])
            owners["measurement_sources"]["compare_q4"] = saved
            Path(self.model_inputs[-1]["path"]).write_bytes(b"CHANGED IMPORTED SOURCE")
            with self.assertRaisesRegex(ValueError, "changed"):
                forecasting.pilot_measurements(spec["path"])

    def test_recovery_binding_failure_never_fabricates_one_completed_owner(self):
        spec, _, _ = self.recovered_measurement_fixture()
        with patch("haltiq_tau_hotpot.recovery.verify_qualified_pilot", create=True,
                   side_effect=ValueError("Original failed owner is not reconciled")), \
                patch.object(forecasting, "terminal_owner") as legacy:
            with self.assertRaisesRegex(ValueError, "not reconciled"):
                forecasting.pilot_measurements(spec["path"])
        legacy.assert_not_called()

    def test_pilot_qualification_pin_is_not_shadowed_by_external_inputs(self):
        external = self.model_inputs[-1]
        self.pilot["pinned_inputs_sha256"] = {external["path"]: external["sha256"]}
        self.pilot_spec = self.save("pilot", self.pilot)
        measured = planning.pilot_measurements(self.pilot_spec["path"])
        self.assertEqual(measured["qualification"], self.pilot_spec)
        self.assertEqual(measured["verification_inputs"][0], self.pilot_spec)
        self.assertIn(external, measured["verification_inputs"])

    def test_exact_manual_launcher_is_part_of_the_source_freeze(self):
        plan = planning.read(self.template / "plan-template.json")
        self.assertIn("scripts/start_tau_hotpot_main.ps1", plan["source_paths"])
        self.assertIn("scripts/start_tau_hotpot_main.ps1", self.review["source_sha256"])

    def test_forecast_failure_writes_no_executable_plan_or_n_freeze(self):
        measured = planning.pilot_measurements(self.pilot_spec["path"])
        measured["steady_role_max_seconds"]["writer"] = 1000
        with patch.object(planning, "pilot_measurements", return_value=measured):
            value = self.compile()
        self.assertEqual(value["status"], "blocked_no_launch")
        self.assertFalse((self.root / "compiled/plan.json").exists())
        self.assertFalse((self.root / "compiled/fresh-n-freeze.json").exists())
        forecast = planning.read(self.root / "compiled/throughput-forecast.json")
        self.assertIsNone(forecast["selected_n"])
        self.assertEqual([row["n"] for row in forecast["candidates"]], [20, 40, 60])

    def test_short_context_cannot_be_promoted_to_full_window_capacity(self):
        measured = planning.pilot_measurements(self.pilot_spec["path"])
        for row in measured["native_samples"]:
            row["input_tokens"] = 512
        value = planning.forecast_route(measured)
        self.assertFalse(value["whole_route_fits"])
        self.assertIsNone(value["selected_n"])
        self.assertIn("full-window", value["measurement_blocker"])

    def test_per_question_route_requires_actual_child_timing_and_prices_all_instances(self):
        measured = planning.pilot_measurements(self.pilot_spec["path"])
        blocked = forecasting.forecast_route(measured)
        self.assertFalse(blocked["whole_route_fits"])
        self.assertIn("recovery child", blocked["measurement_blocker"])
        measured.update(per_question_validation_overhead_max_seconds=20., runtime_instances_measured=6)
        valid = forecasting.forecast_route(measured)
        costs = valid["candidates"][0]["components_seconds"]
        self.assertEqual(costs["question_resource_admissions"], 20 * 30)
        self.assertEqual(costs["question_validation_and_bootstrap"], 20 * 40)
        self.assertEqual(costs["outer_model_stage_admissions"], 4 * 30)
        self.assertEqual(costs["runtime_initialization"], 22 * 5.01)

    def test_real_contract_probe_can_supply_missing_context_without_repricing_cold_writer(self):
        proof = self.make_probe()
        technical = forecasting.runtime_probe_measurements(proof["path"], self.pilot_spec["path"])
        measured = planning.pilot_measurements(self.pilot_spec["path"])
        for row in measured["native_samples"]:
            row["input_tokens"] = 512
        value = planning.forecast_route(measured, runtime_probe=technical)
        self.assertTrue(value["native_context_measurement_sufficient"])
        self.assertFalse(value["evidence_cost_is_unmeasured_proxy"])
        first = value["candidates"][0]
        self.assertEqual(first["components_seconds"]["writer"], 820 * 5.01)
        self.assertGreaterEqual(first["components_seconds"]["cold_first_forward_after_each_placement"], 21 * 24 * 2)
        self.assertEqual(len(technical["native_by_criterion"]["evidence"]), 3)

    def test_v2_both_main_review_timings_replace_legacy_proxy_without_quality_selection(self):
        spec = self.make_probe(version=2)
        technical = forecasting.runtime_probe_measurements(spec['path'], self.pilot_spec['path'])
        self.assertEqual(technical['reviewer_main_max_seconds'], 11.)
        self.assertEqual(len(technical['reviewer_ab']), 4)
        self.assertEqual(technical['reviewer_ab'][0]['status'], 'known_terminal_failure')
        measured = planning.pilot_measurements(self.pilot_spec['path'])
        route = forecasting.forecast_route(measured, runtime_probe=technical)
        self.assertFalse(route['reviewer_cost_is_unmeasured_proxy'])
        self.assertEqual(route['main_reviewer_seconds_per_call'], 11.)
        self.assertEqual(route['candidates'][0]['components_seconds']['reviewer'], 180 * 22.)

    def test_v2_main_review_metadata_or_profile_cannot_be_relabelled(self):
        spec = self.make_probe(version=2)
        proof = planning.read(spec['path'])
        for change in ('timing', 'profile'):
            altered = deepcopy(proof)
            if change == 'timing': altered['reviewer_ab'][1]['seconds'] = .001
            else: altered['generation_profile']['reviewer_max_tokens'] = 384
            Path(spec['path']).write_text(json.dumps(altered))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'timings|profile'):
                forecasting.runtime_probe_measurements(spec['path'], self.pilot_spec['path'])

    def test_probe_qualified_for_a_different_pilot_is_rejected(self):
        spec = self.make_probe()
        value = planning.read(spec["path"])
        value["pilot_proof_sha256"] = "f" * 64
        Path(spec["path"]).write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same-model runtime probe"):
            forecasting.runtime_probe_measurements(spec["path"], self.pilot_spec["path"])

    def test_changed_original_probe_configuration_blocks_later_compilation(self):
        spec = self.make_probe()
        value = planning.read(self.runtime_spec["path"])
        value["writer_digest"] = "f" * 64
        Path(self.runtime_spec["path"]).write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            forecasting.runtime_probe_measurements(spec["path"], self.pilot_spec["path"])

    def test_cold_first_forward_is_separate_from_steady_per_call_maximum(self):
        run = Path(self.pilot["run"])
        path = run / "collect/physical/calls.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        next(row for row in records if row["role"] == "writer" and row["event"] == "completed")["elapsed_seconds"] = 24.
        path.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
        state = planning.read(run / "supervisor.json")
        state["stages"]["collect"]["output_sha256"][str(path)] = file_hash(path)
        (run / "supervisor.json").write_text(json.dumps(state), encoding="utf-8")
        self.pilot["supervisor_sha256"] = file_hash(run / "supervisor.json")
        self.pilot["output_sha256"]["collect/physical/calls.jsonl"] = file_hash(path)
        physical = journal(path.parent)
        self.pilot["physical_calls"]["collect"] = {key: value for key, value in physical.items() if key != "responses"}
        self.pilot_spec = self.save("pilot", self.pilot)
        measured = planning.pilot_measurements(self.pilot_spec["path"])
        self.assertEqual(measured["role_max_seconds"]["writer"], 24.)
        self.assertEqual(measured["steady_role_max_seconds"]["writer"], .01)
        self.assertEqual(measured["cold_placement_max_seconds"]["writer"], 24.)
        route = planning.forecast_route(measured)["candidates"][0]
        self.assertLess(route["components_seconds"]["writer"], 820 * 24)
        self.assertGreaterEqual(route["components_seconds"]["cold_first_forward_after_each_placement"], 21 * 24 * 2)

    def test_missing_real_measurement_does_not_manufacture_readiness(self):
        with patch.object(planning, "pilot_measurements", side_effect=ValueError("No completed native timing")):
            value = self.compile()
        self.assertEqual(value["status"], "blocked_no_launch")
        self.assertIn("native timing", value["reason"])
        self.assertFalse((self.root / "compiled/plan.json").exists())

    def test_source_changed_or_readiness_missing_rejects_before_compile(self):
        self.review["source_sha256"]["haltiq_tau_hotpot/planning.py"] = "f" * 64
        self.review_spec = self.save("review", self.review)
        with self.assertRaisesRegex(ValueError, "current source"):
            self.compile()
        self.assertFalse((self.root / "compiled").exists())
        self.review = self.make_review()
        self.review["readiness"].pop("gpu_coordinator")
        self.review_spec = self.save("review", self.review)
        with self.assertRaisesRegex(ValueError, "readiness"):
            self.compile()

    def test_stale_or_nonfinite_resource_receipt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fresh"):
            planning.verify_resources(self.resource_spec["path"], now=self.now + timedelta(minutes=11))
        self.resource["samples"][1]["free_vram_gib"] = 8.99
        self.resource_spec = self.save("resources", self.resource)
        with self.assertRaisesRegex(ValueError, "reserves"):
            self.compile()
        self.assertFalse((self.root / "compiled").exists())

    def test_main_admission_accepts11gib_boundary_and_rejects_old_startup_floor(self):
        for sample in self.resource["samples"]:
            sample["available_ram_gib"] = 11.
        self.resource_spec = self.save("resources", self.resource)
        planning.verify_resources(self.resource_spec["path"], now=self.now)
        self.resource["samples"][0]["available_ram_gib"] = 10.99
        self.resource_spec = self.save("resources", self.resource)
        with self.assertRaisesRegex(ValueError, "reserves"):
            planning.verify_resources(self.resource_spec["path"], now=self.now)

    def test_resource_receipt_from_different_os_lease_is_rejected(self):
        other = deepcopy(self.runtime)
        other["gpu_lease"] = str(self.root / "different-gpu.lock")
        self.resource["config"] = self.save("different-runtime", other)
        self.resource_spec = self.save("resources", self.resource)
        with self.assertRaisesRegex(ValueError, "different local runtime"):
            self.compile()

    def test_wrong_pilot_owner_or_changed_response_is_blocked(self):
        path = Path(self.pilot["run"]) / "features/physical/responses/0.json"
        path.write_text('{"input_tokens":1024}', encoding="utf-8")
        value = self.compile()
        self.assertEqual(value["status"], "blocked_no_launch")
        self.assertIn("changed", value["reason"])

    def test_actual_start_must_not_be_future_or_backdated(self):
        for start in (self.now + timedelta(seconds=1), self.now - timedelta(minutes=6)):
            with self.assertRaisesRegex(ValueError, "truthful actual start"):
                planning.compile_main(self.template, self.root / "compiled", start_time=start.isoformat(),
                    review_receipt=self.review_spec["path"], pilot_proof=self.pilot_spec["path"],
                    resource_receipt=self.resource_spec["path"], now=self.now)

    def test_blocked_compile_cli_is_nonzero_and_cannot_continue_as_success(self):
        with patch.object(planning, "compile_main", return_value={"status": "blocked_no_launch"}):
            code = planning.main(["compile-main", "--template", str(self.template), "--output", str(self.root / "unused"),
                "--start-time", self.now.isoformat(), "--review-receipt", self.review_spec["path"],
                "--pilot-proof", self.pilot_spec["path"], "--resource-receipt", self.resource_spec["path"]])
        self.assertEqual(code, 2)

    def test_resource_check_is_one_bounded_attempt_with_released_locks(self):
        clock, sleeps, held, enters = [self.now], [], set(), []
        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += timedelta(seconds=seconds)
        class Lease:
            def __init__(_, path, owner): _.path = str(path)
            def __enter__(_):
                self.assertNotIn(_.path, held)
                held.add(_.path)
                enters.append(_.path)
            def __exit__(_, *args): held.remove(_.path)
        def sampler():
            self.assertEqual(len(held), 2)
            return {"available_ram_bytes": 11.1 * 1024**3, "gpu_free_mib": 9.2 * 1024, "gpu_temperature_c": 40}
        output = self.root / "sampled-resources.json"
        with patch("subprocess.Popen") as spawn, patch.object(planning.socket, "socket") as socket:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            result = planning.sample_resources(self.runtime_spec["path"], output, sampler=sampler,
                sleep=sleep, now=lambda: clock[0], lease_factory=Lease)
        spawn.assert_not_called()
        self.assertEqual(result["status"], "PASS_RESOURCE_ADMISSION")
        self.assertEqual(sleeps, [10, 10, 10])
        self.assertEqual(len(result["samples"]), 4)
        self.assertEqual(len(enters), 4)
        self.assertFalse(held)
        planning.verify_resources(output, now=clock[0])
        with self.assertRaises(ValueError):
            planning.sample_resources(self.runtime_spec["path"], output, sampler=sampler, lease_factory=Lease)

    def test_busy_lease_is_a_terminal_resource_failure_without_retry(self):
        attempts = []
        class Busy:
            def __init__(_, path, owner): attempts.append(str(path))
            def __enter__(_): raise RuntimeError("Study lease already owned")
            def __exit__(_, *args): pass
        with patch("time.sleep") as sleep, patch.object(planning.socket, "socket") as socket:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            result = planning.sample_resources(self.runtime_spec["path"], self.root / "busy.json", lease_factory=Busy)
        self.assertEqual(result["status"], "BLOCKED_RESOURCE_ADMISSION")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(result["samples"], [])
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
