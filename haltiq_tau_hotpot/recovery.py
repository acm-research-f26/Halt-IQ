"""Explicit immutable preparation imports and separate comparison ownership.

An interrupted pilot remains interrupted. Recovery imports its three genuinely
completed preparation phases, runs every original heldout question in a fresh
process, and qualifies the combined lineage only after the new owner exits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from . import qualification as q
from .journal import CallJournal, durable_write, fingerprint

PHASES = tuple("compare_q" + str(i) for i in range(1, 5))
ARMS = ["fixed1", "fixed3", "released", "trained"]
PREP = ("collect", "features", "train")


def _external_output(output, *roots):
    output = Path(output).resolve()
    q.require(not output.exists(), "Evidence output must be a NEW file")
    for root in roots + (Path(__file__).resolve().parents[1],):
        root = Path(root).resolve()
        q.require(output != root and root not in output.parents, "Write evidence outside original/current run and source")
    return output


def spec(path):
    path = Path(path).resolve()
    q.require(path.is_file() and not path.is_symlink(), "Existing real receipt required")
    return {"path": str(path), "sha256": q.sha(path)}


def checked(value):
    from haltiq_supervisor.plan import satisfied
    q.require(isinstance(value, dict) and Path(value["path"]).is_absolute()
              and value.get("sha256") and satisfied(value), "Imported evidence is missing or changed")
    return value


def _owner(run, *, purpose, phases, terminal_status):
    from haltiq_supervisor.engine import verify_completion
    run = Path(run).resolve()
    plan, state = q.read(run / "plan.json"), q.read(run / "supervisor.json")
    q.require(plan.get("purpose") == purpose and state.get("status") == terminal_status
        and state.get("plan_sha256") == q.sha(run / "plan.json"), "Wrong actual source/terminal owner")
    source = {p.relative_to(run / "source").as_posix(): q.sha(p)
              for p in (run / "source").rglob("*") if p.is_file()}
    q.require(source == state.get("source_sha256") == plan.get("reviewed_source_sha256")
        and q.sha(plan["python"]) == state.get("python_sha256"), "Original source/interpreter changed")
    for value in plan.get("inputs", []) + [s for group in plan.get("readiness", {}).values() for s in group]:
        checked(value)
    for phase in phases:
        stage = next(s for s in plan["stages"] if s["id"] == phase)
        owner = state["stages"][phase]
        expected = {str(q.within(s["path"].replace("{run}", str(run)), run)) for s in stage["outputs"]}
        q.require(owner.get("status") == "completed" and owner.get("exit_code") == 0
            and owner.get("owned_tree_clear") is True and owner.get("os_handoff_verified") is True
            and set(owner.get("output_sha256", {})) == expected, "A genuinely completed imported phase is required")
        for path, sha in owner["output_sha256"].items():
            q.require(q.sha(q.within(path, run)) == sha, "An original completed output changed")
        completion = run / phase / ("compare/completion.json" if purpose == "qualification_recovery" else "completion.json")
        verify_completion(stage, state, q.read(completion))
    return plan, state


def _preparation_owner(run, reconciliation):
    run, reconciliation = Path(run).resolve(), Path(reconciliation).resolve()
    receipt = q.read(reconciliation)
    plan, state = _owner(run, purpose="qualification_pilot", phases=PREP, terminal_status="failed")
    q.require(receipt.get("schema") == "haltiq-pilot-terminal-reconciliation-v1"
        and receipt.get("status") == "PASS_TERMINAL_RECONCILIATION"
        and all(receipt.get(k) is True for k in ("terminal", "runtime_clear", "all_dispatched_requests_reconciled"))
        and Path(receipt["pilot_run"]).resolve() == run and receipt["plan_sha256"] == state["plan_sha256"]
        and receipt["supervisor_sha256"] == q.sha(run / "supervisor.json")
        and receipt["source_manifest_sha256"] == fingerprint(state["source_sha256"])
        and receipt.get("original_run_status") == "failed" and receipt.get("combined_pilot_qualified") is False,
        "A truthful terminal reconciliation of the failed owner is required")
    q.require(state["stages"]["compare"].get("exit_code") == 1
        and not (run / "compare/completion.json").exists(), "Never relabel a failed comparison as completed")
    for phase, count in zip(PREP, (64, 64, 1)):
        imported = receipt["completed_import_stages"][phase]
        checked(imported["completion"])
        q.require(imported["outputs"] == state["stages"][phase]["output_sha256"]
            and imported["completed_units"] == count and imported["failed_units"] == imported["not_run_units"] == 0,
            "Imported preparation counts/output owners differ")
    for path, sha in receipt["raw_output_sha256"].items():
        q.require(q.sha(q.within(path, run)) == sha, "Reconciled original raw evidence changed")
    for phase in PREP + ("compare",):
        actual = CallJournal(run / phase / "physical").verify()
        recorded = receipt["physical_calls"][phase]
        q.require(not actual["pending_or_uncertain"] and actual["completed"] == recorded["completed"]
            and actual["dispatched"] == recorded["dispatched"] and actual["by_role"] == recorded["by_role"]
            and q.sha(run / phase / "physical/calls.jsonl") == recorded["journal_sha256"],
            "Original physical requests do not reconcile")
    terminal = q.read(run / "compare/runtime-terminal.json")
    q.require(terminal["cleanup"]["verified"] and terminal["cleanup"]["private_port_closed"]
        and terminal["cleanup"]["gpu_lease_released"] and not terminal["calls"]["pending_or_uncertain"],
        "Failed comparison runtime must already be clean")
    return plan, state, receipt


def _preparation_pins(run, plan, state, reconciliation, reconciled):
    """Reconstruct the complete real dependency graph, never trust a subset."""
    run, reconciliation = Path(run).resolve(), Path(reconciliation).resolve()
    pins = {str(run / "plan.json"): q.sha(run / "plan.json"), str(run / "supervisor.json"): q.sha(run / "supervisor.json"),
            str(reconciliation): q.sha(reconciliation), str(Path(plan["python"]).resolve()): q.sha(plan["python"])}
    pins.update({str(run / "source" / key): sha for key, sha in state["source_sha256"].items()})
    pins.update({s["path"]: s["sha256"] for s in plan["inputs"] + [v for group in plan["readiness"].values() for v in group]})
    pins.update(reconciled["raw_output_sha256"])
    pins.update({str(p): q.sha(p) for phase in PREP + ("compare",) for p in (run / phase).rglob("*") if p.is_file()})
    return pins


def import_preparation(run, reconciliation, output):
    """Audit existing model evidence without updates, rescoring or inference."""
    run = Path(run).resolve()
    output = _external_output(output, run)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    plan, state, reconciled = _preparation_owner(run, reconciliation)
    with q.frozen_helpers(run) as helpers:
        old_protocol = helpers["haltiq_supervisor.plan"]
        old_protocol.verify_frozen(plan, state, run)
        old_protocol.qualification(plan)
        config = q.read(plan["qualification_protocol"]["path"])
        freeze, cohorts = helpers["haltiq_tau_hotpot.worker"].qualification_protocol(config)
        q.require(freeze["pilot"]["heldout"] == 4 and freeze["pilot"]["training"] == {"fit": 40, "tune": 10, "calibration": 10},
                  "Only the original four-question/sixty-training qualification may be recovered")
        physical = {phase: q.journal(run / phase / "physical") for phase in PREP}
        collected = q.read(run / "collect/episodes.json")
        cases = cohorts["pilot_heldout"] + cohorts["pilot_fit"] + cohorts["pilot_tune"] + cohorts["pilot_calibration"]
        q.require([r["question_id"] for r in collected] == [c["public"]["id"] for c in cases], "Original initial cohort/order changed")
        for row, case in zip(collected, cases):
            q.require(row == q.read(run / "collect/episodes" / (row["question_id"] + ".json")), "Original initial row changed")
            q.native_episode(row["episode"], case, row["initial_retrieval"]["indices"], helpers, arm="fixed1", max_drafts=1)
            q.require(row["episode"]["status"] == "completed", "Every original first draft must be real and valid")
        rows, baseline = q.read(run / "features/features.json"), q.read(run / "features/untouched-baseline.json")
        q.feature_rows(run, cohorts, rows, baseline, helpers)
        learning = q.learning_proof(run, config, cohorts, rows, baseline, physical, helpers)
        binding = q.model_binding(config, learning)
    pins = _preparation_pins(run, plan, state, reconciliation, reconciled)
    value = {"schema": "haltiq-reusable-pilot-preparation-v1", "status": "PASS_REUSABLE_PREPARATION",
        "source_run": str(run), "source_plan_sha256": state["plan_sha256"], "source_supervisor_sha256": q.sha(run / "supervisor.json"),
        "source_sha256": state["source_sha256"], "source_overall_status": "failed",
        "completed_import_stages": reconciled["completed_import_stages"], "reconciliation": spec(reconciliation),
        "original_config": plan["qualification_protocol"], "model_binding": binding,
        "trained_head": spec(run / "train/learned/answer-head.safetensors"),
        "baseline": spec(run / "features/untouched-baseline.json"), "learning": spec(run / "train/learning.json"),
        "question_ids": [c["public"]["id"] for c in cohorts["pilot_heldout"]], "arms": ARMS,
        "max_drafts": 3, "threshold": .8, "input_sha256": pins,
        "historical_failed_compare": {"run": str(run), "phase": "compare", "physical": reconciled["physical_calls"]["compare"],
            "failure": reconciled["failed_compare"]["failure"], "promoted_to_recovery_episodes": False},
        "new_model_forwards": 0, "new_head_scores": 0, "new_learning": False, "combined_pilot_qualified": False}
    durable_write(output, value)
    return value


def preparation_binding(proof):
    value = q.read(proof) if not isinstance(proof, dict) else proof
    q.require(value.get("schema") == "haltiq-reusable-pilot-preparation-v1"
        and value.get("status") == "PASS_REUSABLE_PREPARATION" and value.get("source_overall_status") == "failed"
        and value.get("combined_pilot_qualified") is False and value.get("arms") == ARMS
        and value.get("max_drafts") == 3 and len(value.get("question_ids", [])) == 4,
        "Explicit bounded preparation import required")
    for path, sha in value["input_sha256"].items():
        checked({"path": path, "sha256": sha})
    run = Path(value["source_run"])
    checked(value["reconciliation"])
    plan, state, reconciled = _preparation_owner(run, value["reconciliation"]["path"])
    q.require(state["plan_sha256"] == value["source_plan_sha256"]
        and q.sha(run / "supervisor.json") == value["source_supervisor_sha256"]
        and state["source_sha256"] == value["source_sha256"]
        and value["input_sha256"] == _preparation_pins(run, plan, state, value["reconciliation"]["path"], reconciled),
        "Imported preparation owner/dependency union changed")
    config = q.read(checked(value["original_config"])["path"])
    from .worker import qualification_protocol
    freeze, cohorts = qualification_protocol(config)
    q.require(value["question_ids"] == [c["public"]["id"] for c in cohorts["pilot_heldout"]], "Never filter or replace recovery questions")
    for key, relative in (("trained_head", "train/learned/answer-head.safetensors"),
                          ("baseline", "features/untouched-baseline.json"), ("learning", "train/learning.json")):
        checked(value[key])
        q.require(Path(value[key]["path"]).resolve() == run / relative, "Imported artifact must retain its actual original path")
    learning = q.read(value["learning"]["path"])
    q.require(value["model_binding"] == q.model_binding(config, learning)
        and value["trained_head"]["sha256"] == learning["criteria"]["answer"]["trained_head_sha256"]
        and value["completed_import_stages"] == q.read(value["reconciliation"]["path"])["completed_import_stages"],
        "Imported model/training lineage metadata changed")
    return value, config, freeze, cohorts


def validate_plan(plan):
    """Supervisor extension: four original comparison units, no training branch."""
    from haltiq_supervisor.plan import _pinned_json
    config = _pinned_json(plan.get("recovery_protocol"), "Recovery config")
    proof = _pinned_json(config.get("preparation_proof"), "Preparation import")
    original = _pinned_json(proof.get("original_config"), "Original preparation config")
    from .planning import RUNTIME_KEYS
    q.require(config.get("purpose") == "qualification_recovery" and config.get("new_learning") is False
        and config.get("native_feature_cache") is True and config.get("response_cache") is True
        and config.get("arms") == ARMS and config.get("max_drafts") == 3 and config.get("threshold") == .8
        and config.get("question_ids") == proof.get("question_ids") and len(config.get("question_ids", [])) == 4
        and proof.get("schema") == "haltiq-reusable-pilot-preparation-v1"
        and proof.get("status") == "PASS_REUSABLE_PREPARATION" and proof.get("source_overall_status") == "failed",
        "Recovery must import the original four questions with unchanged three-draft methods")
    old_plan = q.read(Path(proof["source_run"]) / "plan.json")
    q.require(all(config.get(key) == original.get(key) for key in RUNTIME_KEYS)
        and plan["gpu_lease"] == original["gpu_lease"] and plan["study_lease"] == original["study_lease"]
        and Path(plan["python"]).resolve() == Path(old_plan["python"]).resolve(),
        "Recovery must preserve original models, environment, runtime inputs and OS leases")
    q.require(plan["deadline"] == old_plan["deadline"] and plan["hard_deadline"] == old_plan["hard_deadline"]
        and tuple(s["id"] for s in plan["stages"]) == PHASES and plan["resources"]["startup_ram_gib"] >= 11,
        "Keep the original cutoffs and stronger11GiB per-question admission")
    pins = {(str(Path(s["path"]).resolve()), s.get("sha256")) for s in plan["inputs"]}
    required = [plan["recovery_protocol"], config["preparation_proof"]] + [
        {"path": path, "sha256": sha} for path, sha in proof["input_sha256"].items()]
    q.require(all((str(Path(s["path"]).resolve()), s["sha256"]) in pins for s in required), "Every original import must stay pinned")
    for phase, stage in zip(PHASES, plan["stages"]):
        argv = [plan["python"], "-B", "-m", "haltiq_tau_hotpot.recovery_worker", phase, "--config", plan["recovery_protocol"]["path"],
            "--output", "{run}/" + phase, "--previous", "{run}", "--deadline", "{deadline}", "--control", "{control}", "--allow-model-calls"]
        q.require(stage.get("argv") == argv and stage.get("planned_units") == 4 and stage.get("model_work") is True
            and stage.get("gpu") is True and stage.get("completion_receipt") == "{run}/" + phase + "/compare/completion.json",
            "One separately owned subprocess must execute exactly four arms for each question")
    return config


def prepare_recovery(preparation_proof, source_root, python, output):
    """Write a blocked plan/config only. Independent review/adoption is separate."""
    from haltiq_supervisor.plan import PACKAGES
    from .planning import RESOURCES, RUNTIME_KEYS
    proof, original, _, _ = preparation_binding(preparation_proof)
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    q.require(output != source_root and source_root not in output.parents and output not in source_root.parents,
              "Use a new external preparation directory")
    original_run = Path(proof["source_run"]).resolve()
    q.require(output != original_run and original_run not in output.parents and output not in original_run.parents,
              "Preparation must not add files to or surround the immutable failed source run")
    output.mkdir(parents=True, exist_ok=False)
    config = {key: deepcopy(original[key]) for key in RUNTIME_KEYS}
    config.update(purpose="qualification_recovery", preparation_proof=spec(preparation_proof),
        question_ids=proof["question_ids"], arms=ARMS, max_drafts=3, threshold=.8, new_learning=False,
        native_feature_cache=True, response_cache=True, hosted_services=False, long_study_started=False)
    durable_write(output / "recovery-config.json", config)
    old = q.read(Path(proof["source_run"]) / "plan.json")
    resources = {**RESOURCES, "startup_ram_gib": 11}
    stages = []
    for phase in PHASES:
        base = "{run}/" + phase
        stages.append({"id": phase, "model_work": True, "gpu": True, "planned_units": 4, "check_handoff": True,
            "argv": [str(Path(python).resolve()), "-B", "-m", "haltiq_tau_hotpot.recovery_worker", phase,
                "--config", str(output / "recovery-config.json"), "--output", base, "--previous", "{run}",
                "--deadline", "{deadline}", "--control", "{control}", "--allow-model-calls"],
            "completion_receipt": base + "/compare/completion.json",
            "outputs": [{"path": base + "/" + name} for name in ("compare/completion.json", "compare/episodes.json",
                "compare/runtime-terminal.json", "compare/physical/calls.jsonl", "phase-timing.json")]})
    inputs = [spec(output / "recovery-config.json"), spec(preparation_proof)] + [
        {"path": path, "sha256": sha} for path, sha in proof["input_sha256"].items()]
    plan = {"version": 1, "scope": "custom_tau2_hotpot", "purpose": "qualification_recovery",
        "status": "blocked_preparation_only", "source_root": str(source_root),
        "source_paths": list(PACKAGES) + ["pyproject.toml"], "python": str(Path(python).resolve()),
        "deadline": old["deadline"], "hard_deadline": old["hard_deadline"], "stop_grace_seconds": 60,
        "private_port": 11435, "gpu_lease": original["gpu_lease"], "study_lease": original["study_lease"],
        "resources": resources, "codex_argv": None, "recovery_protocol": spec(output / "recovery-config.json"),
        "readiness": {}, "reviewed_source_sha256": {}, "inputs": inputs, "stages": stages,
        "physical_envelope": {"question_processes": 4, "writer_placements_per_question_max": 1,
            "native_placements_per_question_max": 1, "minimum_admission_seconds_per_question": 30,
            "cold_forward_per_placement": True, "all_original_failed_expense_retained": True}}
    durable_write(output / "recovery-plan-template.json", plan)
    return plan


def _recovery_evidence(child, config, learning, preparation, case, initial_feature, first_draft):
    """Extend the reviewed fresh auditor only for genuine pretraining features."""
    from .main_audit import Evidence
    class RecoveryEvidence(Evidence):
        def __init__(self):
            self.run, self.config, self.learning = child, config, learning
            self.journal = CallJournal(child / "compare/physical")
            self.summary = self.journal.verify()
            q.require(not self.summary["pending_or_uncertain"], "Recovery call remains unresolved")
            self.pending = {r["id"]: r for r in self.journal.records if r["event"] == "pending"}
            self.completed = {r["id"]: r for r in self.journal.records if r["event"] == "completed"}
            self.values = {key: q.read(self.journal.directory / event["response"]) for key, event in self.completed.items()}
            self.generations = [r for r in self.journal.records if r["event"] in {"completed", "exact_response_reuse"}
                                and r["role"] in {"writer", "reviewer"}]
            self.gates = [r for r in self.journal.records if r["event"] == "completed"
                          and r["role"] in {"cpu_head", "native_answer", "cached_head_answer"}]
            self.generation_offset = self.gate_offset = 0
            self.used_retrievals, self.used_native, self.feature_checks, self.labels = set(), set(), set(), []
            self.logical, self.known_pre_dispatch_failures = Counter(), Counter()
            self.released_sha = preparation["model_binding"]["released_head_sha256"]
            self.trained_path = Path(preparation["trained_head"]["path"])
            self.trained_sha = preparation["trained_head"]["sha256"]

        def gate(self, public, draft, event, *, arm, gold):
            if event["status"] == "completed" and self.gate_offset < len(self.gates) and self.gates[self.gate_offset]["role"] == "cpu_head":
                call = self.gates[self.gate_offset]
                self.gate_offset += 1
                value = self.values[call["id"]]
                temperature = learning["criteria"]["answer"][("trained" if arm == "trained" else "released") + "_calibration"]["temperature"]
                feature = initial_feature["observed"]["feature"]
                expected = {"feature": feature["provenance_sha256"], "head": arm, "temperature": temperature}
                q.require(event["criterion"] == "answer" and draft == first_draft and value == event["score"]
                    and value["temperature"] == temperature and arm in {"released", "trained"}
                    and self.pending[call["id"]]["request_sha256"] == fingerprint(expected)
                    and q.sha(feature["path"]) == feature["sha256"], "Shared original feature/head/temperature changed")
                q.probability(value, native=False)
                self.logical["gate_answer"] += 1
                return
            return super().gate(public, draft, event, arm=arm, gold=gold)

        def finish(self):
            q.require(self.generation_offset == len(self.generations) and self.gate_offset == len(self.gates),
                      "Recovery physical/reuse work was not consumed by its episodes")
            q.require({k for k, r in self.completed.items() if r["role"] == "retrieval"} == self.used_retrievals,
                      "Unused recovery retrieval")
            allowed = {"retrieval", "writer", "reviewer", "writer_load", "writer_unload", "native_startup",
                       "native_answer", "cached_head_answer", "cpu_head"}
            q.require(set(self.summary["by_role"]) <= allowed, "Unexpected recovery model work")
            for key, event in self.completed.items():
                if event["role"] in {"writer_load", "writer_unload"}:
                    q.require(self.pending[key]["request_sha256"] == fingerprint({"model": "qwen3:8b"}), "Wrong writer identity")
                if event["role"] == "native_startup":
                    metadata = self.values[key]["metadata"]
                    q.require(self.pending[key]["request_sha256"] == fingerprint({"checkpoint": config["checkpoint"]})
                        and metadata["released_head_sha256"] == self.released_sha
                        and metadata["encoder_sha256"] == learning["encoder_sha256"]
                        and metadata["head_parameters"] == 1053184, "Wrong native placement/model")
            from .recovery_worker import assert_envelope
            assert_envelope(SimpleNamespace(journal=self.journal))
    return RecoveryEvidence()


def _replay(row, case, evidence, first, helpers):
    """Replay real native tools while joining exact callback request/response bytes."""
    from .main_audit import AuditIntegrityError
    queue = iter(e for e in row["trace"] if e["event"] in {"draft", "review", "gate"})
    def event(kind):
        result = next(queue)
        q.require(result["event"] == kind, "Recovery callback order changed")
        return result
    real_agent = helpers["haltiq_tau_hotpot.agent"]
    def traced(case_arg, writer, reviewer, gate, **kwargs):
        def write(state, feedback):
            try:
                current = event("draft")
                if state["previous_draft"] is None:
                    q.require(current.get("draft") == first, "Original first draft changed")
                else:
                    evidence.generation(state, feedback, current, case["public"], review=False)
            except Exception as error:
                raise AuditIntegrityError(str(error)) from error
            return writer(state, feedback)
        def review(state, draft):
            try:
                current = event("review")
                indices = evidence.rank(case["public"], draft["answer"])
                view = {"question": case["public"]["question"], "evidence": [{"title": case["public"]["context"][i][0],
                    "sentences": [{"index": n, "text": text} for n, text in enumerate(case["public"]["context"][i][1])]}
                    for i in indices], "draft": draft}
                evidence.generation(view, "", current, case["public"], review=True)
            except Exception as error:
                raise AuditIntegrityError(str(error)) from error
            return reviewer(state, draft)
        def decide(state, draft, criterion):
            try:
                current = event("gate")
                q.require(current["criterion"] == criterion, "Wrong recovered criterion")
                evidence.gate(case["public"], draft, current, arm=row["arm"], gold=case["evaluator"])
            except Exception as error:
                raise AuditIntegrityError(str(error)) from error
            return gate(state, draft, criterion)
        return real_agent.run_episode(case_arg, write, review, decide, **kwargs)
    proxy = {**helpers, "haltiq_tau_hotpot.agent": SimpleNamespace(run_episode=traced)}
    q.native_episode(row, case, row["passage_indices"], proxy, arm=row["arm"], max_drafts=3)
    q.require(next(queue, None) is None, "Extra recovery callback")


def _measurement(run, phase, *, recovery=False):
    owner = q.read(run / "supervisor.json")
    physical = run / phase / ("compare/physical" if recovery else "physical")
    timing_path = run / phase / "phase-timing.json"
    timing = q.read(timing_path)
    q.require(all(type(timing.get(key)) in (int, float) and math.isfinite(timing[key]) and timing[key] >= 0 for key in
        ("wall_seconds", "active_seconds", "intentional_idle_seconds"))
        and timing["active_seconds"] + timing["intentional_idle_seconds"] <= timing["wall_seconds"] + 2,
        "Invalid nonoverlapping phase timing")
    timing["receipt_sha256"] = q.sha(timing_path)
    stage = owner["stages"][phase]
    timing["supervisor_stage_wall_seconds"] = (q.aware(stage["finished_at"]) - q.aware(stage["started_at"])).total_seconds()
    q.require(math.isfinite(timing["supervisor_stage_wall_seconds"]) and timing["supervisor_stage_wall_seconds"] >= 0
        and timing["wall_seconds"] <= timing["supervisor_stage_wall_seconds"] + 2, "Phase wall escapes its actual owner")
    runtime_path = run / phase / ("compare/runtime-terminal.json" if recovery else "runtime-terminal.json")
    if phase != "train":
        terminal = q.read(runtime_path)
        timing["runtime_startup_and_switch"] = terminal["timings"]
        q.require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in terminal["timings"].values()),
                  "Invalid runtime placement timing")
    journal = CallJournal(physical)
    for event in journal.records:
        q.require(q.aware(stage["started_at"]) <= q.aware(event["at"]) <= q.aware(stage["finished_at"]),
                  "Physical event escapes its actual question owner")
        if event["event"] == "completed":
            q.require(type(event["elapsed_seconds"]) in (int, float) and math.isfinite(event["elapsed_seconds"])
                and 0 <= event["elapsed_seconds"] <= timing["supervisor_stage_wall_seconds"] + 2,
                "Invalid physical duration")
    source = {"run": str(run), "phase": phase, "physical_dir": str(physical), "timing_path": str(timing_path),
        "runtime_terminal_path": str(runtime_path) if phase != "train" else None,
        "owner_plan_sha256": owner["plan_sha256"], "owner_supervisor_sha256": q.sha(run / "supervisor.json")}
    return source, journal.verify(), timing


def verify_qualified_pilot(proof, config=None):
    """Recheck actual two-owner lineage without models, numerical heads or OS locks."""
    value = q.read(proof) if not isinstance(proof, dict) else proof
    q.require(value.get("schema") == "haltiq-tau-hotpot-qualified-pilot-v2"
        and value.get("status") == "PASS_QUALIFIED_ENGINEERING_PILOT" and value.get("planned_units") == [64, 64, 1, 16]
        and all(value.get(k) is True for k in ("qualified", "terminal", "cleanup_verified", "actual_os_handoff_verified"))
        and value.get("pending_or_uncertain") is False, "Actual two-owner qualification required")
    preparation = value["preparation_owner"]
    imported, original, _, _ = preparation_binding(checked(preparation["proof"])["path"])
    run = Path(value["recovery_run"])
    plan, state = _owner(run, purpose="qualification_recovery", phases=PHASES, terminal_status="completed")
    recovery_config = validate_plan(plan)
    q.require(value["run"] == str(run) and value["run_id"] == state["run_id"]
        and value["plan_sha256"] == state["plan_sha256"] and value["supervisor_sha256"] == q.sha(run / "supervisor.json")
        and value["source_sha256"] == state["source_sha256"] and value["model_binding"] == imported["model_binding"]
        and value["trained_head_path"] == imported["trained_head"]["path"]
        and value["trained_head_sha256"] == imported["trained_head"]["sha256"]
        and value["untouched_baseline_sha256"] == imported["baseline"]["sha256"], "Qualified lineage/model binding changed")
    q.require(preparation["proof"] == recovery_config["preparation_proof"], "Recovery owner imports a different preparation proof")
    expected_preparation = {"run": imported["source_run"], "proof": recovery_config["preparation_proof"],
        "plan_sha256": imported["source_plan_sha256"], "supervisor_sha256": imported["source_supervisor_sha256"],
        "source_sha256": imported["source_sha256"], "completed_import_stages": imported["completed_import_stages"],
        "baseline_path": imported["baseline"]["path"], "learning_path": imported["learning"]["path"]}
    q.require(preparation == expected_preparation and value["historical_failed_compare"] == imported["historical_failed_compare"],
              "Qualified preparation/history metadata differs from its actual failed owner")
    expected_pins = dict(imported["input_sha256"])
    expected_pins[preparation["proof"]["path"]] = preparation["proof"]["sha256"]
    expected_pins.update({s["path"]: s["sha256"] for s in plan["inputs"] + [v for group in plan["readiness"].values() for v in group]})
    expected_pins.update({str(p): q.sha(p) for p in run.rglob("*") if p.is_file() and p.name != "supervisor.lock"})
    q.require(value["pinned_inputs_sha256"] == expected_pins, "Qualified lineage omitted or changed an actual dependency")
    for path, sha in value["pinned_inputs_sha256"].items():
        checked({"path": path, "sha256": sha})
    current_outputs = {str(p.relative_to(run)): q.sha(p) for phase in PHASES for p in (run / phase).rglob("*") if p.is_file()}
    q.require(value["output_sha256"] == current_outputs, "Qualified recovery output map changed")
    for phase in PREP + PHASES:
        measurement, physical, timing = _measurement(Path(imported["source_run"]) if phase in PREP else run, phase, recovery=phase in PHASES)
        q.require(value["measurement_sources"][phase] == measurement and value["physical_calls"][phase] == physical
            and value["timing"]["phases"][phase] == timing, "Qualified measurement metadata differs from original receipts")
    if config is not None:
        for key in ("checkpoint", "base", "writer_digest", "tau_revision"):
            q.require(config[key] == original[key], "Downstream model/config differs from qualified pilot")
    return value, Path(value["trained_head_path"]), {"preparation_run": imported["source_run"], "recovery_run": str(run),
        "measurement_sources": value["measurement_sources"], "verification_inputs": [
            {"path": path, "sha256": sha} for path, sha in value["pinned_inputs_sha256"].items()]}


def qualify_recovery(run, output):
    """External terminal-only composition; never invoked by an active supervisor."""
    run = Path(run).resolve()
    plan, state = _owner(run, purpose="qualification_recovery", phases=PHASES, terminal_status="completed")
    config = validate_plan(plan)
    preparation, _, _, cohorts = preparation_binding(checked(config["preparation_proof"])["path"])
    source = Path(preparation["source_run"])
    output = _external_output(output, run, source)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    learning = q.read(preparation["learning"]["path"])
    collected = {r["question_id"]: r for r in q.read(source / "collect/episodes.json")}
    features = {r["question_id"]: r for r in q.read(source / "features/features.json")}
    episodes, initial, audits = [], {}, {}
    with q.frozen_helpers(run) as helpers:
        for phase, case in zip(PHASES, cohorts["pilot_heldout"]):
            child, qid = run / phase, case["public"]["id"]
            rows = q.read(child / "compare/episodes.json")
            first = collected[qid]["episode"]["submitted"]
            initial[qid] = first
            q.require([(r["question_id"], r["arm"]) for r in rows] == [(qid, arm) for arm in ARMS], "All original question/arm pairs are required")
            evidence = _recovery_evidence(child, config, learning, preparation, case, features[qid], first)
            for row in rows:
                q.require(row == q.read(child / "compare/episodes" / (qid + "-" + row["arm"] + ".json"))
                    and row["shared_initial_draft_sha256"] == fingerprint(first) and row["shared_initial_physical_calls"] == 0
                    and row["passage_indices"] == collected[qid]["initial_retrieval"]["indices"]
                    and row["preparation_proof_sha256"] == config["preparation_proof"]["sha256"], "Recovery row/source differs")
                _replay(row, case, evidence, first, helpers)
            evidence.finish()
            completion = q.read(child / "compare/completion.json")
            actual_completed = sum(row["status"] == "completed" for row in rows)
            q.require(completion["completed_units"] == actual_completed and completion["failed_units"] == 4 - actual_completed
                and completion["not_run_units"] == 0, "Recovery completion differs from actual native outcomes")
            terminal = q.read(child / "compare/runtime-terminal.json")
            q.require(terminal["cleanup"]["verified"] and terminal["calls"] == evidence.summary, "Recovery cleanup/journal mismatch")
            episodes.extend(rows)
            audits[phase] = {"physical": evidence.summary, "logical": dict(evidence.logical), "native_replays": 4}
        from .comparison import analyse
        comparison = analyse(episodes, cohorts["pilot_heldout"], ARMS, initial_drafts=initial)
    measurements, physical, timings = {}, {}, {}
    for phase in PREP + PHASES:
        measurements[phase], physical[phase], timings[phase] = _measurement(source if phase in PREP else run, phase, recovery=phase in PHASES)
    pins = dict(preparation["input_sha256"])
    pins[config["preparation_proof"]["path"]] = config["preparation_proof"]["sha256"]
    pins.update({s["path"]: s["sha256"] for s in plan["inputs"] + [v for group in plan["readiness"].values() for v in group]})
    pins.update({str(p): q.sha(p) for p in run.rglob("*") if p.is_file() and p.name != "supervisor.lock"})
    from haltiq_supervisor.engine import check_handoff
    from pc_strands.control import GpuLease
    check_handoff(plan)
    with GpuLease(Path(plan["study_lease"]), "external recovery terminal qualification"):
        with GpuLease(run / "supervisor.lock", "external recovery supervisor terminal proof"):
            pass
    # Recheck outputs and source after replay. No receipt can qualify an active owner.
    _owner(run, purpose="qualification_recovery", phases=PHASES, terminal_status="completed")
    value = {"schema": "haltiq-tau-hotpot-qualified-pilot-v2", "status": "PASS_QUALIFIED_ENGINEERING_PILOT",
        "qualified": True, "terminal": True, "cleanup_verified": True, "pending_or_uncertain": False,
        "actual_os_handoff_verified": True, "ready": {"combined_pilot": True}, "planned_units": [64, 64, 1, 16],
        "run": str(run), "recovery_run": str(run), "run_id": state["run_id"], "plan_sha256": state["plan_sha256"],
        "supervisor_sha256": q.sha(run / "supervisor.json"), "source_sha256": state["source_sha256"],
        "preparation_owner": {"run": str(source), "proof": config["preparation_proof"],
            "plan_sha256": preparation["source_plan_sha256"], "supervisor_sha256": preparation["source_supervisor_sha256"],
            "source_sha256": preparation["source_sha256"], "completed_import_stages": preparation["completed_import_stages"],
            "baseline_path": preparation["baseline"]["path"], "learning_path": preparation["learning"]["path"]},
        "model_binding": preparation["model_binding"], "trained_head_path": preparation["trained_head"]["path"],
        "trained_head_sha256": preparation["trained_head"]["sha256"], "untouched_baseline_sha256": preparation["baseline"]["sha256"],
        "measurement_sources": measurements, "physical_calls": physical, "timing": {"phases": timings,
            "scope": "Separate owner/phase walls contain calls, startup and idle; overlapping scopes are never summed"},
        "historical_failed_compare": preparation["historical_failed_compare"], "pinned_inputs_sha256": pins,
        "output_sha256": {str(p.relative_to(run)): q.sha(p) for phase in PHASES for p in (run / phase).rglob("*") if p.is_file()},
        "comparison": comparison, "audits": audits, "heldout_questions": 4,
        "new_model_forwards": 0, "new_head_scores": 0, "new_learning": False, "long_study_started_by_audit": False,
        "reviewed_at": datetime.now(timezone.utc).isoformat(), "scope": "Recovered engineering pilot; failed original owner remains failed, no population gain claim"}
    verify_qualified_pilot(value)
    durable_write(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    imported = commands.add_parser("import-preparation")
    for key in ("run", "reconciliation", "output"):
        imported.add_argument("--" + key, required=True, type=Path)
    prepare = commands.add_parser("prepare")
    for key in ("preparation-proof", "source-root", "python", "output"):
        prepare.add_argument("--" + key, required=True, type=Path)
    qualify = commands.add_parser("qualify")
    for key in ("run", "output"):
        qualify.add_argument("--" + key, required=True, type=Path)
    args = vars(parser.parse_args())
    action = args.pop("command")
    result = {"import-preparation": import_preparation, "prepare": prepare_recovery, "qualify": qualify_recovery}[action](**args)
    print(json.dumps({"status": result["status"], "new_model_forwards": 0, "new_head_scores": 0}))


if __name__ == "__main__":
    main()
