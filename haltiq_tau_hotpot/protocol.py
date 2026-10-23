"""Strict preparation/dispatch contracts for the separate fourteen-hour study.

This module is data-only: importing or validating it performs no model work.
The earlier qualification worker remains a separate bounded implementation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from pc_strands.data import question_key
from .hobson import file_hash
from haltiq_hotpot.native import PINS

PHASES = ("collect", "features", "train", "compare", "audit")
ARMS = ["fixed1", "fixed2", "fixed3", "fixed5", "fixed10", "released", "decomposed", "trained"]
MAIN_SETTINGS = {"hours": 14, "training": {"fit": 400, "tune": 100, "calibration": 100},
    "development": 20, "criteria": ["answer", "evidence"], "arms": ARMS,
    "max_drafts": 10, "threshold": .8,
    "minimum_valid": {"fit": 320, "tune": 80, "calibration": 80},
    "minimum_evidence_calibration": 80, "trained_criteria": ["answer"],
    "trained_gate": "answer_only", "decomposed_gate": "released_answer_and_released_evidence",
    "baseline_before_weight_updates": True, "shared_initial_drafts": True,
    "native_feature_cache": True, "response_cache": True,
    "model_budget_seconds": 46800, "finalization_reserve_seconds": 3600}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pinned(path, sha256):
    if not isinstance(sha256, str) or len(sha256) != 64 or file_hash(path) != sha256:
        raise ValueError("Main-study input hash is missing or changed")
    return read(path)


def settings_match(settings):
    if not isinstance(settings, dict) or any(type(settings.get(k)) is not type(v) or settings.get(k) != v
                                            for k, v in MAIN_SETTINGS.items()):
        raise ValueError("Main-study settings differ from the predeclared research protocol")


def execution_cutoff(config, wrapper_deadline, phase):
    start, model, wrapper = [datetime.fromisoformat(value) for value in
                            (config["start_time"], config["model_deadline"], wrapper_deadline)]
    if (any(value.tzinfo is None or value.utcoffset() is None for value in (start, model, wrapper))
            or start > datetime.now(timezone.utc) or (model - start).total_seconds() != 46800
            or (wrapper - start).total_seconds() != 50400):
        raise ValueError("The fixed fourteen-hour window must reserve its last hour for finalization")
    return wrapper.isoformat() if phase == "audit" else model.isoformat()


def _cohort(spec, count, partition):
    rows = pinned(spec["path"], spec["sha256"])
    if not isinstance(rows, list) or spec.get("count") != count or len(rows) != count:
        raise ValueError("Main cohort count changed")
    identities, questions, types = set(), set(), {"bridge": 0, "comparison": 0}
    for row in rows:
        public = row.get("public", {})
        gold = row.get("evaluator", {})
        if (row.get("partition") != partition or set(public) != {"id", "question", "type", "level", "context"}
                or any(not isinstance(public.get(k), str) or not public[k].strip() for k in ("id", "question", "type", "level"))
                or public["type"] not in types or set(gold) != {"answer", "supporting_facts"}
                or not isinstance(gold["answer"], str) or not gold["answer"].strip()
                or not isinstance(gold["supporting_facts"], list)
                or not isinstance(public["context"], list) or not public["context"]):
            raise ValueError("Invalid separated public/evaluator main cohort")
        for paragraph in public["context"]:
            if (not isinstance(paragraph, list) or len(paragraph) != 2
                    or not isinstance(paragraph[0], str) or not paragraph[0]
                    or not isinstance(paragraph[1], list) or not paragraph[1]
                    or not all(isinstance(sentence, str) for sentence in paragraph[1])):
                raise ValueError("Complete original public passages are required")
        for fact in gold["supporting_facts"]:
            if (not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str)
                    or type(fact[1]) is not int or fact[1] < 0
                    or not any(title == fact[0] and fact[1] < len(sentences) for title, sentences in public["context"])):
                raise ValueError("Evaluator supporting references must name original public sentences")
        key = question_key(public["question"])
        if public["id"] in identities or key in questions:
            raise ValueError("Duplicate main question or normalized question key")
        identities.add(public["id"])
        questions.add(key)
        types[public["type"]] += 1
    if types != spec.get("type_counts") or types["bridge"] != types["comparison"]:
        raise ValueError("Every prepared main cohort must preserve question-type balance")
    return rows, identities, questions


def load_main(config):
    """Verify the full pre-outcome protocol, disjoint cohorts and fresh selection."""
    if config.get("purpose") != "main_study":
        raise ValueError("The main worker cannot execute a qualification or unspecified configuration")
    if config.get("native_feature_cache") is not True or config.get("response_cache") is not True:
        raise ValueError("Main route requires both predeclared exact-input cache policies")
    research = pinned(config["protocol"], config["protocol_sha256"])
    settings_match(research)
    freeze = pinned(config["data_freeze"], config["data_freeze_sha256"])
    if freeze.get("schema") != "tau2-hotpot-main-v1":
        raise ValueError("Exact main data-freeze schema required")
    settings_match(freeze["main"])
    selection = freeze.get("selection", {})
    if (selection.get("answer_blind") is not True or selection.get("quality_filtering") is not False
            or selection.get("length_filtering") is not False or selection.get("replacement_allowed") is not False):
        raise ValueError("Main preparation must remain blind, unfiltered and without replacements")
    pilot = pinned(config["combined_pilot_proof"], config["combined_pilot_proof_sha256"])
    binding = pilot.get("model_binding", {})
    model_files = {"released_head_sha256": Path(config["checkpoint"]) / "head.safetensors",
        "adapter_sha256": Path(config["checkpoint"]) / "lora/adapter_model.safetensors",
        "base_weights_sha256": Path(config["base"]) / "model.safetensors-00001-of-00001.safetensors"}
    upstream = {name: PINS["upstream/" + name] for name in ("modeling.py", "infer.py", "prompting.py", "schema.py")}
    if (pilot.get("schema") != "haltiq-tau-hotpot-qualified-pilot-v1"
            or pilot.get("status") != "PASS_QUALIFIED_ENGINEERING_PILOT"
            or pilot.get("qualified") is not True or pilot.get("terminal") is not True
            or pilot.get("pending_or_uncertain") is not False or pilot.get("cleanup_verified") is not True
            or binding.get("head_parameters") != 1053184 or binding.get("upstream_sha256") != upstream
            or any(binding.get(key) != file_hash(path) for key, path in model_files.items())
            or not isinstance(pilot.get("source_sha256"), dict) or not pilot["source_sha256"]):
        raise ValueError("Actual qualified combined-pilot proof is required; static tests are insufficient")
    cohorts, seen_ids, seen_questions = {}, set(), set()
    for name, count, partition in (("main_development", 20, "development"),
            ("main_fit", 400, "fit"), ("main_tune", 100, "tune"), ("main_calibration", 100, "calibration")):
        rows, ids, questions = _cohort(freeze["files"][name], count, partition)
        if seen_ids & ids or seen_questions & questions:
            raise ValueError("Main development/FIT/TUNE/CAL partitions overlap")
        cohorts[name] = rows
        seen_ids.update(ids)
        seen_questions.update(questions)
    fresh = {}
    previous = {}
    for count in (20, 40, 60):
        rows, ids, questions = _cohort(freeze["files"]["main_fresh" + str(count)], count, "fresh")
        if seen_ids & ids or seen_questions & questions:
            raise ValueError("Fresh cohort overlaps development or learning questions")
        current = {r["public"]["id"]: r for r in rows}
        if any(current.get(q) != row for q, row in previous.items()):
            raise ValueError("Prepared fresh20/40/60 must be exact nested question subsets")
        previous, fresh[count] = current, rows
    chosen = pinned(config["fresh_n_freeze"], config["fresh_n_freeze_sha256"])
    n = chosen.get("n")
    if (type(n) is not int or n not in fresh or chosen.get("selected_n") != n
            or chosen.get("answer_blind") is not True or chosen.get("fresh_scoring_started") is not False
            or chosen.get("basis") != "measured_throughput_and_resources"
            or chosen.get("data_freeze_sha256") != config["data_freeze_sha256"]
            or chosen.get("protocol_sha256") != config["protocol_sha256"]
            or chosen.get("dataset_sha256") != freeze["files"]["main_fresh" + str(n)]["sha256"]):
        raise ValueError("Fresh N needs one exact pre-outcome throughput/resource freeze")
    timestamp = datetime.fromisoformat(chosen["frozen_at"])
    if timestamp.tzinfo is None or timestamp.utcoffset() is None or timestamp > datetime.now(timezone.utc):
        raise ValueError("Fresh freeze needs a truthful past explicit-offset timestamp")
    throughput = pinned(chosen["throughput_proof"], chosen["throughput_proof_sha256"])
    if throughput.get("answer_blind") is not True or throughput.get("whole_route_fits") is not True:
        raise ValueError("A measured whole-route forecast must fit before adoption")
    cohorts["main_fresh"] = fresh[n]
    return {"config": config, "research": research, "freeze": freeze, "cohorts": cohorts,
            "fresh_n": n, "fresh_freeze": chosen}


def verify_dispatch(args, plan, state):
    """Bind the actual main argv and preceding outputs to this frozen invocation."""
    import os
    from haltiq_supervisor.plan import verify_frozen
    from haltiq_supervisor.engine import verify_completion
    root = args.previous.resolve()
    if (Path(__file__).resolve().parents[1] != root / "source" or plan.get("purpose") != "main_study"
            or tuple(s["id"] for s in plan["stages"]) != PHASES
            or state["run_id"] != os.environ.get("HALTIQ_RUN_ID")
            or state["plan_sha256"] != os.environ.get("HALTIQ_PLAN_SHA256")
            or os.environ.get("HALTIQ_PHASE") != args.phase
            or state.get("stages", {}).get(args.phase, {}).get("status") != "started"
            or args.config.resolve() != Path(plan["main_protocol"]["path"]).resolve()
            or args.output.resolve() != root / args.phase
            or args.control.resolve() != root / "workers" / args.phase or args.deadline != plan["deadline"]):
        raise ValueError("Main worker must match its adopted frozen current phase")
    verify_frozen(plan, state, root)
    for stage in plan["stages"]:
        if stage["id"] == args.phase:
            break
        receipt = state.get("stages", {}).get(stage["id"], {})
        if receipt.get("status") != "completed":
            raise ValueError("Every preceding main phase must be terminal")
        for path, expected in receipt.get("output_sha256", {}).items():
            if file_hash(path) != expected:
                raise ValueError("Preceding main evidence changed")
        verify_completion(stage, state, read(root / stage["id"] / "completion.json"))
