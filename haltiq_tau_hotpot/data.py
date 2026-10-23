"""Answer-blind, question-grouped preparation; no model imports or dispatch."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from haltiq_hotpot.protocol import DATASET_SHA256, PARTNER_CSV_SHA256, digest
from pc_strands.data import question_key
from .journal import durable_write


def records(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(s) for s in text.splitlines() if s.strip()]


def case_from_row(row, partition):
    row = row.get("source_row", row)
    context = row["context"]
    if isinstance(context, dict):
        context = list(zip(context["title"], context["sentences"], strict=True))
    supporting = row["supporting_facts"]
    if isinstance(supporting, dict):
        supporting = list(zip(supporting["title"], supporting["sent_id"], strict=True))
    return {"partition": partition, "public": {
        "id": row.get("_id", row.get("id")), "question": row["question"],
        "type": row["type"], "level": row.get("level", "unknown"),
        "context": [[title, list(sentences)] for title, sentences in context]},
        "evaluator": {"answer": row["answer"], "supporting_facts": [[title, int(index)] for title, index in supporting]}}


def balanced(rows, count, prefix):
    if type(count) is not int or count <= 0 or count % 2:
        raise ValueError("Select an even positive number before outcomes")
    rank = lambda row: hashlib.sha256((prefix + row["_id"]).encode()).hexdigest()
    chosen = []
    for kind in ("bridge", "comparison"):
        pool = sorted((r for r in rows if r["type"] == kind), key=rank)
        if len(pool) < count // 2:
            raise ValueError("Insufficient questions of each type")
        chosen.extend(pool[:count // 2])
    return sorted(chosen, key=rank)


def prepare(testing_root, output):
    """Pin a small qualification pilot and larger unstarted study cohorts."""
    root, output = Path(testing_root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preparation is exclusive: choose a new output directory")
    dataset = root / "preflight-2026-10-18/hotpot_dev_distractor_v1.hf-rows-verified.json"
    partner = root / "partner-pilot-runtime/sources/haltiq_strands_experiments2.csv"
    previous = root / "partner-pilot-runtime/new20-writer-prepared-v1/writer-plan.json"
    heldout = previous.with_name("tasks.jsonl")
    if digest(dataset) != DATASET_SHA256 or digest(partner) != PARTNER_CSV_SHA256:
        raise ValueError("Original dataset/partner exposure inventory changed")
    old = json.loads(previous.read_text(encoding="utf-8"))
    if digest(heldout) != old["dataset_sha256"]:
        raise ValueError("Prepared original20 changed")
    excluded_ids, excluded_questions, inventory = set(), set(), []
    for spec in old["excluded_source_inventory"]:
        path = root / spec["source"]
        if digest(path) != spec["sha256"]:
            raise ValueError("Historical exposure source changed: " + spec["source"])
        inventory.append({"path": str(path), "sha256": spec["sha256"]})
        for item in records(path):
            row = item.get("source_row", item)
            identifier = row.get("_id", row.get("id", row.get("task_id")))
            if isinstance(identifier, str):
                excluded_ids.add(identifier.removeprefix("hotpotqa:"))
            if isinstance(row.get("question"), str):
                excluded_questions.add(question_key(row["question"]))
    with partner.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            excluded_ids.add(row["id"])
            excluded_questions.add(question_key(row["question"]))
    new20 = [r.get("source_row", r) for r in records(heldout)]
    original = records(dataset)
    pool = [r for r in original if r["_id"] not in excluded_ids | {r["_id"] for r in new20}
            and question_key(r["question"]) not in excluded_questions | {question_key(r["question"]) for r in new20}]
    prefix = "42:tau2-hotpot-qualification-v1:"
    pilot_test = balanced(new20, 4, prefix + "heldout:")
    cohorts, used = {}, set()
    for name, count in (("pilot_fit", 40), ("pilot_tune", 10), ("pilot_calibration", 10),
                        ("main_fit", 400), ("main_tune", 100), ("main_calibration", 100), ("main_fresh60", 60)):
        selected = balanced([r for r in pool if r["_id"] not in used], count, prefix + name + ":")
        used.update(r["_id"] for r in selected)
        partition = name.removeprefix("pilot_").removeprefix("main_")
        cohorts[name] = [case_from_row(row, partition) for row in selected]
    cohorts["pilot_heldout"] = [case_from_row(row, "engineering_heldout") for row in pilot_test]
    # The large study is preparation only. A later answer-blind throughput
    # decision chooses a balanced20/40/60 subset; this function never launches.
    output.mkdir(parents=True)
    files = {}
    for name, cases in cohorts.items():
        path = output / (name + ".json")
        durable_write(path, cases)
        files[name] = {"path": str(path), "sha256": digest(path), "count": len(cases),
                       "type_counts": {kind: sum(c["public"]["type"] == kind for c in cases) for kind in ("bridge", "comparison")}}
    protocol = {"schema": "tau2-hotpot-qualification-v1", "status": "prepared_not_started",
        "dataset_sha256": DATASET_SHA256, "partner_csv_sha256": PARTNER_CSV_SHA256,
        "exposure_sources": inventory, "original20_sha256": digest(heldout),
        "selection": {"answer_blind": True, "quality_filtering": False, "length_filtering": False,
                      "replacement_allowed": False, "seed": 42, "prefix": prefix},
        "pilot": {"training": {"fit": 40, "tune": 10, "calibration": 10}, "heldout": 4,
            "criteria": ["answer"],
            "arms": ["fixed1", "fixed3", "released", "trained"], "max_drafts": 3,
            "threshold": .8, "engineering_only": True, "minimum_valid_fit": 40,
            "minimum_valid_tune": 10, "minimum_valid_calibration": 10,
            "both_fit_labels_required": True, "shared_initial_drafts": True,
            "initial_retrieval": "question-only MiniLM top3", "critic_retrieval": "question+draft MiniLM top3",
            "training_label": "official strict answer EM", "baseline_before_weight_updates": True},
        "main": {"status": "prepared_not_started", "hours": 14, "max_drafts": 10,
            "arms": ["fixed1", "fixed2", "fixed3", "fixed5", "fixed10", "released", "decomposed", "trained"],
            "fresh_choices": [20, 40, 60], "selected_fresh_n": None, "start_time": None},
        "files": files}
    durable_write(output / "data-freeze.json", protocol)
    return protocol
