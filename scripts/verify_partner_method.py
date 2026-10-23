"""Export or verify the shared20 scientific observations; never call a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from haltiq_hotpot.metrics import summarize
from haltiq_hotpot.protocol import ARMS, SCOPE, digest, writer_metrics

ORIGINAL_COMMIT = "3f4387d551f0fdee626b8df03f29c30684b4f373"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cohort_and_records(artifact):
    cohort = {"scope": SCOPE, "cases": [{"id": case["id"], "type": case["type"],
        "public": {"draft": case["draft"]}, "evaluator": {"answer": case["reference"]}}
        for case in artifact["cases"]]}
    rows = [{"id": case["id"], "type": case["type"], "model": "hobson", "arm": arm, **case["scores"][arm]}
            for case in artifact["cases"] for arm in ARMS]
    return cohort, rows


def cached_comparison(cases):
    result = {}
    for arm in ARMS:
        gaps = [abs(c["scores"][arm]["p_true"] - c["partner_cached_scores"][arm]) for c in cases]
        result[arm] = {"same_question_count": len(cases),
            "threshold_0_8_decision_mismatches": sum((c["scores"][arm]["p_true"] >= .8)
                != (c["partner_cached_scores"][arm] >= .8) for c in cases),
            "identical_rounded_probabilities": sum(gap == 0 for gap in gaps),
            "mean_absolute_probability_gap": sum(gaps) / len(gaps), "max_absolute_probability_gap": max(gaps),
            "historical_context_construction_recoverable": arm == "minilm_top3",
            "historical_exact_input_hash_available": False}
    return result


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(artifact, *, source_blobs=True):
    check(artifact["schema"] == "haltiq-partner-method-observations-v1" and artifact["scope"] == SCOPE,
          "Wrong scientific scope")
    check(artifact["heldout_confirmation"] is False and artifact["quality_gain_claim"] is False,
          "Development observations must not become confirmation or gain claims")
    cases = artifact["cases"]
    check(len(cases) == 20 and len({c["id"] for c in cases}) == 20, "Exactly20 unique questions required")
    check(all(sum(c["type"] == kind for c in cases) == 10 for kind in ("bridge", "comparison")),
          "The predeclared question-type balance changed")
    cohort, rows = cohort_and_records(artifact)
    for case, original in zip(cases, cohort["cases"], strict=True):
        scored = writer_metrics(original)
        check(type(case["label"]) is bool and case["label"] == scored["exact_match"], "Evaluator label mismatch")
        check(case["answer_token_f1"] == scored["token_f1"], "Answer tokenF1 mismatch")
        retrieval = case["retrieval"]
        scores = retrieval["scores"]
        check(all(math.isfinite(x) for x in scores) and len(scores) == retrieval["pair_count"], "Invalid retrieval scores")
        check(retrieval["selected_paragraph_indices"] == sorted(range(len(scores)), key=lambda i: -scores[i])[:3],
              "Retrieval ordering differs from the frozen raw scores")
        for arm in ARMS:
            row = case["scores"][arm]
            check(row["status"] == "completed" and row["physical_forwards"] == row["head_scores"] == 1,
                  "All40 planned scores must be genuine completed forwards")
            logits, temperature = row["raw_logits_false_true"], row["temperature"]
            check(len(logits) == 2 and all(math.isfinite(x) for x in logits)
                  and temperature == .9107136998460428, "Raw logits or published Noul temperature changed")
            delta = (logits[1] - logits[0]) / temperature
            probability = 1 / (1 + math.exp(-delta)) if delta >= 0 else math.exp(delta) / (1 + math.exp(delta))
            check(abs(probability - row["p_true_unrounded"]) < 3e-7, "Probability does not follow actual raw logits")
            check(round(row["p_true_unrounded"], 4) == row["p_true"], "Official4decimal probability mismatch")
            check(0 < row["input_tokens"] <= 4096, "Full prompt outside trained context window")
    check(summarize(cohort, rows) == artifact["metrics"], "Recomputed common metrics differ")
    check(cached_comparison(cases) == artifact["same20_cached_partner_comparison"], "Cached comparison differs")
    work = artifact["physical_work"]
    check(work["encoder_forwards"] == sum(r["physical_forwards"] for r in rows) == 40, "Native work mismatch")
    check(work["head_scores"] == sum(r["head_scores"] for r in rows) == 40, "Head work mismatch")
    check(work["encoder_input_tokens"] == sum(r["input_tokens"] for r in rows), "Native token accounting mismatch")
    check(work["reranker_forwards"] == sum(c["retrieval"]["physical_forwards"] for c in cases) == 20,
          "Retrieval work mismatch")
    for output, source in (("reranker_pairs", "pair_count"), ("reranker_input_tokens", "input_tokens"),
                           ("reranker_padded_tokens", "padded_tokens")):
        check(work[output] == sum(c["retrieval"][source] for c in cases), "Retrieval accounting mismatch")
    check(work["new_writer_calls"] == work["weight_updates"] == 0, "No writer or training occurred in this pilot")
    timing = artifact["timing_seconds"]
    check(timing["native_forward_and_head"] == sum(r["seconds"] for r in rows), "Native timing mismatch")
    check(timing["retrieval"] == sum(c["retrieval"]["seconds"] for c in cases), "Retrieval timing mismatch")
    check(timing["startup"] == sum(s["seconds"] for s in artifact["startup"]), "Startup timing mismatch")
    components = ("initial_input_verification", "startup", "native_forward_and_head", "retrieval",
                  "intentional_idle", "residual_overhead_including_unmeasured_cleanup_attempts")
    check(all(math.isfinite(timing[k]) and timing[k] >= 0 for k in components), "Invalid timing component")
    check(abs(sum(timing[k] for k in components) - timing["total_wall"]) < 1e-8, "Overlapping or missing wall-time accounting")
    check(timing["cleanup_attempts_separately_measured"] is False, "Unrecorded cleanup duration must remain unmeasured")
    lifecycle = artifact["lifecycle"]
    check(lifecycle["original_status"] == "interrupted" and lifecycle["original_owned_models_released"] is False
          and lifecycle["cleanup_assertion_failed"] is True, "Original cleanup failure must remain visible")
    check(lifecycle["qualification"] == "QUALIFIED_INFERENCE_COMPLETE_WITH_CLEANUP_FAILURE"
          and lifecycle["qualified_after_process_exit"] is True
          and lifecycle["native_pending_or_uncertain"] is False
          and lifecycle["retrieval_pending_or_uncertain"] is False, "Incomplete post-exit qualification")
    for key, value in artifact["input_file_sha256"].items():
        check(not re.match(r"^[A-Za-z]:", key) and not key.startswith("/"), "Private path in public inputs")
        check(bool(re.fullmatch(r"[0-9a-f]{64}", value)), "Invalid input hash")
    check(artifact["source_commit"] == ORIGINAL_COMMIT, "Wrong original scientific source commit")
    if source_blobs:
        aliases = {"controls": "pc_strands/control.py", "data_validation": "pc_strands/data.py",
                   "answer_metrics": "pc_strands/metrics.py"}
        for key, expected in artifact["input_file_sha256"].items():
            relative = "haltiq_hotpot/" + key.removeprefix("source/") if key.startswith("source/") else aliases.get(key)
            if relative:
                blob = subprocess.check_output(["git", "-C", str(REPO), "show", ORIGINAL_COMMIT + ":" + relative])
                check(hashlib.sha256(blob).hexdigest() == expected, "Original Git source blob differs: " + relative)
    return {"status": "PASS", "planned_questions": 20, "actual_native_forwards": 40,
            "actual_retrieval_batches": 20, "original_cleanup_failure_preserved": True,
            "same20_top3_decision_mismatches": artifact["same20_cached_partner_comparison"]["minilm_top3"]["threshold_0_8_decision_mismatches"]}


def build(args):
    pilot, cohort, qualification = read(args.pilot), read(args.cohort), read(args.qualification)
    independent, os_proof = read(args.independent_audit), read(args.postexit_proof)
    check(qualification["original_terminal_sha256"] == digest(args.pilot), "Qualification binds another terminal")
    check(qualification["independent_inference_audit_sha256"] == digest(args.independent_audit)
          and qualification["postexit_os_proof_sha256"] == digest(args.postexit_proof), "Qualification evidence hash mismatch")
    check(independent["inference_audit_status"] == "PASS"
          and qualification["independent_metrics_journal_state_rawlogit_hash_audit"] == "PASS", "Independent inference audit missing")
    check(qualification["independent_current_process_recheck"]["absent"] is True
          and qualification["independent_current_port11435_closed"] is True
          and qualification["independent_actual_gpu_byte0_lock_acquired_released"] is True, "Post-exit owned-runtime clearance missing")
    check(pilot["frozen_before"] == pilot["frozen_after"] and pilot["frozen_files_unchanged"] is True,
          "Frozen inputs changed")
    check(pilot["cohort_sha256"] == digest(args.cohort), "Cohort hash mismatch")
    check(pilot["frozen_before"]["partner_csv"] == digest(args.partner_csv), "Partner CSV hash mismatch")
    events = [json.loads(line) for line in Path(args.journal).read_text(encoding="utf-8").splitlines()]
    events_by_key = {}
    for event in events:
        key = (event["operation"], event["id"], event.get("model"), event.get("arm"))
        events_by_key.setdefault(key, []).append(event["status"])
    check(len(events_by_key) == 60 and all(v == ["pending", "completed"] for v in events_by_key.values()),
          "Physical journal is incomplete or duplicated")
    source_hashes = set(independent["source_sha256"].values())
    check({digest(p) for p in (args.pilot, args.journal, args.cohort, args.partner_csv)} <= source_hashes,
          "Independent audit does not bind these exact inputs")
    with Path(args.partner_csv).open(encoding="utf-8", newline="") as stream:
        partner_rows = list(csv.DictReader(stream))
    partner = {row["id"]: row for row in partner_rows}
    check(len(partner) == len(partner_rows) == 220, "Unexpected partner pool")
    retrieval = {row["id"]: row for row in pilot["retrieval"]}
    decisions = {(row["id"], row["arm"]): row for row in pilot["records"]}
    check(len(retrieval) == 20 and len(decisions) == 40 and len(pilot["records"]) == 40, "Unexpected physical row count")
    cases = []
    for case in cohort["cases"]:
        identifier = case["id"]
        metrics = writer_metrics(case)
        check(case["public"]["draft"] == partner[identifier]["draft"]
              and case["evaluator"]["answer"] == partner[identifier]["gold"]
              and metrics["exact_match"] == bool(int(partner[identifier]["em"])), "Cached draft/reference/label mismatch")
        cases.append({"id": identifier, "type": case["type"], "label": metrics["exact_match"],
            "draft": case["public"]["draft"], "reference": case["evaluator"]["answer"],
            "answer_token_f1": metrics["token_f1"], "retrieval": retrieval[identifier],
            "scores": {arm: {k: v for k, v in decisions[identifier, arm].items() if k not in {"id", "type", "model", "arm"}}
                       for arm in ARMS},
            "partner_cached_scores": {"baseline_3000": float(partner[identifier]["strands"]),
                                      "minilm_top3": float(partner[identifier]["strands_top3"])}})
    native_seconds = sum(row["seconds"] for row in pilot["records"])
    retrieval_seconds = sum(row["seconds"] for row in pilot["retrieval"])
    startup_seconds = sum(row["seconds"] for row in pilot["startup"])
    residual = pilot["wall_seconds"] - (pilot["initial_input_verification_seconds"] + native_seconds
        + retrieval_seconds + startup_seconds + pilot["intentional_idle_seconds"])
    artifact = {"schema": "haltiq-partner-method-observations-v1", "scope": SCOPE,
        "heldout_confirmation": False, "quality_gain_claim": False, "source_commit": ORIGINAL_COMMIT,
        "sources": {"partner_revision": cohort["partner_revision"], "partner_csv_sha256": digest(args.partner_csv),
            "dataset_revision": cohort["dataset_revision"], "dataset_sha256": cohort["dataset_sha256"],
            "cohort_sha256": digest(args.cohort), "original_terminal_sha256": digest(args.pilot),
            "physical_journal_sha256": digest(args.journal), "independent_inference_audit_sha256": digest(args.independent_audit),
            "postexit_os_proof_sha256": digest(args.postexit_proof), "postexit_qualification_sha256": digest(args.qualification)},
        "input_file_sha256": pilot["frozen_before"], "model_revisions": pilot["model_revisions"],
        "packages": pilot["packages"], "selection": cohort["selection"],
        "protocol": {"instructions": cohort["instructions"], "criteria": None, "threshold": .8,
            "baseline": "all public paragraph sentences, no titles, space-joined, first3000characters",
            "retrieval": "cross-encoder question+space+draft query; title-colon-space paragraph;512pairtokens;stable descending raw logits;top3 newline-joined without character cut",
            "precision": "BF16 Strands torso/FP32 pointer;FP32 CPU MiniLM", "window": "strict4096, no silent truncation",
            "historical_model_and_package_revisions_were_unpinned": True,
            "historical_baseline_exact_context_not_recoverable": True},
        "lifecycle": {"original_status": pilot["status"], "original_owned_models_released": pilot["owned_models_released"],
            "cleanup_assertion_failed": True, "original_error_type": pilot["error_type"],
            "native_pending_or_uncertain": pilot["native_pending_or_uncertain"],
            "retrieval_pending_or_uncertain": pilot["retrieval_pending_or_uncertain"],
            "qualification": qualification["status"], "qualified_after_process_exit": True,
            "original_terminal_never_rewritten": qualification["original_terminal_never_rewritten"],
            "exact_residual_allocation_cause_measured": False},
        "physical_work": {"encoder_forwards": pilot["physical_encoder_forwards"], "head_scores": pilot["head_scores"],
            "encoder_input_tokens": pilot["physical_encoder_tokens"], "reranker_forwards": pilot["physical_reranker_forwards"],
            "reranker_pairs": sum(r["pair_count"] for r in pilot["retrieval"]),
            "reranker_input_tokens": pilot["physical_reranker_tokens"],
            "reranker_padded_tokens": sum(r["padded_tokens"] for r in pilot["retrieval"]),
            "new_writer_calls": pilot["new_writer_calls"], "weight_updates": pilot["weight_updates"],
            "hosted_api_calls": 0, "hosted_api_cost_usd": 0,
            "historical_writer_and_partner_inference_expense_included": False, "local_hardware_cost_usd": None},
        "startup": pilot["startup"], "timing_seconds": {"total_wall": pilot["wall_seconds"],
            "initial_input_verification": pilot["initial_input_verification_seconds"], "startup": startup_seconds,
            "native_forward_and_head": native_seconds, "retrieval": retrieval_seconds,
            "intentional_idle": pilot["intentional_idle_seconds"],
            "residual_overhead_including_unmeasured_cleanup_attempts": residual,
            "cleanup_attempts_separately_measured": False},
        "cases": cases, "metrics": pilot["metrics"],
        "same20_cached_partner_comparison": cached_comparison(cases)}
    verify(artifact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(artifact, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return {"status": "PASS", "artifact_sha256": digest(output), "planned_questions": 20, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("build")
    for name in ("pilot", "journal", "cohort", "partner-csv", "qualification", "independent-audit", "postexit-proof", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    validate = commands.add_parser("verify")
    validate.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == "build" else verify(read(args.artifact))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
