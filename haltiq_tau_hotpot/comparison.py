"""Live controlled episodes and question-paired common outcome analysis."""
from __future__ import annotations

from copy import deepcopy
import json
import math

from pc_strands.compare import auroc
from pc_strands.metrics import paired_delta
from .journal import durable_write, fingerprint


def diagnostics(observations, planned_n, threshold=.8):
    """Probabilities use scored first drafts; quality denominators remain plannedN."""
    labels, probabilities, losses = [], [], []
    approved = tp = tn = fp = fn = 0
    for row in observations:
        y, result = row["label"], row["score"]
        p = result["p_true_unrounded"]
        logits, temperature = result["raw_logits_false_true"], result["temperature"]
        if not 0 <= p <= 1 or not all(math.isfinite(v) for v in logits) or temperature <= 0:
            raise ValueError("Invalid decision probability/logits")
        margin = (logits[1] - logits[0]) / temperature
        losses.append(max(margin, 0) - y * margin + math.log1p(math.exp(-abs(margin))))
        labels.append(y)
        probabilities.append(p)
        selected = result["p_true"] >= threshold  # Native served rounding is the route rule.
        approved += selected
        tp += selected and y == 1
        fp += selected and y == 0
        tn += not selected and y == 0
        fn += not selected and y == 1
    n = len(labels)
    return {"planned_questions": planned_n, "scored_questions": n,
        "score_coverage": n / planned_n, "classification_accuracy_at_gate_threshold": (tp + tn) / n if n else None,
        "binary_f1_at_gate_threshold": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "auroc": auroc(labels, probabilities) if n else None,
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n if n else None,
        "nll": sum(losses) / n if n else None, "approved": approved,
        "approval_precision": tp / approved if approved else None,
        "approval_coverage_of_planned": approved / planned_n,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}, "threshold": threshold}


def analyse(episodes, cases, arms, *, initial_drafts=None):
    from .scoring import score
    ids = [case["public"]["id"] for case in cases]
    gold = {case["public"]["id"]: case["evaluator"] for case in cases}
    by_arm = {arm: {} for arm in arms}
    for row in episodes:
        qid, arm = row["question_id"], row["arm"]
        if qid not in gold or arm not in by_arm or qid in by_arm[arm]:
            raise ValueError("Foreign or duplicate question/arm outcome")
        submission = row["submitted"] if row["status"] == "completed" else None
        recomputed = score(submission, gold[qid])
        if recomputed != row["metrics"]:
            raise ValueError("Recorded metrics differ from the common evaluator")
        by_arm[arm][qid] = {**row, **recomputed}
    statistics = {}
    for arm, rows in by_arm.items():
        observations = []
        for qid, row in rows.items():
            draft = next((x["draft"] for x in row["trace"] if x["event"] == "draft" and x.get("status") == "completed"), None)
            gate = next((x for x in row["trace"] if x["event"] == "gate" and x.get("criterion") == "answer" and x.get("status") == "completed"), None)
            if draft is not None and gate is not None:
                observations.append({"label": int(score(draft, gold[qid])["em"]), "score": gate["score"]})
        def mean(metric):
            return sum(row[metric] for row in rows.values()) / len(ids)
        repairs = regressions = lost = 0
        for qid, row in rows.items():
            first = (initial_drafts or {}).get(qid)
            if first is not None:
                initial_correct = bool(score(first, gold[qid])["em"])
                final_correct = bool(row["em"])
                repairs += not initial_correct and final_correct
                regressions += initial_correct and row["status"] == "completed" and not final_correct
                lost += initial_correct and not final_correct
        statistics[arm] = {"planned_n": len(ids), "recorded_n": len(rows),
            "completed": sum(r["status"] == "completed" for r in rows.values()),
            "failures_or_missing": len(ids) - sum(r["status"] == "completed" for r in rows.values()),
            "metrics": {key: mean(key) for key in ("em", "f1", "sp_em", "sp_f1", "joint_em", "joint_f1")},
            "mean_logical_drafts": sum(r["counts"]["answer_drafts"] for r in rows.values()) / len(ids),
            "repairs": repairs, "regressions_among_submissions": regressions,
            "correct_initial_answers_lost_including_failures": lost,
            "first_draft_decision": diagnostics(observations, len(ids)) if arm not in {"fixed1", "fixed2", "fixed3", "fixed5", "fixed10"} else None,
            "types": {kind: {"n": len(group := [c["public"]["id"] for c in cases if c["public"]["type"] == kind]),
                "em": sum(rows[q]["em"] for q in group if q in rows) / len(group),
                "f1": sum(rows[q]["f1"] for q in group if q in rows) / len(group)} for kind in ("bridge", "comparison")}}
    paired = {}
    for candidate, baseline in (("fixed3", "fixed1"), ("released", "fixed1"), ("trained", "released"), ("trained", "fixed3")):
        if candidate in by_arm and baseline in by_arm:
            paired[candidate + "_minus_" + baseline] = {key: paired_delta(by_arm[candidate], by_arm[baseline], ids,
                key, draws=5000, seed=42) for key in ("em", "f1", "sp_f1", "joint_f1")}
    return {"scope": "engineering_qualification_not_population_gain", "planned_n": len(ids),
        "question_ids": ids, "policies": statistics, "paired": paired,
        "bootstrap": {"draws": 5000, "seed": 42, "unit": "question", "failures": "zero quality against all planned questions"},
        "interpretation": "Real live revisions and submissions; shared initial drafts/features have no duplicate physical generation. Small engineering sample cannot establish improvement."}


def compare(args, config, freeze, cohorts, control):
    from .agent import run_episode
    from .hobson import load_head, score_head
    from .runtime import LocalRuntime
    from .worker import bare_case, completion, hydrate, initial_drafts, read
    from haltiq_hotpot.protocol import digest
    saved = initial_drafts(args.previous / "collect")
    features = {r["question_id"]: r for r in read(args.previous / "features/features.json")
                if r["partition"] == "engineering_heldout" and r["status"] == "completed"}
    learning = read(args.previous / "train/learning.json")
    if learning["status"] != "completed" or not learning["frozen_torso_adapter_files_unchanged"]:
        raise ValueError("Verified actual head training/calibration is required before paired comparison")
    criterion = learning["criteria"]["answer"]
    trained_path = args.previous / "train/learned/answer-head.safetensors"
    if digest(trained_path) != criterion["trained_head_sha256"]:
        raise ValueError("Selected trained head changed")
    rows, initial = [], {}
    runtime = LocalRuntime(config, args.output, control)
    try:
        with runtime:
            released_head, trained_head = load_head(config["checkpoint"]), load_head(config["checkpoint"], trained_path)
            for case in cohorts["pilot_heldout"]:
                public, qid = case["public"], case["public"]["id"]
                draft = saved[qid]["episode"]["submitted"]
                initial[qid] = draft
                for arm in freeze["pilot"]["arms"]:
                    def writer(state, feedback):
                        if state["previous_draft"] is None:
                            if draft is None:
                                raise ValueError("Shared initial draft failed; do not redispatch")
                            return deepcopy(draft)
                        return runtime.generate(state, feedback)
                    def reviewer(state, proposed):
                        ranked = runtime.rank(public, proposed["answer"])
                        evidence = [{"title": public["context"][i][0],
                                     "sentences": [{"index": n, "text": text} for n, text in enumerate(public["context"][i][1])]}
                                    for i in ranked["indices"]]
                        return runtime.generate({"question": public["question"], "evidence": evidence,
                            "draft": proposed}, review=True)
                    def gate(state, proposed, component):
                        if component != "answer":
                            raise ValueError("This qualification pilot compares the answer gate only")
                        temperature = criterion[("trained" if arm == "trained" else "released") + "_calibration"]["temperature"]
                        if state["previous_draft"] is None and proposed == draft and qid in features:
                            material = hydrate(features[qid])["feature"]
                            return runtime._physical("cpu_head", {"feature": material["provenance_sha256"],
                                "head": arm, "temperature": temperature}, lambda: score_head(
                                    trained_head if arm == "trained" else released_head, material, temperature=temperature))
                        return runtime.observe(public, proposed, criterion=component,
                            head_path=trained_path if arm == "trained" else None, temperature=temperature)
                    episode = run_episode(bare_case(case), writer, reviewer, gate, arm=arm,
                        max_drafts=freeze["pilot"]["max_drafts"], threshold=.8,
                        passage_indices=saved[qid]["initial_retrieval"]["indices"], timeout_seconds=1800)
                    episode["shared_initial_draft_sha256"] = fingerprint(draft)
                    episode["shared_initial_physical_calls"] = 0
                    durable_write(args.output / "episodes" / f"{qid}-{arm}.json", episode)
                    rows.append(episode)
    finally:
        durable_write(args.output / "runtime-terminal.json", {"cleanup": runtime.cleanup,
            "calls": runtime.journal.verify(), "timings": runtime.timings,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
    durable_write(args.output / "episodes.json", rows)
    result = analyse(rows, cohorts["pilot_heldout"], freeze["pilot"]["arms"], initial_drafts=initial)
    durable_write(args.output / "comparison.json", result)
    completed = sum(row["status"] == "completed" for row in rows)
    return completion(args, planned=len(cohorts["pilot_heldout"]) * len(freeze["pilot"]["arms"]),
        completed=completed, failed=len(rows) - completed, cleanup=runtime.cleanup["verified"],
        pending=runtime.journal.summary()["pending_or_uncertain"])
