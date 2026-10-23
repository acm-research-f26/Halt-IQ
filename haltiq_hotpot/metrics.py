"""Critic classification is separate from the unchanged writer's answer quality."""
from __future__ import annotations

import math
import random

from .protocol import ARMS, writer_metrics


def auc(labels, scores):
    positive = [p for y, p in zip(labels, scores, strict=True) if y]
    negative = [p for y, p in zip(labels, scores, strict=True) if not y]
    if not positive or not negative:
        return None
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def classifier(cases, rows):
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {c["id"] for c in cases}:
        raise ValueError("Every planned case must occur exactly once in each arm")
    tp = fp = fn = tn = failures = 0
    labels, scores, nlls, correctness = [], [], [], []
    for case in cases:
        label = bool(writer_metrics(case)["exact_match"])
        row = by_id[case["id"]]
        if row["status"] != "completed":
            failures += 1
            correctness.append(0)
            continue
        probability = float(row["p_true"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid critic probability")
        decision = probability >= .8
        tp += bool(label and decision)
        fp += bool(not label and decision)
        fn += bool(label and not decision)
        tn += bool(not label and not decision)
        labels.append(label)
        scores.append(probability)
        correctness.append(int(label == decision))
        # Stable likelihood from actual raw false/true logits, not clipped probabilities.
        logits = row["raw_logits_false_true"]
        temperature = float(row["temperature"])
        if len(logits) != 2 or not all(math.isfinite(x) for x in logits) or temperature <= 0:
            raise ValueError("Actual finite binary logits and positive temperature required")
        delta = (logits[1] - logits[0]) / temperature
        nlls.append(max(delta, 0) + math.log1p(math.exp(-abs(delta))) - int(label) * delta)
    n, observed = len(cases), len(labels)
    return {"planned_questions": n, "score_coverage": observed / n,
        "observed_scores": observed, "failed_or_not_run": failures,
        "confusion_observed": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "critic_accuracy_failure_inclusive": (tp + tn) / n,
        "critic_accuracy_observed": (tp + tn) / observed if observed else None,
        "positive_class_binary_f1_observed": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "approval_precision": tp / (tp + fp) if tp + fp else None,
        "approval_coverage_planned": (tp + fp) / n, "wrong_approvals": fp,
        "auroc_observed": auc(labels, scores),
        "brier_observed": sum((p - y) ** 2 for p, y in zip(scores, labels)) / observed if observed else None,
        "nll_raw_logits_observed": sum(nlls) / observed if observed else None,
        "metric_scope": "critic_predicts_strict_first_draft_EM; failures_receive_zero_accuracy_credit",
        "_correctness": correctness}


def interval(values):
    if not values:
        return None
    values = sorted(values)
    def quantile(p):
        rank = (len(values) - 1) * p
        low = int(rank)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (rank - low)
    return [quantile(.025), quantile(.975)]


def paired(cases, baseline, candidate, *, draws=5000, seed=42):
    a = {row["id"]: row for row in baseline}
    b = {row["id"]: row for row in candidate}
    # classifier validates exact cohort consistency before sampling.
    sa, sb = classifier(cases, baseline), classifier(cases, candidate)
    differences = [y - x for x, y in zip(sa["_correctness"], sb["_correctness"], strict=True)]
    rng = random.Random(seed)
    deltas, auc_deltas, undefined = [], [], 0
    for _ in range(draws):
        selected = [rng.randrange(len(cases)) for _ in cases]
        deltas.append(sum(differences[i] for i in selected) / len(selected))
        indices = [i for i in selected if a[cases[i]["id"]]["status"] == b[cases[i]["id"]]["status"] == "completed"]
        labels = [bool(writer_metrics(cases[i])["exact_match"]) for i in indices]
        left = auc(labels, [a[cases[i]["id"]]["p_true"] for i in indices])
        right = auc(labels, [b[cases[i]["id"]]["p_true"] for i in indices])
        if left is None or right is None:
            undefined += 1
        else:
            auc_deltas.append(right - left)
    common = [i for i, c in enumerate(cases) if a[c["id"]]["status"] == b[c["id"]]["status"] == "completed"]
    labels = [bool(writer_metrics(cases[i])["exact_match"]) for i in common]
    left = auc(labels, [a[cases[i]["id"]]["p_true"] for i in common])
    right = auc(labels, [b[cases[i]["id"]]["p_true"] for i in common])
    return {"candidate_minus_baseline_accuracy": sum(differences) / len(differences),
        "accuracy_bootstrap95": interval(deltas),
        "candidate_minus_baseline_auroc_complete_pairs": right - left if left is not None and right is not None else None,
        "auroc_bootstrap95_complete_pairs": interval(auc_deltas), "undefined_auroc_draws": undefined,
        "complete_pairs": len(common), "planned_questions": len(cases), "draws": draws,
        "seed": seed, "unit": "question", "rng": "python.random.Random"}


def summarize(cohort, records):
    cases = cohort["cases"]
    writer = [writer_metrics(c) for c in cases]
    models = list(dict.fromkeys(r["model"] for r in records))
    result = {"scope": cohort["scope"], "planned_questions": len(cases),
        "writer_first_draft": {"answer_exact_match": sum(r["exact_match"] for r in writer) / len(writer),
            "answer_token_f1": sum(r["token_f1"] for r in writer) / len(writer),
            "new_writer_calls": 0, "shared_unchanged_across_critic_arms": True}, "models": {}}
    for model in models:
        arms = {}
        for arm in ARMS:
            rows = [r for r in records if r["model"] == model and r["arm"] == arm]
            metrics = classifier(cases, rows)
            metrics.pop("_correctness")
            metrics["by_question_type"] = {}
            for category in ("bridge", "comparison"):
                subset = [c for c in cases if c["type"] == category]
                subset_ids = {c["id"] for c in subset}
                sub = classifier(subset, [r for r in rows if r["id"] in subset_ids])
                sub.pop("_correctness")
                metrics["by_question_type"][category] = sub
            arms[arm] = metrics
        arms["paired_top3_minus_baseline"] = paired(cases,
            [r for r in records if r["model"] == model and r["arm"] == ARMS[0]],
            [r for r in records if r["model"] == model and r["arm"] == ARMS[1]])
        result["models"][model] = arms
    return result
