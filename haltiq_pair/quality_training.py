"""Exploratory balanced correctness classifier over the teammate's public features.

This is a separately declared objective, not Sri's policy-gradient objective.
It fits train50 exact-match labels, selects on dev20 BCE, freezes every seed,
then reports cached test80. Seed0 and a .5 threshold are fixed in advance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform
import random

from .linear import PROTOCOL_SHA256, QUESTIONS_SHA256, SRI_REVISION
from .replay import CSV_SHA256, TEAM_REVISION, load_trajectories
from .team_training import FEATURES, SEEDS, StrandsHead, replay_one, trajectory_features

OBJECTIVE = "balanced_BCE_exact_match"
POLICIES = ("fixed1", "fixed6", "answer_unchanged", "strands_only", "quality_head", "quality_head_and")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class QualityConfig:
    epochs: int = 200
    learning_rate: float = .01
    l2: float = .01

    def __post_init__(self):
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("A positive fixed epoch count is required")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               for v in (self.learning_rate, self.l2)) or self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("Learning rate must be positive and L2 nonnegative")


def _partition(items, split):
    import torch
    if not items or any(t["split"] != split for t in items) or len({t["id"] for t in items}) != len(items):
        raise ValueError("A unique, nonempty declared partition is required")
    vectors, labels, weights = [], [], []
    for item in items:
        rounds = item["rounds"]
        if len(rounds) != 6:
            raise ValueError("Each question requires the declared six-round trajectory")
        vectors.extend(trajectory_features(rounds))
        for row in rounds:
            label = row["em"]
            if isinstance(label, bool) or not isinstance(label, Real) or label not in (0., 1.):
                raise ValueError("Training labels must be exact-match zeros or ones")
            labels.append(float(label))
            weights.append(1 / len(rounds))
    return (torch.tensor(vectors, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32))


def _balanced_bce(logits, labels, question_weights, positive_weight, questions):
    import torch
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=torch.tensor(positive_weight), reduction="none")
    return (losses * question_weights).sum() / questions


def _fit_seed(training, dev, seed, config, directory, provenance):
    import torch
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError("A predeclared seed is required")
    if {t["id"] for t in training}.intersection(t["id"] for t in dev):
        raise ValueError("Training and dev identities overlap")
    xx, yy, ww = _partition(training, "train")
    dx, dy, dw = _partition(dev, "dev")
    positives = int(yy.sum().item())
    negatives = len(yy) - positives
    if not positives or not negatives:
        raise ValueError("Balanced BCE requires both training label classes")
    positive_weight = negatives / positives
    balance = {"positive_rounds": positives, "negative_rounds": negatives,
               "pos_weight": positive_weight, "source": "training partition only",
               "question_round_weight": 1/6, "normalization": "mean of per-question mean losses"}
    directory.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = torch.nn.Linear(5, 1, device="cpu", dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
    initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    history, best, best_head = [], math.inf, None
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        loss = _balanced_bce(model(xx).squeeze(-1), yy, ww, positive_weight, len(training))
        loss.backward()
        gradient_norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters()))
        if not torch.isfinite(loss) or not math.isfinite(gradient_norm):
            raise ValueError("Nonfinite correctness-classifier loss or gradient")
        before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        optimizer.step()
        with torch.no_grad():
            train_loss = _balanced_bce(model(xx).squeeze(-1), yy, ww, positive_weight, len(training)).item()
            dev_loss = _balanced_bce(model(dx).squeeze(-1), dy, dw, positive_weight, len(dev)).item()
            current = torch.cat([p.detach().flatten() for p in model.parameters()])
            step_delta = torch.linalg.vector_norm(current - before).item()
            initial_delta = torch.linalg.vector_norm(current - initial).item()
        record = {"epoch": epoch, "optimizer_step": epoch, "train_balanced_bce": train_loss,
                  "dev_balanced_bce": dev_loss, "gradient_norm_before_l2_optimizer_step": gradient_norm,
                  "optimizer_parameter_step_l2": step_delta, "parameter_l2_from_initial": initial_delta,
                  "weights": model.weight.detach()[0].tolist(), "bias": model.bias.detach()[0].item()}
        history.append(record)
        if dev_loss < best:
            best = dev_loss
            metadata = dict(provenance, objective=OBJECTIVE, trained=True, selected_epoch=epoch,
                            best_dev_balanced_bce=best, train_balance=balance, training_config=asdict(config),
                            train_ids=sorted(t["id"] for t in training), dev_ids=sorted(t["id"] for t in dev),
                            selected_parameter_l2_from_initial=initial_delta,
                            selection="strict minimum dev balanced BCE using training class ratio; earliest epoch wins ties")
            best_head = StrandsHead(tuple(record["weights"]), record["bias"], seed, metadata)
            _save(directory / "head.json", best_head.as_dict())
    _save(directory / "history.json", {"seed": seed, "objective": OBJECTIVE, "train_balance": balance,
        "optimizer": "Adam with coupled L2 weight_decay", "l2_gradient_coefficient": config.l2,
        "optimizer_steps_completed": len(history), "final_parameter_l2_from_initial": history[-1]["parameter_l2_from_initial"],
        "initial_parameters": initial.tolist(), "selected_epoch": best_head.metadata["selected_epoch"], "epochs": history})
    return best_head


def _ci(values):
    rng = random.Random(42)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(5000))
    return [means[124], means[4874]]


def _classification(head, test):
    scores, labels = [], []
    for item in test:
        scores.extend(head.probability(x) for x in trajectory_features(item["rounds"]))
        labels.extend(r["em"] for r in item["rounds"])
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    auc = (sum((p > n) + .5 * (p == n) for p in positives for n in negatives)
           / (len(positives) * len(negatives))) if positives and negatives else None
    return {"seed": head.seed, "questions": len(test), "correlated_rounds": len(labels),
            "exact_match_positive_rounds": len(positives), "auroc_descriptive": auc,
            "brier_score": sum((s-y)**2 for s, y in zip(scores, labels)) / len(labels),
            "approved_at_point5": sum(s >= .5 for s in scores),
            "correct_approvals": sum(s >= .5 and bool(y) for s, y in zip(scores, labels)),
            "note": "Round-level descriptive classifier metrics; repeated rounds are correlated, not independent fresh questions."}


def evaluate_frozen(trajectories, directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    freeze = json.loads((directory / "training-freeze.json").read_text(encoding="utf-8"))
    if (freeze["seeds"] != list(SEEDS) or len(freeze["checkpoints"]) != 5
            or [entry["seed"] for entry in freeze["checkpoints"]] != list(SEEDS)
            or not freeze["all_selected_checkpoints_frozen_before_test_scoring"]
            or freeze["protocol_sha256"] != _sha(directory / "protocol.json")):
        raise ValueError("Every declared selected checkpoint must be frozen before test scoring")
    test = [t for t in trajectories if t["split"] == "test"]
    test_ids = {t["id"] for t in test}
    if len(test) != 80 or len(test_ids) != 80 or test_ids != set(protocol["test_ids"]):
        raise ValueError("Evaluation differs from the frozen eighty-question test partition")
    heads = []
    for entry in freeze["checkpoints"]:
        path = directory / entry["path"]
        if _sha(path) != entry["sha256"]:
            raise ValueError("A frozen correctness checkpoint changed")
        head = StrandsHead.from_json(path)
        if (head.seed != entry["seed"] or head.metadata["objective"] != OBJECTIVE
                or test_ids.intersection(head.metadata["train_ids"] + head.metadata["dev_ids"])):
            raise ValueError("Checkpoint identity, objective or partition boundary differs")
        heads.append(head)
    rows, summaries, classifier_metrics = [], [], []
    names = {"quality_head": "retrained_head", "quality_head_and": "retrained_head_and"}
    for head in heads:
        seed_rows = []
        for item in test:
            for policy in POLICIES:
                row = replay_one(item, names.get(policy, policy), head)
                row.update(policy=policy, objective=OBJECTIVE)
                seed_rows.append(row)
        rows.extend(seed_rows)
        base = {r["id"]: r for r in seed_rows if r["policy"] == "fixed6"}
        baseline_tokens = sum(r["llm_tokens"] for r in base.values())
        for policy in POLICIES:
            selected = [r for r in seed_rows if r["policy"] == policy]
            differences = [r["answer_f1"] - base[r["id"]]["answer_f1"] for r in selected]
            summaries.append({"seed": head.seed, "policy": policy, "questions": len(test),
                **{f"mean_{k}": sum(r[k] for r in selected) / len(test) for k in
                   ("answer_f1", "answer_em", "rounds", "llm_tokens", "strands_calls", "head_calls")},
                "forced_cap_returns": sum(r["forced_cap"] for r in selected),
                "f1_difference_vs_fixed6": sum(differences) / len(test),
                "paired_f1_difference_ci95_vs_fixed6": _ci(differences),
                "llm_prefix_savings_percent_vs_fixed6": 100 * (1 - sum(r["llm_tokens"] for r in selected) / baseline_tokens)})
        classifier_metrics.append(_classification(head, test))
    report = {"objective": OBJECTIVE, "execution": "cached_test_replay_after_all_classifiers_frozen",
        "evaluation_scope": "Exploratory known teammate cached test80, held out from fitting and checkpoint selection; previously viewed, not new blind confirmation.",
        "unique_test_questions": 80, "seeds": list(SEEDS), "live_deployment_seed_predeclared": 0,
        "thresholds_predeclared": {"quality_classifier": .5, "additional_strands_AND": .8},
        "not_Sri_policy_gradient": True, "not_stop_cost_optimized": True,
        "new_language_model_calls": 0, "new_language_model_training": False,
        "cost_note": "Cached writer/reviewer prefix tokens exclude verifier/reranker compute. Fixed6 includes its terminal reviewer; adaptive arms use reviewer only when continuing. No total-compute or live-latency savings claim.",
        "policy_note": "A linear exact-match correctness classifier supplies a .5 stopping rule; AND .8 is a separate ablation. The final horizon returns the last answer as a cap, without another learned head action.",
        "sample_note": "All five seeds reuse the same eighty questions. All seeds are reported; seed0 was declared before scoring and is not selected by test results.",
        "classification_by_seed": classifier_metrics, "summary": summaries, "rows": rows}
    _save(directory / "test80.json", report)
    return report


def train_study(data, output, config=QualityConfig()):
    import torch
    trajectories = load_trajectories(data)
    if Counter(t["split"] for t in trajectories) != {"train": 50, "dev": 20, "test": 80}:
        raise ValueError("Only the pinned teammate fifty/twenty/eighty partitions are permitted")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    training = [t for t in trajectories if t["split"] == "train"]
    dev = [t for t in trajectories if t["split"] == "dev"]
    provenance = {"objective": OBJECTIVE, "source_sri_revision": SRI_REVISION,
        "source_team_revision": TEAM_REVISION, "joined_csv_sha256": CSV_SHA256,
        "source_questions_sha256": QUESTIONS_SHA256, "source_protocol_sha256": PROTOCOL_SHA256,
        "training_module_sha256": _sha(__file__), "feature_module_sha256": _sha(Path(__file__).with_name("team_training.py")),
        "feature_encoder": "same public lexical/history features plus continuous Strands probability",
        "label": "strict exact-match em column, training labels only for optimizer; dev labels only for checkpoint selection",
        "no_language_model_training": True, "live_seed_predeclared": 0,
        "software": {"python": platform.python_version(), "torch": torch.__version__}}
    protocol = {"schema": "yash-exploratory-correctness-classifier-v1", "objective": OBJECTIVE,
        "declared_before_training_utc": datetime.now(timezone.utc).isoformat(), "source": provenance,
        "features": list(FEATURES), "seeds": list(SEEDS), "configuration": asdict(config),
        "train_ids": sorted(t["id"] for t in training), "dev_ids": sorted(t["id"] for t in dev),
        "test_ids": sorted(t["id"] for t in trajectories if t["split"] == "test"),
        "counts": {"train_questions": 50, "train_rounds": 300, "dev_questions": 20, "dev_rounds": 120, "test_questions": 80},
        "class_balance": "negative/positive exact-match counts from train300 only; same fixed ratio in dev BCE",
        "regularization": f"Adam coupled L2 weight_decay {config.l2}; checkpoint criterion is weighted dev data BCE, excluding L2 penalty",
        "thresholds_predeclared": {"correctness_classifier": .5, "additional_strands_AND": .8},
        "test_labels_used_to_fit_or_select": False, "no_hyperparameter_search": True,
        "evaluation_scope": "Known teammate test80 is held out from fit, previously viewed and exploratory. A separate fresh cohort is needed for confirmation.",
        "all_selected_checkpoints_frozen_before_test_scoring": True, "live_deployment_seed_predeclared": 0}
    _save(output / "protocol.json", protocol)
    for seed in SEEDS:
        _fit_seed(training, dev, seed, config, output / f"seed{seed}", provenance)
    checkpoints = [{"seed": seed, "path": f"seed{seed}/head.json", "sha256": _sha(output / f"seed{seed}/head.json")}
                   for seed in SEEDS]
    _save(output / "training-freeze.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(),
        "objective": OBJECTIVE, "protocol_sha256": _sha(output / "protocol.json"), "seeds": list(SEEDS),
        "all_selected_checkpoints_frozen_before_test_scoring": True, "checkpoints": checkpoints,
        "live_deployment_seed_predeclared": 0})
    return evaluate_frozen(trajectories, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train_study(args.data, args.output)
    print(json.dumps({"output": str(args.output), "objective": OBJECTIVE,
                      "live_seed": 0, "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
