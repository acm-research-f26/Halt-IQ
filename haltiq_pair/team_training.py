"""Retrain Sri's small stopping policy with Vedanshi's continuous Strands score.

Only the pinned teammate train/dev partitions fit or select checkpoints. All five
selected checkpoints are published and hashed before held-out replay begins.
The writer and critic language-model weights are never trained here.
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
import random

from .linear import (PROTOCOL_SHA256, QUESTIONS_SHA256, SRI_REVISION,
                     make_features as original_features)
from .replay import CSV_SHA256, TEAM_REVISION, load_trajectories

FEATURES = ("distance", "round_fraction", "distance_delta", "strands_probability",
            "has_previous_draft")
SCHEMA = "yash-retrained-strands-feature-stop-head-v1"
SEEDS = (0, 1, 2, 3, 4)
POLICIES = ("fixed1", "fixed6", "answer_unchanged", "strands_only",
            "retrained_head", "retrained_head_and")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _number(value, low, high, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def validate_features(values):
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ValueError("Exactly five public numeric features are required")
    result = [_number(x, low, high, name) for x, low, high, name in zip(
        values, (0, 0, -1, 0, 0), (1, 1, 1, 1, 1), FEATURES)]
    if result[1] == 0 or result[4] not in (0., 1.):
        raise ValueError("Round fraction must be positive; history flag must be zero or one")
    return result


def make_strands_features(draft, previous, round_index, max_rounds,
                         strands_probability, previous_distance=None):
    """Use zero-based round indices and only current/past public observations."""
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
        raise ValueError("The declared round budget must be an integer")
    probability = _number(strands_probability, 0, 1, "Strands probability")
    values = original_features(draft, previous, round_index, max_rounds, False,
                               previous_distance)
    values[3] = probability
    return validate_features(values)


@dataclass(frozen=True)
class StrandsHead:
    weights: tuple[float, ...]
    bias: float
    seed: int = 0
    metadata: dict | None = None

    def __post_init__(self):
        if (len(self.weights) != 5 or isinstance(self.seed, bool)
                or not isinstance(self.seed, int) or self.seed not in SEEDS
                or any(isinstance(x, bool) or not isinstance(x, Real)
                       or not math.isfinite(x) for x in (*self.weights, self.bias))):
            raise ValueError("Five finite weights, a finite bias and a declared seed are required")

    def probability(self, features):
        values = validate_features(features)
        z = self.bias + sum(w * x for w, x in zip(self.weights, values))
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

    def as_dict(self):
        payload = {"schema": SCHEMA, "features": list(FEATURES),
                   "weights": list(self.weights), "bias": self.bias, "seed": self.seed,
                   "stop_threshold": .5, "metadata": self.metadata or {}}
        return dict(payload, payload_sha256=hashlib.sha256(_canonical(payload)).hexdigest())

    @property
    def provenance(self):
        return self.as_dict()

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("A portable head must be a JSON object")
        checksum = value.pop("payload_sha256", None)
        if (checksum != hashlib.sha256(_canonical(value)).hexdigest()
                or set(value) != {"schema", "features", "weights", "bias", "seed",
                                  "stop_threshold", "metadata"}
                or value["schema"] != SCHEMA or value["features"] != list(FEATURES)
                or value["stop_threshold"] != .5 or not isinstance(value["metadata"], dict)):
            raise ValueError("The portable head schema or checksum changed")
        return cls(tuple(value["weights"]), value["bias"], value["seed"], value["metadata"])


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    group_size: int = 8
    learning_rate: float = .01
    cost: float = .1
    gradient_clip: float = 1.
    ratio_clip: float = .2

    def __post_init__(self):
        for value, minimum in ((self.epochs, 1), (self.group_size, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("Positive epochs and at least two grouped rollouts are required")
        for name in ("learning_rate", "cost", "gradient_clip", "ratio_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Training settings must be finite numbers")
        if (self.learning_rate <= 0 or self.cost < 0 or self.gradient_clip <= 0
                or not 0 < self.ratio_clip < 1):
            raise ValueError("Invalid learning rate, reward cost or clipping settings")


def trajectory_features(rounds):
    result, previous_distance = [], None
    for i, row in enumerate(rounds):
        values = make_strands_features(row["answer"], rounds[i - 1]["answer"] if i else None,
                                      i, len(rounds), row["strands"], previous_distance)
        result.append(values)
        previous_distance = values[0] if i else None
    return result


def reward(rounds, index, cost):
    """Source reward counts writer plus reviewer through the selected round."""
    fraction = sum(r["tokens"] for r in rounds[:index + 1]) / max(1, sum(r["tokens"] for r in rounds))
    return rounds[index]["f1"] - cost * fraction


def _rollout(model, vectors, *, greedy=False):
    import torch
    decisions = []
    for i, vector in enumerate(vectors[:-1]):
        distribution = torch.distributions.Bernoulli(logits=model(vector).squeeze(-1))
        action = (distribution.probs >= .5).float() if greedy else distribution.sample()
        decisions.append((vector, action, distribution.log_prob(action).detach()))
        if action.item():
            return i, decisions
    return len(vectors) - 1, decisions


def _fit_seed(training, dev, seed, config, directory, provenance):
    """Fit from train/dev arguments only; held-out rows are not accepted here."""
    import torch
    if not training or not dev or seed not in SEEDS:
        raise ValueError("Training and development questions and a declared seed are required")
    ids = [t["id"] for t in training + dev]
    if len(ids) != len(set(ids)) or any(t["split"] != "train" for t in training) or any(t["split"] != "dev" for t in dev):
        raise ValueError("Training/development identities or split labels overlap")
    directory.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = torch.nn.Linear(5, 1, device="cpu", dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    vectors = {t["id"]: [torch.tensor(x, dtype=torch.float32) for x in trajectory_features(t["rounds"])]
               for t in training + dev}
    training = list(training)
    best, history, best_head = -math.inf, [], None
    with (directory / "training-trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
        for epoch in range(1, config.epochs + 1):
            rng.shuffle(training)
            signals = 0
            for item in training:
                xx = vectors[item["id"]]
                with torch.no_grad():
                    group = [_rollout(model, xx) for _ in range(config.group_size)]
                returns = torch.tensor([reward(item["rounds"], i, config.cost) for i, _ in group])
                deviation = returns.std(unbiased=False)
                record = {"epoch": epoch, "question_id": item["id"],
                          "sampled_stopping_rounds": [i + 1 for i, _ in group],
                          "sampled_actions": [[int(a.item()) for _, a, _ in decisions] for _, decisions in group],
                          "returns": returns.tolist(), "group_has_signal": bool(deviation >= 1e-8)}
                if deviation >= 1e-8:
                    advantages = (returns - returns.mean()) / (deviation + 1e-6)
                    losses = []
                    for (_, decisions), advantage in zip(group, advantages):
                        current = sum(torch.distributions.Bernoulli(logits=model(x).squeeze(-1)).log_prob(a)
                                      for x, a, _ in decisions)
                        old = sum(logp for _, _, logp in decisions)
                        ratio = torch.exp(current - old)
                        losses.append(-torch.minimum(ratio * advantage,
                            ratio.clamp(1 - config.ratio_clip, 1 + config.ratio_clip) * advantage))
                    optimizer.zero_grad()
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    if not torch.isfinite(loss) or not torch.isfinite(gradient):
                        raise ValueError("Nonfinite training loss or gradient")
                    optimizer.step()
                    signals += 1
                    record.update(advantages=advantages.tolist(), loss=loss.item(), gradient_norm_before_clip=gradient.item())
                record.update(weights=model.weight.detach()[0].tolist(), bias=model.bias.detach()[0].item())
                trace.write(json.dumps(record, allow_nan=False) + "\n")
            with torch.no_grad():
                score = sum(reward(t["rounds"], _rollout(model, vectors[t["id"]], greedy=True)[0], config.cost)
                            for t in dev) / len(dev)
            epoch_record = {"epoch": epoch, "dev_reward": score, "groups_with_signal": signals,
                            "weights": model.weight.detach()[0].tolist(), "bias": model.bias.detach()[0].item()}
            history.append(epoch_record)
            if score > best:
                best = score
                metadata = dict(provenance, trained=True, selected_epoch=epoch, best_dev_reward=best,
                                train_ids=sorted(t["id"] for t in training), dev_ids=sorted(t["id"] for t in dev),
                                training_config=asdict(config), selection="strict highest greedy dev reward; earliest epoch wins ties")
                best_head = StrandsHead(tuple(epoch_record["weights"]), epoch_record["bias"], seed, metadata)
                _save(directory / "head.json", best_head.as_dict())
    _save(directory / "history.json", {"seed": seed, "epochs": history,
                                        "selected_epoch": best_head.metadata["selected_epoch"]})
    return best_head


def replay_one(trajectory, policy, head):
    if policy not in POLICIES:
        raise ValueError("Unknown retraining comparison policy")
    rounds = trajectory["rounds"]
    vectors = trajectory_features(rounds)
    writer = reviewer = critic_calls = head_calls = 0
    decisions = []
    for i, (row, vector) in enumerate(zip(rounds, vectors)):
        final = i == len(rounds) - 1
        writer += row["writer_tokens"]
        probability, approved = None, row["strands"] >= .8
        if policy in {"strands_only", "retrained_head", "retrained_head_and"}:
            critic_calls += 1
        if policy in {"retrained_head", "retrained_head_and"} and not final:
            probability = head.probability(vector)
            head_calls += 1
        learned_stop = probability is not None and probability >= .5
        if policy == "fixed1":
            stop = True
        elif policy == "fixed6":
            reviewer += row["critic_tokens"]
            stop = final
        elif policy == "answer_unchanged":
            stop = i > 0 and row["answer"].strip().lower() == rounds[i - 1]["answer"].strip().lower()
        elif policy == "strands_only":
            stop = approved
        elif policy == "retrained_head":
            stop = learned_stop
        else:
            stop = learned_stop and approved
        forced_cap = final and not stop and policy != "fixed6"
        decisions.append({"round": i + 1, "strands_probability": row["strands"] if critic_calls else None,
                          "features": vector if probability is not None else None,
                          "p_stop": probability, "gate_stop": stop, "forced_cap": forced_cap})
        if stop or final:
            return {"id": trajectory["id"], "split": trajectory["split"], "seed": head.seed,
                    "policy": policy, "answer": row["answer"], "answer_f1": row["f1"], "answer_em": row["em"],
                    "rounds": i + 1, "writer_tokens": writer, "reviewer_tokens": reviewer,
                    "llm_tokens": writer + reviewer, "strands_calls": critic_calls, "head_calls": head_calls,
                    "forced_cap": forced_cap, "decisions": decisions}
        if policy != "fixed6":
            reviewer += row["critic_tokens"]
    raise ValueError("A trajectory requires at least one round")


def _ci(values):
    rng = random.Random(42)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(5000))
    return [means[124], means[4874]]


def evaluate_frozen(trajectories, directory):
    directory = Path(directory)
    freeze = json.loads((directory / "training-freeze.json").read_text(encoding="utf-8"))
    if (freeze["seeds"] != list(SEEDS) or not freeze["all_selected_checkpoints_frozen_before_test_scoring"]
            or len(freeze["checkpoints"]) != len(SEEDS)
            or [entry["seed"] for entry in freeze["checkpoints"]] != list(SEEDS)
            or freeze["protocol_sha256"] != _sha(directory / "protocol.json")):
        raise ValueError("All declared checkpoints must be frozen before test scoring")
    heads = []
    for entry in freeze["checkpoints"]:
        path = directory / entry["path"]
        if _sha(path) != entry["sha256"]:
            raise ValueError("A selected checkpoint changed after freezing")
        head = StrandsHead.from_json(path)
        if head.seed != entry["seed"]:
            raise ValueError("Checkpoint seed differs from its frozen entry")
        heads.append(head)
    test = [t for t in trajectories if t["split"] == "test"]
    test_ids = {t["id"] for t in test}
    if len(test_ids) != len(test) or not test:
        raise ValueError("Unique held-out test questions are required")
    rows, summaries = [], []
    for head in heads:
        if test_ids.intersection(head.metadata["train_ids"] + head.metadata["dev_ids"]):
            raise ValueError("Train/dev questions overlap held-out replay")
        seed_rows = [replay_one(t, policy, head) for t in test for policy in POLICIES]
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
    report = {"execution": "cached_trajectory_replay_after_all_training_frozen", "unique_test_questions": len(test),
              "seeds": list(SEEDS), "live_deployment_seed_predeclared": 0,
              "new_language_model_calls": 0, "new_language_model_training": False,
              "cost_note": "Saved writer/reviewer prefix tokens exclude Strands/reranker compute; not total compute or live latency savings. Fixed6 includes its terminal reviewer. Other policies require reviewer only when continuing.",
              "training_cost_note": "Training reward preserves Sri's writer-plus-reviewer prefix fraction, including terminal reviewer; replay operational cost separately avoids unneeded final review.",
              "policy_note": "The retrained head is fitted with a continuous Strands feature. AND .8 is a separate ablation, not the policy optimized by the head's training reward. The training horizon forces the final round without another head action.",
              "sample_note": "Five seeds reuse the same eighty questions; they are not 400 independent samples. No best-test-seed selection.",
              "summary": summaries, "rows": rows}
    _save(directory / "test80.json", report)
    return report


def train_study(data, output, config=TrainingConfig()):
    trajectories = load_trajectories(data)
    if Counter(t["split"] for t in trajectories) != {"train": 50, "dev": 20, "test": 80}:
        raise ValueError("The frozen teammate 50/20/80 split is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    training = [t for t in trajectories if t["split"] == "train"]
    dev = [t for t in trajectories if t["split"] == "dev"]
    provenance = {"source_team_revision": TEAM_REVISION, "source_sri_revision": SRI_REVISION,
                  "joined_csv_sha256": CSV_SHA256, "source_questions_sha256": QUESTIONS_SHA256,
                  "source_protocol_sha256": PROTOCOL_SHA256,
                  "new_feature": "continuous Strands probability replaces the original LLM approval flag",
                  "new_training": "six-parameter CPU stop policy only; no LLM weight changes",
                  "training_module_sha256": _sha(__file__), "live_seed_predeclared": 0}
    protocol = {"schema": "yash-team-strands-head-training-v1", "frozen_before_training_utc": datetime.now(timezone.utc).isoformat(),
                "source": provenance, "features": list(FEATURES), "seeds": list(SEEDS),
                "configuration": asdict(config), "counts": {"train": 50, "dev": 20, "test": 80},
                "train_ids": sorted(t["id"] for t in training), "dev_ids": sorted(t["id"] for t in dev),
                "test_ids": sorted(t["id"] for t in trajectories if t["split"] == "test"),
                "test_labels_used_to_fit_or_select": False,
                "all_selected_checkpoints_frozen_before_test_scoring": True,
                "live_deployment_seed_predeclared": 0}
    _save(output / "protocol.json", protocol)
    for seed in SEEDS:
        _fit_seed(training, dev, seed, config, output / f"seed{seed}", provenance)
    checkpoints = [{"seed": seed, "path": f"seed{seed}/head.json",
                    "sha256": _sha(output / f"seed{seed}/head.json")} for seed in SEEDS]
    _save(output / "training-freeze.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": _sha(output / "protocol.json"),
        "seeds": list(SEEDS), "all_selected_checkpoints_frozen_before_test_scoring": True,
        "checkpoints": checkpoints, "live_deployment_seed_predeclared": 0})
    return evaluate_frozen(trajectories, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train_study(args.data, args.output)
    print(json.dumps({"output": str(args.output), "trained_seeds": list(SEEDS),
                      "test_questions": report["unique_test_questions"], "live_seed": 0,
                      "summary": report["summary"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
