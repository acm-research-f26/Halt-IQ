"""Public-feature action classification from measured training counterfactuals.

This is supervised one-step utility imitation, not an on-policy RL claim.
Only the external runner scores answers and constructs utilities. Five fixed
seeds are selected using dev utility, then frozen before the runner's test.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform

from .linear import lexical_distance
from .nov2_orchestration import ACTIONS, State, valid_actions

FEATURES = ("verifier_score", "reviewer_approved", "draft_fraction", "has_previous",
            "lexical_distance", "exact_repeat", "feedback_length_fraction",
            "evidence_issue_cue", "contradiction_cue", "challenger_used",
            "answer_length_fraction")
SEEDS = (0, 1, 2, 3, 4)
OBJECTIVE = "supervised_measured_one_step_utility_action_classification"
SCHEMA = "yash-nov2-three-action-linear-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _vector(value):
    if (not isinstance(value, (list, tuple)) or len(value) != len(FEATURES)
            or any(isinstance(x, bool) or not isinstance(x, Real)
                   or not math.isfinite(x) or not 0 <= x <= 1 for x in value)):
        raise ValueError("Exactly eleven finite bounded public features are required")
    return tuple(float(x) for x in value)


def feature_vector(state):
    if not isinstance(state, State):
        raise ValueError("Features require a typed public state")
    state = state.clone()  # Revalidate nested public fields before any action.
    previous = state.previous_answer
    feedback = state.review["feedback"].lower()
    answer = state.draft["answer"]
    return _vector((state.verifier_score, float(state.review["approved"]),
        state.draft_number / state.max_drafts, float(previous is not None),
        1. if previous is None else lexical_distance(previous, answer),
        float(previous is not None and previous.strip().lower() == answer.strip().lower()),
        min(len(feedback.split()) / 80, 1.),
        float(any(cue in feedback for cue in ("evidence", "support", "citation", "passage", "missing"))),
        float(any(cue in feedback for cue in ("contradict", "conflict", "wrong", "incorrect", "oppos"))),
        float(state.challenger_used), min(len(answer.split()) / 40, 1.)))


def _actions(value):
    if (not isinstance(value, (list, tuple)) or not value or value[0] != "STOP"
            or tuple(value) != tuple(action for action in ACTIONS if action in value)):
        raise ValueError("Available actions must be a unique canonical subset including STOP")
    return tuple(value)


def _best(utilities, available):
    # max preserves the explicitly declared STOP, REVISE, INVOKE tie order.
    return max(available, key=lambda action: utilities[action])


@dataclass(frozen=True)
class PolicyConfig:
    epochs: int = 200
    learning_rate: float = .03
    l2: float = .001

    def __post_init__(self):
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("A positive fixed epoch count is required")
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
               for x in (self.learning_rate, self.l2)) or self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("Learning rate must be positive and L2 nonnegative")


@dataclass(frozen=True)
class LinearPolicy:
    weights: tuple
    bias: tuple
    seed: int
    metadata: dict

    def __post_init__(self):
        if (not isinstance(self.weights, (list, tuple)) or len(self.weights) != 3
                or any(not isinstance(row, (list, tuple)) or len(row) != 11 for row in self.weights)
                or not isinstance(self.bias, (list, tuple)) or len(self.bias) != 3):
            raise ValueError("The action head must have shape 11 to 3")
        values = [x for row in self.weights for x in row] + list(self.bias)
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in values):
            raise ValueError("Head parameters must be finite numbers")
        if type(self.seed) is not int or self.seed not in SEEDS or not isinstance(self.metadata, dict):
            raise ValueError("A predeclared seed and provenance are required")
        if self.metadata.get("objective") != OBJECTIVE:
            raise ValueError("The checkpoint must identify its actual action-classification objective")
        _digest(self.metadata)

    def as_dict(self):
        payload = {"schema": SCHEMA, "objective": OBJECTIVE, "features": list(FEATURES),
                   "actions": list(ACTIONS), "seed": self.seed,
                   "weights": [list(row) for row in self.weights], "bias": list(self.bias),
                   "metadata": self.metadata}
        return dict(payload, payload_sha256=_digest(payload))

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"schema", "objective", "features", "actions", "seed", "weights", "bias", "metadata", "payload_sha256"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("Incomplete or undeclared portable action-head fields")
        digest = value.pop("payload_sha256")
        if (digest != _digest(value) or value["schema"] != SCHEMA
                or value["objective"] != OBJECTIVE or value["features"] != list(FEATURES)
                or value["actions"] != list(ACTIONS)):
            raise ValueError("Checkpoint checksum or declared feature/action schema differs")
        return cls(tuple(tuple(row) for row in value["weights"]), tuple(value["bias"]),
                   value["seed"], value["metadata"])

    def _decision(self, features, available):
        # Revalidate even a manually constructed or subsequently mutated head.
        self.__post_init__()
        features, available = _vector(features), _actions(available)
        scores = {action: sum(w*x for w, x in zip(self.weights[i], features)) + self.bias[i]
                  for i, action in enumerate(ACTIONS)}
        if any(not math.isfinite(x) for x in scores.values()):
            raise ValueError("Action logits overflowed")
        maximum = max(scores[action] for action in available)
        exponentials = {action: math.exp(scores[action] - maximum) if action in available else 0.
                        for action in ACTIONS}
        denominator = sum(exponentials.values())
        return {"action": _best(scores, available), "scores": scores,
                "probabilities": {action: value / denominator for action, value in exponentials.items()},
                "features": list(features), "available_actions": list(available),
                "seed": self.seed, "forced": len(available) == 1,
                "probability_interpretation": "masked action softmax, not calibrated answer correctness"}

    def decide(self, state):
        features = feature_vector(state)
        return self._decision(features, valid_actions(state))


def _records(records, split):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("A nonempty training/dev record list is required")
    seen, cleaned = set(), []
    for record in records:
        required = {"question_id", "state_id", "features", "utilities", "available_actions"}
        if (not isinstance(record, dict) or not required <= set(record)
                or set(record) - required - {"split"}
                or ("split" in record and record["split"] != split)):
            raise ValueError("Action fitting accepts declared train/dev utility records only")
        question, identity = record["question_id"], record["state_id"]
        if not isinstance(question, str) or not question or not isinstance(identity, str) or not identity:
            raise ValueError("Public question/state identities are required")
        if (question, identity) in seen:
            raise ValueError("Duplicate training state identity")
        seen.add((question, identity))
        available = _actions(record["available_actions"])
        utilities = record["utilities"]
        if (not isinstance(utilities, dict) or set(utilities) != set(available)
                or any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
                       for x in utilities.values())):
            raise ValueError("Every available action needs an actual finite measured utility")
        cleaned.append({"question_id": question, "state_id": identity,
            "features": _vector(record["features"]), "available_actions": available,
            "utilities": dict(utilities), "target": _best(utilities, available)})
    return cleaned


def _partition(records):
    import torch
    counts = Counter(record["question_id"] for record in records)
    return (torch.tensor([r["features"] for r in records], dtype=torch.float32),
            torch.tensor([ACTIONS.index(r["target"]) for r in records], dtype=torch.long),
            torch.tensor([[action in r["available_actions"] for action in ACTIONS] for r in records], dtype=torch.bool),
            torch.tensor([1 / counts[r["question_id"]] / len(counts) for r in records], dtype=torch.float32))


def _dev_utility(model, xx, masks, records, weights):
    import torch
    choices = model(xx).masked_fill(~masks, -torch.inf).argmax(dim=1).tolist()
    values = [record["utilities"][ACTIONS[choice]] for record, choice in zip(records, choices)]
    return sum(value * float(weight) for value, weight in zip(values, weights))


def fit_policy(train_records, dev_records, output, config=PolicyConfig(), provenance=None):
    """Fit five heads without accepting, opening, or scoring a test partition."""
    import torch
    if not isinstance(config, PolicyConfig):
        raise ValueError("An explicitly validated policy configuration is required")
    training, dev = _records(train_records, "train"), _records(dev_records, "dev")
    train_ids = sorted({r["question_id"] for r in training})
    dev_ids = sorted({r["question_id"] for r in dev})
    if set(train_ids).intersection(dev_ids):
        raise ValueError("Question groups must be disjoint between train and dev")
    provenance = json.loads(json.dumps(provenance or {}, allow_nan=False))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    xx, yy, masks, weights = _partition(training)
    dx, _, dmasks, dweights = _partition(dev)
    source = dict(provenance, policy_module_sha256=_sha(__file__),
        orchestration_module_sha256=_sha(Path(__file__).with_name("nov2_orchestration.py")),
        software={"python": platform.python_version(), "torch": torch.__version__})
    protocol = {"schema": SCHEMA, "objective": OBJECTIVE, "features": list(FEATURES),
        "actions": list(ACTIONS), "seeds": list(SEEDS), "primary_seed_predeclared": 0,
        "configuration": asdict(config), "source": source, "train_ids": train_ids, "dev_ids": dev_ids,
        "train_states": len(training), "dev_states": len(dev),
        "train_records_sha256": _digest(train_records), "dev_records_sha256": _digest(dev_records),
        "utility": "answer F1 - .02 * incremental attempted agent calls / 10; measured externally",
        "question_weighting": "each question equal; its states share its weight equally",
        "selection": "strict maximum greedy dev utility; earliest epoch wins ties",
        "action_ties": list(ACTIONS), "test_used_to_fit_or_select": False,
        "no_missing_action_counterfactual_imputation": True,
        "declared_utc": datetime.now(timezone.utc).isoformat()}
    _save(output / "protocol.json", protocol)
    for seed in SEEDS:
        directory = output / f"seed{seed}"
        directory.mkdir()
        torch.manual_seed(seed)
        model = torch.nn.Linear(11, 3, device="cpu", dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
        initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        best, selected, history = -math.inf, None, []
        for epoch in range(1, config.epochs + 1):
            optimizer.zero_grad()
            logits = model(xx).masked_fill(~masks, -torch.inf)
            loss = (torch.nn.functional.cross_entropy(logits, yy, reduction="none") * weights).sum()
            loss.backward()
            gradient = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters()))
            if not torch.isfinite(loss) or not math.isfinite(gradient):
                raise ValueError("Nonfinite action-classifier loss or gradient")
            before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
            optimizer.step()
            with torch.no_grad():
                current = torch.cat([p.detach().flatten() for p in model.parameters()])
                dev_utility = _dev_utility(model, dx, dmasks, dev, dweights)
                delta = torch.linalg.vector_norm(current - initial).item()
                record = {"epoch": epoch, "train_cross_entropy_before_step": loss.item(),
                    "dev_greedy_utility_after_step": dev_utility,
                    "gradient_norm_before_l2_optimizer_step": gradient,
                    "optimizer_parameter_step_l2": torch.linalg.vector_norm(current - before).item(),
                    "parameter_l2_from_initial": delta,
                    "weights": model.weight.detach().tolist(), "bias": model.bias.detach().tolist()}
            history.append(record)
            if dev_utility > best:
                best, selected = dev_utility, epoch
                metadata = dict(source, objective=OBJECTIVE, train_ids=train_ids, dev_ids=dev_ids,
                    selected_epoch=epoch, selected_dev_utility=best, configuration=asdict(config),
                    target_counts=dict(Counter(r["target"] for r in training)),
                    selection=protocol["selection"], selected_parameter_l2_from_initial=delta,
                    primary_seed_predeclared=0, trained=True)
                head = LinearPolicy(tuple(tuple(row) for row in record["weights"]),
                                    tuple(record["bias"]), seed, metadata)
                _save(directory / "head.json", head.as_dict())
        _save(directory / "history.json", {"seed": seed, "optimizer": "Adam coupled L2 weight_decay",
            "initial_parameters": initial.tolist(), "selected_epoch": selected,
            "optimizer_steps_completed": len(history), "epochs": history})
        LinearPolicy.from_json(directory / "head.json")
    checkpoints = [{"seed": seed, "path": f"seed{seed}/head.json",
                    "sha256": _sha(output / f"seed{seed}/head.json")} for seed in SEEDS]
    freeze = {"schema": SCHEMA, "objective": OBJECTIVE, "seeds": list(SEEDS),
        "protocol_sha256": _sha(output / "protocol.json"), "checkpoints": checkpoints,
        "all_selected_checkpoints_frozen_before_test_scoring": True,
        "test_used_to_fit_or_select": False, "primary_seed_predeclared": 0,
        "frozen_utc": datetime.now(timezone.utc).isoformat()}
    _save(output / "training-freeze.json", freeze)
    return dict(freeze, output=str(output), primary_head_path=str(output / "seed0/head.json"))
