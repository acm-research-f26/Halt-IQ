"""Declared Strands-feature policies for Yash's separately trained extension.

These controllers receive public drafts and verifier scores only. The original
frozen-transfer controller and its recorded experiments remain unchanged.
"""
from __future__ import annotations

import json
import math

from .linear import lexical_distance


POLICIES = ("fixed1", "fixed6", "repeat", "strands", "orchestrator",
            "retrained", "retrained_and")


def canonical_policy(name):
    aliases = {"baseline": "fixed1", "linear": "retrained",
               "combined": "retrained_and", "answer_unchanged": "repeat"}
    if not isinstance(name, str):
        raise ValueError("A declared policy name is required")
    name = aliases.get(name.lower(), name.lower())
    if name not in POLICIES:
        raise ValueError("Unknown aligned policy")
    return name


class Gate:
    """A trained score-feature head; an extra AND condition is an ablation.

    Public round_index is one based. The new head's fourth feature is the
    continuous Strands probability, independently of the LLM reviewer's flag.
    A false decision at the round cap stays false: the benchmark runner owns
    and records its declared budget-return or action-refusal behavior.
    """

    def __init__(self, *, critic, head, policy_name="strands_feature_retrained",
                 approve_threshold=.8, stop_threshold=.5):
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   and math.isfinite(x) and 0 <= x <= 1
                   for x in (approve_threshold, stop_threshold)):
            raise ValueError("Policy thresholds must be finite probabilities")
        if critic is None or head is None:
            raise ValueError("Both the verifier and separately trained head are required")
        self.critic, self.head = critic, head
        self.policy_name = policy_name
        self.approve_threshold, self.stop_threshold = approve_threshold, stop_threshold

    @property
    def provenance(self):
        return {"policy_family": self.policy_name,
                "head": self.head.provenance,
                "critic": getattr(self.critic, "provenance", {}),
                "new_training": True, "language_model_weights_trained": False,
                "approve_threshold": self.approve_threshold,
                "stop_threshold": self.stop_threshold,
                "head_feature": "continuous Strands probability, not reviewer approval",
                "llm_reviewer": "public feedback for revision; not a head feature",
                "policy_names": {"baseline": "first valid proposal",
                    "strands": "Strands threshold only",
                    "retrained": "trained Strands-feature head",
                    "retrained_AND": "trained head AND Strands threshold"},
                "prior_evidence": "Teammate train/dev trajectories only; superseded Yash evidence excluded",
                "cap_semantics": "runner-owned; a cap is not a learned approval"}

    def decide(self, arm, draft, previous, round_index, max_rounds, question,
               passages, previous_distance=0, critic_approved=False):
        policy = canonical_policy(arm)
        if (type(round_index) is not int or type(max_rounds) is not int
                or not 1 <= round_index <= max_rounds <= 6):
            raise ValueError("Declare a one-based round within a six-round cap")
        if not isinstance(draft, str) or (round_index > 1 and not isinstance(previous, str)):
            raise ValueError("Current and previous public drafts are required")
        distance = 1.0 if round_index == 1 else lexical_distance(previous, draft)
        repeated = previous is not None and draft.strip().lower() == previous.strip().lower()
        probability = p_stop = features = critic_record = None
        score_needed = policy in {"strands", "orchestrator", "retrained", "retrained_and"}
        if score_needed:
            probability = self.critic.score(question, passages, draft)
            if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError("Verifier probability is invalid")
            probability = float(probability)
            critic_record = getattr(self.critic, "last_record", None)
        # Training forces the last round rather than sampling another action.
        # Keep that cap separate from a learned approval in live receipts too.
        head_needed = policy in {"retrained", "retrained_and"} and round_index < max_rounds
        if head_needed:
            from .team_training import make_strands_features
            features = make_strands_features(draft, previous, round_index - 1,
                max_rounds, probability, previous_distance)
            p_stop = self.head.probability(features)
            if (isinstance(p_stop, bool) or not isinstance(p_stop, (int, float))
                    or not math.isfinite(p_stop) or not 0 <= p_stop <= 1):
                raise ValueError("Trained head probability is invalid")
            p_stop = float(p_stop)
        approved = probability is not None and probability >= self.approve_threshold
        head_stop = p_stop is not None and p_stop >= self.stop_threshold
        stopping = {"fixed1": True, "fixed6": round_index == max_rounds,
                    "repeat": repeated, "strands": approved,
                    "orchestrator": approved or repeated,
                    "retrained": head_stop,
                    "retrained_and": head_stop and approved}[policy]
        result = {"stop": stopping, "policy": policy, "p_stop": p_stop,
                  "p_approve": probability, "features": features,
                  "distance": distance, "repeated_answer": repeated,
                  "reason": policy + ("_stop" if stopping else "_continue"),
                  "critic": critic_record, "physical_head_calls": int(head_needed),
                  "physical_critic_calls": int(score_needed)}
        json.dumps(result, allow_nan=False)
        return result


class Runtime:
    """The original pinned 1.5B writer with truthful trained-head receipts."""

    def __init__(self, output, head, timeout=900):
        from .runtime import Runtime as OriginalRuntime
        self.base = OriginalRuntime(output, timeout=timeout)
        self.head = head

    def __enter__(self):
        from .hotpot import _atomic_json
        try:
            agents, previous_gate = self.base.__enter__()
            gate = Gate(critic=previous_gate.critic, head=self.head)
            path = self.base.output / "runtime.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["controller"] = gate.provenance
            _atomic_json(path, metadata)
            return agents, gate
        except BaseException:
            self.base.close()
            raise

    def __exit__(self, *args):
        self.base.close()
