"""Predeclared secondary revision selector using only visited decision scores.

This changes the fallback answer at a round cap. It adds no model calls and never
uses an evaluator label to choose a candidate. It does not guarantee improvement.
"""
from copy import deepcopy
import math

from .metrics import evaluate


def derive_retained_episode(episode: dict, trajectory: dict, references: list[str]) -> dict:
    """Keep the highest-confidence visited candidate, retaining full prefix cost.

    The original policy still determines when to stop and how much work is done.
    Only its already-visited candidates can be retained. Failed episodes remain
    failures; this selector cannot rescue an overflow or missing decision.
    """
    if episode["task_id"] != trajectory["task_id"]:
        raise ValueError("Episode and trajectory refer to different questions")
    result = deepcopy(episode)
    result.update(policy=episode["policy"] + "-retainbest", selected_draft_round=None,
                  retained_earlier_draft=False, selection_rule="highest p_true among visited drafts; earliest tie")
    if episode["status"] != "completed":
        result.update(exact_match=False, token_f1=0.0)
        return result
    count = episode["rounds"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("A completed episode needs its actual visited draft count")
    visible = [r for r in trajectory["rounds"] if r["round"] <= count]
    rounds = {r["round"]: r for r in visible}
    if len(visible) != count or len(rounds) != count or set(rounds) != set(range(1, count + 1)):
        raise ValueError("Visited writer prefix is incomplete")
    decisions = [d for d in episode.get("decisions", []) if d["round"] <= count]
    if [d["round"] for d in decisions] != list(range(1, count + 1)):
        raise ValueError("Each visited draft needs one actual decision score")
    if episode.get("decision_calls") != len(decisions):
        raise ValueError("Actual charged decision count differs from the visited prefix")
    for d in decisions:
        p = d["score"]["p_true"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid visited candidate probability")
        if rounds[d["round"]].get("status") != "completed":
            raise ValueError("Visited writer candidate failed")
    # Labels, answers and future scores are deliberately absent from this key.
    best = max(decisions, key=lambda d: (d["score"]["p_true"], -d["round"]))
    selected = best["round"]
    answer = rounds[selected]["draft"]
    result.update(final_answer=answer, selected_draft_round=selected,
                  retained_earlier_draft=selected < count, decisions=decisions,
                  **evaluate(answer, references))
    # rounds, attempts, tokens and time retain the entire executed prefix.
    return result
