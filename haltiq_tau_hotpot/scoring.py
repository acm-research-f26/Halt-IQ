"""HotpotQA v1 metrics, adapted to an in-memory, failure-inclusive interface.

Copyright 2018 Zhilin Yang, Peng Qi, Saizheng Zhang. Apache License 2.0:
https://www.apache.org/licenses/LICENSE-2.0
Changes: stdlib-only imports, typed in-memory results and explicit absent output.
Source: hotpotqa/hotpot at fa3a36370899e1d85822de61e58c85ea19993154,
hotpot_evaluate_v1.py SHA256 d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0.
"""
from __future__ import annotations

from collections import Counter
import re
import string


OFFICIAL_REVISION = "fa3a36370899e1d85822de61e58c85ea19993154"
OFFICIAL_SHA256 = "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0"
OFFICIAL_URL = f"https://github.com/hotpotqa/hotpot/blob/{OFFICIAL_REVISION}/hotpot_evaluate_v1.py"
METRIC_KEYS = ("em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec", "sp_recall",
               "joint_em", "joint_f1", "joint_prec", "joint_recall")


def normalize_answer(answer: str) -> str:
    lowered = answer.lower()
    unpunctuated = "".join(char for char in lowered if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", unpunctuated).split())


def answer_scores(prediction: str, reference: str) -> dict:
    predicted, expected = normalize_answer(prediction), normalize_answer(reference)
    exact = float(predicted == expected)
    special = {"yes", "no", "noanswer"}
    if predicted != expected and (predicted in special or expected in special):
        return {"em": exact, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    left, right = predicted.split(), expected.split()
    common = sum((Counter(left) & Counter(right)).values())
    precision = common / len(left) if common else 0.0
    recall = common / len(right) if common else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": exact, "f1": f1, "prec": precision, "recall": recall}


def support_scores(prediction: list, reference: list) -> dict:
    left, right = set(map(tuple, prediction)), set(map(tuple, reference))
    common = len(left & right)
    precision = common / len(left) if left else 0.0
    recall = common / len(right) if right else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"sp_em": float(left == right), "sp_f1": f1, "sp_prec": precision, "sp_recall": recall}


def score_submission(submission: dict | None, gold: dict) -> dict:
    """Score one submitted answer; missing submission receives twelve zeros.

    Strict string EM is not an independent semantic judgment. Citation titles
    and sentence IDs use exact set membership, as in the official evaluator.
    """
    if submission is None:
        return dict.fromkeys(METRIC_KEYS, 0.0)
    answer = answer_scores(submission["answer"], gold["answer"])
    support = support_scores(submission["supporting_facts"], gold["supporting_facts"])
    precision = answer["prec"] * support["sp_prec"]
    recall = answer["recall"] * support["sp_recall"]
    joint_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**answer, **support, "joint_em": answer["em"] * support["sp_em"],
            "joint_f1": joint_f1, "joint_prec": precision, "joint_recall": recall}


score = score_submission
