"""Answer-only HotpotQA metrics, matching the official evaluation semantics.

Independent implementation; reference:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
No supporting-fact or joint leaderboard metrics are claimed.
"""

from collections import Counter
import re
import string


def hotpot_normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def hotpot_evaluate(answer: str, gold: list[str]) -> dict:
    candidate = hotpot_normalize(answer)
    exact, best_f1 = False, 0.0
    special = {"yes", "no", "noanswer"}
    for label in gold:
        target = hotpot_normalize(label)
        exact = exact or candidate == target
        if candidate != target and (candidate in special or target in special):
            continue
        predicted_tokens, reference_tokens = candidate.split(), target.split()
        common = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
        if common:
            best_f1 = max(best_f1, 2.0 * common / (len(predicted_tokens) + len(reference_tokens)))
    return {"exact_match": exact, "token_f1": best_f1}
