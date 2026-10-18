"""Official-normalized answer metrics and question-paired descriptive intervals."""
from collections import Counter
import math
import random
import re
import string


def normalize(text: str) -> str:
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text.lower().translate(
        str.maketrans("", "", string.punctuation))).split())


def evaluate(answer: str, references: list[str]) -> dict:
    answer = normalize(answer)
    em, f1 = False, 0.0
    special = {"yes", "no", "noanswer"}
    for reference in references:
        target = normalize(reference)
        em = em or answer == target
        if answer != target and (answer in special or target in special):
            continue
        a, b = answer.split(), target.split()
        common = sum((Counter(a) & Counter(b)).values())
        if common:
            f1 = max(f1, 2 * common / (len(a) + len(b)))
    return {"exact_match": em, "token_f1": f1}


def _quantile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p
    low, high = math.floor(k), math.ceil(k)
    return values[low] + (values[high] - values[low]) * (k - low)


def paired_delta(candidate, baseline, ids, key, *, draws=5000, seed=42):
    """Missing/failed quality rows are zero; independent unit is a question."""
    def value(rows, identifier):
        row = rows.get(identifier)
        return float(row[key]) if row and row["status"] == "completed" else 0.0
    differences = [value(candidate, i) - value(baseline, i) for i in ids]
    if not differences:
        raise ValueError("Empty planned cohort")
    rng = random.Random(seed)
    means = [sum(rng.choice(differences) for _ in differences) / len(differences)
             for _ in range(draws)]
    return {"candidate_minus_baseline": sum(differences) / len(differences),
            "bootstrap95": [_quantile(means, .025), _quantile(means, .975)],
            "planned_n": len(ids), "complete_pair_n": sum(
                i in candidate and i in baseline and candidate[i]["status"] == baseline[i]["status"] == "completed"
                for i in ids), "draws": draws, "seed": seed, "rng": "python.random.Random"}
