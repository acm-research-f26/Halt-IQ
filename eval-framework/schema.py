"""The common results schema: one JSON object per (question, policy), one per line.

Every adapter writes rows with these fields, and compare.py reads only these
fields, so any team's runs can be compared on one cost–quality scale.
"""

import json
import math
from pathlib import Path

# name: (required?, type, meaning)
FIELDS = {
    "question_id":   (True,  str,   "Stable question ID, e.g. 'hotpotqa:5ab833ad5542990e739ec86f'"),
    "split":         (True,  str,   "'dev' or 'test' (the 80 held-out questions); never mix them in one comparison"),
    "policy":        (True,  str,   "Stopping policy name, e.g. 'fixed-3', 'llm-critic', 'oracle'"),
    "source":        (True,  str,   "Where the row came from (run ID or file), for tracing"),
    "final_answer":  (True,  str,   "The answer the policy returns"),
    "rounds":        (True,  int,   "Writer drafts produced before stopping (1 = stopped after the first draft)"),
    "total_tokens":  (False, int,   "All model tokens spent on this question (input + output, writer + critic); null if unknown"),
    "seconds":       (True,  float, "Model time spent on this question (sum of call latencies)"),
    "exact_match":   (True,  bool,  "HotpotQA exact match after normalization"),
    "token_f1":      (True,  float, "HotpotQA token F1 in [0, 1]"),
    "info_score":    (False, float, "Optional information score (e.g. from Sid's or Sri's method); null if not computed"),
    "cost_usd":      (False, float, "API spend for this question; 0 for local models, null if unknown"),
}


def validate(row):
    """Raise ValueError if a row breaks the schema."""
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    for name, (required, kind, _) in FIELDS.items():
        value = row.get(name)
        if value is None:
            if required:
                raise ValueError(f"{name} is required")
            continue
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ValueError(f"{name} must be {kind.__name__}, got {value!r}")
        if kind is float and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if row["rounds"] < 1 or not 0 <= row["token_f1"] <= 1 or row["seconds"] < 0:
        raise ValueError("rounds >= 1, 0 <= token_f1 <= 1, seconds >= 0")
    return row


def write_rows(path, rows):
    rows = [validate(dict(row)) for row in rows]
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def read_rows(path):
    return [validate(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
