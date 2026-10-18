"""Question-disjoint binary temperature fitting; no torch required."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .decider import probability_true, sha256_file


def validate_rows(rows: Sequence[Mapping], *, partition: str = "calibration") -> list[dict]:
    if not rows:
        raise ValueError("no calibration rows")
    seen, validated = set(), []
    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("calibration requires one canonical row per unique question")
        if row.get("partition") != partition:
            raise ValueError(f"expected {partition} partition; refusing cross-partition calibration")
        label = row.get("label")
        if type(label) not in (int, bool) or label not in (0, 1):
            raise ValueError("label must be binary")
        logits = [float(x) for x in row.get("logits", [])]
        probability_true(logits)
        seen.add(qid)
        validated.append({"question_id": qid, "partition": partition, "label": int(label), "logits": logits})
    return validated


def binary_metrics(rows: Sequence[Mapping], temperature: float, *, threshold: float = 0.8,
                   bins: int = 10) -> dict:
    if not rows or not 0 <= threshold <= 1 or bins < 1:
        raise ValueError("nonempty rows, valid threshold and positive bin count required")
    nll, brier, correct, approved, correct_approved = 0.0, 0.0, 0, 0, 0
    groups = [[] for _ in range(bins)]
    for row in rows:
        p = probability_true(row["logits"], temperature)
        y = int(row["label"])
        # Stable logistic NLL remains valid even when probability rounds to 0 or 1.
        delta = (float(row["logits"][1]) - float(row["logits"][0])) / temperature
        nll += max(delta, 0.0) - y * delta + math.log1p(math.exp(-abs(delta)))
        brier += (p - y) ** 2
        correct += int((p > 0.5) == bool(y))
        approved += int(p >= threshold)
        correct_approved += int(p >= threshold and y == 1)
        groups[min(bins - 1, int(p * bins))].append((p, y))
    n = len(rows)
    reliability = [{"bin": i, "n": len(g),
                    "mean_probability": sum(p for p, _ in g) / len(g) if g else None,
                    "positive_fraction": sum(y for _, y in g) / len(g) if g else None}
                   for i, g in enumerate(groups)]
    ece = sum(len(g) / n * abs(sum(p for p, _ in g) / len(g) - sum(y for _, y in g) / len(g))
              for g in groups if g)
    return {"n": n, "nll": nll / n, "brier": brier / n, "accuracy": correct / n,
            "ece_probability_target": ece, "reliability": reliability,
            "approved": approved, "coverage": approved / n,
            "correct_approved": correct_approved,
            "approval_precision": correct_approved / approved if approved else None,
            "threshold": threshold, "temperature": temperature}


def fit_temperature(rows: Sequence[Mapping], *, lo: float = 0.25, hi: float = 6.0,
                    iterations: int = 64, threshold: float = 0.8) -> dict:
    rows = validate_rows(rows)
    if not 0 < lo < hi or iterations < 1:
        raise ValueError("invalid temperature search bounds")
    objective = lambda log_t: binary_metrics(rows, math.exp(log_t), threshold=threshold)["nll"]
    a, b = math.log(lo), math.log(hi)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = objective(d)
    candidates = [lo, hi, math.exp((a + b) / 2.0)]
    if lo <= 1.0 <= hi:
        candidates.append(1.0)
    temperature = min(candidates, key=lambda t: binary_metrics(rows, t)["nll"])
    return {"temperature": temperature, "objective": "calibration-question NLL",
            "bounds": [lo, hi], "question_ids": sorted(r["question_id"] for r in rows),
            "before": binary_metrics(rows, 1.0, threshold=threshold),
            "after": binary_metrics(rows, temperature, threshold=threshold),
            "status": "fitted_on_calibration_only_not_test"}


def calibrate_checkpoint(checkpoint: str | Path, rows: Sequence[Mapping], *,
                         forbidden_question_ids: Sequence[str] = ()) -> dict:
    rows = validate_rows(rows)
    directory = Path(checkpoint)
    config_path = directory / "strands_decider_config.json"
    if not (directory / "training-proof.json").is_file():
        raise ValueError("refusing to mutate an original checkpoint; use a new trained/control-copy checkpoint")
    proof = json.loads((directory / "training-proof.json").read_text(encoding="utf-8"))
    forbidden = set(forbidden_question_ids) | set(proof.get("fit_question_ids", [])) | set(
        proof.get("tune_question_ids", []))
    if forbidden & {r["question_id"] for r in rows}:
        raise ValueError("calibration overlaps fit/tune/test question IDs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    head = directory / "head.safetensors"
    if not head.exists():
        head = directory / "slot_head.pt"
    before_hash = sha256_file(head)
    result = fit_temperature(rows)
    config["temperature"] = result["temperature"]
    config["temperature_by_kind"] = {"noul": result["temperature"]}
    from .control import atomic_json
    atomic_json(config_path, config)
    if sha256_file(head) != before_hash:
        raise RuntimeError("calibration unexpectedly changed head weights")
    result["checkpoint_head_sha256"] = before_hash
    (directory / "calibration-proof.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Optional NEW trained/control-copy checkpoint to update")
    args = parser.parse_args()
    rows = [json.loads(x) for x in Path(args.rows).read_text(encoding="utf-8").splitlines() if x.strip()]
    result = calibrate_checkpoint(args.checkpoint, rows) if args.checkpoint else fit_temperature(rows)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
