"""Frozen, question-disjoint November study plans and failure-inclusive reports.

Selection sees only public IDs/type/level. Gold labels are hashed after selection
and are returned only to the evaluator by ``resolve_plan``. This module neither
starts inference nor treats a budget stop as approval.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from . import hotpot


SCHEMA = "yash-nov2-plan-v1"
PARTITIONS = ("train", "dev", "test")
DATASET = {"id": hotpot.DATASET_ID, "revision": hotpot.DATASET_REVISION,
           "configuration": "distractor", "split": "validation"}
_ID_KEYS = {"question_ids", "question_id", "ids", "id"}
_HOTPOT_ID = re.compile(r"[0-9a-f]{24}\Z")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Question IDs must be nonempty strings")
    return hotpot._bare_id(value.strip())


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite(value: Any, name: str, *, upper: float | None = None) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (upper is not None and value > upper)):
        raise ValueError(f"{name} must be finite and within its nonnegative bounds")
    return float(value)


def _index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Dataset rows must be mappings")
        qid = _id(row.get("id", row.get("_id")))
        if qid in indexed:
            raise ValueError(f"Duplicate dataset question ID {qid}")
        # Normalize the public identity once without mutating the input row.
        indexed[qid] = dict(row, id=qid)
    return indexed


def _strata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((str(row.get("type", "unknown")), str(row.get("level", "unknown")))
                     for row in rows)
    return [{"type": kind, "level": level, "count": count}
            for (kind, level), count in sorted(counts.items())]


def _freeze(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dataset = [hotpot.split_public_gold(row) for row in rows]
    return {"question_ids": [public.id for public, _ in dataset],
            "public_inputs_sha256": _digest([asdict(public) for public, _ in dataset]),
            "label_digest": _digest([asdict(gold) for _, gold in dataset]),
            "strata": _strata(rows)}


def build_plan(rows: Iterable[Mapping[str, Any]], excluded_ids: Iterable[str],
               train_count: int = 20, dev_count: int = 10, test_count: int = 20,
               seed: int = 20261102) -> dict[str, Any]:
    """Freeze sequential proportional public-stratum samples, without gold selection.

    Counts may be zero for an unused partition. Reversing dataset input order or
    mutating gold answers cannot change selected IDs or public-input hashes.
    ``rows`` must come from the declared pinned dataset; no network is used here.
    """
    counts = [_count(value, name) for name, value in
              zip(PARTITIONS, (train_count, dev_count, test_count))]
    if not sum(counts):
        raise ValueError("At least one question must be planned")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    indexed = _index(rows)
    excluded = {_id(value) for value in excluded_ids}
    if len(set(indexed) - excluded) < sum(counts):
        raise ValueError("Not enough eligible questions for all disjoint partitions")
    unavailable = set(excluded)
    chosen = {}
    # Select every partition before reading any gold labels.
    for offset, (partition, count) in enumerate(zip(PARTITIONS, counts)):
        selected = hotpot.select_questions(indexed.values(), count, seed + offset, unavailable) if count else []
        chosen[partition] = selected
        unavailable.update(_id(row.get("id", row.get("_id"))) for row in selected)
    return {"schema": SCHEMA, "dataset": dict(DATASET), "seed": seed,
            "selection": "sequential whole-question proportional type/level SHA256 sampling; seed+partition offset; gold-blind",
            "excluded_ids": sorted(excluded),
            "partitions": {name: _freeze(chosen[name]) for name in PARTITIONS}}


def resolve_plan(plan: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> dict[str, list[tuple[hotpot.PublicQuestion, hotpot.Gold]]]:
    """Resolve exact frozen order; reject pin, identity, label or public-data drift."""
    if (plan.get("schema") != SCHEMA or plan.get("dataset") != DATASET
            or type(plan.get("seed")) is not int):
        raise ValueError("Unsupported plan or dataset pin")
    partitions = plan.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITIONS):
        raise ValueError("Plan requires exactly train, dev and test partitions")
    excluded = plan.get("excluded_ids")
    if not isinstance(excluded, list) or excluded != sorted({_id(value) for value in excluded}):
        raise ValueError("Plan exclusions must be unique canonical sorted IDs")
    indexed = _index(rows)
    seen = set(excluded)
    resolved = {}
    for name in PARTITIONS:
        frozen = partitions[name]
        if not isinstance(frozen, Mapping) or not isinstance(frozen.get("question_ids"), list):
            raise ValueError(f"Invalid {name} partition")
        ids = frozen["question_ids"]
        if any(_id(qid) != qid for qid in ids) or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Partition IDs must be canonical, unique and disjoint from all exclusions/partitions")
        missing = set(ids) - indexed.keys()
        if missing:
            raise ValueError(f"Frozen question IDs are missing: {sorted(missing)}")
        selected = [indexed[qid] for qid in ids]
        if dict(frozen) != _freeze(selected):
            raise ValueError(f"Frozen {name} public-input hash, label digest or strata changed")
        resolved[name] = [hotpot.split_public_gold(row) for row in selected]
        seen.update(ids)
    if not any(resolved.values()):
        raise ValueError("Plan contains no questions")
    return resolved


def _json_ids(value: Any) -> set[str]:
    """Only explicit ID-valued keys may contribute; never scan arbitrary strings."""
    ids = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _ID_KEYS:
                candidates = item if isinstance(item, list) else [item]
                for candidate in candidates:
                    if isinstance(candidate, str) and candidate.strip():
                        qid = _id(candidate)
                        if _HOTPOT_ID.fullmatch(qid):
                            ids.add(qid)
            if isinstance(item, (Mapping, list)):
                ids.update(_json_ids(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (Mapping, list)):
                ids.update(_json_ids(item))
    return ids


def collect_exclusions(repo_root: str | Path, extra_paths: Iterable[str | Path] = ()) -> set[str]:
    """Read teammate ID columns and explicitly supplied JSON/JSONL provenance IDs.

    No recursive directory search, metrics, old models or inferred string IDs are
    inputs. Supplied missing or unsupported paths fail closed. JSON IDs must be
    24 lowercase hexadecimal HotpotQA IDs, optionally with recognized prefixes.
    """
    root = Path(repo_root)
    excluded = set()
    for name in ("haltiq_sri_trajectory_strands.csv", "sri_rounds_with_strands.csv"):
        path = root / "data" / name
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("id"):
                        qid = _id(row["id"])
                        if not _HOTPOT_ID.fullmatch(qid):
                            raise ValueError(f"Invalid teammate question ID in {path}")
                        excluded.add(qid)
    path = root / "strands" / "dev20_ids.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                qid = _id(line)
                if not _HOTPOT_ID.fullmatch(qid):
                    raise ValueError(f"Invalid teammate question ID in {path}")
                excluded.add(qid)
    for item in extra_paths:
        path = Path(item)
        if path.suffix.lower() == ".json":
            excluded.update(_json_ids(json.loads(path.read_text(encoding="utf-8-sig"))))
        elif path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    excluded.update(_json_ids(json.loads(line)))
        else:
            raise ValueError("Additional exclusion paths must be explicit JSON or JSONL files")
    return excluded


def _validate_record(record: Mapping[str, Any], arms: set[str]) -> None:
    required = {"question_id", "arm", "status", "prediction", "metrics", "iterations", "agent_calls",
                "attempted_agent_calls", "total_tokens", "elapsed_seconds", "actions", "termination", "gate_approved"}
    if not isinstance(record, Mapping) or not required <= record.keys():
        raise ValueError("Episode record is missing required fields")
    if (_id(record["question_id"]) != record["question_id"]
            or not isinstance(record["arm"], str) or record["arm"] not in arms):
        raise ValueError("Episode must have a canonical question ID and declared arm")
    if not isinstance(record["status"], str) or record["status"] not in {"completed", "failed", "interrupted"}:
        raise ValueError("Unknown episode status")
    if record["prediction"] is not None and not isinstance(record["prediction"], Mapping):
        raise ValueError("Prediction must be a mapping or None")
    if record["status"] == "completed" and record["prediction"] is None:
        raise ValueError("Completed episodes require a prediction")
    metrics = record["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(hotpot.METRICS):
        raise ValueError("Each record requires all twelve HotpotQA metrics")
    for name, value in metrics.items():
        _finite(value, name, upper=1.0)
    _count(record["iterations"], "iterations")
    calls = record["agent_calls"]
    if not isinstance(calls, Mapping) or any(not isinstance(role, str) or not role for role in calls):
        raise ValueError("agent_calls must map nonempty role names to counts")
    for role, count in calls.items():
        _count(count, role + " calls")
    if _count(record["attempted_agent_calls"], "attempted_agent_calls") < sum(calls.values()):
        raise ValueError("Attempted calls cannot be fewer than completed role calls")
    if record["total_tokens"] is not None:
        _count(record["total_tokens"], "total_tokens")
    if "known_recorded_tokens" in record:
        _count(record["known_recorded_tokens"], "known_recorded_tokens")
    _finite(record["elapsed_seconds"], "elapsed_seconds")
    if (not isinstance(record["actions"], list)
            or any(not isinstance(action, str) or action not in {"STOP", "REVISE", "INVOKE"}
                   for action in record["actions"])):
        raise ValueError("actions must contain STOP, REVISE or INVOKE")
    if not isinstance(record["termination"], str) or not record["termination"]:
        raise ValueError("A termination reason is required")
    if type(record["gate_approved"]) is not bool:
        raise ValueError("gate_approved must be a boolean, never inferred from a cap")


def summarize(records: Sequence[Mapping[str, Any]], arms: Sequence[str], planned_count: int) -> dict[str, Any]:
    """Use all planned questions as denominator; do not silently drop failures.

    The runner must separately check membership against its frozen planned IDs.
    ``agent_calls`` are recorded completed role invocations, not independently
    reconciled physical requests. A missing token receipt (None), an explicitly
    incomplete journal, or pending attempts makes aggregate tokens unknown.
    """
    if _count(planned_count, "planned_count") < 1:
        raise ValueError("planned_count must be positive")
    if (not arms or any(not isinstance(arm, str) or not arm for arm in arms)
            or len(set(arms)) != len(arms)):
        raise ValueError("Arms must be unique nonempty names")
    seen = set()
    for record in records:
        _validate_record(record, set(arms))
        key = (record["question_id"], record["arm"])
        if key in seen:
            raise ValueError(f"Duplicate question-arm attempt {key}")
        seen.add(key)
    summaries = {}
    for arm in arms:
        rows = [record for record in records if record["arm"] == arm]
        if len(rows) > planned_count:
            raise ValueError("Recorded attempts exceed the planned denominator")
        statuses = Counter(row["status"] for row in rows)
        calls = Counter()
        for row in rows:
            calls.update(row["agent_calls"])
        known_tokens = sum(row.get("known_recorded_tokens", row["total_tokens"] or 0) for row in rows)
        journal_complete = len(rows) == planned_count and all(
            row["total_tokens"] is not None and row.get("journal_complete", True) is True for row in rows)
        tokens = sum(row["total_tokens"] for row in rows) if journal_complete else None
        elapsed = sum(row["elapsed_seconds"] for row in rows)
        iterations = sum(row["iterations"] for row in rows)
        summaries[arm] = {
            "planned": planned_count, "attempted": len(rows),
            "completed": statuses["completed"], "failed": statuses["failed"],
            "interrupted": statuses["interrupted"], "pending": planned_count - len(rows),
            "metrics": {name: sum(row["metrics"][name] for row in rows if row["status"] == "completed") / planned_count
                        for name in hotpot.METRICS},
            "iterations": iterations, "mean_writer_iterations": iterations / planned_count,
            "agent_calls": dict(sorted(calls.items())),
            "mean_role_calls": {role: count / planned_count for role, count in sorted(calls.items())},
            "total_calls": sum(calls.values()), "mean_total_calls": sum(calls.values()) / planned_count,
            "attempted_agent_calls": sum(row["attempted_agent_calls"] for row in rows),
            "total_tokens": tokens, "mean_total_tokens": tokens / planned_count if tokens is not None else None,
            "known_recorded_tokens": known_tokens, "journal_complete": journal_complete,
            "elapsed_seconds": elapsed, "mean_elapsed_seconds": elapsed / planned_count,
            "action_distribution": dict(sorted(Counter(action for row in rows for action in row["actions"]).items())),
            "challenger_calls": calls["challenger"],
            "termination_counts": dict(sorted(Counter(row["termination"] for row in rows).items())),
            "gate_approved": sum(row["status"] == "completed" and row["gate_approved"] for row in rows),
        }
    return {"planned_count": planned_count, "arms": summaries,
            "complete": all(value["pending"] == 0 for value in summaries.values()),
            "all_episodes_succeeded": all(value["completed"] == planned_count for value in summaries.values()),
            "denominator": "every planned question; failed, interrupted and pending outputs score zero",
            "means_scope": "recorded resources divided by planned questions; pending resources are not yet known",
            "call_scope": "completed role invocations; physical-response reconciliation is the runner's responsibility"}
