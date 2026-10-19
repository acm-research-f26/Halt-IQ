"""Verify this completed review bundle offline; never load models or call APIs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from haltiq.metrics import hotpot_evaluate
from pc_strands.data import load_tasks


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def verify_data():
    files = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    if any(p.is_symlink() for p in files):
        raise ValueError("Bundle may not contain symbolic links")
    if sum(p.stat().st_size for p in files) > 25 * 1024 * 1024:
        raise ValueError("Unexpected bundle size")
    forbidden = re.compile(r"(?i)(?<![A-Za-z])[A-Z]:[\\/]+|/Users/|/home/|apikey_[A-Za-z0-9]|Bearer\s+[A-Za-z0-9]|[\"'](?:access_token|api_key|authorization)[\"']\s*:")
    for path in files:
        if path.suffix in {".json", ".jsonl", ".md", ".txt", ".bib"}:
            if forbidden.search(path.read_text(encoding="utf-8")):
                raise ValueError("Private machine path or credential pattern: " + path.name)
    fresh = load_tasks(ROOT / "data/fresh20-evaluator-only.jsonl")
    train = load_tasks(ROOT / "data/train90-evaluator-only.jsonl")
    assert len(fresh) == 20 and len(train) == 90
    assert Counter(t.partition for t in train) == {"fit": 60, "tune": 15, "calibration": 15}
    assert not ({t.id for t in fresh} & {t.id for t in train})
    assert Counter(t.category for t in fresh) == {"bridge": 16, "comparison": 4}
    assert all(set(t.public()) == {"question", "evidence"} for t in fresh + train)
    tasks = {t.id: t for t in fresh}
    trajectories = json_read("data/fresh20-writer-trajectories.json")["data"]
    assert [r["task_id"] for r in trajectories] == [t.id for t in fresh]
    repeated = 0
    for trajectory in trajectories:
        rounds = trajectory["rounds"]
        assert len(rounds) == 10 and trajectory["status"] == "completed"
        assert [r["round"] for r in rounds] == list(range(1, 11))
        repeated += len({r["draft"] for r in rounds}) == 1
        task = tasks[trajectory["task_id"]]
        for rnd in rounds:
            assert "response" not in rnd["call"]
            score = hotpot_evaluate(rnd["draft"], task.answers)
            assert score["exact_match"] == rnd["exact_match"]
            assert abs(score["token_f1"] - rnd["token_f1"]) < 1e-12
        assert len({r["exact_match"] for r in rounds}) == 1
    assert repeated == 18
    fixed_reference = {}
    for arm in ("untuned", "trained", "jev"):
        path = ROOT / f"exports/{arm}-common-schema-results.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        counts = Counter(r["policy"] for r in rows)
        assert len(rows) == 140 and len(counts) == 7 and set(counts.values()) == {20}
        for policy in counts:
            subset = [r for r in rows if r["policy"] == policy]
            assert {r["question_id"] for r in subset} == set(tasks)
            assert sum(r["exact_match"] for r in subset) == 10
            assert abs(sum(r["token_f1"] for r in subset) / 20 - .54) < 1e-12
            for row in subset:
                score = hotpot_evaluate(row["final_answer"], tasks[row["question_id"]].answers)
                assert score["exact_match"] == row["exact_match"]
                assert abs(score["token_f1"] - row["token_f1"]) < 1e-12
            if policy.startswith("fixed-"):
                values = [{k: v for k, v in row.items() if k != "source"} for row in subset]
                if policy in fixed_reference:
                    assert values == fixed_reference[policy]
                else:
                    fixed_reference[policy] = values
    primary = json_read("analyses/primary-paired-analysis.json")["data"]
    assert primary["planned_n"] == 20 and len(primary["policies"]) == 13
    assert primary["planned_question_ids"] == [t.id for t in fresh]
    for policy in primary["policies"].values():
        assert policy["correct"] == 10 and policy["failed_or_missing"] == 0
        assert abs(policy["token_f1"] - .54) < 1e-12
    for filename, expected in json_read("protocol/source-and-writer-settings.json")["frozen_15_source_sha256"].items():
        assert digest(ROOT.parents[1] / "pc_strands" / filename) == expected, filename
    selected_head = ROOT / "checkpoints/selected-head/head.safetensors"
    assert selected_head.stat().st_size == 12605896
    assert digest(selected_head) == "d95b640d54ccc004cc136dfbb38d896e942f299d972bed555bf07f7ab648ccf5"
    with selected_head.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < header_size < 100000
        header = json.loads(stream.read(header_size))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    assert len(tensors) == 6 and all(tensor["dtype"] == "F32" for tensor in tensors)
    assert sum(math.prod(tensor["shape"]) for tensor in tensors) == 3151360
    selected_config = ROOT / "checkpoints/selected-head/strands_decider_config.json"
    assert digest(selected_config) == "c8524a28dd0b38acbfd9462f19a03d576cea0533ae148a3b3382c00a32c729d1"
    assert json.loads(selected_config.read_text())["temperature_by_kind"]["noul"] == 1.4266376751158028
    for path in files:
        if path.suffix in {".safetensors", ".bin", ".pt", ".pth"} and path != selected_head:
            raise ValueError("Only the exact selected trained head is allowed in this bundle")
        if path.suffix in {".sqlite", ".db"}:
            raise ValueError("Live budget ledger must remain external")
    return {"status": "PASS", "fresh_questions": 20, "train_partitions": {"fit": 60, "tune": 15, "calibration": 15},
            "writer_rounds": 200, "repeated_answers_all_ten": repeated, "common_schema_rows": 420,
            "independent_question_n": 20, "primary_policies": 13, "frozen_source_files": 15,
            "selected_head_parameters": 3151360, "selected_head_exact_hash_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Explicitly refresh bundle hashes after reviewed artifact additions")
    parser.add_argument("--as-of", help="Explicit ISO8601 timestamp required when writing a manifest")
    args = parser.parse_args()
    result = verify_data()
    manifest_path = ROOT / "artifact-manifest.json"
    if args.write_manifest:
        if not args.as_of or not re.search(r"[+-]\d\d:\d\d$|Z$", args.as_of):
            parser.error("--write-manifest requires --as-of with explicit UTC offset")
        files = {p.relative_to(ROOT).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(ROOT.rglob("*")) if p.is_file() and p != manifest_path and "__pycache__" not in p.parts}
        manifest_path.write_text(json.dumps({"scope": "GitHub review bundle; original evidence SHA differs from sanitized export SHA",
            "as_of": args.as_of, "manifest_excludes_itself": True, "files": files}, indent=2) + "\n", encoding="utf-8")
    manifest = json_read("artifact-manifest.json")
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p != manifest_path and "__pycache__" not in p.parts}
    assert actual == set(manifest["files"]), "Bundle file inventory changed"
    for name, entry in manifest["files"].items():
        path = ROOT / name
        assert path.is_relative_to(ROOT) and path.is_file()
        assert digest(path) == entry["sha256"] and path.stat().st_size == entry["bytes"], name
    result["hashed_files"] = len(actual)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
