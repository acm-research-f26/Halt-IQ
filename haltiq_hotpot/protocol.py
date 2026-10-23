"""Answer-blind cohort selection and the partner's public evidence formats."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from pc_strands.data import question_key
from pc_strands.metrics import evaluate

PARTNER_REVISION = "d0688aed897c18d45fea8aacf040896ab06c86ee"
PARTNER_CSV_SHA256 = "5cfad9b8b3e31e21138aa0fd6867491f3abdc25b5e42cfa847eaa569f789fbe8"
DATASET_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
DATASET_SHA256 = "674e6c8b4e8c7e020ddd59b9b9b801966c663ae170d4e177a214b8c8c6439e7e"
SCOPE = "partner_shared_first_draft_development_method_check"
INSTRUCTIONS = "Is the draft answer correct and fully supported by the evidence?"
ARMS = ("baseline_3000", "minilm_top3")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reconstruct_rows(partner_rows, source_rows):
    """Join the whole shared pool; CSV context/labels/probabilities are ignored."""
    raw = {}
    for row in source_rows:
        key = question_key(row["question"])
        if key in raw:
            raise ValueError("Ambiguous normalized question in official dataset")
        raw[key] = row
    result, seen = [], set()
    for shared in partner_rows:
        source = raw.get(question_key(shared["question"]))
        if source is None or source["_id"] != shared["id"]:
            raise ValueError("Partner question/ID does not uniquely match the official row")
        if shared["question"] != source["question"] or source["_id"] in seen:
            raise ValueError("Changed question text or duplicate partner ID")
        if source["type"] not in {"bridge", "comparison"}:
            raise ValueError("Unknown question type")
        paragraphs = [{"title": title, "sentences": list(sentences)}
                      for title, sentences in source["context"]]
        if not paragraphs or any(not p["sentences"] for p in paragraphs):
            raise ValueError("Missing complete public paragraphs")
        if not isinstance(shared["draft"], str) or not shared["draft"].strip():
            raise ValueError("Missing saved first draft")
        seen.add(source["_id"])
        result.append({"id": source["_id"], "type": source["type"],
            "public": {"question": source["question"], "draft": shared["draft"],
                       "paragraphs": paragraphs},
            "evaluator": {"answer": source["answer"],
                "supporting_titles": list(dict.fromkeys(title for title, _ in source["supporting_facts"]))}})
    return result


def select_balanced(rows, *, per_type=10, seed=42):
    if per_type != 10 or seed != 42:
        raise ValueError("This first pilot freezes ten questions of each type, seed42")
    def rank(row):
        return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()
    chosen = []
    for category in ("bridge", "comparison"):
        candidates = [row for row in rows if row["type"] == category]
        if len(candidates) < per_type:
            raise ValueError("Insufficient question-type coverage")
        chosen.extend(sorted(candidates, key=rank)[:per_type])
    # Sorting reads only ID/type. No draft quality, labels or critic outputs rank rows.
    return sorted(chosen, key=rank)


def prepare_cohort(csv_path, dataset_path):
    if digest(csv_path) != PARTNER_CSV_SHA256 or digest(dataset_path) != DATASET_SHA256:
        raise ValueError("Partner CSV or official dataset differs from its pinned source")
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        shared = list(csv.DictReader(stream))
    sources = json.loads(Path(dataset_path).read_text(encoding="utf-8-sig"))
    if len(shared) != 220 or len(sources) != 7405:
        raise ValueError("Unexpected source cohort size")
    rows = reconstruct_rows(shared, sources)
    counts = {kind: sum(row["type"] == kind for row in rows) for kind in ("bridge", "comparison")}
    if counts != {"bridge": 176, "comparison": 44}:
        raise ValueError("Partner pool type counts changed")
    selected = select_balanced(rows)
    return {"schema": "haltiq-hotpot-partner-pilot-v1", "scope": SCOPE,
        "partner_revision": PARTNER_REVISION, "partner_csv_sha256": PARTNER_CSV_SHA256,
        "dataset_revision": DATASET_REVISION, "dataset_sha256": DATASET_SHA256,
        "selection": {"answer_blind": True, "seed": 42, "rank": "sha256('42:'+official_id)",
            "per_type": 10, "source_pool_counts": counts, "model_scoring_started": False},
        "instructions": INSTRUCTIONS, "criteria": None, "threshold": .8,
        "arms": list(ARMS), "cases": selected}


def validate_cohort(cohort):
    if (cohort.get("schema") != "haltiq-hotpot-partner-pilot-v1" or cohort.get("scope") != SCOPE
            or cohort.get("instructions") != INSTRUCTIONS or cohort.get("criteria") is not None
            or cohort.get("threshold") != .8 or cohort.get("arms") != list(ARMS)
            or cohort.get("partner_revision") != PARTNER_REVISION
            or cohort.get("partner_csv_sha256") != PARTNER_CSV_SHA256
            or cohort.get("dataset_revision") != DATASET_REVISION
            or cohort.get("dataset_sha256") != DATASET_SHA256):
        raise ValueError("Pilot protocol/source identity changed")
    selection = cohort.get("selection", {})
    if selection != {"answer_blind": True, "seed": 42, "rank": "sha256('42:'+official_id)",
            "per_type": 10, "source_pool_counts": {"bridge": 176, "comparison": 44},
            "model_scoring_started": False}:
        raise ValueError("Selection receipt does not match the predeclared protocol")
    cases = cohort.get("cases", [])
    if (len(cases) != 20 or len({c["id"] for c in cases}) != 20
            or len({question_key(c["public"]["question"]) for c in cases}) != 20
            or any(sum(c["type"] == kind for c in cases) != 10 for kind in ("bridge", "comparison"))):
        raise ValueError("Exactly20 unique type-balanced cases required")
    for case in cases:
        if set(case) != {"id", "type", "public", "evaluator"}:
            raise ValueError("Unexpected cohort fields")
        public = case["public"]
        if set(public) != {"question", "draft", "paragraphs"}:
            raise ValueError("Evaluator metadata must remain outside public state")
        if not all(isinstance(public[k], str) and public[k].strip() for k in ("question", "draft")):
            raise ValueError("Empty public question/draft")
        if not public["paragraphs"] or any(set(p) != {"title", "sentences"}
                or not isinstance(p["title"], str) or not isinstance(p["sentences"], list)
                or not p["sentences"] or not all(isinstance(s, str) for s in p["sentences"])
                for p in public["paragraphs"]):
            raise ValueError("Invalid public paragraphs")
        if (set(case["evaluator"]) != {"answer", "supporting_titles"}
                or not isinstance(case["evaluator"]["answer"], str)
                or not isinstance(case["evaluator"]["supporting_titles"], list)):
            raise ValueError("Invalid evaluator-only references")
    return cohort


def baseline_evidence(public):
    return " ".join(sentence for p in public["paragraphs"] for sentence in p["sentences"])[:3000]


def retrieval_pairs(public):
    query = public["question"] + " " + public["draft"]
    return [(query, p["title"] + ": " + " ".join(p["sentences"])) for p in public["paragraphs"]]


def ranked_evidence(public, scores):
    if len(scores) != len(public["paragraphs"]):
        raise ValueError("Every paragraph needs one retrieval score")
    import math
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Non-finite retrieval score")
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
    paragraphs = retrieval_pairs(public)
    return "\n".join(paragraphs[i][1] for i in indices), indices


def state_text(public, evidence):
    # The explicit allowlist prevents evaluator labels/IDs from entering the prompt.
    if set(public) != {"question", "draft", "paragraphs"}:
        raise ValueError("Unexpected public-state fields")
    return f"Question: {public['question']}\nEvidence: {evidence}\nDraft answer: {public['draft']}"


def writer_metrics(case):
    return evaluate(case["public"]["draft"], [case["evaluator"]["answer"]])
