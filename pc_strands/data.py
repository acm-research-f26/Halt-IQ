"""Question-grouped HotpotQA readers. Labels never enter public model state."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import string
import unicodedata


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def question_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    evidence: list[str]
    answers: list[str]
    category: str = "unknown"
    partition: str = "unspecified"

    def public(self, draft: str | None = None) -> dict:
        result = {"question": self.question, "evidence": list(self.evidence)}
        if draft is not None:
            result["draft"] = draft
        return result


def public_state(state: dict, *, draft_required=True) -> dict:
    """Explicit allowlist: never copy IDs, references or support labels."""
    keys = ("question", "evidence", "draft") if draft_required else ("question", "evidence")
    for key in ("question", "draft") if draft_required else ("question",):
        if not isinstance(state.get(key), str) or not state[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    evidence = state.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(not isinstance(s, str) for s in evidence):
        raise ValueError("evidence must be a nonempty list of strings")
    return {key: state[key] for key in keys}


def _paragraphs(context) -> list[str]:
    if isinstance(context, dict):
        pairs = list(zip(context["title"], context["sentences"], strict=True))
    else:
        pairs = context
    return [str(title) + "\n" + "\n".join(f"[{i}] {sentence}" for i, sentence in enumerate(sentences))
            for title, sentences in pairs]


def task_from_row(row: dict) -> Task:
    partition = row.get("local_partition", row.get("split", "unspecified"))
    source = row.get("source_row", row)
    identifier = source.get("id", source.get("_id"))
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Task lacks a source ID")
    identifier = identifier if identifier.startswith("hotpotqa:") else "hotpotqa:" + identifier
    evidence = source.get("evidence")
    if evidence is None:
        evidence = _paragraphs(source["context"])
    answers = source.get("answers", [source.get("answer")])
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) for a in answers):
        raise ValueError("Task requires evaluator-only string references")
    task = Task(identifier, source["question"], evidence, answers,
                source.get("category", source.get("type", "unknown")), partition)
    public_state(task.public(), draft_required=False)
    return task


def load_tasks(path: str | Path, *, partition=None, limit=None, ids=None) -> list[Task]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(s) for s in raw.splitlines() if s.strip()]
    tasks = [task_from_row(r) for r in rows]
    if partition:
        tasks = [t for t in tasks if t.partition == partition]
    if ids is not None:
        wanted = set(ids)
        tasks = [t for t in tasks if t.id in wanted]
        if {t.id for t in tasks} != wanted:
            raise ValueError("Requested IDs are absent from this partition")
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate source IDs")
    if len({question_key(t.question) for t in tasks}) != len(tasks):
        raise ValueError("Duplicate normalized questions")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("No questions selected")
    return tasks
