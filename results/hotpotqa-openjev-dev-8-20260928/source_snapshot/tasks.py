"""Dataset loading and evaluation. Gold labels stay outside model inputs."""

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import unicodedata

from .metrics import hotpot_evaluate


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    category: str
    question: str
    evidence: list[str]
    answers: list[str]
    demo_drafts: list[str]
    demo_scores: list[dict]
    benchmark: str = "diagnostic"
    metric: str = "diagnostic"
    metadata: dict = field(default_factory=dict)

    def state(self, draft: str | None = None) -> dict:
        # Explicit allowlist: never serialize this dataclass into a model prompt.
        state = {"question": self.question, "evidence": self.evidence}
        if draft is not None:
            state["draft"] = draft
        return state

    def evaluate(self, answer: str) -> dict:
        if self.metric == "hotpotqa":
            return hotpot_evaluate(answer, self.answers)
        return evaluate(answer, self.answers)


def load_tasks(path: str | Path, split: str = "dev", limit: int | None = None) -> list[Task]:
    tasks = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        for field in ("id", "split", "category", "question"):
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                raise ValueError(f"Line {line_number}: {field} must be a nonempty string")
        for field in ("evidence", "answers"):
            if not isinstance(raw.get(field), list) or not raw[field] or any(
                not isinstance(v, str) or not v.strip() for v in raw[field]
            ):
                raise ValueError(f"Line {line_number}: {field} must be a nonempty list of strings")
        if raw["split"] not in {"dev", "test"}:
            raise ValueError(f"Line {line_number}: split must be dev or test")
        if raw["id"] in seen:
            raise ValueError(f"Duplicate task id: {raw['id']}")
        seen.add(raw["id"])
        benchmark, metric, metadata = raw.get("benchmark", "diagnostic"), raw.get("metric", "diagnostic"), raw.get("metadata", {})
        if not isinstance(benchmark, str) or not benchmark or metric not in {"diagnostic", "hotpotqa"} or not isinstance(metadata, dict):
            raise ValueError(f"Line {line_number}: invalid benchmark, metric, or metadata")
        task = Task(**{k: raw[k] for k in ("id", "split", "category", "question", "evidence", "answers")},
                    demo_drafts=raw.get("demo_drafts", []), demo_scores=raw.get("demo_scores", []),
                    benchmark=benchmark, metric=metric, metadata=metadata)
        if split == "all" or task.split == split:
            tasks.append(task)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError(f"No tasks in split {split!r}")
    return tasks


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("\N{MINUS SIGN}", "-")
    # Retain signs and decimal points in numeric answers: -3 and 3 are different.
    text = re.sub(r"(?<!\d)\.(?!\d)|(?<=\d)\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s.\-+]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def evaluate(answer: str, gold: list[str]) -> dict:
    candidate = normalize(answer)
    exact = any(candidate == normalize(g) for g in gold)
    best_f1 = 0.0
    for label in gold:
        target = normalize(label)
        if candidate == target:
            best_f1 = 1.0
            break
        overlap = sum((Counter(candidate.split()) & Counter(target.split())).values())
        if overlap:
            precision = overlap / len(candidate.split())
            recall = overlap / len(target.split())
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return {"exact_match": exact, "token_f1": best_f1}
