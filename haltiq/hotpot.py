"""Prepare reproducible, answer-blind HotpotQA distractor research subsets.

Only questions and the complete provided context become model inputs. Labels and
supporting-fact annotations are retained for offline evaluation, never selection.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


SOURCE_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOMEPAGE = "https://hotpotqa.github.io/"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def normalize_question(question: str) -> str:
    """Conservative duplicate key: Unicode, case, punctuation, and whitespace."""
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate(rows: object) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("HotpotQA source must be a nonempty JSON array")
    seen_ids = set()
    for index, row in enumerate(rows):
        where = f"HotpotQA row {index + 1}"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected an object")
        for field in ("_id", "question", "answer", "type", "level"):
            if not _nonempty_string(row.get(field)):
                raise ValueError(f"{where}: {field} must be a nonempty string")
        if not normalize_question(row["question"]):
            raise ValueError(f"{where}: question has no normalized content")
        if row["_id"] in seen_ids:
            raise ValueError(f"Duplicate HotpotQA source ID: {row['_id']}")
        seen_ids.add(row["_id"])
        if row["type"] not in {"bridge", "comparison"}:
            raise ValueError(f"{where}: type must be bridge or comparison")
        context = row.get("context")
        if not isinstance(context, list) or not 2 <= len(context) <= 10:
            raise ValueError(f"{where}: distractor context must contain 2 to 10 paragraphs")
        for paragraph in context:
            if (not isinstance(paragraph, list) or len(paragraph) != 2
                    or not _nonempty_string(paragraph[0])
                    or not isinstance(paragraph[1], list) or not paragraph[1]
                    or any(not isinstance(sentence, str) for sentence in paragraph[1])):
                raise ValueError(f"{where}: each context paragraph needs a title and sentence strings")
    return rows


def _annotation_warnings(row: dict) -> list[dict]:
    """Retain upstream annotation errors without conditioning selection on gold."""
    warnings = []
    paragraph_lengths = defaultdict(list)
    for title, sentences in row["context"]:
        paragraph_lengths[title].append(len(sentences))
    for title, lengths in paragraph_lengths.items():
        if len(lengths) > 1:
            warnings.append({"code": "duplicate_context_title", "title": title})
    supporting = row.get("supporting_facts")
    if not isinstance(supporting, list) or not supporting:
        warnings.append({"code": "missing_or_invalid_supporting_facts"})
        return warnings
    for index, fact in enumerate(supporting):
        if (not isinstance(fact, list) or len(fact) != 2
                or not _nonempty_string(fact[0]) or type(fact[1]) is not int):
            warnings.append({"code": "invalid_supporting_fact_shape", "fact_index": index})
        elif fact[0] not in paragraph_lengths:
            warnings.append({"code": "supporting_title_not_in_context", "fact_index": index})
        elif not any(0 <= fact[1] < length for length in paragraph_lengths[fact[0]]):
            warnings.append({"code": "supporting_sentence_out_of_range", "fact_index": index})
    return warnings


def _context_counts(rows: list[dict]) -> dict[str, int]:
    return {str(count): total for count, total in sorted(Counter(len(row["context"]) for row in rows).items())}


def _warning_summary(rows: list[dict]) -> dict:
    issues = [
        {"source_id": row["_id"], "warnings": warnings}
        for row in sorted(rows, key=lambda item: item["_id"])
        if (warnings := _annotation_warnings(row))
    ]
    return {
        "task_count": len(issues),
        "code_counts": dict(sorted(Counter(warning["code"] for item in issues for warning in item["warnings"]).items())),
        "details": issues,
    }


def _rank(seed: int, split: str, row: dict) -> tuple[str, str]:
    key = json.dumps([seed, split, row["_id"]], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(key).hexdigest(), row["_id"]


def _quotas(size: int, groups: dict[str, list[dict]]) -> dict[str, int]:
    """Largest-remainder proportional allocation without floating-point ties."""
    population = sum(len(rows) for rows in groups.values())
    counts = {key: size * len(rows) // population for key, rows in groups.items()}
    order = sorted(groups, key=lambda key: (-(size * len(groups[key]) % population), key))
    for key in order[:size - sum(counts.values())]:
        counts[key] += 1
    return counts


def _task(row: dict, split: str) -> dict:
    return {
        "id": "hotpotqa:" + row["_id"],
        "split": split,
        "category": row["type"],
        "question": row["question"],
        "evidence": [
            title + "\n" + "\n".join(f"[{index}] {sentence}" for index, sentence in enumerate(sentences))
            for title, sentences in row["context"]
        ],
        "answers": [row["answer"]],
        "benchmark": "hotpotqa-distractor",
        "metric": "hotpotqa",
        "metadata": {
            "source_id": row["_id"],
            "source_split": "official_dev",
            "type": row["type"],
            "level": row["level"],
            "supporting_facts": row.get("supporting_facts"),
            "annotation_warnings": _annotation_warnings(row),
            "context_titles": [title for title, _ in row["context"]],
            "context_paragraph_count": len(row["context"]),
        },
    }


def prepare_hotpot(
    source: Path,
    output_dir: Path,
    dev_size: int = 20,
    test_size: int = 80,
    seed: int = 42,
    source_url: str = SOURCE_URL,
) -> dict:
    """Validate, stratify, and save two internal splits from official dev data.

    Refuses existing output directories and insufficient unique questions.
    ``test`` is a local held-out split, never HotpotQA's hidden official test.
    """
    source, output_dir = Path(source), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    for name, size in (("dev_size", dev_size), ("test_size", test_size)):
        if type(size) is not int or size < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not _nonempty_string(source_url):
        raise ValueError("source_url must be a nonempty string")
    source_bytes = source.read_bytes()
    rows = _validate(json.loads(source_bytes))

    # Keep the smallest ID for each normalized question, independent of source
    # ordering. No answers, annotations, lengths, or model outcomes affect this.
    questions = {}
    exclusions = []
    for row in sorted(rows, key=lambda item: item["_id"]):
        normalized = normalize_question(row["question"])
        if normalized in questions:
            exclusions.append({"source_id": row["_id"], "kept_source_id": questions[normalized]["_id"]})
        else:
            questions[normalized] = row
    eligible = list(questions.values())
    if dev_size + test_size > len(eligible):
        raise ValueError(
            f"Requested {dev_size + test_size} tasks but only {len(eligible)} unique questions "
            f"remain after excluding {len(exclusions)} duplicate questions"
        )

    selected = {}
    remaining = eligible
    for split, size in (("dev", dev_size), ("test", test_size)):
        groups = defaultdict(list)
        for row in remaining:
            groups[row["type"]].append(row)
        quotas = _quotas(size, groups)
        chosen = []
        for category in sorted(groups):
            chosen.extend(sorted(groups[category], key=lambda row: _rank(seed, split, row))[:quotas[category]])
        selected[split] = sorted(chosen, key=lambda row: _rank(seed, split, row))
        chosen_ids = {row["_id"] for row in chosen}
        remaining = [row for row in remaining if row["_id"] not in chosen_ids]

    tasks = [_task(row, split) for split in ("dev", "test") for row in selected[split]]
    output_bytes = "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "benchmark": "hotpotqa-distractor",
        "source": {
            "url": source_url,
            "homepage": HOMEPAGE,
            "filename": source.name,
            "split": "official_dev",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "row_count": len(rows),
            "type_counts": dict(sorted(Counter(row["type"] for row in rows).items())),
            "context_paragraph_count_distribution": _context_counts(rows),
            "annotation_warnings": _warning_summary(rows),
            "license": "CC-BY-SA-4.0",
            "license_url": LICENSE_URL,
        },
        "parameters": {"dev_size": dev_size, "test_size": test_size, "seed": seed},
        "sampling": {
            "method": "Proportional largest-remainder type quotas; SHA-256(seed, split, source ID) ranking",
            "order": "dev first, then test from remaining unique questions",
            "question_normalization": "NFKC, casefold, punctuation to spaces, collapse whitespace",
            "duplicate_policy": "Keep lexicographically smallest source ID per normalized question",
            "eligible_count": len(eligible),
            "eligible_type_counts": dict(sorted(Counter(row["type"] for row in eligible).items())),
            "excluded_duplicate_question_count": len(exclusions),
            "excluded_duplicate_questions": exclusions,
            "selection_uses_gold_answers": False,
            "selection_uses_model_outcomes": False,
        },
        "splits": {
            split: {
                "count": len(selected[split]),
                "type_counts": dict(sorted(Counter(row["type"] for row in selected[split]).items())),
                "context_paragraph_count_distribution": _context_counts(selected[split]),
                "annotation_warnings": _warning_summary(selected[split]),
                "source_ids": [row["_id"] for row in selected[split]],
            }
            for split in ("dev", "test")
        },
        "output": {"filename": "tasks.jsonl", "sha256": hashlib.sha256(output_bytes).hexdigest()},
        "evaluation": {
            "metric": "hotpotqa",
            "scope": "Answer exact match and token F1 only; not supporting-fact or joint leaderboard scores",
            "test_split_is_official_test": False,
            "split_notice": "Both internal splits derive from official development data; test is locally held out.",
            "model_inputs": ["question", "evidence"],
            "gold_only_fields": ["answers", "metadata.supporting_facts", "metadata.annotation_warnings"],
            "context_policy": "All supplied paragraphs (up to 10) in original order; no truncation or gold filtering",
            "annotation_policy": "Supporting-fact issues are evaluator-only warnings; no rows are excluded based on gold annotations",
        },
        "derived_data_license": "CC-BY-SA-4.0",
    }
    # Validation and serialization finish before any output is created. mkdir
    # also refuses a directory created concurrently by another process.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "tasks.jsonl").write_bytes(output_bytes)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# HaltIQ HotpotQA distractor subset\n\n"
        f"Derived from [HotpotQA]({HOMEPAGE}) by Yang, Qi, Zhang, Bengio, Cohen, "
        "Salakhutdinov, and Manning (2018). See `CITATION.bib`.\n\n"
        f"Source: [{source.name}]({source_url}). Source and derived data are distributed under "
        f"[CC BY-SA 4.0]({LICENSE_URL}). The changes are deterministic sampling, duplicate-question "
        "exclusion, local splitting, and formatting as HaltIQ JSONL.\n\n"
        f"The {dev_size} `dev` and {test_size} `test` tasks both come from **official development data**. "
        "The `test` split is locally held out, **not the official hidden test set**. "
        "Tune the critic on dev; freeze settings before evaluating test.\n\n"
        "Sampling is proportional by question type (bridge/comparison), seeded and stable under source row "
        "reordering. Duplicate normalized questions keep their smallest source ID. The manifest records "
        "parameters, IDs, type and context-count distributions, excluded duplicates, and source/output SHA-256 hashes. "
        "Selection never uses gold answers or model outcomes.\n\n"
        "Every task preserves all supplied context paragraphs (up to 10), their titles, all sentences, and original "
        "order. Sentence numbers are zero-based. No gold supporting-fact filter or context truncation is applied. "
        "Only `question` and `evidence` enter writer/critic prompts. Answers and supporting-fact annotations "
        "are private evaluation data. Upstream annotation issues are recorded as evaluator-only warnings "
        "in metadata and the manifest; affected rows remain eligible. Empty sentence strings are preserved. "
        "No scripted mock answers or critic scores are included.\n\n"
        "Report answer exact match and token F1 using the HotpotQA answer convention. This answer-only "
        "experiment does not compute supporting-fact or joint leaderboard scores. A small development-derived "
        "subset supports a critic/halting pilot, not an official leaderboard claim.\n",
        encoding="utf-8",
    )
    (output_dir / "CITATION.bib").write_text(
        "@inproceedings{yang2018hotpotqa,\n"
        "  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},\n"
        "  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and "
        "Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},\n"
        "  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},\n"
        "  year={2018},\n"
        "  url={https://aclanthology.org/D18-1259/}\n"
        "}\n",
        encoding="utf-8",
    )
    return manifest
