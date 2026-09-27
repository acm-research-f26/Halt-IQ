"""Pinned, checksum-verified HotpotQA mirror download; no model calls."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


HF_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HF_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/" + HF_REVISION +
          "/distractor/validation-00000-of-00001.parquet")
PARQUET_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
PARQUET_BYTES = 27_452_575


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_hotpot(cache_dir: Path, progress=print) -> tuple[Path, dict]:
    """Return canonical upstream-format JSON and exact download provenance.

    Only preparation needs pyarrow. Existing prepared tasks run with stdlib.
    The immutable revision and LFS hash are pinned, rather than following main.
    """
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except ImportError:
        raise ValueError('Preparing the benchmark requires pyarrow. Install the data extra: python -m pip install -e ".[data]"') from None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "hotpotqa-distractor-validation.parquet"
    if not cached.exists():
        progress("Downloading the pinned HotpotQA validation mirror (27.5 MB)...", flush=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as target:
                temp_path = Path(target.name)
                with urllib.request.urlopen(HF_URL, timeout=60) as response:
                    count = 0
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > PARQUET_BYTES:
                            raise ValueError("Downloaded file exceeds the pinned source size")
                        target.write(chunk)
            if temp_path.stat().st_size != PARQUET_BYTES or sha256(temp_path) != PARQUET_SHA256:
                raise ValueError("Downloaded HotpotQA checksum or size does not match the pinned mirror")
            temp_path.replace(cached)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    if cached.stat().st_size != PARQUET_BYTES or sha256(cached) != PARQUET_SHA256:
        raise ValueError(f"Cached HotpotQA checksum mismatch: {cached}; use a new cache directory")
    rows = parquet.read_table(cached).to_pylist()
    if len(rows) != 7405:
        raise ValueError("Unexpected row count in pinned HotpotQA validation release")
    converted = []
    for row in rows:
        context, facts = row["context"], row["supporting_facts"]
        if len(context["title"]) != len(context["sentences"]) or len(facts["title"]) != len(facts["sent_id"]):
            raise ValueError("HotpotQA parallel column lengths differ")
        converted.append({"_id": row["id"], "question": row["question"], "answer": row["answer"],
                          "type": row["type"], "level": row["level"],
                          "context": [list(pair) for pair in zip(context["title"], context["sentences"])],
                          "supporting_facts": [list(pair) for pair in zip(facts["title"], facts["sent_id"])]})
    canonical = cache_dir / "hotpot_dev_distractor_v1.json"
    # Deterministic conversion; source data is never executed as code.
    encoded = (json.dumps(converted, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".json.part", delete=False) as target:
        target.write(encoded)
        temp_path = Path(target.name)
    try:
        temp_path.replace(canonical)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    metadata = {"url": HF_URL, "revision": HF_REVISION, "parquet_sha256": PARQUET_SHA256,
                "parquet_bytes": PARQUET_BYTES, "canonical_json_sha256": hashlib.sha256(encoded).hexdigest(),
                "rows": len(rows), "pyarrow_version": pyarrow.__version__,
                "context_count_distribution": dict(Counter(len(r["context"]) for r in converted)),
                "note": "Hugging Face mirror of the official development/validation distractor split; converted to upstream JSON field layout."}
    return canonical, metadata
