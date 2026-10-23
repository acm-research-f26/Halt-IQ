"""Fsynced, single-writer physical-call accounting with no uncertain retries."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def durable_write(path, value):
    """Exclusive result first, fsync, then the journal may bind its bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class UncertainCall(RuntimeError):
    """A dispatched operation has no trustworthy terminal result; stop the worker."""


class CallJournal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "calls.jsonl"
        self.records = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                self.records.append(json.loads(line))
            if self.summary()["pending_or_uncertain"]:
                raise UncertainCall("Existing call journal is unresolved; reconcile it without redispatch.")
        self.next_id = sum(r["event"] == "pending" for r in self.records)

    def _append(self, record):
        record = {"at": datetime.now(timezone.utc).isoformat(), **record}
        with self.path.open("ab") as stream:
            stream.write(canonical(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)

    def call(self, role, request, operation):
        identifier = f"call-{self.next_id:06d}"
        self.next_id += 1
        self._append({"event": "pending", "id": identifier, "role": role,
                      "request_sha256": fingerprint(request)})
        started = time.monotonic()
        try:
            result = operation()
            result_path = self.directory / "responses" / (identifier + ".json")
            durable_write(result_path, result)
            self._append({"event": "completed", "id": identifier, "role": role,
                          "elapsed_seconds": time.monotonic() - started,
                          "response": result_path.relative_to(self.directory).as_posix(),
                          "response_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()})
            return result
        except BaseException as error:
            # No network/native operation is automatically retried. A response can
            # have completed even when transport, serialization or fsync failed.
            self._append({"event": "uncertain", "id": identifier, "role": role,
                          "elapsed_seconds": time.monotonic() - started,
                          "error_type": type(error).__name__})
            raise UncertainCall(f"{role} has an unresolved physical call: {identifier}") from error

    def summary(self):
        pending = {r["id"] for r in self.records if r["event"] == "pending"}
        completed = {r["id"] for r in self.records if r["event"] == "completed"}
        uncertain = {r["id"] for r in self.records if r["event"] == "uncertain"}
        by_role = {}
        for record in self.records:
            if record["event"] == "completed":
                by_role[record["role"]] = by_role.get(record["role"], 0) + 1
        return {"dispatched": len(pending), "completed": len(completed),
                "uncertain": len(uncertain), "pending_or_uncertain": bool(pending - completed or uncertain),
                "by_role": by_role}

    def verify(self):
        for record in self.records:
            if record["event"] == "completed":
                path = self.directory / record["response"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["response_sha256"]:
                    raise ValueError("Physical response bytes changed")
        return self.summary()
