"""Owned, pinned Colab services and primary receipts for the November study.

Importing this module never loads models. GPU execution is explicit and is
performed only when a user starts one of the new Colab notebooks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from .hotpot import _atomic_json
from .server import MODEL, REVISION, Writer, stream_response


def correlate_response(output, body, response):
    """Bind a client attempt to its already-published physical response bytes."""
    metadata = body.get("metadata") or {}
    call_id = metadata.get("yash_call_id")
    if call_id is None:
        return None
    if (not isinstance(call_id, str) or not re.fullmatch(r"[0-9a-f]{32}", call_id)
            or metadata.get("yash_role") not in {"writer", "reviewer", "challenger", "qualification"}):
        raise ValueError("Invalid owned call metadata")
    response_id = response["id"]
    if not re.fullmatch(r"chatcmpl-[0-9a-f]{32}", response_id):
        raise ValueError("Invalid physical response identity")
    root = Path(output)
    primary = root / "requests" / (response_id + ".json")
    raw = primary.read_bytes()
    saved = json.loads(raw)
    if saved.get("id") != response_id or saved.get("request", {}).get("metadata") != metadata:
        raise ValueError("Physical request receipt does not match the client attempt")
    usage = response["usage"]
    if saved.get("usage") != usage:
        raise ValueError("Primary and HTTP response token usage disagree")
    record = {"call_id": call_id, "role": metadata["yash_role"],
              "response_id": response_id, "primary_receipt": "requests/" + primary.name,
              "primary_sha256": hashlib.sha256(raw).hexdigest(), "usage": usage,
              "finish_reason": response["choices"][0]["finish_reason"], "metadata": metadata}
    target = root / "by-client" / (call_id + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("A client attempt cannot be silently reused")
    _atomic_json(target, record)
    return record


class CorrelatedWriter(Writer):
    """The owned HTTP handler serializes synchronous completions in one process."""
    def complete(self, body):
        metadata = body.get("metadata") or {}
        if "yash_call_id" in metadata:
            call_id = metadata["yash_call_id"]
            if not isinstance(call_id, str) or not re.fullmatch(r"[0-9a-f]{32}", call_id):
                raise ValueError("Invalid client attempt ID")
            if metadata.get("yash_role") not in {"writer", "reviewer", "challenger", "qualification"}:
                raise ValueError("Invalid owned role refused before generation")
            if (self.output / "by-client" / (call_id + ".json")).exists():
                raise ValueError("Repeated client attempt refused before generation")
        response = super().complete(body)
        correlate_response(self.output, body, response)
        return response


def serve_writer(output, port=8001):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    writer = CorrelatedWriter(output, "cuda", max_context=16384)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {**writer.metadata, "client_correlation": "by-client/<call_id>.json"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        try:
            response = writer.complete(body)
        except Exception as exc:
            return JSONResponse({"error": {"type": type(exc).__name__, "message": str(exc)}}, status_code=400)
        if body.get("stream"):
            return StreamingResponse(stream_response(response), media_type="text/event-stream")
        return response

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


class RecordedCritic:
    """One durable record for every attempted verifier/reranker evaluation."""
    def __init__(self, critic, output):
        self.critic, self.output = critic, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.usage = []
        self.last_record = None
        self.context = {}
        self.provenance = {**getattr(critic, "provenance", {}),
                           "recording": "exclusive call identity; public input hashes; per-call elapsed time"}

    def score(self, question, passages, answer):
        started = time.monotonic()
        call_id = uuid.uuid4().hex
        if (self.output / (call_id + ".json")).exists():
            raise ValueError("Repeated critic identity refused before inference")
        record = {"call_id": call_id, "context": dict(self.context),
                  "public_input_sha256": hashlib.sha256(json.dumps(
                      [question, passages, answer], ensure_ascii=False).encode()).hexdigest(),
                  "status": "failed", "score": None}
        self.last_record = None
        try:
            score = self.critic.score(question, passages, answer)
            if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid verifier observation")
            record.update(status="completed", score=float(score),
                          critic=getattr(self.critic, "last_record", None))
            self.last_record = {**(getattr(self.critic, "last_record", None) or {}),
                                "call_id": call_id}
            return float(score)
        except BaseException as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            self.usage.append(record)
            _atomic_json(self.output / (call_id + ".json"), record)


class Runtime:
    def __init__(self, output, model_variant="1p5b", timeout=1800, qualify=True):
        if model_variant != "1p5b":
            raise ValueError("This frozen primary/replication protocol uses only the pinned 1.5B writer")
        self.output = Path(output)
        self.timeout, self.do_qualify = timeout, qualify
        self.writer = self.critic_server = self.log = None
        self.agents = self.critic = None

    def start(self):
        import torch
        from importlib.metadata import version
        from .core import Agents
        from .critic import Critic, start_critic
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU in this new Yash Colab, then rerun the setup")
        for port in (8000, 8001):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Port {port} is occupied; no existing service was changed")
        self.output.mkdir(parents=True, exist_ok=True)
        self.critic_server = start_critic(self.output / "critic-cache", dtype="float16",
            log_path=self.output / "critic-server.log", timeout=self.timeout)
        self.log = (self.output / "writer-server.log").open("x", encoding="utf-8")
        self.writer = subprocess.Popen([sys.executable, "-u", "-m", "haltiq_pair.nov2_runtime",
            "serve-writer", "--output", str(self.output / "writer")],
            stdout=self.log, stderr=subprocess.STDOUT)
        end = time.monotonic() + self.timeout
        metadata = None
        while time.monotonic() < end:
            if self.writer.poll() is not None:
                raise RuntimeError("Owned writer startup failed; inspect writer-server.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=3) as response:
                    metadata = json.load(response)
                break
            except OSError:
                time.sleep(1)
        if not metadata:
            raise TimeoutError("Writer startup did not finish within the declared setup timeout")
        if (metadata.get("model") != MODEL or metadata.get("revision") != REVISION
                or metadata.get("dtype") != "torch.float16"
                or metadata.get("client_correlation") != "by-client/<call_id>.json"):
            raise ValueError("Health does not identify the owned pinned FP16 writer")
        self.agents = Agents(max_tokens=512)
        self.agents.receipt_root = self.output / "writer" / "by-client"
        self.agents.request_receipt_root = self.output / "writer" / "requests"
        self.critic = RecordedCritic(Critic(endpoint=self.critic_server.endpoint,
            ranker_device="cpu"), self.output / "critic-requests")
        self.provenance = {"writer": metadata, "critic_server": self.critic_server.provenance,
            "gpu": torch.cuda.get_device_name(), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "python": sys.version, "packages": {name: version(name) for name in
                ("torch", "transformers", "strands-agents", "strands-decider", "peft", "datasets")},
            "qualification_is_benchmark_evidence": False}
        _atomic_json(self.output / "runtime.json", self.provenance)
        if self.do_qualify:
            self.qualify()
        return self.agents, self.critic

    def qualify(self):
        from .hotpot import PublicQuestion
        from .nov2_orchestration import RoleAgents, Orchestrator
        agents = RoleAgents.from_agents(self.agents)
        manager = Orchestrator(agents, self.critic, max_drafts=3, format_corrections=1,
                               max_agent_calls=10)
        public = PublicQuestion("0" * 24, "What is the capital of France?", (
            ("France", ("Paris is the capital of France.",)),
            ("Germany", ("Berlin is the capital of Germany.",))))
        self.critic.context = {"phase": "engineering_qualification", "not_benchmark": True}
        trace = []
        state = manager.begin(public, trace=trace.append)
        if "INVOKE" not in manager.valid_actions(state):
            raise RuntimeError("Synthetic role qualification cannot invoke the challenger")
        revised = manager.transition(state, "INVOKE", trace=trace.append)
        roles = {item.get("role") for item in agents.usage if item.get("status") == "completed"}
        if not {"writer", "reviewer", "challenger"} <= roles:
            raise RuntimeError("Qualification did not execute all three agent roles")
        receipt = {"status": "PASS", "kind": "invented engineering input; not benchmark accuracy",
            "real_three_role_interaction": True, "initial": state.to_dict(), "after_invoke": revised.to_dict(),
            "trace": trace, "role_usage": agents.usage, "critic_usage": self.critic.usage,
            "context_limit": "16384 tokens; writer rejects overflow without truncation",
            "long_context_generation_probe_performed": False}
        _atomic_json(self.output / "qualification.json", receipt)
        self.critic.context = {}
        # Engineering calls remain in immutable receipts but do not enter study deltas.

    def close(self):
        try:
            if self.writer is not None and self.writer.poll() is None:
                self.writer.terminate()
                try:
                    self.writer.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.writer.kill()
                    self.writer.wait(timeout=10)
        finally:
            try:
                if self.log is not None:
                    self.log.close()
            finally:
                if self.critic_server is not None:
                    self.critic_server.close()

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve-writer"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    serve_writer(args.output, args.port)


if __name__ == "__main__":
    main()
