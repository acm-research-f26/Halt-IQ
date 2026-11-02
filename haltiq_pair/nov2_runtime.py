"""Owned, pinned Colab services and primary receipts for the November study.

Importing this module never loads models. GPU execution is explicit and is
performed only when a user starts one of the new Colab notebooks.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
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
from .nov2_orchestration import ROLE_FORMAT_VERSION, ROLE_JSON_SCHEMAS

LMFE_VERSION = "0.11.3"
LMFE_EFFECTIVE_CONFIG = {"max_json_array_length": 20, "max_consecutive_whitespaces": 12,
                         "force_json_field_order": False}

# Tokenizer bridge adapted from lm-format-enforcer v0.11.3's MIT-licensed
# integrations/transformers.py lines49-91. Its legacy tokenizer import does not
# exist in Transformers5.17, so this owned adapter uses the public core API.
LMFE_BRIDGE_LICENSE = '''MIT License
Copyright (c) 2023 Noam Gat
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''


def constraint_dependencies():
    """Fail closed before loading a model when the pinned public API is absent."""
    from importlib.metadata import version
    if version("lm-format-enforcer") != LMFE_VERSION:
        raise RuntimeError("Install the owned requirements-nov2-colab.txt exactly")
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    return JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


def constraint_provenance():
    return {"library": "lm-format-enforcer", "version": LMFE_VERSION,
        "schema_version": ROLE_FORMAT_VERSION, "schemas": deepcopy(ROLE_JSON_SCHEMAS),
        "schemas_sha256": hashlib.sha256(json.dumps(ROLE_JSON_SCHEMAS, sort_keys=True).encode()).hexdigest(),
        "effective_parser_config": dict(LMFE_EFFECTIVE_CONFIG),
        "parser_alphabet": "tokenizer-derived alphabet supplied before parser context initialization",
        "adapter": "owned public TokenEnforcer callback; v0.11.3 algorithm with no legacy Transformers import or model mutation",
        "source": "https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/integrations/transformers.py",
        "scope": "every owned writer/reviewer/challenger call in every arm",
        "fresh_parser_and_token_enforcer_per_request": True, "tokenizer_data_cached_per_writer": True,
        "unconstrained_fallback": False,
        "validation": "strict raw JSON types/keys and nonnegative integer indices after generation; no content repair",
        "limitation": "LMFE integer minimum is not token-enforced; strict postvalidation rejects negative indices. Public citation membership/answer correctness are not constrained.",
        "tokenization_limitation": "Some valid merged escaped-quote tokenizations are filtered; an alternative characterwise tokenization of the same escaped JSON passes. No claim that every valid encoding is admitted."}


def build_constraint_tokenizer_data(tokenizer):
    """Precompute the pinned library's reusable tokenizer tree, without a model."""
    _, _, data_type = constraint_dependencies()
    zero = tokenizer.encode("0")[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        after_zero = tokenizer.decode([zero, token_id])[1:]
        single = tokenizer.decode([token_id])
        regular.append((token_id, after_zero, len(after_zero) > len(single)))
    return data_type(regular, lambda ids: tokenizer.decode(ids).rstrip("\ufffd"),
                     tokenizer.eos_token_id, False, len(tokenizer))


def build_constraint_prefix(tokenizer_data, role):
    parser_type, enforcer_type, _ = constraint_dependencies()
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    if role not in ROLE_JSON_SCHEMAS:
        raise ValueError("Every constrained request needs a declared role")
    config = CharacterLevelParserConfig(alphabet=tokenizer_data.tokenizer_alphabet,
                                       **LMFE_EFFECTIVE_CONFIG)
    parser = parser_type(deepcopy(ROLE_JSON_SCHEMAS[role]), config=config)
    enforcer = enforcer_type(tokenizer_data, parser)
    # TokenEnforcer replaces parser.config; explicitly freeze the effective values
    # after construction rather than trust environment-controlled defaults.
    for name, value in LMFE_EFFECTIVE_CONFIG.items():
        setattr(parser.config, name, value)

    def prefix_allowed_tokens_fn(batch_id, sent):
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

    prefix_allowed_tokens_fn.token_enforcer = enforcer
    prefix_allowed_tokens_fn.role = role
    return prefix_allowed_tokens_fn


def _validate_role_json(text, role):
    def pairs(items):
        if len({key for key, _ in items}) != len(items):
            raise ValueError("Duplicate JSON keys are refused")
        return dict(items)

    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))

    def validate(item, schema):
        kind = schema["type"]
        if kind == "object":
            if not isinstance(item, dict) or set(item) != set(schema["required"]):
                raise ValueError("Generated object has missing or extra schema keys")
            for key, nested in schema["properties"].items():
                validate(item[key], nested)
        elif kind == "array":
            if not isinstance(item, list) or len(item) > schema.get("maxItems", 20):
                raise ValueError("Generated citation list is invalid")
            for nested in item:
                validate(nested, schema["items"])
        elif kind == "string":
            if not isinstance(item, str) or len(item) < schema.get("minLength", 0):
                raise ValueError("Generated string is invalid")
        elif kind == "integer":
            if type(item) is not int or item < schema.get("minimum", 0):
                raise ValueError("Sentence indices must be nonnegative integers")
        elif kind == "boolean":
            if type(item) is not bool:
                raise ValueError("Review approval must be a JSON boolean")
        else:
            raise ValueError("Unknown owned schema type")
    validate(value, ROLE_JSON_SCHEMAS[role])
    return value


class _ConstrainedModel:
    """Request-local public-generate adapter; all other model attributes delegate."""
    def __init__(self, model, prefix):
        self._model, self._prefix = model, prefix

    def __getattr__(self, name):
        return getattr(self._model, name)

    def generate(self, *args, **kwargs):
        if "prefix_allowed_tokens_fn" in kwargs:
            raise ValueError("An owned request cannot override its schema constraint")
        return self._model.generate(*args, prefix_allowed_tokens_fn=self._prefix, **kwargs)


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
    def __init__(self, *args, **kwargs):
        constraint_dependencies()
        super().__init__(*args, **kwargs)
        self._constraint_tokenizer_data = None
        self.metadata["format_constraints"] = constraint_provenance()
        _atomic_json(self.output / "writer.json", self.metadata)

    def _prefix_for(self, role):
        if getattr(self, "_constraint_tokenizer_data", None) is None:
            self._constraint_tokenizer_data = build_constraint_tokenizer_data(self.tokenizer)
        return build_constraint_prefix(self._constraint_tokenizer_data, role)

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
        else:
            raise ValueError("Every owned generation requires its correlated call identity")
        role = metadata.get("yash_role")
        if role not in ROLE_JSON_SCHEMAS or body.get("tools"):
            raise ValueError("Only the three frozen JSON roles are supported")
        prefix = self._prefix_for(role)
        request_writer = copy(self)
        request_writer.model = _ConstrainedModel(self.model, prefix)
        response = Writer.complete(request_writer, body)
        correlate_response(self.output, body, response)
        primary = self.output / "requests" / (response["id"] + ".json")
        raw = json.loads(primary.read_bytes())["response_text"]
        check = {"call_id": call_id, "role": role, "response_id": response["id"],
                 "schema_version": ROLE_FORMAT_VERSION,
                 "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                 "status": "incomplete_length" if response["choices"][0]["finish_reason"] == "length" else "failed"}
        error = None
        try:
            if check["status"] == "incomplete_length":
                raise ValueError("The output exhausted its declared token budget")
            _validate_role_json(raw, role)
            if response["choices"][0]["message"].get("content", "").strip() != raw.strip():
                raise ValueError("The response content differs from the retained raw JSON")
            check["status"] = "validated"
        except (ValueError, TypeError, KeyError) as exc:
            error = exc
            check["error"] = {"type": type(exc).__name__, "message": str(exc)}
        checks = self.output / "format-checks"
        checks.mkdir(exist_ok=True)
        _atomic_json(checks / (call_id + ".json"), check)
        if error is not None:
            raise ValueError("Generated raw output failed strict owned schema validation") from error
        return response


def create_writer_app(writer):
    """Build the real HTTP surface without loading a model or starting a port."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    app = FastAPI()

    @app.get("/health")
    def health():
        return {**writer.metadata, "client_correlation": "by-client/<call_id>.json"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    async def complete(request):
        body = await request.json()
        try:
            response = writer.complete(body)
        except Exception as exc:
            return JSONResponse({"error": {"type": type(exc).__name__, "message": str(exc)}}, status_code=400)
        if body.get("stream"):
            return StreamingResponse(stream_response(response), media_type="text/event-stream")
        return response

    # Future annotations would turn a locally imported Request into an
    # unresolved string. Bind the actual class before FastAPI inspects it.
    complete.__annotations__["request"] = Request
    app.add_api_route("/v1/chat/completions", complete, methods=["POST"])
    return app


def serve_writer(output, port=8001):
    import uvicorn
    writer = CorrelatedWriter(output, "cuda", max_context=16384)
    app = create_writer_app(writer)
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
                or metadata.get("format_constraints") != constraint_provenance()
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
                ("torch", "transformers", "strands-agents", "strands-decider", "peft", "datasets", "lm-format-enforcer")},
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
        try:
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
        except BaseException as exc:
            _atomic_json(self.output / "qualification.json", {
                "status": "FAILED", "kind": "invented engineering input; not benchmark accuracy",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "trace": trace, "role_usage": agents.usage, "critic_usage": self.critic.usage,
                "real_three_role_interaction": False, "long_context_generation_probe_performed": False})
            raise
        finally:
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
