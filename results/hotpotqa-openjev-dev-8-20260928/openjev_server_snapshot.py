"""Loopback-only HaltIQ wrapper around daseinlabs/open-jev's unchanged scorer."""

import argparse
from contextlib import asynccontextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
import uvicorn

from openjev.scorer import OptionScorer
from openjev.systemone import NoulQuestion, SystemOneRequest, render_noul, system_one


ROOT = Path(__file__).resolve().parents[1]


def create_app(runtime, scorer_factory=OptionScorer):
    lock = threading.Lock()
    state = {}
    provenance = {key: runtime[key] for key in ("implementation", "upstream_revision", "model_id", "model_repository",
                                               "model_revision", "backend", "max_prompt_tokens", "files")}
    provenance.update(decision_method="softmax of yes/no continuation log probabilities; uncalibrated",
                      normalization="sum", packages=runtime["packages"],
                      server_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @asynccontextmanager
    async def lifespan(app):
        scorer = scorer_factory(runtime["model_path"], backend=runtime["backend"], device=runtime["device"], batch_size=2)
        scorer.score("warm up", ["yes", "no"], norm="sum")
        state["scorer"] = scorer
        yield
        state.clear()

    app = FastAPI(title="HaltIQ local OpenJev", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": "scorer" in state, **provenance}

    @app.post("/v1/systemone")
    def judge(request: SystemOneRequest):
        if request.model not in (None, "openjev-latest", runtime["model_id"]):
            raise HTTPException(400, "Requested model is not the installed OpenJev model")
        if len(request.questions) > 3 or any(not isinstance(q, NoulQuestion) for q in request.questions.values()):
            raise HTTPException(400, "This research server supports at most three Noul questions per request")
        if "scorer" not in state:
            raise HTTPException(503, "Model is not ready")
        with lock:
            scorer = state["scorer"]
            # Use the same renderer/tokenizer as upstream inference. Reject long
            # inputs before any scoring, preserving all supplied evidence.
            for question in request.questions.values():
                prompt, labels = render_noul(request.state, question)
                size = len(scorer.context_ids(prompt)) + max(len(scorer.option_ids(label)) for label in labels)
                if size > runtime["max_prompt_tokens"]:
                    raise HTTPException(400, "Prompt exceeds the configured token limit; evidence was not truncated")
            result = system_one(scorer, request, model_name=runtime["model_id"], norm="sum").model_dump()
            result["openjev_metadata"] = provenance
            return result

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    runtime = json.loads((ROOT / ".haltiq/openjev-runtime.json").read_text())
    runtime["device"] = args.device
    for package, expected in runtime["packages"].items():
        if version(package) != expected:
            parser.error(f"OpenJev package version changed: {package}; rerun setup to record it")
    # Checkpoint hashes were recorded by setup. Verify all files before loading
    # so edited weights/configs cannot retain the previous provenance label.
    for name, expected in runtime["files"].items():
        path = Path(runtime["model_path"]) / name
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected["sha256"]:
            parser.error(f"Checkpoint file changed: {name}; rerun setup to record it")
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
