"""Vedanshi's Hobson-v19 critic with BGE top-two evidence and an owned server.

The model/format follow the teammate's integration notebook, not Yash's prior
Strands training. Imports are light; downloads and GPU work require explicit calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

TEAM_REVISION = "6c72991652400dcfec8f464a331efb95d5d1cf8b"
HOBSON_MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"
HOBSON_REVISION = "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
BASE_MODEL = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
BGE_MODEL = "BAAI/bge-reranker-base"
BGE_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
DECIDER_VERSION = "0.2.0"
INSTRUCTION = "Is the draft answer correct and fully supported by the evidence?"


class Critic:
    """Lazy BGE ranker and the teammate's exact System One HTTP payload.

    ``ranker`` and ``transport`` can be injected for offline contract tests.
    A ranker receives (question + ' ' + answer, passages) and returns one finite
    score per passage. HTTP transports receive a JSON payload and return JSON.
    """
    def __init__(self, endpoint="http://127.0.0.1:8000", *, ranker=None,
                 transport=None, ranker_device="cpu", timeout=120):
        self.endpoint = endpoint.rstrip("/")
        self._ranker = ranker
        self._transport = transport
        self.ranker_device = ranker_device
        self.timeout = timeout
        self.last_record = None
        self._tokenizer = self._model = None
        self.provenance = {"method_source_revision": TEAM_REVISION,
                           "critic_model": HOBSON_MODEL, "critic_revision": HOBSON_REVISION,
                           "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                           "reranker_model": BGE_MODEL, "reranker_revision": BGE_REVISION,
                           "top_k": 2, "rerank_max_tokens": 512,
                           "instructions": INSTRUCTION, "strands_threshold": 0.8}

    def _rank(self, query, passages):
        if self._ranker is not None:
            return list(self._ranker(query, passages))
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL, revision=BGE_REVISION)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                BGE_MODEL, revision=BGE_REVISION, use_safetensors=True).to(self.ranker_device).eval()
            self._model.requires_grad_(False)
        x = self._tokenizer([query] * len(passages), passages, truncation=True,
                            max_length=512, padding=True, return_tensors="pt").to(self.ranker_device)
        with torch.inference_mode():
            return self._model(**x).logits.view(-1).float().tolist()

    def _post(self, payload):
        if self._transport is not None:
            return self._transport(payload)
        url = self.endpoint if self.endpoint.endswith("/v1/systemone") else self.endpoint + "/v1/systemone"
        req = urllib.request.Request(url, json.dumps(payload, allow_nan=False).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def score(self, question: str, passages: list[str], answer: str) -> float:
        self.last_record = None
        if (not isinstance(question, str) or not question.strip() or not isinstance(answer, str)
                or not isinstance(passages, list) or not passages
                or any(not isinstance(p, str) or not p.strip() for p in passages)):
            raise ValueError("A public question, draft and nonempty passages are required")
        started = time.perf_counter()
        scores = self._rank(question + " " + answer, passages)
        if len(scores) != len(passages) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError("Every passage requires one finite reranker score")
        indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:2]
        evidence = "\n".join(passages[i] for i in indices)
        state = f"Question: {question}\nEvidence: {evidence}\nDraft answer: {answer}"
        payload = {"state": state, "questions": {"sufficient": {"type": "noul", "instructions": INSTRUCTION}}}
        response = self._post(payload)
        value = response["answers"]["sufficient"]
        p = value.get("noul", value.get("value"))
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Strands did not return a finite probability in [0,1]")
        self.last_record = {**self.provenance, "selected_indices": indices,
                            "reranker_scores": [float(s) for s in scores],
                            "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                            "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                            "p_true": float(p), "seconds": time.perf_counter() - started,
                            "usage": response.get("usage", {}), "model": response.get("model"),
                            "physical_critic_calls": 1, "physical_reranker_forwards": 1}
        return float(p)


@dataclass
class CriticServer:
    process: subprocess.Popen
    endpoint: str
    provenance: dict
    log_handle: object

    def close(self):
        """Stop only the process this object created."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        self.log_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _serving_dtype(device, dtype):
    if dtype != "auto":
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16 or float32")
        return dtype
    import torch
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA runtime is required; select GPU in Colab")
        # T4 (compute capability 7.5) has no native BF16. Do not promise native
        # BF16 support merely because a newer Torch build permits emulation.
        return "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
    return "float32"


def start_critic(cache_dir: str | Path, *, log_path: str | Path | None = None,
                 port=8000, device="cuda", dtype="auto", timeout=600) -> CriticServer:
    """Download the pinned teammate model and start a bounded, owned localhost server.

    Notebook install: ``%pip install strands-decider==0.2.0``. Unlike the original
    notebook's unpinned model name, the launch serves an immutable HF revision.
    A local serving copy changes only precision (FP16 on T4); the original cached
    checkpoint stays intact. This numerical change is recorded explicitly and
    requires a successful real runtime smoke test before benchmark execution.
    """
    from importlib.metadata import version
    if version("strands-decider") != DECIDER_VERSION:
        raise RuntimeError(f"Install strands-decider=={DECIDER_VERSION} before starting")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is occupied; the existing service was not changed")
    from huggingface_hub import snapshot_download
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(snapshot_download(HOBSON_MODEL, revision=HOBSON_REVISION))
    precision = _serving_dtype(device, dtype)
    serving = root / f"hobson-v19-{HOBSON_REVISION[:8]}-{precision}"
    if not serving.exists():
        shutil.copytree(source, serving)
    for original_file in source.rglob("*"):
        if original_file.is_file() and original_file.name != "hobson_config.json":
            served_file = serving / original_file.relative_to(source)
            if (not served_file.is_file()
                    or hashlib.sha256(served_file.read_bytes()).digest() != hashlib.sha256(original_file.read_bytes()).digest()):
                raise ValueError("The local serving copy differs from the pinned model snapshot")
    config_path = serving / "hobson_config.json"
    original = json.loads((source / "hobson_config.json").read_text())
    config = {**original, "torch_dtype": precision, "base_revision": BASE_REVISION}
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    provenance = {"model": HOBSON_MODEL, "revision": HOBSON_REVISION,
                  "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                  "strands_decider_version": DECIDER_VERSION, "device": device,
                  "original_dtype": original["torch_dtype"], "serving_dtype": precision,
                  "precision_changed": precision != original["torch_dtype"],
                  "original_config_sha256": hashlib.sha256((source / "hobson_config.json").read_bytes()).hexdigest(),
                  "served_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  "historical_probability_identity_claimed": False}
    (serving / "haltiq-serving-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log_path = Path(log_path) if log_path else root / "strands-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    # Upstream has a CLI entry point, but no strands_decider.__main__ module.
    command = [sys.executable, "-c", "from strands_decider.cli import main; main()", "serve",
               str(serving), "--host", "127.0.0.1", "--port", str(port), "--device", device,
               "--max-batch", "1", "--strict-window", "--no-prefix-cache"]
    try:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    except BaseException:
        handle.close()
        raise
    server = CriticServer(process, f"http://127.0.0.1:{port}", provenance, handle)
    end = time.monotonic() + timeout
    try:
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(f"Strands server exited ({process.returncode}); inspect {log_path}")
            try:
                with urllib.request.urlopen(server.endpoint + "/health", timeout=3) as response:
                    health = json.load(response)
                if health.get("status") == "ok" and Path(health.get("checkpoint", "")).resolve() == serving:
                    server.provenance["health"] = health
                    return server
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError(f"Strands startup exceeded {timeout}s; inspect {log_path}")
    except BaseException:
        server.close()
        raise
