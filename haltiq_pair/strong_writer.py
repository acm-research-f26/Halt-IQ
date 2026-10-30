"""Optional 7B NF4 engineering variant; the recorded 1.5B writer is unchanged.

The model pin is from Hugging Face's Qwen model API. The bitsandbytes pin is
from PyPI. A GPU qualification is required before interpreting live results.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from .runtime import Runtime
from .server import (ATTENTION_IMPLEMENTATION, LOGITS_TO_KEEP, PREFILL_CHUNK_SIZE,
                     Writer, stream_response)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
BITSANDBYTES_VERSION = "0.50.2"
PIN_SOURCES = {
    "model": "https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct",
    "quantizer": "https://pypi.org/project/bitsandbytes/0.50.2/",
    "quantization_api": "https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes",
}


class StrongWriter(Writer):
    """Reuse the tested normalization, full-context generation and receipts."""

    def __init__(self, output, device="cuda", max_context=16384):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if device not in {"cuda", "cuda:0"} or not torch.cuda.is_available():
            raise RuntimeError("The optional 7B NF4 variant requires a CUDA GPU")
        if version("bitsandbytes") != BITSANDBYTES_VERSION:
            raise RuntimeError(f"Install bitsandbytes=={BITSANDBYTES_VERSION} for this frozen variant")
        if type(max_context) is not int or not 1024 <= max_context <= 16384:
            raise ValueError("Declare a context budget between 1024 and 16384 tokens")
        self.output, self.max_context = Path(output), max_context
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            torch_dtype=torch.float16, device_map={"": 0},
            quantization_config=quantization,
            attn_implementation=ATTENTION_IMPLEMENTATION).eval()
        self.metadata = {
            "model": MODEL, "revision": REVISION, "device": self.device,
            "dtype": "torch.float16 (nonquantized layers and computation)",
            "quantization": {"weight_bits": 4, "type": "nf4", "double_quant": True,
                             "compute_dtype": "torch.float16", "cpu_offload": False,
                             "library": "bitsandbytes", "version": BITSANDBYTES_VERSION},
            "max_context": max_context, "context_policy": "reject_overflow_without_truncation",
            "attention_implementation": self.model.config._attn_implementation,
            "prefill_chunk_size": PREFILL_CHUNK_SIZE, "logits_to_keep": LOGITS_TO_KEEP,
            "use_cache": True, "request_receipt_primary": "requests/<response_id>.json",
            "request_receipt_policy": "exclusive_temp_flush_fsync_replace",
            "request_receipt_compatibility": "model_requests.jsonl (best-effort mirror)",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "pin_sources": PIN_SOURCES, "new_training": False,
            "engineering_variant": "7B NF4; different from the recorded 1.5B study",
        }
        (self.output / "writer.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def complete(self, body):
        if body.get("model", MODEL) != MODEL:
            raise ValueError("Request model does not match the owned pinned 7B writer")
        # The inherited method owns normalization/chunking/usage/persistence. Its
        # historical model label is replaced before any HTTP/SSE response leaves.
        response = super().complete(body)
        response["model"] = MODEL
        return response


def create_app(writer):
    """Build the transport without loading a model, for offline contract checks."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    app = FastAPI()

    @app.get("/health")
    def health():
        return writer.metadata

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        try:
            response = writer.complete(body)
        except Exception as exc:
            return JSONResponse({"error": {"message": str(exc), "type": type(exc).__name__}},
                                status_code=400)
        if body.get("stream"):
            return StreamingResponse(stream_response(response), media_type="text/event-stream")
        return response

    return app


def serve(output, port=8001, device="cuda"):
    import uvicorn
    uvicorn.run(create_app(StrongWriter(output, device)), host="127.0.0.1",
                port=port, log_level="warning")


def validate_health(metadata):
    """A listening service is not proof that our process loaded the right model."""
    quantization = metadata.get("quantization", {})
    if (metadata.get("model") != MODEL or metadata.get("revision") != REVISION
            or quantization.get("weight_bits") != 4 or quantization.get("type") != "nf4"
            or quantization.get("compute_dtype") != "torch.float16"
            or quantization.get("version") != BITSANDBYTES_VERSION):
        raise RuntimeError("Writer health does not match the owned pinned 7B NF4 variant")


class StrongRuntime(Runtime):
    """Own exactly the critic and 7B child processes that this instance starts."""

    def __init__(self, output, head, timeout=1200):
        if getattr(head, "seed", None) != 0:
            raise ValueError("The deployment seed was declared as zero before held-out evaluation")
        super().__init__(output, timeout=timeout)
        self.head = head

    def start(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Select a Colab GPU runtime before running this variant")
        from .aligned_gate import Gate
        from .core import Agents
        from .critic import Critic, start_critic
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8001)) == 0:
                raise RuntimeError("Writer port 8001 is occupied; existing service was not changed")
        self.critic_server = start_critic(self.output / "critic-cache",
            log_path=self.output / "critic-server.log", timeout=self.timeout)
        self.log = (self.output / "writer-server.log").open("w", encoding="utf-8")
        self.writer = subprocess.Popen([sys.executable, "-m", "haltiq_pair.strong_writer",
            "serve", "--output", str(self.output / "writer")],
            stdout=self.log, stderr=subprocess.STDOUT)
        end = time.monotonic() + self.timeout
        while time.monotonic() < end:
            if self.writer.poll() is not None:
                raise RuntimeError("7B writer startup failed; read writer-server.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as response:
                    self.writer_metadata = json.load(response)
                validate_health(self.writer_metadata)
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("7B writer did not load before the startup deadline")
        agents = Agents(model_id=MODEL)
        critic = Critic(endpoint=self.critic_server.endpoint, ranker_device="cpu")
        gate = Gate(critic=critic, head=self.head, policy_name="strands_feature_retrained")
        (self.output / "runtime.json").write_text(json.dumps({
            "writer": self.writer_metadata, "critic_server": self.critic_server.provenance,
            "controller": gate.provenance, "engineering_variant": "7B NF4"}, indent=2),
            encoding="utf-8")
        return agents, gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    serve(args.output, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
