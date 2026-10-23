"""Explicit local-only Strands action gate child; JSON in, one JSON result out."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import uuid

from .core import CRITERIA, INSTRUCTIONS, CONTEXT_PROTOCOL, GateScore, fingerprint, validate_gate_view


def request_state(value):
    if not isinstance(value, dict) or set(value) != {"state", "instructions", "criteria"}:
        raise ValueError("expected the exact shared action-gate request schema")
    if value["instructions"] != INSTRUCTIONS or value["criteria"] != CRITERIA:
        raise ValueError("action criterion differs from the frozen shared question")
    return validate_gate_view(value["state"])


def tokenize_request(checkpoint, state):
    """Tokenization only: no model construction, weights, inference or CUDA API."""
    from transformers import AutoTokenizer
    from strands_decider.prompting import build_prompt
    from strands_decider.schema import NoulQuestion
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.no_truncation()
    prompt, _ = build_prompt(validate_gate_view(state), NoulQuestion(instructions=INSTRUCTIONS, criteria=CRITERIA),
                             option_order=(0, 1))
    count = len(tokenizer(prompt, truncation=False, padding=False)["input_ids"])
    config = json.loads((Path(checkpoint) / "strands_decider_config.json").read_text(encoding="utf-8"))
    window = min(4096, config["max_length"])
    return {"context_protocol": CONTEXT_PROTOCOL, "input_tokens": count, "maximum_tokens": window,
            "fits": count <= window, "state_sha256": fingerprint(state), "prompt_sha256":
            __import__("hashlib").sha256(prompt.encode()).hexdigest(), "model_forwards": 0,
            "silent_truncation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "score"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base")
    parser.add_argument("--deadline")
    parser.add_argument("--gpu-lock")
    parser.add_argument("--control-root")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    state = request_state(json.load(sys.stdin))
    with redirect_stdout(sys.stderr):
        preflight = tokenize_request(args.checkpoint, state)
        if args.mode == "preflight":
            result = preflight
        else:
            if not args.allow_model_calls or not all((args.base, args.deadline, args.gpu_lock, args.control_root)):
                parser.error("score requires explicit model opt-in, local base, deadline, shared GPU lock and control root")
            if not preflight["fits"]:
                raise ValueError("full action prompt exceeds native window; no model forward or truncation")
            from pc_strands.control import GpuLease, RunControl
            from pc_strands.decider import DeciderPaths
            from .providers import native_strands_backend
            control = RunControl(Path(args.control_root) / uuid.uuid4().hex, args.deadline)
            backend = None
            started = time.perf_counter()
            with GpuLease(Path(args.gpu_lock), "tau-policy-action-gate"):
                control.before_work(require_gpu=args.device == "cuda")
                try:
                    startup_started = time.perf_counter()
                    backend = native_strands_backend(DeciderPaths(args.checkpoint, args.base),
                        device=args.device, allow_model_calls=True, cpu_threads=args.cpu_threads)
                    backend.activate()
                    startup = time.perf_counter() - startup_started
                    control.rest_after_block(startup)
                    control.before_work(require_gpu=args.device == "cuda")
                    score = backend.score(state, temperature=1.0)
                    control.rest_after_block(score.elapsed_seconds)
                    result = asdict(GateScore(score.p_true, score.model_id, score.physical_input_tokens,
                        0, None, score.elapsed_seconds, tuple(score.logits), startup,
                        control.idle_seconds, time.perf_counter() - started))
                finally:
                    if backend is not None:
                        backend.offload()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
