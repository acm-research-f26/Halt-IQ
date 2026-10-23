"""Explicit provider boundaries. Importing this module performs no IO/model load."""
from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import time
from pathlib import Path

from .core import CRITERIA, INSTRUCTIONS, CONTEXT_PROTOCOL, GateScore, PublicState, fingerprint, gate_view, validate_gate_view


class CommandGate:
    """One explicit child request via JSON stdin/stdout, with no shell or retries.

    The child must print exactly a GateScore JSON object. It owns its model
    runtime, resource admission and any credentials; this adapter never writes
    credentials or command arguments to experiment outputs.
    """
    def __init__(self, argv, *, allow_model_calls=False, timeout=60, deadline=None, preflight_argv=None):
        if not allow_model_calls:
            raise ValueError("model execution needs explicit allow_model_calls")
        if not isinstance(argv, list) or not argv or not all(type(x) is str and x for x in argv):
            raise ValueError("argv must be a nonempty argument list")
        if type(timeout) not in (int, float) or not 0 < timeout <= 600:
            raise ValueError("timeout must be between zero and 600 seconds")
        self.argv, self.timeout = list(argv), timeout
        self.deadline = deadline
        if preflight_argv is not None and (not isinstance(preflight_argv, list) or not preflight_argv or
                                           not all(type(x) is str and x for x in preflight_argv)):
            raise ValueError("preflight_argv must be an explicit nonempty argument list")
        self.preflight_argv = preflight_argv

    def identity(self):
        return {"provider": "explicit_native_subprocess", "command_sha256": fingerprint(self.argv),
                "preflight_command_sha256": fingerprint(self.preflight_argv),
                "context_protocol": CONTEXT_PROTOCOL, "criteria_sha256": fingerprint([INSTRUCTIONS, CRITERIA])}

    def preflight(self, state: PublicState):
        if self.preflight_argv is None:
            raise ValueError("an explicit tokenizer-only preflight command is required before pilot dispatch")
        request = {"state": gate_view(state), "instructions": INSTRUCTIONS, "criteria": CRITERIA}
        result = subprocess.run(self.preflight_argv, input=json.dumps(request, allow_nan=False),
                                text=True, capture_output=True, timeout=60, check=False, shell=False)
        if result.returncode or len(result.stdout.encode("utf-8")) > 256 * 1024:
            raise RuntimeError("tokenizer-only preflight failed; no model call was attempted")
        row = json.loads(result.stdout)
        if (not isinstance(row, dict) or row.get("context_protocol") != CONTEXT_PROTOCOL or
                row.get("model_forwards") != 0 or row.get("silent_truncation") is not False or
                type(row.get("input_tokens")) is not int or row["input_tokens"] <= 0 or
                row.get("maximum_tokens") != 4096 or type(row.get("fits")) is not bool or
                row["fits"] != (row["input_tokens"] <= 4096)):
            raise ValueError("invalid tokenizer-only preflight receipt")
        return row

    def score(self, state: PublicState):
        payload = {"state": gate_view(state), "instructions": INSTRUCTIONS, "criteria": CRITERIA}
        started = time.perf_counter()
        timeout = self.timeout
        if self.deadline is not None:
            remaining = self.deadline.timestamp() - time.time()
            if remaining <= 0:
                raise RuntimeError("deadline reached before gate dispatch")
            timeout = min(timeout, remaining)
        try:
            result = subprocess.run(self.argv, input=json.dumps(payload, allow_nan=False),
                                    text=True, capture_output=True, timeout=timeout,
                                    check=False, shell=False)
        except subprocess.TimeoutExpired:
            raise RuntimeError("gate request timed out; no automatic retry") from None
        if result.returncode:
            raise RuntimeError("gate process failed; no automatic retry")
        if len(result.stdout.encode("utf-8")) > 256 * 1024:
            raise ValueError("gate output exceeds the response limit")
        row = json.loads(result.stdout, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite output")))
        if not isinstance(row, dict) or set(row) - set(GateScore.__dataclass_fields__):
            raise ValueError("gate must return only GateScore fields")
        score = GateScore(**row)
        return GateScore(**{**asdict(score), "wall_seconds": time.perf_counter() - started})


class JevActionGate:
    """Use the tested budget-reserving Jev client with the NEW action criterion."""
    def __init__(self, client, *, allow_hosted_calls=False, deadline=None):
        from haltiq.providers import JevClient
        if not allow_hosted_calls:
            raise ValueError("hosted calls need explicit allow_hosted_calls")
        if not isinstance(client, JevClient) or client.local or client.budget is None:
            raise ValueError("official Jev client with an existing persistent ledger required")
        if client.budget.limit_usd > 4.50:
            raise ValueError("preserve the existing $4.50 soft budget")
        if not Path(client.budget.path).is_file():
            raise ValueError("existing budget ledger required")
        self.client = client
        self.deadline = deadline

    def identity(self):
        return {"provider": "official_typesafe", "model": self.client.model,
                "context_protocol": CONTEXT_PROTOCOL, "criteria_sha256": fingerprint([INSTRUCTIONS, CRITERIA]),
                "cumulative_soft_budget_usd": self.client.budget.limit_usd}

    def score(self, state: PublicState):
        if self.deadline is not None:
            remaining = self.deadline.timestamp() - time.time()
            if remaining <= 0:
                raise RuntimeError("deadline reached before Jev dispatch")
            self.client.timeout = min(self.client.timeout, remaining)
        if self.client.budget.snapshot()["unsettled_requests"]:
            raise RuntimeError("unsettled hosted requests block further dispatch")
        questions = {"action_allowed": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}
        call = self.client.judge(gate_view(state), questions)
        return GateScore(call.data["answers"]["action_allowed"]["noul"], call.model,
                         call.input_tokens, call.output_tokens, call.cost_usd,
                         call.latency_ms / 1000, None)


def native_strands_backend(paths, *, device="cuda", allow_model_calls=False,
                           cpu_threads=4):
    """Load the real frozen Strands model with a new policy-action question.

    The caller must hold the study GPU lease and resource governor. No Hotpot
    temperature or fine-tuning is silently transferred to this new task.
    """
    if not allow_model_calls:
        raise ValueError("native model load needs explicit allow_model_calls")
    from pc_strands.decider import StrandsDecider, ContextWindowError, option_token_indices

    class ActionDecider(StrandsDecider):
        def _encode(self, state, option_order):
            from strands_decider.prompting import build_prompt
            from strands_decider.schema import NoulQuestion
            public = validate_gate_view(state)
            if sorted(option_order) != [0, 1]:
                raise ValueError("binary false/true option order required")
            question = NoulQuestion(instructions=INSTRUCTIONS, criteria=CRITERIA)
            prompt, rendered = build_prompt(public, question, option_order=option_order)
            encoded = self.model.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                                           truncation=False, padding=False)
            count = int(encoded["input_ids"].shape[1])
            if count > self.max_length:
                raise ContextWindowError(count, self.max_length)
            offsets = encoded.pop("offset_mapping")[0].tolist()
            positions = option_token_indices(offsets, rendered.option_spans, len(prompt) - len(rendered.text))
            return public, prompt, encoded, positions, count

        def metadata(self):
            from .core import fingerprint
            metadata = super().metadata()
            metadata["checkpoint_stored_temperature"] = metadata["temperature"]
            metadata["temperature"] = 1.0
            metadata["model_fingerprint"] = fingerprint({"head": metadata["checkpoint_head_sha256"],
                "adapter": metadata["adapter_sha256"], "base": metadata["base_revision"],
                "instructions": INSTRUCTIONS, "criteria": CRITERIA})
            metadata.update(criterion="tau-policy-action-v1", context_protocol=CONTEXT_PROTOCOL, tau_calibration="not_fitted",
                            hotpot_calibration_applied=False)
            return metadata

    return ActionDecider(paths, device=device, cpu_threads=cpu_threads)


class NativeStrandsGate:
    def __init__(self, backend, governor, *, allow_model_calls=False):
        if not allow_model_calls:
            raise ValueError("native calls need explicit allow_model_calls")
        if governor is None:
            raise ValueError("explicit resource governor required")
        self.backend, self.governor = backend, governor

    def identity(self):
        return {"provider": "local_native_strands", "model": self.backend.metadata(),
                "context_protocol": CONTEXT_PROTOCOL}

    def preflight(self, state: PublicState):
        from pc_strands.decider import ContextWindowError
        view = gate_view(state)
        try:
            _, prompt, _, _, count = self.backend._encode(view, (0, 1))
            prompt_hash = __import__("hashlib").sha256(prompt.encode()).hexdigest()
        except ContextWindowError as exc:
            count, prompt_hash = exc.token_count, None
        return {"context_protocol": CONTEXT_PROTOCOL, "input_tokens": count,
                "maximum_tokens": self.backend.max_length, "fits": count <= self.backend.max_length,
                "model_forwards": 0, "silent_truncation": False, "state_sha256": fingerprint(view),
                "prompt_sha256": prompt_hash}

    def score(self, state: PublicState):
        self.governor.before_work(require_gpu=self.backend.device == "cuda")
        started = time.perf_counter()
        try:
            # Raw released-head temperature: do not reuse Hotpot-fitted temperature.
            result = self.backend.score(gate_view(state), temperature=1.0)
        finally:
            self.governor.rest_after_block(time.perf_counter() - started)
        return GateScore(result.p_true, result.model_id, result.physical_input_tokens,
                         0, None, result.elapsed_seconds, tuple(result.logits))

    def close(self):
        self.backend.offload()
