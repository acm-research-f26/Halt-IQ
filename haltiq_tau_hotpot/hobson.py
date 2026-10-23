"""Hobson-v19 pointer features; the caller owns admission, GPU lease and journals.

Importing this module never imports Torch or loads a model. Construct/activate a
backend only inside the runtime's shared GPU lease. Features contain public input
provenance and CPU tensors, never evaluator labels or question identifiers.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

from haltiq_hotpot.native import OfficialDecider, HOBSON_TEMPERATURE, PINS
from haltiq_hotpot.protocol import INSTRUCTIONS

FEATURE_SCHEMA = "haltiq-hobson-pointer-feature-v1"
HEAD_PARAMETERS = 1053184
HIDDEN_SIZE = 2048
INSTRUCTIONS_BY_CRITERION = {
    "answer": INSTRUCTIONS,
    "evidence": "Do the cited evidence sentences jointly support the draft answer and all reasoning hops needed to answer the question?",
}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def checkpoint_config(checkpoint):
    path = Path(checkpoint)
    choices = [path / n for n in ("hobson_config.json", "strands_decider_config.json") if (path / n).is_file()]
    if len(choices) != 1:
        raise ValueError("One unambiguous Hobson configuration is required")
    data = json.loads(choices[0].read_text(encoding="utf-8"))
    if (data.get("base_model") != "Qwen/Qwen3.5-2B-Base" or data.get("head_type") != "pointer"
            or data.get("pointer_dim") != 256 or data.get("lora_r") != 16
            or data.get("temperature_by_kind", {}).get("noul") != HOBSON_TEMPERATURE):
        raise ValueError("Only the released Hobson-v19 256/rank16 configuration is supported")
    return data


def encoder_binding(checkpoint, base):
    """Hash actual frozen files, without substituting old 768/rank48 features."""
    checkpoint, base = Path(checkpoint), Path(base)
    checkpoint_config(checkpoint)
    files = {"checkpoint/" + name: checkpoint / name for name in (
        "hobson_config.json", "strands_decider_config.json", "lora/adapter_model.safetensors",
        "lora/adapter_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        if (checkpoint / name).is_file()}
    files.update({"base/" + p.name: p for p in base.glob("*.safetensors")})
    files.update({"base/" + name: base / name for name in ("config.json", "model.safetensors.index.json")
                  if (base / name).is_file()})
    spec = importlib.util.find_spec("strands_decider")  # Top-level lookup executes no package code.
    if spec is None or spec.origin is None:
        raise ValueError("The reviewed local official Strands source is required")
    upstream = Path(spec.origin).parent
    for name in ("modeling.py", "infer.py", "prompting.py", "schema.py"):
        path = upstream / name
        if not path.is_file() or file_hash(path) != PINS["upstream/" + name]:
            raise ValueError("Official native source differs from the reviewed upstream revision")
        files["upstream/" + name] = path
    if ("checkpoint/lora/adapter_model.safetensors" not in files
            or "checkpoint/tokenizer.json" not in files or "base/config.json" not in files
            or not any(k.startswith("base/") and k.endswith(".safetensors") for k in files)):
        raise ValueError("Complete local base, adapter and tokenizer files are required")
    return {name: file_hash(path) for name, path in sorted(files.items())}


def load_head(checkpoint, head_path=None):
    """Load only the existing compatible 1,053,184-parameter pointer head on CPU."""
    from strands_decider.modeling import StrandsDeciderConfig, build_head, config_path
    from safetensors.torch import load_file
    checkpoint_config(checkpoint)
    config = StrandsDeciderConfig.from_json(config_path(str(checkpoint)))
    head = build_head(config, HIDDEN_SIZE).float().cpu()
    head.load_state_dict(load_file(str(head_path or Path(checkpoint) / "head.safetensors"), device="cpu"), strict=True)
    if sum(p.numel() for p in head.parameters()) != HEAD_PARAMETERS:
        raise ValueError("Hobson head parameter count differs")
    return head.eval().requires_grad_(False)


def feature_tensors(feature, *, expected_encoder=None, criterion=None):
    """Validate an in-memory or rehydrated CPU cache record, without gold labels."""
    import torch
    if (feature.get("schema") != FEATURE_SCHEMA or feature.get("option_order") != ["false", "true"]
            or feature.get("criterion") not in INSTRUCTIONS_BY_CRITERION):
        raise ValueError("Incompatible canonical Hobson feature schema")
    if criterion is not None and feature["criterion"] != criterion:
        raise ValueError("Criterion mismatch in cached feature")
    if expected_encoder is not None and feature.get("encoder_sha256") != expected_encoder:
        raise ValueError("Cached feature belongs to a different frozen encoder")
    provenance = feature.get("provenance", {})
    if (stable_hash(provenance) != feature.get("provenance_sha256")
            or provenance.get("criterion") != feature["criterion"]
            or provenance.get("instructions_sha256") != stable_hash(INSTRUCTIONS_BY_CRITERION[feature["criterion"]])
            or provenance.get("source_sha256") != file_hash(__file__)
            or provenance.get("option_order") != ["false", "true"]
            or not all(isinstance(provenance.get(k), str) and len(provenance[k]) == 64
                       for k in ("state_sha256", "token_ids_sha256", "rendered_question_sha256", "source_sha256"))):
        raise ValueError("Missing or changed head-independent input provenance")
    if any(isinstance(feature[k], torch.Tensor) and feature[k].device.type != "cpu"
           for k in ("decide", "options")):
        raise ValueError("Cached training features must already be on CPU")
    decide = torch.as_tensor(feature["decide"], dtype=torch.float32, device="cpu")
    options = torch.as_tensor(feature["options"], dtype=torch.float32, device="cpu")
    if (tuple(decide.shape) != (HIDDEN_SIZE,) or tuple(options.shape) != (2, HIDDEN_SIZE)
            or not bool(torch.isfinite(decide).all()) or not bool(torch.isfinite(options).all())):
        raise ValueError("Invalid cached pointer tensor shape or values")
    return decide, options


def score_head(head, feature, *, temperature):
    import torch
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("A finite positive criterion-specific temperature is required")
    decide, options = feature_tensors(feature)
    head.eval()
    with torch.no_grad():
        logits = head(decide.unsqueeze(0), options.unsqueeze(0))[0].float()
        if tuple(logits.shape) != (2,) or not bool(torch.isfinite(logits).all()):
            raise ValueError("Invalid canonical binary head logits")
        probability = float(torch.softmax(logits / temperature, dim=-1)[1])
    return {"raw_logits_false_true": logits.tolist(), "temperature": float(temperature),
            "p_true_unrounded": probability, "p_true": round(probability, 4),
            "physical_forwards": 0, "head_scores": 1}


class HobsonBackend:
    def __init__(self, checkpoint, base):
        self.checkpoint, self.base = Path(checkpoint).resolve(), Path(base).resolve()
        self.encoder_sha256 = encoder_binding(self.checkpoint, self.base)
        self.released_head_sha256 = file_hash(self.checkpoint / "head.safetensors")
        self._official = OfficialDecider(self.checkpoint, self.base, kind="hobson")
        self.torch = self._official.torch
        self._released = copy.deepcopy(self._official.model.head).float().cpu().eval().requires_grad_(False)
        if sum(p.numel() for p in self._released.parameters()) != HEAD_PARAMETERS:
            raise ValueError("Expected the actual Hobson 256-dimensional pointer head")
        self._heads = {}
        self.active = False
        self.metadata = {**self._official.metadata, "feature_schema": FEATURE_SCHEMA,
            "encoder_sha256": self.encoder_sha256, "released_head_sha256": self.released_head_sha256,
            "head_parameters": HEAD_PARAMETERS, "criteria": dict(INSTRUCTIONS_BY_CRITERION),
            "canonical_option_order": ["false", "true"], "feature_cache_head_independent": True}

    def activate(self):
        if self.active:
            raise RuntimeError("Hobson backend is already active")
        self._official.activate()
        self.active = True

    def _prepare(self, state, criterion):
        if not self.active or self._official is None:
            raise RuntimeError("Caller must activate inside its GPU lease before observation")
        if not isinstance(state, str) or not state.strip() or criterion not in INSTRUCTIONS_BY_CRITERION:
            raise ValueError("Nonempty public state and declared criterion required")
        from strands_decider.prompting import render_question, render_state
        from strands_decider.schema import NoulQuestion
        rendered = render_question(NoulQuestion(instructions=INSTRUCTIONS_BY_CRITERION[criterion]))
        if tuple(rendered.slot_labels) != ("false", "true"):
            raise ValueError("Unexpected native Noul option ordering")
        engine = self._official.engine
        state_ids, questions, offsets = engine._fit(render_state(state), [rendered.text])
        return rendered, state_ids, questions[0], offsets

    def preflight(self, state, criterion="answer"):
        _, state_ids, question_ids, _ = self._prepare(state, criterion)
        return {"input_tokens": len(state_ids) + len(question_ids), "physical_forwards": 0}

    def prepare_state(self, state, criterion="answer"):
        rendered, state_ids, question_ids, offsets = self._prepare(state, criterion)
        return {"state_sha256": stable_hash(state), "criterion": criterion, "rendered": rendered,
                "state_ids": state_ids, "question_ids": question_ids, "offsets": offsets,
                "input_tokens": len(state_ids) + len(question_ids), "physical_forwards": 0}

    def _selected_head(self, head_path):
        if head_path is None:
            if file_hash(self.checkpoint / "head.safetensors") != self.released_head_sha256:
                raise ValueError("Released head file changed during the backend lifetime")
            return self._released
        path = str(Path(head_path).resolve())
        sha = file_hash(path)
        if path not in self._heads:
            self._heads[path] = (sha, load_head(self.checkpoint, path))
        if self._heads[path][0] != sha:
            raise ValueError("Selected head file changed after first load")
        return self._heads[path][1]

    def score_features(self, feature, *, head_path=None, temperature=None):
        feature_tensors(feature, expected_encoder=self.encoder_sha256)
        if temperature is None:
            if head_path is not None:
                raise ValueError("A trained head requires its separately fitted temperature")
            temperature = HOBSON_TEMPERATURE if feature["criterion"] == "answer" else 1.0
        return score_head(self._selected_head(head_path), feature, temperature=temperature)

    def observe(self, state, criterion="answer", head_path=None, temperature=None, *, prepared=None):
        fitted = prepared if prepared is not None else self.prepare_state(state, criterion)
        if fitted["state_sha256"] != stable_hash(state) or fitted["criterion"] != criterion:
            raise ValueError("Prepared tokens do not belong to this public state and criterion")
        rendered, state_ids, question_ids, offsets = [fitted[k] for k in
            ("rendered", "state_ids", "question_ids", "offsets")]
        torch, engine, model = self.torch, self._official.engine, self._official.model
        selected = self._selected_head(head_path)
        if temperature is None:
            if head_path is not None:
                raise ValueError("A trained head requires its separately fitted temperature")
            temperature = HOBSON_TEMPERATURE if criterion == "answer" else 1.0
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Finite positive criterion-specific temperature required")
        # Select compatible head values in memory only; the same frozen torso and
        # adapter produce features for every head. Released files are never edited.
        model.head.load_state_dict(selected.state_dict(), strict=True)
        captured = []
        def capture(_module, inputs):
            if len(inputs) != 2:
                raise ValueError("Native head did not receive pointer decision/option tensors")
            # Only detached CPU values leave the inference block; no hidden GPU ref survives.
            captured.append(tuple(v.detach().float().cpu().tolist() for v in inputs))
        handle = model.head.register_forward_pre_hook(capture)
        started = time.perf_counter()
        try:
            ids, mask = engine._pad([state_ids + question_ids])
            options = engine._option_idx([rendered], len(state_ids), offsets)
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask,
                    n_slots=torch.tensor([2], device=engine.device), temperature=1.0, opt_idx=options)
                native_logits = output["logits"][0, :2].float().cpu().tolist()
            torch.cuda.synchronize()
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("One native forward must invoke the pointer head exactly once")
        provenance = {"criterion": criterion, "state_sha256": stable_hash(state),
            "instructions_sha256": stable_hash(INSTRUCTIONS_BY_CRITERION[criterion]),
            "rendered_question_sha256": stable_hash(rendered.text),
            "token_ids_sha256": stable_hash(state_ids + question_ids),
            "option_order": ["false", "true"], "source_sha256": file_hash(__file__)}
        feature = {"schema": FEATURE_SCHEMA, "criterion": criterion,
            "option_order": ["false", "true"], "encoder_sha256": dict(self.encoder_sha256),
            "provenance": provenance, "provenance_sha256": stable_hash(provenance),
            "decide": torch.tensor(captured[0][0][0], dtype=torch.float32),
            "options": torch.tensor(captured[0][1][0], dtype=torch.float32)}
        replay = score_head(selected, feature, temperature=temperature)
        error = max(abs(a - b) for a, b in zip(native_logits, replay["raw_logits_false_true"]))
        if not math.isfinite(error) or error > 2e-4:
            raise RuntimeError("Cached head-input replay does not reproduce native logits")
        result = dict(replay)
        result.update(native_logits_false_true=native_logits, feature=feature,
            cached_replay_max_logit_error=error, input_tokens=len(state_ids) + len(question_ids),
            physical_forwards=1, head_scores=2, seconds=time.perf_counter() - started)
        # One native selected-head call plus one CPU replay, without torso rerun.
        return result

    def close(self):
        self.active = False
        self._heads.clear()
        self._released = None
        official, self._official = self._official, None
        if official is None:
            return {"already_closed": True}
        return official.close()
