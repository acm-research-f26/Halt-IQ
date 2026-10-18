"""Offline-only Strands pointer backend; imports torch only when a model is loaded.

One public correctness question is shared by untouched and trained-head arms.
Cached-head scoring is explicitly distinguished from real encoder inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ID = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
MODEL_REVISION = "216d1ccb46a9e9b036eb9187b248db98480f77d4"
BASE_ID = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
CODE_REVISION = "930e6d5de16309be32579f5631a7369a3322140c"
CORRECTNESS_INSTRUCTIONS = (
    "Does the draft correctly answer the question using the supplied evidence?"
)
CORRECTNESS_CRITERIA = {
    "false": "The draft is incorrect, incomplete, or not established by the supplied evidence.",
    "true": "The draft correctly and adequately answers the question from the supplied evidence.",
}


class ContextWindowError(ValueError):
    def __init__(self, token_count: int, max_length: int):
        self.token_count = token_count
        self.max_length = max_length
        super().__init__(f"full rendered prompt needs {token_count} tokens; limit is {max_length}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_public_state(state: Mapping[str, Any], *, allow_history: bool = False) -> dict:
    """Reject metadata/gold fields instead of silently forwarding or deleting them."""
    if not isinstance(state, Mapping):
        raise ValueError("state must be an object")
    required = {"question", "evidence", "draft"}
    allowed = required | ({"history", "round", "budget"} if allow_history else set())
    if set(state) - allowed or required - set(state):
        raise ValueError(f"public state keys must be {sorted(allowed)}; got {sorted(state)}")
    for key in ("question", "draft"):
        if not isinstance(state[key], str) or not state[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    evidence = state["evidence"]
    if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) for x in evidence):
        raise ValueError("evidence must be a nonempty list of strings")
    result = {"question": state["question"], "evidence": list(evidence), "draft": state["draft"]}
    if "history" in state:
        if not isinstance(state["history"], list) or len(state["history"]) > 10:
            raise ValueError("history must be a list of at most ten public turns")
        for row in state["history"]:
            if not isinstance(row, dict) or set(row) != {"draft", "feedback"} or not all(
                isinstance(v, str) for v in row.values()
            ):
                raise ValueError("history turns may contain only draft and feedback strings")
        result["history"] = [dict(x) for x in state["history"]]
    if "round" in state:
        if type(state["round"]) is not int or not 1 <= state["round"] <= 10:
            raise ValueError("round must be an integer from one through ten")
        result["round"] = state["round"]
    if "budget" in state:
        budget = state["budget"]
        if not isinstance(budget, dict) or set(budget) != {"drafts_remaining"} or type(
            budget["drafts_remaining"]
        ) is not int or not 0 <= budget["drafts_remaining"] <= 10:
            raise ValueError("budget must contain only drafts_remaining in [0,10]")
        result["budget"] = dict(budget)
    return result


def option_token_indices(offsets: Sequence[Sequence[int]], spans: Sequence[Sequence[int]],
                         base: int) -> list[int]:
    """The official pointer's last complete token in each option line."""
    indices = []
    for start, end in spans:
        found = [i for i, (lo, hi) in enumerate(offsets)
                 if hi > lo and lo >= base + start and hi <= base + end]
        if not found:
            raise ValueError("an option has no complete token; refusing malformed pointer input")
        indices.append(found[-1])
    return indices


def probability_true(logits: Sequence[float], temperature: float = 1.0) -> float:
    if len(logits) != 2 or not all(math.isfinite(float(x)) for x in logits):
        raise ValueError("expected two finite canonical false/true logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    delta = (float(logits[1]) - float(logits[0])) / temperature
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    exp = math.exp(delta)
    return exp / (1.0 + exp)


def label_for_order(label: int, option_order: Sequence[int]) -> int:
    if type(label) not in (int, bool) or label not in (0, 1) or sorted(option_order) != [0, 1]:
        raise ValueError("binary canonical label and a two-option permutation required")
    return list(option_order).index(int(label))


@dataclass(frozen=True)
class DeciderPaths:
    checkpoint_dir: str | Path
    base_dir: str | Path


@dataclass(frozen=True)
class FeatureRecord:
    decide: list[float]
    options: list[list[float]]
    option_order: tuple[int, int]
    input_tokens: int
    prompt_sha256: str
    state_sha256: str
    encoding_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "FeatureRecord":
        obj = cls(**{k: row[k] for k in cls.__dataclass_fields__})
        if len(obj.decide) != 2048 or len(obj.options) != 2 or any(len(x) != 2048 for x in obj.options):
            raise ValueError("released head requires one 2048-vector and two option 2048-vectors")
        if len(obj.option_order) != 2 or any(type(v) is not int for v in obj.option_order) or sorted(obj.option_order) != [0, 1]:
            raise ValueError("invalid option permutation")
        if not all(math.isfinite(float(v)) for x in [obj.decide, *obj.options] for v in x):
            raise ValueError("nonfinite cached features")
        if type(obj.input_tokens) is not int or not 0 < obj.input_tokens <= 4096:
            raise ValueError("cached feature token count outside trained window")
        if (not isinstance(obj.prompt_sha256, str) or not obj.prompt_sha256
                or not isinstance(obj.state_sha256, str) or not obj.state_sha256
                or type(obj.encoding_seconds) not in (int, float)
                or not math.isfinite(obj.encoding_seconds) or obj.encoding_seconds < 0):
            raise ValueError("cached feature hashes and nonnegative finite encoder timing required")
        return obj


@dataclass(frozen=True)
class ScoreResult:
    p_true: float
    logits: tuple[float, float]
    input_tokens: int
    elapsed_seconds: float
    answer: str
    model_id: str
    physical_input_tokens: int
    reused_encoding: bool
    encoding_seconds: float
    temperature: float

    def to_dict(self) -> dict:
        return asdict(self)


def _checkpoint_config(directory: Path) -> dict:
    path = directory / "strands_decider_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("head_type") != "pointer" or config.get("pointer_dim") != 768:
        raise ValueError("expected the released 768-dimensional pointer head; no SlotHead substitution")
    if not config.get("use_lora") or config.get("lora_r") != 48:
        raise ValueError("expected the frozen released rank48 adapter")
    if config.get("base_model") != BASE_ID or config.get("lora_alpha") != 96:
        raise ValueError("unexpected base or released LoRA scale")
    return config


class StrandsDecider:
    """One frozen torso, swappable pointer heads, and no implicit network downloads."""

    def __init__(self, paths: DeciderPaths, *, device: str = "cuda", max_length: int = 4096,
                 seed: int = 42, allow_history: bool = False, cpu_threads: int = 4):
        if device not in {"cpu", "cuda"} or not 0 < max_length <= 4096:
            raise ValueError("device must be cpu/cuda; max_length must be in [1,4096]")
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 4:
            raise ValueError("CPU thread limit must be an integer from one through four")
        self.paths = DeciderPaths(Path(paths.checkpoint_dir).resolve(), Path(paths.base_dir).resolve())
        for p in (self.paths.checkpoint_dir, self.paths.base_dir):
            if not p.is_dir():
                raise FileNotFoundError(f"local model directory missing: {p}")
        self.original_config = _checkpoint_config(self.paths.checkpoint_dir)
        self.max_length = min(max_length, self.original_config["max_length"])
        self.allow_history, self.device, self.seed = allow_history, device, seed
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer
        from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel, load_head_state

        self.torch = torch
        self.cpu_threads = cpu_threads
        torch.set_num_threads(cpu_threads)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing an implicit CPU fallback")
        torch.manual_seed(seed)
        # A local-only base override avoids changing the immutable downloaded snapshot.
        cfg = StrandsDeciderConfig.from_json(str(self.paths.checkpoint_dir / "strands_decider_config.json"))
        cfg.base_model, cfg.base_revision = str(self.paths.base_dir), None
        tok = AutoTokenizer.from_pretrained(str(self.paths.checkpoint_dir), local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        torso = StrandsDeciderModel._load_torso(cfg, None, None)
        torso = PeftModel.from_pretrained(torso, str(self.paths.checkpoint_dir / "lora"),
                                         is_trainable=False, local_files_only=True)
        self.model = StrandsDeciderModel(cfg, torso, tok)
        self.model.head.load_state_dict(load_head_state(str(self.paths.checkpoint_dir)))
        self.model.head.float()
        self.model.freeze_torso()
        self.model.to("cpu")
        if device == "cpu":
            from strands_decider.infer import EngineConfig, SystemOneEngine
            self.model = SystemOneEngine(self.model, EngineConfig(device="cpu")).model
        self.model.eval()
        self._active = "cpu"
        self._head_checkpoint = self.paths.checkpoint_dir
        self.temperature = float(self.original_config.get("temperature_by_kind", {}).get(
            "noul", self.original_config.get("temperature", 1.0)))
        self._adapter_hash = sha256_file(self.paths.checkpoint_dir / "lora/adapter_model.safetensors")

    def activate(self) -> float:
        start = time.perf_counter()
        if self._active != self.device:
            self.model.to(self.device)
            self.model.head.float()
            self._active = self.device
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        self.model.eval()
        return time.perf_counter() - start

    def offload(self) -> float:
        start = time.perf_counter()
        if self._active == "cuda":
            self.torch.cuda.synchronize()
            self.model.to("cpu")
            self._active = "cpu"
            self.torch.cuda.empty_cache()
        self.model.eval()
        return time.perf_counter() - start

    def _encode(self, state: Mapping[str, Any], option_order: Sequence[int]):
        from strands_decider.prompting import build_prompt
        from strands_decider.schema import NoulQuestion

        public = validate_public_state(state, allow_history=self.allow_history)
        if sorted(option_order) != [0, 1]:
            raise ValueError("option_order must be a permutation of false/true")
        question = NoulQuestion(instructions=CORRECTNESS_INSTRUCTIONS, criteria=CORRECTNESS_CRITERIA)
        prompt, rendered = build_prompt(public, question, option_order=option_order)
        encoded = self.model.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                                       truncation=False, padding=False)
        count = int(encoded["input_ids"].shape[1])
        if count > self.max_length:
            raise ContextWindowError(count, self.max_length)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        positions = option_token_indices(offsets, rendered.option_spans, len(prompt) - len(rendered.text))
        return public, prompt, encoded, positions, count

    def extract(self, state: Mapping[str, Any], option_order: Sequence[int] = (0, 1)) -> FeatureRecord:
        from strands_decider.modeling import gather_options, pool_last_token

        start = time.perf_counter()
        public, prompt, encoded, positions, count = self._encode(state, option_order)
        self.activate()
        torch = self.torch
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        opt = torch.tensor([positions], dtype=torch.long, device=self.device)
        with torch.no_grad():
            hidden = self.model.encode(ids, mask)
            decide = pool_last_token(hidden, mask).float()[0].cpu().tolist()
            options = gather_options(hidden, opt).float()[0].cpu().tolist()
        if self.device == "cuda":
            torch.cuda.synchronize()
        return FeatureRecord(decide, options, tuple(option_order), count,
                             hashlib.sha256(prompt.encode()).hexdigest(), stable_hash(public),
                             time.perf_counter() - start)

    extract_features = extract

    def score_features(self, feature: FeatureRecord | Mapping[str, Any], *, head=None,
                       temperature: float | None = None) -> ScoreResult:
        """Head-only replay: never activates the torso; physical encoder tokens are zero."""
        start = time.perf_counter()
        feature = FeatureRecord.from_dict(feature.to_dict() if isinstance(feature, FeatureRecord) else feature)
        head = self.model.head if head is None else head
        head.eval()
        device = next(head.parameters()).device
        torch = self.torch
        with torch.no_grad():
            logits = head(torch.tensor([feature.decide], dtype=torch.float32, device=device),
                          torch.tensor([feature.options], dtype=torch.float32, device=device))[0].cpu().tolist()
        canonical = tuple(float(logits[list(feature.option_order).index(i)]) for i in (0, 1))
        temp = (getattr(head, "_haltiq_temperature", self.temperature)
                if temperature is None else temperature)
        p = probability_true(canonical, temp)
        return ScoreResult(p, canonical, feature.input_tokens, time.perf_counter() - start,
                           "true" if p > 0.5 else "false",
                           getattr(head, "_haltiq_model_id", self._head_checkpoint.name),
                           0, True, feature.encoding_seconds, temp)

    def score(self, state: Mapping[str, Any], *, temperature: float | None = None) -> ScoreResult:
        start = time.perf_counter()
        feature = self.extract(state)
        result = self.score_features(feature, temperature=temperature)
        return ScoreResult(**{**result.to_dict(), "elapsed_seconds": time.perf_counter() - start,
                              "physical_input_tokens": feature.input_tokens, "reused_encoding": False})

    def make_head(self, checkpoint_dir: str | Path):
        """Load a second CPU head against exactly the same frozen adapter."""
        from strands_decider.modeling import StrandsDeciderConfig, build_head, load_head_state

        path = Path(checkpoint_dir).resolve()
        _checkpoint_config(path)
        if sha256_file(path / "lora/adapter_model.safetensors") != self._adapter_hash:
            raise ValueError("head checkpoint has a different frozen adapter")
        cfg = StrandsDeciderConfig.from_json(str(path / "strands_decider_config.json"))
        head = build_head(cfg, 2048)
        head.load_state_dict(load_head_state(str(path)))
        head._haltiq_model_id = path.name
        head._haltiq_temperature = float(cfg.temperature_by_kind.get("noul", cfg.temperature))
        return head.float().cpu().eval()

    def load_head(self, checkpoint_dir: str | Path) -> dict:
        path = Path(checkpoint_dir).resolve()
        head = self.make_head(path)
        self.model.head.load_state_dict(head.state_dict())
        cfg = _checkpoint_config(path)
        self.temperature = float(cfg.get("temperature_by_kind", {}).get("noul", cfg.get("temperature", 1.0)))
        self._head_checkpoint = path
        self.model.eval()
        return self.metadata()

    def metadata(self) -> dict:
        path = self._head_checkpoint
        head_path = path / "head.safetensors"
        if not head_path.exists():
            head_path = path / "slot_head.pt"
        return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "base_model": BASE_ID, "base_revision": BASE_REVISION, "code_revision": CODE_REVISION,
                "checkpoint_head_sha256": sha256_file(head_path), "adapter_sha256": self._adapter_hash,
                "model_fingerprint": stable_hash({"head": sha256_file(head_path), "adapter": self._adapter_hash,
                                                   "base": BASE_REVISION, "question": CORRECTNESS_INSTRUCTIONS,
                                                   "criteria": CORRECTNESS_CRITERIA}),
                "max_length": self.max_length, "temperature": self.temperature,
                "head_parameters": sum(p.numel() for p in self.model.head.parameters()),
                "torso_trainable_parameters": sum(p.numel() for p in self.model.torso.parameters() if p.requires_grad),
                "current_device": self._active, "allow_history": self.allow_history,
                "cpu_threads": self.cpu_threads, "checkpoint_name": path.name}


def cache_trajectories(decider: StrandsDecider, trajectory_dir: str | Path, output: str | Path, *,
                       partition: str | None = None, all_rounds: bool = False,
                       both_orders: bool = True, control=None, resume: bool = False,
                       startup_metadata=None) -> dict:
    """Encode public trajectory states once; evaluator labels stay outside state.

    Call under the study's GPU lease after unloading its writer. Resume refuses
    an incomplete inference journal or changed source/model/data fingerprints.
    """
    from .data import Task
    from .metrics import evaluate

    phase_started = time.perf_counter()
    idle_at_phase_entry = getattr(control, "idle_seconds", 0.0)
    source, destination = Path(trajectory_dir), Path(output)
    tasks_file, rounds_file = source / "tasks-evaluator-only.json", source / "trajectories.json"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") not in {"completed", "completed_with_errors"} or source_manifest.get(
        "writer_unload_verified"
    ) is not True:
        raise ValueError("feature extraction requires a terminal writer phase with verified unload")
    tasks = [Task(**r) for r in json.loads(tasks_file.read_text(encoding="utf-8"))]
    if partition is not None:
        tasks = [t for t in tasks if t.partition == partition]
    if not tasks or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("empty or duplicate task selection")
    trajectories = json.loads(rounds_file.read_text(encoding="utf-8"))
    by_id = {r["task_id"]: r for r in trajectories}
    if len(by_id) != len(trajectories) or any(t.id not in by_id for t in tasks):
        raise ValueError("trajectory IDs missing or duplicated")
    metadata = decider.metadata()
    frozen = {"tasks_sha256": sha256_file(tasks_file), "trajectories_sha256": sha256_file(rounds_file),
              "backend_sha256": sha256_file(__file__), "encoder_adapter_sha256": metadata["adapter_sha256"],
              "base_revision": BASE_REVISION, "partition": partition, "all_rounds": all_rounds,
              "both_orders": both_orders, "instructions": CORRECTNESS_INSTRUCTIONS,
              "criteria": CORRECTNESS_CRITERIA, "max_length": decider.max_length}
    manifest_path = Path(str(destination) + ".manifest.json")
    events_path = Path(str(destination) + ".events.jsonl")
    failures_path = Path(str(destination) + ".failures.jsonl")
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or previous["frozen"] != frozen:
            raise ValueError("cache exists or source/data/model changed; explicit matching resume required")
        for path, key in ((destination, "cached_rows_sha256"), (events_path, "events_sha256"),
                          (failures_path, "failures_sha256")):
            recorded = previous.get(key)
            actual = sha256_file(path) if path.exists() else None
            if recorded is not None and recorded != actual:
                raise ValueError(f"changed feature output: {path.name}; preserve and reconcile")
    elif any(p.exists() for p in (destination, events_path, failures_path)):
        raise ValueError("unrecognized existing feature output")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def append(path: Path, value: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    pending, completed, successful_keys, failed_keys = set(), set(), set(), set()
    prior_failures = 0
    if events_path.exists():
        raw = events_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("incomplete feature journal tail; preserve and reconcile")
        for line in raw.decode("utf-8").splitlines():
            event = json.loads(line)
            key = event["key"]
            if event["event"] == "started":
                if key in pending or key in completed:
                    raise ValueError("duplicate feature dispatch")
                pending.add(key)
            elif event["event"] in {"completed", "failed"}:
                if key not in pending:
                    raise ValueError("feature completion without dispatch")
                pending.remove(key)
                completed.add(key)
                prior_failures += int(event["event"] == "failed")
                (successful_keys if event["event"] == "completed" else failed_keys).add(key)
            else:
                raise ValueError("unknown feature journal event")
        if pending:
            raise ValueError("unreconciled feature inference exists; automatic resume refused")
    if destination.exists() and destination.stat().st_size and not destination.read_bytes().endswith(b"\n"):
        raise ValueError("incomplete feature cache tail")
    def stored_keys(path: Path) -> set[str]:
        result = set()
        if not path.exists():
            return result
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError(f"incomplete feature output tail: {path.name}")
        for line in raw.decode("utf-8").splitlines():
            row = json.loads(line)
            order = row["option_order"]
            key = f"{row['question_id']}|{row['round']}|{order[0]}{order[1]}"
            if key in result:
                raise ValueError(f"duplicate stored feature key: {key}")
            result.add(key)
        return result
    if successful_keys != stored_keys(destination) or failed_keys != stored_keys(failures_path):
        raise ValueError("feature outputs and completed/failed journal entries do not reconcile")
    from .control import atomic_json

    manifest = {"status": "running", "frozen": frozen, "encoder": metadata,
                "selected_questions": len(tasks), "resumed": resume,
                "source_trajectories_not_completed": sum(by_id[t.id].get("status") != "completed" for t in tasks),
                "measurement_mode": "real_frozen_encoder_features_not_head_replay"}
    if startup_metadata is not None:
        manifest["backend_startup_this_invocation"] = startup_metadata
    atomic_json(manifest_path, manifest)
    orders = ((0, 1), (1, 0)) if both_orders else ((0, 1),)
    block_start, failed, encoded = None, 0, 0
    try:
        if control is not None:
            control.before_work(require_gpu=decider.device == "cuda")
        block_start = time.perf_counter()
        for task in tasks:
            rows = by_id[task.id]["rounds"]
            for rnd in rows if all_rounds else rows[:1]:
                if rnd.get("status") != "completed":
                    continue
                for order in orders:
                    key = f"{task.id}|{rnd['round']}|{order[0]}{order[1]}"
                    if key in completed:
                        continue
                    if sha256_file(__file__) != frozen["backend_sha256"]:
                        raise RuntimeError("backend changed during feature extraction")
                    if control is not None and time.perf_counter() - block_start >= 5:
                        control.rest_after_block(time.perf_counter() - block_start)
                        control.before_work(require_gpu=decider.device == "cuda")
                        block_start = time.perf_counter()
                    append(events_path, {"event": "started", "key": key})
                    pending.add(key)
                    state = task.public(rnd["draft"])
                    try:
                        feature = decider.extract(state, order)
                    except ContextWindowError as exc:
                        append(failures_path, {"question_id": task.id, "round": rnd["round"],
                               "option_order": order, "failure": "ContextWindowError",
                               "input_tokens": exc.token_count, "max_length": exc.max_length})
                        append(events_path, {"event": "failed", "key": key, "failure": "ContextWindowError"})
                        pending.remove(key)
                        completed.add(key)
                        failed += 1
                        continue
                    # Gold is used only HERE, after the public model forward.
                    label = int(evaluate(rnd["draft"], task.answers)["exact_match"])
                    append(destination, {"question_id": task.id, "partition": task.partition,
                           "round": rnd["round"], "label": label, **feature.to_dict()})
                    append(events_path, {"event": "completed", "key": key,
                                        "input_tokens": feature.input_tokens,
                                        "encoding_seconds": feature.encoding_seconds})
                    pending.remove(key)
                    completed.add(key)
                    encoded += 1
                    manifest.update(new_encoded_rows=encoded, new_overflow_rows=failed,
                                    finished_dispatches=len(completed), current_question_id=task.id)
                    atomic_json(manifest_path, manifest)
                    print(json.dumps({"phase": "feature_cache", "question_id": task.id,
                                      "round": rnd["round"], "new_encoded_rows": encoded,
                                      "finished_dispatches": len(completed)}), flush=True)
        manifest["status"] = ("completed_with_source_failures" if manifest["source_trajectories_not_completed"]
                              else "completed_with_overflows" if failed or prior_failures else "completed")
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        decider.offload()
        manifest.update(new_encoded_rows=encoded, new_overflow_rows=failed,
                        total_overflow_rows=failed + prior_failures, unreconciled_inferences=len(pending),
                        finished_dispatches=len(completed), cached_rows_sha256=(
                            sha256_file(destination) if destination.exists() else None),
                        events_sha256=sha256_file(events_path) if events_path.exists() else None,
                        failures_sha256=sha256_file(failures_path) if failures_path.exists() else None,
                        this_invocation_wall_seconds=time.perf_counter() - phase_started,
                        this_invocation_intentional_idle_seconds=getattr(control, "idle_seconds", 0.0) - idle_at_phase_entry,
                        timing_scope="feature-cache phase only; startup admission/construction/placement/rest are separate")
        events = ([json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines()]
                  if events_path.exists() else [])
        successes = [e for e in events if e["event"] == "completed"]
        manifest["physical_encoding_usage"] = {
            "completed_encoder_forwards": len(successes),
            "input_tokens": sum(e["input_tokens"] for e in successes),
            "encoding_seconds": sum(e["encoding_seconds"] for e in successes),
            "unreconciled_inferences": len(pending),
            "note": "One real frozen forward per stored option order. Later head replay adds zero physical encoder tokens."}
        atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    import argparse
    from .control import GpuLease, RunControl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "cache"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-json")
    parser.add_argument("--trajectory-dir")
    parser.add_argument("--partition")
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--canonical-order-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deadline", required=True, help="ISO8601 with timezone offset")
    parser.add_argument("--gpu-lock", required=True)
    args = parser.parse_args()
    if args.mode == "cache" and not args.trajectory_dir:
        parser.error("cache requires --trajectory-dir")
    if args.mode == "preflight" and not args.state_json:
        parser.error("preflight requires --state-json")
    control = RunControl(Path(str(args.output) + ".control"), args.deadline)
    with GpuLease(Path(args.gpu_lock), "strands-feature-encoding"):
        startup_started = time.perf_counter()
        idle_before_startup = getattr(control, "idle_seconds", 0.0)
        control.before_work(require_gpu=args.device == "cuda")
        decider = None
        try:
            started = time.perf_counter()
            decider = StrandsDecider(DeciderPaths(args.checkpoint, args.base), device=args.device)
            construction = time.perf_counter() - started
            started = time.perf_counter()
            decider.activate()
            startup = {"device": args.device, "explicit_activation": True,
                       "construction_seconds": construction,
                       "placement_seconds": time.perf_counter() - started}
            control.rest_after_block(startup["construction_seconds"] + startup["placement_seconds"])
            startup["intentional_idle_seconds"] = getattr(control, "idle_seconds", 0.0) - idle_before_startup
            startup["wall_seconds"] = time.perf_counter() - startup_started
            if args.mode == "cache":
                cache_trajectories(decider, args.trajectory_dir, args.output, partition=args.partition,
                                   all_rounds=args.all_rounds, both_orders=not args.canonical_order_only,
                                   control=control, resume=args.resume, startup_metadata=startup)
            else:
                control.before_work(require_gpu=args.device == "cuda")
                state = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
                result = decider.score(state).to_dict()
                result["metadata"] = decider.metadata()
                result["backend_startup_this_invocation"] = startup
                Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        finally:
            if decider is not None:
                decider.offload()


if __name__ == "__main__":
    main()
