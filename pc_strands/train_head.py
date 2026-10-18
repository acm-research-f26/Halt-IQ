"""Bounded CPU training of the RELEASED pointer head on frozen cached features.

No torso, pretrained model download or test examples are loaded by this module.
CLI requires evidence that untouched-head evaluation preceded weight updates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Sequence

from .decider import BASE_ID, BASE_REVISION, FeatureRecord, _checkpoint_config, label_for_order, sha256_file


def load_feature_rows(path: str | Path, partition: str) -> list[dict]:
    if partition not in {"fit", "tune"}:
        raise ValueError("training accepts fit/tune only; calibration and test are forbidden")
    rows, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("partition") != partition:
            raise ValueError(f"unexpected partition in {partition} cache")
        if not isinstance(row.get("question_id"), str) or not row["question_id"]:
            raise ValueError("question ID required outside model state")
        if type(row.get("label")) not in (int, bool) or row["label"] not in (0, 1):
            raise ValueError("canonical binary correctness label required")
        feature = FeatureRecord.from_dict(row)
        key = (row["question_id"], tuple(feature.option_order), feature.prompt_sha256)
        if key in seen:
            raise ValueError("duplicate question/order/prompt training record")
        seen.add(key)
        rows.append({**row, "feature": feature})
    if not rows:
        raise ValueError(f"empty {partition} cache")
    return rows


def validate_disjoint(fit: Sequence[dict], tune: Sequence[dict]) -> dict:
    fit_ids = {x["question_id"] for x in fit}
    tune_ids = {x["question_id"] for x in tune}
    if fit_ids & tune_ids:
        raise ValueError("fit/tune question overlap; all variants must remain question-grouped")
    return {"fit_question_ids": sorted(fit_ids), "tune_question_ids": sorted(tune_ids),
            "fit_questions": len(fit_ids), "tune_questions": len(tune_ids)}


def validate_baseline_receipt(receipt: dict, initial_head_sha256: str, *,
                              min_questions: int = 20) -> list[str]:
    """Require a finished, failure-inclusive untouched run before any updates."""
    ids = receipt.get("planned_task_ids", [])
    if (not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids)
            or len(set(ids)) != len(ids) or len(ids) < min_questions):
        raise ValueError("declared untouched baseline question IDs required")
    counts = ("attempted", "completed", "failed", "valid_decision_forwards",
              "pending_requests", "uncertain_requests")
    if any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in counts):
        raise ValueError("explicit integer baseline completion/failure/dispatch counts required")
    if (receipt.get("phase") != "untuned_pretraining"
            or receipt.get("checkpoint_head_sha256") != initial_head_sha256
            or receipt["attempted"] != len(ids)
            or receipt["completed"] + receipt["failed"] != len(ids)
            or receipt["valid_decision_forwards"] < 1
            or receipt["pending_requests"] != 0 or receipt["uncertain_requests"] != 0):
        raise ValueError("finished untouched evaluation with matching head and a real forward required")
    return ids


def load_pointer_head(checkpoint: str | Path):
    from strands_decider.modeling import StrandsDeciderConfig, build_head, load_head_state

    path = Path(checkpoint)
    _checkpoint_config(path)
    head = build_head(StrandsDeciderConfig.from_json(str(path / "strands_decider_config.json")), 2048)
    head.load_state_dict(load_head_state(str(path)))
    return head.float().cpu()


def tensor_value_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().float().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _batch(rows: Sequence[dict], torch):
    decide = torch.tensor([r["feature"].decide for r in rows], dtype=torch.float32)
    options = torch.tensor([r["feature"].options for r in rows], dtype=torch.float32)
    labels = torch.tensor([label_for_order(r["label"], r["feature"].option_order) for r in rows],
                          dtype=torch.long)
    return decide, options, labels


def _nll(head, rows: Sequence[dict], torch, batch_size: int) -> float:
    head.eval()
    by_question = {}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            decide, options, labels = _batch(chunk, torch)
            losses = torch.nn.functional.cross_entropy(head(decide, options), labels, reduction="none").tolist()
            for row, value in zip(chunk, losses):
                by_question.setdefault(row["question_id"], []).append(float(value))
    return sum(sum(values) / len(values) for values in by_question.values()) / len(by_question)


def _save_checkpoint(source: Path, output: Path, state: dict, proof: dict) -> None:
    from safetensors.torch import save_file

    if output.exists() and any(output.iterdir()):
        raise ValueError("output checkpoint must be absent or empty; original weights stay immutable")
    output.mkdir(parents=True, exist_ok=True)
    config = _checkpoint_config(source)
    config.update(base_model=BASE_ID, base_revision=BASE_REVISION, temperature=1.0,
                  temperature_by_kind={})
    (output / "strands_decider_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file({k: v.detach().cpu().float().contiguous() for k, v in state.items()},
              str(output / "head.safetensors"))
    shutil.copytree(source / "lora", output / "lora", dirs_exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                 "merges.txt", "added_tokens.json", "chat_template.jinja"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    (output / "training-proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")


def train_cached_head(checkpoint: str | Path, fit_cache: str | Path, tune_cache: str | Path,
                      output: str | Path, baseline_receipt: str | Path, *,
                      learning_rates: Sequence[float] = (1e-5, 5e-5), epochs: int = 5,
                      batch_size: int = 32, seed: int = 42, cpu_threads: int = 2,
                      max_seconds: float = 900.0, control=None,
                      min_baseline_questions: int = 20) -> dict:
    source, destination = Path(checkpoint).resolve(), Path(output).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("new checkpoint must not overlap the released snapshot")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output must be empty")
    if not 1 <= epochs <= 5 or batch_size < 1 or not 1 <= cpu_threads <= 4 or max_seconds <= 0:
        raise ValueError("invalid bounded training settings")
    if not learning_rates or any(not math.isfinite(x) or not 0 < x <= 1e-3 for x in learning_rates):
        raise ValueError("invalid learning rate")
    head_path = source / "head.safetensors"
    if not head_path.is_file():
        head_path = source / "slot_head.pt"
    initial_file_hash = sha256_file(head_path)
    receipt = json.loads(Path(baseline_receipt).read_text(encoding="utf-8"))
    baseline_ids = validate_baseline_receipt(receipt, initial_file_hash,
                                            min_questions=min_baseline_questions)
    for filename, key in (("manifest.json", "manifest_sha256"), ("episodes.json", "episodes_sha256")):
        file_path = Path(baseline_receipt).parent / filename
        if not file_path.is_file() or receipt.get(key) != sha256_file(file_path):
            raise ValueError("untouched evaluation receipt does not match its final result files")
    fit = load_feature_rows(fit_cache, "fit")
    tune = load_feature_rows(tune_cache, "tune")
    groups = validate_disjoint(fit, tune)
    if set(baseline_ids) & (set(groups["fit_question_ids"]) | set(groups["tune_question_ids"])):
        raise ValueError("training questions overlap the pretraining development baseline")
    import torch

    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    original = load_pointer_head(source)
    initial = {k: v.detach().clone() for k, v in original.state_dict().items()}
    initial_hash = tensor_value_hash(initial)
    adapter_hash = sha256_file(source / "lora/adapter_model.safetensors")
    feature_manifest_hashes = {}
    from .decider import CORRECTNESS_CRITERIA, CORRECTNESS_INSTRUCTIONS
    for cache_path, expected_partition in ((fit_cache, "fit"), (tune_cache, "tune")):
        manifest_path = Path(str(cache_path) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = manifest.get("frozen", {})
        if (not manifest.get("status", "").startswith("completed")
                or manifest.get("unreconciled_inferences", 0) != 0
                or manifest.get("cached_rows_sha256") != sha256_file(cache_path)
                or frozen.get("encoder_adapter_sha256") != adapter_hash
                or frozen.get("base_revision") != BASE_REVISION
                or frozen.get("partition") != expected_partition
                or frozen.get("instructions") != CORRECTNESS_INSTRUCTIONS
                or frozen.get("criteria") != CORRECTNESS_CRITERIA):
            raise ValueError("feature cache provenance is incomplete, changed or incompatible with this frozen encoder")
        feature_manifest_hashes[expected_partition] = sha256_file(manifest_path)
    started, history = time.perf_counter(), []
    block_started = time.perf_counter()
    progress_path = destination.parent / (destination.name + ".progress.json")
    from .control import ResourcePressure, StudyStopped, atomic_json
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if control is not None:
        control.before_work(require_gpu=False)
    best_loss, best_state, best_settings = float("inf"), None, None
    stopped, stop_reason = False, None
    for lr in learning_rates:
        torch.manual_seed(seed)
        head = copy.deepcopy(original)
        head.requires_grad_(True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01,
                                     betas=(0.9, 0.95), eps=1e-8)
        candidate_best, stale = float("inf"), 0
        for epoch in range(epochs):
            order = list(range(len(fit)))
            random.Random(seed + epoch).shuffle(order)
            head.train()
            losses, steps, gradient_seen = [], 0, False
            for start in range(0, len(order), batch_size):
                if control is not None and time.perf_counter() - block_started >= 5.0:
                    try:
                        control.rest_after_block(time.perf_counter() - block_started)
                        control.before_work(require_gpu=False)
                    except (ResourcePressure, StudyStopped) as exc:
                        stopped, stop_reason = True, type(exc).__name__
                        break
                    block_started = time.perf_counter()
                if time.perf_counter() - started > max_seconds:
                    stopped, stop_reason = True, "training_time_cap"
                    break
                chunk = [fit[i] for i in order[start:start + batch_size]]
                decide, options, labels = _batch(chunk, torch)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(decide, options), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite head training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("nonfinite head gradients")
                gradient_seen |= float(norm) > 0
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
                atomic_json(progress_path, {"status": "training", "lr": lr, "epoch": epoch + 1,
                            "step_in_epoch": steps, "rows_seen_this_epoch": min(start + batch_size, len(order)),
                            "fit_loss_last_batch": losses[-1], "elapsed_seconds": time.perf_counter() - started})
            if stopped:
                break  # choose only fully completed epochs
            val = _nll(head, tune, torch, batch_size)
            entry = {"lr": lr, "epoch": epoch + 1, "fit_loss": sum(losses) / len(losses),
                     "tune_nll": val, "optimizer_steps": steps, "nonzero_gradient_seen": gradient_seen}
            history.append(entry)
            if val < best_loss:
                best_loss = val
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                best_settings = {"lr": lr, "epoch": epoch + 1}
            if val < candidate_best:
                candidate_best, stale = val, 0
            else:
                stale += 1
            if stale >= 2:
                break
        if stopped:
            break
    if best_state is None:
        raise RuntimeError("no complete training epoch; no trained-checkpoint claim is possible")
    final_value_hash = tensor_value_hash(best_state)
    delta_norm = math.sqrt(sum(float((best_state[k] - initial[k]).square().sum()) for k in initial))
    if final_value_hash == initial_hash or delta_norm <= 0:
        raise RuntimeError("head tensor values did not change")
    if sha256_file(head_path) != initial_file_hash or sha256_file(
        source / "lora/adapter_model.safetensors"
    ) != adapter_hash:
        raise RuntimeError("original head or frozen adapter changed")
    proof = {"status": "trained_bounded_early_stop" if stopped else "trained",
             "target": "one official-EM correctness proxy; not separate rubric labels",
             "head_parameters": sum(p.numel() for p in original.parameters()),
             "frozen_torso_adapter": True, "original_head_file_sha256": initial_file_hash,
             "frozen_adapter_sha256": adapter_hash, "initial_head_tensor_sha256": initial_hash,
             "trained_head_tensor_sha256": final_value_hash, "head_delta_l2": delta_norm,
             "fit_cache_sha256": sha256_file(fit_cache), "tune_cache_sha256": sha256_file(tune_cache),
             "feature_manifest_sha256": feature_manifest_hashes,
             "baseline_receipt_sha256": sha256_file(baseline_receipt),
             "untuned_baseline_questions": len(baseline_ids),
             "untuned_baseline_successful_questions": receipt["completed"],
             "untuned_baseline_failed_questions": receipt["failed"],
             "untuned_baseline_valid_forwards": receipt["valid_decision_forwards"], **groups,
             "fit_rows": len(fit), "tune_rows": len(tune), "seed": seed,
             "history": history, "selected": best_settings, "selection": "minimum tuning-question NLL",
             "elapsed_seconds": time.perf_counter() - started, "calibration_used": False,
             "fresh100_used": False, "source_head_unchanged": True, "source_adapter_unchanged": True,
             "bounded_stop_reason": stop_reason, "resume_supported": False}
    _save_checkpoint(source, destination, best_state, proof)
    restored = load_pointer_head(destination)
    restored_hash = tensor_value_hash(restored.state_dict())
    if restored_hash != final_value_hash:
        raise RuntimeError("save/reload tensor values differ")
    original.load_state_dict(best_state)
    original.eval()
    decide, options, _ = _batch(tune[:batch_size], torch)
    with torch.no_grad():
        max_error = float((original(decide, options) - restored.eval()(decide, options)).abs().max())
    if max_error > 1e-6:
        raise RuntimeError("checkpoint reload logits differ")
    if sha256_file(destination / "lora/adapter_model.safetensors") != adapter_hash:
        raise RuntimeError("copied frozen adapter differs")
    proof.update(reload_max_logit_error=max_error, reload_verified=True,
                 checkpoint_head_sha256=sha256_file(destination / "head.safetensors"))
    (destination / "training-proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    atomic_json(progress_path, {"status": proof["status"], "selected": best_settings,
                               "reload_verified": True, "head_delta_l2": delta_norm})
    return proof


def copy_control_checkpoint(checkpoint: str | Path, output: str | Path, *,
                            forbidden_question_ids: Sequence[str] = ()) -> dict:
    """A new, unchanged head copy for local calibration; never edits the release."""
    source, destination = Path(checkpoint).resolve(), Path(output).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("control copy must not overlap source")
    head = load_pointer_head(source)
    proof = {"status": "untuned_control_copy", "weights_updated": False,
             "fit_question_ids": list(forbidden_question_ids), "tune_question_ids": [],
             "head_tensor_sha256": tensor_value_hash(head.state_dict())}
    _save_checkpoint(source, destination, head.state_dict(), proof)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "fit-cache", "tune-cache", "output", "baseline-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--deadline", required=True, help="ISO8601 with timezone offset")
    args = parser.parse_args()
    from .control import RunControl
    control = RunControl(Path(str(args.output) + ".control"), args.deadline)
    result = train_cached_head(args.checkpoint, args.fit_cache, args.tune_cache, args.output,
                               args.baseline_receipt, epochs=args.epochs, max_seconds=args.max_seconds,
                               control=control)
    print(json.dumps({k: result[k] for k in ("status", "selected", "head_delta_l2", "reload_verified")}, indent=2))


if __name__ == "__main__":
    main()
