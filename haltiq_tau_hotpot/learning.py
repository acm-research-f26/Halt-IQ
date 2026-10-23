"""CPU-only supervised updates of Hobson's existing pointer head.

FIT updates weights, TUNE selects a fixed learning-rate/epoch grid, and CAL fits
one scalar temperature separately for each criterion and released/trained head.
No torso is loaded, no LoRA weights are trained, and no RL claim follows.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import time

from .hobson import (HEAD_PARAMETERS, INSTRUCTIONS_BY_CRITERION, encoder_binding,
    feature_tensors, file_hash, load_head, stable_hash)


def tensor_hash(state):
    import hashlib
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().float().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_partitions(fit, tune, calibration, *, minimum_questions=None):
    """Question grouping applies across criteria, revisions and feature variants."""
    minima = minimum_questions or {"fit": 40, "tune": 10, "calibration": 10}
    groups, seen = {}, set()
    for partition, rows in (("fit", fit), ("tune", tune), ("calibration", calibration)):
        if not rows:
            raise ValueError("Every FIT/TUNE/CAL partition must contain valid observations")
        group, keys = set(), set()
        for row in rows:
            if (row.get("partition") != partition or not isinstance(row.get("question_id"), str)
                    or not row["question_id"] or row.get("criterion") not in INSTRUCTIONS_BY_CRITERION
                    or type(row.get("label")) not in (bool, int) or row["label"] not in (0, 1)):
                raise ValueError("Explicit grouped partition, criterion and evaluator-only binary label required")
            feature = row["feature"]
            key = (row["question_id"], row["criterion"], feature.get("provenance_sha256"))
            if key in keys:
                raise ValueError("Duplicate question/criterion/input feature record")
            keys.add(key)
            group.add(row["question_id"])
        if len(group) < minima[partition] or seen & group:
            raise ValueError("Insufficient unique questions or FIT/TUNE/CAL question overlap")
        seen.update(group)
        groups[partition] = sorted(group)
    criteria = sorted({r["criterion"] for r in fit})
    if "answer" not in criteria or any({r["criterion"] for r in rows} != set(criteria)
                                       for rows in (tune, calibration)):
        raise ValueError("Answer criterion and matching criterion coverage are required")
    for criterion in criteria:
        if {int(r["label"]) for r in fit if r["criterion"] == criterion} != {0, 1}:
            raise ValueError("Both labels are required for each FIT criterion; never resample TUNE/CAL")
    return groups, criteria


def validate_baseline(receipt, released_head_sha256, forbidden_ids, *, minimum_questions=4):
    ids = receipt.get("planned_question_ids", [])
    counts = [receipt.get(k) for k in ("completed", "failed", "valid_native_forwards")]
    if (receipt.get("phase") != "untouched_pretraining" or receipt.get("terminal") is not True
            or receipt.get("pending_or_uncertain") is not False or receipt.get("cleanup_verified") is not True
            or receipt.get("checkpoint_head_sha256") != released_head_sha256
            or not isinstance(ids, list) or not all(isinstance(q, str) and q for q in ids)
            or len(set(ids)) != len(ids) or len(ids) < minimum_questions
            or any(type(n) is not int or n < 0 for n in counts)
            or counts[0] != len(ids) or counts[1] != 0 or counts[2] != len(ids)
            or set(ids) & set(forbidden_ids)):
        raise ValueError("Finished disjoint untouched native baseline must precede all weight updates")
    return sorted(ids)


def _rows_hash(rows):
    return stable_hash([{k: row[k] for k in ("question_id", "partition", "criterion", "label")} | {
        "provenance_sha256": row["feature"]["provenance_sha256"],
        "tensor_sha256": tensor_hash(dict(zip(("decide", "options"), feature_tensors(row["feature"]))))}
        for row in rows])


def _batch(rows, torch):
    values = [feature_tensors(r["feature"], criterion=r["criterion"]) for r in rows]
    return (torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values]),
            torch.tensor([int(r["label"]) for r in rows], dtype=torch.long))


def head_logits(head, rows, *, batch_size=16, accounting=None, role=None):
    import torch
    values = []
    head.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            decide, options, _ = _batch(rows[offset:offset + batch_size], torch)
            logits = head(decide, options)
            if accounting is not None:
                record = accounting.setdefault(role, {"head_forward_batches": 0, "head_scored_states": 0})
                record["head_forward_batches"] += 1
                record["head_scored_states"] += len(decide)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite head logits")
            values.extend(logits.float().tolist())
    return values


def grouped_nll(logits, rows, temperature=1.0):
    if not logits or len(logits) != len(rows) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Matched finite logits and positive temperature required")
    grouped = {}
    for pair, row in zip(logits, rows):
        if len(pair) != 2 or not all(math.isfinite(x) for x in pair):
            raise ValueError("Canonical false/true finite logits required")
        a, b = [x / temperature for x in pair]
        maximum = max(a, b)
        loss = maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum)) - (b if row["label"] else a)
        grouped.setdefault(row["question_id"], []).append(loss)
    return sum(sum(v) / len(v) for v in grouped.values()) / len(grouped)


def fit_temperature(logits, rows):
    """Fixed bounded deterministic scalar search; CAL fitting NLL is not test quality."""
    low, high = math.log(.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = high - ratio * (high - low), low + ratio * (high - low)
    for _ in range(64):
        if grouped_nll(logits, rows, math.exp(c)) <= grouped_nll(logits, rows, math.exp(d)):
            high, d = d, c
            c = high - ratio * (high - low)
        else:
            low, c = c, d
            d = low + ratio * (high - low)
    candidates = [.05, 1.0, 20.0, math.exp((low + high) / 2)]
    temperature = min(candidates, key=lambda t: grouped_nll(logits, rows, t))
    return {"temperature": temperature, "calibration_fitting_nll": grouped_nll(logits, rows, temperature),
            "uncalibrated_calibration_nll": grouped_nll(logits, rows), "bounds": [.05, 20.0],
            "selection": "minimum question-grouped CAL NLL; not held-out quality"}


class _Budget:
    def __init__(self, control, max_seconds):
        self.control, self.started = control, time.perf_counter()
        self.block = self.started
        self.block_idle = getattr(control, "idle_seconds", 0.)
        self.max_seconds = max_seconds
        if control is not None:
            control.before_work(require_gpu=False)

    def check(self):
        if time.perf_counter() - self.started >= self.max_seconds:
            raise TimeoutError("Bounded head-learning time expired; no complete proof")
        if self.control is not None:
            self.control.before_work(require_gpu=False)
            active = max(0., time.perf_counter() - self.block
                         - (getattr(self.control, "idle_seconds", 0.) - self.block_idle))
            if active >= 5:
                self.control.rest_after_block(active)
                self.control.before_work(require_gpu=False)
                self.block = time.perf_counter()
                self.block_idle = getattr(self.control, "idle_seconds", 0.)

    def finish(self):
        if self.control is not None:
            self.control.rest_after_block(max(0., time.perf_counter() - self.block
                - (getattr(self.control, "idle_seconds", 0.) - self.block_idle)))


def train_heads(checkpoint, base, fit_rows, tune_rows, calibration_rows, output, baseline_receipt,
                *, learning_rates=(1e-5, 5e-5), epochs=3, batch_size=16, seed=42,
                max_seconds=900, control=None, minimum_questions=None):
    """Train only compatible existing heads; serialize into a new external directory.

    Rows carry evaluator labels outside ``feature``. The caller must preserve
    failures/denominators in its own stage receipt and journal the training phase.
    This function intentionally offers no resume or implicit retry.
    """
    checkpoint, base, output = Path(checkpoint).resolve(), Path(base).resolve(), Path(output).resolve()
    if (output.exists() or checkpoint == output or checkpoint in output.parents or output in checkpoint.parents
            or base == output or base in output.parents or output in base.parents):
        raise ValueError("A new output directory must not overlap immutable model sources")
    if (type(epochs) is not int or not 1 <= epochs <= 5 or type(batch_size) is not int
            or not 1 <= batch_size <= 64 or not math.isfinite(max_seconds) or not 0 < max_seconds <= 3600
            or not learning_rates or len(learning_rates) > 4
            or any(not math.isfinite(lr) or not 0 < lr <= 1e-3 for lr in learning_rates)):
        raise ValueError("Bounded fixed training grid required")
    fit, tune, calibration = list(fit_rows), list(tune_rows), list(calibration_rows)
    groups, criteria = validate_partitions(fit, tune, calibration, minimum_questions=minimum_questions)
    frozen = encoder_binding(checkpoint, base)
    released_sha = file_hash(checkpoint / "head.safetensors")
    baseline = json.loads(Path(baseline_receipt).read_text(encoding="utf-8")) if isinstance(baseline_receipt, (str, Path)) else baseline_receipt
    baseline_ids = validate_baseline(baseline, released_sha, [q for ids in groups.values() for q in ids])
    for row in fit + tune + calibration:
        feature_tensors(row["feature"], expected_encoder=frozen, criterion=row["criterion"])
    import torch
    from safetensors.torch import save_file, load_file
    torch.set_num_threads(4)
    original = load_head(checkpoint)
    initial = {k: v.detach().clone() for k, v in original.state_dict().items()}
    initial_tensor_sha = tensor_hash(initial)
    cache_hashes = {name: _rows_hash(rows) for name, rows in (("fit", fit), ("tune", tune), ("calibration", calibration))}
    budget = _Budget(control, max_seconds)
    output.mkdir(parents=True, exist_ok=False)
    proof = {"status": "incomplete", "method": "supervised existing Hobson pointer-head CE; no RL or LoRA updates",
        "head_parameters": sum(p.numel() for p in original.parameters()), "encoder_sha256": frozen,
        "released_head_sha256": released_sha, "initial_head_tensor_sha256": initial_tensor_sha,
        "partitions": groups, "baseline_question_ids": baseline_ids, "baseline_receipt_sha256": stable_hash(baseline),
        "cache_sha256": cache_hashes, "seed": seed, "criteria": {}, "resume_supported": False}
    if proof["head_parameters"] != HEAD_PARAMETERS:
        raise ValueError("Unexpected Hobson head capacity")
    for criterion in criteria:
        train = [r for r in fit if r["criterion"] == criterion]
        validation = [r for r in tune if r["criterion"] == criterion]
        cal = [r for r in calibration if r["criterion"] == criterion]
        best_loss, best, selected, history, total_steps = math.inf, None, None, [], 0
        physical = {}
        for lr in learning_rates:
            torch.manual_seed(seed)
            head = copy.deepcopy(original).requires_grad_(True)
            optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01, betas=(.9, .95))
            for epoch in range(epochs):
                order = list(range(len(train)))
                random.Random(seed + epoch).shuffle(order)
                head.train()
                epoch_steps, fit_losses, gradient_seen = 0, [], False
                for offset in range(0, len(order), batch_size):
                    budget.check()
                    chunk = [train[i] for i in order[offset:offset + batch_size]]
                    decide, options, labels = _batch(chunk, torch)
                    optimizer.zero_grad(set_to_none=True)
                    # Inverse-multiplicity weights reduce variant overrepresentation.
                    losses = torch.nn.functional.cross_entropy(head(decide, options), labels, reduction="none")
                    multiplicity = {q: sum(r["question_id"] == q for r in train) for q in {r["question_id"] for r in chunk}}
                    weights = torch.tensor([1 / multiplicity[r["question_id"]] for r in chunk])
                    loss = (losses * weights).sum() / weights.sum()
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite head training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    if not bool(torch.isfinite(norm)):
                        raise RuntimeError("Nonfinite head gradients")
                    gradient_seen |= float(norm) > 0
                    optimizer.step()
                    work = physical.setdefault("fit", {"head_forward_batches": 0, "head_scored_states": 0})
                    work["head_forward_batches"] += 1
                    work["head_scored_states"] += len(chunk)
                    epoch_steps += 1
                    total_steps += 1
                    fit_losses.append(float(loss.detach()))
                budget.check()
                tune_nll = grouped_nll(head_logits(head, validation, batch_size=batch_size,
                    accounting=physical, role="tune"), validation)
                history.append({"learning_rate": lr, "epoch": epoch + 1, "optimizer_steps": epoch_steps,
                    "nonzero_gradient_seen": gradient_seen,
                    "fit_batch_mean_loss": sum(fit_losses) / len(fit_losses), "tune_nll": tune_nll})
                if tune_nll < best_loss:
                    best_loss, selected = tune_nll, {"learning_rate": lr, "epoch": epoch + 1}
                    best = {k: v.detach().clone() for k, v in head.state_dict().items()}
        original_trained = copy.deepcopy(original)
        original_trained.load_state_dict(best, strict=True)
        original_trained.eval().requires_grad_(False)
        delta = math.sqrt(sum(float((best[k] - initial[k]).square().sum()) for k in initial))
        if tensor_hash(best) == initial_tensor_sha or not math.isfinite(delta) or delta <= 0:
            raise RuntimeError("No actual head weight change")
        budget.check()
        destination = output / (criterion + "-head.safetensors")
        save_file({k: v.detach().cpu().float().contiguous() for k, v in best.items()}, str(destination))
        restored = copy.deepcopy(original)
        restored.load_state_dict(load_file(str(destination), device="cpu"), strict=True)
        restored.eval().requires_grad_(False)
        before = head_logits(original_trained, validation, batch_size=batch_size, accounting=physical, role="reload_before")
        after = head_logits(restored, validation, batch_size=batch_size, accounting=physical, role="reload_after")
        reload_error = max(abs(a - b) for first, second in zip(before, after) for a, b in zip(first, second))
        if tensor_hash(restored.state_dict()) != tensor_hash(best) or reload_error != 0:
            raise RuntimeError("Saved/reloaded CPU head does not reproduce exact logits")
        released_cal = head_logits(original, cal, batch_size=batch_size, accounting=physical, role="released_calibration")
        trained_cal = head_logits(restored, cal, batch_size=batch_size, accounting=physical, role="trained_calibration")
        label_counts = {partition: {str(label): sum(int(r["label"]) == label for r in rows if r["criterion"] == criterion)
            for label in (0, 1)} for partition, rows in (("fit", fit), ("tune", tune), ("calibration", calibration))}
        proof["criteria"][criterion] = {"selected": selected, "history": history,
            "total_optimizer_steps": total_steps, "head_delta_l2": delta,
            "physical_head_work": physical, "backward_passes": total_steps,
            "physical_encoder_forwards": 0, "torso_loaded": False,
            "label_counts": label_counts,
            "warnings": [p + " contains one class; retain all cases, NLL/calibration remain defined, AUROC is undefined"
                         for p, counts in label_counts.items() if 0 in counts.values()],
            "trained_head_sha256": file_hash(destination), "trained_head_tensor_sha256": tensor_hash(best),
            "reload_max_logit_error": reload_error, "reload_verified": True,
            "released_calibration": fit_temperature(released_cal, cal),
            "trained_calibration": fit_temperature(trained_cal, cal),
            "calibration_questions": len({r["question_id"] for r in cal}),
            "calibration_logits_false_true": {"released": released_cal, "trained": trained_cal},
            "calibration_labels": [int(r["label"]) for r in cal]}
    budget.check()
    if encoder_binding(checkpoint, base) != frozen or file_hash(checkpoint / "head.safetensors") != released_sha:
        raise RuntimeError("Immutable released torso/adapter/head files changed")
    budget.finish()
    proof.update(status="completed", frozen_torso_adapter_files_unchanged=True,
        released_head_file_unchanged=True, elapsed_seconds=time.perf_counter() - budget.started,
        calibration_scope="FIT/TUNE/CAL only; no held-out quality or live rollout gain is claimed")
    (output / "learning-proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return proof
