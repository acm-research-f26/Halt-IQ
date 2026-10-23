"""One bounded, journaled execution of a frozen20-question shared pilot."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import socket
import time

from pc_strands.control import GpuLease, RunControl, StudyStopped, atomic_json, parse_deadline, utc_now

from .metrics import summarize
from .native import BASE_REVISION, HOBSON_REVISION, MINILM_REVISION, UPSTREAM_REVISION, PINS, MiniLMRanker, OfficialDecider
from .protocol import ARMS, baseline_evidence, digest, prepare_cohort, ranked_evidence, stable_hash, state_text, validate_cohort

REPO = Path(__file__).resolve().parents[1]


def ensure_port_closed():
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", 11435)) == 0:
            raise RuntimeError("Private writer port is open")


def startup_admission(control):
    samples = []
    started = time.perf_counter()
    for index in range(4):
        if index:
            control._sleep(10)
        control.before_work(require_gpu=True)
        samples.append(dict(control.last_resources))
    return {"samples": samples, "wall_seconds": time.perf_counter() - started}


def frozen_files(args):
    roots = {"hobson": args.checkpoint_dir.resolve(), "base": args.base_dir.resolve(),
             "minilm": args.reranker_dir.resolve()}
    if args.current_checkpoint:
        roots["current"] = args.current_checkpoint.resolve()
    paths = {}
    for kind, root in roots.items():
        if not root.is_dir():
            raise ValueError("All models must already exist locally; this runner never downloads")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                paths[kind + "/" + path.relative_to(root).as_posix()] = path
    paths.update({"source/" + path.name: path for path in Path(__file__).parent.glob("*.py")})
    paths.update({"cohort": args.cohort, "partner_csv": args.partner_csv, "dataset": args.dataset,
        "controls": REPO / "pc_strands/control.py", "data_validation": REPO / "pc_strands/data.py",
        "answer_metrics": REPO / "pc_strands/metrics.py"})
    spec = importlib.util.find_spec("strands_decider")
    if not spec or not spec.origin:
        raise ValueError("Use the installed local Strands runtime")
    upstream = Path(spec.origin).parent
    for name in ("modeling.py", "infer.py", "prompting.py", "schema.py"):
        paths["upstream/" + name] = upstream / name
    hashes = {key: digest(path) for key, path in paths.items()}
    for key, expected in PINS.items():
        if hashes.get(key) != expected:
            raise ValueError("Pinned model file differs: " + key)
    if args.current_checkpoint:
        current = {"head.safetensors": "ace56742545b505e6fae34ada85c8edbeedb23ba8974d8680aebe0380d6083a1",
            "lora/adapter_model.safetensors": "ae1d0cd1ac3edec6df0a519a3ad18f1c382099032d493f8f98c96a80cfebb4c7"}
        if any(hashes.get("current/" + key) != value for key, value in current.items()):
            raise ValueError("Optional current arm requires the actual unchanged released model")
    return paths, hashes


def journal(path, event):
    # A pending row is durably saved before dispatch; no automatic resume/retry exists.
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"at": utc_now(), **event}, allow_nan=False) + "\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())


def run(args, *, ranker_factory=MiniLMRanker, decider_factory=OfficialDecider,
        control_class=RunControl, freeze=frozen_files):
    if digest(args.cohort) != args.cohort_sha256:
        raise ValueError("Explicit cohort hash does not match")
    cohort = validate_cohort(json.loads(args.cohort.read_text(encoding="utf-8")))
    if prepare_cohort(args.partner_csv, args.dataset) != cohort:
        raise ValueError("Cohort is not the exact answer-blind source selection")
    models = ["hobson"] + (["current"] if args.current_checkpoint else [])
    if not args.allow_model_calls:
        return {"status": "dry_run_no_model_import_or_calls", "questions": 20,
                "planned_native_forwards": 40 * len(models), "planned_retrieval_batches": 20,
                "scope": cohort["scope"], "cohort_sha256": args.cohort_sha256}
    output = args.output.resolve()
    roots = [args.checkpoint_dir.resolve(), args.base_dir.resolve(), args.reranker_dir.resolve()]
    if args.current_checkpoint:
        roots.append(args.current_checkpoint.resolve())
    if output == REPO or REPO in output.parents or any(output == p or p in output.parents or output in p.parents for p in roots):
        raise ValueError("A new external output must not overlap source or model directories")
    if not args.lease_path.is_file():
        raise ValueError("Use the existing shared study GPU lease file")
    now = datetime.now(timezone.utc)
    deadline = parse_deadline(args.deadline) if args.deadline else now + timedelta(hours=1)
    if not 0 < (deadline - now).total_seconds() <= 3600:
        raise ValueError("The pilot deadline must be within the next hour with an explicit offset")
    wall_started = time.perf_counter()
    paths, hashes = freeze(args)
    if datetime.now(timezone.utc) >= deadline:
        raise StudyStopped("Deadline expired during input verification")
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "pilot.json"
    log = output / "physical-work.jsonl"
    records = [{"id": c["id"], "type": c["type"], "model": model, "arm": arm,
                "status": "not_run"} for model in models for c in cohort["cases"] for arm in ARMS]
    result = {"scope": cohort["scope"], "status": "waiting_for_admission", "started_at": now.isoformat(),
        "initial_input_verification_seconds": time.perf_counter() - wall_started,
        "deadline": deadline.isoformat(), "deadline_kind": "cooperative_safe_boundary",
        "cohort_sha256": args.cohort_sha256, "frozen_before": hashes,
        "model_revisions": {"hobson": HOBSON_REVISION, "minilm": MINILM_REVISION,
                            "base": BASE_REVISION, "upstream_code": UPSTREAM_REVISION},
        "physical_encoder_forwards": 0, "physical_reranker_forwards": 0, "head_scores": 0,
        "physical_encoder_tokens": 0, "physical_reranker_tokens": 0,
        "native_pending_or_uncertain": False, "retrieval_pending_or_uncertain": False,
        "new_writer_calls": 0, "weight_updates": 0, "records": records, "retrieval": [],
        "startup": [], "admissions": [], "gpu_lease_released": False, "owned_models_released": False,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "safetensors", "tokenizers")}}
    atomic_json(receipt, result)
    guard = control_class(output / "control", deadline.isoformat(), duty_cycle=.5, min_rest_seconds=5,
        minimum_ram_gib=7, minimum_free_vram_gib=3)
    admission = control_class(output / "admission", deadline.isoformat(), duty_cycle=.5, min_rest_seconds=5,
        minimum_ram_gib=8.5, minimum_free_vram_gib=9, maximum_pause_seconds=0)
    ranker, decider, error = None, None, None
    try:
        with GpuLease(args.lease_path, "hotpot-partner-shared20-method-check"):
            result["gpu_lease_held_before_model_import"] = True
            try:
                ensure_port_closed()
                result["admissions"].append({"model": "minilm_cpu", **startup_admission(admission)})
                ensure_port_closed()
                started = time.perf_counter()
                ranker = ranker_factory(args.reranker_dir)
                result["startup"].append({"model": "minilm_cpu", "seconds": time.perf_counter() - started})
                atomic_json(receipt, result)
                guard.rest_after_block(result["startup"][-1]["seconds"])
                evidence = {}
                for case in cohort["cases"]:
                    public = case["public"]
                    guard.before_work(require_gpu=False)
                    result["retrieval_pending_or_uncertain"] = True
                    atomic_json(receipt, result)
                    journal(log, {"operation": "retrieval", "id": case["id"], "status": "pending"})
                    measured = ranker.rank(public)
                    result["physical_reranker_forwards"] += measured["physical_forwards"]
                    result["physical_reranker_tokens"] += measured["input_tokens"]
                    top, indices = ranked_evidence(public, measured["scores"])
                    evidence[case["id"]] = {ARMS[0]: baseline_evidence(public), ARMS[1]: top}
                    journal(log, {"operation": "retrieval", "id": case["id"], "status": "completed", **measured})
                    result["retrieval"].append({"id": case["id"], "selected_paragraph_indices": indices,
                        "public_evidence_sha256": {arm: stable_hash(text) for arm, text in evidence[case["id"]].items()}, **measured})
                    result["retrieval_pending_or_uncertain"] = False
                    atomic_json(receipt, result)
                    guard.rest_after_block(measured["seconds"])
                ranker.close()
                ranker = None
                for model in models:
                    ensure_port_closed()
                    result["admissions"].append({"model": model, **startup_admission(admission)})
                    atomic_json(receipt, result)
                    ensure_port_closed()
                    started = time.perf_counter()
                    decider = decider_factory(args.checkpoint_dir if model == "hobson" else args.current_checkpoint,
                                               args.base_dir, kind=model)
                    decider.activate()
                    result["startup"].append({"model": model, "seconds": time.perf_counter() - started,
                                               "metadata": decider.metadata})
                    atomic_json(receipt, result)
                    guard.rest_after_block(result["startup"][-1]["seconds"])
                    for case in cohort["cases"]:
                        for arm in ARMS:
                            row = next(r for r in records if (r["id"], r["model"], r["arm"]) == (case["id"], model, arm))
                            state = state_text(case["public"], evidence[case["id"]][arm])
                            row["public_state_sha256"] = stable_hash(state)
                            try:
                                prepared = decider.prepare(state)
                            except ValueError as exc:
                                # Tokenization-only overflow is a failed planned score, never a native forward.
                                if "context window" not in str(exc):
                                    raise
                                row.update(status="context_window_failure", error_type=type(exc).__name__)
                                journal(log, {"operation": "native", "id": case["id"], "model": model,
                                              "arm": arm, "status": "preflight_failed", "physical_forwards": 0})
                                atomic_json(receipt, result)
                                continue
                            guard.before_work(require_gpu=True)
                            row["status"] = "native_pending"
                            result.update(status="running", native_pending_or_uncertain=True)
                            journal(log, {"operation": "native", "id": case["id"], "model": model,
                                          "arm": arm, "status": "pending"})
                            atomic_json(receipt, result)
                            measured = decider.score(prepared)
                            result["physical_encoder_forwards"] += measured["physical_forwards"]
                            result["head_scores"] += measured["head_scores"]
                            result["physical_encoder_tokens"] += measured["input_tokens"]
                            # Refuse nonfinite outputs before they can poison the durable receipt.
                            # A returned forward is counted; unresolved journal work remains pending.
                            json.dumps(measured, allow_nan=False)
                            journal(log, {"operation": "native", "id": case["id"], "model": model,
                                          "arm": arm, "status": "completed", **measured})
                            row.update(status="completed", **measured)
                            result["native_pending_or_uncertain"] = False
                            atomic_json(receipt, result)
                            guard.rest_after_block(measured["seconds"])
                    started = time.perf_counter()
                    decider.close()
                    decider = None
                    result.setdefault("release_seconds", []).append(time.perf_counter() - started)
                result["status"] = "completed" if all(r["status"] == "completed" for r in records) else "completed_with_failures"
            finally:
                # Cleanup remains inside the OS lease, even after an uncertain request.
                if decider is not None:
                    decider.close()
                if ranker is not None:
                    ranker.close()
                result["owned_models_released"] = True
    except BaseException as exc:
        error = exc
        result.update(status="interrupted", error_type=type(exc).__name__)
    finally:
        try:
            with GpuLease(args.lease_path, "hotpot-pilot-release-verification"):
                pass
            result["gpu_lease_released"] = True
        except Exception as exc:
            error = error or exc
            result["release_error_type"] = type(exc).__name__
        try:
            after = {key: digest(path) for key, path in paths.items()}
            result.update(frozen_after=after, frozen_files_unchanged=hashes == after)
            if hashes != after:
                raise RuntimeError("Frozen source/data/model bytes changed")
            result["metrics"] = summarize(cohort, records)
        except Exception as exc:
            error = error or exc
            result["finalization_error_type"] = type(exc).__name__
        result.update(finished_at=utc_now(), wall_seconds=time.perf_counter() - wall_started,
            intentional_idle_seconds=guard.idle_seconds + admission.idle_seconds)
        if error:
            result["status"] = "interrupted"
        atomic_json(receipt, result)
    if error:
        raise error
    return result
