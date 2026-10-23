"""Bounded synthetic runtime qualification, separate from benchmark outcomes.

Six actual native forwards cover both criteria at three declared input lengths.
Each genuine feature is rescored with released and trained heads. Two generation
responses are reused only for exact repeated requests. No learning or fresh
research question is permitted; a completed real pilot is required first.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time

from pc_strands.control import RunControl
from .hobson import file_hash
from .journal import durable_write
from .protocol import pinned, read
from .runtime import LocalRuntime, decision_state
from .worker import completion

TARGETS = (512, 2048, 4096)
CRITERIA = ("answer", "evidence")
UNITS = 22  # 6 native + 12 CPU rescoring + 2 generation + 2 exact-response reuse.


def synthetic_input(backend, target, criterion):
    """Fit only invented filler; never truncate any benchmark question/passages."""
    draft = {"answer": "control", "supporting_facts": [["Synthetic runtime note", 0]]}
    public = {"id": f"synthetic-{criterion}-{target}", "type": "synthetic_control",
        "level": "engineering_only", "question": "What synthetic key is stated in the note?",
        "context": [["Synthetic runtime note", [""]]]}
    low, high, best = 0, target * 2, None
    while low <= high:
        repeat = (low + high) // 2
        public["context"][0][1][0] = "The synthetic key is control." + " filler" * repeat
        state = decision_state(public, draft, [0], criterion)
        try:
            prepared = backend.prepare_state(state, criterion=criterion)
            n = prepared["input_tokens"]
        except ValueError:
            high = repeat - 1
            continue
        if n <= target:
            best = repeat, n
            low = repeat + 1
        else:
            high = repeat - 1
    if best is None or not target - 8 <= best[1] <= target:
        raise ValueError("The fixed synthetic input did not cover its declared token boundary")
    public["context"][0][1][0] = "The synthetic key is control." + " filler" * best[0]
    return public, draft, best[1]


def verify_dispatch(args, plan, state):
    from haltiq_supervisor.plan import verify_frozen
    root = args.previous.resolve()
    if (Path(__file__).resolve().parents[1] != root / "source" or plan.get("purpose") != "runtime_probe"
            or [s["id"] for s in plan["stages"]] != ["probe"]
            or state["run_id"] != os.environ.get("HALTIQ_RUN_ID")
            or state["plan_sha256"] != os.environ.get("HALTIQ_PLAN_SHA256")
            or os.environ.get("HALTIQ_PHASE") != "probe"
            or state.get("stages", {}).get("probe", {}).get("status") != "started"
            or args.config.resolve() != Path(plan["runtime_probe"]["path"]).resolve()
            or args.output.resolve() != root / "probe" or args.control.resolve() != root / "workers/probe"
            or args.deadline != plan["deadline"]):
        raise ValueError("Synthetic runtime probe must use its separately adopted frozen invocation")
    verify_frozen(plan, state, root)


def pilot_binding(config):
    from .qualification import terminal_owner
    proof = pinned(config["combined_pilot_proof"], config["combined_pilot_proof_sha256"])
    if proof.get("schema") == "haltiq-tau-hotpot-qualified-pilot-v2":
        from .recovery import verify_qualified_pilot
        qualified, trained, _ = verify_qualified_pilot(proof, config=config)
        return qualified, trained
    if (proof.get("schema") != "haltiq-tau-hotpot-qualified-pilot-v1"
            or proof.get("status") != "PASS_QUALIFIED_ENGINEERING_PILOT"
            or proof.get("qualified") is not True or proof.get("pending_or_uncertain") is not False
            or proof.get("cleanup_verified") is not True):
        raise ValueError("A genuine completed benchmark pilot must precede the synthetic probe")
    run = Path(proof["run"])
    _, state = terminal_owner(run, proof["plan_sha256"])
    head = run / "train/learned/answer-head.safetensors"
    model_files = {"released_head_sha256": Path(config["checkpoint"]) / "head.safetensors",
        "adapter_sha256": Path(config["checkpoint"]) / "lora/adapter_model.safetensors",
        "base_weights_sha256": Path(config["base"]) / "model.safetensors-00001-of-00001.safetensors"}
    if (proof["supervisor_sha256"] != file_hash(run / "supervisor.json") or state["run_id"] != proof["run_id"]
            or proof["trained_head_sha256"] != file_hash(head)
            or any(proof["model_binding"][key] != file_hash(path) for key, path in model_files.items())):
        raise ValueError("Original pilot owner or released/trained head changed")
    return proof, head


def probe(args, config, control):
    proof, trained = pilot_binding(config)
    points, generation = [], []
    runtime = LocalRuntime(config, args.output, control)
    try:
        with runtime:
            # Materialize exact generation prefixes before native placement.
            state = {"question": "What is the synthetic key?", "evidence": [{
                "title": "Synthetic runtime note", "sentences": [{"index": 0, "text": "The synthetic key is control."}]}],
                "previous_draft": None}
            answer = runtime.generate(state)
            repeated = runtime.generate(state)
            review_state = {**state, "draft": answer}
            feedback = runtime.generate(review_state, review=True)
            repeated_feedback = runtime.generate(review_state, review=True)
            if answer != repeated or feedback != repeated_feedback:
                raise ValueError("An exact repeated generation request changed its committed response")
            generation = [answer, feedback]
            runtime._native_ready()
            # One declared public passage: no retrieval model is needed to choose
            # among nonexistent alternatives. This is not a retrieval benchmark.
            runtime.rank = lambda *_: {"indices": [0], "source": "scripted_single_synthetic_passage"}
            for target in TARGETS:
                for criterion in CRITERIA:
                    public, draft, tokens = synthetic_input(runtime.backend, target, criterion)
                    actual = runtime.observe(public, draft, criterion=criterion, temperature=1.)
                    same = runtime.observe(public, draft, criterion=criterion, temperature=1.)
                    learned = runtime.observe(public, draft, criterion=criterion, head_path=trained, temperature=1.)
                    if (actual["physical_forwards"] != 1 or actual["input_tokens"] != tokens
                            or any(v["physical_forwards"] != 0 or v["head_scores"] != 1 for v in (same, learned))
                            or same["raw_logits_false_true"] != actual["raw_logits_false_true"]):
                        raise ValueError("Genuine native-to-cached CPU head replay did not reproduce")
                    row = {"criterion": criterion, "target_tokens": target, "input_tokens": tokens,
                        "public": public, "draft": draft, "actual": actual,
                        "cached_released": same, "cached_trained": learned,
                        "cold_first_forward_after_placement": not points}
                    durable_write(args.output / "points" / f"{criterion}-{target}.json", row)
                    points.append(row)
    finally:
        durable_write(args.output / "runtime-terminal.json", {"cleanup": runtime.cleanup,
            "calls": runtime.journal.verify(), "timings": runtime.timings,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
    calls = runtime.journal.verify()
    expected = {"native_startup": 1, "native_answer": 3, "native_evidence": 3,
                "cached_head_answer": 6, "cached_head_evidence": 6,
                "writer_load": 1, "writer": 1, "reviewer": 1, "writer_unload": 1}
    if calls["by_role"] != expected or calls["exact_response_reuses"] != 2 or calls["pending_or_uncertain"]:
        raise ValueError("Synthetic probe work differed from its fixed bounded physical route")
    value = {"schema": "haltiq-runtime-probe-v1", "status": "PASS_RUNTIME_PROBE",
        "engineering_only": True, "benchmark_quality_claim": False, "new_learning": False,
        "score_scope": "T=1 technical replay only; the trained weights learned answer correctness, not evidence correctness",
        "points": points, "generation_results": generation, "calls": calls,
        "pilot_proof_sha256": config["combined_pilot_proof_sha256"], "pilot_run_id": proof["run_id"],
        "released_head_sha256": proof["model_binding"]["released_head_sha256"],
        "trained_head_sha256": file_hash(trained), "wrapper_plan_sha256": os.environ["HALTIQ_PLAN_SHA256"],
        "source_sha256": read(args.previous / "supervisor.json")["source_sha256"],
        "finished_at": datetime.now(timezone.utc).isoformat(), "long_study_started": False}
    durable_write(args.output / "probe.json", value)
    return completion(args, planned=UNITS, completed=UNITS, failed=0, cleanup=runtime.cleanup["verified"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["probe"])
    for name in ("config", "output", "previous", "control"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_calls:
        raise ValueError("Explicit bounded synthetic model-call opt-in required")
    plan, state = read(args.previous / "plan.json"), read(args.previous / "supervisor.json")
    verify_dispatch(args, plan, state)
    config = pinned(args.config, plan["runtime_probe"]["sha256"])
    import importlib.util
    for module, expected in config["runtime_modules"].items():
        found = importlib.util.find_spec(module)
        if found is None or Path(found.origin).resolve() != Path(expected).resolve():
            raise ValueError("Synthetic probe imported a different pinned package: " + module)
    args.output.mkdir(parents=True, exist_ok=False)
    control, started = RunControl(args.control, args.deadline), time.monotonic()
    try:
        probe(args, config, control)
        durable_write(args.output / "phase-timing.json", {"wall_seconds": time.monotonic() - started,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
    except BaseException as error:
        durable_write(args.output / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "no_automatic_retry": True})
        raise


if __name__ == "__main__":
    main()
