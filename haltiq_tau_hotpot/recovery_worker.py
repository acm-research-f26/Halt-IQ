"""Exactly one original engineering question per separately owned process."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import time

from pc_strands.control import RunControl
from . import qualification as q
from .journal import durable_write, fingerprint
from .recovery import ARMS, PHASES, checked, preparation_binding, validate_plan
from .runtime import LocalRuntime


def verify_dispatch(args, plan, state):
    from haltiq_supervisor.plan import verify_frozen
    from haltiq_supervisor.engine import verify_completion
    root = args.previous.resolve()
    q.require(Path(__file__).resolve().parents[1] == root / "source"
        and plan.get("purpose") == "qualification_recovery" and args.phase in PHASES
        and state["run_id"] == os.environ.get("HALTIQ_RUN_ID")
        and state["plan_sha256"] == os.environ.get("HALTIQ_PLAN_SHA256")
        and os.environ.get("HALTIQ_PHASE") == args.phase
        and state.get("stages", {}).get(args.phase, {}).get("status") == "started"
        and args.config.resolve() == Path(plan["recovery_protocol"]["path"]).resolve()
        and args.output.resolve() == root / args.phase
        and args.control.resolve() == root / "workers" / args.phase and args.deadline == plan["deadline"],
        "Recovery worker arguments/source must match its actual frozen current owner")
    verify_frozen(plan, state, root)
    validate_plan(plan)
    for phase in PHASES[:PHASES.index(args.phase)]:
        owner = state["stages"][phase]
        q.require(owner.get("status") == "completed", "Previous question owner must be completed")
        for path, expected in owner["output_sha256"].items():
            q.require(q.sha(path) == expected, "Previous question output changed")
        stage = next(s for s in plan["stages"] if s["id"] == phase)
        verify_completion(stage, state, q.read(root / phase / "compare/completion.json"))


def assert_envelope(runtime):
    """Cache/prompt/order changes cannot silently restore repeated GPU swaps."""
    roles = [event["role"] for event in runtime.journal.records if event["event"] == "pending"]
    if roles.count("writer_load") > 1 or roles.count("native_startup") > 1:
        raise RuntimeError("One question exceeded its reviewed placement envelope")
    if "native_startup" in roles:
        if "writer_load" in roles[roles.index("native_startup") + 1:]:
            raise RuntimeError("A native-to-writer reload invalidated the bounded host-memory route")


class QuestionRuntime(LocalRuntime):
    """Reject extra placement BEFORE dispatch, outside policy ValueError handling."""
    def _writer_ready(self):
        roles = self.journal.summary()["by_role"]
        if roles.get("native_startup", 0) or roles.get("writer_load", 0) and not self.writer_loaded:
            raise RuntimeError("Question route would reload writer after its reviewed placement")
        if roles.get("writer_load", 0):
            resident = self._http("/api/ps", timeout=5).get("models")
            if not isinstance(resident, list) or len(resident) != 1 or resident[0].get("digest") != self.config["writer_digest"]:
                raise RuntimeError("The single writer placement was evicted; do not reload automatically")
            return
        super()._writer_ready()

    def _native_ready(self):
        if self.backend is None and self.journal.summary()["by_role"].get("native_startup", 0):
            raise RuntimeError("Question route would reload the native model")
        super()._native_ready()


def run_question(args, config, control):
    from .agent import run_episode
    from .hobson import load_head, score_head
    from .worker import bare_case, completion, hydrate, initial_drafts
    proof, _, _, cohorts = preparation_binding(checked(config["preparation_proof"])["path"])
    case = cohorts["pilot_heldout"][PHASES.index(args.phase)]
    public, qid = case["public"], case["public"]["id"]
    q.require(qid == config["question_ids"][PHASES.index(args.phase)], "Recovery question/order changed")
    source = Path(proof["source_run"])
    saved = initial_drafts(source / "collect")[qid]
    draft = saved["episode"]["submitted"]
    feature = next(row for row in q.read(source / "features/features.json") if row["question_id"] == qid)
    learned = q.read(checked(proof["learning"])["path"])
    calibration = learned["criteria"]["answer"]
    trained = Path(checked(proof["trained_head"])["path"])
    output = args.output / "compare"
    output.mkdir(exist_ok=False)
    runtime, rows = QuestionRuntime(config, output, control), []
    try:
        with runtime:
            released_head, trained_head = load_head(config["checkpoint"]), load_head(config["checkpoint"], trained)
            for arm in ARMS:
                def writer(state, feedback):
                    if state["previous_draft"] is None:
                        return deepcopy(draft)
                    result = runtime.generate(state, feedback)
                    assert_envelope(runtime)
                    return result
                def reviewer(state, proposed):
                    ranked = runtime.rank(public, proposed["answer"])
                    evidence = [{"title": public["context"][i][0], "sentences": [
                        {"index": n, "text": text} for n, text in enumerate(public["context"][i][1])]}
                        for i in ranked["indices"]]
                    result = runtime.generate({"question": public["question"], "evidence": evidence,
                                               "draft": proposed}, review=True)
                    assert_envelope(runtime)
                    return result
                def gate(state, proposed, criterion):
                    q.require(criterion == "answer", "Recovery cannot add an evidence-head training claim")
                    temperature = calibration[("trained" if arm == "trained" else "released") + "_calibration"]["temperature"]
                    if state["previous_draft"] is None and proposed == draft:
                        material = hydrate(feature)["feature"]
                        result = runtime._physical("cpu_head", {"feature": material["provenance_sha256"],
                            "head": arm, "temperature": temperature}, lambda: score_head(
                                trained_head if arm == "trained" else released_head, material, temperature=temperature))
                    else:
                        result = runtime.observe(public, proposed, criterion="answer",
                            head_path=trained if arm == "trained" else None, temperature=temperature)
                    assert_envelope(runtime)
                    return result
                row = run_episode(bare_case(case), writer, reviewer, gate, arm=arm, max_drafts=3,
                    threshold=.8, passage_indices=saved["initial_retrieval"]["indices"], seed=42, timeout_seconds=1800)
                row.update(shared_initial_draft_sha256=fingerprint(draft), shared_initial_physical_calls=0,
                    preparation_proof_sha256=config["preparation_proof"]["sha256"], recovery_question_phase=args.phase)
                durable_write(output / "episodes" / (qid + "-" + arm + ".json"), row)
                rows.append(row)
            assert_envelope(runtime)
    finally:
        durable_write(output / "runtime-terminal.json", {"cleanup": runtime.cleanup,
            "calls": runtime.journal.verify(), "timings": runtime.timings,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
    durable_write(output / "episodes.json", rows)
    original_output, args.output = args.output, output
    try:
        completed = sum(row["status"] == "completed" for row in rows)
        return completion(args, planned=4, completed=completed, failed=4 - completed,
            cleanup=runtime.cleanup["verified"], pending=runtime.journal.summary()["pending_or_uncertain"],
            extra={"question_id": qid, "new_learning": False, "source_preparation_sha256": config["preparation_proof"]["sha256"]})
    finally:
        args.output = original_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    for name in ("config", "output", "previous", "control"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    q.require(args.allow_model_calls, "Explicit model-call opt-in required for a new recovery owner")
    plan, state = q.read(args.previous / "plan.json"), q.read(args.previous / "supervisor.json")
    verify_dispatch(args, plan, state)
    config = q.read(args.config)
    q.require(q.sha(args.config) == plan["recovery_protocol"]["sha256"], "Recovery config changed")
    for module, expected in config["runtime_modules"].items():
        found = importlib.util.find_spec(module)
        q.require(found is not None and Path(found.origin).resolve() == Path(expected).resolve(), "Recovery runtime package origin differs")
    args.output.mkdir(parents=True, exist_ok=False)
    control, started = RunControl(args.control, args.deadline), time.monotonic()
    try:
        run_question(args, config, control)
        durable_write(args.output / "phase-timing.json", {"wall_seconds": time.monotonic() - started,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
    except BaseException as error:
        durable_write(args.output / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "phase": args.phase, "no_automatic_retry": True, "wall_seconds": time.monotonic() - started,
            "actual_partial_episode": getattr(error, "haltiq_episode_partial", None)})
        raise


if __name__ == "__main__":
    main()
