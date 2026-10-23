"""Finite unattended qualification phases; never starts the14-hour study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from haltiq_hotpot.protocol import digest
from pc_strands.control import GpuLease, RunControl
from .data import records
from .journal import CallJournal, durable_write, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def qualification_protocol(config):
    """The worker checks the same bounded scope as the parent, before imports."""
    if config.get("purpose") != "qualification_pilot":
        raise ValueError("This worker is a small qualification pilot only; no long-study dispatch")
    freeze = read(config["data_freeze"])
    if digest(config["data_freeze"]) != config["data_freeze_sha256"]:
        raise ValueError("Qualification protocol bytes changed")
    p = freeze["pilot"]
    if (sum(p["training"].values()) > 60 or p["heldout"] > 8 or p["max_drafts"] > 3
            or p["arms"] != ["fixed1", "fixed3", "released", "trained"] or p["threshold"] != .8):
        raise ValueError("Qualification cannot expand into the long study")
    all_ids, all_questions = set(), set()
    cohorts = {}
    from pc_strands.data import question_key
    for name in ("pilot_fit", "pilot_tune", "pilot_calibration", "pilot_heldout"):
        spec = freeze["files"][name]
        if digest(spec["path"]) != spec["sha256"]:
            raise ValueError("Question cohort changed")
        cases = read(spec["path"])
        if len(cases) != spec["count"]:
            raise ValueError("Question count changed")
        for case in cases:
            public = case["public"]
            key = question_key(public["question"])
            if public["id"] in all_ids or key in all_questions:
                raise ValueError("Question-grouped partitions overlap")
            all_ids.add(public["id"])
            all_questions.add(key)
        cohorts[name] = cases
    if any(len(cohorts["pilot_" + name]) != count for name, count in p["training"].items()):
        raise ValueError("Frozen partition counts differ")
    if len(cohorts["pilot_heldout"]) != p["heldout"]:
        raise ValueError("Frozen heldout count differs")
    return freeze, cohorts


def bare_case(case):
    return {"public": case["public"], "evaluator": case["evaluator"]}


def verify_dispatch(args, plan, state):
    if (Path(__file__).resolve().parents[1] != args.previous.resolve() / "source"
            or state["run_id"] != os.environ.get("HALTIQ_RUN_ID")
            or state["plan_sha256"] != os.environ.get("HALTIQ_PLAN_SHA256")
            or os.environ.get("HALTIQ_PHASE") != args.phase
            or state.get("stages", {}).get(args.phase, {}).get("status") != "started"
            or args.config.resolve() != Path(plan["qualification_protocol"]["path"]).resolve()
            or args.output.resolve() != args.previous.resolve() / args.phase
            or args.control.resolve() != args.previous.resolve() / "workers" / args.phase
            or args.deadline != plan["deadline"]):
        raise ValueError("Worker arguments must match its adopted current phase exactly")
    from haltiq_supervisor.engine import verify_completion
    for stage in plan["stages"]:
        if stage["id"] == args.phase:
            break
        receipt = state.get("stages", {}).get(stage["id"], {})
        if receipt.get("status") != "completed":
            raise ValueError("Every preceding phase must be terminal and verified")
        for path, expected in receipt.get("output_sha256", {}).items():
            if digest(path) != expected:
                raise ValueError("Preceding generated evidence changed")
        verify_completion(stage, state, read(args.previous / stage["id"] / "completion.json"))


def verify_native_rows(args, config, cohorts, rows, baseline):
    """Join actual native receipts/public inputs/labels before any CPU update."""
    from .hobson import stable_hash
    from .scoring import score
    journal = CallJournal(args.previous / "features/physical")
    if journal.verify()["pending_or_uncertain"]:
        raise ValueError("Native feature requests are unresolved")
    responses = {}
    for event in journal.records:
        if event["event"] == "completed" and event["role"] == "native_answer":
            response = read(journal.directory / event["response"])
            responses[fingerprint(response)] = response
    drafts = initial_drafts(args.previous / "collect")
    cases = {c["public"]["id"]: c for cases in cohorts.values() for c in cases}
    for row in rows:
        if row["status"] != "completed":
            continue
        case, observed = cases[row["question_id"]], row["observed"]
        draft = drafts[row["question_id"]]["episode"]["submitted"]
        evidence = "\n".join(case["public"]["context"][i][0] + ": " + " ".join(case["public"]["context"][i][1])
                             for i in observed["retrieval_indices"])
        public_state = f"Question: {case['public']['question']}\nEvidence: {evidence}\nDraft answer: {draft['answer']}"
        feature = observed["feature"]
        if (fingerprint(observed) not in responses or observed["physical_forwards"] != 1
                or feature["provenance"]["state_sha256"] != stable_hash(public_state)
                or row["label"] != int(score(draft, case["evaluator"])["em"])
                or digest(feature["path"]) != feature["sha256"]):
            raise ValueError("Feature/physical-forward/public-state/evaluator join failed")
    heldout = [r for r in rows if r["partition"] == "engineering_heldout"]
    if baseline["rows_sha256"] != fingerprint(heldout):
        raise ValueError("Untouched baseline is not bound to its genuine native rows")


def completion(args, *, planned, completed, failed, cleanup, pending=False, extra=None):
    value = {"run_id": os.environ["HALTIQ_RUN_ID"], "plan_sha256": os.environ["HALTIQ_PLAN_SHA256"],
        "phase": os.environ["HALTIQ_PHASE"], "terminal": True,
        "pending_or_uncertain": pending, "cleanup_verified": cleanup,
        "planned_units_accounted": True, "planned_units": planned,
        "completed_units": completed, "failed_units": failed,
        "not_run_units": planned - completed - failed,
        "finished_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
    durable_write(args.output / "completion.json", value)
    return value


def collect(args, config, freeze, cohorts, control):
    from .agent import run_episode
    from .runtime import LocalRuntime
    cases = cohorts["pilot_heldout"] + cohorts["pilot_fit"] + cohorts["pilot_tune"] + cohorts["pilot_calibration"]
    rows = []
    runtime = LocalRuntime(config, args.output, control)
    try:
        with runtime:
            for case in cases:
                public = case["public"]
                ranked = runtime.rank(public)
                episode = run_episode(bare_case(case), runtime.generate, arm="fixed1", max_drafts=1,
                                      passage_indices=ranked["indices"])
                row = {"question_id": public["id"], "partition": case["partition"],
                       "initial_retrieval": ranked, "episode": episode}
                durable_write(args.output / "episodes" / (public["id"] + ".json"), row)
                rows.append(row)
    finally:
        durable_write(args.output / "runtime-terminal.json", {"cleanup": runtime.cleanup,
            "calls": runtime.journal.verify(), "timings": runtime.timings,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
    durable_write(args.output / "episodes.json", rows)
    successful = sum(bool(r["episode"].get("submitted")) for r in rows)
    return completion(args, planned=len(cases), completed=successful, failed=len(rows) - successful,
                      cleanup=runtime.cleanup["verified"], pending=runtime.journal.summary()["pending_or_uncertain"])


def initial_drafts(path):
    rows = read(Path(path) / "episodes.json")
    return {row["question_id"]: row for row in rows}


def features(args, config, freeze, cohorts, control):
    from .runtime import LocalRuntime
    from .scoring import score
    drafts = initial_drafts(args.previous / "collect")
    rows, baseline = [], []
    runtime = LocalRuntime(config, args.output, control)
    cases = cohorts["pilot_heldout"] + cohorts["pilot_fit"] + cohorts["pilot_tune"] + cohorts["pilot_calibration"]
    planned = len(cohorts["pilot_heldout"]) + sum(freeze["pilot"]["training"].values())
    try:
        with runtime:
            for case in cases:
                public = case["public"]
                saved = drafts[public["id"]]["episode"]
                draft = saved.get("submitted")
                criteria = ["answer"]
                for criterion in criteria:
                    row = {"question_id": public["id"], "partition": case["partition"], "criterion": criterion}
                    if not draft:
                        row.update(status="failed", failure="no_initial_submission", label=0)
                    else:
                        metrics = score(draft, case["evaluator"])
                        row["label"] = int(metrics["em"] if criterion == "answer" else metrics["sp_em"])
                        try:
                            observed = runtime.observe(public, draft, criterion=criterion)
                            row.update(status="completed", observed=observed)
                        except ValueError as error:
                            row.update(status="failed", failure=type(error).__name__, message=str(error))
                    durable_write(args.output / "rows" / f"{public['id']}-{criterion}.json", row)
                    rows.append(row)
                    if case["partition"] == "engineering_heldout":
                        baseline.append(row)
    finally:
        durable_write(args.output / "runtime-terminal.json", {"cleanup": runtime.cleanup,
            "calls": runtime.journal.verify(), "timings": runtime.timings,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
    durable_write(args.output / "features.json", rows)
    proof = {"phase": "untouched_pretraining", "terminal": True,
        "pending_or_uncertain": runtime.journal.summary()["pending_or_uncertain"],
        "cleanup_verified": runtime.cleanup["verified"],
        "checkpoint_head_sha256": digest(Path(config["checkpoint"]) / "head.safetensors"),
        "valid_native_forwards": sum(r["status"] == "completed" for r in baseline),
        "planned_question_ids": [c["public"]["id"] for c in cohorts["pilot_heldout"]],
        "completed": sum(r["status"] == "completed" for r in baseline),
        "failed": sum(r["status"] != "completed" for r in baseline),
        "rows_sha256": fingerprint(baseline), "finished_at": datetime.now(timezone.utc).isoformat()}
    durable_write(args.output / "untouched-baseline.json", proof)
    return completion(args, planned=planned, completed=sum(r["status"] == "completed" for r in rows),
        failed=sum(r["status"] != "completed" for r in rows), cleanup=runtime.cleanup["verified"],
        pending=runtime.journal.summary()["pending_or_uncertain"])


def hydrate(row):
    from safetensors.torch import load_file
    feature = row["observed"]["feature"]
    if digest(feature["path"]) != feature["sha256"]:
        raise ValueError("Native feature cache changed")
    return {"question_id": row["question_id"], "partition": row["partition"],
            "criterion": row["criterion"], "label": row["label"],
            "feature": {**feature, **load_file(feature["path"], device="cpu")}}


def train(args, config, freeze, cohorts, control):
    rows = read(args.previous / "features/features.json")
    baseline = read(args.previous / "features/untouched-baseline.json")
    verify_native_rows(args, config, cohorts, rows, baseline)
    expected = {"fit": 40, "tune": 10, "calibration": 10}
    for criterion in ("answer",):
        for partition, minimum in expected.items():
            selected = [r for r in rows if r["partition"] == partition and r["criterion"] == criterion]
            if len(selected) != minimum or any(r["status"] != "completed" for r in selected):
                raise ValueError("Training qualification minima not met; preserve failures, never replace cases")
    with GpuLease(Path(config["gpu_lease"]), "CPU-only existing Hobson head updates"):
        # A lease also covers Torch import; no native/writer model is resident.
        from .learning import train_heads
        material = [hydrate(r) for r in rows if r["partition"] in expected]
        journal = CallJournal(args.output / "physical")
        result = journal.call("head_training_and_calibration", {
            "features_sha256": digest(args.previous / "features/features.json"),
            "baseline_sha256": digest(args.previous / "features/untouched-baseline.json")}, lambda: train_heads(config["checkpoint"], config["base"],
            [r for r in material if r["partition"] == "fit"],
            [r for r in material if r["partition"] == "tune"],
            [r for r in material if r["partition"] == "calibration"],
            args.output / "learned", baseline, control=control))
    durable_write(args.output / "learning.json", result)
    return completion(args, planned=1, completed=1, failed=0, cleanup=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "features", "train", "compare"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_calls:
        raise ValueError("Explicit model-call opt-in required")
    config = read(args.config)
    # A model worker is never a loose script: require the exact adopted frozen
    # supervisor instance even when somebody invokes this module directly.
    from haltiq_supervisor.plan import qualification, verify_frozen
    plan, state = read(args.previous / "plan.json"), read(args.previous / "supervisor.json")
    verify_dispatch(args, plan, state)
    verify_frozen(plan, state, args.previous.resolve())
    qualification(plan)
    import importlib.util
    for module, expected in config["runtime_modules"].items():
        found = importlib.util.find_spec(module)
        if found is None or Path(found.origin).resolve() != Path(expected).resolve():
            raise ValueError("Runtime imported a different pinned package: " + module)
    freeze, cohorts = qualification_protocol(config)
    args.output.mkdir(parents=True, exist_ok=False)
    control = RunControl(args.control, args.deadline)
    started = time.monotonic()
    functions = {"collect": collect, "features": features, "train": train}
    if args.phase == "compare":
        from .comparison import compare
        functions["compare"] = compare
    try:
        value = functions[args.phase](args, config, freeze, cohorts, control)
        durable_write(args.output / "phase-timing.json", {"wall_seconds": time.monotonic() - started,
            "active_seconds": control.active_seconds, "intentional_idle_seconds": control.idle_seconds})
        print(json.dumps({"phase": args.phase, "terminal": value["terminal"],
                          "completed": value["completed_units"], "failed": value["failed_units"]}))
    except BaseException as error:
        durable_write(args.output / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "phase": args.phase, "no_automatic_retry": True, "wall_seconds": time.monotonic() - started,
            "actual_partial_episode": getattr(error, "haltiq_episode_partial", None)})
        raise


if __name__ == "__main__":
    main()
