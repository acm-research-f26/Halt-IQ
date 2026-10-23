"""Offline-first CLI; model runs require explicit provider construction by an operator."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from . import UPSTREAM_REVISION, UPSTREAM_VERSION
from .core import CONTEXT_PROTOCOL
from .pinning import verify_upstream


def validate_pilot(value):
    expected = {"upstream_revision", "benchmark_version", "context_protocol", "language", "domain", "task_ids", "seeds", "arms",
                "max_steps", "max_reviews", "threshold", "writer_model", "reviewer_model", "user_model",
                "evaluation_type", "communicate_scorer"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("pilot keys differ from the explicit public example schema")
    if value["upstream_revision"] != UPSTREAM_REVISION or value["benchmark_version"] != UPSTREAM_VERSION:
        raise ValueError("benchmark pin/version mismatch")
    if value["context_protocol"] != CONTEXT_PROTOCOL:
        raise ValueError("decision-model context protocol mismatch")
    if value["language"] != "en" or value["domain"] not in {"retail", "telecom", "airline"}:
        raise ValueError("this pilot is English text-only in a standard tool domain")
    if type(value["task_ids"]) is not list or not value["task_ids"] or len(set(value["task_ids"])) != len(value["task_ids"]) or not all(type(x) is str and x for x in value["task_ids"]):
        raise ValueError("explicit unique task IDs required")
    if type(value["seeds"]) is not list or not value["seeds"] or len(set(value["seeds"])) != len(value["seeds"]) or not all(type(x) is int and x >= 0 for x in value["seeds"]):
        raise ValueError("explicit unique nonnegative seeds required")
    if (type(value["arms"]) is not list or len(set(value["arms"])) != len(value["arms"]) or
            not {"no_gate", "strands"} <= set(value["arms"]) <= {"no_gate", "fixed_review", "strands", "jev"}):
        raise ValueError("declare unique arms including no_gate and strands; Jev is optional before dispatch")
    if type(value["max_steps"]) is not int or not 1 <= value["max_steps"] <= 200:
        raise ValueError("max_steps must be from one through 200")
    if type(value["max_reviews"]) is not int or not 1 <= value["max_reviews"] <= 10:
        raise ValueError("max_reviews must be from one through ten")
    if type(value["threshold"]) not in (int, float) or not 0 < value["threshold"] < 1:
        raise ValueError("threshold must be between zero and one")
    if not all(type(value[x]) is str and value[x].strip() for x in ("writer_model", "reviewer_model", "user_model")):
        raise ValueError("all model identities must be explicit")
    if value["evaluation_type"] != "all" or value["communicate_scorer"] != "semantic":
        raise ValueError("the research pilot retains official reward_basis and semantic communication scoring")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    check = subs.add_parser("validate", help="Validate pin/config only; no model/API work")
    check.add_argument("--config", required=True)
    check.add_argument("--upstream-root", required=True)
    mock = subs.add_parser("mock-check", help="Execute scripted official tools/evaluator; no model/API calls")
    mock.add_argument("--upstream-root", required=True)
    mock.add_argument("--output", required=True)
    run = subs.add_parser("run", help="Explicit live pilot; never invoked by validation/tests")
    run.add_argument("--config", required=True)
    run.add_argument("--upstream-root", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--deadline", required=True, help="ISO8601 including timezone")
    run.add_argument("--allow-model-calls", action="store_true")
    run.add_argument("--allow-hosted-calls", action="store_true")
    run.add_argument("--strands-command", required=True, help="JSON argv for a separately governed local Strands gate process")
    run.add_argument("--strands-preflight-command", required=True, help="JSON argv for tokenizer-only native_command preflight")
    run.add_argument("--jev-ledger", help="EXISTING cumulative Jev SQLite file, required only for a declared Jev arm")
    run.add_argument("--generation-args", default="{}", help="Nonsecret LiteLLM settings JSON; credentials only in environment")
    run.add_argument("--control-root", required=True, help="Local resource/deadline control directory for all generation calls")
    args = parser.parse_args()
    if args.command == "validate":
        config = validate_pilot(json.loads(Path(args.config).read_text(encoding="utf-8")))
        pin = verify_upstream(args.upstream_root)
        print(json.dumps({**pin, "status": "valid_not_executed", "planned_task_seed_pairs_per_arm":
                          len(config["task_ids"]) * len(config["seeds"]), "model_calls": 0}, indent=2))
    elif args.command == "mock-check":
        output = Path(args.output)
        if output.exists():
            parser.error("output exists; preserve earlier receipts")
        from .mock_check import run_mock_check
        result = run_mock_check(args.upstream_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
    else:
        if not args.allow_model_calls:
            parser.error("run needs --allow-model-calls; no calls were made")
        config = validate_pilot(json.loads(Path(args.config).read_text(encoding="utf-8")))
        if any("REPLACE_" in config[key] for key in ("writer_model", "reviewer_model", "user_model")):
            parser.error("replace example model identities before live execution")
        deadline = datetime.fromisoformat(args.deadline)
        if deadline.tzinfo is None:
            parser.error("deadline needs an explicit timezone offset")
        from .providers import CommandGate, JevActionGate
        from .pilot import run_pilot
        command = json.loads(args.strands_command)
        preflight_command = json.loads(args.strands_preflight_command)
        generation_args = json.loads(args.generation_args)
        local_generation = any(config[key].startswith("ollama_chat/") for key in
                               ("writer_model", "reviewer_model", "user_model"))
        if local_generation and generation_args.get("num_gpu") != 0:
            parser.error("CLI local generation requires explicit num_gpu=0; GPU generation needs the separately owned same-process coordinator")
        factories = {"strands": lambda: CommandGate(command, allow_model_calls=True, timeout=600, deadline=deadline,
                                                     preflight_argv=preflight_command)}
        if "jev" in config["arms"]:
            if not args.allow_hosted_calls or not args.jev_ledger:
                parser.error("a declared Jev arm needs --allow-hosted-calls and --jev-ledger before dispatch")
            ledger_path = Path(args.jev_ledger).resolve()
            if not ledger_path.is_file():
                parser.error("existing cumulative Jev ledger required; no ledger is created")
            from pc_strands.jev_replay import read_existing_ledger
            audit = read_existing_ledger(ledger_path)
            if audit["unsettled"] or audit["used_usd"] >= 4.50:
                parser.error("existing Jev budget has unsettled requests or reached its soft cap")
            from haltiq.budget import BudgetLedger
            from haltiq.providers import JevClient
            ledger = BudgetLedger(ledger_path, limit_usd=4.50)
            client = JevClient(budget=ledger)
            factories["jev"] = lambda: JevActionGate(client, allow_hosted_calls=True, deadline=deadline)
        from pc_strands.control import RunControl
        control = RunControl(Path(args.control_root), args.deadline)
        result = run_pilot(config, args.upstream_root, args.output, factories,
                           allow_model_calls=True, generation_args=generation_args, deadline=deadline,
                           before_call=lambda: control.before_work(require_gpu=False),
                           after_call=control.rest_after_block)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
