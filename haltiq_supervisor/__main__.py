"""Prepare or control explicit local phases; defaults never start model work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pc_strands.campaign import load, save, stamp
from .plan import inspect, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("inspect", help="Read-only readiness, no models or Codex")
    check.add_argument("--plan", type=Path, required=True)
    adopt = sub.add_parser("prepare", help="Freeze a ready explicit plan; no process launch")
    adopt.add_argument("--plan", type=Path, required=True)
    adopt.add_argument("--run-dir", type=Path, required=True)
    adopt.add_argument("--allow-model-calls", action="store_true")
    run = sub.add_parser("run", help="Execute from frozen cwd only; keep your PC powered on")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-model-calls", action="store_true")
    run.add_argument("--allow-codex-handoff", action="store_true")
    for action in ("status", "pause", "continue", "stop"):
        control = sub.add_parser(action, help="Local control only; no model/Codex requests")
        control.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "inspect":
        result = inspect(load(args.plan))
    elif args.action == "prepare":
        result = prepare(args.plan, args.run_dir, allow_model_calls=args.allow_model_calls)
    elif args.action == "run":
        from .engine import run
        result = run(args.run_dir, resume=args.resume, allow_model_calls=args.allow_model_calls,
                     allow_codex_handoff=args.allow_codex_handoff)
    else:
        receipt = args.run_dir / "supervisor.json"
        if not receipt.is_file():
            parser.error("Existing supervised run required; nothing was created")
        if args.action in ("pause", "stop"):
            save(args.run_dir / (args.action + ".request"), {"source": "explicit_local_command", "at": stamp()})
        elif args.action == "continue":
            (args.run_dir / "pause.request").unlink(missing_ok=True)
        result = load(receipt)
        result.update(pause_requested=(args.run_dir / "pause.request").exists(),
                      stop_requested=(args.run_dir / "stop.request").exists())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
