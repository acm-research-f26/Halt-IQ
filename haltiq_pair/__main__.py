from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Teammate-only HaltIQ pair simulations")
    parser.add_argument("command", choices=["hotpot", "tau", "replay", "serve-writer"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream-root", default="/content/tau2-bench")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "serve-writer":
        from .server import serve
        serve(args.output, port=args.port, device=args.device)
        return
    if args.command == "replay":
        from .replay import run_replay
        summary = run_replay(output=args.output)
        from .report import write_report
        write_report(args.output, replay=summary)
        summary = {k: v for k, v in summary.items() if k not in {"rows", "checkpoints"}}
    else:
        from .runtime import Runtime
        with Runtime(args.output / "runtime") as (agents, gate):
            if args.command == "hotpot":
                from .hotpot import run_hotpot
                summary = run_hotpot(agents, gate, args.output, limit=args.limit, seed=args.seed)
            else:
                from .tau import run
                summary = run(agents, gate, args.output, limit=args.limit, seed=args.seed,
                              upstream_root=args.upstream_root)
        from .report import write_report
        write_report(args.output, live=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
