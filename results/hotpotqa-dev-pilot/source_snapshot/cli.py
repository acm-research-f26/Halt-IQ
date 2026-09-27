"""Run from the checkout with python -m haltiq. No Python dependencies needed."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

from .budget import BudgetLedger
from .critics import JevCritic, LLMCritic, LocalTypedCritic, Writer
from .providers import JevClient, OllamaClient
from .runner import run_experiment
from .tasks import load_tasks


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "diagnostic.jsonl"
HOTPOT_DATA = ROOT / "data" / "hotpotqa" / "tasks.jsonl"


def add_provider_flags(parser):
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--writer-model", default="qwen3:8b")
    parser.add_argument("--critic-model", default="qwen3:8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--context-tokens", type=int, default=32768,
                        help="Ollama context window; default 32768 for complete HotpotQA evidence")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--openjev-url", default="http://localhost:8080")
    parser.add_argument("--openjev-model", default="openjev-latest")
    parser.add_argument("--budget-usd", type=float, default=5.0,
                        help="Cumulative allowance on this persistent ledger, max $30 (default $5)")
    parser.add_argument("--budget-ledger", type=Path, default=ROOT / ".haltiq" / "budget.sqlite")


def make_parser():
    parser = argparse.ArgumentParser(description="HaltIQ: local writer + swappable critic, with optional hosted Jev")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("demo", "Offline scripted control-flow demonstration; no model results"),
                            ("run", "Run a real local writer with selected critics")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dataset", type=Path, help="Override the selected environment's JSONL file")
        p.add_argument("--environment", choices=["hotpotqa", "diagnostic"],
                       default="diagnostic" if name == "demo" else "hotpotqa")
        p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
        p.add_argument("--limit", type=int)
        p.add_argument("--max-rounds", type=int, default=3)
        p.add_argument("--output", type=Path, help="New directory; existing directories are never overwritten")
        if name == "run":
            p.add_argument("--critics", default="fixed,llm", help="Comma-separated: fixed,llm,jev,local-typed,openjev")
        add_provider_flags(p)
    judge = sub.add_parser("judge", help="Review one JSON state for integration with an existing harness")
    judge.add_argument("--input", required=True, type=Path, help="JSON {question, evidence:[...], draft}")
    judge.add_argument("--backend", choices=["llm", "jev", "local-typed", "openjev"], default="llm")
    add_provider_flags(judge)
    doctor = sub.add_parser("doctor", help="Show local model availability and whether a Jev key is configured")
    doctor.add_argument("--ollama-url", default="http://localhost:11434")
    prepare = sub.add_parser("prepare-hotpot", help="Download and prepare reproducible HotpotQA distractor subsets")
    prepare.add_argument("--source", type=Path, help="Optional original-format HotpotQA dev JSON; otherwise download the pinned mirror")
    prepare.add_argument("--output", type=Path, default=HOTPOT_DATA.parent)
    prepare.add_argument("--dev-size", type=int, default=20)
    prepare.add_argument("--test-size", type=int, default=80)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--cache-dir", type=Path, default=ROOT / ".haltiq" / "datasets")
    return parser


def local_metadata(base_url):
    # Constructor validates local URL. Do not use a configured remote proxy.
    OllamaClient(base_url=base_url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.rstrip("/") + "/api/tags", timeout=5) as response:
        tags = json.load(response)
    return [{k: m.get(k) for k in ("name", "digest", "size", "details", "remote_host") if k in m}
            for m in tags.get("models", [])]


def make_critics(names: list[str], args):
    budget = None
    if "jev" in names:
        # Validate key before running any local generations or creating a ledger.
        if not os.environ.get("TYPESAFE_API_KEY", "").strip():
            raise ValueError("Jev requires TYPESAFE_API_KEY in your environment. Use --critics fixed,llm for a free local run.")
        budget = BudgetLedger(args.budget_ledger, limit_usd=args.budget_usd)
    critics = {}
    for name in names:
        if name == "fixed":
            critics[name] = None
        elif name == "jev":
            critics[name] = JevCritic(JevClient(model=args.jev_model, budget=budget, timeout=min(args.timeout, 60)), args.threshold)
        elif name == "openjev":
            # Only loopback OpenJev allowed; cannot accidentally select the paid endpoint.
            OllamaClient(base_url=args.openjev_url)
            critics[name] = JevCritic(JevClient(base_url=args.openjev_url, model=args.openjev_model,
                                                timeout=args.timeout), args.threshold)
        else:
            client = OllamaClient(base_url=args.ollama_url, model=args.critic_model, seed=args.seed,
                                  timeout=args.timeout, num_ctx=args.context_tokens)
            critics[name] = LocalTypedCritic(client, args.threshold) if name == "local-typed" else LLMCritic(client)
    return critics, budget


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.command == "prepare-hotpot":
            from .hotpot import prepare_hotpot, SOURCE_URL
            if args.output.exists():
                raise ValueError(f"Output already exists: {args.output}. Use a new --output directory to create another subset.")
            if args.dev_size < 1 or args.test_size < 1:
                raise ValueError("dev-size and test-size must be positive")
            download = None
            source, source_url = args.source, SOURCE_URL
            if source is None:
                from .benchmark_download import download_hotpot
                source, download = download_hotpot(args.cache_dir)
                source_url = download["url"]
            manifest = prepare_hotpot(source, args.output, args.dev_size, args.test_size, args.seed, source_url)
            if download:
                manifest["download"] = download
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Prepared {args.dev_size} development + {args.test_size} local held-out tasks: {args.output.resolve() / 'tasks.jsonl'}")
            print("Both subsets derive from the public official development split; this is not the hidden official test set.")
            return 0
        if args.command == "doctor":
            report = {"python": sys.version.split()[0], "jev_key_configured": bool(os.environ.get("TYPESAFE_API_KEY")),
                      "ollama_url": args.ollama_url, "hotpotqa_prepared": HOTPOT_DATA.exists(),
                      "hotpotqa_tasks": str(HOTPOT_DATA)}
            try:
                report["models"] = local_metadata(args.ollama_url)
                report["ollama_reachable"] = True
            except Exception as exc:
                report.update(ollama_reachable=False, error=f"{type(exc).__name__}: {exc}")
            print(json.dumps(report, indent=2))
            return 0 if report["ollama_reachable"] else 1
        if not math.isfinite(args.threshold) or not 0 < args.threshold <= 1:
            raise ValueError("threshold must be finite and in (0, 1]")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if not 2048 <= args.context_tokens <= 65536:
            raise ValueError("context-tokens must be between 2048 and 65536")
        if not math.isfinite(args.budget_usd) or not 0 < args.budget_usd <= 30:
            raise ValueError("budget-usd must be finite and in (0, 30]")
        if args.command == "judge":
            state = json.loads(args.input.read_text(encoding="utf-8"))
            critics, budget = make_critics([args.backend], args)
            review = critics[args.backend].review(state)
            result = {"backend": args.backend, **asdict(review)}
            if budget:
                result["budget"] = budget.snapshot()
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        args.dataset = args.dataset or (HOTPOT_DATA if args.environment == "hotpotqa" else DEFAULT_DATA)
        if not args.dataset.exists():
            raise ValueError(f"Dataset not found: {args.dataset}. Run python -m haltiq prepare-hotpot first, or use --environment diagnostic.")
        tasks = load_tasks(args.dataset, args.split, args.limit)
        if not 1 <= args.max_rounds <= 10:
            raise ValueError("max-rounds must be between 1 and 10")
        output = args.output or ROOT / "results" / "runs" / (args.command + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        if output.exists():
            raise ValueError(f"Output already exists: {output}. Choose a new directory.")
        demo = args.command == "demo"
        budget, writer, metadata = None, None, []
        if demo:
            critics = {"fixed": None, "mock-typed": None}
        else:
            names = [n.strip() for n in args.critics.split(",")]
            if len(set(names)) != len(names) or any(n not in {"fixed", "llm", "jev", "local-typed", "openjev"} for n in names):
                raise ValueError("critics must be unique values from fixed,llm,jev,local-typed,openjev")
            critics, budget = make_critics(names, args)
            metadata = local_metadata(args.ollama_url)
            writer = Writer(OllamaClient(base_url=args.ollama_url, model=args.writer_model,
                                        seed=args.seed, timeout=args.timeout, num_ctx=args.context_tokens))
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(output=str(output.resolve()), dataset=str(args.dataset.resolve()), arms=list(critics), local_models=metadata)
        result = run_experiment(tasks, writer, critics, output, config, budget=budget, demo=demo)
        print(f"\nSaved report: {output.resolve() / 'report.md'}")
        return 1 if result["manifest"]["status"] != "completed" else 0
    except KeyboardInterrupt:
        print("\nInterrupted. Completed traces and any budget reservations have been preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
