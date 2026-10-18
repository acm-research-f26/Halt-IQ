"""Optional hosted Jev1.13 comparator on an already-frozen Strands cohort.

Importing this module makes no network call and reads no credential. The CLI
prompts privately only when explicitly invoked with --allow-hosted-calls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import getpass
import json
from pathlib import Path
import sqlite3
import warnings

from haltiq.budget import BudgetLedger, JEV_INPUT_RATE, JEV_MAX_INPUT_TOKENS
from haltiq.providers import JevClient

from . import evaluate as core
from .data import fingerprint, load_tasks
from .decider import CORRECTNESS_CRITERIA, CORRECTNESS_INSTRUCTIONS, sha256_file, validate_public_state

MODEL = "jev-1.13.0"
QUESTIONS = {"correct": {"type": "noul", "instructions": CORRECTNESS_INSTRUCTIONS,
                          "criteria": CORRECTNESS_CRITERIA}}
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "preflight-2026-10-18/jev-budget.sqlite"
SOFT_CAP = 4.50


def hidden_key_entry() -> str:
    """Never let getpass fall back to echoed/noninteractive credential input."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass("Jev API key (hidden; never logged): ")
        except getpass.GetPassWarning:
            raise RuntimeError("Hidden credential entry is unavailable; use a real interactive terminal. No key was read or sent.") from None


def read_existing_ledger(path: str | Path = DEFAULT_LEDGER) -> dict:
    """Read-only audit: never creates/replaces/resets the shared cumulative file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("existing cumulative Jev ledger is required; no new ledger is created")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        rate = connection.execute("SELECT value FROM settings WHERE key='rate_per_million'").fetchone()
        row = connection.execute("SELECT COUNT(*),COALESCE(SUM(settled),0),COALESCE(SUM(charge_nanos),0),"
                                 "COALESCE(SUM(CASE WHEN settled=0 THEN charge_nanos ELSE 0 END),0) FROM attempts").fetchone()
    finally:
        connection.close()
    if rate is None or float(rate[0]) != JEV_INPUT_RATE:
        raise ValueError("existing ledger rate differs from verified Jev pricing")
    attempts, settled, charge, reserved = row
    return {"attempts": attempts, "settled": settled, "unsettled": attempts - settled,
            "used_usd": charge / 1e9, "reserved_usd": reserved / 1e9,
            "soft_cap_usd": SOFT_CAP, "absolute_cap_usd": 5.0}


class HostedJevDecider:
    """Public-state adapter over the existing tested, budget-reserving client."""
    def __init__(self, client: JevClient):
        if client.model != MODEL or client.local or client.budget is None:
            raise ValueError("this optional arm requires official pinned Jev1.13 with persistent accounting")
        if client.budget.limit_usd > SOFT_CAP:
            raise ValueError("optional comparator must preserve the existing $4.50 soft cap")
        self.client = client
        self.source = {"adapter": sha256_file(__file__),
                       "client": sha256_file(Path(core.__file__).parent.parent / "haltiq/providers.py"),
                       "budget": sha256_file(Path(core.__file__).parent.parent / "haltiq/budget.py")}

    def score(self, state, *, temperature=None):
        if temperature is not None:
            raise ValueError("hosted Jev probabilities are not replaced by local temperature scaling")
        public = validate_public_state(state)
        current = {"adapter": sha256_file(__file__),
                   "client": sha256_file(Path(core.__file__).parent.parent / "haltiq/providers.py"),
                   "budget": sha256_file(Path(core.__file__).parent.parent / "haltiq/budget.py")}
        if current != self.source:
            raise RuntimeError("Jev adapter changed during a paid stage")
        call = self.client.judge(public, QUESTIONS)
        p = call.data["answers"]["correct"]["noul"]
        return {"p_true": p, "input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
                "physical_input_tokens": call.input_tokens, "elapsed_seconds": call.latency_ms / 1000,
                "cost_usd": call.cost_usd, "model_id": call.model,
                "logits": None, "temperature": None, "reused_encoding": False,
                "probability_semantics": "provider Noul probability; raw logits unavailable; no fabricated logits"}

    def offload(self):
        return 0.0


class PaidGuard:
    """Budget/admission checks before journaling dispatch; the client reserves atomically."""
    def __init__(self, guard, ledger, client):
        self.guard, self.ledger, self.client = guard, ledger, client

    @property
    def idle_seconds(self):
        return self.guard.idle_seconds

    def before(self):
        state = self.ledger.snapshot()
        if state["unsettled_requests"]:
            raise core.UncertainRequest("Paid admission stopped before dispatch: existing unsettled attempts require reconciliation")
        worst = JEV_MAX_INPUT_TOKENS * JEV_INPUT_RATE / 1e6
        if state["remaining_usd"] < worst:
            raise core.UncertainRequest("Paid admission stopped before dispatch: insufficient cumulative soft budget")
        try:
            self.guard.before()
        except Exception as exc:
            raise core.UncertainRequest("Paid admission stopped before dispatch by deadline/resource/user control") from exc
        remaining = self.guard.deadline - datetime.now().timestamp()
        if remaining <= 0:
            raise core.UncertainRequest("Paid admission stopped before dispatch: deadline reached")
        self.client.timeout = min(60, remaining)

    def after(self, elapsed):
        self.guard.after(elapsed)


def _finalize_paid_usage(output: Path, tasks, config, ledger, initial_budget):
    """Add hosted outputs/dollars to Strands-zero-cost generic replay accounting."""
    if not (output / "episodes.json").is_file():
        return
    existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if existing["frozen"]["config"] != config:
        return  # Never rewrite an unrelated run after its initialization was refused.
    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    for episode in episodes:
        decisions = episode.get("decisions", [])
        episode["decision_output_tokens"] = sum(d["score"].get("output_tokens", 0) for d in decisions)
        episode["decision_cost_usd"] = sum(d["score"].get("cost_usd", 0) for d in decisions)
        # Idempotent resume/finalization: derive writer-only cost from its stored field.
        writer_cost = episode.setdefault("writer_cost_usd", episode["cost_usd"])
        episode["cost_usd"] = writer_cost + episode["decision_cost_usd"]
        episode["hosted_probability_only"] = True
    core.json_write(output / "episodes.json", episodes)
    policies = list(dict.fromkeys(row["policy"] for row in episodes))
    summary = core.summarize(episodes, [t.id for t in tasks], policies)
    for name in policies:
        rows = [r for r in episodes if r["policy"] == name]
        summary[name]["mean_raw_reported_tokens"] = sum(r["writer_input_tokens"] + r["writer_output_tokens"]
            + r["decision_input_tokens"] + r["decision_output_tokens"] for r in rows) / len(tasks)
        summary[name]["counterfactual_hosted_cost_usd"] = sum(r["decision_cost_usd"] for r in rows)
        summary[name]["raw_usage_includes_decision_output_tokens"] = True
    core.json_write(output / "summary.json", summary)
    exported = core.export_common_schema(output, episodes, tasks, config)
    episodes_by_key = {(r["task_id"], r["policy"]): r for r in episodes}
    for row in exported:
        if row["total_tokens"] is not None:
            row["total_tokens"] += episodes_by_key[(row["question_id"], row["policy"])]["decision_output_tokens"]
    (output / "common-schema-results.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in exported), encoding="utf-8")
    current = ledger.snapshot()
    core.json_write(output / "jev-budget-receipt.json", {"before": initial_budget, "after": current,
        "note": "Same cumulative SQLite ledger; never reset. Settled charges plus retained uncertain reservations bound paid work. Shared writer prefixes are separate."})
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(hosted_accounting_finalized=True, budget_receipt_sha256=sha256_file(output / "jev-budget-receipt.json"),
                    provider_probability_only=True, no_raw_provider_logits=True,
                    main_strands_training_or_campaign_modified=False)
    core.json_write(output / "manifest.json", manifest)


def run_matched_jev(reference_run: str | Path, trajectory_dir: str | Path, dataset: str | Path,
                    output: str | Path, client: JevClient, guard, *, resume=False):
    """Execute only when root explicitly invokes this optional hosted stage later."""
    reference, source, output = Path(reference_run), Path(trajectory_dir), Path(output)
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    frozen, reference_config = manifest["frozen"], manifest["frozen"]["config"]
    if manifest.get("status") not in {"completed", "completed_with_errors"} or reference_config.get("phase") != "confirmation":
        raise ValueError("an existing terminal frozen Strands confirmation run is required first")
    tasks = load_tasks(dataset, ids=frozen["task_ids"])
    if [t.id for t in tasks] != frozen["task_ids"] or fingerprint([asdict(t) for t in tasks]) != frozen["dataset_state_sha256"]:
        raise ValueError("optional comparator questions/evidence/references/order differ from frozen confirmation")
    trajectory_path = source / "trajectories.json"
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if original.get("status") not in {"completed", "completed_with_errors"} or original.get("writer_unload_verified") is not True:
        raise ValueError("terminal unloaded writer trajectories required")
    if sha256_file(trajectory_path) != reference_config["trajectory_sha256"]:
        raise ValueError("optional comparator trajectory differs from frozen Strands source")
    source_tasks = {r["id"]: r for r in json.loads((source / "tasks-evaluator-only.json").read_text(encoding="utf-8"))}
    if any(source_tasks.get(t.id) != asdict(t) for t in tasks):
        raise ValueError("optional comparator task state differs from the writer source")
    decider = HostedJevDecider(client)
    initial_budget = client.budget.snapshot()
    if initial_budget["unsettled_requests"]:
        raise RuntimeError("unsettled paid requests prevent optional launch")
    if reference_config.get("threshold") != .8 or not 1 <= reference_config["max_drafts"] <= 10:
        raise ValueError("optional Jev parity requires frozen threshold .8 and cap at most10")
    config = {**reference_config, "mode": "hosted-jev-replay", "policy_name": "jev-1.13-single-correctness",
              "temperature": None, "phase": "confirmation", "decision_device": "hosted",
              "model_fingerprint": fingerprint({"model": MODEL, "questions": QUESTIONS, "source": decider.source}),
              "hosted_source_sha256": decider.source,
              "paired_reference_manifest_sha256": sha256_file(reference / "manifest.json"),
              "shared_ledger_path": str(client.budget.path), "soft_budget_usd": SOFT_CAP,
              "hosted_accounting_finalization_required": True,
              "decision_window_note": "Provider's64Ktotalcontext differs from localStrands4096; samecompletepublicstate, no truncation."}
    expected = {"config": config, "source_sha256": core.source_hash(), "task_ids": [t.id for t in tasks],
                "dataset_state_sha256": fingerprint([asdict(t) for t in tasks])}
    if (output / "manifest.json").exists():
        if not resume or json.loads((output / "manifest.json").read_text(encoding="utf-8"))["frozen"] != expected:
            raise ValueError("existing optional output requires explicit identical-source/config resume")
        receipt = output / "jev-budget-receipt.json"
        if receipt.exists():
            initial_budget = json.loads(receipt.read_text(encoding="utf-8"))["before"]
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    keep = {t.id for t in tasks}
    trajectories = [r for r in trajectories if r["task_id"] in keep]
    try:
        core.replay_trajectories(tasks, trajectories, decider, output, config,
                                PaidGuard(guard, client.budget, client), resume=resume)
    finally:
        _finalize_paid_usage(output, tasks, config, client.budget, initial_budget)
    return json.loads((output / "episodes.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-run", "trajectory-dir", "dataset", "output", "deadline"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allow-hosted-calls", action="store_true", help="Explicit optional operator launch; never part of the primary campaign")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.allow_hosted_calls:
        parser.error("optional paid stage requires explicit --allow-hosted-calls")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline.timestamp() <= datetime.now().timestamp():
        parser.error("future deadline with timezone offset required")
    audit = read_existing_ledger()
    if audit["unsettled"] or audit["used_usd"] >= SOFT_CAP:
        parser.error("shared ledger has unsettled attempts or exhausted soft budget")
    ledger = BudgetLedger(DEFAULT_LEDGER, limit_usd=SOFT_CAP)
    key = hidden_key_entry()
    try:
        client = JevClient(api_key=key, budget=ledger, timeout=60)
    finally:
        del key
    guard = core.Governor(args.output, deadline.timestamp(), require_gpu=False)
    run_matched_jev(args.reference_run, args.trajectory_dir, args.dataset, args.output,
                    client, guard, resume=args.resume)


if __name__ == "__main__":
    main()
