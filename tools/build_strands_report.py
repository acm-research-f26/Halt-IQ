"""Build an evidence-bounded professor PDF without models, network or inference.

Use the bundled document Python (reportlab/pypdf), not the study environment.
An explicit fresh N of0 produces a draft without selecting a confirmation panel.
Existing output directories must be empty; historical reports are never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
import importlib.util
import json
import math
from pathlib import Path
import sys

# Direct script execution must resolve the maintained sibling package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TERMINAL = {"completed", "completed_with_errors"}
DEVELOPMENT_AUDITOR_SHA = "5fa27dcaa09030d95ef0d77057a417b4343b05671029918c71f931f844e60fb1"
DEVELOPMENT_EVIDENCE_SHA = "2ecdc14f67781d16b8e242b13bdd6909b411a5270a00a29545bfd1f54b596062"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("--as-of requires an explicit UTC offset")
    return result


def number(value, places=3):
    if value is None:
        return "not recorded"
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("nonfinite or nonnumeric report metric")
    return f"{value:,.{places}f}"


class Evidence:
    """Read each selected input once and bind the report to its exact bytes."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files, self.missing, self.cache = {}, [], {}

    def path(self, value):
        value = Path(value)
        return value.resolve() if value.is_absolute() else (self.root / value).resolve()

    def label(self, path):
        for base, prefix in ((self.root, "runtime/"), (self.root.parent.parent, "project/")):
            try:
                return prefix + Path(path).relative_to(base).as_posix()
            except ValueError:
                pass
        return "external/" + Path(path).name

    def read(self, value, role, *, lines=False):
        path = self.path(value)
        if path in self.cache:
            return self.cache[path]
        if not path.is_file():
            self.missing.append({"path": str(path), "role": role})
            return None
        raw = path.read_bytes()
        raw_hash = hashlib.sha256(raw).hexdigest()
        if path in self.files:
            if self.files[path]["sha256"] != raw_hash:
                raise ValueError("evidence changed before parsing")
        else:
            self.files[path] = {"id": f"S{len(self.files)+1}", "path": str(path), "label": self.label(path), "role": role,
                "sha256": raw_hash, "bytes": len(raw)}
        text = raw.decode("utf-8-sig")
        if lines:
            if raw and not raw.endswith(b"\n"):
                raise ValueError(f"incomplete journal: {path}")
            result = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            result = json.loads(text)
        self.cache[path] = result
        return result

    def stable(self):
        if any(not p.is_file() or digest(p) != r["sha256"] for p, r in self.files.items()):
            raise ValueError("report inputs changed during collection; rerun at a stable checkpoint")

    def record_file(self, value, role):
        path = self.path(value)
        if not path.is_file():
            self.missing.append({"path": str(path), "role": role})
            return None
        if path not in self.files:
            self.files[path] = {"id": f"S{len(self.files)+1}", "path": str(path),
                "label": self.label(path), "role": role, "sha256": digest(path), "bytes": path.stat().st_size}
        return self.files[path]["sha256"]


def journal_counts(rows):
    started, completed, errors = set(), set(), 0
    for row in rows or []:
        key, event = row["request_id"], row["event"]
        if event == "request_started":
            if key in started:
                raise ValueError("duplicate request identifier")
            started.add(key)
        elif event == "request_completed":
            if key not in started or key in completed:
                raise ValueError("unmatched request completion")
            completed.add(key)
        elif event == "request_error":
            errors += 1
        else:
            raise ValueError("unknown journal event")
    return {"started": len(started), "completed": len(completed),
            "pending": len(started - completed), "errors": errors}


def train_reconciliation(manifest, trajectories, events):
    """Distinguish physical completions from durable, scored trajectory rows."""
    ids = manifest.get("frozen", {}).get("task_ids", [])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate planned TRAIN identifiers")
    owners, completed, dispatched = {}, set(), set()
    for event in events:
        key = event["request_id"]
        pieces = key.rsplit("|fixed|writer|", 1)
        if len(pieces) != 2 or pieces[0] not in ids or not pieces[1].isdigit() or not 1 <= int(pieces[1]) <= 10:
            raise ValueError("foreign or malformed TRAIN writer request")
        owners[key] = pieces[0]
        if event["event"] == "request_started":
            dispatched.add(pieces[0])
        elif event["event"] == "request_completed":
            completed.add(key)
    represented = {f"{t['task_id']}|fixed|writer|{r['round']}"
        for t in trajectories for r in t.get("rounds", [])}
    if not represented <= completed:
        raise ValueError("saved TRAIN round lacks a completed physical call")
    saved = {t["task_id"] for t in trajectories}
    return {"planned_questions": len(ids), "saved_trajectories": len(saved),
        "missing_trajectories": len(set(ids) - saved),
        "undispatched_questions": len(set(ids) - dispatched),
        "completed_calls_without_saved_round": len(completed - represented),
        "completed_questions_without_saved_trajectory": len({owners[k] for k in completed} - saved),
        "saved_trajectory_failures": sum(t.get("status") != "completed" for t in trajectories)}


def train_invocation_timings(evidence, train, manifest, archive_dirs):
    """Sum only explicitly supplied, hash-bound, nonoverlapping invocations."""
    current_raw = (train / "requests.jsonl").read_bytes()
    selected, previous_raw = [], b""
    for directory in archive_dirs:
        directory = evidence.path(directory)
        receipt = evidence.read(directory / "snapshot-receipt.json", "archived TRAIN segment identity")
        hashes = (receipt or {}).get("files_sha256", {})
        if (not {"manifest.json", "requests.jsonl"} <= set(hashes)
                or evidence.path(receipt["source"]) != train):
            raise ValueError("explicit TRAIN segment receipt does not bind this source")
        for name, expected in hashes.items():
            if Path(name).name != name or evidence.record_file(directory / name, "archived TRAIN segment bytes") != expected:
                raise ValueError("archived TRAIN segment fingerprint differs")
        source = evidence.read(directory / "manifest.json", "archived TRAIN invocation timing")
        raw = (directory / "requests.jsonl").read_bytes()
        if source.get("frozen") != manifest.get("frozen") or not raw.startswith(previous_raw) or not current_raw.startswith(raw):
            raise ValueError("TRAIN segment source or cumulative journal prefix differs")
        journal = journal_counts(evidence.read(directory / "requests.jsonl", "archived TRAIN journal", lines=True))
        usage = source.get("actual_evaluation_usage", {})
        if journal["pending"] or usage.get("completed_requests_including_invalid_outputs") != journal["completed"]:
            raise ValueError("archived TRAIN segment physical journal is not reconciled")
        selected.append((source, journal["completed"]))
        previous_raw = raw
    if "this_invocation_wall_seconds" not in manifest:
        if selected:
            raise ValueError("current TRAIN invocation timing missing")
        return []
    selected.append((manifest, journal_counts(evidence.read(train / "requests.jsonl", "current TRAIN journal", lines=True))["completed"]))
    result, last_finish, last_calls = [], None, 0
    for index, (source, calls) in enumerate(selected, 1):
        wall, idle = source.get("this_invocation_wall_seconds"), source.get("this_invocation_intentional_idle_seconds")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (wall, idle)) or idle > wall:
            raise ValueError("invalid TRAIN invocation wall/idle duration")
        finish = timestamp(source["finished_at_utc"])
        start = finish - timedelta(seconds=wall)
        if (last_finish is not None and start < last_finish) or calls < last_calls:
            raise ValueError("TRAIN invocation intervals overlap or cumulative calls decrease")
        result.append({"invocation": index, "archived": index < len(selected),
            "derived_start_utc": start.astimezone(timezone.utc).isoformat(),
            "finish_before_unload_utc": finish.astimezone(timezone.utc).isoformat(),
            "wall_seconds": wall, "intentional_idle_seconds": idle,
            "cumulative_physical_calls": calls,
            "additional_physical_calls": calls - last_calls if index > 1 else None})
        last_finish, last_calls = finish, calls
    return result


def train_source_evidence(evidence, train, archives, role="TRAIN"):
    """Reconcile one physical writer source; never merge different source journals."""
    manifest = evidence.read(train / "manifest.json", role + " generation manifest") or {}
    control = evidence.read(train / "control-status.json", role + " resource/control observation") or {}
    trajectories = evidence.read(train / "trajectories.json", role + " attempted trajectories") or []
    tasks = evidence.read(train / "tasks-evaluator-only.json", role + " original partition metadata") or []
    events = evidence.read(train / "requests.jsonl", role + " physical request journal", lines=True) or []
    journal = journal_counts(events)
    by_id = {t["id"]: t for t in tasks}
    ids = manifest.get("frozen", {}).get("task_ids", [])
    saved = [t["task_id"] for t in trajectories]
    if (len(by_id) != len(tasks) or len(set(saved)) != len(saved)
            or set(by_id) != set(ids) or any(q not in ids or q not in by_id for q in saved)):
        raise ValueError("duplicate or foreign TRAIN trajectory/task identifiers")
    if manifest.get("status") in TERMINAL and len(saved) != len(ids):
        raise ValueError("terminal TRAIN source lacks its full planned trajectories")
    usage = manifest.get("actual_evaluation_usage", {})
    if (usage.get("completed_requests_including_invalid_outputs") != journal["completed"]
            or usage.get("unreconciled_requests", journal["pending"]) != journal["pending"]):
        raise ValueError("TRAIN manifest and physical journal counts differ")
    return {"manifest": manifest, "control": control, "trajectories": trajectories,
        "journal": journal, "planned": len(ids),
        "journal_sha256": evidence.files[evidence.path(train / "requests.jsonl")]["sha256"],
        "physical_dispatches": {(r["request_id"], timestamp(r["at_utc"]).astimezone(timezone.utc).isoformat())
            for r in events if r["event"] == "request_started" and r.get("at_utc")},
        "partitions": Counter(by_id[t["task_id"]].get("partition", "unknown") for t in trajectories),
        "reconciliation": train_reconciliation(manifest, trajectories, events),
        "timings": train_invocation_timings(evidence, train, manifest, archives)}


def baseline_valid(receipt):
    from pc_strands.train_head import validate_baseline_receipt
    try:
        validate_baseline_receipt(receipt or {}, (receipt or {}).get("checkpoint_head_sha256"))
    except (ValueError, KeyError, TypeError):
        return False
    return bool(receipt.get("checkpoint_head_sha256"))


def baseline_diagnostics(evidence, receipt_path, receipt, plan, campaign, campaign_dir):
    """Recompute cap-one development evidence; never consume its loose summary."""
    from pc_strands.data import fingerprint
    from pc_strands.metrics import evaluate

    directory = receipt_path.parent
    manifest = evidence.read(directory / "manifest.json", "bound untouched development manifest")
    episodes = evidence.read(directory / "episodes.json", "bound untouched development episodes")
    tasks = evidence.read(directory / "tasks-evaluator-only.json", "untouched evaluator task snapshot")
    events = evidence.read(directory / "requests.jsonl", "untouched physical native request journal", lines=True)
    if any(value is None for value in (manifest, episodes, tasks, events)):
        raise ValueError("complete baseline manifest/episodes/tasks/journal required")
    for name, key in (("manifest.json", "manifest_sha256"), ("episodes.json", "episodes_sha256")):
        if evidence.record_file(directory / name, "baseline receipt identity") != receipt.get(key):
            raise ValueError("baseline receipt/result fingerprint differs")
    journal_path = directory / "requests.jsonl"
    anchors = [receipt["requests_sha256"]] if receipt.get("requests_sha256") else []
    reuse = (plan or {}).get("baseline_reuse", {})
    if reuse:
        if evidence.path(reuse["run"]) != directory:
            raise ValueError("reviewed baseline reuse points at a different run")
        hashes = reuse.get("hashes", {})
        for name, key in ((receipt_path.name, "baseline_receipt"), ("manifest.json", "baseline_manifest"),
                          ("episodes.json", "baseline_episodes")):
            if hashes.get(key) != evidence.record_file(directory / name, "reviewed baseline reuse identity"):
                raise ValueError("reviewed baseline reuse fingerprint differs")
        if hashes.get("baseline_requests"):
            anchors.append(hashes["baseline_requests"])
    for stage in (campaign or {}).get("stages", {}).values():
        captured = {evidence.path(k): value for k, value in stage.get("output_sha256", {}).items()}
        if stage.get("status") == "completed" and captured.get(receipt_path) == digest(receipt_path):
            if (not campaign_dir or campaign.get("plan_sha256") != evidence.record_file(
                    evidence.path(campaign_dir) / "plan.json", "executed baseline plan")):
                raise ValueError("captured baseline journal lacks matching executed plan")
            if captured.get(journal_path):
                anchors.append(captured[journal_path])
    journal_hash = evidence.record_file(journal_path, "hash-bound baseline native journal")
    if not anchors or any(anchor != journal_hash for anchor in anchors):
        raise ValueError("baseline native journal requires an exact receipt/plan/captured-output hash")
    frozen, ids = manifest.get("frozen", {}), receipt["planned_task_ids"]
    config = frozen.get("config", {})
    if (manifest.get("status") not in TERMINAL or frozen.get("task_ids") != ids
            or [t["id"] for t in tasks] != ids or fingerprint(tasks) != frozen.get("dataset_state_sha256")
            or frozen.get("source_sha256") != receipt.get("source_sha256")
            or config.get("phase") != "untuned_pretraining" or config.get("max_drafts") != 1
            or config.get("mode") != "replay" or config.get("decision_device") != "cuda"
            or config.get("threshold") != .8 or receipt.get("model", {}).get("max_length") != 4096):
        raise ValueError("baseline requires exact original task/source/native cap-one protocol")
    counts = journal_counts(events)
    if counts["pending"] or counts["errors"]:
        raise ValueError("baseline journal has unresolved or error requests")
    policy = config["policy_name"]
    rows = {(r["task_id"], r["policy"]): r for r in episodes}
    if len(rows) != len(episodes) or set(rows) != {(qid, p) for qid in ids for p in (policy, "fixed-1-prefix")}:
        raise ValueError("baseline planned question-policy coverage differs")
    started = {r["request_id"]: r for r in events if r["event"] == "request_started"}
    completed = {r["request_id"]: r for r in events if r["event"] == "request_completed"}
    if any(key not in {f"{qid}|{policy}|decision|1" for qid in ids} for key in started):
        raise ValueError("foreign baseline native request")
    quality = {p: {"planned_n": len(ids), "completed": 0, "em": 0., "token_f1": 0.}
               for p in (policy, "fixed-1-prefix")}
    approvals = correct_approvals = correct_rejections = valid = 0
    for task in tasks:
        qid = task["id"]
        primary, fixed = rows[qid, policy], rows[qid, "fixed-1-prefix"]
        if primary["final_answer"] != fixed["final_answer"]:
            raise ValueError("cap-one baseline and Fixed1 answers differ")
        score = evaluate(fixed["final_answer"], task["answers"])
        first_correct = fixed["status"] == "completed" and score["exact_match"]
        for name, row in ((policy, primary), ("fixed-1-prefix", fixed)):
            if row["status"] not in {"completed", "failed"}:
                raise ValueError("nonterminal baseline episode")
            actual = score if row["status"] == "completed" else {"exact_match": False, "token_f1": 0.}
            if (type(row.get("exact_match")) is not bool or row["exact_match"] != actual["exact_match"]
                    or not math.isclose(row["token_f1"], actual["token_f1"], abs_tol=1e-12)
                    or row.get("first_draft_exact_match") != first_correct):
                raise ValueError("baseline stored quality differs from canonical answer scoring")
            quality[name]["completed"] += row["status"] == "completed"
            quality[name]["em"] += actual["exact_match"] / len(ids)
            quality[name]["token_f1"] += actual["token_f1"] / len(ids)
        key = f"{qid}|{policy}|decision|1"
        decisions, call = primary.get("decisions", []), completed.get(key)
        if primary.get("decision_calls") != int(call is not None):
            raise ValueError("baseline episode/native request count differs")
        if call is None or "failure" in call["result"]:
            if decisions or primary["status"] != "failed":
                raise ValueError("missing baseline native score represented as success")
            continue
        raw, request = call["result"], started[key]
        expected_state = {"question": task["question"], "evidence": task["evidence"], "draft": fixed["final_answer"]}
        if (request["input"].get("state") != expected_state
                or set(request["input"]["state"]) != {"question", "evidence", "draft"}
                or request["input"].get("model") != config.get("model_fingerprint")
                or request["input_sha256"] != fingerprint(request["input"])
                or call["input_sha256"] != request["input_sha256"]
                or len(decisions) != 1 or decisions[0].get("score") != raw):
            raise ValueError("baseline public state/raw critic response differs from journal")
        probability, logits, temperature = raw["p_true"], raw["logits"], raw["temperature"]
        if (type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1
                or len(logits) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)
                or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
                or type(raw.get("physical_input_tokens")) is not int or raw["physical_input_tokens"] <= 0
                or raw.get("reused_encoding") is not False or raw.get("input_tokens") != raw["physical_input_tokens"]
                or raw["input_tokens"] > 4096 or primary.get("decision_input_tokens") != raw["input_tokens"]):
            raise ValueError("baseline requires finite raw logits and a real untruncated native forward")
        x = (logits[1] - logits[0]) / temperature
        calculated = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        approved = probability >= config["threshold"]
        if (not math.isclose(calculated, probability, abs_tol=1e-12)
                or decisions[0].get("round") != 1 or type(decisions[0].get("approved")) is not bool
                or decisions[0]["approved"] != approved or decisions[0].get("draft_exact_match") != first_correct
                or primary.get("stop_reason") != ("approved" if approved else "round_cap")):
            raise ValueError("baseline probability/approval differs from raw score or threshold")
        valid += 1
        approvals += approved
        correct_approvals += approved and first_correct
        correct_rejections += not approved and first_correct
    if (quality[policy]["completed"] != receipt["completed"] or valid != receipt["valid_decision_forwards"]
            or manifest.get("actual_decision_requests") != counts["completed"]):
        raise ValueError("baseline receipt counts differ from actual episodes/native calls")
    calls = list(completed.values())
    usage = {"completed_requests_including_invalid_outputs": len(calls), "unreconciled_requests": 0,
        "reported_physical_input_tokens": sum(c["result"].get("physical_input_tokens", 0) for c in calls),
        "reported_output_tokens": sum(c["result"].get("output_tokens", 0) for c in calls),
        "requests_with_unknown_usage": sum(c["result"].get("input_tokens") is None for c in calls),
        "reported_request_seconds": sum(c["request_elapsed_seconds"] for c in calls),
        "hosted_cost_usd": sum(c["result"].get("cost_usd", 0) for c in calls)}
    recorded = manifest.get("actual_evaluation_usage", {})
    for key, value in usage.items():
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or type(recorded.get(key)) not in (int, float)
                or not math.isclose(value, recorded[key], abs_tol=1e-9)):
            raise ValueError("baseline physical usage differs from native journal")
    def duration(value):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("invalid bound baseline timing")
        return value
    phase, idle = (duration(manifest.get(k)) for k in ("this_invocation_wall_seconds", "this_invocation_intentional_idle_seconds"))
    startup = manifest.get("backend_startup_this_invocation", {})
    startup_wall, startup_idle = (duration(startup.get(k)) for k in ("wall_seconds", "intentional_idle_seconds"))
    if (phase is not None and idle is not None and idle > phase
            or startup_wall is not None and startup_idle is not None and startup_idle > startup_wall):
        raise ValueError("baseline idle exceeds its corresponding wall scope")
    return {"scope": "DEVELOPMENT cap1 before training; not fresh or repair performance", "policies": quality,
        "approvals": approvals, "correct_approvals": correct_approvals, "wrong_approvals": approvals - correct_approvals,
        "rejections_of_correct_drafts": correct_rejections, "valid_native_forwards": valid, "usage": usage,
        "timing": {"startup_wall_seconds": startup_wall, "startup_idle_seconds": startup_idle,
            "phase_wall_seconds": phase, "phase_idle_seconds": idle,
            "startup_plus_phase_wall_seconds": startup_wall + phase if startup_wall is not None and phase is not None else None},
        "journal_sha256": journal_hash}


def training_valid(proof):
    if not proof:
        return False
    fit, tune = set(proof.get("fit_question_ids", [])), set(proof.get("tune_question_ids", []))
    delta, error = proof.get("head_delta_l2"), proof.get("reload_max_logit_error")
    finite_delta = type(delta) in (int, float) and math.isfinite(delta) and delta > 0
    finite_error = type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1e-6
    return (proof.get("status") in {"trained", "trained_bounded_early_stop"}
        and bool(fit) and bool(tune) and not fit & tune
        and finite_delta and finite_error and proof.get("reload_verified") is True
        and proof.get("initial_head_tensor_sha256") != proof.get("trained_head_tensor_sha256")
        and all(proof.get(k) is True for k in ("source_head_unchanged", "source_adapter_unchanged", "frozen_torso_adapter"))
        and proof.get("calibration_used") is False and proof.get("fresh100_used") is False)


def validate_comparison(data, n):
    """Check the published comparison contract, without choosing or rescoring runs."""
    if data.get("status") != "independently_joined_terminal_paired_replays":
        raise ValueError("terminal independently joined replay analysis required")
    ids = data.get("planned_question_ids", [])
    if data.get("planned_n") != n or len(ids) != n or len(set(ids)) != n:
        raise ValueError("paired analysis differs from explicit fresh N")
    if not data.get("policies") or not data.get("paired_comparisons"):
        raise ValueError("comparison policies/paired intervals missing")
    bootstrap = data.get("bootstrap", {})
    if bootstrap.get("draws") != 5000 or bootstrap.get("seed") != 42 or bootstrap.get("unit") != "original question":
        raise ValueError("expected predeclared question-paired 5,000-draw seed 42 intervals")
    for row in data["policies"].values():
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ("planned_n", "completed", "failed_or_missing")):
            raise ValueError("invalid integer policy coverage")
        if row["planned_n"] != n or row["completed"] + row["failed_or_missing"] != n:
            raise ValueError("policy denominator/coverage differs")
        for key in ("em", "token_f1"):
            value = row[key]
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid policy quality metric")
        for key in ("mean_actual_drafts", "mean_counterfactual_prefix_model_seconds", "mean_raw_reported_tokens"):
            if row.get(key) is not None and (not math.isfinite(row[key]) or row[key] < 0):
                raise ValueError("invalid usage metric")
        probabilities = row.get("first_draft_probabilities", {})
        for key in ("auroc", "brier", "nll", "first_draft_approval_precision", "approval_coverage_of_planned"):
            value = probabilities.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (key != "nll" and value > 1)):
                raise ValueError("invalid probability diagnostic")
    for name, contrast in data["paired_comparisons"].items():
        candidate, baseline = name.split("_minus_")
        for metric, row in contrast.items():
            low, high = row["bootstrap95"]
            if row["planned_n"] != n or not 0 <= row["complete_pair_n"] <= n:
                raise ValueError("invalid paired coverage")
            if not all(math.isfinite(x) for x in (low, high, row["candidate_minus_baseline"])) or low > high:
                raise ValueError("invalid paired interval")
            key = "em" if metric == "exact_match" else "token_f1"
            expected = data["policies"][candidate][key] - data["policies"][baseline][key]
            if not math.isclose(expected, row["candidate_minus_baseline"], abs_tol=1e-12):
                raise ValueError("paired delta differs from policy means")
    for usage in data.get("physical_evaluation_usage", {}).values():
        if isinstance(usage, dict) and (usage.get("unreconciled_requests", 0) or usage.get("unreconciled_inferences", 0)):
            raise ValueError("unreconciled physical work")
    return data


def checkpoint_fingerprint(evidence, directory):
    from pc_strands.data import fingerprint
    path = evidence.path(directory)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("current checkpoint files are absent")
    return fingerprint({str(p.relative_to(path)): evidence.record_file(p, "current checkpoint identity") for p in files})


def artifact_paths(evidence, plan, n, campaign_dir=None, *, training_roles=None):
    """Resolve declared roles; an explicit plan never falls back to old artifacts."""
    defaults = {
        "baseline": "study-runs/untuned-dev20-decisions/untuned-pretraining-receipt.json",
        "released": "models/decider", "trained": "checkpoints/head-trained",
        "untuned": "checkpoints/released-calibrated", "train_source": None,
        "freeze": "fresh-N-freeze.json",
        **{f"cache_{p}": f"feature-caches/{p}.jsonl" for p in ("fit", "tune", "calibration")},
        **{f"calibration_{k}": f"study-runs/calibration-{k}/temperature-fit.json" for k in ("untuned", "trained")},
    }
    if plan is None:
        return {k: evidence.path(v) if v is not None else None for k, v in defaults.items()}
    roles = ({k: training_roles.get(k) for k in (*defaults, "control_copy_stage")}
             if training_roles is not None else {k: None for k in defaults})

    def path(value):
        if campaign_dir:
            value = value.replace("{run}", str(evidence.path(campaign_dir)))
        if "{" in value or "}" in value:
            raise ValueError("unresolved campaign artifact path")
        return evidence.path(value)

    def arg(step, flag, required=True):
        argv = step.get("argv", [])
        places = [i for i, value in enumerate(argv) if value == flag]
        if not places and not required:
            return None
        if len(places) != 1 or places[0] + 1 >= len(argv):
            raise ValueError(f"exactly one {flag} required for report artifact role")
        return argv[places[0] + 1]

    def module(step, name, mode=None):
        argv = step.get("argv", [])
        return any(argv[i:i+2] == ["-m", name] and (mode is None or argv[i+2:i+3] == [mode])
                   for i in range(len(argv)))

    def put(role, value):
        value = path(value)
        if roles.get(role) is not None and roles[role] != value:
            raise ValueError(f"conflicting reviewed campaign paths for {role}")
        roles[role] = value

    steps = plan.get("steps", [])
    for step in steps:
        if module(step, "pc_strands.train_head"):
            for role, flag in (("trained", "--output"), ("released", "--checkpoint"),
                               ("baseline", "--baseline-receipt"), ("cache_fit", "--fit-cache"),
                               ("cache_tune", "--tune-cache")):
                put(role, arg(step, flag))
        elif module(step, "pc_strands.evaluate", "replay") and arg(step, "--phase", False) == "untuned_pretraining":
            put("baseline", str(path(arg(step, "--output")) / "untuned-pretraining-receipt.json"))
            put("released", arg(step, "--checkpoint-dir"))
        elif module(step, "pc_strands.decider", "cache"):
            partition = arg(step, "--partition", False)
            if partition in ("fit", "tune", "calibration"):
                put("cache_" + partition, arg(step, "--output"))
                put("train_source", arg(step, "--trajectory-dir"))
    for step in steps:
        if module(step, "pc_strands.calibrate"):
            checkpoint = path(arg(step, "--checkpoint"))
            kind = "trained" if checkpoint == roles["trained"] else "untuned"
            put(kind, str(checkpoint))
            put("calibration_" + kind, arg(step, "--output"))
    if roles["untuned"] is not None and training_roles is None:
        copied = {roles["untuned"] / name for name in ("head.safetensors", "training-proof.json")}
        owners = [s for s in steps if copied <= {path(o["path"]) for o in s.get("outputs", [])}]
        if len(owners) > 1:
            raise ValueError("ambiguous reviewed control-copy output owner")
        roles["control_copy_stage"] = owners[0]["id"] if owners else None
    roles["freeze"] = path(plan.get("operator_fresh_freeze_path", "fresh-N-freeze.json"))
    if not n:
        return roles
    options = [o for o in plan.get("fresh_options", []) if o.get("n") == n]
    if len(options) != 1:
        raise ValueError("exactly one reviewed fresh option required")
    option = options[0]
    put("dataset", option["dataset"])
    comparisons = [s for s in option.get("steps", []) if module(s, "pc_strands.compare")]
    if len(comparisons) != 1:
        raise ValueError("one explicit canonical comparison stage required")
    comparison = comparisons[0]
    for role, flag in (("fresh_untuned", "--untuned"), ("fresh_trained", "--tuned"),
                       ("fresh_source", "--trajectory-dir"), ("fresh_cache", "--feature-cache"),
                       ("fresh_analysis", "--output")):
        put(role, arg(comparison, flag))
    generators = [s for s in option.get("steps", []) if module(s, "pc_strands.evaluate", "generate")
                  and path(arg(s, "--output")) == roles["fresh_source"]]
    encoders = [s for s in option.get("steps", []) if module(s, "pc_strands.decider", "cache")
                and path(arg(s, "--output")) == roles["fresh_cache"]]
    if (len(generators) != 1 or len(encoders) != 1
            or path(arg(generators[0], "--dataset")) != roles["dataset"]
            or path(arg(encoders[0], "--trajectory-dir")) != roles["fresh_source"]):
        raise ValueError("fresh generator/encoder roles differ from reviewed comparison")
    for kind in ("untuned", "trained"):
        matches = [s for s in option.get("steps", []) if module(s, "pc_strands.evaluate", "head-replay")
                   and path(arg(s, "--output")) == roles["fresh_" + kind]]
        if len(matches) != 1 or roles[kind] is None:
            raise ValueError(f"explicit {kind} head/calibration report roles required")
        step = matches[0]
        for flag, role in (("--checkpoint-dir", kind), ("--feature-cache", "fresh_cache"),
                           ("--trajectory-dir", "fresh_source"), ("--dataset", "dataset")):
            if path(arg(step, flag)) != roles[role]:
                raise ValueError("fresh replay paths differ from reviewed comparison/checkpoints")
        roles[kind + "_policy"] = arg(step, "--policy-name")
    return roles


def external_training_owner(evidence, directory, fresh_dir, fresh_plan, fresh_campaign):
    """Validate a separate completed owner; never synthesize imported stages."""
    from pc_strands.campaign import condition

    directory, fresh_dir = evidence.path(directory), evidence.path(fresh_dir)
    if directory == fresh_dir or fresh_plan.get("steps"):
        raise ValueError("separate fresh plan must not import training stages")
    fresh_ids = [s.get("id") for option in fresh_plan.get("fresh_options", []) for s in option.get("steps", [])]
    if (not fresh_ids or any(not isinstance(name, str) or not name or
            any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
            for name in fresh_ids) or len(set(fresh_ids)) != len(fresh_ids)
            or not set((fresh_campaign or {}).get("stages", {})) <= set(fresh_ids)):
        raise ValueError("fresh campaign state contains foreign/imported or ambiguous stage IDs")
    plan = evidence.read(directory / "plan.json", "separate training frozen plan")
    campaign = evidence.read(directory / "campaign.json", "separate completed training campaign")
    if not plan or not campaign or plan.get("fresh_options") or campaign.get("status") != "completed":
        raise ValueError("separate training owner requires its completed training-only campaign")
    plan_hash = evidence.record_file(directory / "plan.json", "separate training plan identity")
    reference = fresh_plan.get("training_campaign_reference", {})
    if (not reference.get("path") or evidence.path(reference["path"]) != directory
            or reference.get("final_plan_sha256") != plan_hash or campaign.get("plan_sha256") != plan_hash
            or reference.get("terminal_campaign_sha256") != evidence.record_file(
                directory / "campaign.json", "separate terminal training identity")):
        raise ValueError("fresh plan does not bind the exact separate training plan/campaign")
    source = plan.get("reviewed_source_sha256", {})
    if (len(source) != 15 or source != campaign.get("source_sha256")
            or source != fresh_plan.get("reviewed_source_sha256")
            or source != (fresh_campaign or {}).get("source_sha256")
            or not plan.get("model_pins") or plan["model_pins"] != fresh_plan.get("model_pins")):
        raise ValueError("separate training/fresh source or model pins differ")
    for owner in (directory, fresh_dir):
        package = owner / "source/pc_strands"
        if {p.name for p in package.glob("*.py")} != set(source):
            raise ValueError("separate campaign frozen source coverage differs")
        for name, expected in source.items():
            if evidence.record_file(package / name, "separate campaign frozen runtime") != expected:
                raise ValueError("separate campaign frozen source changed")
    steps = plan.get("steps", [])
    ids = [s.get("id") for s in steps]
    if (not ids or None in ids or len(set(ids)) != len(ids) or set(campaign.get("stages", {})) != set(ids)
            or set(ids) & set(fresh_ids)):
        raise ValueError("separate training stage coverage differs from its actual plan")
    for step in steps:
        stage = campaign["stages"][step["id"]]
        expected_argv = [x.replace("{source}", str(directory / "source")).replace("{run}", str(directory))
                         for x in step["argv"]]
        outputs = {evidence.path(o["path"]): o for o in step.get("outputs", [])}
        captured = {evidence.path(p): h for p, h in stage.get("output_sha256", {}).items()}
        if (stage.get("status") != "completed" or stage.get("exit_code") != 0
                or stage.get("argv") != expected_argv or not outputs or set(outputs) != set(captured)):
            raise ValueError("separate training completed stage/captured-output contract differs")
        for path, spec in outputs.items():
            if (evidence.record_file(path, "actual training owning-stage output") != captured[path]
                    or not condition(spec)):
                raise ValueError("separate training captured stage output changed")
    return plan, campaign, directory


def control_valid(evidence, paths, trained, baseline, campaign):
    """Receipt tensor identity permits honest safetensors reserialization."""
    if not trained or not baseline or any(paths.get(k) is None for k in ("released", "trained", "untuned")):
        return False
    proof = evidence.read(paths["untuned"] / "training-proof.json", "unchanged control tensor proof") or {}
    stage = (campaign or {}).get("stages", {}).get(paths.get("control_copy_stage"), {})
    captured = {evidence.path(path): value for path, value in stage.get("output_sha256", {}).items()}
    if stage.get("status") != "completed":
        return False
    for name in ("head.safetensors", "training-proof.json"):
        actual = evidence.record_file(paths["untuned"] / name, "pre-calibration control-copy output")
        if not actual or actual != captured.get(paths["untuned"] / name):
            return False
    original = evidence.record_file(paths["released"] / "head.safetensors", "released control source head")
    adapters = [evidence.record_file(paths[k] / "lora/adapter_model.safetensors", k + " frozen adapter identity")
                for k in ("released", "trained", "untuned")]
    return bool(proof.get("status") == "untuned_control_copy" and proof.get("weights_updated") is False
        and proof.get("head_tensor_sha256")
        and proof["head_tensor_sha256"] == trained.get("initial_head_tensor_sha256")
        and original and original == trained.get("original_head_file_sha256") == baseline.get("checkpoint_head_sha256")
        and all(value and value == trained.get("frozen_adapter_sha256") for value in adapters))


def calibration_valid(evidence, kind, excluded, paths):
    directory, fit_path = paths.get(kind), paths.get("calibration_" + kind)
    if directory is None or fit_path is None:
        return False
    fit = evidence.read(fit_path, f"{kind} temperature fit")
    proof = evidence.read(directory / "calibration-proof.json", f"{kind} checkpoint calibration proof")
    config = evidence.read(directory / "strands_decider_config.json", f"{kind} calibrated config")
    weight = evidence.record_file(directory / "head.safetensors", f"{kind} calibrated head weights")
    ids = (fit or {}).get("question_ids", [])
    return bool(fit and fit == proof and fit.get("status") == "fitted_on_calibration_only_not_test"
        and ids and len(set(ids)) == len(ids) and not set(ids) & set(excluded)
        and weight == fit.get("checkpoint_head_sha256") and config
        and config.get("temperature") == fit.get("temperature")
        and config.get("temperature_by_kind", {}).get("noul") == fit.get("temperature"))


def confirmation(evidence, n, as_of, campaign_dir, campaign, plan, paths, baseline, trained, *, control_campaign=None):
    """Verify cohort timing and independently recompute the canonical pure join."""
    from pc_strands.compare import compare_runs
    from pc_strands.data import load_tasks
    if not campaign_dir or not campaign or not plan:
        raise ValueError("positive fresh N requires an explicit frozen campaign and plan")
    freeze = evidence.read(paths["freeze"], "answer-blind fresh cohort freeze")
    if not freeze or (freeze.get("n") != n or freeze.get("answer_blind") is not True
            or freeze.get("fresh_scoring_started") is not False or freeze.get("basis") != "measured_throughput_and_resources"):
        raise ValueError("positive fresh N requires its explicit matching answer-blind pre-score freeze")
    plan_hash = digest(evidence.path(campaign_dir) / "plan.json")
    if freeze.get("plan_sha256") != plan_hash or campaign.get("plan_sha256") != plan_hash:
        raise ValueError("campaign/fresh freeze plan identity differs")
    copied = evidence.record_file(evidence.path(campaign_dir) / "fresh-freeze.json", "campaign copied pre-score freeze")
    operator_hash = digest(paths["freeze"])
    if copied != operator_hash or campaign.get("fresh_freeze_sha256") != copied or campaign.get("fresh_n") != n:
        raise ValueError("campaign did not adopt the exact operator fresh freeze")
    options = [o for o in plan.get("fresh_options", []) if o.get("n") == n]
    dataset = paths["dataset"]
    dataset_hash = evidence.record_file(dataset, "exact frozen confirmation dataset")
    if (len(options) != 1 or evidence.path(options[0]["dataset"]) != dataset
            or dataset_hash != freeze.get("dataset_sha256") or options[0].get("dataset_sha256", dataset_hash) != dataset_hash):
        raise ValueError("exact plan confirmation dataset differs from freeze")
    ids = [t.id for t in load_tasks(dataset)]
    if len(ids) != n:
        raise ValueError("actual fresh cohort differs from explicit N")
    source = paths["fresh_source"]
    generation = evidence.read(source / "manifest.json", "fresh generation start and cleanup")
    frozen_at = timestamp(freeze["frozen_at"])
    if frozen_at > timestamp(as_of) or frozen_at > timestamp(generation["started_at_utc"]):
        raise ValueError("fresh freeze must precede fresh generation and report checkpoint")
    raw = evidence.read(paths["fresh_analysis"] / "paired-analysis.json", "terminal paired confirmation analysis")
    if not raw:
        return None
    validate_comparison(raw, n)
    if raw["planned_question_ids"] != ids or raw["shared_protocol"].get("dataset_sha256") != dataset_hash:
        raise ValueError("comparison IDs/order or dataset differs from actual frozen cohort")
    if not raw.get("input_sha256"):
        raise ValueError("comparison input fingerprints absent")
    for path, expected in raw["input_sha256"].items():
        if evidence.record_file(path, "comparison-bound source input") != expected:
            raise ValueError("comparison input fingerprint changed or disappeared")
    retain = bool(raw.get("secondary_retain_best_policies"))
    planned_retain = any("--retain-best" in s.get("argv", []) for s in options[0].get("steps", []))
    if retain != planned_retain:
        raise ValueError("secondary analysis differs from frozen plan")
    actual = compare_runs(paths["fresh_untuned"], paths["fresh_trained"], source,
        paths["fresh_cache"], retain_best=retain)
    for key, value in actual.items():
        if key not in {"created_at_utc", "input_sha256"} and raw.get(key) != value:
            raise ValueError(f"published comparison differs from authoritative pure comparison: {key}")
    for path, expected in actual["input_sha256"].items():
        if evidence.record_file(path, "independently rejoined canonical input") != expected:
            raise ValueError("canonical comparison input changed")
    for kind in ("untuned", "trained"):
        name, directory = paths[kind + "_policy"], paths[kind]
        if raw.get("model_fingerprints", {}).get(name) != checkpoint_fingerprint(evidence, directory):
            raise ValueError("comparison does not bind current primary checkpoints")
    if not control_valid(evidence, paths, trained, baseline, control_campaign if control_campaign is not None else campaign):
        raise ValueError("unchanged control tensor/adapter/source-head linkage is not verified")
    excluded = ids + trained.get("fit_question_ids", []) + trained.get("tune_question_ids", [])
    if not all(calibration_valid(evidence, kind, excluded, paths) for kind in ("untuned", "trained")):
        raise ValueError("both current primary heads require disjoint hash-bound calibration")
    fits = [evidence.read(paths["calibration_" + k], k + " shared calibration protocol") for k in ("untuned", "trained")]
    if (fits[0].get("question_ids") != fits[1].get("question_ids")
            or any(f.get("bounds") != [.25, 6.] or f.get("objective") != "calibration-question NLL"
                   or any(f.get(phase, {}).get("threshold") != .8 for phase in ("before", "after")) for f in fits)):
        raise ValueError("untouched/trained calibration IDs or predeclared protocol differ")
    return raw


def section(title, paragraphs=(), headers=(), rows=()):
    return {"title": title, "paragraphs": list(paragraphs), "headers": list(headers), "rows": list(rows)}


def development_auditor(evidence):
    """Load only the exact reviewed, read-only stdlib audit, never receipt code."""
    helper = evidence.root / "operator_dev20_diagnostics.py"
    for path, expected in ((helper, DEVELOPMENT_AUDITOR_SHA),
                           (helper.with_name("operatorqualification.py"), DEVELOPMENT_EVIDENCE_SHA)):
        if evidence.record_file(path, "pinned read-only DEVELOPMENT auditor") != expected:
            raise ValueError("reviewed DEVELOPMENT auditor is missing or changed")
    spec = importlib.util.spec_from_file_location("report_development20_audit", helper)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        raw = helper.read_bytes()
        if hashlib.sha256(raw).hexdigest() != DEVELOPMENT_AUDITOR_SHA:
            raise ValueError("reviewed DEVELOPMENT auditor changed before loading")
        # Execute the pinned bytes, never a potentially stale cached .pyc.
        exec(compile(raw, str(helper), "exec"), module.__dict__)
    finally:
        sys.dont_write_bytecode = previous
    return module


def development_diagnostics(evidence, receipt_path, paths, owner_dir, as_of):
    """Re-audit explicit secondary evidence without changing primary gates."""
    receipt = evidence.read(receipt_path, "explicit secondary DEVELOPMENT20 diagnostics")
    if not isinstance(receipt, dict) or not receipt:
        raise ValueError("explicit DEVELOPMENT diagnostic receipt is missing")
    # Strict types prevent JSON true from masquerading as an integer denominator.
    if (receipt.get("status") != "verified_secondary_development20_posttraining_diagnostics"
            or receipt.get("phase") != "development"
            or type(receipt.get("planned_n")) is not int or receipt["planned_n"] != 20
            or type(receipt.get("max_drafts")) is not int or receipt["max_drafts"] != 1
            or len(receipt.get("task_ids", [])) != 20 or len(set(receipt["task_ids"])) != 20
            or receipt.get("analysis_helper_sha256") != DEVELOPMENT_AUDITOR_SHA
            or receipt.get("all20_original_development_questions_verified") is not True
            or receipt.get("no_fresh_confirmation") is not True
            or receipt.get("no_further_weight_or_temperature_selection") is not True
            or receipt.get("held_out_improvement_claim") is not False):
        raise ValueError("secondary evidence must be all 20 DEVELOPMENT questions at cap1")
    if timestamp(receipt["audited_at_utc"]) > timestamp(as_of):
        raise ValueError("DEVELOPMENT audit postdates this report checkpoint")
    roles = receipt["artifact_roles"]
    expected = {"original_baseline_directory": paths["baseline"].parent,
                "untuned_checkpoint_directory": paths["untuned"],
                "trained_checkpoint_directory": paths["trained"]}
    if not owner_dir or any(value is None or evidence.path(roles[key]) != value
                            for key, value in expected.items()):
        raise ValueError("DEVELOPMENT baseline/checkpoints differ from report training owner")
    qualification = evidence.read(roles["matched_training_control_CAL_qualification"],
                                  "DEVELOPMENT matched training/control/CAL owner")
    owner = evidence.path(owner_dir)
    if (not isinstance(qualification, dict)
            or qualification.get("actual_training_campaign_plan_sha256") != evidence.record_file(
                owner / "plan.json", "DEVELOPMENT actual training plan")
            or qualification.get("actual_training_campaign_sha256") != evidence.record_file(
                owner / "campaign.json", "DEVELOPMENT actual training terminal state")):
        raise ValueError("DEVELOPMENT qualification belongs to another training campaign")
    hashes = receipt.get("input_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("DEVELOPMENT input fingerprints absent")
    for path, expected_hash in hashes.items():
        if evidence.record_file(path, "secondary DEVELOPMENT bound input") != expected_hash:
            raise ValueError("DEVELOPMENT input fingerprint changed or disappeared")
    auditor = development_auditor(evidence)
    actual = auditor.analyze(evidence.path(roles["plan"]), evidence.path(roles["campaign"]).parent)
    # JSON normalizes integer mapping keys just as the persisted receipt does.
    actual = json.loads(json.dumps(actual, allow_nan=False))
    if {k:v for k,v in receipt.items() if k != "audited_at_utc"} != {
            k:v for k,v in actual.items() if k != "audited_at_utc"}:
        raise ValueError("DEVELOPMENT receipt differs from the authoritative read-only re-audit")
    # Bind all newly rejoined source evidence too, then freeze these bytes at report end.
    for path, expected_hash in actual["input_files_sha256"].items():
        if evidence.record_file(path, "independently rejoined DEVELOPMENT input") != expected_hash:
            raise ValueError("DEVELOPMENT evidence changed during re-audit")
    return actual


def development_sections(result):
    """Keep reused-cohort diagnostics visually separate from fresh confirmation."""
    metric = lambda value: "undefined" if value is None else number(value, 6)
    rows = []
    for kind in ("untuned", "trained"):
        arm = result["per_policy"][kind]; p = arm["first_draft_diagnostics"]
        rows.append([kind + " calibrated", f"{p['scored_first_drafts']}/20",
            metric(p["auroc"]), metric(p["brier"]), metric(p["nll"]),
            metric(p["first_draft_approval_precision"]), metric(p["approval_coverage_of_planned"])])
    quality = result["source_writer_answer_quality"]
    sections = [section("Secondary post-training DEVELOPMENT probability diagnostics", [
        "The same 20 development questions were inspected before training. These cap1 results describe probabilities on saved first drafts; they establish neither repair benefit, answer improvement nor fresh generalization. Released and trained heads each use their separately fitted CAL temperature; weights and temperatures are not selected again here.",
        f"The shared saved writer answers have EM {100*quality['em']:.2f}% and token F1 {quality['token_f1']:.6f}. A head failure can reduce its failure-inclusive episode EM, but cannot improve these saved answers. Missing/overflow probabilities remain missing; all 20 stay in coverage denominators. Undefined AUROC or precision is not zero.",
    ], ["Calibrated head", "Scored/N", "AUROC", "Brier", "NLL", "Precision", "Approval/N"], rows)]
    paired = result["paired_deltas"]; rows = []
    for name in ("nll", "brier", "accuracy", "approval_coverage_of_planned"):
        value = paired[name + "_trained_minus_untuned"]
        rows.append(["Decision classification accuracy" if name == "accuracy" else name,
            metric(value["candidate_minus_baseline"]) if value else "undefined",
            "[" + ", ".join(metric(x) for x in value["bootstrap95"]) + "]" if value else "undefined",
            str(value["planned_n"]) if value else "0", "5,000 / 0" if value else "not drawn"])
    for name in ("auroc", "approval_precision"):
        value = paired[name + "_paired_bootstrap"]
        a = result["per_policy"]["trained"]["first_draft_diagnostics"]
        b = result["per_policy"]["untuned"]["first_draft_diagnostics"]
        key = "auroc" if name == "auroc" else "first_draft_approval_precision"
        delta = a[key] - b[key] if a[key] is not None and b[key] is not None else None
        interval = value["bootstrap95"]
        rows.append([name, metric(delta), "[" + ", ".join(metric(x) for x in interval) + "]" if interval else "undefined",
            str(paired["paired_scored_n"]), f"{value['valid_draws']} / {value['undefined_draws']}"])
    sections.append(section("Secondary DEVELOPMENT paired probability differences", [
        f"Trained minus released calibrated head; 5,000 paired draws, seed 42, original development question as unit. Probability differences use the {paired['paired_scored_n']}/20 observed pairs; {paired['missing_probability_pairs']} absent pairs are not imputed. Approval coverage uses all 20, with absent decisions counted as no approval. Decision classification accuracy concerns correctness prediction, not answer EM.",
        "AUROC/precision point differences use each arm's observed scores; their paired intervals use only the observed-pair intersection. Nonadditive intervals exclude and count draws undefined in either arm. Additive intervals, when defined, use all 5,000 draws. These small, reused-cohort intervals are descriptive.",
    ], ["Statistic", "Point delta", "Paired95% interval", "Pair N", "Valid/undefined draws"], rows))
    return sections


def selected_train_description(evidence, plan, campaign, owner, manifest, tasks):
    """Describe the selected source; ancestor proposals are a separate cohort."""
    counts = Counter(t.get("partition", "unknown") for t in tasks)
    description = (f"Selected TRAIN for this report: {len(tasks)} planned questions, "
        f"FIT {counts['fit']} / TUNE {counts['tune']} / CALIBRATION {counts['calibration']}. "
        "This source's generation and native features are separate from historical preparation expense.")
    source = (plan or {}).get("training_source", {})
    if not source or not owner:
        return description
    if (not isinstance(source, dict) or not isinstance(campaign, dict) or not isinstance(manifest, dict)
            or not isinstance(manifest.get("frozen"), dict) or not isinstance(manifest["frozen"].get("config"), dict)):
        raise ValueError("selected TRAIN owner/source state is missing or malformed")
    selected = [s for specs in plan.get("prerequisites", {}).values()
        for s in (specs if isinstance(specs, list) else [specs])
        if s.get("sha256") == source.get("manifest_sha256")]
    if len(selected) != 1:
        raise ValueError("selected TRAIN manifest is not uniquely pinned by its owner")
    selection = evidence.read(selected[0]["path"], "selected TRAIN blind original-partition sampling")
    if not isinstance(selection, dict) or not isinstance(selection.get("selection"), dict):
        raise ValueError("selected TRAIN sampling manifest is missing or malformed")
    if (campaign.get("status") != "completed" or campaign.get("plan_sha256") != evidence.record_file(
            evidence.path(owner) / "plan.json", "selected TRAIN adopted plan")
            or evidence.record_file(selected[0]["path"], "selected TRAIN sampling identity") != source["manifest_sha256"]
            or source.get("dataset_sha256") != manifest.get("frozen", {}).get("config", {}).get("dataset_sha256")
            or source.get("planned_n") != len(tasks)
            or selection.get("count") != len(tasks) or selection.get("ordered_task_ids") != [t["id"] for t in tasks]
            or selection.get("partition_counts") != dict(counts)
            or selection.get("selection", {}).get("uses_only") != ["canonical_question_id", "local_partition", "source_row.type"]
            or selection["selection"].get("no_repartitioning") is not True):
        raise ValueError("selected TRAIN source/sampling/adoption identities differ")
    return description + (" Its answer-blind reduction ranked canonical IDs by a seed42 hash within original "
        "partition and public question type, retained original order/partitions, and excluded answers, outcomes, "
        "draft availability and context length from selection. The preserved selection manifest still says "
        "PROPOSED; actual execution is established separately by the completed owning campaign.")


def captured_phase_usage(evidence, plan, campaign, owner, step, expected_ids, *, head=False):
    """Reuse captured-stage audit and the existing pure physical-usage reducer."""
    from types import SimpleNamespace
    from pc_strands.evaluate import physical_usage
    if not isinstance(plan, dict) or not isinstance(campaign, dict):
        raise ValueError("physical-phase owning plan/state is missing or malformed")
    q = development_auditor(evidence).q
    owner = evidence.path(owner); bound = q.Evidence()
    if (campaign.get("status") != "completed" or campaign.get("plan_sha256") != evidence.record_file(
            owner / "plan.json", "physical-phase owning plan")):
        raise ValueError("physical-phase campaign is not its completed frozen owner")
    stage = q.stage(bound, campaign, step, owner / "source", owner)
    for path, expected in bound.files.items():
        if evidence.record_file(path, "captured physical-phase input") != expected:
            raise ValueError("physical-phase captured output changed")
    directory = evidence.path(q.argument(step, "--output"))
    captured = {evidence.path(p) for p in stage.get("output_sha256", {})}
    if not {directory / name for name in ("manifest.json", "requests.jsonl", "tasks-evaluator-only.json")} <= captured:
        raise ValueError("physical-phase owning capture omits manifest/journal/tasks")
    manifest = evidence.read(directory / "manifest.json", "physical-phase manifest")
    events = evidence.read(directory / "requests.jsonl", "physical-phase raw journal", lines=True)
    tasks = evidence.read(directory / "tasks-evaluator-only.json", "physical-phase planned tasks")
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("frozen"), dict)
            or not isinstance(manifest["frozen"].get("config"), dict)
            or not isinstance(tasks, list) or any(not isinstance(t, dict) for t in tasks)):
        raise ValueError("physical-phase manifest/tasks are missing or malformed")
    counts = journal_counts(events); cfg = manifest["frozen"]["config"]
    if (manifest["status"] not in TERMINAL or manifest["frozen"]["task_ids"] != expected_ids
            or [t["id"] for t in tasks] != expected_ids
            or any(t.get("partition") != ("calibration" if head else "dev") for t in tasks)
            or cfg.get("max_drafts") != 1 or cfg.get("mode") != ("head-replay" if head else "generate")
            or cfg.get("phase") != "development"
            or counts["pending"] or counts["errors"]):
        raise ValueError("physical-phase protocol/coverage/journal differs")
    policy = cfg.get("policy_name") if head else "fixed"
    wanted = {f"{qid}|{policy}|{'decision' if head else 'writer'}|1" for qid in expected_ids}
    if {e["request_id"] for e in events} != wanted:
        raise ValueError("physical-phase requests differ from planned first drafts")
    completed = {e["request_id"]: e for e in events if e["event"] == "request_completed"}
    for event in completed.values():
        result = event["result"]
        if not isinstance(result, dict):
            raise ValueError("physical-phase raw result is malformed")
        values = (result.get("input_tokens"), result.get("physical_input_tokens", result.get("input_tokens")),
            result.get("output_tokens", 0), result.get("cost_usd", 0),
            event.get("request_elapsed_seconds", result.get("elapsed_seconds")))
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("physical-phase request usage is missing or invalid")
    actual = physical_usage(SimpleNamespace(completed=completed, pending=set()))
    usage = manifest["actual_evaluation_usage"]
    if not isinstance(usage, dict):
        raise ValueError("physical-phase usage is malformed")
    for key, value in actual.items():
        if key == "note": continue
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or type(usage.get(key)) not in (int, float) or not math.isclose(usage[key], value, abs_tol=1e-9)):
            raise ValueError("physical-phase manifest/raw usage differs")
    if actual["requests_with_unknown_usage"] or (head and (cfg.get("decision_device") != "cpu"
            or cfg.get("head_replay_no_torso_loaded") is not True or actual["reported_physical_input_tokens"] != 0
            or any(e["result"].get("reused_encoding") is not True or e["result"].get("physical_input_tokens") != 0 for e in completed.values()))):
        raise ValueError("physical-phase usage is unknown or CPU cache reuse differs")
    return directory, actual


def collect_report(input_root, train_run, fresh_n, as_of, *, cohort_manifest=None, campaign_run=None,
                   train_segments=(), historical_train_run=None, historical_train_segments=(),
                   training_campaign_run=None, development_diagnostic=None):
    if fresh_n not in (0, 20, 50, 100):
        raise ValueError("explicit fresh N must be0,20,50 or100")
    as_of = timestamp(as_of).isoformat()
    evidence = Evidence(input_root)
    campaign = evidence.read(evidence.path(campaign_run) / "campaign.json", "explicit campaign state") if campaign_run else None
    plan = evidence.read(evidence.path(campaign_run) / "plan.json", "explicit frozen campaign plan") if campaign_run else None
    if campaign_run and plan is None:
        raise ValueError("explicit campaign plan is missing; canonical artifacts cannot supply its roles")
    owner_plan, owner_campaign, owner_dir = plan, campaign, campaign_run
    if training_campaign_run:
        if not campaign_run or not plan:
            raise ValueError("separate training owner requires an explicit fresh campaign/plan")
        owner_plan, owner_campaign, owner_dir = external_training_owner(
            evidence, training_campaign_run, campaign_run, plan, campaign)
    paths = artifact_paths(evidence, owner_plan, 0, owner_dir)
    training_roles = paths.copy() if training_campaign_run else None
    train = evidence.path(train_run)
    if paths["train_source"] is not None and train != paths["train_source"]:
        raise ValueError("explicit TRAIN source differs from reviewed campaign cache source")
    current = train_source_evidence(evidence, train, train_segments)
    manifest, train_control, trajectories = current["manifest"], current["control"], current["trajectories"]
    journal, partitions, planned = current["journal"], current["partitions"], current["planned"]
    reconciliation, timings = current["reconciliation"], current["timings"]
    if historical_train_segments and not historical_train_run:
        raise ValueError("historical TRAIN segments require an explicit historical source")
    historical_path = evidence.path(historical_train_run) if historical_train_run else None
    if historical_path == train:
        raise ValueError("historical TRAIN source duplicates the current physical source")
    historical = (train_source_evidence(evidence, historical_path, historical_train_segments, "Historical TRAIN")
                  if historical_path else None)
    if historical and (current["physical_dispatches"] & historical["physical_dispatches"]
            or (current["journal"]["started"] and current["journal_sha256"] == historical["journal_sha256"])):
        raise ValueError("historical TRAIN evidence repeats current physical dispatches")
    baseline = evidence.read(paths["baseline"], "untouched real-forward gate") if paths["baseline"] else None
    proof = evidence.read(paths["trained"] / "training-proof.json", "actual trained-head proof") if paths["trained"] else None
    gate, trained = baseline_valid(baseline), training_valid(proof)
    warnings, validation_errors, baseline_result = [], [], None
    if reconciliation["completed_calls_without_saved_round"]:
        warnings.append(f"{reconciliation['completed_calls_without_saved_round']} completed TRAIN provider call(s) have no saved trajectory round. Their physical work is charged; they are not invented scored drafts.")
    if historical:
        saved_missing = historical["reconciliation"]["completed_calls_without_saved_round"]
        warnings.append(f"Historical TRAIN preparation is a separate physical source: {historical['journal']['completed']} completed calls; {saved_missing} completed call(s) without a saved round. Its questions, caches and denominator are not pooled with the selected TRAIN source.")
    if baseline:
        released_hash = evidence.record_file(paths["released"] / "head.safetensors", "released untouched head weights") if paths["released"] else None
        if released_hash != baseline.get("checkpoint_head_sha256"):
            gate = False
            warnings.append("Untouched gate does not bind the actual released head.")
        for name, key in (("manifest.json", "manifest_sha256"), ("episodes.json", "episodes_sha256")):
            path = paths["baseline"].parent / name
            evidence.read(path, "untouched receipt bound result")
            if not path.is_file() or digest(path) != baseline.get(key):
                gate = False
                warnings.append("Untouched gate result hash is missing or inconsistent.")
        if gate:
            try:
                baseline_result = baseline_diagnostics(evidence, paths["baseline"], baseline, owner_plan, owner_campaign, owner_dir)
            except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
                gate = False
                validation_errors.append({"stage": "untouched development baseline", "type": type(error).__name__, "detail": str(error)})
                warnings.append("Untouched development diagnostics and native expense omitted: bound baseline evidence is incomplete or inconsistent.")
    if proof and baseline:
        if proof.get("original_head_file_sha256") != baseline.get("checkpoint_head_sha256"):
            trained = False
            warnings.append("Training proof and untouched released-head identity differ.")
        baseline_path = paths["baseline"]
        if proof.get("baseline_receipt_sha256") != digest(baseline_path):
            trained = False
            warnings.append("Training proof does not bind the supplied untouched gate.")
        if set(baseline.get("planned_task_ids", [])) & (
                set(proof.get("fit_question_ids", [])) | set(proof.get("tune_question_ids", []))):
            trained = False
            warnings.append("Untouched development IDs overlap the training/tuning proof.")
    if proof:
        head_hash = evidence.record_file(paths["trained"] / "head.safetensors", "saved trained pointer weights")
        if not head_hash or head_hash != proof.get("checkpoint_head_sha256"):
            trained = False
            warnings.append("Actual trained-head file is missing or differs from training proof.")
        adapter = evidence.record_file(paths["released"] / "lora/adapter_model.safetensors", "released frozen adapter") if paths["released"] else None
        if not adapter or adapter != proof.get("frozen_adapter_sha256"):
            trained = False
            warnings.append("Training proof does not bind the actual released adapter.")
        for partition in ("fit", "tune"):
            cache_path = paths["cache_" + partition]
            if cache_path is None:
                trained = False
                warnings.append(f"Training proof {partition} cache is not declared by the reviewed plan.")
                continue
            actual_hash = evidence.record_file(cache_path, f"{partition} training cache")
            manifest_hash = evidence.record_file(str(cache_path)+".manifest.json", f"{partition} training cache manifest")
            if (not actual_hash or actual_hash != proof.get(f"{partition}_cache_sha256")
                    or manifest_hash != proof.get("feature_manifest_sha256", {}).get(partition)):
                trained = False
                warnings.append(f"Training proof {partition} cache fingerprints are not verified.")
    subset = evidence.read(cohort_manifest, "explicit selected TRAIN cohort") if cohort_manifest else None
    if subset and "PROPOSED" in subset.get("status", ""):
        warnings.append("Supplied TRAIN subset remains proposed; it is not an adopted training result.")
    control = control_valid(evidence, paths, proof, baseline, owner_campaign)
    development = None
    if development_diagnostic:
        try:
            if not (gate and trained and control):
                raise ValueError("secondary diagnostics require this report's verified baseline/trained/control proofs")
            development = development_diagnostics(evidence, development_diagnostic, paths, owner_dir, as_of)
        except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
            validation_errors.append({"stage": "secondary development diagnostics", "type": type(error).__name__, "detail": str(error)})
            warnings.append("Secondary post-training DEVELOPMENT diagnostics omitted: the explicit receipt or its bound evidence is missing/inconsistent. Primary confirmation gates are unchanged.")
    paired = None
    if fresh_n:
        if not campaign_run:
            raise ValueError("positive fresh N requires an explicit frozen campaign and plan")
        try:
            paths = artifact_paths(evidence, plan, fresh_n, campaign_run, training_roles=training_roles)
            paired = confirmation(evidence, fresh_n, as_of, campaign_run, campaign, plan, paths, baseline, proof or {},
                control_campaign=owner_campaign)
            if paired and not (gate and trained):
                raise ValueError("untouched gate and trained-head proof must both be verified")
        except (ValueError, KeyError, TypeError, OSError) as error:
            validation_errors.append({"stage": "paired confirmation", "type": type(error).__name__, "detail": str(error)})
            warnings.append(f"Paired quality evidence omitted ({type(error).__name__}); inspect private provenance and validation inputs.")
            paired = None
    complete = bool(paired and gate and trained)
    status = "TERMINAL PAIRED REPLAY CHECKPOINT" if complete else "DRAFT / PARTIAL CHECKPOINT"
    sections = []
    coverage = [
        ["Selected TRAIN source", str(planned or "not recorded"), str(len(trajectories)),
         f"{manifest.get('status', 'not recorded')}; partitions {dict(partitions)}; saved trajectory failures {reconciliation['saved_trajectory_failures']}"],
        ["TRAIN physical requests", str(journal["started"]), str(journal["completed"]), f"pending {journal['pending']}; error events {journal['errors']}"],
        ["TRAIN missing trajectories", str(planned), str(reconciliation["missing_trajectories"]),
         f"{reconciliation['completed_questions_without_saved_trajectory']} completed-call question(s) lack a saved trajectory; {reconciliation['undispatched_questions']} questions never dispatched"],
        ["Untouched Strands baseline", "20 before learning", str((baseline or {}).get("attempted", 0)),
         "gate verified" if gate else "not verified; no baseline accuracy claim"],
        ["Pointer-head training", "disjoint fit/tune", str((proof or {}).get("fit_questions", 0)),
         "weight/reload proof verified" if trained else "not verified; no trained-model claim"],
        ["Fresh paired confirmation", str(fresh_n or "not supplied"), str(paired["planned_n"] if paired else 0),
         "terminal matched replay" if paired else "no validated paired result supplied"],
    ]
    sections.append(section("1. Evidence and completion", [
        "This report separates completed computation, incomplete coverage and proposed experiments. It does not claim that elapsed monitoring time establishes a completed six-hour experiment.",
        f"TRAIN cleanup: writer_unload_verified={manifest.get('writer_unload_verified', 'not recorded')}. "
        "An interrupted source retains its original planned denominator; neither omitted questions nor a derived subset can silently relabel it complete.",
        f"Last recorded TRAIN control state: {train_control.get('state', 'not recorded')}; "
        f"reasons: {', '.join(train_control.get('reasons', [])) or 'none recorded'}. "
        "A resource hold preserves completed requests rather than repeating them.",
        f"Explicit campaign state: {(campaign or {}).get('status', 'not supplied')}. Launching a supervisor is not a model forward.",
        *( ["A separate completed training campaign owns TRAIN caches, checkpoint updates and calibration. Its actual stage captures are verified directly; fresh evaluation does not import or recreate those stages."] if training_campaign_run else [] ),
    ], ["Evidence", "Declared", "Observed", "Interpretation"], coverage))
    config = manifest.get("frozen", {}).get("config", {})
    writer, settings = config.get("writer", {}), config.get("writer_settings", {})
    tasks = evidence.read(train / "tasks-evaluator-only.json", "selected TRAIN planned partitions") or []
    try:
        selected_description = selected_train_description(evidence, owner_plan, owner_campaign, owner_dir, manifest, tasks)
    except (ValueError, KeyError, TypeError, OSError) as error:
        selected_description = f"Selected TRAIN source: {planned} planned questions; saved partitions {dict(partitions)}. Original-partition selection provenance was not verified."
        warnings.append("Selected TRAIN sampling description omitted: owner-bound evidence differs.")
    dataset_metadata = evidence.read(evidence.root.parent / "research-2026-10-18-strands/official-train600-candidate/manifest.json", "official TRAIN selection and exclusion audit") or {}
    evidence.record_file(evidence.root.parent / "preflight-2026-10-18/PROFESSOR-METHODS-DRAFT.txt", "historical methods and HotpotQA attribution")
    sections.append(section("2. Dataset and architecture", [
        "The task is short-answer HotpotQA with all supplied distractor paragraphs, titles and sentence order. Questions, evidence and the current draft are public model inputs; references, supporting facts and correctness labels are evaluator-only. Answer metrics use official-style normalized exact match and token F1.",
        "Official TRAIN and public validation must be disjoint by ID and normalized question. The original 600-question training proposal uses 400 fit / 100 tune / 100 calibration cases; its six fixed 100-row source blocks are not an IID sample of all 90,447 TRAIN questions. A resource-driven TRAIN reduction requires a separately adopted manifest before learning, with original partitions and parent provenance preserved.",
        selected_description,
        f"Dataset attribution: HotpotQA (https://hotpotqa.github.io/), CC-BY-SA 4.0; selected source revision {dataset_metadata.get('source_revision', 'not supplied')}. Preserve source attribution and the recorded ID/question exclusion audit when distributing derived data.",
        f"Selected training dataset SHA256: {config.get('dataset_sha256', 'not supplied')}. "
        f"Explicit cohort manifest status: {(subset or {}).get('status', 'none supplied')}; "
        f"recorded partition counts: {(subset or {}).get('partition_counts', dict(partitions))}.",
        "The proposed Strands controller scores binary draft correctness and stops at probability >=0.8 or continues a generic revision loop. Strands accepts at most 4,096 fully rendered tokens, without truncation; overflow remains an explicit coverage failure. The writer has a separate 16k context. Ten total drafts and cap-three parity share recorded trajectories. The released 2B encoder and adapter remain frozen; actual supervised fine-tuning updates the released pointer head only. One EM correctness proxy is not three independently labeled rubric targets. LoRA/backbone training and RL are separate future work.",
        "Confirmation is a once-frozen subset of 20, 50 or 100 public validation questions, not an official hidden-test score. Shared Fixed 1/2/3/5/10 controls use the identical observations. The unchanged Vedanshi v0 export is a schema contract, not proof of identical hardware; canonical planned-N summaries retain failures/zero-draft cases that v0 cannot express.",
        "The shared comparison measures all-planned-question answer EM and token F1, first-draft repairs and regressions, approval precision with coverage, and first-decision AUROC, Brier score and NLL. Question-paired bootstrap intervals use 5,000 draws and seed42; counterfactual prefix costs and actual physical work are separate. These are the comparison methods, not additional measured results in a partial checkpoint.",
    ]))
    install = evidence.read("install-status.json", "actual installation state") or {}
    health = evidence.read("health.json", "actual import/CUDA health") or {}
    packages = evidence.read("installed-packages.json", "actual package versions") or []
    versions = {r["name"].lower(): r["version"] for r in packages if r.get("name", "").lower() in {"torch", "transformers", "peft", "accelerate", "numpy"}}
    targets = evidence.read("dependency-lock.json", "pinned wheel targets") or []
    target_versions = {r["name"].lower(): r["version"] for r in targets if r.get("name", "").lower() in {"torch", "transformers", "peft", "accelerate", "numpy"}}
    queue = evidence.read("download-queue.json", "public pinned download queue") or {}
    wheels = [r for r in queue.get("files", []) if r["path"].startswith("wheels/")]
    for row in wheels:
        if Path(row["path"]).name.startswith("torch-"):
            target_versions["torch"] = Path(row["path"]).name.split("-")[1]
    receipts = evidence.read("download-receipts.json", "download verification receipts") or {}
    verified = sum(receipts.get(r["path"], {}).get("sha256") == r["sha256"] for r in wheels)
    hardware = evidence.read(evidence.root.parent.parent / "pc-evidence/hardware.json", "dated hardware inventory") or {}
    gpu = str(hardware.get("gpu", "not supplied")).split(",")[:2]
    environment_rows = [
        ["Python / platform", f"{manifest.get('python', 'not recorded')} / {manifest.get('platform', 'not recorded')}"],
        ["Dated hardware inventory", f"{hardware.get('collected_at', 'not supplied')}; {hardware.get('cpu', {}).get('Name', '').strip()}; {', '.join(gpu)}"],
        ["Installed physical RAM capacity", number(hardware.get("memory", {}).get("TotalPhysicalMemory") / 1024**3, 2) + " GiB" if hardware.get("memory", {}).get("TotalPhysicalMemory") else "not recorded"],
        ["Writer identity", f"{writer.get('name', 'not recorded')} / digest {writer.get('digest', 'not recorded')}"],
        ["Writer settings", json.dumps(settings, sort_keys=True)],
        ["Serving profile", json.dumps(writer.get("serving_profile", {}), sort_keys=True)],
        ["Resource admission", json.dumps(config.get("resource_profile", {}), sort_keys=True)],
        ["Installation / health", f"{install.get('state', 'not supplied')}; CUDA health {'present' if health else 'absent'}"],
        ["Actually installed packages", json.dumps(versions, sort_keys=True) if versions else "No installed-package receipt supplied"],
        ["Download receipts / target packages", f"{verified}/{len(wheels)} pinned wheels have matching verification receipts; targets {json.dumps(target_versions, sort_keys=True)}"],
    ]
    sections.append(section("3. Environment and model identity", [
        "The profile preserves one study GPU owner, four CPU threads, background rests and memory admission. A resource pause does not authorize lowering the floor or closing user applications. Package targets are not installation or kernel evidence.",
        "Pinned target: upstream Strands source at 930e6d5de16309be32579f5631a7369a3322140c; released decider at 216d1ccb46a9e9b036eb9187b248db98480f77d4; Qwen3.5-2B-Base revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c. These are intended source identities until actual weight/inference receipts establish use. New 16k/q8 results must not be pooled with historical 32k/f16 development results.",
    ], ["Item", "Recorded evidence"], environment_rows))
    proof_rows = [["Untouched gate", "verified" if gate else "missing or invalid"],
        ["Attempted / successful / failed / valid forwards", " / ".join(str((baseline or {}).get(k, "not recorded")) for k in ("attempted", "completed", "failed", "valid_decision_forwards"))],
        ["Training status / actual head parameters", f"{(proof or {}).get('status', 'not recorded')} / {(proof or {}).get('head_parameters', 'not recorded')}"],
        ["Fit questions / tune questions / rows", f"{(proof or {}).get('fit_questions', 'not recorded')} / {(proof or {}).get('tune_questions', 'not recorded')} / {(proof or {}).get('fit_rows', 'not recorded')}+{(proof or {}).get('tune_rows', 'not recorded')}"],
        ["Head delta L2 / reload error", f"{number((proof or {}).get('head_delta_l2'), 8)} / {number((proof or {}).get('reload_max_logit_error'), 8)}"],
        ["Selection / elapsed seconds", f"{(proof or {}).get('selection', 'not recorded')} / {number((proof or {}).get('elapsed_seconds'))}"],
        ["Total candidate-fit epochs / optimizer steps", f"{len((proof or {}).get('history', []))} / {sum(r.get('optimizer_steps', 0) for r in (proof or {}).get('history', []))}" if (proof or {}).get('history') else "not recorded"],
        ["Selected saved checkpoint LR / epoch", f"{(proof or {}).get('selected', {}).get('lr', 'not recorded')} / {(proof or {}).get('selected', {}).get('epoch', 'not recorded')}"],
        ["Source head / adapter unchanged", f"{(proof or {}).get('source_head_unchanged', 'not recorded')} / {(proof or {}).get('source_adapter_unchanged', 'not recorded')}"],
        ["Untouched control tensor / adapter linkage", "verified" if control else "not verified"],
        ["Released head SHA256", str((baseline or {}).get("checkpoint_head_sha256", "not recorded"))],
        ["Saved trained head SHA256", str((proof or {}).get("checkpoint_head_sha256", "not recorded"))],
    ]
    calibration_rows = []
    for kind in ("untuned", "trained"):
        fit_path = paths.get("calibration_" + kind)
        fit = evidence.read(fit_path, f"{kind} temperature fit") if fit_path else None
        if fit:
            excluded = set((proof or {}).get("fit_question_ids", [])) | set((proof or {}).get("tune_question_ids", []))
            if calibration_valid(evidence, kind, excluded, paths):
                calibration_rows.append([kind, str(len(fit.get("question_ids", []))), number(fit.get("temperature")),
                    number(fit.get("before", {}).get("nll")), number(fit.get("after", {}).get("nll"))])
            else:
                warnings.append(f"{kind} calibration table omitted: current checkpoint/calibration linkage not verified.")
    sections.append(section("4. Untouched baseline, training and calibration", [
        "The untouched real-model 20-question gate must precede weight updates and bind final manifest/episode hashes. Successful native inference is distinct from mocked tests or CPU head replay. Training proof requires changed tensor hashes, positive weight delta, unchanged released source/adapter and matching reload logits.",
        "Training selects by disjoint tuning-question NLL; separate calibration questions fit each head's temperature using the same predeclared procedure. Neither phase may read fresh outcomes. Better training loss, successful reload and a changed head are not held-out improvement.",
        "Optimizer steps and candidate epochs describe total candidate-fit work across learning-rate trials. The selected saved checkpoint contains its selected trial/epoch prefix; it need not accumulate every candidate's updates.",
    ], ["Proof", "Observed"], proof_rows))
    if baseline_result:
        sections.append(section("Untouched DEVELOPMENT first-draft results", [
            "Recomputed from receipt-bound episodes and evaluator tasks, with raw critic responses/public inputs reconciled against the hash-bound native journal. The standalone summary is not used. All original planned questions remain in the denominator; failed episodes score zero.",
            "This is the pre-training cap-one development gate. Fixed1 and completed Strands episodes return the same first answer; there is no revision opportunity, fresh comparison or training gain in these numbers.",
            f"Approvals: {baseline_result['approvals']}; EM-correct approvals: {baseline_result['correct_approvals']}; "
            f"EM-wrong approvals: {baseline_result['wrong_approvals']}; rejections of EM-correct drafts: {baseline_result['rejections_of_correct_drafts']}. "
            "Approval correctness is conditional on approval, not all-question answer accuracy.",
        ], ["Policy", "Complete/N", "Answer EM", "Token F1", "Scope"], [[
            name, f"{row['completed']}/{row['planned_n']}", number(100 * row["em"], 2) + "%",
            number(row["token_f1"], 6), "DEVELOPMENT / cap1"] for name, row in baseline_result["policies"].items()]))
    if calibration_rows:
        sections.append(section("Calibration evidence", ["Calibration NLL is measured on the fitting data, not confirmation performance."],
            ["Head", "Questions", "Temperature", "NLL T1", "NLL fitted"], calibration_rows))
    if development:
        sections.extend(development_sections(development))
    if paired:
        quality = []
        probability = []
        transitions = []
        for name, row in paired["policies"].items():
            quality.append([name, f"{row['completed']}/{fresh_n}", number(100*row["em"], 2),
                number(row["token_f1"], 4), number(row.get("mean_actual_drafts")),
                number(row.get("mean_raw_reported_tokens"), 1), number(row.get("mean_counterfactual_prefix_model_seconds"), 4)])
            p = row.get("first_draft_probabilities", {})
            if name.startswith("strands-"):
                transitions.append([name, str(row.get("repairs_completed", "not recorded")),
                    str(row.get("regressions_completed", "not recorded")),
                    str(row.get("correct_first_draft_lost_including_failures", "not recorded")),
                    str(row.get("em_wrong_approvals", "not recorded")),
                    number(row.get("approval_precision")), number(row.get("approval_coverage"))])
            if p.get("scored_first_drafts", 0) and not name.endswith("-retainbest"):
                probability.append([name, f"{p['scored_first_drafts']}/{fresh_n}", number(p.get("auroc")),
                    number(p.get("brier")), number(p.get("nll")), number(p.get("first_draft_approval_precision")),
                    number(p.get("approval_coverage_of_planned"))])
        sections.append(section("5. Matched confirmation quality", [
            "All planned questions remain in each denominator. Failed/missing policy episodes score zero for EM/F1. These are terminal common-trajectory replay results; prefix model seconds are not measured live end-to-end latency. Raw reported tokens across different tokenizers are usage records, not equal compute.",
            f"Secondary visited-confidence policies: {', '.join(paired.get('secondary_retain_best_policies', [])) or 'none'}. They cannot inspect future drafts, rescue failures or erase prior costs.",
        ], ["Policy", "Complete", "EM %", "F1", "Drafts/q", "Tokens/q", "Prefix s/q"], quality))
        contrasts = []
        for name, metrics in paired["paired_comparisons"].items():
            values = []
            for key in ("exact_match", "token_f1"):
                row = metrics[key]
                lo, hi = row["bootstrap95"]
                values.append(f"{100*row['candidate_minus_baseline']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
            contrasts.append([name, *values, str(metrics["exact_match"]["complete_pair_n"])])
        sections.append(section("Paired differences", [
            "Candidate minus baseline, percentage points; 95% paired percentile bootstrap, 5,000 draws, seed 42, original question as sampling unit. Intervals are descriptive and do not establish general superiority, equivalence or non-inferiority. Repeated rounds do not increase independent N.",
        ], ["Contrast", "EM delta [95%CI]", "F1 delta [95%CI]", "Complete pairs"], contrasts))
        if probability:
            sections.append(section("Probability diagnostics and coverage", [
                "AUROC/Brier/NLL use observed first decisions only; missing/context-overflow probabilities remain coverage failures. Null AUROC for one label class is undefined, not zero. Approval precision is conditional on approval; show coverage alongside it.",
            ], ["Policy", "Scored/N", "AUROC", "Brier", "NLL", "Precision", "Approval/N"], probability))
        if transitions:
            sections.append(section("Repairs, regressions and approvals", [
                "Transitions use answer EM as the recorded correctness proxy. Correct-first-draft loss includes failures; approval precision is conditional and must be read with approval coverage. Missing values are not invented.",
            ], ["Policy", "Repairs", "Regressions", "Correct lost", "Wrong approvals", "Precision", "Approval/N"], transitions))
    else:
        sections.append(section("5. Confirmation result not yet established", [
            "No validated terminal paired Strands confirmation metrics are supplied. No accuracy gain, probability advantage, trained-controller improvement or operational saving is claimed. Partial training generation is not a substitute for a matched fresh comparison.",
        ]))
    usage_rows = []
    def add_usage(name, usage):
        usage_rows.append([name, str(usage.get("completed_requests_including_invalid_outputs", usage.get("completed_encoder_forwards", "not recorded"))),
            number(usage.get("reported_physical_input_tokens", usage.get("input_tokens")), 0),
            "not generative" if "completed_encoder_forwards" in usage else number(usage.get("reported_output_tokens"), 0),
            number(usage.get("reported_request_seconds", usage.get("encoding_seconds"))), number(usage.get("hosted_cost_usd"), 6)])
    if manifest.get("actual_evaluation_usage"):
        add_usage("TRAIN writer", manifest["actual_evaluation_usage"])
    if historical and historical["manifest"].get("actual_evaluation_usage"):
        add_usage("Historical TRAIN preparation", historical["manifest"]["actual_evaluation_usage"])
    if baseline_result:
        add_usage("Untouched DEVELOPMENT native decisions", baseline_result["usage"])
    extra_phases = {train, paths["baseline"].parent} if paths.get("baseline") else {train}
    if historical_path: extra_phases.add(historical_path)
    if owner_plan and owner_dir and baseline_result:
        try:
            wanted = owner_plan.get("baseline_reuse", {}).get("actual_v3_plan_sha256")
            specs = [s for group in owner_plan.get("prerequisites", {}).values()
                for s in (group if isinstance(group, list) else [group]) if wanted and s.get("sha256") == wanted]
            if len(specs) == 1:
                old_plan = evidence.read(specs[0]["path"], "original DEVELOPMENT writer owner")
                old_owner = evidence.path(specs[0]["path"]).parent
                old_state = evidence.read(old_owner / "campaign.json", "original DEVELOPMENT completed owner")
                if evidence.record_file(specs[0]["path"], "original DEVELOPMENT plan identity") != wanted:
                    raise ValueError("original DEVELOPMENT plan changed")
                candidates = [s for s in old_plan["steps"] if "pc_strands.evaluate" in s["argv"] and "generate" in s["argv"]]
                if len(candidates) != 1: raise ValueError("ambiguous original DEVELOPMENT writer")
                directory, usage = captured_phase_usage(evidence, old_plan, old_state, old_owner, candidates[0], baseline["planned_task_ids"])
                if evidence.read(directory / "tasks-evaluator-only.json", "original DEVELOPMENT writer tasks") != evidence.read(paths["baseline"].parent / "tasks-evaluator-only.json", "original DEVELOPMENT baseline tasks"):
                    raise ValueError("original DEVELOPMENT writer tasks differ")
                if directory in extra_phases: raise ValueError("duplicate physical phase")
                extra_phases.add(directory); add_usage("Original DEVELOPMENT writer once", usage)
        except (ValueError, KeyError, TypeError, OSError):
            warnings.append("Original DEVELOPMENT writer expense omitted: owning-stage/raw accounting was not verified.")
    if (owner_plan and owner_dir and gate and trained and control
            and any("head-replay" in s.get("argv", []) for s in owner_plan.get("steps", []))):
        for kind in ("untuned", "trained"):
            try:
                q = development_auditor(evidence).q
                fit_path = paths.get("calibration_" + kind)
                if fit_path is None: raise ValueError("CAL fit role not supplied")
                directory = fit_path.parent
                matches = [s for s in owner_plan["steps"] if "head-replay" in s["argv"]
                    and evidence.path(q.argument(s, "--output")) == directory
                    and evidence.path(q.argument(s, "--checkpoint-dir")) == paths[kind]
                    and q.argument(s, "--partition") == "calibration"]
                if len(matches) != 1: continue
                fit = evidence.read(paths["calibration_" + kind], kind + " CAL planned questions")
                ids = evidence.read(directory / "manifest.json", kind + " CAL order")["frozen"]["task_ids"]
                if set(ids) != set(fit["question_ids"]): raise ValueError("CAL planned IDs differ")
                directory, usage = captured_phase_usage(evidence, owner_plan, owner_campaign, owner_dir, matches[0], ids, head=True)
                if directory in extra_phases: raise ValueError("duplicate physical phase")
                extra_phases.add(directory); add_usage("CALIBRATION " + kind + " CPU head", usage)
            except (ValueError, KeyError, TypeError, OSError):
                warnings.append(kind + " CAL CPU expense omitted: owning-stage/raw accounting was not verified.")
    if development:
        encoding = development["encoding_expense"]
        add_usage("Secondary DEVELOPMENT encoder once", {"completed_encoder_forwards": encoding["physical_encoder_forwards"],
            "input_tokens": encoding["physical_encoder_input_tokens"], "encoding_seconds": encoding["encoding_seconds"], "hosted_cost_usd": 0})
        for kind in ("untuned", "trained"):
            arm = development["per_policy"][kind]
            add_usage("Secondary DEVELOPMENT " + kind + " CPU head", {"completed_requests_including_invalid_outputs": arm["head_score_requests_including_known_failures"],
                "reported_physical_input_tokens": 0, "reported_output_tokens": 0,
                "reported_request_seconds": arm["request_seconds"], "hosted_cost_usd": 0})
    if paired:
        for key in ("writer_generation_once", "real_encoder_features_once", "untuned_head_scoring", "tuned_head_scoring"):
            if paired.get("physical_evaluation_usage", {}).get(key):
                add_usage(key, paired["physical_evaluation_usage"][key])
    for partition in ("fit", "tune", "calibration"):
        cache_path = paths.get("cache_" + partition)
        feature = evidence.read(str(cache_path) + ".manifest.json", f"{partition} real encoder coverage") if cache_path else None
        if feature:
            add_usage(f"TRAIN encoder {partition}", feature.get("physical_encoding_usage", {}))
            warnings.append(f"{partition} features: selected={feature.get('selected_questions')}; overflow rows={feature.get('total_overflow_rows')}; unreconciled={feature.get('unreconciled_inferences')}.")
    sections.append(section("6. Physical work and interpretation", [
        "This table records physical work once per phase; it is not a total system-cost estimate. Shared writer generation and real encoder features are not repeated expenses for each replayed policy. CPU head replay adds zero physical encoder tokens. Training/calibration/startup and intentional idle remain separate evidence; inspect provenance and phase manifests for omitted components.",
        f"TRAIN invocation wall={number(manifest.get('this_invocation_wall_seconds'))}s; intentional idle={number(manifest.get('this_invocation_intentional_idle_seconds'))}s. Resumed invocations must not be confused with cumulative journal usage.",
        "Local hosted API cost of $0 leaves hardware/electricity unpriced. Counterfactual policy prefixes are not additional physical API calls or demonstrated deployment savings. Missing usage stays unknown or a stated lower bound, never fabricated zero. The historical Jev development PDF is a separate cohort/profile and is preserved unchanged.",
    ], ["Phase", "Calls/forwards", "Input tokens", "Output tokens", "Model seconds", "Hosted USD"], usage_rows))
    if baseline_result:
        timing = baseline_result["timing"]
        sections.append(section("Untouched DEVELOPMENT wall and idle scopes", [
            "Native request seconds in the physical-work table are a subset of replay-phase wall time. Backend startup is separate from that phase; only startup wall plus phase wall is summed here. Do not add request time again. Writer draft generation is a separate expense and is not included in this native-decision timing.",
            "Phase intentional idle includes rests, resource holds and scheduling overshoot. Their residual is not an exact RAM-pause measurement. Hardware/electricity remain unpriced, and local hosted USD0 is not zero total cost.",
        ], ["Timing scope", "Recorded seconds", "Interpretation"], [
            ["Backend startup wall", number(timing["startup_wall_seconds"], 6), "Admission/construction/placement/rest"],
            ["Replay phase wall", number(timing["phase_wall_seconds"], 6), "Native requests, phase rests and holds"],
            ["Phase intentional idle", number(timing["phase_idle_seconds"], 6), "Inside phase wall; not added again"],
            ["Startup plus phase wall", number(timing["startup_plus_phase_wall_seconds"], 6), "Separate scopes summed once"],
        ]))
    if historical:
        old = historical["reconciliation"]
        sections.append(section("Historical TRAIN preparation coverage", [
            "This explicitly selected earlier source contributes physical preparation expense only. Its questions, trajectories, references, feature caches and planned denominator are not added to the selected training cohort or confirmation quality. Repeated questions in a new writer run still incur new physical calls.",
            "Historical cumulative journal usage is charged once. Its archived cumulative counters supply timing provenance and are not additional calls/tokens. These rows remain a partial accounting of the campaign, with hardware/electricity and unprovided phases unpriced.",
        ], ["Source state", "Planned", "Saved", "Physical calls", "Missing", "Completed without saved round"], [[
            historical["manifest"].get("status", "not recorded"), str(old["planned_questions"]),
            str(old["saved_trajectories"]), str(historical["journal"]["completed"]),
            str(old["missing_trajectories"]), str(old["completed_calls_without_saved_round"])]]))
    timing_sources = [("TRAIN invocation timing", timings, train_segments),
        ("Historical TRAIN invocation timing", historical["timings"] if historical else [], historical_train_segments)]
    for timing_title, source_timings, source_archives in timing_sources:
        if not source_timings:
            continue
        rows = [[f"Invocation {t['invocation']} ({'archived' if t['archived'] else 'current'})",
            number(t["wall_seconds"]), number(t["intentional_idle_seconds"]),
            f"+{t['additional_physical_calls']} since prior snapshot" if t["additional_physical_calls"] is not None else f"{t['cumulative_physical_calls']} cumulative; increment not isolated",
            t["finish_before_unload_utc"]] for t in source_timings]
        if source_archives:
            rows.append(["Sum of selected invocations", number(sum(t["wall_seconds"] for t in source_timings)),
                number(sum(t["intentional_idle_seconds"] for t in source_timings)),
                f"{source_timings[-1]['cumulative_physical_calls']} final cumulative", "excludes gaps / unload"])
        sections.append(section(timing_title, [
            "Archived invocations are included only when explicitly selected, receipt-hash verified and journal-prefix matched to the current source. Their worker intervals must not overlap. Starts are derived from recorded finish minus measured invocation wall time; source started_at may retain the first invocation timestamp after resume.",
            "Wall/idle sums exclude the gap between invocations, wrapper startup and final unload/cleanup. Physical token/call usage above is already cumulative and is counted once. The first selected snapshot has no proven prior counter; later increments are differences between selected snapshots, not additional expenses to add to the cumulative total. Without selected archives, only the latest invocation is shown.",
        ], ["Invocation", "Worker wall s", "Intentional idle s", "Physical call counter", "Recorded finish UTC"], rows))
    sections.append(section("7. Two next architectural ideas", [
        "Evidence-specialist repair under Strands stop authority. On rejection, a specialist returns short diagnostics cited to supplied evidence before revision; Strands retains final STOP authority, with at most one specialist call per revision and two per question. Pros: inspectable, targeted repairs. Cons: extra cost/latency, wrong diagnostics and more failure modes. Test generic versus cited feedback under matched decisions and count every role's work.",
        "Budget-aware STOP/VERIFY utility learning after the supervised baseline. Use disjoint TRAIN continuations to learn whether verification is worth its token/time cost, initially with bounded one-step full-information targets. Pros: directly aligns decisions with budget and may avoid unnecessary verification. Cons: reward-weight sensitivity, proxy-label errors and replay/off-policy bias; actual action effects need later matched live validation. Supervised head training is not an RL result, and fresh confirmation must never tune the policy.",
    ]))
    if warnings:
        sections.append(section("Evidence qualifications", warnings))
    evidence.stable()
    return {"title": "HaltIQ Strands study", "status": status, "as_of": as_of,
        "fresh_n": fresh_n, "train_run": str(train), "sections": sections,
        "training_campaign_run": str(evidence.path(training_campaign_run)) if training_campaign_run else None,
        "input_files": list(evidence.files.values()), "missing_inputs": evidence.missing,
        "artifact_roles": {k: str(v) if isinstance(v, Path) else v for k, v in paths.items()},
        "validation": {"untouched_gate": gate, "trained_head_proof": trained,
                       "unchanged_control_proof": control, "paired_analysis": bool(paired), "complete_checkpoint": complete,
                       "baseline_diagnostics": bool(baseline_result), "secondary_development_diagnostics": bool(development)},
        "baseline_development": baseline_result,
        "secondary_development": development,
        "train_reconciliation": reconciliation, "train_invocations": timings,
        "historical_train_expense": ({"run": str(historical_path),
            "status": historical["manifest"].get("status"),
            "actual_evaluation_usage": historical["manifest"].get("actual_evaluation_usage", {}),
            "reconciliation": historical["reconciliation"], "invocations": historical["timings"]}
            if historical else None),
        "warnings": warnings, "validation_errors": validation_errors}


def plain_text(report):
    lines = [report["title"], report["status"], "Evidence checkpoint: " + report["as_of"], ""]
    for part in report["sections"]:
        lines += [part["title"], *part["paragraphs"]]
        if part["headers"]:
            lines.append(" | ".join(part["headers"]))
            lines += [" | ".join(str(x) for x in row) for row in part["rows"]]
        lines.append("")
    lines += ["Input provenance (complete SHA256s also in report-provenance.json)"]
    lines += [f"[{r['id']}] {r['role']}: {r['label']} SHA256 {r['sha256']}" for r in report["input_files"]]
    return "\n".join(lines) + "\n"


def render_pdf(report, path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyReport", fontName="Helvetica", fontSize=9.5, leading=13.4, spaceAfter=8, splitLongWords=True))
    styles.add(ParagraphStyle(name="CellReport", fontName="Helvetica", fontSize=8, leading=10.5, splitLongWords=True))
    styles.add(ParagraphStyle(name="SourceReport", fontName="Helvetica", fontSize=7.4, leading=10, spaceAfter=5, splitLongWords=True))
    styles["Title"].fontName, styles["Title"].fontSize, styles["Title"].leading = "Helvetica-Bold", 28, 32
    styles["Title"].alignment = TA_LEFT
    styles["Heading2"].spaceBefore, styles["Heading2"].spaceAfter = 15, 8
    styles["Heading2"].keepWithNext = True
    def p(text, style="BodyReport"):
        # ASCII punctuation is intentional; escape source strings before XML parsing.
        clean = str(text).replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
        return Paragraph(escape(clean), styles[style])
    story = [p(report["title"], "Title"), p(report["status"]), p("Evidence checkpoint: " + report["as_of"]), Spacer(1, 7)]
    width = letter[0] - 96
    for part in report["sections"]:
        story.append(p(part["title"], "Heading2"))
        story += [p(text) for text in part["paragraphs"]]
        if part["headers"] and part["rows"]:
            count = len(part["headers"])
            fractions = {2: [.32, .68], 4: [.31, .17, .16, .36],
                         5: [.34, .16, .17, .17, .16], 6: [.26, .13, .16, .14, .16, .15],
                         7: [.28, .12, .10, .10, .11, .14, .15]}.get(count, [1/count]*count)
            # Different four-column tables need more room for paired intervals.
            if part["title"] == "Paired differences":
                fractions = [.36, .24, .24, .16]
            rows = [[p(x, "CellReport") for x in part["headers"]]]
            rows += [[p(x, "CellReport") for x in row] for row in part["rows"]]
            table = LongTable(rows, colWidths=[width*f for f in fractions], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#edf2f4")),
                ("VALIGN", (0,0), (-1,-1), "TOP"), ("LINEBELOW", (0,0), (-1,0), .8, colors.HexColor("#153f4b")),
                ("LINEBELOW", (0,1), (-1,-1), .25, colors.HexColor("#cbd4d8")),
                ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6)]))
            story += [table, Spacer(1, 7)]
    story.append(p("Evidence references", "Heading2"))
    story.append(p("Input SHA256s and missing-file records are preserved in the provenance JSON. Source paths below identify the actual receipts; they are not model outputs.", "SourceReport"))
    for record in report["input_files"]:
        story.append(p(f"[{record['id']}] {record['role']}: {record['label']} (SHA256 {record['sha256'][:12]}...)", "SourceReport"))
    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd4d8"))
        canvas.line(48, 755, 564, 755)
        canvas.line(48, 37, 564, 37)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#43515a"))
        canvas.drawString(48, 764, "HALTIQ  |  PROFESSOR REVIEW  |  " + report["status"])
        canvas.drawString(48, 25, "Yash  |  " + report["as_of"])
        canvas.drawRightString(564, 25, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=48, rightMargin=48,
        topMargin=51, bottomMargin=49, title=report["title"], author="Yash")
    doc.build(story, onFirstPage=page, onLaterPages=page)


def write_report(report, output_dir):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("new report output must be empty; previous reports are preserved")
    output.mkdir(parents=True, exist_ok=True)
    text, pdf = output / "professor-review.txt", output / "professor-review.pdf"
    text.write_text(plain_text(report), encoding="utf-8")
    render_pdf(report, pdf)
    from pypdf import PdfReader
    reader = PdfReader(str(pdf))
    if not reader.pages or any(not page.extract_text().strip() for page in reader.pages):
        raise ValueError("generated PDF contains an empty page")
    provenance = {k:v for k,v in report.items() if k != "sections"}
    provenance.update(generator_sha256=digest(__file__), generated_at_utc=datetime.now(timezone.utc).isoformat(),
        outputs={p.name: {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size} for p in (pdf, text)},
        pdf_pages=len(reader.pages), visual_review="PENDING: render and inspect every page before delivery")
    (output / "report-provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--fresh-n", required=True, type=int, choices=(0,20,50,100))
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cohort-manifest")
    parser.add_argument("--campaign-run")
    parser.add_argument("--training-campaign-run", help="Explicit completed training-only owner for a separate fresh-only campaign; exact plan/state/source/output identities required")
    parser.add_argument("--development-diagnostic", help="Explicit secondary DEVELOPMENT20/cap1 receipt; missing/inconsistent evidence suppresses secondary diagnostics only")
    parser.add_argument("--train-segment", action="append", default=[], help="Explicit archived TRAIN segment directory, in chronological order; receipt required")
    parser.add_argument("--historical-train-run", help="One explicit earlier TRAIN writer source, physical expense only; never merged into the selected cohort")
    parser.add_argument("--historical-train-segment", action="append", default=[], help="Explicit archive of that historical source, in chronological order; receipt required")
    args = parser.parse_args()
    report = collect_report(args.input_root, args.train_run, args.fresh_n, args.as_of,
        cohort_manifest=args.cohort_manifest, campaign_run=args.campaign_run, train_segments=args.train_segment,
        historical_train_run=args.historical_train_run, historical_train_segments=args.historical_train_segment,
        training_campaign_run=args.training_campaign_run, development_diagnostic=args.development_diagnostic)
    provenance = write_report(report, args.output_dir)
    print(json.dumps({"status": report["status"], "pdf_pages": provenance["pdf_pages"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
