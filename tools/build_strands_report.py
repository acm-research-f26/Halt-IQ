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
import json
import math
from pathlib import Path
import sys

# Direct script execution must resolve the maintained sibling package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TERMINAL = {"completed", "completed_with_errors"}


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


def baseline_valid(receipt):
    from pc_strands.train_head import validate_baseline_receipt
    try:
        validate_baseline_receipt(receipt or {}, (receipt or {}).get("checkpoint_head_sha256"))
    except (ValueError, KeyError, TypeError):
        return False
    return bool(receipt.get("checkpoint_head_sha256"))


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


def calibration_valid(evidence, kind, excluded):
    directory = "checkpoints/released-calibrated" if kind == "untuned" else "checkpoints/head-trained"
    fit = evidence.read(f"study-runs/calibration-{kind}/temperature-fit.json", f"{kind} temperature fit")
    proof = evidence.read(f"{directory}/calibration-proof.json", f"{kind} checkpoint calibration proof")
    config = evidence.read(f"{directory}/strands_decider_config.json", f"{kind} calibrated config")
    weight = evidence.record_file(f"{directory}/head.safetensors", f"{kind} calibrated head weights")
    ids = (fit or {}).get("question_ids", [])
    return bool(fit and fit == proof and fit.get("status") == "fitted_on_calibration_only_not_test"
        and ids and len(set(ids)) == len(ids) and not set(ids) & set(excluded)
        and weight == fit.get("checkpoint_head_sha256") and config
        and config.get("temperature") == fit.get("temperature")
        and config.get("temperature_by_kind", {}).get("noul") == fit.get("temperature"))


def confirmation(evidence, n, as_of, campaign_dir, campaign, plan):
    """Verify cohort timing and independently recompute the canonical pure join."""
    from pc_strands.compare import compare_runs
    from pc_strands.data import load_tasks
    if not campaign_dir or not campaign or not plan:
        raise ValueError("positive fresh N requires an explicit frozen campaign and plan")
    freeze = evidence.read("fresh-N-freeze.json", "answer-blind fresh cohort freeze")
    if not freeze or (freeze.get("n") != n or freeze.get("answer_blind") is not True
            or freeze.get("fresh_scoring_started") is not False or freeze.get("basis") != "measured_throughput_and_resources"):
        raise ValueError("positive fresh N requires its explicit matching answer-blind pre-score freeze")
    plan_hash = digest(evidence.path(campaign_dir) / "plan.json")
    if freeze.get("plan_sha256") != plan_hash or campaign.get("plan_sha256") != plan_hash:
        raise ValueError("campaign/fresh freeze plan identity differs")
    copied = evidence.record_file(evidence.path(campaign_dir) / "fresh-freeze.json", "campaign copied pre-score freeze")
    operator_hash = digest(evidence.path("fresh-N-freeze.json"))
    if copied != operator_hash or campaign.get("fresh_freeze_sha256") != copied or campaign.get("fresh_n") != n:
        raise ValueError("campaign did not adopt the exact operator fresh freeze")
    options = [o for o in plan.get("fresh_options", []) if o.get("n") == n]
    dataset = evidence.path(f"candidate-panels/fresh{n}/tasks.jsonl")
    dataset_hash = evidence.record_file(dataset, "exact frozen confirmation dataset")
    if (len(options) != 1 or evidence.path(options[0]["dataset"]) != dataset
            or dataset_hash != freeze.get("dataset_sha256") or options[0].get("dataset_sha256", dataset_hash) != dataset_hash):
        raise ValueError("exact plan confirmation dataset differs from freeze")
    ids = [t.id for t in load_tasks(dataset)]
    if len(ids) != n:
        raise ValueError("actual fresh cohort differs from explicit N")
    source = evidence.path(f"study-runs/fresh{n}-fixed10")
    generation = evidence.read(source / "manifest.json", "fresh generation start and cleanup")
    frozen_at = timestamp(freeze["frozen_at"])
    if frozen_at > timestamp(as_of) or frozen_at > timestamp(generation["started_at_utc"]):
        raise ValueError("fresh freeze must precede fresh generation and report checkpoint")
    raw = evidence.read(f"study-runs/fresh{n}-paired-analysis/paired-analysis.json", "terminal paired confirmation analysis")
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
    actual = compare_runs(evidence.path(f"study-runs/fresh{n}-untuned"),
        evidence.path(f"study-runs/fresh{n}-trained"), source,
        evidence.path(f"feature-caches/fresh{n}-allrounds.jsonl"), retain_best=retain)
    for key, value in actual.items():
        if key not in {"created_at_utc", "input_sha256"} and raw.get(key) != value:
            raise ValueError(f"published comparison differs from authoritative pure comparison: {key}")
    for path, expected in actual["input_sha256"].items():
        if evidence.record_file(path, "independently rejoined canonical input") != expected:
            raise ValueError("canonical comparison input changed")
    for name, directory in (("strands-untuned", "checkpoints/released-calibrated"),
                            ("strands-trained", "checkpoints/head-trained")):
        if raw.get("model_fingerprints", {}).get(name) != checkpoint_fingerprint(evidence, directory):
            raise ValueError("comparison does not bind current primary checkpoints")
    proof = evidence.read("checkpoints/head-trained/training-proof.json", "actual trained-head proof") or {}
    excluded = ids + proof.get("fit_question_ids", []) + proof.get("tune_question_ids", [])
    if not all(calibration_valid(evidence, kind, excluded) for kind in ("untuned", "trained")):
        raise ValueError("both current primary heads require disjoint hash-bound calibration")
    return raw


def section(title, paragraphs=(), headers=(), rows=()):
    return {"title": title, "paragraphs": list(paragraphs), "headers": list(headers), "rows": list(rows)}


def collect_report(input_root, train_run, fresh_n, as_of, *, cohort_manifest=None, campaign_run=None, train_segments=()):
    if fresh_n not in (0, 20, 50, 100):
        raise ValueError("explicit fresh N must be0,20,50 or100")
    as_of = timestamp(as_of).isoformat()
    evidence = Evidence(input_root)
    train = evidence.path(train_run)
    manifest = evidence.read(train / "manifest.json", "TRAIN generation manifest") or {}
    train_control = evidence.read(train / "control-status.json", "last TRAIN resource/control observation") or {}
    trajectories = evidence.read(train / "trajectories.json", "TRAIN attempted trajectories") or []
    tasks = evidence.read(train / "tasks-evaluator-only.json", "TRAIN original partition metadata") or []
    train_events = evidence.read(train / "requests.jsonl", "TRAIN physical request journal", lines=True) or []
    journal = journal_counts(train_events)
    by_id = {t["id"]: t for t in tasks}
    planned_ids = manifest.get("frozen", {}).get("task_ids", [])
    trajectory_ids = [t["task_id"] for t in trajectories]
    if (len(by_id) != len(tasks) or len(set(trajectory_ids)) != len(trajectory_ids)
            or any(q not in planned_ids or q not in by_id for q in trajectory_ids)):
        raise ValueError("duplicate or foreign TRAIN trajectory/task identifiers")
    physical = manifest.get("actual_evaluation_usage", {})
    if (physical.get("completed_requests_including_invalid_outputs") != journal["completed"]
            or physical.get("unreconciled_requests", journal["pending"]) != journal["pending"]):
        raise ValueError("TRAIN manifest and physical journal counts differ")
    partitions = Counter(by_id.get(t["task_id"], {}).get("partition", "unknown") for t in trajectories)
    planned = len(manifest.get("frozen", {}).get("task_ids", []))
    reconciliation = train_reconciliation(manifest, trajectories, train_events)
    timings = train_invocation_timings(evidence, train, manifest, train_segments)
    baseline = evidence.read("study-runs/untuned-dev20-decisions/untuned-pretraining-receipt.json", "untouched real-forward gate")
    proof = evidence.read("checkpoints/head-trained/training-proof.json", "actual trained-head proof")
    gate, trained = baseline_valid(baseline), training_valid(proof)
    warnings, validation_errors = [], []
    if reconciliation["completed_calls_without_saved_round"]:
        warnings.append(f"{reconciliation['completed_calls_without_saved_round']} completed TRAIN provider call(s) have no saved trajectory round. Their physical work is charged; they are not invented scored drafts.")
    if baseline:
        released_hash = evidence.record_file("models/decider/head.safetensors", "released untouched head weights")
        if released_hash != baseline.get("checkpoint_head_sha256"):
            gate = False
            warnings.append("Untouched gate does not bind the actual released head.")
        for name, key in (("manifest.json", "manifest_sha256"), ("episodes.json", "episodes_sha256")):
            path = evidence.path("study-runs/untuned-dev20-decisions") / name
            evidence.read(path, "untouched receipt bound result")
            if not path.is_file() or digest(path) != baseline.get(key):
                gate = False
                warnings.append("Untouched gate result hash is missing or inconsistent.")
    if proof and baseline:
        if proof.get("original_head_file_sha256") != baseline.get("checkpoint_head_sha256"):
            trained = False
            warnings.append("Training proof and untouched released-head identity differ.")
        baseline_path = evidence.path("study-runs/untuned-dev20-decisions/untuned-pretraining-receipt.json")
        if proof.get("baseline_receipt_sha256") != digest(baseline_path):
            trained = False
            warnings.append("Training proof does not bind the supplied untouched gate.")
        if set(baseline.get("planned_task_ids", [])) & (
                set(proof.get("fit_question_ids", [])) | set(proof.get("tune_question_ids", []))):
            trained = False
            warnings.append("Untouched development IDs overlap the training/tuning proof.")
    if proof:
        head_hash = evidence.record_file("checkpoints/head-trained/head.safetensors", "saved trained pointer weights")
        if not head_hash or head_hash != proof.get("checkpoint_head_sha256"):
            trained = False
            warnings.append("Actual trained-head file is missing or differs from training proof.")
        adapter = evidence.record_file("models/decider/lora/adapter_model.safetensors", "released frozen adapter")
        if not adapter or adapter != proof.get("frozen_adapter_sha256"):
            trained = False
            warnings.append("Training proof does not bind the actual released adapter.")
        for partition in ("fit", "tune"):
            cache_path = evidence.path(f"feature-caches/{partition}.jsonl")
            actual_hash = evidence.record_file(cache_path, f"{partition} training cache")
            manifest_hash = evidence.record_file(str(cache_path)+".manifest.json", f"{partition} training cache manifest")
            if (not actual_hash or actual_hash != proof.get(f"{partition}_cache_sha256")
                    or manifest_hash != proof.get("feature_manifest_sha256", {}).get(partition)):
                trained = False
                warnings.append(f"Training proof {partition} cache fingerprints are not verified.")
    subset = evidence.read(cohort_manifest, "explicit selected TRAIN cohort") if cohort_manifest else None
    if subset and "PROPOSED" in subset.get("status", ""):
        warnings.append("Supplied TRAIN subset remains proposed; it is not an adopted training result.")
    campaign = evidence.read(evidence.path(campaign_run) / "campaign.json", "explicit campaign state") if campaign_run else None
    plan = evidence.read(evidence.path(campaign_run) / "plan.json", "explicit frozen campaign plan") if campaign_run else None
    paired = None
    if fresh_n:
        if not campaign_run:
            raise ValueError("positive fresh N requires an explicit frozen campaign and plan")
        try:
            paired = confirmation(evidence, fresh_n, as_of, campaign_run, campaign, plan)
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
        ["Fresh paired confirmation", str(fresh_n or "not selected"), str(paired["planned_n"] if paired else 0),
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
    ], ["Evidence", "Declared", "Observed", "Interpretation"], coverage))
    config = manifest.get("frozen", {}).get("config", {})
    writer, settings = config.get("writer", {}), config.get("writer_settings", {})
    dataset_metadata = evidence.read(evidence.root.parent / "research-2026-10-18-strands/official-train600-candidate/manifest.json", "official TRAIN selection and exclusion audit") or {}
    evidence.record_file(evidence.root.parent / "preflight-2026-10-18/PROFESSOR-METHODS-DRAFT.txt", "historical methods and HotpotQA attribution")
    sections.append(section("2. Dataset and architecture", [
        "The task is short-answer HotpotQA with all supplied distractor paragraphs, titles and sentence order. Questions, evidence and the current draft are public model inputs; references, supporting facts and correctness labels are evaluator-only. Answer metrics use official-style normalized exact match and token F1.",
        "Official TRAIN and public validation must be disjoint by ID and normalized question. The original 600-question training proposal uses 400 fit / 100 tune / 100 calibration cases; its six fixed 100-row source blocks are not an IID sample of all 90,447 TRAIN questions. A resource-driven repartition requires a separately adopted manifest before learning, with original partitions and parent provenance preserved.",
        f"Dataset attribution: HotpotQA (https://hotpotqa.github.io/), CC-BY-SA 4.0; selected source revision {dataset_metadata.get('source_revision', 'not supplied')}. Preserve source attribution and the recorded ID/question exclusion audit when distributing derived data.",
        f"Selected training dataset SHA256: {config.get('dataset_sha256', 'not supplied')}. "
        f"Explicit cohort manifest status: {(subset or {}).get('status', 'none supplied')}; "
        f"recorded partition counts: {(subset or {}).get('partition_counts', dict(partitions))}.",
        "The proposed Strands controller scores binary draft correctness and stops at probability >=0.8 or continues a generic revision loop. Strands accepts at most 4,096 fully rendered tokens, without truncation; overflow remains an explicit coverage failure. The writer has a separate 16k context. Ten total drafts and cap-three parity share recorded trajectories. The released 2B encoder and adapter remain frozen; actual supervised fine-tuning updates the released pointer head only. One EM correctness proxy is not three independently labeled rubric targets. LoRA/backbone training and RL are separate future work.",
        "Confirmation is a once-frozen subset of 20, 50 or 100 public validation questions, not an official hidden-test score. Shared Fixed 1/2/3/5/10 controls use the identical observations. The unchanged Vedanshi v0 export is a schema contract, not proof of identical hardware; canonical planned-N summaries retain failures/zero-draft cases that v0 cannot express.",
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
        ["Source head / adapter unchanged", f"{(proof or {}).get('source_head_unchanged', 'not recorded')} / {(proof or {}).get('source_adapter_unchanged', 'not recorded')}"],
        ["Released head SHA256", str((baseline or {}).get("checkpoint_head_sha256", "not recorded"))],
        ["Saved trained head SHA256", str((proof or {}).get("checkpoint_head_sha256", "not recorded"))],
    ]
    calibration_rows = []
    for kind in ("untuned", "trained"):
        fit = evidence.read(f"study-runs/calibration-{kind}/temperature-fit.json", f"{kind} temperature fit")
        if fit:
            excluded = set((proof or {}).get("fit_question_ids", [])) | set((proof or {}).get("tune_question_ids", []))
            if calibration_valid(evidence, kind, excluded):
                calibration_rows.append([kind, str(len(fit.get("question_ids", []))), number(fit.get("temperature")),
                    number(fit.get("before", {}).get("nll")), number(fit.get("after", {}).get("nll"))])
            else:
                warnings.append(f"{kind} calibration table omitted: current checkpoint/calibration linkage not verified.")
    sections.append(section("4. Untouched baseline, training and calibration", [
        "The untouched real-model 20-question gate must precede weight updates and bind final manifest/episode hashes. Successful native inference is distinct from mocked tests or CPU head replay. Training proof requires changed tensor hashes, positive weight delta, unchanged released source/adapter and matching reload logits.",
        "Training selects by disjoint tuning-question NLL; separate calibration questions fit each head's temperature using the same predeclared procedure. Neither phase may read fresh outcomes. Better training loss, successful reload and a changed head are not held-out improvement.",
    ], ["Proof", "Observed"], proof_rows))
    if calibration_rows:
        sections.append(section("Calibration evidence", ["Calibration NLL is measured on the fitting data, not confirmation performance."],
            ["Head", "Questions", "Temperature", "NLL T1", "NLL fitted"], calibration_rows))
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
    if paired:
        for key in ("writer_generation_once", "real_encoder_features_once", "untuned_head_scoring", "tuned_head_scoring"):
            if paired.get("physical_evaluation_usage", {}).get(key):
                add_usage(key, paired["physical_evaluation_usage"][key])
    for partition in ("fit", "tune", "calibration"):
        feature = evidence.read(f"feature-caches/{partition}.jsonl.manifest.json", f"{partition} real encoder coverage")
        if feature:
            add_usage(f"TRAIN encoder {partition}", feature.get("physical_encoding_usage", {}))
            warnings.append(f"{partition} features: selected={feature.get('selected_questions')}; overflow rows={feature.get('total_overflow_rows')}; unreconciled={feature.get('unreconciled_inferences')}.")
    sections.append(section("6. Physical work and interpretation", [
        "This table records physical work once per phase; it is not a total system-cost estimate. Shared writer generation and real encoder features are not repeated expenses for each replayed policy. CPU head replay adds zero physical encoder tokens. Training/calibration/startup and intentional idle remain separate evidence; inspect provenance and phase manifests for omitted components.",
        f"TRAIN invocation wall={number(manifest.get('this_invocation_wall_seconds'))}s; intentional idle={number(manifest.get('this_invocation_intentional_idle_seconds'))}s. Resumed invocations must not be confused with cumulative journal usage.",
        "Local hosted API cost of $0 leaves hardware/electricity unpriced. Counterfactual policy prefixes are not additional physical API calls or demonstrated deployment savings. Missing usage stays unknown or a stated lower bound, never fabricated zero. The historical Jev development PDF is a separate cohort/profile and is preserved unchanged.",
    ], ["Phase", "Calls/forwards", "Input tokens", "Output tokens", "Model seconds", "Hosted USD"], usage_rows))
    if timings:
        rows = [[f"Invocation {t['invocation']} ({'archived' if t['archived'] else 'current'})",
            number(t["wall_seconds"]), number(t["intentional_idle_seconds"]),
            f"+{t['additional_physical_calls']} since prior snapshot" if t["additional_physical_calls"] is not None else f"{t['cumulative_physical_calls']} cumulative; increment not isolated",
            t["finish_before_unload_utc"]] for t in timings]
        if train_segments:
            rows.append(["Sum of selected invocations", number(sum(t["wall_seconds"] for t in timings)),
                number(sum(t["intentional_idle_seconds"] for t in timings)),
                f"{timings[-1]['cumulative_physical_calls']} final cumulative", "excludes gaps / unload"])
        sections.append(section("TRAIN invocation timing", [
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
        "input_files": list(evidence.files.values()), "missing_inputs": evidence.missing,
        "validation": {"untouched_gate": gate, "trained_head_proof": trained,
                       "paired_analysis": bool(paired), "complete_checkpoint": complete},
        "train_reconciliation": reconciliation, "train_invocations": timings,
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
    parser.add_argument("--train-segment", action="append", default=[], help="Explicit archived TRAIN segment directory, in chronological order; receipt required")
    args = parser.parse_args()
    report = collect_report(args.input_root, args.train_run, args.fresh_n, args.as_of,
        cohort_manifest=args.cohort_manifest, campaign_run=args.campaign_run, train_segments=args.train_segment)
    provenance = write_report(report, args.output_dir)
    print(json.dumps({"status": report["status"], "pdf_pages": provenance["pdf_pages"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
