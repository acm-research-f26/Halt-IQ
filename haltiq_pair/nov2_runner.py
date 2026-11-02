"""Manually started November study, with gold confined to this evaluator.

No inference runs on import. Training branches retain their shared public path
but charge only newly executed calls to physical collection. Every fitted seed
freezes before the first held-out call; seed zero is never selected on test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from . import hotpot, nov2_protocol
from .core import Gate
from .linear import LinearHead, lexical_distance, make_features
from .nov2_orchestration import Orchestrator, OrchestrationError, RoleAgents


ARMS = ("fixed1", "legacy_combined6", "strands", "binary_and",
        "always_revise3", "always_invoke", "learned3")
UTILITY_CALL_PENALTY = .02 / 10


class StudyTimeout(TimeoutError):
    """The frozen wall-clock allowance expired; missing attempts stay pending."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    hotpot._atomic_json(Path(path), value)


def _append(path, value):
    hotpot._append_jsonl(Path(path), value)


def _read(path):
    """Primary journals fail closed: no malformed-line dropping or repair."""
    path = Path(path)
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_bytes().split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Corrupt primary journal {path.name} line {number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Non-object primary record in {path.name} line {number}")
        result.append(value)
    return result


def _usage_tokens(value):
    usage = value.get("usage", value) if isinstance(value, dict) else {}
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("total_tokens", usage.get("totalTokens"))
    return tokens if type(tokens) is int and tokens >= 0 else None


class _Ledger:
    def __init__(self, output, deadline, data_mode):
        self.output, self.deadline, self.data_mode = Path(output), deadline, data_mode
        self.context, self.calls, self.critics = {}, [], []
        self.call_ids, self.response_ids = set(), set()
        self.progress = None

    def check(self):
        if time.monotonic() >= self.deadline:
            raise StudyTimeout("Study wall-clock allowance exhausted before the next call")

    def trace(self, event):
        _append(self.output / "traces.jsonl", {**self.context, **event})

    def receipt(self, entry, agents):
        root = getattr(agents, "receipt_root", None)
        call_id = entry.get("call_id")
        if root is None:
            if self.data_mode == "injected_rows_offline_verification":
                return {"status": "injected_api_usage_only", "tokens": _usage_tokens(entry)}
            return {"status": "missing_receipt_root", "tokens": None}
        if not isinstance(call_id, str) or not re.fullmatch(r"[0-9a-f]{32}", call_id):
            return {"status": "missing_or_invalid_call_id", "tokens": None}
        if call_id in self.call_ids:
            return {"status": "duplicate_call_id", "tokens": None}
        self.call_ids.add(call_id)
        root = Path(root)
        by_client = root if root.name == "by-client" else root / "by-client"
        path = by_client / (call_id + ".json")
        try:
            value = json.loads(path.read_bytes())
            response_id = value["response_id"]
            if not isinstance(response_id, str) or not re.fullmatch(r"chatcmpl-[0-9a-f]{32}", response_id):
                raise ValueError("Invalid response ID")
            metadata = {"yash_call_id": call_id, "yash_role": entry["role"]}
            if (value.get("call_id") != call_id or value.get("role") != entry["role"]
                    or value.get("metadata") != metadata or response_id in self.response_ids
                    or value.get("primary_receipt") != "requests/" + response_id + ".json"):
                raise ValueError("Mismatched or duplicate response identity")
            primary = by_client.parent / "requests" / (response_id + ".json")
            raw = primary.read_bytes()
            physical = json.loads(raw)
            tokens = _usage_tokens(value)
            usage = value.get("usage")
            request = physical.get("request") if isinstance(physical, dict) else None
            valid_usage = (isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")))
            if (not isinstance(physical, dict) or physical.get("id") != response_id
                    or value.get("primary_sha256") != hashlib.sha256(raw).hexdigest()
                    or not isinstance(request, dict) or request.get("metadata") != metadata
                    or not valid_usage or usage["prompt_tokens"] + usage["completion_tokens"] != tokens
                    or physical.get("usage") != usage
                    or value.get("finish_reason") not in {"stop", "length", "tool_calls"}
                    or value["finish_reason"] != physical.get("finish_reason")):
                raise ValueError("Response identity or token receipt mismatch")
            self.response_ids.add(response_id)
            return {"status": "reconciled", "tokens": tokens, "response_id": response_id,
                    "client_receipt": str(path), "client_receipt_sha256": _sha(path),
                    "primary_receipt": str(primary), "primary_receipt_sha256": _sha(primary)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {"status": "missing_or_invalid_response_receipt", "tokens": None,
                    "error_type": type(exc).__name__}

    def agent_call(self, entry, agents):
        receipt = self.receipt(entry, agents)
        record = {**self.context, "invocation_index": len(self.calls), **entry,
                  "sdk_reported_tokens": _usage_tokens(entry), "total_tokens": receipt["tokens"],
                  "response_receipt": receipt}
        _append(self.output / "physical-calls.jsonl", record)
        self.calls.append(record)
        if self.progress is not None:
            self.progress()
        return record

    def critic_call(self, entry):
        record = {**self.context, "critic_index": len(self.critics), **entry}
        _append(self.output / "critic-calls.jsonl", record)
        self.critics.append(record)

    def costs(self, start):
        calls = self.calls[start:]
        known = sum(item["total_tokens"] for item in calls if item["total_tokens"] is not None)
        complete = all(item["total_tokens"] is not None for item in calls)
        return {"attempted_agent_calls": len(calls),
                "agent_calls": dict(Counter(item["role"] for item in calls if item["status"] == "completed")),
                "total_tokens": known if complete else None, "known_recorded_tokens": known,
                "journal_complete": complete}


class _TrackedAgents:
    def __init__(self, delegate, ledger):
        self.delegate, self.ledger, self.usage = delegate, ledger, []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def generate(self, role, system, prompt):
        self.ledger.check()
        self.ledger.trace({"event": "agent_invocation_started", "role": role})
        before, started = len(self.delegate.usage), time.monotonic()
        status, error = "completed", None
        try:
            return self.delegate.generate(role, system, prompt)
        except BaseException as exc:
            status, error = "failed", type(exc).__name__
            raise
        finally:
            entries = self.delegate.usage[before:]
            entry = dict(entries[0]) if len(entries) == 1 else {"usage_accounting_error": "expected one invocation entry"}
            entry.update(role=role, status=status, seconds=time.monotonic() - started)
            if error:
                entry["error_type"] = error
            accounted = self.ledger.agent_call(entry, self.delegate)
            self.usage.append(accounted)


class _TrackedCritic:
    def __init__(self, delegate, ledger):
        self.delegate, self.ledger = delegate, ledger

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def score(self, *args):
        self.ledger.check()
        self.ledger.trace({"event": "critic_invocation_started"})
        if hasattr(self.delegate, "context"):
            self.delegate.context = dict(self.ledger.context)
        started = time.monotonic()
        source = getattr(self.delegate, "usage", [])
        before, status, error = len(source), "completed", None
        try:
            return self.delegate.score(*args)
        except BaseException as exc:
            status, error = "failed", type(exc).__name__
            raise
        finally:
            entries = getattr(self.delegate, "usage", [])[before:]
            entry = dict(entries[0]) if len(entries) == 1 else {}
            entry.update(status=status, seconds=time.monotonic() - started,
                         receipt=getattr(self.delegate, "last_record", None))
            if error:
                entry["error_type"] = error
            self.ledger.critic_call(entry)


def _interrupted(exc):
    return isinstance(exc, (StudyTimeout, KeyboardInterrupt, SystemExit)) or getattr(exc, "error_type", "") in {"StudyTimeout", "KeyboardInterrupt", "SystemExit"}


def _collect_partition(name, dataset, orchestrator, ledger):
    from .nov2_policy import feature_vector
    training = []
    for public, gold in dataset:
        ledger.check()
        ledger.context = {"stage": name, "question_id": public.id, "state_id": "initial"}
        before = len(ledger.calls)
        try:
            initial = orchestrator.begin(public, ledger.trace)
            _append(ledger.output / "training-states.jsonl", {**ledger.context, "state": initial.to_dict(),
                    "incremental_physical_calls": len(ledger.calls) - before})
        except BaseException as exc:
            _append(ledger.output / "training-branches.jsonl", {**ledger.context, "action": "INITIAL",
                "status": "interrupted" if _interrupted(exc) else "failed", "metrics": {k: 0. for k in hotpot.METRICS},
                "error": {"type": type(exc).__name__, "message": str(exc)}, **ledger.costs(before)})
            if _interrupted(exc):
                raise StudyTimeout("Collection interrupted") from exc
            continue
        queue = [("initial", initial, 0)]
        while queue:
            state_id, state, depth = queue.pop(0)
            utilities = {}
            available = tuple(orchestrator.valid_actions(state))
            for action in available:
                ledger.check()
                ledger.context = {"stage": name, "question_id": public.id, "state_id": state_id,
                                  "branch_action": action}
                before, critic_before, started = len(ledger.calls), len(ledger.critics), time.monotonic()
                result, error, status = None, None, "completed"
                ledger.trace({"event": "training_branch_started", "action": action})
                try:
                    result = orchestrator.transition(state.clone(), action, ledger.trace)
                except BaseException as exc:
                    error = {"type": getattr(exc, "error_type", type(exc).__name__), "message": str(exc)}
                    status = "interrupted" if _interrupted(exc) else "failed"
                metrics = hotpot.score_prediction(result.draft, gold) if result is not None else {k: 0. for k in hotpot.METRICS}
                costs = ledger.costs(before)
                utility = metrics["f1"] - UTILITY_CALL_PENALTY * costs["attempted_agent_calls"]
                utilities[action] = utility
                _append(ledger.output / "training-branches.jsonl", {**ledger.context,
                    "action": action, "status": status, "metrics": metrics, "utility": utility,
                    "utility_scope": "one-transition returned answer F1 minus .02*incremental attempted LLM calls/10",
                    "error": error, "parent_path_calls": state.agent_calls,
                    "child_path_calls": result.agent_calls if result else None,
                    "incremental_critic_calls": len(ledger.critics) - critic_before,
                    "elapsed_seconds": time.monotonic() - started, **costs,
                    "result_state": result.to_dict() if result is not None else None})
                if status == "interrupted":
                    raise StudyTimeout("Collection interrupted")
                if result is not None and depth == 0 and action != "STOP":
                    child_id = state_id + "/" + action
                    _append(ledger.output / "training-states.jsonl", {"stage": name, "question_id": public.id,
                            "state_id": child_id, "state": result.to_dict(), "incremental_physical_calls": 0,
                            "reused_result_of_branch": action})
                    queue.append((child_id, result, 1))
            sample = {"split": name, "question_id": public.id, "state_id": state_id,
                      "features": feature_vector(state), "available_actions": list(available), "utilities": utilities}
            _append(ledger.output / "training-examples.jsonl", sample)
            training.append(sample)
    return training


def _choose(arm, state, policy, head):
    available = Orchestrator.valid_actions(state)
    if arm == "learned3":
        return policy.decide(state)
    flag = state.verifier_score >= .8
    if arm == "strands" and flag:
        action = "STOP"
    elif arm == "binary_and" and flag and head.probability(make_features(
            state.draft["answer"], state.previous_answer, state.draft_number - 1,
            state.max_drafts, flag, state.previous_distance)) >= .5:
        action = "STOP"
    elif arm == "always_invoke" and "INVOKE" in available:
        action = "INVOKE"
    else:
        action = "REVISE" if "REVISE" in available else "STOP"
    return {"action": action, "available_actions": list(available), "policy": arm}


def _episode(public, gold, arm, orchestrator, agents, critic, policy, ledger, *, best_prefix):
    before, critic_before, started = len(ledger.calls), len(ledger.critics), time.monotonic()
    state, legacy, error, actions = None, None, None, []
    status, termination, approved, iterations, prediction = "completed", "unknown", False, 0, None
    ledger.trace({"event": "episode_started"})
    try:
        if arm in {"fixed1", "legacy_combined6"}:
            legacy = hotpot.run_episode(agents, Gate(critic=critic), public,
                "fixed1" if arm == "fixed1" else "combined", max_rounds=6, trace=ledger.trace)
            status, prediction, iterations = legacy["status"], legacy["prediction"], legacy["rounds"]
            error = legacy["error"]
            if status != "completed":
                status = "interrupted" if error and error["type"] == "StudyTimeout" else "failed"
                termination = "error"
            else:
                approved = arm != "fixed1" and bool(legacy.get("gate_decision", {}).get("stop")) and not legacy["forced_budget_stop"]
                termination = "fixed_one_draft" if arm == "fixed1" else "budget_cap" if legacy["forced_budget_stop"] else "legacy_gate_stop"
                actions = ["REVISE"] * max(0, iterations - 1) + ["STOP"]
        else:
            state = orchestrator.begin(public, ledger.trace)
            while not state.terminal:
                decision = _choose(arm, state, policy, LinearHead.pretrained(0))
                ledger.trace({"event": "route_decision", "decision": decision})
                actions.append(decision["action"])
                state = orchestrator.transition(state, decision["action"], ledger.trace)
            prediction, iterations, termination = state.draft, state.draft_number, state.terminal_reason
            approved = termination == "policy_stop" and state.verifier_score >= .8
    except BaseException as exc:
        status = "interrupted" if _interrupted(exc) else "failed"
        termination, error = "error", {"type": getattr(exc, "error_type", type(exc).__name__), "message": str(exc)}
        if isinstance(exc, OrchestrationError) and exc.state is not None:
            state = exc.state
        if state is not None:
            prediction, iterations = state.draft, state.draft_number
    costs = ledger.costs(before)
    record = {"question_id": public.id, "arm": arm, "status": status, "prediction": prediction,
        "metrics": hotpot.score_prediction(prediction, gold) if status == "completed" else {k: 0. for k in hotpot.METRICS},
        "iterations": sum(item["role"] == "writer" for item in ledger.calls[before:]),
        "returned_draft_number": iterations,
        "iteration_scope": "attempted writer generations, including format corrections and failures",
        **costs, "elapsed_seconds": time.monotonic() - started,
        "actions": actions, "termination": termination, "gate_approved": approved,
        "error": error, "critic_calls": len(ledger.critics) - critic_before,
        "primary_answer_selection": "last returned draft", "state": state.to_dict() if state else None,
        "legacy_trajectory": legacy["trajectory"] if legacy else None,
        "budget_comparability": "secondary unmatched six-draft legacy loop" if arm == "legacy_combined6" else
                                "writer-only one draft" if arm == "fixed1" else "matched draft and role-call caps"}
    if best_prefix and status == "completed" and state is not None:
        history = [step for step in state.history if "draft" in step and "verifier_score" in step]
        selected = max(enumerate(history), key=lambda pair: (pair[1]["verifier_score"], -pair[0]))[1]
        record["best_prefix_diagnostic"] = {"is_primary": False, "prediction": selected["draft"],
            "metrics": hotpot.score_prediction(selected["draft"], gold),
            "selection": "maximum observed verifier score, earliest tie; no extra generation; full path cost retained"}
    ledger.trace({"event": "episode_finished", "status": status, "termination": termination})
    return record


def _policy_analysis(output, summary, episodes, checkpoints):
    """Declared diagnostics; no threshold/head/configuration is changed here."""
    examples = _read(output / "training-examples.jsonl")
    labels = {}
    for split in ("train", "dev"):
        records = [row for row in examples if row["split"] == split]
        counts = Counter(max(row["available_actions"], key=lambda action: row["utilities"][action]) for row in records)
        labels[split] = {"questions_with_valid_states": len({row["question_id"] for row in records}),
                         "states": len(records), "target_actions": dict(counts)}
    seeds = []
    for item in checkpoints:
        head = json.loads((output / item["path"]).read_text(encoding="utf-8"))
        seeds.append({"seed": item["seed"], "sha256": item["sha256"],
            "selected_epoch": head["metadata"]["selected_epoch"],
            "selected_dev_utility": head["metadata"]["selected_dev_utility"]})
    observations = []
    inherited = LinearHead.pretrained(0)
    for episode in episodes:
        if not episode.get("state"):
            continue
        history = [row for row in episode["state"]["history"] if "draft" in row]
        for index, step in enumerate(history):
            flag = step["verifier_score"] >= .8
            previous = history[index - 1]["draft"]["answer"] if index else None
            previous_distance = lexical_distance(history[index - 2]["draft"]["answer"], previous) if index >= 2 else None
            probability = inherited.probability(make_features(step["draft"]["answer"], previous,
                index, 3, flag, previous_distance))
            observations.append({"question_id": episode["question_id"], "arm": episode["arm"],
                "draft_number": index + 1, "inherited_p_stop": probability,
                "strands_stop": flag, "inherited_head_stop": probability >= .5,
                "binary_and_stop": flag and probability >= .5})
    comparisons = {}
    learned = summary["arms"]["learned3"]
    for arm in ("strands", "binary_and", "always_revise3"):
        baseline = summary["arms"][arm]
        complete = learned["pending"] == baseline["pending"] == 0
        known_tokens = complete and learned["mean_total_tokens"] is not None and baseline["mean_total_tokens"] is not None
        comparisons[arm] = {"both_panels_fully_attempted": complete,
            "answer_f1_delta_learned_minus_control": learned["metrics"]["f1"] - baseline["metrics"]["f1"] if complete else None,
            "joint_f1_delta_learned_minus_control": learned["metrics"]["joint_f1"] - baseline["metrics"]["joint_f1"] if complete else None,
            "mean_attempted_agent_calls_delta": (learned["attempted_agent_calls"] - baseline["attempted_agent_calls"]) / summary["planned_count"] if complete else None,
            "mean_verifier_calls_delta": learned["mean_critic_calls"] - baseline["mean_critic_calls"] if complete else None,
            "mean_response_tokens_delta": learned["mean_total_tokens"] - baseline["mean_total_tokens"] if known_tokens else None}
    return {"scope": "post-run predeclared descriptive diagnostics; no test-driven tuning or superiority claim",
        "training_label_distribution": labels, "seed_dev_checkpoint_selection": seeds,
        "deployed_actions_by_arm": {arm: value["action_distribution"] for arm, value in summary["arms"].items()},
        "failure_coverage_by_arm": {arm: {key: value[key] for key in ("planned", "attempted", "completed", "failed", "interrupted", "pending")}
                                    for arm, value in summary["arms"].items()},
        "inherited_head_vs_strands": {"scope": "same observed public draft prefixes in matched test trajectories; repeated tasks across arms are not independent samples",
            "checkpoint_seed": 0, "head_threshold": .5, "strands_threshold": .8,
            "observed_states": len(observations),
            "raw_head_vs_strands_disagreements": sum(row["inherited_head_stop"] != row["strands_stop"] for row in observations),
            "AND_vs_strands_disagreements": sum(row["binary_and_stop"] != row["strands_stop"] for row in observations),
            "observations": observations},
        "learned_vs_matched_controls": comparisons}


def _versions():
    result = {}
    for name in ("strands-agents", "transformers", "strands-decider", "torch", "datasets"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _write_outputs(output, summary, episodes):
    import csv
    import io
    _json(output / "summary.json", summary)
    stream = io.StringIO(newline="")
    columns = ["question_id", "arm", "status", *hotpot.METRICS, "iterations", "attempted_agent_calls",
               "total_tokens", "elapsed_seconds", "termination", "gate_approved", "critic_calls"]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for episode in episodes:
        writer.writerow({key: (episode["metrics"][key] if key in hotpot.METRICS else episode.get(key)) for key in columns})
    (output / "results.csv").write_text(stream.getvalue(), encoding="utf-8")
    lines = ["YASH - NOVEMBER STUDY RECEIPT", "", "Status: " + summary["study_status"],
        "Data mode: " + summary["data_mode"],
        "These are this run's recorded outputs, not results claimed by the video paper.",
        "Every planned question remains in the metric denominator; pending attempts are not completed simulations.",
        "Primary answer selection: last returned draft. Budget caps are not gate approval.",
        "Primary learned comparisons: matched strands, binary_and, always_revise3. Legacy six-round control has a different budget.",
        "", "Training and physical accounting", json.dumps(summary["collection"], indent=2),
        "Physical collection calls are separate from shared-prefix counterfactual branch path costs.",
        "Tokens describe correlated writer/reviewer/challenger responses. Critic/reranker calls and time are separate; hardware compute is not free.",
        "", "Held-out arms (failure-inclusive)"]
    for arm, value in summary["arms"].items():
        lines.append(f"{arm}: {value['completed']}/{value['planned']} valid; failed={value['failed']}; interrupted={value['interrupted']}; pending={value['pending']}; answer F1={value['metrics']['f1']:.6f}; joint F1={value['metrics']['joint_f1']:.6f}; calls={value['attempted_agent_calls']}; tokens={value['total_tokens']}")
    lines += ["", "Best-prefix figures, when present in episodes, are same-candidate diagnostics and do not replace primary scores.",
        "No statistical superiority, synergy, calibrated-probability or total-compute-saving claim follows from this small study.",
        "", "Policy analysis (descriptive, no test-driven tuning)",
        json.dumps({key: value for key, value in summary["policy_analysis"].items() if key != "inherited_head_vs_strands"}, indent=2),
        "Inherited AND versus Strands on the same observed draft prefixes: " + str(summary["policy_analysis"]["inherited_head_vs_strands"]["AND_vs_strands_disagreements"])
        + " disagreements across " + str(summary["policy_analysis"]["inherited_head_vs_strands"]["observed_states"]) + " states (not independent tasks).",
        "Full public state comparisons are in policy-analysis.json.",
        "", "Journal integrity", json.dumps(summary["journal_integrity"], indent=2),
        "", "Configuration and all five dev-selected head hashes are in study-config.json and freeze.json."]
    (output / "Yash-Nov2-Study-Report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    offline = summary["data_mode"] == "injected_rows_offline_verification"
    figure, axes = plt.subplots(1, 3, figsize=(13, 7.2), dpi=100, sharey=True)
    colors = ["#315e85" if arm != "legacy_combined6" else "#8b919a" for arm in ARMS]
    rows = [summary["arms"][arm] for arm in ARMS]
    panels = [("Answer F1", [row["metrics"]["f1"] for row in rows], "Score (all planned questions)", ".3f"),
              ("Attempted role calls", [row["attempted_agent_calls"] / row["planned"] for row in rows], "Mean per planned question", ".1f"),
              ("Writer generations", [row["mean_writer_iterations"] for row in rows], "Mean per planned question", ".1f")]
    for index, (axis, (title, values, xlabel, number_format)) in enumerate(zip(axes, panels)):
        bars = axis.barh(range(len(ARMS)), values, color=colors, height=.55)
        maximum = max(1., max(values, default=0.))
        axis.set_xlim(0, maximum * 1.24)
        axis.set_title(title, fontsize=12, loc="left", pad=15)
        axis.set_xlabel(xlabel, fontsize=9)
        axis.grid(axis="x", color="#e2e8ef", linewidth=.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=9, length=0, pad=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(value + maximum * .025, bar.get_y() + bar.get_height() / 2,
                      format(value, number_format), ha="left", va="center", fontsize=9, color="#172335")
        if index == 0:
            labels = [f"{arm}\nvalid {row['completed']}/{row['planned']}; pending {row['pending']}"
                      for arm, row in zip(ARMS, rows)]
            axis.set_yticks(range(len(ARMS)), labels)
    axes[0].invert_yaxis()
    title = "SYNTHETIC OFFLINE VERIFICATION - no benchmark inference" if offline else "Yash November study - paired held-out comparison"
    figure.suptitle(title, x=.025, y=.965, ha="left", fontsize=15, weight="bold", color="#172335")
    figure.text(.025, .91, "Status: " + summary["study_status"] + "; " + str(summary["planned_count"]) + " planned questions per arm", fontsize=10, color="#334155")
    figure.text(.025, .065, "Last returned draft. Failed/interrupted/pending score zero. Gray legacy6 uses a different draft budget.", fontsize=9, color="#334155")
    figure.text(.025, .03, "Calls and writer generations include failed attempts/corrections; pending resources are unknown. Verifier/reranker work is reported separately.", fontsize=8.5, color="#334155")
    figure.subplots_adjust(left=.215, right=.965, top=.82, bottom=.16, wspace=.3)
    figure.savefig(output / "Yash-Nov2-Answer-F1.png", dpi=100, facecolor="white")
    plt.close(figure)


def run_study(agents, critic, plan, rows, output, *, seed=0, max_drafts=3,
              max_agent_calls=10, timeout_seconds=27000, provenance=None,
              role_agents=None, data_mode="injected_rows_offline_verification",
              policy_config=None, best_prefix_diagnostic=True):
    """Collect, fit/freeze, and evaluate once; existing study outputs are refused.

    Supply ``role_agents`` for injected fakes. In production it is created from
    original Agents. This function never starts a model server or resumes a run.
    The CLI's runtime owns inference only after the user manually starts it.
    """
    from .nov2_policy import LinearPolicy, PolicyConfig, fit_policy
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("Deployment seed must be predeclared zero or replication one")
    if type(max_drafts) is not int or max_drafts != 3:
        raise ValueError("The named matched arms require exactly three drafts")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("A positive finite wall-clock allowance is required")
    if data_mode not in {"pinned_dataset_live", "injected_rows_offline_verification"}:
        raise ValueError("Declare pinned live data or injected offline verification")
    dataset = nov2_protocol.resolve_plan(plan, rows)
    if any(not dataset[name] for name in nov2_protocol.PARTITIONS):
        raise ValueError("The study requires nonempty train, dev and test partitions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [name for name in output.iterdir() if name.name != "runtime"]
    if artifacts:
        raise FileExistsError("Study output already contains artifacts; overwrite and resume are refused")
    # Exclusive claim prevents two callers from appending to the same primary journals.
    with (output / "study-claim.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema": "yash-nov2-study-v1", "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
    for name in ("episodes.jsonl", "traces.jsonl", "training-branches.jsonl", "training-states.jsonl",
                 "training-examples.jsonl", "physical-calls.jsonl", "critic-calls.jsonl"):
        (output / name).touch(exist_ok=False)
    role_agents = role_agents if role_agents is not None else RoleAgents.from_agents(agents)
    ledger = _Ledger(output, time.monotonic() + timeout_seconds, data_mode)
    tracked = _TrackedAgents(role_agents, ledger)
    verified = _TrackedCritic(critic, ledger)
    orchestrator = Orchestrator(tracked, verified, max_drafts=max_drafts, max_agent_calls=max_agent_calls, format_corrections=1)
    source = {path.name: _sha(path) for path in Path(__file__).parent.glob("*.py")}
    config = {"schema": "yash-nov2-study-v1", "plan": plan, "deployment_seed": seed,
        "all_fitted_seeds": [0, 1, 2, 3, 4], "seed_selection": "dev only; seed0 primary, seed1 predeclared replication",
        "max_drafts": max_drafts, "max_agent_calls": max_agent_calls, "format_corrections": 1,
        "timeout_seconds": timeout_seconds, "data_mode": data_mode, "arms": list(ARMS),
        "primary_comparators": ["strands", "binary_and", "always_revise3"],
        "legacy_combined6": "historical function/Gate, six drafts, no format retry; shared current transport/output caps; unmatched secondary",
        "primary_answer_selection": "last returned draft", "best_prefix_diagnostic": best_prefix_diagnostic,
        "utility": "one-transition answer F1 - .02*incremental attempted LLM calls/10",
        "state_collection": "initial state and every successful non-STOP depth1 child; all available one-step action branches",
        "policy_config": asdict(policy_config or PolicyConfig()), "source_sha256": source,
        "package_versions": _versions(), "writer": getattr(role_agents, "provenance", {}),
        "critic": getattr(critic, "provenance", {}), "inherited_head": LinearHead.pretrained(0).provenance,
        "provenance": provenance or {}}
    _json(output / "study-config.json", config)
    episodes, failure, stage, fit, checkpoints = [], None, "train", None, []

    def checkpoint(status):
        _json(output / "status.json", {"stage": stage, "status": status,
            "recorded_test_attempts": len(episodes),
            "valid_test_predictions": sum(row["status"] == "completed" for row in episodes),
            "current": dict(ledger.context), "planned_test_attempts": len(dataset["test"]) * len(ARMS),
            "recorded_agent_invocations": len(ledger.calls), "recorded_critic_calls": len(ledger.critics)})

    ledger.progress = lambda: checkpoint("running")
    checkpoint("running")
    try:
        train = _collect_partition("train", dataset["train"], orchestrator, ledger)
        stage = "dev"
        checkpoint("running")
        dev = _collect_partition("dev", dataset["dev"], orchestrator, ledger)
        ledger.check()
        stage = "fit_and_freeze"
        checkpoint("running")
        fit = fit_policy(train, dev, output / "heads", config=policy_config or PolicyConfig(),
                         provenance={"plan_sha256": _sha(output / "study-config.json"), "source_sha256": source})
        checkpoints = [{**item, "path": str(Path("heads") / item["path"])} for item in fit["checkpoints"]]
        if {item["seed"] for item in checkpoints} != set(range(5)) or len(checkpoints) != 5:
            raise ValueError("All five predeclared dev-selected checkpoints must freeze before test")
        for item in checkpoints:
            if _sha(output / item["path"]) != item["sha256"]:
                raise ValueError("Fitted checkpoint hash changed")
        freeze = {"schema": "yash-nov2-freeze-v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _sha(output / "study-config.json"), "checkpoints": checkpoints,
            "deployment_seed": seed, "test_calls_at_freeze": sum(row.get("stage") == "test" for row in ledger.calls),
            "physical_collection_invocations": len(ledger.calls), "physical_collection_critic_calls": len(ledger.critics)}
        _json(output / "freeze.json", freeze)
        policy = LinearPolicy.from_json(output / next(item["path"] for item in checkpoints if item["seed"] == seed))
        stage = "test"
        checkpoint("running")
        for public, gold in dataset["test"]:
            offset = int(hashlib.sha256(f"{seed}:{public.id}:arm-order".encode()).hexdigest()[:8], 16) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                ledger.check()
                ledger.context = {"stage": "test", "question_id": public.id, "arm": arm}
                episode = _episode(public, gold, arm, orchestrator, tracked, verified, policy, ledger,
                                   best_prefix=best_prefix_diagnostic)
                # This scope alone receives labels. No action policy sees gold or metrics.
                _append(output / "episodes.jsonl", episode)
                episodes.append(episode)
                checkpoint("running")
                if episode["status"] == "interrupted":
                    raise StudyTimeout("Held-out attempt interrupted")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        stage = "interrupted" if _interrupted(exc) else "failed"
    # Validate exact membership, not merely counts, before presenting any metrics.
    planned = {(public.id, arm) for public, _ in dataset["test"] for arm in ARMS}
    saved = _read(output / "episodes.jsonl")
    if saved != episodes or any((row["question_id"], row["arm"]) not in planned for row in saved):
        raise ValueError("Primary episodes do not match this run's frozen membership")
    for name in ("training-branches.jsonl", "training-states.jsonl", "training-examples.jsonl"):
        _read(output / name)
    traces = _read(output / "traces.jsonl")
    calls, critics = _read(output / "physical-calls.jsonl"), _read(output / "critic-calls.jsonl")
    if calls != ledger.calls or critics != ledger.critics:
        raise ValueError("Primary physical journals diverged from recorded invocations")
    if (sum(row.get("event") == "agent_invocation_started" for row in traces) != len(calls)
            or sum(row.get("event") == "critic_invocation_started" for row in traces) != len(critics)):
        raise ValueError("Started calls lack matching durable invocation receipts")
    summary = nov2_protocol.summarize(saved, ARMS, len(dataset["test"]))
    collection = [item for item in calls if item.get("stage") in {"train", "dev"}]
    collection_critics = [item for item in critics if item.get("stage") in {"train", "dev"}]
    branches = _read(output / "training-branches.jsonl")
    states = _read(output / "training-states.jsonl")
    initial_coverage = {}
    for name in ("train", "dev"):
        complete_ids = {item["question_id"] for item in states if item["stage"] == name and item["state_id"] == "initial"}
        failed = [item for item in branches if item["stage"] == name and item["action"] == "INITIAL"]
        attempted_ids = complete_ids | {item["question_id"] for item in failed}
        initial_coverage[name] = {"planned": len(dataset[name]), "attempted": len(attempted_ids),
            "valid_initial_states": len(complete_ids), "failed": sum(item["status"] == "failed" for item in failed),
            "interrupted": sum(item["status"] == "interrupted" for item in failed),
            "pending": len(dataset[name]) - len(attempted_ids)}
    unknown = sum(item["total_tokens"] is None for item in calls)
    summary.update({"schema": "yash-nov2-results-v1", "data_mode": data_mode,
        "study_status": "completed" if summary["all_episodes_succeeded"] and failure is None else
                        "completed_with_failed_attempts" if summary["complete"] and failure is None else stage,
        "failure": failure, "deployment_seed": seed,
        "collection": {"planned_questions": {name: len(dataset[name]) for name in ("train", "dev")},
            "initial_question_coverage": initial_coverage,
            "branch_outcomes": len(branches), "branch_statuses": dict(Counter(item["status"] for item in branches)),
            "physical_agent_invocations": len(collection), "physical_critic_calls": len(collection_critics),
            "physical_tokens": sum(item["total_tokens"] for item in collection) if all(item["total_tokens"] is not None for item in collection) else None,
            "scope": "actual new collection invocations; cloned ancestral path calls counted once"},
        "journal_integrity": {"critical_jsonl_valid": True, "agent_invocations": len(calls),
            "critic_invocations": len(critics), "unknown_response_token_receipts": unknown,
            "receipt_statuses": dict(Counter(item["response_receipt"]["status"] for item in calls)),
            "physical_response_ids": len(ledger.response_ids), "all_response_tokens_reconciled": unknown == 0 and data_mode == "pinned_dataset_live",
            "cost_scope": "writer/reviewer/challenger response tokens; critic/reranker compute/time separate"},
        "fitted_checkpoints": checkpoints,
        "primary_answer_selection": "last returned draft", "best_prefix_is_primary": False,
        "statistical_scope": "paired small pilot; no superiority or total-compute-savings claim"})
    for arm, value in summary["arms"].items():
        arm_rows = [row for row in saved if row["arm"] == arm]
        value["critic_calls"] = sum(row["critic_calls"] for row in arm_rows)
        value["mean_critic_calls"] = value["critic_calls"] / len(dataset["test"])
    summary["policy_analysis"] = _policy_analysis(output, summary, saved, checkpoints)
    _json(output / "policy-analysis.json", summary["policy_analysis"])
    _write_outputs(output, summary, saved)
    checkpoint(summary["study_status"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout-seconds", type=float, default=27000)
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--model-variant", default="1p5b")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = json.loads(args.data_json.read_text(encoding="utf-8")) if args.data_json else hotpot.load_validation()
    # Validate before starting the explicitly manual inference runtime.
    nov2_protocol.resolve_plan(plan, rows)
    from .nov2_runtime import Runtime
    with Runtime(args.output / "runtime", model_variant=args.model_variant) as (agents, critic):
        summary = run_study(agents, critic, plan, rows, args.output, seed=args.seed,
            timeout_seconds=args.timeout_seconds,
            provenance={"runtime": json.loads((args.output / "runtime/runtime.json").read_text())}
                       if (args.output / "runtime/runtime.json").exists() else None,
            data_mode="injected_rows_offline_verification" if args.data_json else "pinned_dataset_live")
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
