"""Fresh-episode provenance checks using saved evidence only, never a model.

Logical callbacks bind either a completed physical response or its explicit
zero-call reuse. Native Tau replay uses the recorded callback results; it does
not regenerate answers, critic probabilities, embeddings, or head logits.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .journal import CallJournal, fingerprint
from .qualification import native_episode, probability, require, safe_tensors, sha, within


class AuditIntegrityError(RuntimeError):
    """Must escape the agent's deliberate ValueError-to-policy-failure handling."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def payload(state, feedback, *, review=False, config=None):
    # Reconstruct the selected owner's exact request. Original qualification
    # keeps its legacy profile; future main uses the declared shared profile.
    from .generation import generation_payload
    return generation_payload(config or {}, state, feedback, review=review)


def parsed_response(raw, *, review):
    """Repeat only the runtime's output validation, including known failures."""
    from .generation import parse_response
    return parse_response(raw, review=review)


class Evidence:
    """One sequential compare journal, with exact logical-use accounting."""
    def __init__(self, run, config, learning):
        self.run, self.config, self.learning = Path(run), config, learning
        self.journal = CallJournal(self.run / "compare/physical")
        self.summary = self.journal.verify()
        require(not self.summary["pending_or_uncertain"], "Fresh physical work is unresolved")
        self.pending = {r["id"]: r for r in self.journal.records if r["event"] == "pending"}
        self.completed, self.values = {}, {}
        for record in self.journal.records:
            if record["event"] == "completed":
                self.completed[record["id"]] = record
                self.values[record["id"]] = read(self.journal.directory / record["response"])
        self.generations = [r for r in self.journal.records if r["event"] in {"completed", "exact_response_reuse"}
                            and r["role"] in {"writer", "reviewer"}]
        self.gates = [r for r in self.journal.records if r["event"] == "completed"
                      and r["role"] in {"native_answer", "native_evidence", "cached_head_answer", "cached_head_evidence"}]
        self.generation_offset = self.gate_offset = 0
        self.used_retrievals, self.used_native, self.feature_checks, self.labels = set(), set(), set(), []
        self.logical = Counter()
        self.known_pre_dispatch_failures = Counter()
        self.released_sha = sha(Path(config["checkpoint"]) / "head.safetensors")
        self.trained_path = self.run / "train/learned/answer-head.safetensors"
        self.trained_sha = sha(self.trained_path)
        require(self.released_sha == learning["released_head_sha256"]
                and self.trained_sha == learning["criteria"]["answer"]["trained_head_sha256"],
                "Fresh heads differ from the actual learning proof")

    def rank(self, public, draft=""):
        query = {"question": public["question"], "draft": draft, "paragraphs": [
            {"title": title, "sentences": sentences} for title, sentences in public["context"]]}
        matches = [r for key, r in self.completed.items() if r["role"] == "retrieval"
                   and self.pending[key]["request_sha256"] == fingerprint(query)]
        require(len(matches) == 1, "Public retrieval lacks one exact completed original query")
        event = matches[0]
        value = self.values[event["id"]]
        scores = value["scores"]
        require(len(scores) == len(public["context"]) and value["pair_count"] == len(scores)
                and value["physical_forwards"] == 1
                and all(type(s) in (float, int) and math.isfinite(s) for s in scores), "Invalid original retrieval result")
        self.used_retrievals.add(event["id"])
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]

    def generation(self, state, feedback, event, public, *, review):
        request = payload(state, feedback, review=review, config=self.config)
        role = "reviewer" if review else "writer"
        self.logical[role] += 1
        overflow = sum(len(m["content"].encode("utf-8")) for m in request["messages"]) + 1536 > 16384
        if overflow:
            require(event["status"] == "failed", "Successful generation exceeds its pre-dispatch bound")
            self.known_pre_dispatch_failures[role] += 1
            return
        require(self.generation_offset < len(self.generations), "Logical generation has no physical/reuse evidence")
        call = self.generations[self.generation_offset]
        self.generation_offset += 1
        require(call["role"] == role and self.pending[call["id"]]["request_sha256"] == fingerprint(request),
                "Fresh writer/reviewer request differs from its public input/feedback")
        if call["event"] == "exact_response_reuse":
            require(self.config.get("response_cache") is True and call["new_physical_calls"] == 0,
                    "Undeclared generation reuse or invented physical expense")
        try:
            value = parsed_response(self.values[call["id"]], review=review)
        except ValueError:
            require(event["status"] == "failed", "Invalid physical output was recorded as successful")
            return
        require(event["status"] != "failed", "A known provider failure has a valid completed output")
        if not review and event["status"] == "invalid_submission":
            from .domain import validate_submission
            try:
                validate_submission(value, public)
            except ValueError:
                return
            require(False, "A valid physical answer was recorded as an invalid citation/schema")
        expected = event["feedback"] if review else event["draft"]
        require(value == expected, "Logical prediction/feedback differs from its actual response")

    def public_state(self, public, draft, criterion):
        indices = self.rank(public, draft["answer"])
        evidence = "\n".join(public["context"][i][0] + ": " + " ".join(public["context"][i][1]) for i in indices)
        state = f"Question: {public['question']}\nEvidence: {evidence}\nDraft answer: {draft['answer']}"
        if criterion == "evidence":
            state += "\nCitations: " + json.dumps(draft["supporting_facts"], ensure_ascii=False)
        return state, indices

    def feature(self, value, state, criterion, indices):
        from .hobson import INSTRUCTIONS_BY_CRITERION
        feature = value["feature"]
        path = within(feature["path"], self.run / "compare/features")
        require(sha(path) == feature["sha256"], "Fresh native feature bytes changed")
        provenance = feature["provenance"]
        require(feature["schema"] == "haltiq-hobson-pointer-feature-v1" and feature["criterion"] == criterion
                and feature["option_order"] == ["false", "true"]
                and feature["encoder_sha256"] == self.learning["encoder_sha256"]
                and fingerprint(provenance) == feature["provenance_sha256"]
                and provenance["criterion"] == criterion and provenance["state_sha256"] == fingerprint(state)
                and provenance["instructions_sha256"] == fingerprint(INSTRUCTIONS_BY_CRITERION[criterion])
                and provenance["source_sha256"] == sha(Path(__file__).with_name("hobson.py"))
                and value["retrieval_indices"] == indices, "Fresh feature is not bound to this public state/criterion")
        if feature["sha256"] not in self.feature_checks:
            safe_tensors(path, {"decide": [2048], "options": [2, 2048]})
            self.feature_checks.add(feature["sha256"])

    def gate(self, public, draft, event, *, arm, gold):
        criterion = event["criterion"]
        require(criterion == "answer" or (criterion == "evidence" and arm == "decomposed"), "Wrong gate criterion/arm")
        state, indices = self.public_state(public, draft, criterion)
        self.logical["gate_" + criterion] += 1
        if event["status"] == "failed":
            # Native token/window validation is pre-dispatch. A failure is zero
            # quality, not an invented score/forward. Unused real calls fail finish().
            self.known_pre_dispatch_failures["gate_" + criterion] += 1
            return
        require(self.gate_offset < len(self.gates), "Completed gate lacks a real scored response")
        call = self.gates[self.gate_offset]
        self.gate_offset += 1
        value = self.values[call["id"]]
        require(value == event["score"], "Gate trace differs from its actual head/native response")
        trained = arm == "trained"
        expected_head = self.trained_path if trained else None
        expected_sha = self.trained_sha if trained else self.released_sha
        temperature = (self.learning["released_evidence_calibration"]["temperature"] if criterion == "evidence"
            else self.learning["criteria"]["answer"][("trained" if trained else "released") + "_calibration"]["temperature"])
        require(value["temperature"] == temperature, "Gate used the wrong calibrated head temperature")
        if call["role"] == "native_" + criterion:
            request = {"state": state, "criterion": criterion,
                       "head": str(expected_head) if expected_head else "released", "temperature": temperature}
            require(self.pending[call["id"]]["request_sha256"] == fingerprint(request), "Native gate head/state/request mismatch")
            probability(value, native=True)
        else:
            require(call["role"] == "cached_head_" + criterion and self.config.get("native_feature_cache") is True,
                    "Unexpected cached gate role")
            original_id = value["original_native_call_id"]
            original_event = self.completed.get(original_id, {})
            original = self.values.get(original_id)
            require(original_event.get("role") == "native_" + criterion and original is not None
                    and original_id in self.used_native
                    and value["original_native_response_sha256"] == original_event["response_sha256"]
                    and value["feature"] == original["feature"] and value["cache_reused"] is True,
                    "Cached head lacks its original genuine native response")
            original_request = self.pending[original_id]
            # The original selected head can differ: cached encoder features are
            # head-independent. Validate its own selected head/T via its trace.
            require(original_request["role"] == "native_" + criterion, "Wrong native cache ancestor")
            self.feature(original, state, criterion, indices)
            probability(original, native=True)
            request = {"original_native_call_id": original_id, "feature_sha256": value["feature"]["sha256"],
                       "head_sha256": expected_sha, "criterion": criterion, "temperature": temperature}
            require(self.pending[call["id"]]["request_sha256"] == fingerprint(request), "Cached gate head/request mismatch")
            probability(value, native=False)
        self.feature(value, state, criterion, indices)
        if call["role"] == "native_" + criterion:
            self.used_native.add(call["id"])
        from .scoring import score
        self.labels.append({"question_id": public["id"], "arm": arm, "criterion": criterion,
            "label": int(score(draft, gold)["em" if criterion == "answer" else "sp_em"]),
            "physical_call_id": call["id"]})

    def finish(self):
        require(self.generation_offset == len(self.generations) and self.gate_offset == len(self.gates),
                "Actual physical/reuse operations were not consumed by the native episodes")
        retrieval_ids = {k for k, r in self.completed.items() if r["role"] == "retrieval"}
        require(retrieval_ids == self.used_retrievals, "A physical retrieval has no matching public logical query")
        allowed = {"retrieval", "writer", "reviewer", "writer_load", "writer_unload", "native_startup",
                   "native_answer", "native_evidence", "cached_head_answer", "cached_head_evidence"}
        require(set(self.summary["by_role"]) <= allowed, "Undeclared fresh physical operation")
        for identifier, event in self.completed.items():
            if event["role"] in {"writer_load", "writer_unload"}:
                require(self.pending[identifier]["request_sha256"] == fingerprint({"model": "qwen3:8b"}), "Wrong writer placement identity")
            if event["role"] == "native_startup":
                require(self.pending[identifier]["request_sha256"] == fingerprint({"checkpoint": self.config["checkpoint"]}),
                        "Wrong native placement identity")
                metadata = self.values[identifier]["metadata"]
                require(metadata["released_head_sha256"] == self.released_sha
                        and metadata["encoder_sha256"] == self.learning["encoder_sha256"]
                        and metadata["head_parameters"] == 1053184, "Native startup model binding changed")


def replay(row, case, evidence, *, initial=False):
    """Reuse the pilot's strict native replay, capturing exact logical inputs."""
    from . import agent, domain
    callbacks = iter([event for event in row["trace"] if event["event"] in {"draft", "review", "gate"}])
    def next_event(kind):
        event = next(callbacks)
        require(event["event"] == kind, "Callback event order changed")
        return event
    def traced(case_arg, writer, reviewer, gate, **kwargs):
        def write(state, feedback):
            try:
                event = next_event("draft")
                if not initial and state["previous_draft"] is None:
                    evidence.logical["shared_initial_writer_zero_new_calls"] += 1
                    require(event.get("draft") == case["initial"] if case["initial"] is not None else event["status"] == "failed",
                            "An arm did not reuse its genuine first draft/failure")
                else:
                    evidence.generation(state, feedback, event, case["public"], review=False)
            except Exception as error:
                raise AuditIntegrityError("Writer evidence failed: " + str(error)) from error
            return writer(state, feedback)
        def review(state, proposed):
            try:
                event = next_event("review")
                indices = evidence.rank(case["public"], proposed["answer"])
                view = {"question": case["public"]["question"], "evidence": [
                    {"title": case["public"]["context"][i][0], "sentences": [
                        {"index": n, "text": text} for n, text in enumerate(case["public"]["context"][i][1])]}
                    for i in indices], "draft": proposed}
                evidence.generation(view, "", event, case["public"], review=True)
            except Exception as error:
                raise AuditIntegrityError("Reviewer evidence failed: " + str(error)) from error
            return reviewer(state, proposed)
        def decide(state, proposed, criterion):
            try:
                event = next_event("gate")
                require(event["criterion"] == criterion, "Gate order changed")
                evidence.gate(case["public"], proposed, event, arm=row["arm"], gold=case["evaluator"])
            except Exception as error:
                raise AuditIntegrityError("Gate evidence failed: " + str(error)) from error
            return gate(state, proposed, criterion)
        return agent.run_episode(case_arg, write, review, decide, **kwargs)
    helpers = {"haltiq_tau_hotpot.agent": SimpleNamespace(run_episode=traced), "haltiq_tau_hotpot.domain": domain}
    native_episode(row, case, row["passage_indices"], helpers, arm="fixed1" if initial else row["arm"],
                   max_drafts=1 if initial else 10)
    require(next(callbacks, None) is None, "Unconsumed callback trace")


def verify_question_owners(run, state, cases, records):
    """Bind each serial child, fresh admission and disjoint journal slice."""
    from .protocol import ARMS
    from .main_question import validate_process_binding
    import sys
    terminal = read(run / "compare/runtime-terminal.json")
    plan = read(run / "plan.json")
    def summary(prefix):
        pending = {e["id"] for e in prefix if e["event"] == "pending"}
        completed = {e["id"] for e in prefix if e["event"] == "completed"}
        uncertain = {e["id"] for e in prefix if e["event"] == "uncertain"}
        return {"dispatched": len(pending), "completed": len(completed), "uncertain": len(uncertain),
            "pending_or_uncertain": bool(pending-completed or uncertain),
            "exact_response_reuses": sum(e["event"] == "exact_response_reuse" for e in prefix),
            "by_role": dict(Counter(e["role"] for e in prefix if e["event"] == "completed"))}
    entries = terminal.get("question_receipts", [])
    require(terminal.get("comparison_process_layout") == "per_question" and len(entries) == len(cases),
            "Every question requires its own actual process receipt")
    previous_end, previous_time, child_active, child_idle = 0, None, 0., 0.
    startup, switches, child_pids = 0., 0., []
    for index, (entry, case) in enumerate(zip(entries, cases)):
        qid = case["public"]["id"]
        directory = run / "compare/questions" / f"{index:03d}"
        path = directory / "receipt.json"
        require(entry.get("question_index") == index and entry.get("question_id") == qid
                and Path(entry["path"]).resolve() == path and sha(path) == entry["sha256"]
                and entry.get("owned_runtime_clear") is True, "Question receipt order/path/hash changed")
        receipt, dispatched, exited, started, identity = [read(directory / name) for name in
            ("receipt.json", "dispatch-pending.json", "process-exit.json", "started.json", "identity-verified.json")]
        require(sha(directory / "dispatch-pending.json") == entry["dispatch_sha256"]
                and sha(directory / "process-exit.json") == entry["process_exit_sha256"], "Question dispatch/exit bytes changed")
        require(sha(directory / "identity-verified.json") == entry["identity_sha256"]
                == receipt["identity_sha256"] == exited["identity_sha256"]
                and identity["started_sha256"] == sha(directory / "started.json")
                and identity["dispatch_sha256"] == sha(directory / "dispatch-pending.json")
                and identity.get("schema") == "haltiq-main-question-process-binding-v1"
                and identity.get("verified_before_admission") is True, "Question live identity binding changed")
        executables = list(dict.fromkeys([plan["python"], sys.executable, sys._base_executable]))
        require(identity.get("executables") == executables
                and identity.get("executable_sha256") == {str(Path(p).resolve()): sha(p) for p in executables},
                "Question interpreter identity differs from its frozen runtime")
        validate_process_binding(identity["launcher"], identity["worker"], started, dispatched, identity["observed_at"],
            launcher_pid=exited["launcher_pid"], parent_pid=identity["compare_parent_pid"], executables=executables,
            compare_parent=identity["compare_parent"])
        for value in (receipt, dispatched, exited, started):
            require(value.get("run_id") == state["run_id"] and value.get("plan_sha256") == state["plan_sha256"]
                    and value.get("question_index") == index, "Question belongs to a different current owner")
        expected_argv = [plan["python"], "-B", "-m", "haltiq_tau_hotpot.main_question",
            "--config", plan["main_protocol"]["path"], "--output", str(run / "compare"), "--previous", str(run),
            "--control", str(run / "workers/compare"), "--deadline", plan["deadline"], "--question-index", str(index),
            "--allow-model-calls"]
        require(dispatched.get("argv") == expected_argv and dispatched.get("automatic_retry") is False,
                "Question process was not the exact frozen finite command")
        require(receipt.get("schema") == "haltiq-main-question-v1" and receipt.get("question_id") == qid
                and dispatched.get("question_id") == qid and exited.get("question_id") == qid
                and receipt.get("terminal") is True and receipt.get("cleanup_verified") is True
                and receipt.get("pending_or_uncertain") is False and exited.get("exit_code") == 0
                and exited.get("process_exit_observed") is True
                and type(receipt.get("pid")) is int and receipt["pid"] > 0
                and receipt["pid"] == exited.get("worker_pid") == started.get("pid") == identity["worker"]["pid"]
                and receipt.get("parent_pid") == started.get("parent_pid") == identity["worker"]["parent_pid"]
                and receipt.get("launcher_pid") == exited["launcher_pid"] == identity["launcher"]["pid"]
                and started.get("at") == receipt.get("started_at")
                and receipt.get("source_sha256") == fingerprint(state["source_sha256"])
                and receipt.get("config_sha256") == sha(plan["main_protocol"]["path"]), "Question terminal identity/source changed")
        for role in ("launcher", "worker"):
            cleared = exited["identity_exit_check"][role]
            current, original = cleared["current_identity"], identity[role]
            require(cleared.get("original_identity_absent") is True and cleared["pid"] == original["pid"]
                    and cleared["original_created_at"] == original["created_at"]
                    and (current is None or current["created_at"] != original["created_at"]),
                    "Original owned launcher/worker identity was not cleared")
        begin, end = [datetime.fromisoformat(receipt[key]) for key in ("started_at", "finished_at")]
        outer_begin, outer_end = [datetime.fromisoformat(exited[key]) for key in ("started_at", "finished_at")]
        require(all(t.tzinfo is not None for t in (begin,end,outer_begin,outer_end)) and outer_begin <= begin <= end <= outer_end
                and (previous_time is None or previous_time <= outer_begin), "Question processes overlap or have invalid clocks")
        previous_time = outer_end
        child_pids.append(receipt["pid"])
        for key in ("wall_seconds", "active_seconds", "intentional_idle_seconds"):
            require(type(receipt.get(key)) in (int,float) and math.isfinite(receipt[key]) and receipt[key] >= 0,
                    "Question timing is not finite")
        require(receipt["active_seconds"] + receipt["intentional_idle_seconds"] <= receipt["wall_seconds"] + 2,
                "Question active/idle scopes exceed containing wall")
        child_active += receipt["active_seconds"]
        child_idle += receipt["intentional_idle_seconds"]
        require(sha(directory / "admission.json") == receipt["admission_sha256"]
                and sha(directory / "runtime-terminal.json") == receipt["runtime_terminal_sha256"], "Question resource/runtime evidence changed")
        admitted, runtime = read(directory / "admission.json"), read(directory / "runtime-terminal.json")
        from .admission_wait import verify_wait_history
        verify_wait_history(read(plan["main_protocol"]["path"]), admitted, directory)
        if admitted.get("wait"):
            from .admission_wait import CLOCK_TOLERANCE_SECONDS
            wait = admitted["wait"]
            require(begin <= datetime.fromisoformat(wait["started_at"]) <= datetime.fromisoformat(wait["finished_at"]) <= end
                    and wait["idle_seconds"] <= receipt["intentional_idle_seconds"] + CLOCK_TOLERANCE_SECONDS,
                    "Admission wait escapes its owned child clock or idle accounting")
        require(admitted.get("schema") == "haltiq-main-question-admission-v1" and admitted.get("status") == "PASS_ADMISSION"
                and admitted.get("gpu_lease_released") is True and admitted.get("private_port_closed") is True
                and admitted.get("ram_gib") == 11 and admitted.get("vram_gib") == 9
                and type(admitted.get("model_calls")) is int and admitted.get("model_calls") == 0
                and admitted.get("automatic_retry") is False,
                "Fresh child admission was not actually established")
        samples = admitted.get("samples", [])
        times = [datetime.fromisoformat(sample["at"]) for sample in samples]
        require(len(samples) == 4 and all(t.tzinfo is not None for t in times)
                and begin <= times[0] <= times[-1] <= end and (times[-1]-times[0]).total_seconds() >= 30
                and all((b-a).total_seconds() >= 10 for a,b in zip(times,times[1:])), "Fresh question admission window changed")
        require(begin <= datetime.fromisoformat(identity["observed_at"]) <= times[0],
                "Question admission preceded live parent/worker identity approval")
        for sample in samples:
            values = [sample.get(k) for k in ("available_ram_gib", "free_vram_gib", "gpu_temperature_c")]
            require(all(type(v) in (int,float) and math.isfinite(v) for v in values)
                    and values[0] >= 11 and values[1] >= 9 and 0 <= values[2] < 80,
                    "Question startup reserves were lowered or unavailable")
        cleanup = runtime["cleanup"]
        require(all(cleanup.get(k) is True for k in ("verified", "private_port_closed", "owned_server_exited", "gpu_lease_released"))
                and not cleanup["errors"] and all(u["allocated_bytes"] == u["reserved_bytes"] == 0 for u in cleanup["native_unloads"]),
                "Question owned-runtime cleanup did not succeed")
        start, stop = receipt["physical_record_start"], receipt["physical_record_end"]
        require(type(start) is int and type(stop) is int and start == previous_end <= stop <= len(records)
                and receipt["physical_record_prefix_sha256"] == fingerprint(records[:start])
                and receipt["physical_record_slice_sha256"] == fingerprint(records[start:stop]), "Question physical slice overlaps or changed")
        roles = [r["role"] for r in records[start:stop] if r["event"] == "pending"]
        require(roles.count("writer_load") <= 1 and roles.count("native_startup") <= 1
                and ("native_startup" not in roles or "writer_load" not in roles[roles.index("native_startup")+1:]),
                "Question exceeded its physical placement envelope")
        require(runtime["active_seconds"] == receipt["active_seconds"]
                and runtime["intentional_idle_seconds"] == receipt["intentional_idle_seconds"]
                and runtime["calls"] == summary(records[:stop]) and not runtime["calls"]["pending_or_uncertain"],
                "Question cumulative journal/control totals changed")
        call_times = [datetime.fromisoformat(e["at"]) for e in records[start:stop]]
        require(all(t.tzinfo is not None and times[-1] <= t <= end for t in call_times)
                and call_times == sorted(call_times), "Question physical operation precedes admission or escapes its clock")
        require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in runtime["timings"].values()),
                "Invalid question startup/switch timing")
        startup += runtime["timings"]["startup_seconds"]
        switches += runtime["timings"]["switch_seconds"]
        expected_outputs = {"initial/" + qid + ".json"} | {"episodes/" + qid + "-" + arm + ".json" for arm in ARMS}
        require(set(receipt["outputs_sha256"]) == expected_outputs, "Question source outcome set is incomplete")
        for relative, expected_hash in receipt["outputs_sha256"].items():
            require(sha(within(run / "compare" / relative, run / "compare")) == expected_hash, "Question source outcome bytes changed")
        previous_end = stop
    require(previous_end == len(records) and terminal["calls"] == summary(records) and terminal["active_seconds"] == child_active
            and terminal["intentional_idle_seconds"] == child_idle
            and terminal["timings"] == {"startup_seconds": startup, "switch_seconds": switches},
            "Question aggregate double-counts cumulative work or omits physical records")
    return {"verified": True, "questions": len(cases), "processes": len(child_pids),
            "physical_record_slices": len(cases), "timing_scope": "Disjoint child control totals; cumulative physical journal counted once"}


def verify_compare(args, context, episodes, initial):
    """Validate every planned fresh episode against its real physical evidence."""
    from haltiq_supervisor.engine import verify_completion
    from .protocol import ARMS
    run, config = Path(args.previous).resolve(), context["config"]
    state, plan = read(run / "supervisor.json"), read(run / "plan.json")
    stage = next(s for s in plan["stages"] if s["id"] == "compare")
    owner = state["stages"]["compare"]
    require(owner["status"] == "completed" and owner["owned_tree_clear"] is True
            and owner["os_handoff_verified"] is True and owner["exit_code"] == 0, "Fresh compare owner is not terminal and clear")
    for path, expected in owner.get("output_sha256", {}).items():
        require(sha(within(path, run)) == expected, "Bound compare output changed")
    completion = read(run / "compare/completion.json")
    verify_completion(stage, state, completion)
    cases = context["cohorts"]["main_fresh"]
    ids = [c["public"]["id"] for c in cases]
    require(len(ids) == context["fresh_n"] and set(initial) == set(ids), "Missing/foreign fresh initial outcome")
    require([(e["question_id"], e["arm"]) for e in episodes] == [(qid, arm) for qid in ids for arm in ARMS],
            "Every planned fresh question/arm must have one native outcome")
    require(episodes == read(run / "compare/episodes.json") and initial == read(run / "compare/initial-drafts.json"),
            "Fresh audit arguments differ from saved source outcomes")
    if config.get("purpose") == "main_study":
        require(config.get("cap_fallback") == "last_valid"
                and all(row.get("cap_fallback") == "last_valid" for row in episodes),
                "Main policy must preserve its predeclared last-valid draft-cap behavior")
    evidence = Evidence(run, config, read(run / "train/learning.json"))
    question_owners = None
    if config.get("compare_process_layout") == "per_question":
        question_owners = verify_question_owners(run, state, cases, evidence.journal.records)
    for case in cases:
        qid = case["public"]["id"]
        first = read(run / "compare/initial" / (qid + ".json"))
        if config.get("purpose") == "main_study":
            require(first.get("cap_fallback") == "last_valid", "Main initial control belongs to a different cap policy")
        require(first["submitted"] == initial[qid] and first["passage_indices"] == evidence.rank(case["public"]),
                "Initial native answer/retrieval changed")
        replay(first, case, evidence, initial=True)
        for row in [e for e in episodes if e["question_id"] == qid]:
            require(row == read(run / "compare/episodes" / (qid + "-" + row["arm"] + ".json"))
                    and row["passage_indices"] == first["passage_indices"]
                    and row["shared_initial_draft_sha256"] == fingerprint(initial[qid])
                    and row["shared_initial_physical_calls"] == 0, "Fresh row/shared first-draft binding changed")
            replay(row, {**case, "initial": initial[qid]}, evidence)
    evidence.finish()
    completed = sum(e["status"] == "completed" for e in episodes)
    require(completion["planned_units"] == len(ids) * len(ARMS) and completion["completed_units"] == completed
            and completion["failed_units"] == len(episodes) - completed and completion["not_run_units"] == 0,
            "Fresh completion counts differ from genuine native outcomes")
    return {"verified": True, "planned_questions": len(ids), "planned_episodes": len(episodes),
        "completed_episodes": completed, "failed_policy_episodes": len(episodes) - completed,
        "native_initial_and_arm_replays": len(ids) + len(episodes), "physical": evidence.summary,
        "logical_callbacks": dict(evidence.logical), "known_pre_dispatch_failures": dict(evidence.known_pre_dispatch_failures),
        "gate_label_bindings_sha256": fingerprint(evidence.labels), "compare_journal_sha256": sha(evidence.journal.path),
        "episodes_sha256": sha(run / "compare/episodes.json"), "initial_sha256": sha(run / "compare/initial-drafts.json"),
        "new_encoder_forwards": 0, "new_head_scores": 0,
        "question_processes": question_owners,
        "scope": "Saved physical evidence and deterministic native replay; no new inference or quality-gain claim"}
