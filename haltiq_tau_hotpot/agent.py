"""Bounded writer/reviewer/decision teams inside real Tau2 HotpotQA episodes.

This module performs no model loading. Provider callbacks own their physical
journals, resource admission, deadlines and any real inference/training.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import time

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator_base import HalfDuplexUser

from .domain import DOMAIN, POLICY, TAU_REVISION, get_environment, json_copy, make_task, register_domain, validate_case, validate_indices, validate_submission, validate_task
from .scoring import OFFICIAL_REVISION, OFFICIAL_SHA256, OFFICIAL_URL, score_submission


ABSTAIN = "No accepted answer could be submitted within the declared episode limits."
FIXED = {f"fixed{i}": i for i in (1, 2, 3, 5, 10)}
GATED = {"released", "trained", "decomposed", "released_submission_gate",
         "trained_submission_gate", "separate_answer_and_evidence_gate"}


@dataclass
class EpisodeState:
    evidence: list[dict] = field(default_factory=list)
    remaining_indices: list[int] = field(default_factory=list)
    searched: bool = False
    previous_draft: dict | None = None
    counts: dict = field(default_factory=lambda: dict.fromkeys(
        ("answer_drafts", "reviewer_calls", "gate_calls", "tool_calls", "agent_steps"), 0))
    trace: list[dict] = field(default_factory=list)


class DeterministicQuestionUser(HalfDuplexUser):
    """No hidden facts, model calls, semantic feedback or correctness hints."""
    def __init__(self, question: str):
        super().__init__(instructions=question, tools=[])
        self.question = question

    def get_init_state(self, message_history=None):
        return {"question_sent": bool(message_history)}

    def generate_next_message(self, message, state):
        state["question_sent"] = True
        return UserMessage.text(self.question), state

    def set_seed(self, seed):
        pass


class HotpotEpisodeAgent(HalfDuplexAgent[EpisodeState]):
    additional_stop_tool_names = frozenset({"submit_answer"})

    def __init__(self, tools, domain_policy, *, public, writer, reviewer, gate,
                 arm, max_drafts, threshold, caps, timeout_seconds, cap_fallback=None):
        super().__init__(tools, domain_policy)
        self.public, self.writer, self.reviewer, self.gate = json_copy(public), writer, reviewer, gate
        self.arm, self.max_drafts, self.threshold, self.caps = arm, max_drafts, threshold, caps
        self.timeout_seconds, self.wall_deadline = timeout_seconds, None
        self.state, self.failure_category = None, None
        self.cap_fallback, self.submission_reason = cap_fallback, None

    def set_seed(self, seed):
        # Model identities/seeds and mutable runtime configuration stay outside this adapter.
        pass

    def get_init_state(self, message_history=None):
        self.state = EpisodeState()
        self.wall_deadline = time.monotonic() + self.timeout_seconds
        return self.state

    @classmethod
    def is_stop(cls, message):
        return message.content == ABSTAIN and not message.tool_calls

    def _stop(self, state, reason):
        self.failure_category = reason
        state.trace.append({"event": "stopped_without_submission", "reason": reason})
        return AssistantMessage.text(ABSTAIN), state

    def _call_tool(self, state, name, arguments):
        if time.monotonic() >= self.wall_deadline:
            return self._stop(state, "episode_wall_timeout")
        if state.counts["tool_calls"] >= self.caps["tool_calls"]:
            return self._stop(state, "tool_call_cap")
        state.counts["tool_calls"] += 1
        state.trace.append({"event": "tool_requested", "name": name, "arguments": json_copy(arguments)})
        return AssistantMessage(role="assistant", tool_calls=[ToolCall(
            id=f"hotpot-tool-{state.counts['tool_calls']}", name=name, arguments=arguments)]), state

    def _public_state(self, state):
        return {"question": self.public["question"], "evidence": deepcopy(state.evidence),
                "previous_draft": deepcopy(state.previous_draft)}

    def _gate(self, state, draft, criterion):
        if time.monotonic() >= self.wall_deadline:
            self.failure_category = "episode_wall_timeout"
            return None
        if state.counts["gate_calls"] >= self.caps["gate_calls"]:
            return None
        state.counts["gate_calls"] += 1
        event = {"event": "gate", "criterion": criterion, "status": "pending"}
        state.trace.append(event)
        try:
            result = self.gate(self._public_state(state), deepcopy(draft), criterion)
            if (type(result) is not dict or type(result.get("p_true")) not in (int, float)
                    or not math.isfinite(result["p_true"]) or not 0 <= result["p_true"] <= 1):
                raise ValueError("gate must return a finite p_true probability")
            event.update(status="completed", score=json_copy(result))
        except ValueError:
            event.update(status="failed", error_type="ValueError")
            self.failure_category = "gate_value_error"
            return None
        return result["p_true"]

    def _drafts(self, state):
        feedback = None
        target = FIXED.get(self.arm)
        while state.counts["answer_drafts"] < self.max_drafts:
            if time.monotonic() >= self.wall_deadline:
                return self._stop(state, "episode_wall_timeout")
            state.counts["answer_drafts"] += 1
            event = {"event": "draft", "number": state.counts["answer_drafts"], "status": "pending"}
            state.trace.append(event)
            try:
                draft = self.writer(self._public_state(state), feedback)
            except ValueError:
                event.update(status="failed", error_type="ValueError")
                return self._stop(state, "writer_value_error")
            try:
                draft = validate_submission(draft, self.public)
            except ValueError:
                event.update(status="invalid_submission")
                return self._stop(state, "invalid_submission")
            event.update(status="completed", draft=deepcopy(draft))
            if target is not None:
                approved = state.counts["answer_drafts"] == target
            else:
                criteria = ("answer", "evidence") if self.arm in {
                    "decomposed", "separate_answer_and_evidence_gate"} else ("answer",)
                scores = []
                for criterion in criteria:
                    probability = self._gate(state, draft, criterion)
                    if probability is None:
                        return self._stop(state, self.failure_category or "gate_call_cap")
                    scores.append(probability)
                approved = all(score >= self.threshold for score in scores)
            state.previous_draft = deepcopy(draft)
            if approved:
                event["route"] = "submit"
                self.submission_reason = "fixed_target" if target is not None else "gate_approved"
                return self._call_tool(state, "submit_answer", draft)
            event["route"] = "review"
            if state.counts["answer_drafts"] >= self.max_drafts:
                if self.cap_fallback == "last_valid" and target is None:
                    event["route"] = "submit_budget_fallback"
                    self.submission_reason = "budget_fallback"
                    state.trace.append({"event": "budget_fallback", "reason": "draft_cap",
                        "selected_draft_number": state.counts["answer_drafts"], "gate_approved": False})
                    return self._call_tool(state, "submit_answer", draft)
                return self._stop(state, "draft_cap_no_accepted_submission")
            if state.counts["reviewer_calls"] >= self.caps["reviewer_calls"]:
                return self._stop(state, "reviewer_call_cap")
            if time.monotonic() >= self.wall_deadline:
                return self._stop(state, "episode_wall_timeout")
            state.counts["reviewer_calls"] += 1
            review = {"event": "review", "number": state.counts["reviewer_calls"], "status": "pending"}
            state.trace.append(review)
            try:
                feedback = self.reviewer(self._public_state(state), deepcopy(draft))
                if type(feedback) is not str or not feedback.strip():
                    raise ValueError("reviewer must return nonempty text")
            except ValueError:
                review.update(status="failed", error_type="ValueError")
                return self._stop(state, "reviewer_value_error")
            review.update(status="completed", feedback=feedback)
        return self._stop(state, "draft_cap_no_accepted_submission")

    def generate_next_message(self, message, state):
        state.counts["agent_steps"] += 1
        if state.counts["agent_steps"] > self.caps["agent_steps"]:
            return self._stop(state, "agent_step_cap")
        if isinstance(message, MultiToolMessage):
            raise ValueError("the frozen custom domain uses one tool per native turn")
        if isinstance(message, ToolMessage):
            if message.requestor != "assistant" or message.error:
                return self._stop(state, "native_tool_error")
            result = json.loads(message.content)
            if not state.searched:
                state.searched = True
                state.remaining_indices = [row["paragraph_index"] for row in result]
            else:
                state.evidence.append(result)
        if not state.searched:
            return self._call_tool(state, "search_passages", {"query": self.public["question"]})
        if state.remaining_indices:
            return self._call_tool(state, "read_passage", {"paragraph_index": state.remaining_indices.pop(0)})
        return self._drafts(state)


def run_episode(case, writer, reviewer=None, gate=None, arm="fixed1", max_drafts=3,
                threshold=.8, passage_indices=None, caps=None, seed=42, timeout_seconds=600,
                cap_fallback=None) -> dict:
    """Run one real native episode; provider failures retain actual partial traces.

    No automatic retry, model import, training or hosted request occurs here.
    All counters are logical callback/tool counts, not invented physical costs.
    """
    public, gold = validate_case(case)
    indices = validate_indices(passage_indices, public)
    if arm not in FIXED.keys() | GATED or type(max_drafts) is not int or not 1 <= max_drafts <= 10:
        raise ValueError("declare a supported arm and one through ten total drafts")
    if cap_fallback not in (None, "last_valid"):
        raise ValueError("Declare legacy abstention or the explicit last-valid draft-cap policy")
    if arm in FIXED and FIXED[arm] > max_drafts:
        raise ValueError("fixed control exceeds the episode-wide draft cap")
    if not callable(writer) or (max_drafts > 1 and arm != "fixed1" and not callable(reviewer)):
        raise ValueError("explicit writer and any required independent reviewer are needed")
    if arm in GATED and not callable(gate):
        raise ValueError("gated arms require an explicit gate callback")
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 < threshold < 1 or type(seed) is not int or seed < 0
            or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError("invalid threshold, seed or finite episode timeout")
    limits = {"reviewer_calls": max_drafts - 1, "gate_calls": 2 * max_drafts,
              "tool_calls": len(indices) + 2, "agent_steps": len(indices) + 2,
              "tau_steps": 2 * (len(indices) + 2) + 2}
    if caps is not None:
        if type(caps) is not dict or set(caps) - set(limits):
            raise ValueError("unknown episode cap")
        limits.update(caps)
    if any(type(value) is not int or value < 0 for value in limits.values()) or limits["tau_steps"] < 1:
        raise ValueError("episode caps must be nonnegative integers with at least one native step")
    register_domain()
    environment = get_environment(public_case=public, passage_indices=indices)
    task = make_task(public, gold)
    validate_task(task)
    agent = HotpotEpisodeAgent(environment.get_tools(), POLICY, public=public,
        writer=writer, reviewer=reviewer, gate=gate, arm=arm, max_drafts=max_drafts,
        threshold=threshold, caps=limits, timeout_seconds=timeout_seconds, cap_fallback=cap_fallback)
    orchestrator = Orchestrator(domain=DOMAIN, agent=agent, user=DeterministicQuestionUser(public["question"]),
        environment=environment, task=task, max_steps=limits["tau_steps"], max_errors=1,
        seed=seed, timeout=timeout_seconds, validate_communication=True)
    try:
        simulation = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL,
                                   env_kwargs={"public_case": public, "passage_indices": indices})
    except Exception as exc:
        # The outer worker must abort on an uncertain physical callback. Preserve
        # only the actual partial native trace, never invent a finished run.
        state = agent.state or EpisodeState()
        exc.haltiq_episode_partial = json_copy({"question_id": public["id"], "arm": arm,
            "counts": state.counts, "trace": state.trace,
            "native_partial_messages": [m.model_dump(mode="json") for m in orchestrator.get_messages()]})
        raise
    submitted = deepcopy(environment.tools.db.submission)
    normal = simulation is not None and simulation.termination_reason.value in {"agent_stop", "user_stop"}
    completed = normal and submitted is not None
    state = agent.state or EpisodeState()
    result = {"scope": "custom_HotpotQA_using_native_Tau2_orchestration",
        "upstream_revision": TAU_REVISION, "question_id": public["id"], "arm": arm,
        "status": "completed" if completed else "completed_with_failure",
        "failure_category": agent.failure_category or (simulation.termination_reason.value if not normal else None),
        "error_type": None, "submitted": submitted,
        "metrics": score_submission(submitted if completed else None, gold),
        "native_reward": simulation.reward_info.reward if simulation is not None else None,
        "native_simulation": simulation.model_dump(mode="json") if simulation is not None else None,
        "native_partial_messages": [],
        "counts": dict(state.counts), "caps": {"answer_drafts": max_drafts, **limits},
        "trace": deepcopy(state.trace), "passage_indices": indices, "seed": seed,
        "critic_threshold": threshold, "physical_cost_scope": "provider callbacks own actual physical work accounting",
        "scorer": {"revision": OFFICIAL_REVISION, "sha256": OFFICIAL_SHA256, "url": OFFICIAL_URL}}
    if cap_fallback is not None:
        result.update(cap_fallback=cap_fallback,
            submission_reason=agent.submission_reason if completed else None,
            budget_fallback_used=completed and agent.submission_reason == "budget_fallback",
            gate_approved=completed and agent.submission_reason == "gate_approved")
    return json_copy(result)
