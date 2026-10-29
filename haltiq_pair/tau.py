"""Strands/linear policy controls in the pinned official Tau² retail simulator.

The native simulator owns tool execution and hidden scenarios. The agent sees
only delivered messages, the domain policy and public tool schemas. Local model
judges retain upstream evaluation logic but do not produce a leaderboard score.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlparse
import uuid

UPSTREAM_REVISION = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"
ARMS = ("baseline", "strands", "linear", "combined")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
API_BASE = "http://127.0.0.1:8001/v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def error_message(exc):
    message = str(exc)[:2000]
    message = re.sub(r"(?i)(api[_-]?key|authorization|password)([\"'\s:=]+)[^,\s}]+",
                     r"\1\2[redacted]", message)
    return re.sub(r"https?://[^/@\s]+:[^/@\s]+@", "http://[redacted]@", message)


def critic_inputs(policy, schemas, messages, proposal):
    """Declared Tau transfer retrieval: full candidates, short public query."""
    last_request = next((m["content"] for m in reversed(messages)
                         if m["role"] == "user" and m.get("content")), "")
    question = ("Is the proposed NEXT assistant action allowed by the retail policy, "
                "supported by observed tool results, and responsive to the customer's "
                "latest delivered request?\nLatest customer request: " + last_request)
    # Every policy character remains available. Overlap preserves constraints at
    # chunk boundaries; retrieval's declared top-two choice can still miss one.
    candidates = [f"Retail policy excerpt at character {start}:\n" + policy[start:start + 1500]
                  for start in range(0, len(policy), 1350)]
    proposed_names = {call["name"] for call in proposal["tool_calls"]}
    candidates.extend("Proposed tool schema:\n" + canonical(schema) for schema in schemas
                      if schema["name"] in proposed_names)
    candidates.extend("Observed public message:\n" + canonical(m) for m in messages)
    return question, candidates


def local_model_args(api_base=API_BASE, *, seed=42):
    """Constrain all simulator and judge requests to the local model server."""
    parsed = urlparse(api_base)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("Tau model calls require a loopback HTTP /v1 endpoint")
    return {"api_base": api_base.rstrip("/"), "api_key": "local", "temperature": 0,
            "seed": seed, "max_tokens": 768, "timeout": 180, "num_retries": 0}


def parse_proposal(text, available_names):
    """Validate a proposal without executing a tool or repairing model output."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    row = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("nonfinite action JSON")))
    if not isinstance(row, dict) or set(row) != {"content", "tool_calls"}:
        raise ValueError("proposal requires exactly content and tool_calls")
    content, calls = row["content"], row["tool_calls"]
    if content is not None and (not isinstance(content, str) or not content.strip()):
        raise ValueError("content must be nonempty text or null")
    if not isinstance(calls, list) or len(calls) > 4 or bool(content) == bool(calls):
        raise ValueError("propose either user-facing text or one to four tool calls")
    for call in calls:
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or call["name"] not in available_names
                or not isinstance(call["arguments"], dict)):
            raise ValueError("invalid or unavailable proposed tool")
    # A complete JSON round trip also rejects objects outside the public contract.
    return json.loads(canonical(row))


def verify_upstream(upstream_root):
    """Bind imported Tau and clean source/data to the reviewed source revision."""
    root = Path(upstream_root).resolve(strict=True)
    def git(*argv):
        result = subprocess.run(["git", "-C", str(root), *argv], text=True,
                                capture_output=True, check=True, timeout=30)
        return result.stdout.strip()
    if git("rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError("Tau source differs from the pinned upstream revision")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tau tracked source/data changed")
    spec = importlib.util.find_spec("tau2")
    if (spec is None or spec.origin is None or Path(spec.origin).resolve()
            != root / "src" / "tau2" / "__init__.py"):
        raise ValueError("install the pinned Tau checkout in editable mode")
    return root


@contextmanager
def local_judges(api_base=API_BASE, model_id=MODEL_ID, *, seed=42):
    """Scope local judge selection and no-retry call receipts; restore globals."""
    from tau2.user import user_simulator
    from tau2.evaluator import evaluator_communicate, evaluator_nl_assertions
    from tau2.utils.llm_utils import llm_log_dir
    if llm_log_dir.get() is not None:
        raise ValueError("disable upstream request-kwargs debug logging")
    args = local_model_args(api_base, seed=seed)
    model = "openai/" + model_id
    old_model = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS
    old_args = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS
    originals, calls = [], []
    def wrap(original, role):
        def generate(*positional, **kwargs):
            if positional or kwargs.get("model") != model:
                raise ValueError("unexpected model/provider at official generation seam")
            kwargs.update(args)
            kwargs["response_validation_retries"] = 0
            receipt = {"role": role, "model": model, "status": "pending",
                       "input_tokens": None, "output_tokens": None,
                       "hosted_api_calls": 0}
            calls.append(receipt)
            started = time.perf_counter()
            try:
                response = original(**kwargs)
                if role == "nl_judge":
                    # Upstream treats an absent/empty results list as all([]),
                    # which is true. Refuse incomplete local model replies before
                    # they can reach that unchanged evaluator implementation.
                    expected = ast.literal_eval(kwargs["messages"][-1].content.rsplit(
                        "expectedOutcomes:", 1)[1].strip())
                    reply = json.loads(response.content or "")
                    results = reply.get("results") if isinstance(reply, dict) else None
                    if (not isinstance(results, list) or len(results) != len(expected)
                            or any(not isinstance(r, dict) or set(r) != {
                                "expectedOutcome", "reasoning", "metExpectation"}
                                or type(r["metExpectation"]) is not bool
                                or not isinstance(r["reasoning"], str) or not r["reasoning"].strip()
                                or not isinstance(r["expectedOutcome"], str) for r in results)
                            or Counter(r["expectedOutcome"] for r in results) != Counter(expected)):
                        raise ValueError("local NL judge did not return every expected outcome exactly once")
                usage = response.usage or {}
                receipt.update(status="completed", input_tokens=usage.get("prompt_tokens"),
                               output_tokens=usage.get("completion_tokens"))
                return response
            except Exception:
                receipt["status"] = "failed"
                raise
            finally:
                receipt["seconds"] = time.perf_counter() - started
        return generate
    try:
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = model
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(args)
        for module, role in ((user_simulator, "user"),
                             (evaluator_communicate, "communication_judge"),
                             (evaluator_nl_assertions, "nl_judge")):
            originals.append((module, module.generate))
            module.generate = wrap(module.generate, role)
        yield calls
    finally:
        for module, original in reversed(originals):
            module.generate = original
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = old_model
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = old_args


def make_agent(agents, gate, tools, domain_policy, arm, *, max_rounds=6, trace=None):
    """Construct the official HalfDuplexAgent wrapper without a hidden task."""
    from tau2.agent.base_agent import AgentError, HalfDuplexAgent
    from tau2.data_model.message import (AssistantMessage, MultiToolMessage,
                                        ToolCall, ToolMessage, UserMessage)
    if arm not in ARMS or type(max_rounds) is not int or not 1 <= max_rounds <= 10:
        raise ValueError("invalid arm or revision budget")
    schemas = [tool.openai_schema.get("function", tool.openai_schema) for tool in tools]
    names = {row["name"] for row in schemas}
    trace = trace or (lambda event: None)

    @dataclass
    class State:
        messages: list = field(default_factory=list)

    def public(message):
        if isinstance(message, UserMessage) and not message.is_audio and not message.tool_calls:
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage) and message.requestor == "assistant":
            return {"role": "tool", "content": message.content,
                    "tool_call_id": message.id, "error": message.error}
        if isinstance(message, AssistantMessage) and not message.is_audio:
            return {"role": "assistant", "content": message.content,
                    "tool_calls": [{"name": c.name, "arguments": c.arguments}
                                   for c in message.tool_calls or []]}
        raise ValueError("only delivered public text/tool observations are allowed")

    class PairAgent(HalfDuplexAgent):
        def __init__(self):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.events = []

        def get_init_state(self, message_history=None):
            return State(messages=[public(m) for m in message_history or []])

        def set_seed(self, seed):
            # The generator/server and official user receive the frozen run seed.
            self.seed = seed

        def generate_next_message(self, message, state):
            delivered = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
            state.messages.extend(public(m) for m in delivered)
            observed = canonical({"policy": domain_policy, "tools": schemas,
                                  "messages": state.messages})
            previous, previous_distance, review = None, 0.0, None
            def review_draft(draft, round_index):
                feedback = agents.generate("reviewer", "Independently review the proposed NEXT action. "
                    "Use only public policy and observations. Give concise corrective feedback; "
                    "return exactly APPROVED only when it is valid.", observed + "\nProposal:\n" + draft).strip()
                if not feedback:
                    raise ValueError("empty independent review")
                trace({"event": "reviewer_output", "round": round_index, "review": feedback})
                return feedback
            for round_index in range(1, max_rounds + 1):
                prompt = (observed + "\nReturn exactly JSON: {\"content\": string|null, "
                          "\"tool_calls\": [{\"name\": string, \"arguments\": object}]}. "
                          "Use either nonempty content with [] calls, or null content with tools. "
                          "The proposal has not been sent or executed.")
                if previous is not None:
                    prompt += "\nRejected proposal:\n" + previous + "\nIndependent review:\n" + review
                try:
                    raw = agents.generate("writer", "Follow the supplied retail policy exactly. "
                                          "Propose only the NEXT action, never invent tool results.", prompt)
                    trace({"event": "writer_output", "round": round_index, "raw": raw})
                    proposal = parse_proposal(raw, names)
                    draft = canonical(proposal)
                    question, passages = critic_inputs(domain_policy, schemas, state.messages, proposal)
                    current_review = review_draft(draft, round_index) if arm == "linear" else None
                    decision = {"stop": True, "reason": "first_proposal_baseline",
                                "p_stop": None, "p_approve": None, "distance": None}
                    if arm != "baseline":
                        decision = gate.decide(arm, draft, previous, round_index, max_rounds,
                            question, passages, previous_distance=previous_distance,
                            critic_approved=current_review == "APPROVED")
                    event = {"round": round_index, "proposal": proposal, "decision": decision}
                    if current_review is not None:
                        event["review"] = current_review
                    self.events.append(event)
                    trace({"event": "gate_decision", **event,
                           "public_critic_input_sha256": hashlib.sha256(canonical({
                               "question": question, "passages": passages, "draft": draft}).encode()).hexdigest()})
                    if decision["stop"]:
                        response = AssistantMessage(role="assistant", content=proposal["content"],
                            tool_calls=[ToolCall(id=f"pair-{len(state.messages)}-{index}", **call)
                                        for index, call in enumerate(proposal["tool_calls"])] or None)
                        state.messages.append(public(response))
                        event["route"] = "execute"
                        return response, state
                    event["route"] = "review" if round_index < max_rounds else "rejected_at_cap"
                    if round_index == max_rounds:
                        raise AgentError("pair rejected the action at the declared revision cap")
                    review = current_review if current_review is not None else review_draft(draft, round_index)
                    event["review"] = review
                    previous, previous_distance = draft, decision.get("distance", 0.0)
                except AgentError:
                    raise
                except Exception as exc:
                    self.events.append({"round": round_index, "route": "technical_failure",
                                        "error_type": type(exc).__name__, "error_message": error_message(exc)})
                    trace({"event": "technical_failure", **self.events[-1]})
                    raise AgentError("pair provider/parser failure: " + type(exc).__name__) from exc
            raise AgentError("no permitted proposal")

    return PairAgent()


def run(agents, gate, output, limit=10, seed=42, *, upstream_root="/content/tau2-bench",
        arms=ARMS, max_rounds=6, max_steps=40, api_base=API_BASE, model_id=MODEL_ID):
    """Run matched real retail tasks; failure-inclusive reward is the outcome."""
    root = verify_upstream(upstream_root)
    if (type(limit) is not int or limit < 1 or type(seed) is not int or seed < 0
            or not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS)
            or type(max_rounds) is not int or not 1 <= max_rounds <= 6
            or type(max_steps) is not int or not 1 <= max_steps <= 200):
        raise ValueError("declare valid unique arms, a positive limit and bounded steps")
    args = local_model_args(api_base, seed=seed)
    from tau2.data_model.semantic_communication import CommunicateScoring, SemanticCommunicateSettings
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.multilingual.runtime import pin_language_artifacts
    from tau2.registry import registry
    from tau2.runner import build_text_orchestrator, run_simulation
    from tau2.runner.helpers import select_run_tasks
    model = "openai/" + model_id
    scoring = CommunicateScoring(judge=SemanticCommunicateSettings(model=model, model_args=args))
    config = TextRunConfig(domain="retail", language="en", agent="llm_agent",
        llm_agent=model, llm_user=model, num_trials=1, max_concurrency=1,
        max_steps=max_steps, seed=seed, task_split_name="base", llm_args_agent=args,
        llm_args_user=args, communicate_scoring=scoring, enforce_communication_protocol=True)
    loaded = pin_language_artifacts(config)
    tasks = select_run_tasks(config, loaded)
    tasks = sorted(tasks, key=lambda task: hashlib.sha256(f"{seed}:{task.id}".encode()).hexdigest())[:limit]
    if len(tasks) != limit:
        raise ValueError("requested task count exceeds the official base split")
    output = Path(output)
    if output.exists() and any(item.name != "runtime" for item in output.iterdir()):
        raise ValueError("output already contains study artifacts; select a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"benchmark": "official_tau2_retail", "upstream_revision": UPSTREAM_REVISION,
        "benchmark_version": "2.0.0", "domain": "retail", "language": "en",
        "task_split": "base", "task_ids": [task.id for task in tasks], "seed": seed,
        "task_selection": "sha256(seed:task_id), independent of answers/rewards",
        "tau_critic_transfer": {"query": "fixed next-action policy criterion + latest delivered customer request",
            "candidate_evidence": "entire public policy in 1500-character chunks at 1350-character stride; complete proposed tool schemas; individual delivered public messages",
            "retrieval": "teammate BGE question+draft reranker, top two candidates",
            "native_context": "strict 4096 tokens; overflow is a technical failure; no silent clipping",
            "calibration": "Hobson QA probability threshold transferred without Tau calibration",
            "linear_features": "canonical action JSON replaces answer text; current reviewer APPROVED flag in linear arm, Strands approval in combined arm",
            "distribution_shift": True},
        "task_sha256": {t.id: hashlib.sha256(canonical(t.model_dump(mode="json")).encode()).hexdigest()
                        for t in tasks}, "arms": list(arms), "max_rounds": max_rounds,
        "max_steps": max_steps, "evaluation_type": "all", "communicate_scorer": "semantic",
        "simulator_and_judge_model": model_id, "local_api_base": api_base,
        "hosted_api_calls": 0, "leaderboard_comparable": False,
        "scope": "Official tools, scenarios and evaluator; local user/judges differ from released leaderboard conditions.",
        "language_artifacts": config.language_artifacts.model_dump(mode="json"),
        "agents_provenance": getattr(agents, "provenance", {"class": type(agents).__name__}),
        "gate_provenance": getattr(gate, "provenance", {"class": type(gate).__name__}),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "frozen_before_model_calls": True, "no_automatic_retries": True,
        "rejection_cap_policy": "fail action/episode; rejected actions never execute",
        "arm_order": "rotate by task index"}
    save(output / "plan.json", plan)
    rows = []
    for task_index, task in enumerate(tasks):
        ordered = list(arms[task_index % len(arms):]) + list(arms[:task_index % len(arms)])
        for arm in ordered:
            verify_upstream(root)
            agent_holder = []
            def trace(event):
                append(output / "traces.jsonl", {"task_id": task.id, "arm": arm, **event})
            def factory(tools, domain_policy, **unused):
                # `unused` can contain a hidden task: never read it or pass it on.
                agent = make_agent(agents, gate, tools, domain_policy, arm, max_rounds=max_rounds, trace=trace)
                agent_holder.append(agent)
                return agent
            registered = "pair_" + uuid.uuid4().hex
            registry.register_agent_factory(factory, registered)
            current = config.model_copy(deep=True, update={"agent": registered, "task_ids": [task.id]})
            row = {"task_id": task.id, "arm": arm, "seed": seed, "status": "pending", "reward": 0.0}
            started, usage_start = time.perf_counter(), len(agents.usage)
            calls = []
            trace({"event": "episode_started"})
            try:
                with local_judges(api_base, model_id, seed=seed) as calls:
                    orchestrator = build_text_orchestrator(current, task, seed=seed)
                    simulation = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
                row.update(status="completed", reward=simulation.reward_info.reward,
                    termination_reason=simulation.termination_reason.value,
                    reward_info=simulation.reward_info.model_dump(mode="json"))
                save(output / f"simulation-{task_index}-{arm}.json", simulation.model_dump(mode="json"))
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error_message=error_message(exc))
            finally:
                row.update(seconds=time.perf_counter() - started, official_calls=calls,
                    writer_reviewer_usage=agents.usage[usage_start:],
                    route_events=agent_holder[0].events if agent_holder else [])
                rows.append(row)
                append(output / "outcomes.jsonl", row)
                trace({"event": "episode_finished", "status": row["status"], "reward": row["reward"]})
                summary = {"benchmark": plan["benchmark"], "upstream_revision": UPSTREAM_REVISION,
                    "leaderboard_comparable": False, "local_judges": True, "hosted_api_calls": 0,
                    "planned_episodes": len(tasks) * len(arms), "reported_episodes": len(rows),
                    "arms": {name: {"planned": len(tasks),
                        "reported": sum(r["arm"] == name for r in rows),
                        "completed": sum(r["arm"] == name and r["status"] == "completed" for r in rows),
                        "successful_tasks": sum(r["arm"] == name and r["reward"] == 1.0 for r in rows),
                        "technical_failures": sum(r["arm"] == name and (r["status"] == "failed"
                            or any(e.get("route") == "technical_failure" for e in r["route_events"])) for r in rows),
                        "success_failure_inclusive": sum(r["reward"] for r in rows if r["arm"] == name) / len(tasks)}
                        for name in arms}}
                summary["complete"] = len(rows) == summary["planned_episodes"]
                summary["all_simulations_executed"] = summary["complete"] and all(r["status"] == "completed" for r in rows)
                summary["status"] = ("completed" if summary["all_simulations_executed"]
                    and not any(a["technical_failures"] for a in summary["arms"].values())
                    else "completed_with_technical_failures" if summary["complete"] else "incomplete")
                save(output / "summary.json", summary)
    return summary
