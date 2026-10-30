"""Fresh official Tau retail study with a retrained Strands-feature head.

This is an additional 7B engineering study, not a rewrite of the recorded 1.5B
panel. Tau owns tools, hidden tasks and evaluation. Model outputs remain data
until a valid proposal passes the declared policy.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
import uuid

from .strong_writer import MODEL, REVISION, BITSANDBYTES_VERSION
from .tau import (API_BASE, UPSTREAM_REVISION, append, canonical, critic_inputs,
                  error_message, local_judges, local_model_args, parse_proposal,
                  save, verify_upstream)

ARMS = ("first_valid", "strands", "retrained", "retrained_AND")
ARM_ALIASES = {"first_valid": "baseline", "strands": "strands",
               "retrained": "retrained", "retrained_AND": "retrained_AND"}
ORIGINAL_TASK_IDS = frozenset({"88", "1", "45", "53", "86", "84", "41", "13", "106", "64"})
FORMAT_POLICY = {
    "proposal_contract": "exactly content and tool_calls; nonempty text XOR one to four available tool calls",
    "max_writer_attempts_per_proposal": 2,
    "max_format_corrections_per_proposal": 1,
    "correction_trigger": "parse_proposal contract ValueError only; never provider/truncation errors",
    "correction_input": "original public observation/prompt, original invalid output and validation error",
    "all_invalid_outputs_and_usage_preserved": True,
    "gold_or_reward_based_repairs": False,
    "semantic_retries_or_evaluator_retries": False,
}


def select_task_ids(task_ids, *, limit=10, seed=43, phase="evaluation", extra_exclusions=(),
                    manifest=None):
    """Freeze two development IDs and a disjoint evaluation prefix by public ID."""
    ids = list(task_ids)
    if (type(seed) is not int or seed < 0 or type(limit) is not int or limit < 1
            or phase not in {"smoke", "evaluation"}
            or limit > (2 if phase == "smoke" else 10)
            or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids)):
        raise ValueError("invalid task IDs, phase, seed or bounded task count")
    extra = set(extra_exclusions)
    if any(not isinstance(i, str) or not i for i in extra):
        raise ValueError("excluded task IDs must be nonempty strings")
    ordered = sorted(set(ids) - ORIGINAL_TASK_IDS - extra,
                     key=lambda i: (hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), i))
    if len(ordered) < 2 + (limit if phase == "evaluation" else 0):
        raise ValueError("not enough fresh official tasks for development and evaluation")
    smoke_ids = ordered[:2]
    selected = smoke_ids[:limit] if phase == "smoke" else ordered[2:2 + limit]
    if manifest is not None:
        if not isinstance(manifest, dict) or set(manifest) != {"task_ids"}:
            raise ValueError("task manifest is data only: exactly one task_ids list")
        if manifest["task_ids"] != selected:
            raise ValueError("task manifest differs from the frozen public-hash selection")
    return selected, smoke_ids


def propose_valid(agents, system, prompt, available_names, *, round_index, trace):
    """One explicit format correction; no label lookup, string repair or execution."""
    current_prompt = prompt
    for attempt in range(2):
        start = len(getattr(agents, "usage", []))
        raw = agents.generate("writer", system, current_prompt)
        trace({"event": "writer_output", "round": round_index, "writer_attempt": attempt + 1,
               "raw": raw, "call_usage": getattr(agents, "usage", [])[start:]})
        try:
            return parse_proposal(raw, available_names)
        except ValueError as exc:
            trace({"event": "invalid_proposal", "round": round_index,
                   "writer_attempt": attempt + 1, "raw": raw,
                   "error_type": type(exc).__name__, "error_message": error_message(exc)})
            if attempt:
                raise ValueError("proposal remains invalid after one declared format correction") from exc
            trace({"event": "format_correction_requested", "round": round_index,
                   "original_raw": raw, "validation_error": error_message(exc)})
            current_prompt = (prompt + "\nFORMAT CORRECTION ONLY. The previous output violated the "
                "public proposal contract: " + error_message(exc) +
                "\nReturn the required JSON object using only the same public observations. "
                "No tool has executed; do not invent its result.\nOriginal invalid output:\n" + raw)
    raise AssertionError("bounded proposal loop did not return or raise")


def make_agent(agents, gate, tools, domain_policy, arm, *, max_rounds=6, trace=None):
    from tau2.agent.base_agent import AgentError, HalfDuplexAgent
    from tau2.data_model.message import (AssistantMessage, MultiToolMessage,
                                        ToolCall, ToolMessage, UserMessage)
    if arm not in ARMS or type(max_rounds) is not int or not 1 <= max_rounds <= 6:
        raise ValueError("invalid policy arm or revision budget")
    schemas = [tool.openai_schema.get("function", tool.openai_schema) for tool in tools]
    names = {schema["name"] for schema in schemas}
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
                    "tool_calls": [{"name": call.name, "arguments": call.arguments}
                                   for call in message.tool_calls or []]}
        raise ValueError("only delivered public text and tool observations are allowed")

    class AlignedAgent(HalfDuplexAgent):
        def __init__(self):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.events = []

        def get_init_state(self, message_history=None):
            return State(messages=[public(message) for message in message_history or []])

        def set_seed(self, seed):
            self.seed = seed

        def generate_next_message(self, message, state):
            delivered = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
            state.messages.extend(public(observation) for observation in delivered)
            observed = canonical({"policy": domain_policy, "tools": schemas, "messages": state.messages})
            previous, previous_distance, review = None, 0.0, None
            for round_index in range(1, max_rounds + 1):
                prompt = (observed + "\nReturn exactly JSON: {\"content\": string|null, "
                    "\"tool_calls\": [{\"name\": string, \"arguments\": object}]}. "
                    "Use nonempty content with [] calls OR null content with one to four tool calls. "
                    "The proposal has not been sent or executed.")
                if previous is not None:
                    prompt += "\nRejected proposal:\n" + previous + "\nIndependent review:\n" + review
                try:
                    proposal = propose_valid(agents,
                        "Follow the supplied retail policy exactly. Propose only the NEXT action; "
                        "never invent tool results.", prompt, names,
                        round_index=round_index, trace=trace)
                    draft = canonical(proposal)
                    question, passages = critic_inputs(domain_policy, schemas, state.messages, proposal)
                    decision = {"stop": True, "reason": "first_valid_proposal",
                                "p_stop": None, "p_approve": None, "distance": None}
                    if arm != "first_valid":
                        decision = gate.decide(ARM_ALIASES[arm], draft, previous, round_index,
                            max_rounds, question, passages, previous_distance=previous_distance)
                    if type(decision.get("stop")) is not bool:
                        raise ValueError("gate must return an explicit Boolean stop decision")
                    event = {"round": round_index, "proposal": proposal, "decision": decision}
                    self.events.append(event)
                    trace({"event": "gate_decision", **event,
                        "public_critic_input_sha256": hashlib.sha256(canonical({
                            "question": question, "passages": passages, "draft": draft}).encode()).hexdigest()})
                    if decision["stop"]:
                        response = AssistantMessage(role="assistant", content=proposal["content"],
                            tool_calls=[ToolCall(id=f"aligned-{len(state.messages)}-{index}", **call)
                                        for index, call in enumerate(proposal["tool_calls"])] or None)
                        state.messages.append(public(response))
                        event["route"] = "execute"
                        return response, state
                    event["route"] = "review" if round_index < max_rounds else "rejected_at_cap"
                    if round_index == max_rounds:
                        raise AgentError("aligned pair rejected the action at the declared revision cap")
                    review = agents.generate("reviewer", "Independently review the proposed NEXT action. "
                        "Use only public policy and observations. Give concise corrective feedback; "
                        "return exactly APPROVED only when it is valid.", observed + "\nProposal:\n" + draft).strip()
                    if not review:
                        raise ValueError("empty independent review")
                    event["review"] = review
                    trace({"event": "reviewer_output", "round": round_index, "review": review})
                    previous, previous_distance = draft, decision.get("distance", 0.0)
                except AgentError:
                    raise
                except Exception as exc:
                    event = {"round": round_index, "route": "technical_failure",
                             "error_type": type(exc).__name__, "error_message": error_message(exc)}
                    self.events.append(event)
                    trace({"event": "technical_failure", **event})
                    raise AgentError("aligned provider/parser failure: " + type(exc).__name__) from exc
            raise AgentError("no permitted proposal")

    return AlignedAgent()


def summarize(rows, planned_tasks, arms):
    """Coverage and failures remain in the planned denominator."""
    summary = {"benchmark": "official_tau2_retail_aligned_7b", "leaderboard_comparable": False,
        "local_judges": True, "hosted_api_calls": 0, "planned_episodes": planned_tasks * len(arms),
        "reported_episodes": len(rows), "arms": {}}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        summary["arms"][arm] = {
            "planned": planned_tasks, "reported": len(selected),
            "completed": sum(row["status"] == "completed" for row in selected),
            "successful_tasks": sum(row["reward"] == 1.0 for row in selected),
            "failed_or_refused": sum(row["status"] == "failed" for row in selected),
            "interrupted": sum(row["status"] == "interrupted" for row in selected),
            "controller_cap_refusals": sum(any(e.get("route") == "rejected_at_cap"
                for e in row["route_events"]) for row in selected),
            "format_corrections_requested": sum(row["format_corrections_requested"] for row in selected),
            "writer_reviewer_invocations": sum(len(row["writer_reviewer_usage"]) for row in selected),
            "official_generation_invocations": sum(len(row["official_calls"]) for row in selected),
            "success_failure_inclusive": sum(row["reward"] for row in selected) / planned_tasks,
        }
    summary["complete"] = (len(rows) == summary["planned_episodes"]
                           and all(row["status"] in {"completed", "failed"} for row in rows))
    summary["all_simulations_executed"] = summary["complete"] and all(row["status"] == "completed" for row in rows)
    summary["status"] = ("completed" if summary["all_simulations_executed"] else
                         "completed_with_failures" if summary["complete"] else "incomplete")
    return summary


def run(agents, gate, output, limit=10, seed=43, *, upstream_root="/content/tau2-bench",
        arms=ARMS, max_rounds=6, max_steps=40, phase="evaluation", task_manifest=None,
        extra_exclusions=(), api_base=API_BASE):
    """Use unchanged official tools/evaluation and a separate fresh task panel."""
    root = verify_upstream(upstream_root)
    if (not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS)
            or type(max_rounds) is not int or not 1 <= max_rounds <= 6
            or type(max_steps) is not int or not 1 <= max_steps <= 200):
        raise ValueError("declare unique policy arms and bounded episode/revision budgets")
    if getattr(agents, "model_id", MODEL) != MODEL:
        raise ValueError("this engineering panel requires the declared local 7B model")
    args = local_model_args(api_base, seed=seed)
    from tau2.data_model.semantic_communication import CommunicateScoring, SemanticCommunicateSettings
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.multilingual.runtime import pin_language_artifacts
    from tau2.registry import registry
    from tau2.runner import build_text_orchestrator, run_simulation
    from tau2.runner.helpers import select_run_tasks
    model = "openai/" + MODEL
    scoring = CommunicateScoring(judge=SemanticCommunicateSettings(model=model, model_args=args))
    config = TextRunConfig(domain="retail", language="en", agent="llm_agent",
        llm_agent=model, llm_user=model, num_trials=1, max_concurrency=1,
        max_steps=max_steps, seed=seed, task_split_name="base", llm_args_agent=args,
        llm_args_user=args, communicate_scoring=scoring, enforce_communication_protocol=True)
    loaded = pin_language_artifacts(config)
    available = select_run_tasks(config, loaded)
    chosen, development = select_task_ids([task.id for task in available], limit=limit, seed=seed,
        phase=phase, extra_exclusions=extra_exclusions, manifest=task_manifest)
    by_id = {task.id: task for task in available}
    tasks = [by_id[task_id] for task_id in chosen]
    output = Path(output)
    if output.exists() and any(item.name != "runtime" for item in output.iterdir()):
        raise ValueError("output already contains study artifacts; select a new directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"benchmark": "official_tau2_retail_aligned_7b", "upstream_revision": UPSTREAM_REVISION,
        "benchmark_version": "2.0.0", "domain": "retail", "language": "en", "task_split": "base",
        "phase": phase, "development_smoke_pooled_into_evaluation": False,
        "task_ids": chosen, "development_smoke_task_ids": development,
        "old_seed42_task_ids_excluded": sorted(ORIGINAL_TASK_IDS),
        "extra_task_ids_excluded": sorted(extra_exclusions), "seed": seed,
        "task_selection": "sha256(seed:public_task_id); first2development, nextNevaluation; no rewards/outcomes read",
        "task_sha256": {task.id: hashlib.sha256(canonical(task.model_dump(mode="json")).encode()).hexdigest()
                        for task in tasks}, "arms": list(arms), "internal_arm_aliases": ARM_ALIASES,
        "max_rounds": max_rounds, "max_steps": max_steps, "evaluation_type": "all",
        "communicate_scorer": "semantic", "simulator_and_judge_model": MODEL,
        "model_revision": REVISION, "quantization": "NF4 double quantization, FP16 computation",
        "bitsandbytes_version": BITSANDBYTES_VERSION, "local_api_base": api_base,
        "format_policy": FORMAT_POLICY, "hosted_api_calls": 0, "leaderboard_comparable": False,
        "scope": "Official tools/scenarios/evaluator; all writer, reviewer, local user and judges use the 7B variant.",
        "tau_transfer": {"training_domain": "HotpotQA only; no Tau calibration",
            "distribution_shift": True, "strands_threshold": 0.8, "head_threshold": 0.5,
            "critic_query": "next-action policy criterion plus latest delivered customer request",
            "candidate_evidence": "whole public policy in1500-character chunks/1350 stride, proposed tool schemas, public delivered messages",
            "reranking": "same pinned BGE top2 question+draft retrieval; no hidden task fields",
            "first_valid": "execute first contract-valid proposal, allowing at most one format correction",
            "strands": "execute only Strands approval",
            "retrained": "execute only QA-trained Strands-feature head stop",
            "retrained_AND": "execute only head stop AND Strands approval"},
        "language_artifacts": config.language_artifacts.model_dump(mode="json"),
        "agents_provenance": getattr(agents, "provenance", {"class": type(agents).__name__}),
        "gate_provenance": getattr(gate, "provenance", {"class": type(gate).__name__}),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "frozen_before_model_calls": True, "provider_and_evaluator_retries": 0,
        "rejection_cap_policy": "fail action/episode; rejected actions never execute",
        "arm_order": "rotate by task index"}
    save(output / "plan.json", plan)
    rows = []
    save(output / "summary.json", summarize(rows, len(tasks), arms))
    for task_index, task in enumerate(tasks):
        ordered = list(arms[task_index % len(arms):]) + list(arms[:task_index % len(arms)])
        for arm in ordered:
            verify_upstream(root)
            holder, episode_events = [], []
            def trace(event):
                episode_events.append(event)
                append(output / "traces.jsonl", {"task_id": task.id, "arm": arm, **event})
            def factory(tools, domain_policy, **unused):
                # Official code may supply a hidden task in unused. Never read it.
                agent = make_agent(agents, gate, tools, domain_policy, arm,
                                   max_rounds=max_rounds, trace=trace)
                holder.append(agent)
                return agent
            registered = "aligned_" + uuid.uuid4().hex
            registry.register_agent_factory(factory, registered)
            current = config.model_copy(deep=True, update={"agent": registered, "task_ids": [task.id]})
            row = {"task_id": task.id, "arm": arm, "seed": seed, "status": "pending", "reward": 0.0}
            started, usage_start = time.perf_counter(), len(agents.usage)
            calls = []
            trace({"event": "episode_started"})
            try:
                with local_judges(api_base, MODEL, seed=seed) as calls:
                    orchestrator = build_text_orchestrator(current, task, seed=seed)
                    simulation = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
                row.update(status="completed", reward=simulation.reward_info.reward,
                    termination_reason=simulation.termination_reason.value,
                    reward_info=simulation.reward_info.model_dump(mode="json"))
                save(output / f"simulation-{task_index}-{arm}.json", simulation.model_dump(mode="json"))
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error_message=error_message(exc))
            except BaseException as exc:
                row.update(status="interrupted", error_type=type(exc).__name__,
                           error_message=error_message(exc))
                raise
            finally:
                row.update(seconds=time.perf_counter() - started, official_calls=calls,
                    writer_reviewer_usage=agents.usage[usage_start:],
                    route_events=holder[0].events if holder else [],
                    format_corrections_requested=sum(event.get("event") == "format_correction_requested"
                                                     for event in episode_events))
                rows.append(row)
                append(output / "outcomes.jsonl", row)
                trace({"event": "episode_finished", "status": row["status"], "reward": row["reward"]})
                summary = summarize(rows, len(tasks), arms)
                save(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--upstream-root", default="/content/tau2-bench")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--phase", choices=["smoke", "evaluation"], default="evaluation")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--taskids-manifest", type=Path)
    parser.add_argument("--exclude-task-ids", nargs="*", default=[])
    args = parser.parse_args()
    from .strong_writer import StrongRuntime
    from .team_training import StrandsHead
    if args.output.exists():
        parser.error("--output must be a brand-new directory")
    head = StrandsHead.from_json(args.head)
    manifest = None
    if args.taskids_manifest:
        if args.taskids_manifest.stat().st_size > 100000:
            parser.error("task manifest exceeds the data-only size bound")
        manifest = json.loads(args.taskids_manifest.read_text(encoding="utf-8"))
    with StrongRuntime(args.output / "runtime", head) as (agents, gate):
        summary = run(agents, gate, args.output, limit=args.limit, seed=args.seed,
            upstream_root=args.upstream_root, max_rounds=args.max_rounds,
            max_steps=args.max_steps, phase=args.phase, task_manifest=manifest,
            extra_exclusions=args.exclude_task_ids)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
