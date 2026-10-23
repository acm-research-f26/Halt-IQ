"""Sequential live text pilot using the official simulator and reward evaluator.

Output is local research evidence. This module never publishes reports. The
caller explicitly supplies gate factories and permits model calls; nothing is
inferred from a credential, installed package or old Hotpot result.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

from . import UPSTREAM_REVISION
from .core import Action, Controller, PublicState, fingerprint, summarize_planned
from .pinning import verify_upstream


def run_pilot(config, upstream_root, output_dir, gate_factories, *, allow_model_calls=False,
              generation_args=None, deadline=None, before_call=None, after_call=None):
    """Execute the predeclared matched task/seed cohort; never retry a unit.

    All arms use the same writer/reviewer/user identities and task context.
    The user can respond differently after an intervention: this is a live
    policy comparison, not a claim of identical post-intervention transcripts.
    """
    from .__main__ import validate_pilot
    config = validate_pilot(config)
    if not allow_model_calls:
        raise ValueError("live pilot needs explicit allow_model_calls")
    if not callable(before_call) or not callable(after_call):
        raise ValueError("live writer, reviewer and user calls require explicit admission/rest hooks")
    if any("REPLACE_" in config[key] for key in ("writer_model", "reviewer_model", "user_model")):
        raise ValueError("replace example model identities before live execution")
    expected_gates = set(config["arms"]) & {"strands", "jev"}
    if set(gate_factories) != expected_gates:
        raise ValueError("explicit factories must exactly match the declared decision-model arms")
    if deadline is None or deadline.tzinfo is None or deadline.timestamp() <= time.time():
        raise ValueError("a future timezone-aware deadline is required")
    verify_upstream(upstream_root, require_import=True)
    from tau2.data_model.semantic_communication import CommunicateScoring, SemanticCommunicateSettings
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.multilingual.runtime import pin_language_artifacts
    from tau2.registry import registry
    from tau2.runner import build_text_orchestrator, run_simulation
    from tau2.runner.helpers import select_run_tasks
    from .agent import TauPolicyAgent
    from .llm import TauReviewer, TauWriter
    from .dispatch import guard_official_calls
    from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS

    output = Path(output_dir)
    if output.exists():
        raise ValueError("output exists; no automatic retry/resume or overwrite")
    args = dict(generation_args or {})
    if set(args) - {"temperature", "timeout", "max_tokens", "max_completion_tokens", "api_base", "reasoning_effort",
                    "num_thread", "num_ctx", "num_gpu", "think", "keep_alive"}:
        raise ValueError("only nonsecret generation settings are accepted")
    args.setdefault("temperature", 0)
    args.setdefault("timeout", 60)
    if all(config[key].startswith("ollama_chat/") for key in ("writer_model", "reviewer_model", "user_model")):
        args.setdefault("num_thread", 4)
        args.setdefault("num_ctx", 16384)
        args.setdefault("think", False)
        args.setdefault("max_tokens", 512)
    # MeteredLLM checks these arguments again. User and judge retries are also zero.
    scoring = CommunicateScoring(judge=SemanticCommunicateSettings(model_args={
        "reasoning_effort": "low", "num_retries": 0, "timeout": 60}))
    base = TextRunConfig(domain=config["domain"], language="en", agent="llm_agent",
        llm_agent=config["writer_model"], llm_user=config["user_model"], task_ids=config["task_ids"],
        num_trials=1, max_concurrency=1, max_steps=config["max_steps"], seed=config["seeds"][0],
        llm_args_agent={**args, "num_retries": 0}, llm_args_user={**args, "num_retries": 0},
        communicate_scoring=scoring, enforce_communication_protocol=True)
    loaded = pin_language_artifacts(base)
    selected = select_run_tasks(base, loaded)
    by_id = {task.id: task for task in selected}
    if set(by_id) != set(config["task_ids"]):
        raise ValueError("official language-bound task selection differs from plan")
    tasks = [by_id[task_id] for task_id in config["task_ids"]]
    gates = {arm: factory() for arm, factory in gate_factories.items()}
    if any(not callable(getattr(gate, "identity", None)) for gate in gates.values()):
        raise ValueError("explicit gate provider identities are required before dispatch")
    plan = {"config": config, "generation_args": args, "upstream_revision": UPSTREAM_REVISION,
            "task_sha256": {task.id: fingerprint(task.model_dump(mode="json")) for task in tasks},
            "language_artifacts": base.language_artifacts.model_dump(mode="json"),
            "communicate_scoring": scoring.model_dump(mode="json"),
            "nl_assertion_judge": {"model": DEFAULT_LLM_NL_ASSERTIONS, "model_args": {
                **DEFAULT_LLM_NL_ASSERTIONS_ARGS, "num_retries": 0, "timeout": 60}},
            "frozen_before_model_calls": True, "created_at": datetime.now().astimezone().isoformat(),
            "deadline": deadline.isoformat(), "no_automatic_retries": True,
            "all_generation_calls_have_admission_and_rest_hooks": True,
            "arms": config["arms"], "counterbalanced_arm_order": "rotate by task-seed index"}
    plan["gate_identities"] = {arm: gate.identity() for arm, gate in gates.items()}
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    frozen = fingerprint(plan)
    rows = {arm: [] for arm in config["arms"]}
    planned = [(task.id, seed) for task in tasks for seed in config["seeds"]]
    # Construction is key/model-call free. Only inspect public policy/tool schemas,
    # never the hidden task or simulator state. This is a lower-bound admission
    # check; actual complete prompts still fail closed if later context overflows.
    admission_orchestrator = build_text_orchestrator(base, tasks[0], seed=config["seeds"][0])
    schemas = []
    for tool in admission_orchestrator.environment.get_tools():
        schema = tool.openai_schema.get("function", tool.openai_schema)
        schemas.append({"name": schema["name"], "description": schema.get("description", ""),
                        "parameters": schema["parameters"]})
    largest = max(schemas, key=lambda row: len(json.dumps(row, sort_keys=True)))
    admission_messages = ({"role": "user", "content": "Hello, please help me."},)
    proposals = [Action("How can I help you?"), Action(tool_calls=({"id": "preflight-only", "name": largest["name"], "arguments": {}},))]
    preflight_rows = [gates["strands"].preflight(PublicState(admission_orchestrator.agent.domain_policy,
                      admission_messages, tuple(schemas), proposal)) for proposal in proposals]
    (output / "admission-preflight.json").write_text(json.dumps({"cases": preflight_rows,
        "scope": "text and largest-by-schema-bytes hypothetical single-tool proposal; no inference; later full prompts checked separately"},
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not all(row.get("fits") is True for row in preflight_rows):
        blocked = {"status": "blocked_context_window_before_any_model_dispatch", "model_calls": 0,
                   "plan_sha256": frozen, "planned_pairs_per_arm": len(planned), "benchmark_performance_measured": False}
        (output / "summary.json").write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        return blocked
    started, stop = time.perf_counter(), False
    for index, (task_id, seed) in enumerate(planned):
        arms = config["arms"][index % len(config["arms"]):] + config["arms"][:index % len(config["arms"])]
        for arm in arms:
            if time.time() >= deadline.timestamp():
                stop = True
                break
            verify_upstream(upstream_root, require_import=True)
            if fingerprint(json.loads((output / "plan.json").read_text(encoding="utf-8"))) != frozen:
                raise RuntimeError("frozen plan changed during the pilot")
            agent_holder = []
            def factory(tools, domain_policy, **unused):
                # Deliberately never inspect unused['task'] or any other private factory kwargs.
                writer = TauWriter(config["writer_model"], tools, allow_model_calls=True, model_args=args, deadline=deadline,
                                   before_call=before_call, after_call=after_call)
                reviewer = TauReviewer(config["reviewer_model"], allow_model_calls=True, model_args=args, deadline=deadline,
                                       before_call=before_call, after_call=after_call)
                gate = gates.get(arm)
                mode = "gated" if gate is not None else arm
                agent = TauPolicyAgent(tools, domain_policy, Controller(writer, reviewer, gate, mode=mode,
                    threshold=config["threshold"], max_reviews=config["max_reviews"]))
                agent_holder.append(agent)
                return agent
            registered_name = "haltiq_tau_live_" + frozen[:12] + "_" + str(index) + "_" + arm
            registry.register_agent_factory(factory, registered_name)
            current = base.model_copy(deep=True, update={"agent": registered_name, "seed": seed,
                                                       "timeout": deadline.timestamp() - time.time()})
            row = {"arm": arm, "task_id": task_id, "seed": seed, "status": "dispatched", "reward": None}
            unit_started = time.perf_counter()
            official_calls = []
            try:
                orchestrator = build_text_orchestrator(current, by_id[task_id], seed=seed)
                with guard_official_calls(deadline, before_call=before_call, after_call=after_call) as official_calls:
                    simulation = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
                row.update(status="completed", reward=simulation.reward_info.reward,
                    termination_reason=simulation.termination_reason.value,
                    db_match=simulation.reward_info.db_check.db_match if simulation.reward_info.db_check else None,
                    reward_breakdown=simulation.reward_info.model_dump(mode="json").get("reward_breakdown"),
                    official_agent_cost=simulation.agent_cost, official_user_cost=simulation.user_cost,
                    semantic_judge_cost_usd=None)
                (output / f"simulation-{index}-{arm}.json").write_text(
                    simulation.model_dump_json(indent=2) + "\n", encoding="utf-8")
                row["controller_failure_category"] = getattr(orchestrator.agent, "failure_category", None)
                # A known task failure is a measured zero, not permission to skip
                # the other paired arms. Unknown provider/infra outcomes halt.
                if simulation.termination_reason.value in {"timeout", "user_error", "infrastructure_error", "unexpected_error"}:
                    stop = True
                if row["controller_failure_category"] == "controller_provider_error":
                    stop = True
            except Exception as exc:
                row.update(status="failed_or_uncertain", error_type=type(exc).__name__)
                stop = True
            finally:
                row["wall_seconds"] = time.perf_counter() - unit_started
                row["official_user_and_judge_calls"] = official_calls
                if agent_holder:
                    agent = agent_holder[0]
                    row["writer_calls"] = agent.controller.writer.calls
                    row["reviewer_calls"] = agent.controller.reviewer.calls
                    row["route_events"] = agent.physical_route_events
                rows[arm].append(row)
                with (output / "outcomes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            if stop:
                break
        if stop:
            break
    result = {"plan_sha256": frozen, "status": "stopped_with_missing_or_failed" if stop else "completed",
              "wall_seconds": time.perf_counter() - started,
              "arms": {arm: summarize_planned(planned, rows[arm]) for arm in config["arms"]},
              "semantic_judge_cost_accounting": "physical calls/tokens/provider-reported dollars retained; verified dollar rates unavailable, so cost_usd remains null",
              "model_quality_claim": "small paired pilot only; scripted harness tests are separate"}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
