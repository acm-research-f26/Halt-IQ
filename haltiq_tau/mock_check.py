"""Key-free harness check against the real official MOCK tools and evaluator.

This executes SCRIPTED paths, not Strands/Jev model inference or benchmark
performance. The intentionally wrong write is a declared negative control.
"""
from __future__ import annotations

from .pinning import verify_upstream


def run_mock_check(upstream_root):
    pin = verify_upstream(upstream_root, require_import=True)
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.domains.mock.environment import get_environment, get_tasks
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = next(task for task in get_tasks() if task.id == "create_task_1")
    rows = []
    for label, title in (("correct_write", "Important Meeting"), ("wrong_write", "Wrong Meeting")):
        env = get_environment()
        call = ToolCall(id="scripted-call", name="create_task",
                        arguments={"user_id": "user_1", "title": title})
        response = env.get_response(call)
        if response.error:
            raise RuntimeError("scripted mock tool execution failed")
        trajectory = [AssistantMessage(role="assistant", tool_calls=[call]), response]
        reward = EnvironmentEvaluator.calculate_reward(get_environment, task, trajectory)
        rows.append({"path": label, "task_id": task.id, "reward": reward.reward,
                     "db_match": reward.db_check.db_match,
                     "actual_tool_executed": True, "native_model_calls": 0,
                     "hosted_calls": 0})
    if [row["reward"] for row in rows] != [1.0, 0.0]:
        raise AssertionError("official evaluator did not distinguish correct and wrong final state")
    return {"kind": "scripted_official_mock_harness_check", **pin, "results": rows,
            "model_quality_result": False, "scope": "Official environment/DB evaluator; no LLM/user simulator/judge calls."}
