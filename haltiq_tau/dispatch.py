"""Sequential, process-local call guard for the official user and judge seams.

Prompts, task rewards and judge models are unchanged. Calls are metered and
bounded before dispatch; the original functions are restored on every exit.
"""
from __future__ import annotations

from contextlib import contextmanager
import time


@contextmanager
def guard_official_calls(deadline, *, before_call=None, after_call=None):
    from tau2.user import user_simulator
    from tau2.evaluator import evaluator_communicate, evaluator_nl_assertions
    from tau2.utils.llm_utils import llm_log_dir
    if llm_log_dir.get() is not None:
        raise ValueError("disable upstream request-kwargs logging before model work")
    calls, saved = [], []
    def wrap(original, role):
        def guarded(*args, **kwargs):
            if before_call is not None:
                before_call()
            remaining = deadline.timestamp() - time.time()
            if remaining <= 0:
                raise RuntimeError("deadline reached before official user/judge dispatch")
            kwargs["num_retries"] = 0
            kwargs["response_validation_retries"] = 0
            kwargs["timeout"] = min(kwargs.get("timeout", 60), remaining)
            entry = {"role": role, "model": kwargs.get("model", args[0] if args else None),
                     "status": "dispatched", "input_tokens": None, "output_tokens": None,
                     "cost_usd": None, "reported_cost_usd": None}
            calls.append(entry)
            started = time.perf_counter()
            try:
                response = original(*args, **kwargs)
                usage = response.usage or {}
                entry.update(status="completed", input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"), reported_cost_usd=response.cost)
                return response
            except Exception:
                entry["status"] = "failed_or_uncertain"
                raise
            finally:
                entry["elapsed_seconds"] = time.perf_counter() - started
                if after_call is not None:
                    after_call(entry["elapsed_seconds"])
        return guarded
    try:
        for module, role in ((user_simulator, "official_user_simulator"),
                             (evaluator_communicate, "official_communication_judge"),
                             (evaluator_nl_assertions, "official_nl_assertion_judge")):
            original = module.generate
            saved.append((module, original))
            module.generate = wrap(original, role)
        yield calls
    finally:
        for module, original in reversed(saved):
            module.generate = original
