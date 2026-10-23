"""Optional Tau/LiteLLM writer and independent reviewer with physical call meters."""
from __future__ import annotations

import json
import time

from .agent import action_from_tau
from .core import PublicState


def _messages(rows):
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
    result = []
    for row in rows:
        if row["role"] == "user":
            result.append(UserMessage.text(row["content"]))
        elif row["role"] == "assistant":
            result.append(AssistantMessage(role="assistant", content=row["content"],
                          tool_calls=[ToolCall(**call) for call in row["tool_calls"]] or None))
        elif row["role"] == "tool":
            result.append(ToolMessage(role="tool", id=row["tool_call_id"], content=row["content"],
                                      error=row["error"], requestor="assistant"))
        else:
            raise ValueError("only public delivered messages are allowed")
    return result


class MeteredLLM:
    """Never retries. Unknown monetary pricing is null, not a claim of free compute."""
    def __init__(self, model, *, allow_model_calls=False, model_args=None, pricing_verified=False, deadline=None,
                 before_call=None, after_call=None):
        if not allow_model_calls:
            raise ValueError("generation needs explicit allow_model_calls")
        if not isinstance(model, str) or not model:
            raise ValueError("explicit model required")
        args = dict(model_args or {})
        allowed = {"temperature", "seed", "timeout", "max_tokens", "max_completion_tokens", "api_base", "reasoning_effort",
                   "num_thread", "num_ctx", "num_gpu", "think", "keep_alive"}
        if set(args) - allowed:
            raise ValueError("unsupported model arguments; credentials belong only in the process environment")
        if any(key in args for key in ("num_thread", "num_ctx", "num_gpu", "think", "keep_alive")):
            if not model.startswith("ollama_chat/"):
                raise ValueError("Ollama resource options require an explicit ollama_chat model")
            if type(args.get("num_thread", 4)) is not int or not 1 <= args.get("num_thread", 4) <= 4:
                raise ValueError("local generation uses at most four CPU threads")
            if args.get("num_ctx", 16384) != 16384:
                raise ValueError("the local controlled pilot uses the declared 16384-token context")
            if "num_gpu" in args and (type(args["num_gpu"]) is not int or args["num_gpu"] < 0):
                raise ValueError("num_gpu must be an explicit nonnegative integer")
            if "think" in args and args["think"] is not False:
                raise ValueError("the local controlled pilot disables thinking")
        args.update(num_retries=0)
        self.model, self.args, self.pricing_verified = model, args, pricing_verified
        self.deadline = deadline
        self.before_call, self.after_call = before_call, after_call
        self.calls = []

    def set_seed(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("nonnegative integer seed required")
        self.args["seed"] = seed

    def generate(self, messages, *, tools=None, call_name):
        from tau2.utils.llm_utils import generate, llm_log_dir
        if llm_log_dir.get() is not None:
            raise ValueError("disable upstream request-kwargs logging before model work")
        if self.before_call is not None:
            self.before_call()
        started = time.perf_counter()
        args = dict(self.args)
        if self.deadline is not None:
            remaining = self.deadline.timestamp() - time.time()
            if remaining <= 0:
                raise RuntimeError("model deadline reached before dispatch")
            args["timeout"] = min(args.get("timeout", 60), remaining)
        entry = {"role": call_name, "model": self.model, "status": "dispatched",
                 "input_tokens": None, "output_tokens": None, "cost_usd": None}
        self.calls.append(entry)
        try:
            message = generate(self.model, messages, tools=tools, call_name=call_name,
                               response_validation_retries=0, **args)
            usage = message.usage or {}
            entry.update(status="completed", input_tokens=usage.get("prompt_tokens"),
                         output_tokens=usage.get("completion_tokens"),
                         cost_usd=message.cost if self.pricing_verified else None,
                         elapsed_seconds=time.perf_counter() - started)
            return message
        except Exception:
            entry.update(status="failed_or_uncertain", elapsed_seconds=time.perf_counter() - started)
            raise
        finally:
            if self.after_call is not None:
                self.after_call(entry["elapsed_seconds"])


class TauWriter(MeteredLLM):
    def __init__(self, model, tools, **kwargs):
        super().__init__(model, **kwargs)
        self.tools = tools

    def __call__(self, policy, messages, tools, feedback):
        from tau2.data_model.message import SystemMessage
        instruction = ("You are a customer-service agent. Follow the domain policy and use only the available tools. "
                       "Each response must contain either a user-facing message or tool calls, never both. "
                       "Do not claim an action succeeded before observing its tool result.\n\nPOLICY:\n" + policy)
        prompt = [SystemMessage(role="system", content=instruction), *_messages(messages)]
        if feedback is not None:
            prompt.append(SystemMessage(role="system", content="Internal independent policy review:\n" + feedback +
                           "\nProduce a corrected NEXT action. The rejected proposal was not executed or sent to the user."))
        return action_from_tau(self.generate(prompt, tools=self.tools, call_name="writer"))


class TauReviewer(MeteredLLM):
    def __call__(self, state: PublicState):
        from tau2.data_model.message import SystemMessage, UserMessage
        prompt = [SystemMessage(role="system", content=(
            "Review the proposed NEXT action against the supplied public policy, tool schemas and observed transcript. "
            "Give short, specific corrective feedback identifying missing evidence or policy constraints. "
            "Do not invent tool outputs or hidden user facts; you have no access to hidden evaluation targets.")),
            UserMessage.text(json.dumps(state.to_dict(), ensure_ascii=False, allow_nan=False))]
        message = self.generate(prompt, call_name="independent_reviewer")
        if message.tool_calls or not message.content or not message.content.strip():
            raise ValueError("reviewer must return actionable text, not tool calls")
        return message.content
