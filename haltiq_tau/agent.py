"""Optional actual Tau2 HalfDuplexAgent extension, loaded only explicitly.

The Tau orchestrator remains the sole executor of tools. This adapter receives
no task object, hidden simulator scenario, DB snapshot or evaluator criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

from tau2.agent.base_agent import AgentError, HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage

from .core import Action, Controller, RejectedAction, RouteFailure, public_message


def action_from_tau(message: AssistantMessage) -> Action:
    if not isinstance(message, AssistantMessage) or message.is_audio:
        raise ValueError("only a text-mode assistant proposal is supported")
    calls = []
    for call in message.tool_calls or []:
        if call.requestor != "assistant":
            raise ValueError("assistant proposals cannot call user tools")
        calls.append({"id": call.id, "name": call.name, "arguments": call.arguments})
    return Action(message.content, tuple(calls))


def message_from_tau(message):
    if isinstance(message, AssistantMessage):
        return {"role": "assistant", **action_from_tau(message).to_dict()}
    if isinstance(message, UserMessage):
        if message.is_audio or message.tool_calls:
            raise ValueError("private simulator tool calls/audio cannot enter the agent state")
        return public_message({"role": "user", "content": message.content})
    if isinstance(message, ToolMessage) and message.requestor == "assistant":
        return public_message({"role": "tool", "content": message.content,
                               "tool_call_id": message.id, "error": message.error})
    raise ValueError("input is not a delivered public agent observation")


@dataclass
class AgentState:
    messages: list[dict] = field(default_factory=list)
    route_events: list[dict] = field(default_factory=list)


class TauPolicyAgent(HalfDuplexAgent[AgentState]):
    """A writer/reviewer team whose proposed action is gated before execution."""
    def __init__(self, tools, domain_policy, controller: Controller):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.controller = controller
        self.public_tools = []
        for tool in tools:
            schema = tool.openai_schema
            function = schema.get("function", schema)
            self.public_tools.append({"name": function["name"],
                                      "description": function.get("description", ""),
                                      "parameters": function["parameters"]})
        self.physical_route_events = []
        self.failure_category = None

    def set_seed(self, seed):
        for component in (self.controller.writer, self.controller.reviewer, self.controller.gate):
            method = getattr(component, "set_seed", None)
            if method is not None:
                method(seed)

    def stop(self, message=None, state=None):
        close = getattr(self.controller.gate, "close", None)
        if close is not None:
            close()

    def get_init_state(self, message_history=None):
        return AgentState(messages=[message_from_tau(message) for message in message_history or []])

    def generate_next_message(self, message, state):
        delivered = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        state.messages.extend(message_from_tau(item) for item in delivered)
        try:
            result = self.controller.route(self.domain_policy, state.messages, self.public_tools)
        except RejectedAction as exc:
            self.failure_category = "known_rejection"
            state.route_events.extend(deepcopy(exc.events))
            self.physical_route_events.extend(deepcopy(exc.events))
            raise AgentError(str(exc)) from exc
        except RouteFailure as exc:
            self.failure_category = "controller_provider_error"
            state.route_events.extend(deepcopy(exc.events))
            self.physical_route_events.extend(deepcopy(exc.events))
            raise AgentError(str(exc)) from exc
        except Exception as exc:
            self.failure_category = "controller_provider_error"
            # Preserve the actual error as an agent failure, never silently execute.
            raise AgentError(f"policy-action controller failed: {type(exc).__name__}") from exc
        state.route_events.extend(deepcopy(result.events))
        self.physical_route_events.extend(deepcopy(result.events))
        action = result.action
        response = AssistantMessage(role="assistant", content=action.content,
                                    tool_calls=[ToolCall(**row) for row in action.tool_calls] or None)
        state.messages.append({"role": "assistant", **action.to_dict()})
        return response, state
