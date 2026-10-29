"""The deterministic controller and AWS Strands writer/reviewer agents.

No reference answer, task evaluator, reward, or supporting-fact label is an
argument to these components. Only public observations enter the controller.
"""

from __future__ import annotations

import time
from typing import Any


class Agents:
    """Two actual Strands SDK agents sharing an explicitly local model server.

    Agents are reset for each bounded call. The runner supplies the complete
    permitted history; state cannot leak from another question or policy arm.
    """

    def __init__(self, base_url="http://127.0.0.1:8001/v1",
                 model_id="Qwen/Qwen2.5-1.5B-Instruct", max_tokens=768):
        from urllib.parse import urlparse
        url = urlparse(base_url)
        if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("This keyless configuration requires a loopback model server")
        self.base_url, self.model_id = base_url, model_id
        self.max_tokens = max_tokens
        self.usage: list[dict] = []

    @property
    def provenance(self):
        return {"framework": "AWS Strands Agents SDK", "model_id": self.model_id,
                "base_url": self.base_url, "temperature": 0,
                "max_tokens": self.max_tokens, "roles": ["writer", "reviewer"]}

    def generate(self, role: str, system: str, prompt: str) -> str:
        from strands import Agent
        from strands.models.openai import OpenAIModel
        if role not in {"writer", "reviewer"}:
            raise ValueError("Unknown agent role")
        model = OpenAIModel(
            client_args={"api_key": "local", "base_url": self.base_url,
                         "timeout": 180.0, "max_retries": 0},
            model_id=self.model_id,
            params={"temperature": 0, "max_tokens": self.max_tokens},
        )
        agent = Agent(name=role, model=model, system_prompt=system,
                      callback_handler=None, retry_strategy=None)
        started = time.monotonic()
        try:
            result = agent(prompt)
            text = "".join(block.get("text", "")
                           for block in result.message.get("content", []))
            usage = dict(getattr(result.metrics, "accumulated_usage", {}) or {})
            if not text.strip():
                raise ValueError("Model returned no text")
            if getattr(result, "stop_reason", None) in {"max_tokens", "length"}:
                raise ValueError("Model output was truncated at the generation cap")
            self.usage.append({"role": role, "seconds": time.monotonic()-started,
                               "usage": usage, "status": "completed"})
            return text
        except Exception as exc:
            self.usage.append({"role": role, "seconds": time.monotonic()-started,
                               "status": "failed", "error_type": type(exc).__name__})
            raise


class Gate:
    """Frozen transfer AND gate, plus explicit component ablations.

    Shree's frozen head was trained with an LLM critic flag. Substituting a
    Strands flag is an explicitly recorded transfer, never a trained pair.
    Public round_index is ONE based; the source head uses zero based indices.
    """

    def __init__(self, critic=None, head=None, approve_threshold=0.8,
                 stop_threshold=0.5):
        from .linear import LinearHead
        self.critic = critic
        self.head = head or LinearHead.pretrained(seed=0)
        self.approve_threshold = approve_threshold
        self.stop_threshold = stop_threshold

    @property
    def provenance(self):
        return {"policy": "frozen_transfer_AND", "approve_threshold": self.approve_threshold,
                "stop_threshold": self.stop_threshold, "seed": 0,
                "head": getattr(self.head, "provenance", getattr(self.head, "metadata", {})),
                "critic": getattr(self.critic, "provenance", {}),
                "new_training": False,
                "prior_evidence": "Vedanshi and Shree only; no Yash results",
                "feature_transfer": "LLM critic flag replaced by Strands flag in combined arm"}

    def decide(self, arm: str, draft: str, previous: str | None,
               round_index: int, max_rounds: int, question: str,
               passages: list[str], previous_distance: float = 0,
               critic_approved: bool = False) -> dict[str, Any]:
        import math
        from .linear import make_features
        if not 1 <= round_index <= max_rounds:
            raise ValueError("round_index is one based and must fit the round budget")
        aliases = {"fixed1": "fixed1", "baseline": "fixed1", "fixed6": "fixed6",
                   "linear": "linear", "strands": "strands", "combined": "combined"}
        if arm not in aliases:
            raise ValueError(f"Unknown policy arm: {arm}")
        mode = aliases[arm]
        p_approve = None
        flag = bool(critic_approved)
        critic_record = None
        if mode in {"strands", "combined"}:
            if self.critic is None:
                from .critic import Critic
                self.critic = Critic()
            p_approve = float(self.critic.score(question, passages, draft))
            if not math.isfinite(p_approve) or not 0 <= p_approve <= 1:
                raise ValueError("Critic probability is invalid")
            flag = p_approve >= self.approve_threshold
            critic_record = getattr(self.critic, "last_record", None)
        features = make_features(draft, previous, round_index - 1,
                                 max_rounds, flag, previous_distance)
        p_stop = float(self.head.probability(features))
        if not math.isfinite(p_stop) or not 0 <= p_stop <= 1:
            raise ValueError("Linear-head probability is invalid")
        head_stop = p_stop >= self.stop_threshold
        stop = {"fixed1": True, "fixed6": round_index == max_rounds,
                "linear": head_stop, "strands": flag,
                "combined": head_stop and flag}[mode]
        return {"stop": stop, "p_stop": p_stop, "p_approve": p_approve,
                "features": features, "distance": features[0],
                "reason": (f"{mode}_approved" if stop else f"{mode}_continue"),
                "critic": critic_record}
