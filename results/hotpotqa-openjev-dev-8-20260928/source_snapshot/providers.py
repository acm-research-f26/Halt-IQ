"""Small HTTP clients for local Ollama and the TypeSafe-compatible Jev API.

No retries are automatic: a timeout may already have incurred a charge. HTTP
redirects are blocked, and API credentials are never included in error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .budget import BudgetLedger, JEV_INPUT_RATE, JEV_MAX_INPUT_TOKENS


class ProviderError(RuntimeError):
    """An endpoint, request, or model response could not be used safely."""


@dataclass(frozen=True)
class CallResult:
    data: dict
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model: str
    context_tokens: int | None = None
    context_budget_estimate_tokens: int | None = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _endpoint(base_url: str, *, official_allowed: bool) -> tuple[str, bool]:
    try:
        parsed = urlsplit(base_url)
        host, port = parsed.hostname, parsed.port
    except (TypeError, ValueError):
        raise ProviderError("Invalid provider base URL") from None
    if (
        not host
        or parsed.scheme not in ("http", "https")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ProviderError("Use an http(s) API root URL without a path, credentials, or query")
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "localhost"
    official = host == "api.typesafe.ai" and parsed.scheme == "https" and port in (None, 443)
    if not local and not (official_allowed and official):
        raise ProviderError("Only loopback local servers and the official TypeSafe HTTPS API are allowed")
    return base_url.rstrip("/"), local


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ProviderError("timeout must be a finite positive number")
    return timeout


def _validate_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ProviderError("model must be a nonempty string")
    return model.strip()


def _integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProviderError(f"Response {name} must be a nonnegative integer")
    return value


def _reject_nonfinite(value: str):
    raise ValueError("non-finite JSON value")


def _json_object(raw: str | bytes, name: str) -> dict:
    try:
        result = json.loads(raw, parse_constant=_reject_nonfinite)
    except (ValueError, UnicodeDecodeError):
        raise ProviderError(f"{name} was not valid JSON") from None
    if not isinstance(result, dict):
        raise ProviderError(f"{name} must be a JSON object")
    return result


def _encode(payload: dict) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError):
        raise ProviderError("Request must contain only JSON-compatible finite values") from None


def _http_json(url: str, body: bytes | None, headers: dict, timeout: float, local: bool) -> dict:
    # A configured proxy must not turn a local-only run into a remote request.
    handlers = [_NoRedirect()]
    if local:
        handlers.append(ProxyHandler({}))
    opener = build_opener(*handlers)
    request = Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST" if body is not None else "GET")
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(8_000_001)
    except HTTPError as error:
        code = error.code
        error.close()
        raise ProviderError(f"Provider returned HTTP {code}; no automatic retry was made") from None
    except (URLError, TimeoutError, OSError):
        raise ProviderError("Provider connection failed or timed out; no automatic retry was made") from None
    if len(raw) > 8_000_000:
        raise ProviderError("Provider response exceeded the 8 MB safety limit")
    result = _json_object(raw, "Provider response")
    if "error" in result:
        raise ProviderError("Provider returned an error response")
    return result


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        seed: int = 42,
        timeout: float = 180,
        num_ctx: int = 8192,
    ) -> None:
        self.base_url, self.local = _endpoint(base_url, official_allowed=False)
        self.model = _validate_model(model)
        if "cloud" in self.model.lower():
            raise ProviderError("Cloud model names are disabled; use a downloaded local Ollama model")
        if type(seed) is not int:
            raise ProviderError("seed must be an integer")
        self.seed = seed
        self.timeout = _validate_timeout(timeout)
        if type(num_ctx) is not int or not 2048 <= num_ctx <= 65536:
            raise ProviderError("num_ctx must be an integer from 2048 to 65536")
        self.num_ctx = num_ctx

    def _ensure_local_model(self) -> None:
        # Cloud aliases can have innocuous names. Inspect metadata before each
        # generation rather than relying only on a ':cloud' spelling check.
        tags = _http_json(self.base_url + "/api/tags", None, {}, self.timeout, self.local)
        models = tags.get("models")
        if not isinstance(models, list):
            raise ProviderError("Ollama did not return an installed model list")
        names = {self.model}
        if ":" not in self.model.rsplit("/", 1)[-1]:
            names.add(self.model + ":latest")
        matching = [model for model in models if isinstance(model, dict) and (model.get("name") in names or model.get("model") in names)]
        if not matching:
            raise ProviderError(f"Ollama model {self.model!r} is not installed; pull it before running")
        for model in matching:
            if model.get("remote_host") or model.get("remote_model"):
                raise ProviderError("Ollama model is a remote alias; use a downloaded local model")
            if type(model.get("size")) is not int or model["size"] <= 0:
                raise ProviderError("Ollama model has no local weights; use a downloaded local model")

    def chat(self, messages: list[dict], schema: dict | None = None) -> CallResult:
        if not isinstance(messages, list) or not messages:
            raise ProviderError("messages must be a nonempty list")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str) or message.get("role") not in ("system", "user", "assistant"):
                raise ProviderError("Every message must have a supported role and string content")
        if schema is not None and not isinstance(schema, dict):
            raise ProviderError("schema must be a JSON schema object")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": self.seed, "num_predict": 512, "num_ctx": self.num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        body = _encode(payload)
        # For the selected Qwen byte-level tokenizer, count one token per UTF-8
        # byte of the entire JSON request, including the schema, then reserve
        # 1024 tokens for chat framing and 512 for output. This intentionally
        # conservative budget estimate is not an exact tokenizer or a guarantee
        # for arbitrary model templates. Reject instead of truncating evidence.
        context_budget_estimate = len(body) + 1024 + 512
        if context_budget_estimate > self.num_ctx:
            raise ProviderError(
                f"Ollama request exceeds the conservative context budget: "
                f"{context_budget_estimate} estimated tokens including framing and output "
                f"> {self.num_ctx} configured tokens. Choose a larger --context-tokens "
                "value (up to 65536); evidence was not truncated and no request was sent."
            )
        self._ensure_local_model()
        started = time.perf_counter()
        result = _http_json(self.base_url + "/api/chat", body, {}, self.timeout, self.local)
        latency_ms = (time.perf_counter() - started) * 1000
        if result.get("done") is not True or result.get("done_reason") in ("length", "max_tokens", "max_length"):
            raise ProviderError("Ollama output was unfinished or truncated")
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama returned an empty message")
        data = _json_object(content, "Ollama structured output") if schema is not None else {"text": content}
        input_tokens = _integer(result.get("prompt_eval_count"), "prompt_eval_count")
        output_tokens = _integer(result.get("eval_count"), "eval_count")
        response_model = _validate_model(result.get("model"))
        if "cloud" in response_model.lower() or result.get("remote_host"):
            raise ProviderError("Ollama reported a remote model; use OLLAMA_NO_CLOUD=1 on the server")
        return CallResult(
            data, input_tokens, output_tokens, latency_ms, 0.0, response_model,
            context_tokens=self.num_ctx,
            context_budget_estimate_tokens=context_budget_estimate,
        )


class JevClient:
    def __init__(
        self,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-1.13.0",
        api_key: str | None = None,
        budget: BudgetLedger | None = None,
        timeout: float = 60,
    ) -> None:
        self.base_url, self.local = _endpoint(base_url, official_allowed=True)
        self.model = _validate_model(model)
        self.timeout = _validate_timeout(timeout)
        self.budget = budget
        # Never forward a cloud credential to a local community implementation.
        self._api_key = api_key if self.local else api_key or os.environ.get("TYPESAFE_API_KEY")
        if self._api_key is not None:
            if not isinstance(self._api_key, str) or not self._api_key or any(ord(c) < 33 or ord(c) > 126 for c in self._api_key):
                raise ProviderError("API key must be a nonempty ASCII value without whitespace")
        if not self.local:
            if not self._api_key:
                raise ProviderError("Set TYPESAFE_API_KEY to use the official Jev API")
            if self.model != "jev-1.13.0":
                raise ProviderError("Paid mode pins jev-1.13.0 to its verified price and context limit")
            if budget is None:
                raise ProviderError("Paid Jev calls require an explicit persistent BudgetLedger")
            if budget.rate_per_million < JEV_INPUT_RATE:
                raise ProviderError("Budget token rate is below the verified Jev input price")

    def judge(self, state: dict, questions: dict) -> CallResult:
        if not isinstance(state, dict) or not isinstance(questions, dict) or not questions:
            raise ProviderError("state must be an object and questions a nonempty object")
        for key, question in questions.items():
            if not isinstance(key, str) or not isinstance(question, dict) or question.get("type") != "noul" or not question.get("instructions"):
                raise ProviderError("This critic adapter requires named Noul questions with instructions")
        body = _encode({"model": self.model, "state": state, "questions": questions})
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        reservation = self.budget.reserve(JEV_MAX_INPUT_TOKENS) if not self.local else None
        started = time.perf_counter()
        result = _http_json(self.base_url + "/v1/systemone", body, headers, self.timeout, self.local)
        latency_ms = (time.perf_counter() - started) * 1000
        answers, usage = result.get("answers"), result.get("usage")
        if not isinstance(answers, dict) or not isinstance(usage, dict):
            raise ProviderError("Jev response must include answers and usage objects")
        for key in questions:
            answer = answers.get(key)
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise ProviderError(f"Jev response is missing a Noul answer for {key!r}")
            probability = answer.get("noul")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ProviderError(f"Jev Noul answer for {key!r} must be a finite probability in [0, 1]")
        input_tokens = _integer(usage.get("input_tokens"), "input_tokens")
        output_tokens = _integer(usage.get("output_tokens"), "output_tokens")
        response_model = _validate_model(result.get("model"))
        if not self.local and response_model != self.model:
            raise ProviderError("Jev returned a different model than the pinned paid version")
        cost = self.budget.settle(reservation, input_tokens) if reservation is not None else 0.0
        return CallResult(result, input_tokens, output_tokens, latency_ms, cost, response_model)
