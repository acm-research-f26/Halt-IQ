"""Loopback OpenAI-compatible transport for the pinned, local Qwen writer.

Both AWS Strands agents and the official Tau user/evaluators share this model.
No tool is executed here: tool requests are data returned to Tau's orchestrator.
"""
from copy import deepcopy
import json
import re
import time
import uuid
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ATTENTION_IMPLEMENTATION = "sdpa"
PREFILL_CHUNK_SIZE = 512
LOGITS_TO_KEEP = 1


def normalize_messages(messages):
    """Convert text-only OpenAI chat messages to the pinned Qwen template.

    Strands sends standard OpenAI text-block lists. Qwen2.5's template requires
    string content and serializes tool arguments itself, so arguments must be
    objects rather than JSON strings. The caller's request remains unchanged.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("A nonempty list of chat messages is required")
    normalized = deepcopy(messages)
    for message in normalized:
        if not isinstance(message, dict) or message.get("role") not in {
                "system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported chat message role")
        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif isinstance(content, str):
            pass
        elif isinstance(content, list):
            parts = []
            for block in content:
                if (not isinstance(block, dict) or block.get("type") != "text"
                        or not isinstance(block.get("text"), str)):
                    raise ValueError("This local writer supports only OpenAI text content blocks")
                parts.append(block["text"])
            message["content"] = "".join(parts)
        else:
            raise ValueError("Chat content must be text, text blocks, or null")
        calls = message.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("Invalid tool call")
            function = call.get("function", call)
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ValueError("Invalid tool-call function")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call arguments must be a JSON object")
            function["arguments"] = arguments
    return normalized


def parse_tool_calls(text):
    """Translate Qwen's documented tool-call markup without evaluating code."""
    calls = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        item = json.loads(match.group(1))
        args = item.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(item.get("name"), str) or not isinstance(args, dict):
            raise ValueError("Invalid generated tool call")
        calls.append({"id": "call_" + uuid.uuid4().hex,
                      "type": "function", "function": {
                          "name": item["name"], "arguments": json.dumps(args)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    return content or None, calls


class Writer:
    def __init__(self, output, device="cuda", max_context=16384):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.output, self.max_context = Path(output), max_context
        self.output.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=dtype,
            attn_implementation=ATTENTION_IMPLEMENTATION).to(device).eval()
        self.device = device
        self.metadata = {"model": MODEL, "revision": REVISION,
                         "dtype": str(dtype), "device": device,
                         "max_context": max_context,
                         "context_policy": "reject_overflow_without_truncation",
                         "attention_implementation": self.model.config._attn_implementation,
                         "prefill_chunk_size": PREFILL_CHUNK_SIZE,
                         "logits_to_keep": LOGITS_TO_KEEP, "use_cache": True,
                         "torch": torch.__version__,
                         "transformers": transformers.__version__, "new_training": False}
        (self.output / "writer.json").write_text(json.dumps(self.metadata, indent=2))

    def complete(self, body):
        import torch
        messages = normalize_messages(body["messages"])
        response_format = body.get("response_format", {})
        if response_format.get("type") in {"json_object", "json_schema"}:
            schema = response_format.get("json_schema", {}).get("schema")
            instruction = "Return only one valid JSON object."
            if schema:
                instruction += " It must satisfy this JSON schema: " + json.dumps(schema)
            messages.insert(0, {"role": "system", "content": instruction})
        encoded = self.tokenizer.apply_chat_template(
            messages, tools=body.get("tools") or None,
            add_generation_prompt=True, tokenize=True, return_tensors="pt",
            return_dict=True).to(self.device)
        prompt_tokens = encoded["input_ids"].shape[-1]
        wanted = int(body.get("max_completion_tokens") or body.get("max_tokens") or 768)
        max_new = min(wanted, 2048)
        if prompt_tokens + max_new > self.max_context:
            raise ValueError(f"Context overflow: {prompt_tokens}+{max_new}>{self.max_context}; no silent truncation")
        started = time.monotonic()
        with torch.inference_mode():
            # Process every prompt token while bounding temporary prefill tensors.
            # The cumulative KV cache preserves the complete conversation.
            result = self.model.generate(**encoded, max_new_tokens=max_new,
                                         do_sample=False,
                                         prefill_chunk_size=PREFILL_CHUNK_SIZE,
                                         logits_to_keep=LOGITS_TO_KEEP, use_cache=True,
                                         pad_token_id=self.tokenizer.eos_token_id)
        ids = result[0, prompt_tokens:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        content, calls = parse_tool_calls(text)
        finish = ("length" if len(ids) == max_new else "tool_calls" if calls else "stop")
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(ids),
                 "total_tokens": prompt_tokens + len(ids)}
        rid = "chatcmpl-" + uuid.uuid4().hex
        record = {"id": rid, "timestamp": time.time(), "usage": usage,
                  "seconds": time.monotonic() - started, "finish_reason": finish,
                  "request": body, "response_text": text}
        with (self.output / "model_requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        return {"id": rid, "object": "chat.completion", "created": int(time.time()),
                "model": MODEL, "choices": [{"index": 0, "message": message,
                    "finish_reason": finish}], "usage": usage}


def stream_response(response):
    """OpenAI SSE transport, including token usage for SDK accounting."""
    base = {k: response[k] for k in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    choice = response["choices"][0]
    delta = dict(choice["message"])
    if delta.get("tool_calls"):
        delta["tool_calls"] = [dict(c, index=i) for i, c in enumerate(delta["tool_calls"])]
    yield "data: " + json.dumps(dict(base, choices=[{"index": 0, "delta": delta,
                                                      "finish_reason": None}])) + "\n\n"
    yield "data: " + json.dumps(dict(base, choices=[{"index": 0, "delta": {},
                        "finish_reason": choice["finish_reason"]}])) + "\n\n"
    yield "data: " + json.dumps(dict(base, choices=[], usage=response["usage"])) + "\n\n"
    yield "data: [DONE]\n\n"


def serve(output, port=8001, device="cuda"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    writer = Writer(output, device)
    app = FastAPI()

    @app.get("/health")
    def health():
        return writer.metadata

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        try:
            response = writer.complete(body)
        except Exception as exc:
            return JSONResponse({"error": {"message": str(exc),
                                          "type": type(exc).__name__}}, status_code=400)
        if body.get("stream"):
            return StreamingResponse(stream_response(response), media_type="text/event-stream")
        return response

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
