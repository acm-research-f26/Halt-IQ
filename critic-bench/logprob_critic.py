"""qwen3:8b logprob critic: P("yes") for "Is the proposed answer correct and supported by the evidence?"

Thinking is off, so the first output token is the answer. We read that token's
top-20 logprobs from Ollama and normalize: P(yes) / (P(yes) + P(no)), summing
case and spacing variants ("Yes", " yes", ...). Same model, seed, temperature 0,
and context size as the Sept 28 run; the state is built with Yash's public_state.
"""

import json
import math
import time
import urllib.request

from drafts import RUN_DIR

from haltiq.critics import BOUNDARY, public_state

CONFIG = json.loads((RUN_DIR / "manifest.json").read_text())["config"]
MODEL = "qwen3:8b"
SYSTEM = ("You are a careful evidence-grounded answer critic. " + BOUNDARY +
          "Question for you: Is the proposed answer correct and supported by the evidence? "
          "Reply with exactly one word: yes or no.")


def ask(draft):
    """One Ollama call; returns (P(yes) normalized over yes/no, seconds, first token)."""
    state = public_state({"question": draft["question"], "evidence": draft["evidence"], "draft": draft["draft"]})
    payload = {"model": MODEL, "stream": False, "think": False, "logprobs": True, "top_logprobs": 20,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(state)}],
               "options": {"temperature": 0, "seed": CONFIG["seed"], "num_predict": 1, "num_ctx": CONFIG["context_tokens"]}}
    request = urllib.request.Request(CONFIG["ollama_url"] + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=CONFIG["timeout"]) as response:
        result = json.loads(response.read())
    seconds = time.perf_counter() - started
    first = result["logprobs"][0]
    mass = {"yes": 0.0, "no": 0.0}
    for option in first["top_logprobs"]:
        word = option["token"].strip().lower()
        if word in mass:
            mass[word] += math.exp(option["logprob"])
    total = mass["yes"] + mass["no"]
    return (mass["yes"] / total if total else 0.5), seconds, first["token"]


def logprob(draft):
    p, seconds, _ = ask(draft)
    return p, seconds
