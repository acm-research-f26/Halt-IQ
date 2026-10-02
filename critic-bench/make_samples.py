"""Extra writer samples for the consistency critic.

    python3 make_samples.py          # subset80 questions; resumable, appends to samples/consistency.jsonl
    python3 make_samples.py extra    # also every remaining `extra` question

For each question (by default the 80 in subset80: the 20 first-draft questions + the first 60 of `extra`),
draw 3 more answers with Yash's unmodified Writer (same model, prompt, and schema)
at temperature 0.7. His OllamaClient hardcodes temperature 0 and seed 42, so we
pass the Writer a small client that sends the same request with temperature 0.7
and a different seed per sample (with one fixed seed all 3 samples would be identical).
"""

import json
import sys
import time

from drafts import EXTRA_DIR, RUN_DIR, read_jsonl, subset80_task_ids, HERE

from haltiq.critics import Writer
from haltiq.providers import CallResult, OllamaClient, _encode, _http_json

SAMPLES = HERE / "samples" / "consistency.jsonl"
SEEDS = (1, 2, 3)
TEMPERATURE = 0.7
CONFIG = json.loads((RUN_DIR / "manifest.json").read_text())["config"]


class SamplingClient(OllamaClient):
    """Same request as OllamaClient.chat, but with temperature 0.7 and a chosen seed."""

    def chat(self, messages, schema=None):
        payload = {"model": self.model, "messages": messages, "stream": False, "think": False, "format": schema,
                   "options": {"temperature": TEMPERATURE, "seed": self.seed, "num_predict": 512,
                               "num_ctx": self.num_ctx}}
        started = time.perf_counter()
        result = _http_json(self.base_url + "/api/chat", _encode(payload), {}, self.timeout, self.local)
        return CallResult(json.loads(result["message"]["content"]), result["prompt_eval_count"],
                          result["eval_count"], (time.perf_counter() - started) * 1000, 0.0, result["model"])


def main(which="subset80"):
    tasks = {t["id"]: t for t in read_jsonl(RUN_DIR / "dataset_snapshot.jsonl") if t["split"] == "dev"}
    tasks.update({t["id"]: t for t in read_jsonl(HERE / "extra" / "tasks.jsonl")})
    done = {(r["task_id"], r["seed"]) for r in read_jsonl(SAMPLES)} if SAMPLES.exists() else set()
    task_ids = subset80_task_ids()
    if which == "extra":
        task_ids += [t["id"] for t in read_jsonl(EXTRA_DIR / "tasks.jsonl")[60:]]
    todo = [(task_id, seed) for task_id in task_ids for seed in SEEDS if (task_id, seed) not in done]
    print(f"{len(done)} samples cached, {len(todo)} to draw", flush=True)
    SAMPLES.parent.mkdir(exist_ok=True)
    with SAMPLES.open("a", encoding="utf-8") as out:
        for i, (task_id, seed) in enumerate(todo, 1):
            writer = Writer(SamplingClient(base_url=CONFIG["ollama_url"], model=CONFIG["writer_model"], seed=seed,
                                           timeout=CONFIG["timeout"], num_ctx=CONFIG["context_tokens"]))
            task = tasks[task_id]
            try:
                answer, call = writer.write({"question": task["question"], "evidence": task["evidence"]})
            except Exception as error:
                print(f"  [{i}/{len(todo)}] {task_id} seed {seed}: {type(error).__name__}: {error}", flush=True)
                continue
            out.write(json.dumps({"task_id": task_id, "seed": seed, "temperature": TEMPERATURE, "answer": answer,
                                  "seconds": round(call.latency_ms / 1000, 3)}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{i}/{len(todo)}] {task_id} seed {seed}: {answer!r} ({call.latency_ms / 1000:.1f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
