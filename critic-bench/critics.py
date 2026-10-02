"""Pluggable critics. Each takes one draft dict (see drafts.py) and returns:

- a float = P(answer is correct) in [0, 1]   (probability critics), or
- "approve" / "revise"                        (text-only critics), or
- None = no score available for this draft (reused critics only; nothing is cached).

A critic may return (value, seconds) to report its own timing; otherwise the
harness times the call. To add a critic, write a function and add it to CRITICS.
"""

from drafts import normalize, read_jsonl, saved_reviews

CHECKS = ("supported", "complete", "relevant")


# --- Reused critics: decisions saved in Yash's Sept 28 run (no model calls) ---

def reused(arm):
    reviews = None

    def critic(draft):
        nonlocal reviews
        reviews = reviews or saved_reviews(arm)
        review = reviews.get(draft["draft_id"])
        if review is None:
            return None  # this arm never saw this draft
        if review["scores"] is None:  # llm critic: approve/revise only
            value = "approve" if review["approved"] else "revise"
        else:  # typed critics: the loop approves only if all 3 checks pass, so use the min
            value = min(review["scores"][check] for check in CHECKS)
        return value, review["seconds"]
    return critic


# --- Live versions of the same critics, for drafts the run never judged (e.g. `extra`) ---
# They call Yash's own critic classes with the Sept 28 run's settings, so prompts match.

def live(arm):
    critic = None

    def judge(draft):
        nonlocal critic
        if critic is None:
            import json
            from drafts import RUN_DIR
            from haltiq.critics import JevCritic, LLMCritic
            from haltiq.providers import JevClient, OllamaClient
            config = json.loads((RUN_DIR / "manifest.json").read_text())["config"]
            if arm == "openjev":
                critic = JevCritic(JevClient(base_url=config["openjev_url"], model=config["openjev_model"],
                                             timeout=config["timeout"]), config["threshold"])
            else:
                critic = LLMCritic(OllamaClient(base_url=config["ollama_url"], model=config["critic_model"],
                                                seed=config["seed"], timeout=config["timeout"],
                                                num_ctx=config["context_tokens"]))
        review = critic.review({"question": draft["question"], "evidence": draft["evidence"], "draft": draft["draft"]})
        value = min(review.scores.values()) if review.scores else ("approve" if review.approved else "revise")
        return value, review.call.latency_ms / 1000
    return judge


def saved_then_live(arm):
    """Reuse the run's saved decision when there is one; otherwise call the critic live."""
    saved, fresh = reused(arm), live(arm)
    return lambda draft: saved(draft) or fresh(draft)


# --- No-model critics ---

def evidence_match(draft):
    """1 if the normalized answer appears word-for-word in the evidence the critic sees, else 0.

    A yes/no answer never appears verbatim, so it gets 0.5 (no information). UNKNOWN gets 0.
    Uses all evidence paragraphs, not gold supporting facts (those are evaluation-only).
    """
    answer = normalize(draft["draft"])
    if answer == "unknown":
        return 0.0  # a non-answer, even if the word "unknown" appears in the evidence
    if answer in ("yes", "no"):
        return 0.5
    evidence = " ".join(normalize(paragraph) for paragraph in draft["evidence"])
    return 1.0 if answer and f" {answer} " in f" {evidence} " else 0.0


_samples = None


def consistency(draft):
    """Share of 3 temperature-0.7 writer samples whose normalized answer equals this first draft.

    Only defined for first drafts of the 80 sampled questions (see make_samples.py).
    """
    global _samples
    if _samples is None:
        from make_samples import SAMPLES
        _samples = {}
        for row in read_jsonl(SAMPLES) if SAMPLES.exists() else []:
            _samples.setdefault(row["task_id"], []).append(row)
    samples = _samples.get(draft["task_id"], [])
    if not draft["is_first"] or len(samples) < 3:
        return None  # not sampled (yet); nothing is cached
    agree = sum(normalize(s["answer"]) == normalize(draft["draft"]) for s in samples) / len(samples)
    return agree, sum(s["seconds"] for s in samples)  # cost = the 3 extra writer calls


def laya(draft):
    from laya_critic import laya as run  # imported lazily so other critics don't load MLX
    return run(draft)


CRITICS = {
    "laya": laya,                          # Laya noul: P(correct and supported)
    "evidence_match": evidence_match,      # answer string found in evidence (no model)
    "consistency": consistency,            # agreement of 3 resampled writer answers
    "llm": saved_then_live("llm"),         # qwen3:8b approve/revise
    "local-typed": reused("local-typed"),  # qwen3:8b, 3 self-reported scores (saved only)
    "kev": saved_then_live("openjev"),     # Kev-0.8B via the openjev arm
}
