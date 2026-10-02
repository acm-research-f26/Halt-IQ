"""Pluggable critics. Each takes one draft dict (see drafts.py) and returns:

- a float = P(answer is correct) in [0, 1]   (probability critics), or
- "approve" / "revise"                        (text-only critics), or
- None = no score available for this draft (reused critics only; nothing is cached).

A critic may return (value, seconds) to report its own timing; otherwise the
harness times the call. To add a critic, write a function and add it to CRITICS.
"""

from drafts import saved_reviews

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


CRITICS = {
    "llm": reused("llm"),                  # qwen3:8b approve/revise
    "local-typed": reused("local-typed"),  # qwen3:8b, 3 self-reported scores
    "kev": reused("openjev"),              # Kev-0.8B via the openjev arm
}
