"""Laya (laya_mlx 0.3.0, model aac6fef/laya-mlx) as a yes/no correctness critic.

Laya reads at most 512 tokens: the question prefix, then the JSON state. It cuts
a dict state from the END, so we put the proposed answer before the evidence and
trim the evidence ourselves, keeping the sentences most relevant to the answer.

    python3 laya_critic.py measure   # token lengths vs the 512 limit (tokenizer only)
"""

import json
from pathlib import Path

from drafts import load_drafts, normalize

MODEL = "aac6fef/laya-mlx"
QUESTION = {
    "type": "noul",
    "instructions": "Is the proposed answer correct and supported by the evidence?",
    "criteria": {"true": "The proposed answer is correct and supported by the evidence.",
                 "false": "The proposed answer is wrong, or the evidence does not support it."},
}
STOPWORDS = set("of in on at to for by with from and or is was are were be as that this which who what when "
                "where how did does do its it his her their he she they".split())

_agent = None


def agent():
    global _agent
    if _agent is None:
        import laya_mlx
        _agent = laya_mlx.load(MODEL)
    return _agent


def tokenizer():
    """Laya's own tokenizer, loaded without the model weights."""
    from huggingface_hub import snapshot_download
    from laya_mlx.tokenizer import Tokenizer
    return Tokenizer(Path(snapshot_download(MODEL, allow_patterns=["tokenizer/*"])) / "tokenizer")


def state_room(tok):
    """Tokens left for the state after Laya's question prefix (same math as laya_mlx.common)."""
    from laya_mlx.common import build_prefix
    from laya_mlx.agent import Agent
    prefix, _ = build_prefix(tok, Agent._to_internal(QUESTION), 192)
    return 512 - len(prefix) - 1


def n_tokens(tok, state):
    return len(tok(json.dumps(state, ensure_ascii=False))["input_ids"])


def words(text):
    return set(normalize(text).split()) - STOPWORDS


def sentences(evidence):
    """Split each 'Title\\n[0] sentence\\n[1] ...' paragraph into 'Title: sentence' strings."""
    out = []
    for paragraph in evidence:
        title, *lines = paragraph.split("\n")
        out += [f"{title}: {line.split('] ', 1)[-1]}" for line in lines]
    return out


def build_state(draft, tok, room):
    """Answer first, then evidence sentences ranked by overlap with the answer (x2) and question.

    Sentences that don't fit are skipped. Ranking uses only the question and draft,
    never gold answers or supporting facts. Returns (state, info about truncation).
    """
    answer_words, question_words = words(draft["draft"]), words(draft["question"])
    ranked = sorted(sentences(draft["evidence"]), reverse=True,
                    key=lambda s: 2 * len(words(s) & answer_words) + len(words(s) & question_words))
    state = {"question": draft["question"], "proposed_answer": draft["draft"], "evidence": []}
    for sentence in ranked:
        state["evidence"].append(sentence)
        if n_tokens(tok, state) > room:
            state["evidence"].pop()
    full = {"question": draft["question"], "proposed_answer": draft["draft"], "evidence": draft["evidence"]}
    return state, {"full_tokens": n_tokens(tok, full), "kept": len(state["evidence"]), "total": len(ranked)}


def laya(draft):
    a = agent()
    state, _ = build_state(draft, a.tok, state_room(a.tok))
    result = a.predict(state, {"correct": QUESTION})
    assert not result["usage"]["truncated"], "state should already fit"
    return float(result["answers"]["correct"]["noul"])


def measure():
    tok = tokenizer()
    room = state_room(tok)
    for which in ("first", "all"):
        infos = [build_state(d, tok, room)[1] for d in load_drafts(which)]
        over = [i for i in infos if i["full_tokens"] > room]
        lengths = sorted(i["full_tokens"] for i in infos)
        share = sorted(i["kept"] / i["total"] for i in infos)
        print(f"{which}: {len(infos)} drafts; state room {room} tokens (512 minus question prefix)")
        print(f"  full state tokens: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
        print(f"  truncated: {len(over)}/{len(infos)}; share of evidence sentences kept: "
              f"min {share[0]:.0%}, median {share[len(share) // 2]:.0%}, max {share[-1]:.0%}")


if __name__ == "__main__":
    measure()
