"""Shared helpers: answer scoring (official HotpotQA/SQuAD normalization),
model loading, chat generation with token counting, and dataset loaders."""
import json
import re
import string
from collections import Counter


# ---------- scoring (same normalization as the official HotpotQA scorer) ----------
def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        if p != g:
            return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def em_score(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def best_scores(pred, golds):
    """MuSiQue gives an answer plus aliases: take the best match."""
    return max(f1_score(pred, g) for g in golds), max(em_score(pred, g) for g in golds)


def extract_answer(text):
    """Text after the last 'Answer:' line (also '**Answer:**', 'Final Answer**:').
    Returns None when the writer gave no answer line (e.g. cut off mid-reasoning)."""
    m = re.findall(r"answer\s*\**\s*:\s*\**\s*(.+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    ans = m[-1].strip().strip("*. \"'").strip()
    return ans[:200] or None


# ---------- model ----------
def load_model(name, four_bit=False):
    """four_bit=True loads the model in 4-bit (NF4) so a 7B model fits on a free T4."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return tok, model


def chat(tok, model, user_msg, max_new_tokens=160):
    """Greedy generation. Returns (text, prompt_tokens, generated_tokens)."""
    import torch
    msgs = [{"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True), int(ids["input_ids"].shape[1]), int(gen.shape[0])


def format_paragraphs(paras):
    return "\n\n".join(f"[{i+1}] {p['title']}: {p['text']}" for i, p in enumerate(paras))


# ---------- datasets ----------
MUSIQUE_HF_IDS = ["dgslibisey/MuSiQue", "bdsaglam/musique"]


def _musique_row(r):
    paras = [{"title": p.get("title", ""), "text": p.get("paragraph_text", p.get("text", "")),
              "supporting": bool(p.get("is_supporting", False))} for p in r["paragraphs"]]
    golds = [r["answer"]] + list(r.get("answer_aliases") or [])
    return {"id": r["id"], "question": r["question"], "golds": golds, "paragraphs": paras}


def load_musique(n, seed=0, local_jsonl=None):
    """Answerable MuSiQue dev questions. Uses a local official .jsonl if given, else Hugging Face."""
    import random
    rows = []
    if local_jsonl:
        with open(local_jsonl) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        from datasets import load_dataset
        last_err = None
        for hf_id in MUSIQUE_HF_IDS:
            try:
                ds = load_dataset(hf_id, split="validation")
                rows = list(ds)
                print(f"Loaded MuSiQue from {hf_id}: {len(rows)} rows")
                break
            except Exception as e:  # try the next mirror
                last_err = e
        if not rows:
            raise RuntimeError(f"Could not load MuSiQue from Hugging Face ({last_err}). "
                               "Download musique_ans_v1.0_dev.jsonl and pass --local_jsonl.")
    rows = [r for r in rows if r.get("answerable", True)]
    random.Random(seed).shuffle(rows)
    return [_musique_row(r) for r in rows[:n]]
