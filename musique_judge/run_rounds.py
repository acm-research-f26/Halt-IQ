"""Writer agent: draft an answer, then revise it for up to R rounds.
Saves one row per (question, round), same columns as data/sri_rounds_with_strands.csv
(minus strands/critic), plus a paragraphs file the judge and Strands can reuse.

Colab:  python run_rounds.py --n 60 --rounds 6 --out data/musique_rounds.csv
"""
import argparse
import csv
import json
import os
import time

from common import best_scores, chat, extract_answer, format_paragraphs, load_model, load_musique

DRAFT_PROMPT = """Answer the question using the paragraphs below. The answer is a short phrase (a name, date, place or number).

{paras}

Question: {q}

Reason in at most three short sentences. Then write the final line exactly as:
Answer: <short answer>"""

REVISE_PROMPT = """Answer the question using the paragraphs below. The answer is a short phrase (a name, date, place or number).

{paras}

Question: {q}

Your previous answer was: {prev}

Check each hop of the question against the paragraphs, in at most three short sentences. If the previous answer is wrong, correct it; if it is right, repeat it.
Then write the final line exactly as:
Answer: <short answer>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="data/musique_rounds.csv")
    ap.add_argument("--paras_out", default="data/musique_paragraphs.jsonl")
    ap.add_argument("--local_jsonl", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    qs = load_musique(args.n, local_jsonl=args.local_jsonl)
    with open(args.paras_out, "w") as f:
        for q in qs:
            f.write(json.dumps({"id": q["id"], "question": q["question"], "golds": q["golds"],
                                "paragraphs": q["paragraphs"]}) + "\n")

    done = set()
    if os.path.exists(args.out):  # resume after a Colab disconnect
        with open(args.out) as f:
            done = {r["id"] for r in csv.DictReader(f) if r["round"] == str(args.rounds)}
    tok, model = load_model(args.model)
    cols = ["id", "split", "round", "answer", "parsed", "f1", "em", "tokens", "prompt_tokens", "gen_tokens", "raw"]
    new_file = not os.path.exists(args.out)
    t0 = time.time()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        for i, q in enumerate(qs):
            if q["id"] in done:
                continue
            paras = format_paragraphs(q["paragraphs"])
            prev, rows = None, []
            for r in range(1, args.rounds + 1):
                prompt = (DRAFT_PROMPT.format(paras=paras, q=q["question"]) if r == 1 else
                          REVISE_PROMPT.format(paras=paras, q=q["question"], prev=prev))
                text, pt, gt = chat(tok, model, prompt, args.max_new_tokens)
                ans = extract_answer(text)
                parsed = int(ans is not None)
                if ans is None:
                    # No "Answer:" line (usually cut off mid-reasoning). The orchestrator keeps the
                    # previous answer; in round 1 there is none, so fall back to the last short line.
                    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
                    ans = prev if prev is not None else (lines[-1][:200] if lines else "")
                f1, em = best_scores(ans, q["golds"])
                rows.append({"id": q["id"], "split": "test", "round": r, "answer": ans, "parsed": parsed,
                             "f1": round(f1, 4), "em": em, "tokens": pt + gt,
                             "prompt_tokens": pt, "gen_tokens": gt, "raw": text[:2000]})
                prev = ans
            w.writerows(rows)  # write whole questions only, so resume stays clean
            f.flush()
            print(f"[{i+1}/{len(qs)}] {time.time()-t0:.0f}s  gold={q['golds'][0]!r}  "
                  f"parsed={sum(x['parsed'] for x in rows)}/{len(rows)}  answers={[x['answer'] for x in rows]}")


if __name__ == "__main__":
    main()
