"""Correctness-prompted judge: the controlled comparison for H3.

Same model and inputs as `coct_judge.py --with_reasoning` (paragraphs, question, both answers,
both reasonings). The only change is the question asked: "is the EARLIER / REVISED answer
correct?" instead of "did the revision help?". Judges only changes that helped or hurt.
Tie rule (fixed before running): if both answers get the same verdict, predict BETTER
(keep the revision, the default action) and mark the case undecided.
"""
import argparse
import csv
import os
import re
from types import SimpleNamespace

from common import chat, format_paragraphs, load_model, normalize_answer
from coct_judge import CTX, REASONING, clip, load_context, load_rounds

JUDGE_CORRECT = CTX + """
Is the {which} answer correct? Check it against the paragraphs.
End with exactly one line: Verdict: CORRECT  or  Verdict: INCORRECT"""


def parse_correct(text):
    m = re.findall(r"verdict\s*:\s*\**\s*(incorrect|correct)", text, flags=re.IGNORECASE)
    return m[-1].upper() if m else "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--paras", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--with_reasoning", action="store_true")
    args = ap.parse_args()

    by_id = load_rounds(args.rounds)
    changes = []
    for qid, rows in by_id.items():
        for prev, cur in zip(rows, rows[1:]):
            if (normalize_answer(prev["answer"]) != normalize_answer(cur["answer"])
                    and float(prev["f1"]) != float(cur["f1"])):
                changes.append((qid, prev, cur))
    done = set()
    if os.path.exists(args.out):  # resume
        with open(args.out) as f:
            done = {(r["id"], r["round"]) for r in csv.DictReader(f)}
    print(f"{len(changes)} help/hurt changes, {len(done)} already judged")

    ctx = load_context(SimpleNamespace(paras=args.paras, hotpot=False), by_id.keys())
    tok, model = load_model(args.model, four_bit=args.four_bit)
    cols = ["id", "round", "old", "new", "f1_old", "f1_new", "label",
            "old_verdict", "new_verdict", "correctness", "undecided", "tokens"]
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        for i, (qid, prev, cur) in enumerate(changes):
            if (qid, cur["round"]) in done:
                continue
            q, paras = ctx.get(qid, (prev.get("question", ""), []))
            if not q:
                print(f"  no question text for {qid}; skipping")
                continue
            fmt = dict(paras=(format_paragraphs(paras) + "\n\n") if paras else "",
                       q=q, old=prev["answer"], new=cur["answer"],
                       reasoning=REASONING.format(r_old=clip(prev.get("raw")), r_new=clip(cur.get("raw")))
                       if args.with_reasoning else "")
            t_old, p1, g1 = chat(tok, model, JUDGE_CORRECT.format(which="EARLIER", **fmt), 120)
            t_new, p2, g2 = chat(tok, model, JUDGE_CORRECT.format(which="REVISED", **fmt), 120)
            v_old, v_new = parse_correct(t_old), parse_correct(t_new)
            if v_old == "INCORRECT" and v_new == "CORRECT":
                pred, undecided = "BETTER", 0
            elif v_old == "CORRECT" and v_new == "INCORRECT":
                pred, undecided = "WORSE", 0
            else:
                pred, undecided = "BETTER", 1
            f_old, f_new = float(prev["f1"]), float(cur["f1"])
            label = "help" if f_new > f_old else "hurt"
            w.writerow({"id": qid, "round": cur["round"], "old": prev["answer"], "new": cur["answer"],
                        "f1_old": f_old, "f1_new": f_new, "label": label,
                        "old_verdict": v_old, "new_verdict": v_new, "correctness": pred,
                        "undecided": undecided, "tokens": p1 + g1 + p2 + g2})
            f.flush()
            print(f"[{i+1}/{len(changes)}] {label:4s} old={v_old} new={v_new} -> {pred}"
                  f"{' (undecided)' if undecided else ''}")


if __name__ == "__main__":
    main()