"""Help-or-hurt judge for revisions, after the professor's Chain of Counterfactual Thought
(CoCT; Zhang, Sun, Guo, Gogate, arXiv 2506.22385).

Each time the writer CHANGES its answer between rounds, the revision is an "update".
CoCT:   1) argue the revision is a strengthener (new answer better)
        2) argue it is a weakener (new answer worse)
        3) a judge reads both arguments and picks BETTER or WORSE.
Direct: the same judge asked BETTER/WORSE with no arguments (the paper's ablation).
Ground truth: did token F1 go up (help) or down (hurt)? Ties are reported separately.

Colab:  python coct_judge.py --rounds data/musique_rounds.csv --paras data/musique_paragraphs.jsonl \
            --out data/musique_judge.csv
HotpotQA (Sri's saved rounds):
        python coct_judge.py --rounds data/sri_rounds_with_strands.csv --hotpot --out data/hotpot_judge.csv
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

from common import chat, format_paragraphs, load_model, normalize_answer

CTX = """{paras}Question: {q}
Earlier answer: {old}
Revised answer: {new}
{reasoning}"""

REASONING = """
Writer's reasoning for the earlier answer:
{r_old}

Writer's reasoning for the revised answer:
{r_new}
"""


def clip(text, n=700):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + " ..."

ARGUE_BETTER = CTX + """
Argue that the REVISED answer is more correct than the earlier one. Cite the evidence that supports it. Be brief (3 sentences)."""

ARGUE_WORSE = CTX + """
Argue that the REVISED answer is less correct than the earlier one, i.e. the revision introduced an error. Cite the evidence. Be brief (3 sentences)."""

JUDGE_COCT = CTX + """
Argument that the revision helped:
{pro}

Argument that the revision hurt:
{con}

Both arguments may contain mistakes. Check each against the paragraphs and correct any errors, then decide which argument is right.
End with exactly one line: Verdict: BETTER  (revision helped) or  Verdict: WORSE  (revision hurt)."""

JUDGE_DIRECT = CTX + """
Is the revised answer more correct than the earlier one?
End with exactly one line: Verdict: BETTER  or  Verdict: WORSE"""


def parse_verdict(text):
    m = re.findall(r"verdict\s*:\s*\**\s*(better|worse)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    t = text.upper()
    return "WORSE" if t.rfind("WORSE") > t.rfind("BETTER") else "BETTER"


def load_rounds(path):
    by_id = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_id[r["id"]].append(r)
    for rows in by_id.values():
        rows.sort(key=lambda r: int(r["round"]))
    return by_id


def load_context(args, ids):
    """id -> (question, paragraphs). Paragraphs may be empty if unavailable."""
    ctx = {}
    if args.paras:
        with open(args.paras) as f:
            for l in f:
                d = json.loads(l)
                ctx[d["id"]] = (d["question"], d["paragraphs"])
    if args.hotpot:
        from datasets import load_dataset
        # Sri's ids look like "hotpot_<hotpot id>"; her questions come from the distractor validation set.
        want = {qid.removeprefix("hotpot_"): qid for qid in ids if qid not in ctx}
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        for r in ds:
            if r["id"] in want:
                paras = [{"title": t, "text": " ".join(s)}
                         for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
                ctx[want[r["id"]]] = (r["question"], paras)
        print(f"Found HotpotQA context for {sum(q in ctx for q in want.values())} of {len(want)} questions")
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--paras", default=None)
    ap.add_argument("--hotpot", action="store_true", help="look up HotpotQA questions/paragraphs by id")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_direct", action="store_true", help="skip the no-argument baseline judge")
    ap.add_argument("--four_bit", action="store_true", help="load the judge in 4-bit (for 7B on a T4)")
    ap.add_argument("--with_reasoning", action="store_true",
                    help="show the judge the writer's own reasoning (the 'raw' column) for both answers")
    ap.add_argument("--decisive_first", action="store_true",
                    help="judge changes that helped or hurt first, ties last (accuracy needs only the first)")
    args = ap.parse_args()

    by_id = load_rounds(args.rounds)
    changes = []
    for qid, rows in by_id.items():
        for prev, cur in zip(rows, rows[1:]):
            if normalize_answer(prev["answer"]) != normalize_answer(cur["answer"]):
                changes.append((qid, prev, cur))
    if args.decisive_first:  # stable sort: help/hurt first, ties last
        changes.sort(key=lambda c: float(c[1]["f1"]) == float(c[2]["f1"]))
    done = set()
    if os.path.exists(args.out):  # resume: skip changes already judged
        with open(args.out) as f:
            done = {(r["id"], r["round"]) for r in csv.DictReader(f)}
    print(f"{len(by_id)} questions, {len(changes)} answer changes, {len(done)} already judged")
    if args.with_reasoning and "raw" not in next(iter(by_id.values()))[0]:
        raise SystemExit("--with_reasoning needs the 'raw' column from run_rounds.py")

    ctx = load_context(args, by_id.keys())
    tok, model = load_model(args.model, four_bit=args.four_bit)
    cols = ["id", "round", "old", "new", "f1_old", "f1_new", "label",
            "coct", "coct_tokens", "direct", "direct_tokens"]
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
            pro, p1, g1 = chat(tok, model, ARGUE_BETTER.format(**fmt), 140)
            con, p2, g2 = chat(tok, model, ARGUE_WORSE.format(**fmt), 140)
            jt, p3, g3 = chat(tok, model, JUDGE_COCT.format(pro=pro, con=con, **fmt), 200)
            coct = parse_verdict(jt)
            direct, dtok = "", 0
            if not args.no_direct:
                dt, p4, g4 = chat(tok, model, JUDGE_DIRECT.format(**fmt), 120)
                direct, dtok = parse_verdict(dt), p4 + g4
            f_old, f_new = float(prev["f1"]), float(cur["f1"])
            label = "help" if f_new > f_old else "hurt" if f_new < f_old else "tie"
            w.writerow({"id": qid, "round": cur["round"], "old": prev["answer"], "new": cur["answer"],
                        "f1_old": f_old, "f1_new": f_new, "label": label,
                        "coct": coct, "coct_tokens": p1 + g1 + p2 + g2 + p3 + g3,
                        "direct": direct, "direct_tokens": dtok})
            f.flush()
            print(f"[{i+1}/{len(changes)}] {label:4s} coct={coct} direct={direct}  "
                  f"{prev['answer']!r} -> {cur['answer']!r}")


if __name__ == "__main__":
    main()
