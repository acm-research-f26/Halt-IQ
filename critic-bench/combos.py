"""Compare stop rules built from cached critic scores (strict label); writes combos.md.

    python3 combos.py

A rule says "stop" (approve) or "revise" for each draft. Cheap checks run first,
and a slower critic's time counts only when the rule actually calls it.
evidence_match passes only at 1; a yes/no draft (0.5) can't be string-checked,
so the gates (rules 4 and 11) hand it to the critic.
"""

import math
from pathlib import Path

from drafts import load_drafts
from report import jev_cost_per_decision, load_scores

HERE = Path(__file__).resolve().parent


def run(scores, critic, draft_id):
    """(score or decision, seconds) for one critic on one draft."""
    row = scores[critic][draft_id]
    return (row["score"] if row["score"] is not None else row["decision"] == "approve"), row["seconds"]


# Each rule returns (approved, seconds, costly critic it called: "llm", "jev", or None).
def approve_all(s, d):
    return True, 0.0, None


def llm_alone(s, d):
    ok, sec = run(s, "llm", d)
    return ok, sec, "llm"


def evidence_alone(s, d):
    return run(s, "evidence_match", d)[0] == 1, 0.0, None


def gate(s, d):
    if run(s, "evidence_match", d)[0] == 0:
        return False, 0.0, False  # answer not in the evidence: revise, skip llm
    ok, sec = run(s, "llm", d)
    return ok, sec, "llm"


def laya_alone(s, d):
    p, sec = run(s, "laya", d)
    return p >= 0.8, sec, None


def evidence_and_laya(s, d):
    if run(s, "evidence_match", d)[0] != 1:
        return False, 0.0, None
    p, sec = run(s, "laya", d)
    return p >= 0.5, sec, None


def consistency_and_evidence(s, d):
    if run(s, "evidence_match", d)[0] != 1:
        return False, 0.0, None
    c, sec = run(s, "consistency", d)
    return c == 1.0, sec, None


def logprob_alone(s, d):
    p, sec = run(s, "qwen8b-logprob", d)
    return p >= 0.8, sec, None


def evidence_and_logprob(s, d):
    if run(s, "evidence_match", d)[0] != 1:
        return False, 0.0, None
    p, sec = run(s, "qwen8b-logprob", d)
    return p >= 0.5, sec, None


def jev_alone(s, d):
    p, sec = run(s, "jev", d)
    return p >= 0.8, sec, "jev"


def jev_gate(s, d):
    if run(s, "evidence_match", d)[0] == 0:
        return False, 0.0, None  # answer not in the evidence: revise, skip the paid Jev call
    p, sec = run(s, "jev", d)
    return p >= 0.8, sec, "jev"


# (name, rule, critics it needs). A rule runs on the drafts that every needed critic has scored.
RULES = [
    ("1. approve everything", approve_all, []),
    ("2. llm alone", llm_alone, ["llm"]),
    ("3. evidence_match alone", evidence_alone, ["evidence_match"]),
    ("4. gate: evidence_match, then llm", gate, ["evidence_match", "llm"]),
    ("5. laya ≥ 0.8 alone", laya_alone, ["laya"]),
    ("6. evidence_match = 1 AND laya ≥ 0.5", evidence_and_laya, ["evidence_match", "laya"]),
    ("7. consistency = 1.0 AND evidence_match = 1", consistency_and_evidence, ["evidence_match", "consistency"]),
    ("8. logprob ≥ 0.8 alone", logprob_alone, ["qwen8b-logprob"]),
    ("9. evidence_match = 1 AND logprob ≥ 0.5", evidence_and_logprob, ["evidence_match", "qwen8b-logprob"]),
    ("10. jev ≥ 0.8 alone", jev_alone, ["jev"]),
    ("11. gate: evidence_match, then jev ≥ 0.8", jev_gate, ["evidence_match", "jev"]),
]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def table(which, scores):
    all_drafts = load_drafts(which)
    n, n_right = len(all_drafts), sum(d["correct"] for d in all_drafts)
    lines = [f"## `{which}`: {n} drafts, {n_right} correct ({n_right / n:.0%})", "",
             "| Rule | Approved | Accuracy when approved (95% CI) | Wrong approvals | Wrongly rejected "
             "| Avg s/draft | llm calls saved vs rule 2 | API $ per draft |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    jev_cost = jev_cost_per_decision()
    for name, rule, needs in RULES:
        drafts = [d for d in all_drafts if all(d["draft_id"] in scores.get(c, {}) for c in needs)]
        if not drafts:
            continue  # no scores for this critic on this set
        if len(drafts) < n:
            name += f" (only {len(drafts)}/{n} drafts scored)"
        rule_n, rule_right = len(drafts), sum(d["correct"] for d in drafts)
        results = [(d, *rule(scores, d["draft_id"])) for d in drafts]
        approved = [d for d, ok, _, _ in results if ok]
        right = sum(d["correct"] for d in approved)
        ci = wilson(right, len(approved))
        accuracy = f"{right}/{len(approved)} ({right / len(approved):.0%}, {ci[0]:.0%}–{ci[1]:.0%})" if approved else "—"
        wrong_ok = sum(not d["correct"] for d in approved)
        seconds = sum(sec for _, _, sec, _ in results) / rule_n
        saved = 1 - sum(called == "llm" for *_, called in results) / rule_n
        jev_calls = sum(called == "jev" for *_, called in results)
        api = f"${jev_calls * jev_cost / rule_n:.6f}" if jev_calls and jev_cost else "$0"
        lines.append(f"| {name} | {len(approved)}/{rule_n} | {accuracy} | {wrong_ok}/{rule_n - rule_right} "
                     f"| {rule_right - right}/{rule_right} | {seconds:.2f} | {saved:.0%} | {api} |")
    return lines


SUMMARY = ("Jev is the first critic that clearly separates right from wrong drafts: gating it with the free "
           "evidence_match check (rule 11) is right 74% of the time when it stops on `extra` (95% interval 66–81%), "
           "the only rule whose interval does not overlap \"approve everything\" (51–64%), with 32 wrong approvals "
           "against 59 for the current llm critic, at about 0.2 s and $0.00008 per draft. "
           "The trade-off is more rejected right answers (22 vs 15 for llm alone), which would cost extra rounds. "
           "Among the free rules, the evidence_match gate before llm (rule 4) and consistency + evidence_match (rule 7) "
           "remain the best, at 66–69% when they stop.")


def main():
    scores = load_scores()
    lines = ["# Stop-rule combinations (strict label)", "",
             "Each rule decides, per draft, whether the loop stops (approves) or revises. "
             "Seconds count only the critics the rule actually calls (evidence_match is free). "
             "Generated by `combos.py` from cached scores; no new model calls.", ""]
    for which in ("extra", "subset80"):
        lines += table(which, scores) + [""]
    lines += ["## Summary", "", SUMMARY, "",
              "Notes: \"llm calls saved\" counts calls to the qwen3:8b llm critic only (Jev rules call the paid Jev API instead; "
              "API $ per draft is from the Jev budget ledger: mean spend per settled call x share of drafts that call Jev); rule 7's time is its 3 extra writer "
              "samples (temperature 0.7) per question. A yes/no draft (evidence_match = 0.5) is sent on to the critic by the gates (rules 4 and 11) and rejected by rules 3, 6, 7, and 9.", ""]
    text = "\n".join(lines)
    (HERE / "combos.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
