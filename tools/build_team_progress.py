"""Build the October 18 team update from the audited study facts.

This is a narrative delivery document, not a benchmark runner or metric scorer.
Use Python 3.12 and reportlab 4.4.9. Outputs are PDF and editable Markdown.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, StyleSheet1
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


REPO = "https://github.com/acm-research-f26/Halt-IQ"
BUNDLE = f"{REPO}/tree/Yash/results/strands-20261018"
PAGES = [
    [
        ("title", "HaltIQ Strands team progress report"),
        ("meta", "Yash | Research team update | October 18 2026"),
        ("p", "I completed the Strands decision-controller implementation, native baseline testing, supervised head fine-tuning and a matched HotpotQA comparison at up to ten drafts. The implementation and experiment outputs are ready for team review. The fresh comparison found no answer-quality improvement from additional revisions or head fine-tuning: all 13 primary policies scored 50% exact match and 0.540 token F1 on the same 20 questions."),
        ("h1", "Implementation completed"),
        ("bullet", "Set up and ran the actual open Strands decider locally, including verified model files, CUDA health and native model forwards."),
        ("bullet", "Tested the untouched released model on all 20 original development questions before updating weights."),
        ("bullet", "Updated the existing 3,151,360-parameter pointer head using disjoint fit, tuning and calibration partitions. Verified changed head weights, frozen encoder and released adapter, and an exact reload match."),
        ("bullet", "Implemented Fixed 1/2/3/5/10 controls, calibrated released and trained controllers at caps three and ten, and four predeclared visited-confidence retention variants."),
        ("bullet", "Completed a separate hosted Jev 1.13 comparison on the same fresh question set and saved writer trajectories, with settled API accounting."),
        ("h1", "Tested architecture and depth"),
        ("p", "Qwen3:8B writes an answer from the question and all supplied evidence. Strands scores the current draft's binary answer correctness. A calibrated probability of at least 0.8 accepts the draft; otherwise a generic revision is visited until the cap. Ten total drafts means the initial answer plus nine revisions. Cap-three results preserve the shorter comparison. Retention variants can select only a draft already visited and retain all costs incurred."),
        ("p", "The comparison replays stopping policies on common recorded trajectories so policies share identical draft observations. It evaluates the decision controller; it does not establish live orchestration latency. Evidence-specialist routing, backbone or LoRA updates and reinforcement learning were not run."),
        ("h1", "Cohorts kept separate"),
        ("table", [
            ["Cohort", "Questions", "Use"],
            ["Original development", "20", "Untouched baseline and clearly labeled reused diagnostics"],
            ["Adopted TRAIN", "90", "60 fit, 15 tune and 15 calibration"],
            ["Fresh public validation", "20", "16 bridge and 4 comparison questions; all ten drafts"],
        ], [150, 66, 300]),
    ],
    [
        ("h1", "Dataset and evaluation protocol"),
        ("p", "I used HotpotQA distractor questions with the supplied paragraph titles, sentence order and full evidence retained. Reference answers, supporting facts and correctness labels remain evaluator-only inputs. TRAIN and validation are checked for overlap by ID and normalized question. The adopted training reduction preserves the original partitions and uses answer-blind seed-42 hash selection within partition and public question type."),
        ("p", "The 20 fresh questions were frozen before their writer or decision outcomes were observed. They are a selected public-validation panel, not an official hidden-test result. The same 20 questions remain in every quality denominator; missing or failed episodes would score zero rather than being omitted."),
        ("h1", "Recorded environment"),
        ("table", [
            ["Component", "Recorded setting"],
            ["PC and operating system", "Windows 11; Ryzen 7 9800X3D; RTX 5070 Ti with about 16 GB VRAM; 31.06 GiB RAM"],
            ["Python and packages", "Python 3.12.10; Torch 2.7.1+cu128; Transformers 5.17.0; PEFT 0.21.0"],
            ["Writer", "Qwen3:8B; 16,384 context tokens; 512 maximum output tokens; temperature 0; seed 42; thinking off"],
            ["Strands decider", "Released 2B encoder and adapter frozen; 4,096 fully rendered tokens; no silent truncation"],
            ["Background use", "Four CPU threads, one study GPU owner, 7 GiB available RAM and 3 GiB free VRAM reserves; deliberate rests"],
        ], [126, 390]),
        ("h1", "Fine tuning and calibration completed"),
        ("p", "The 60 fit examples train the existing head; 15 disjoint tuning questions select the checkpoint by NLL. The selected learning rate is 1e-5 at epoch one. Its saved weights contain two optimizer steps; 12 steps describe all candidate fitting across the trials. Head weight change has L2 magnitude 0.02861134 and reload logit error is zero. The encoder and released adapter remained frozen."),
        ("p", "Each head has its own temperature fitted on the same 15 calibration questions: 1.438330 for released and 1.426638 for trained. Calibration fitting loss is not held-out quality. The pre-training development baseline was 45% EM and 0.527619 F1 on a different cohort, so the fresh 50% score is not a measured training gain."),
        ("h1", "Common metrics and partner compatibility"),
        ("p", "The common answer metrics are HotpotQA normalized exact match and token F1. I also report repairs, regressions, approval precision with coverage, first-decision AUROC, Brier score and NLL. Paired intervals use 5,000 question-level bootstrap draws with seed 42. Exports follow Vedanshi's v0 result schema; additional provenance and failure accounting are retained separately. Schema and scoring compatibility do not imply identical hardware or context limits."),
    ],
    [
        ("h1", "Results on the fresh 20 questions"),
        ("p", "Every primary policy completed all 20 planned questions. There were no repairs, regressions or failed episodes. Eighteen questions had the same raw answer throughout all ten drafts; the other two had incorrect wording changes without an exact-match transition."),
        ("table", [
            ["Policy", "EM", "Token F1", "Mean drafts"],
            ["Fixed 1 / 2 / 3 / 5 / 10", "50%", "0.540", "1 / 2 / 3 / 5 / 10"],
            ["Released Strands cap ten", "50%", "0.540", "5.50"],
            ["Trained Strands cap ten", "50%", "0.540", "5.50"],
            ["Either Strands head cap three", "50%", "0.540", "2.00"],
            ["Four retention variants caps three / ten", "50%", "0.540", "2.00 / 5.50"],
            ["Jev 1.13 cap ten", "50%", "0.540", "3.75"],
            ["Jev 1.13 cap three", "50%", "0.540", "1.65"],
        ], [252, 58, 70, 136]),
        ("h1", "First decision correctness diagnostics"),
        ("table", [
            ["Metric", "Released Strands", "Trained Strands", "Jev"],
            ["AUROC", "0.380", "0.380", "0.915"],
            ["Brier score", "0.336260", "0.336322", "0.141200"],
            ["NLL", "0.910647", "0.910822", "0.469715"],
            ["Correct approved / approved", "5/10 = 50%", "5/10 = 50%", "10/13 = 76.92%"],
            ["Approved / all planned", "10/20 = 50%", "10/20 = 50%", "13/20 = 65%"],
        ], [174, 114, 114, 114]),
        ("p", "Jev ranked first-draft correctness better on this panel but returned the same answer scores. Its final stopping approvals are a separate statistic: 10 correct out of 14 approvals, with 14/20 approval coverage. Approval precision is conditional on approving and must not be called the all-question match rate. Jev provides probabilities, not raw logits."),
        ("p", "All primary paired answer-quality differences are zero, with observed 95% intervals [0, 0]. This small-panel result does not establish population equivalence or a generalization advantage. The trained head's Brier and NLL were slightly worse than the released control. Repeated drafts do not increase the independent sample size beyond 20."),
    ],
    [
        ("h1", "Actual expense and incomplete preparation"),
        ("p", "Fresh physical work was 200 writer calls, 200 native forwards and 110 CPU scores per Strands head. The primary campaign took about 38 minutes 25 seconds, including startup and rests. Replay prefixes do not measure live speed or token savings."),
        ("p", "Jev made 75 calls costing $0.006847848; cumulative accounting ends at 606 settled calls and $0.054730032, with zero reserved or unsettled. Local hardware and electricity are unpriced. Earlier 600-question preparation retained 590 trajectories from 591 calls, with one unsaved call and nine questions undispatched. Its expense is separate from the completed 90-question training source."),
        ("h1", "Two next experiments"),
        ("h2", "Evidence specialist feedback under Strands control"),
        ("p", "When Strands rejects an answer, a specialist would return short feedback cited to the supplied evidence before the writer revises. Strands would retain final STOP authority. A proposed limit is one specialist call per revision and two per question."),
        ("p", "Pros: feedback could target an identifiable error instead of repeating a generic draft, and citations make the proposed correction inspectable. Cons: extra calls add latency and cost, and incorrect specialist feedback can introduce new errors. The next test would compare generic revisions with cited feedback on a new disjoint panel, while charging every role's work and retaining the same answer and decision metrics."),
        ("h2", "Budget aware STOP and VERIFY utility learning"),
        ("p", "A second experiment would use disjoint TRAIN continuations to learn whether the expected answer benefit of verification exceeds its token and time cost. I would start with bounded one-step full-information targets before considering reinforcement learning."),
        ("p", "Pros: this trains the controller toward the joint quality and budget objective rather than correctness confidence alone. Cons: results depend on reward weights and an imperfect exact-match proxy; replay or off-policy bias can misestimate action value. A later matched live test would be required to establish action effects and real latency. Neither proposal is a completed experiment."),
        ("h1", "GitHub and review materials"),
        ("p", "The team entry point is results/strands-20261018 on the Yash branch. It indexes the reports, question-level exports, data and trajectory snapshots, the selected trained head, model identities and portable protocol records, with original hashes and sanitization notes."),
        ("link", "Open the final experiment bundle", BUNDLE),
        ("link", "Open the Strands implementation", f"{REPO}/tree/Yash/pc_strands"),
        ("link", "Read the full professor report", f"{REPO}/blob/Yash/results/strands-20261018/reports/professor-review.pdf"),
        ("link", "Read the optional Jev comparison", f"{REPO}/blob/Yash/results/strands-20261018/reports/optional-jev-companion.pdf"),
        ("p", "The full write-up contains detailed methods and proofs. The GitHub bundle excludes credentials, raw API ledgers, private machine paths, large upstream weights and feature caches. Public HotpotQA inputs retain attribution; reference answers remain evaluator-only."),
        ("meta", "Evidence: finalized October 18 primary paired replay and separate Jev terminal audits. Implementation source baseline 8f48a545bd4d538d74e78e40c73cdd4de63d45c6. Report data and methods are described in the linked 18-page primary report and four-page companion."),
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "team-progress.pdf"
    md = out / "team-progress.md"
    if pdf.exists() or md.exists():
        parser.error("Use a new output directory to preserve reviewed versions")
    styles = StyleSheet1()
    for name, size, leading, after in [
        ("body", 10.8, 14.5, 8), ("title", 23, 27, 13),
        ("h1", 14, 18, 8), ("h2", 11.8, 15, 6),
        ("meta", 8.7, 11, 9), ("cell", 9.6, 12.2, 0),
    ]:
        styles.add(ParagraphStyle(name, fontName="Helvetica-Bold" if name in ("title", "h1", "h2") else "Helvetica", fontSize=size, leading=leading, textColor=colors.black, spaceAfter=after, spaceBefore=9 if name in ("h1", "h2") else 0, keepWithNext=name in ("title", "h1", "h2")))
    story = []
    markdown = []
    for page_i, blocks in enumerate(PAGES):
        if page_i:
            story.append(PageBreak())
        for block in blocks:
            kind = block[0]
            if kind == "table":
                rows, widths = block[1:]
                rendered = [[Paragraph(escape(str(value)), styles["cell"]) for value in row] for row in rows]
                table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF2")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F8FA")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story += [table, Spacer(1, 8)]
                markdown += ["| " + " | ".join(row) + " |" for row in rows[:1]]
                markdown += ["| " + " | ".join(["---"] * len(rows[0])) + " |"]
                markdown += ["| " + " | ".join(row) + " |" for row in rows[1:]]
                markdown.append("")
            elif kind == "link":
                label, url = block[1:]
                story.append(Paragraph(f'<link href="{escape(url)}" color="#174A7A">{escape(label)}</link>', styles["body"]))
                markdown += [f"[{label}]({url})", ""]
            else:
                value = block[1]
                style = "body" if kind in ("p", "bullet") else kind
                story.append(Paragraph(escape(value), styles[style], bulletText="-" if kind == "bullet" else None))
                prefix = {"title": "# ", "h1": "## ", "h2": "### ", "bullet": "- "}.get(kind, "")
                markdown += [prefix + value, ""]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(48, 25, "HaltIQ team progress | October 18 2026")
        canvas.drawRightString(564, 25, str(document.page))
        canvas.restoreState()

    doc = BaseDocTemplate(str(pdf), pagesize=letter, leftMargin=48, rightMargin=48, topMargin=38, bottomMargin=42, title="HaltIQ Strands team progress report", author="Yash", subject="Completed implementation and matched evaluation for team review")
    doc.addPageTemplates(PageTemplate(id="body", frames=[Frame(48, 42, 516, 712, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer))
    doc.build(story)
    md.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Created {pdf.name} and {md.name}")


if __name__ == "__main__":
    main()
