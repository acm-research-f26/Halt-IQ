"""Build a plain-language team account of the completed Strands study.

Detailed settings and evidence remain in the linked technical materials.
This document builder runs no models and makes no API calls.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors

from build_team_progress import BUNDLE, REPO, build_document


def pipeline_figure():
    figure = Drawing(516, 80)
    labels = [("Question and", "supplied evidence"), ("Writer creates", "an answer"),
              ("Strands checks", "the answer"), ("Accept it or", "try another draft")]
    for i, lines in enumerate(labels):
        x = i * 132
        figure.add(Rect(x, 22, 120, 44, fillColor=colors.HexColor("#EEF3F7"), strokeColor=colors.HexColor("#657787"), strokeWidth=.65))
        for j, value in enumerate(lines):
            figure.add(String(x + 60, 49 - 12 * j, value, textAnchor="middle", fontName="Helvetica", fontSize=9))
        if i < 3:
            figure.add(Line(x + 120, 44, x + 130, 44, strokeColor=colors.black, strokeWidth=.8))
            figure.add(Polygon([x+130,44,x+126,46,x+126,42], fillColor=colors.black, strokeColor=None))
    return figure


PAGES = [
    [
        ("title", "Strands study: what I built and learned"),
        ("meta", "Yash | Team research update | October 18, 2026"),
        ("h1", "The main finding"),
        ("p", "I tested whether Strands could decide when an answer was good enough to stop revising, and whether fine-tuning would improve those decisions. I also compared it with Jev. On the final 20 questions, every tested approach answered 10 correctly. Extra drafts and fine-tuning did not improve the final answers in this test."),
        ("h1", "What I built"),
        ("bullet", "Set up Strands locally and checked that it could run on my PC."),
        ("bullet", "Tested the original model before training it, then fine-tuned its decision-making part."),
        ("bullet", "Built a fair comparison using the same questions and saved answers for each approach, with up to ten drafts per question."),
        ("bullet", "Recorded answer quality, the quality of the model's judgments, failures and actual work performed. The code, results and trained weights are on GitHub."),
        ("h1", "How the system works"),
        ("p", "The writer, Qwen, produces an answer using the question and the supplied evidence. Strands acts as the checker: it estimates whether the answer is correct. If its confidence reaches 80%, it accepts the answer. Otherwise, the system considers another draft, until it reaches the draft limit."),
        ("diagram", pipeline_figure(), "The decision rule being tested. For the comparison, I saved the drafts first and then applied each checker to the same answers.", "flowchart LR\n    Q[Question and evidence] --> W[Writer creates an answer]\n    W --> S[Strands checks the answer]\n    S --> D[Accept or try another draft]"),
        ("p", "Strands decides whether to stop; it does not write corrections or explain how to fix an answer in this experiment. That distinction matters because a checker can recognize a problem without helping the writer produce a better answer."),
        ("h1", "What fine-tuning changed"),
        ("p", "I trained only the small decision layer that turns the model's understanding into a correctness judgment. The rest of Strands stayed unchanged. I checked that training changed the intended weights and that the saved model worked the same way after reloading. This was supervised training; reinforcement learning was not used."),
    ],
    [
        ("h1", "How I tested it fairly"),
        ("p", "I used HotpotQA, a question-answering dataset that provides a question and several evidence passages. The writer and checker saw the supplied evidence, but never the reference answer. The reference was used only to score their output."),
        ("table", [
            ["Question group", "Size", "Purpose"],
            ["Initial check", "20", "Test the original Strands model before training"],
            ["Training and preparation", "90", "60 for learning, 15 for choosing the trained version, 15 for adjusting confidence scores"],
            ["Final comparison", "20 new", "Compare every approach on questions kept separate from training"],
        ], [156, 60, 300]),
        ("p", "The final questions were chosen before seeing the models' answers. Each question had ten drafts: one initial answer and nine revisions. I compared stopping after exactly 1, 2, 3, 5 or 10 drafts with letting Strands or Jev decide when to stop. I also tested keeping the highest-confidence answer among drafts already considered."),
        ("p", "I tested both the original and fine-tuned Strands versions. Each had its confidence scores adjusted using the same separate preparation questions. This adjustment, called calibration, aims to make confidence more meaningful. It does not change an answer or prove better performance."),
        ("h1", "What the scores mean"),
        ("bullet", "Exact match: the share of answers matching an accepted reference after standard text cleanup. This is the main answer-quality score."),
        ("bullet", "Token F1: how much the answer's words overlap the reference. It gives partial credit when an answer is not an exact match."),
        ("bullet", "Judgment quality: whether the checker gives correct answers higher confidence than incorrect ones, and how reliable those confidence scores are. The technical report includes ranking and confidence-error measures."),
        ("bullet", "Repairs and regressions: whether later drafts fix an initially wrong answer or spoil an initially correct one. Failed questions still count in the final score."),
        ("p", "I used the same standard answer scoring and compatible result format as Vedanshi's framework. Hardware, prompts and model limits can differ, so scores from separate experiments are not automatically direct comparisons."),
        ("h1", "Where it ran"),
        ("p", "Qwen and Strands ran on my Windows PC with an NVIDIA graphics card. I limited background resource use and added rests so the PC remained available for other work. Jev ran through its hosted API. Full software versions and settings are available in the technical report."),
    ],
    [
        ("h1", "What happened"),
        ("p", "All approaches scored 50% exact match and 0.54 token F1 on the same 20 final questions. No later draft fixed an initially wrong answer, and none spoiled an initially correct one. All final questions completed without a failed episode."),
        ("table", [
            ["Approach", "Correct answers", "Average drafts selected"],
            ["Stop after a fixed number of drafts", "10 out of 20 at every tested depth", "1, 2, 3, 5 or 10"],
            ["Original Strands, up to ten drafts", "10 out of 20", "5.5"],
            ["Fine-tuned Strands, up to ten drafts", "10 out of 20", "5.5"],
            ["Jev, up to ten drafts", "10 out of 20", "3.75"],
        ], [240, 144, 132]),
        ("p", "The shorter three-draft checks and the versions that kept the highest-confidence visited answer also returned the same answer scores. The fine-tuned Strands model made the same stopping choices as the original version."),
        ("h1", "Why extra drafts did not help here"),
        ("p", "For 18 of the 20 questions, the writer repeated the same answer through all ten drafts. The remaining two changed their wording but stayed incorrect. The checker was choosing among answers with the same correctness, so stopping earlier or later could not improve the final score."),
        ("p", "For example, one question expected 'the Waters of Death', but every draft answered 'the shore'. Strands kept asking for another attempt; Jev accepted the first answer. Both ended with the same wrong answer. Detecting doubt is useful only if the next step can produce a better candidate."),
        ("h1", "What I learned about the checkers"),
        ("p", "Jev did a better job of giving correct first answers higher confidence than incorrect ones. That did not translate into more correct final answers. Fine-tuning did not improve Strands' judgments on this final set; its confidence-error scores were slightly worse."),
        ("p", "A checker's accuracy among answers it approves is different from overall answer accuracy. I kept those measures separate in the detailed results. Recognizing correctness and improving an answer are two different capabilities."),
        ("h1", "How far these results go"),
        ("p", "This was a small test of 20 independent questions. The 200 saved drafts are repeated attempts on those questions, not 200 independent examples. The result shows no improvement in this test; it does not prove that fine-tuning or additional revisions can never help. The initial check used different questions, so it is not a before-and-after accuracy comparison."),
    ],
    [
        ("h1", "What I would try next"),
        ("h2", "1. Give the writer useful feedback"),
        ("p", "Proposal: when Strands rejects an answer, ask a separate evidence checker to identify a specific mistake and cite the relevant passage. Give that feedback to the writer before it revises."),
        ("p", "Benefit: this could produce a genuinely different answer instead of repeating a generic self-check. Risk: the extra checker adds cost and delay, and its feedback could be wrong."),
        ("p", "Test: compare ordinary revisions with evidence-based feedback on new questions. Keep the writer settings the same, limit extra checker calls, and measure repairs, regressions, answer quality and total cost."),
        ("h2", "2. Train the decision to ask whether another step is worth it"),
        ("p", "Proposal: train the controller to estimate whether another verification step is likely to improve the answer enough to justify its cost. Start with supervised examples before considering reinforcement learning."),
        ("p", "Benefit: this connects the stopping decision to the value of another action, rather than confidence alone. Risk: it needs richer training examples, and a poorly chosen reward could teach the wrong behavior."),
        ("p", "Test: develop it on separate training questions, then compare it with the current checker and fixed draft limits on a new final set. Measure answer quality and the work each approach actually performs. Neither proposed experiment has been run."),
        ("h1", "Time, cost and practical limits"),
        ("p", "The final local comparison took about 38 minutes, including startup and planned rests. It generated 200 drafts and checked each with Strands. The final Jev comparison cost less than one cent; total hosted spending across the study was about five cents. Local hardware and electricity were not priced."),
        ("p", "Because all drafts were generated before comparing stopping rules, choosing fewer drafts does not demonstrate real speed or token savings. That needs a later test where the system actually stops generation. An earlier preparation run ended incomplete; it was kept separate from the completed training and final comparison."),
        ("h1", "Where the work is available"),
        ("p", "GitHub contains the implementation, recorded answers, scores, trained decision layer and instructions for checking the results. The detailed reports hold the exact model settings and evidence needed for reproduction."),
        ("link", "Code, results and trained model", BUNDLE),
        ("link", "Detailed methods and Strands results", f"{REPO}/blob/Yash/results/strands-20261018/reports/professor-review.pdf"),
        ("link", "Detailed Jev comparison", f"{REPO}/blob/Yash/results/strands-20261018/reports/optional-jev-companion.pdf"),
        ("meta", "Dataset: HotpotQA, Yang and colleagues, EMNLP 2018, shared under CC BY-SA 4.0. Model credits and full licenses are included with the GitHub materials."),
    ],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_document(PAGES, args.output_dir, "strands-head-finetuning-writeup", "Strands study: what I built and learned", "HaltIQ team update | October 18, 2026")


if __name__ == "__main__":
    main()
