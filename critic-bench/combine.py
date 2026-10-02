"""Can critics combined beat each critic alone? Logistic regression on `extra` (200 drafts).

    python3 combine.py      # cached scores only; no model calls; writes combine.md

Features per draft: evidence_match, consistency, laya, qwen8b-logprob (as log-odds,
because 138/200 raw scores are >= 0.99), llm decision (approve = 1), and flags for
yes/no and UNKNOWN drafts. Label: strict exact match. 5-fold stratified
cross-validation gives each draft a score from a model that never saw it
("out-of-fold"), so the AUROC is not inflated by fitting.
"""

import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drafts import load_drafts, normalize
from report import load_scores

HERE = Path(__file__).resolve().parent
FEATURES = ["evidence_match", "consistency", "laya", "logprob_logodds", "llm_approve", "is_yes_no", "is_unknown"]
LLM_APPROVALS = 159  # how many `extra` drafts llm alone approves


def features(draft, scores):
    get = lambda critic: scores[critic][draft["draft_id"]]
    p = min(max(get("qwen8b-logprob")["score"], 1e-11), 1 - 1e-11)
    answer = normalize(draft["draft"])
    return [get("evidence_match")["score"], get("consistency")["score"], get("laya")["score"],
            max(-25.0, min(25.0, math.log(p / (1 - p)))),
            1.0 if get("llm")["decision"] == "approve" else 0.0,
            1.0 if answer in ("yes", "no") else 0.0, 1.0 if answer == "unknown" else 0.0]


def main():
    scores, drafts = load_scores(), load_drafts("extra")
    X = np.array([features(d, scores) for d in drafts])
    y = np.array([d["correct"] for d in drafts], dtype=int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]

    lines = ["# Combining critics (logistic regression, `extra`, strict label)", "",
             f"{len(y)} drafts, {y.sum()} correct. Out-of-fold = 5-fold stratified cross-validation (seed 42).", "",
             "| Score | AUROC |", "|---|---:|",
             f"| **combined (out-of-fold)** | **{roc_auc_score(y, oof):.3f}** |"]
    for i, name in enumerate(FEATURES[:5]):  # the flags alone are not critics
        lines.append(f"| {name} alone | {roc_auc_score(y, X[:, i]):.3f} |")

    # Approve the ~159 drafts with the highest out-of-fold score, as many as llm alone approves.
    top = np.argsort(-oof)[:LLM_APPROVALS]
    right = int(y[top].sum())
    llm = X[:, 4] == 1
    lines += ["", f"Approving the top {LLM_APPROVALS} drafts by out-of-fold score (llm alone approves {int(llm.sum())}):", "",
              "| Rule | Approved | Accuracy when approved | Wrong approvals | Wrongly rejected |", "|---|---:|---:|---:|---:|",
              f"| combined, top {LLM_APPROVALS} | {LLM_APPROVALS} | {right}/{LLM_APPROVALS} ({right / LLM_APPROVALS:.0%}) "
              f"| {LLM_APPROVALS - right}/{len(y) - y.sum()} | {y.sum() - right}/{y.sum()} |",
              f"| llm alone | {int(llm.sum())} | {int(y[llm].sum())}/{int(llm.sum())} ({y[llm].mean():.0%}) "
              f"| {int((1 - y[llm]).sum())}/{len(y) - y.sum()} | {int(y.sum() - y[llm].sum())}/{y.sum()} |"]

    model.fit(X, y)  # coefficients from a fit on all 200, on standardized features
    coefs = model.named_steps["logisticregression"].coef_[0]
    lines += ["", "Coefficients (fit on all 200; standardized, so sizes are comparable; + means more likely correct):", "",
              "| Feature | Coefficient |", "|---|---:|"]
    lines += [f"| {name} | {c:+.3f} |" for name, c in zip(FEATURES, coefs)]
    lines += [f"| intercept | {model.named_steps['logisticregression'].intercept_[0]:+.3f} |", ""]
    text = "\n".join(lines)
    (HERE / "combine.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
