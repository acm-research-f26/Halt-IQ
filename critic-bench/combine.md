# Combining critics (logistic regression, `extra`, strict label)

200 drafts, 115 correct. Out-of-fold = 5-fold stratified cross-validation (seed 42).

| Score | AUROC |
|---|---:|
| **combined (out-of-fold)** | **0.686** |
| evidence_match alone | 0.597 |
| consistency alone | 0.584 |
| laya alone | 0.605 |
| logprob_logodds alone | 0.633 |
| llm_approve alone | 0.588 |

Approving the top 159 drafts by out-of-fold score (llm alone approves 159):

| Rule | Approved | Accuracy when approved | Wrong approvals | Wrongly rejected |
|---|---:|---:|---:|---:|
| combined, top 159 | 159 | 103/159 (65%) | 56/85 | 12/115 |
| llm alone | 159 | 100/159 (63%) | 59/85 | 15/115 |

Coefficients (fit on all 200; standardized, so sizes are comparable; + means more likely correct):

| Feature | Coefficient |
|---|---:|
| evidence_match | +0.791 |
| consistency | +0.555 |
| laya | +0.214 |
| logprob_logodds | +0.142 |
| llm_approve | +0.202 |
| is_yes_no | +0.417 |
| is_unknown | -0.337 |
| intercept | +0.184 |

