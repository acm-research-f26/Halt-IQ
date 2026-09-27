# Plug the Jev critic into Vedanshi's harness

Use the small adapter without replacing an existing writer, dataset, or episode runner. Install this checkout in that harness's Python environment with `python -m pip install -e "/path/to/Halt IQ"`, or place this checkout on `PYTHONPATH`. Dataset CLI commands are designed to run from the checkout; this is not a standalone packaged dataset release.

```python
from pathlib import Path
from haltiq.budget import BudgetLedger
from haltiq.providers import JevClient
from haltiq.critics import JevCritic

# Reuse one persistent ledger for all runs in this project.
ledger = BudgetLedger(Path("/path/to/Halt IQ/.haltiq/budget.sqlite"), limit_usd=5)
critic = JevCritic(JevClient(budget=ledger), threshold=0.8)

review = critic.review({
    "question": question,
    "evidence": retrieved_passages,  # list[str]
    "draft": writer_answer,
})

if review.approved:
    final_answer = writer_answer
else:
    revision_feedback = review.feedback
    # Feed revision_feedback into the existing writer, while respecting its round cap.

# Save all of these to preserve the comparison and cost accounting.
print(review.scores, review.call.model, review.call.input_tokens)
print(review.call.latency_ms, review.call.cost_usd)
```

Set `TYPESAFE_API_KEY` in the environment before constructing the client. Model requests contain only question, evidence, and draft. Do not supply reference answers, experiment labels, or previous evaluator verdicts as evidence.

`review.approved` is the control decision. `review.feedback` is a deterministic repair instruction from failed checks, not a generated explanation of a particular factual error. `review.scores` contains three raw marginal probabilities. Retain your independent answer evaluator to measure correctness.

An existing critic interface that returns text can translate this result into `"APPROVED" if review.approved else review.feedback`, while separately logging the structured fields. That is a suggested integration boundary, not a tested patch against Vedanshi's current harness.

Run a paired comparison on the same task IDs and round cap. Our loop shares the initial draft but lets feedback create different later answers. If her harness replays fixed trajectories, label that as a stopping-policy comparison: changing the critic during replay cannot change an already-recorded writer trajectory.

Suggested division: Yash owns the typed rubric, adapter, threshold development, and failure inspection. Vedanshi owns the same-task comparisons and analysis of success, rounds, judge calls, latency, and costs. Freeze thresholds before evaluating the test split.

The standalone implementation is maintained on the [`Yash` branch](https://github.com/acm-research-f26/Halt-IQ/tree/Yash). Clone that branch using the [README setup instructions](../README.md). The adapter can be imported from this checkout; integrating it into Vedanshi's harness remains a separate step.
