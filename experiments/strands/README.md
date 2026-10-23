# Native Strands checks

These small controlled tests use real model inference to check whether changing
an answer or an observed order status changes the decision score. They are
diagnostics, not HotpotQA or official Tau2 task-performance results.

| Check | Correct / permitted score | Wrong / prohibited score |
|---|---:|---:|
| Eiffel Tower city | 0.9046 | 0.1511 |
| Largest planet | 0.9107 | 0.2039 |
| Cancel pending versus delivered order | 0.5563 | 0.5559 |
| Return delivered versus pending order | 0.5951 | 0.5857 |

The factual rows show the released head; the accompanying CSV also records the
previously trained head. Both separate these simple correct and incorrect answers.
The order-status changes barely affect scores in the two action examples.
At the predeclared 0.8 threshold, all four proposed actions are rejected,
including the two permitted actions. Action-specific data and calibration need
testing before this model can be relied on as an action gate.

The fact and action checks use different criteria and temperatures; their score
levels are not directly comparable. The action check uses the complete pinned
retail policy, public conversation and proposed-tool schema, with temperature 1
and no Hotpot calibration. Within each action pair, only the observed order
status changes. Labels stay outside model inputs, and no benchmark tasks or
real customer data are loaded. No proposed tool is executed.

`input-sensitivity-*` contains four encoder forwards and eight head scores;
`tau-action-*` contains four forwards and four head scores. JSON summaries retain
earlier zero-forward setup attempts separately. Startup, encoding, scoring, idle
and wall timings overlap and must not be added. Configurations are in
`../../configs/strands/`; native reproduction uses the corresponding
`../../scripts/probe_*` scripts with external local weights and resource checks.

For the action probe, first export an unchanged CPU control copy of the pinned
released checkpoint in the Strands runtime:

```python
from pc_strands.train_head import copy_control_checkpoint
copy_control_checkpoint("path/to/released-checkpoint", "path/to/new-control-copy")
```

Pass the new copy as `--checkpoint-dir`. This preparation reproduced the frozen
head-file hash locally without training or inference. The raw downloaded file
uses different serialization. Action scoring still uses temperature 1; the
factual checks use the separately recorded Hotpot calibration temperatures.
