# HaltIQ: writer–critic collaboration with a learned stopping controller

This document distinguishes the **completed cached-trajectory experiment** from the **proposed live integration**. The controller predicts **p(stop)**, not answer correctness.

## 1. Completed experiment: collect, score, train, replay

```mermaid
flowchart TD
    Q["150 HotpotQA questions + supplied passages<br/>50 train / 20 validation / 80 test"]
    W["Writer role: frozen Qwen2.5-1.5B<br/>Produce or revise a short answer"]
    C["Critic role: frozen Qwen2.5-1.5B<br/>APPROVED or correction"]
    CACHE["Save all six rounds per question<br/>Answers, feedback, approval, tokens<br/>900 rounds total"]
    R["BGE reranker<br/>Question + saved answer select top-2 passages"]
    S["Strands Decider 2B<br/>Score each unique saved draft<br/>166 question–answer pairs"]
    JOIN["Join score by exact question ID + answer<br/>900 rounds with Strands scores"]
    TRAIN["50 training questions<br/>Train original / replace / add heads<br/>Five seeds per variant"]
    GOLD["Reference answers / offline F1 labels"]
    REWARD["Training reward<br/>Answer F1 minus token-use penalty"]
    VAL["20 validation questions<br/>Select checkpoint by validation reward<br/>Keep stop threshold at 0.5"]
    TEST["80 test questions<br/>Replay saved prefixes with selected heads"]
    METRICS["Compare with fixed rounds 1–6<br/>Answer F1, EM, average rounds<br/>Cached Qwen tokens as % of Fixed-6"]

    Q --> W
    W --> C
    C -->|"Feedback for the next revision, until six rounds"| W
    W --> CACHE
    C --> CACHE
    CACHE --> R
    Q -->|"Supplied evidence"| R
    R --> S
    CACHE -->|"Saved draft"| S
    CACHE --> JOIN
    S --> JOIN
    JOIN -->|"Training partition"| TRAIN
    GOLD -->|"Offline scoring only"| REWARD
    REWARD --> TRAIN
    TRAIN --> VAL
    JOIN -->|"Validation partition"| VAL
    VAL --> TEST
    JOIN -->|"Test partition"| TEST
    TEST --> METRICS
    GOLD -->|"Offline test scoring only"| METRICS
```

**What happened:** Qwen generated all six rounds before controller evaluation. Strands subsequently scored the saved drafts. The learned controller selected a stopping point during replay; it did not interrupt collection or cause new answers to be generated.

Gold answers, F1, and EM are used for offline rewards and evaluation only. They are not provided to the writer, critic, reranker, or prediction features. Validation selects checkpoints; test outcomes do not update the weights.

## 2. Proposed live loop: where multi-agent orchestration fits

The diagram shows the **add-Strands variant**. It is a proposed execution schedule, not a live experiment already measured.

```mermaid
flowchart TD
    INPUT["Question + supplied evidence"]
    WRITER["WRITER AGENT<br/>Frozen Qwen drafts or revises an answer"]
    CRITIC["CRITIC AGENT<br/>Frozen Qwen returns approval or correction"]
    RERANK["BGE top-2 evidence selection"]
    STRANDS["STRANDS CRITIC SCORE<br/>Evaluate the current draft"]
    FEATURES["Build features from observed rounds only<br/>Distance, round progress, change trend,<br/>Qwen approval, previous-draft flag, Strands score"]
    HEAD["LEARNED CONTROLLER<br/>Linear score + sigmoid → p(stop)"]
    DECISION{"p(stop) ≥ 0.5<br/>or round six reached?"}
    OUTPUT["Return current answer"]
    NEXT["Supply current draft + Qwen critique<br/>to the next writer round"]

    INPUT --> WRITER
    WRITER --> CRITIC
    WRITER --> RERANK
    INPUT -->|"Evidence"| RERANK
    RERANK --> STRANDS
    WRITER -->|"Current answer"| STRANDS
    WRITER -->|"Observed answer history"| FEATURES
    CRITIC -->|"Approval"| FEATURES
    STRANDS --> FEATURES
    FEATURES --> HEAD
    HEAD --> DECISION
    DECISION -->|"Yes: STOP"| OUTPUT
    DECISION -->|"No: CONTINUE"| NEXT
    CRITIC -->|"Correction / feedback"| NEXT
    NEXT --> WRITER
```

The writer and Qwen critic are separate **roles**, even though they share the same frozen model. Strands supplies an additional critic score. The controller orchestrates **whether another revision occurs**; it does not write answers, route tasks among specialists, or determine when an unrelated customer conversation should end.

Only a live run would establish which calls are actually avoided. The two critic paths are logically separate; this diagram does not claim they were executed concurrently.

## 3. Three classifier variants

```mermaid
flowchart LR
    COMMON["Shared inputs<br/>1. Answer distance<br/>2. Round / 6<br/>3. Change in distance<br/>4. Previous-draft flag"]
    APPROVAL["Qwen critic approval<br/>0 or 1"]
    SCORE["Strands score<br/>Continuous value in 0–1"]
    ORIGINAL["Original head<br/>5 inputs / 6 parameters"]
    REPLACE["Variant A: replace approval<br/>5 inputs / 6 parameters"]
    ADD["Variant B: add Strands<br/>6 inputs / 7 parameters"]
    P["p(stop)<br/>Stop at ≥ 0.5; force stop at round six"]

    COMMON --> ORIGINAL
    APPROVAL --> ORIGINAL
    COMMON --> REPLACE
    SCORE --> REPLACE
    COMMON --> ADD
    APPROVAL --> ADD
    SCORE --> ADD
    ORIGINAL --> P
    REPLACE --> P
    ADD --> P
```

The shared-input list above groups features for readability; the code preserves the trained feature order:

| Variant | Exact feature order |
|---|---|
| Original | distance, round_fraction, distance_delta, critic_approved, has_previous_draft |
| A: replace | distance, round_fraction, distance_delta, strands, has_previous_draft |
| B: add | distance, round_fraction, distance_delta, critic_approved, has_previous_draft, strands |

**Replacing a feature is not replacing the live feedback loop.** Variant A removes Qwen approval from the controller's inputs, but the saved later drafts were still generated using Qwen critic feedback. A future implementation could obtain the Strands score before deciding whether a Qwen critique is needed for another revision. That schedule and its full cost require separate validation.

## 4. Prediction interface and decision rule

```python
from classifier import StopPredictor

controller = StopPredictor("classifier/results/add/seed0/stop_head.pt")
p_stop = controller.predict(question, rounds_so_far)
stop = controller.should_stop(question, rounds_so_far)
```

Each observed round supplies answer text, a Strands score for variants A/B, and Boolean Qwen approval for original/B. The controller uses no future rounds or gold labels. `question` is retained for interface compatibility; the linear head does not embed its text. Round progress uses the saved cap of six, not the current prefix length.

```text
z = sum(weight × input) + bias
p(stop) = sigmoid(z)
stop = (p(stop) >= 0.5) OR (round == 6)
```

## 5. Measured outcome and cost boundary

All three variants stopped after round one on all 80 test questions across five seeds:

| Variant | Answer F1 | EM | Average rounds | Qwen tokens (% of six rounds) |
|---|---:|---:|---:|---:|
| Original | 0.3588 | 0.2125 | 1.00 | 16.61% |
| A: replace | 0.3588 | 0.2125 | 1.00 | 16.61% |
| B: add | 0.3588 | 0.2125 | 1.00 | 16.61% |

These percentages include original cached **Qwen writer + Qwen critic** tokens through the selected round. They exclude unrecorded Strands/reranker overhead. They are replay cost estimates, not measured live integrated savings. Adding Strands did not improve the stopping decisions under the unchanged training setup.

## Code map

- [Feature calculation and prediction](model.py)
- [Training and validation checkpoint selection](train.py)
- [Replay evaluation and baselines](evaluate.py)
- [Input audit and score join](prepare_inputs.py)
- [Detailed methods and limitations](REPORT.md)
- [Results by seed](results/summary_by_seed.csv)

GitHub renders the Mermaid diagrams in this Markdown file. Other Markdown viewers need Mermaid support.
