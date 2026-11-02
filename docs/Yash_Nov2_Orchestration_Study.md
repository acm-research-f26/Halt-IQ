# Yash — November 2 orchestration study

Prepared November 2, 2026 (America/Chicago). **No new GPU benchmark results are claimed by this document.** Teammate files and previous results are preserved. The user approved this new study and selected Colab rather than local GPU execution. Subsequent account/presence/Drive approvals allowed both new Yash Drive copies to be started; the pre-benchmark engineering checkpoint below supersedes the initial manual-start-only status.

## Research question

Can a Strands-informed linear policy improve the quality/resource tradeoff by deciding **STOP, REVISE, or INVOKE**, rather than applying a second binary stopping gate?

The inherited six-round gate had a concrete limitation: the verified affine feature bounds made all five original heads approve at threshold 0.5 throughout that feature contract. An AND with an always-approving component cannot add a stopping restriction. This motivates changing the learned decision and collecting new action branches. It does not establish that the replacement will improve performance. Newer team heads must be assessed separately.

The latest inspected team integration is `dd3f4a6d2984f300590f0aa760d907388cf226bf`. Its cached answer-selection and headroom analyses are useful references, but do not provide observed outcomes for a newly invoked challenger. This study therefore generates its own training branches instead of relabeling old trajectories as third-agent interactions.

## What is multi-agent about it?

An explicit orchestrator maintains public question/evidence, current draft, reviewer feedback, verifier score, recent answer change, challenger usage, and remaining budget. It executes separate AWS Strands SDK agent calls with independent role prompts and structured responses. All roles use the same pinned language model; this is role-specialized orchestration, not a claim of different model weights or parallel execution.

1. **Writer:** proposes an answer with supporting sentence citations.
2. **Reviewer:** independently checks that answer against the supplied public passages and returns approval plus feedback.
3. **Challenger, when invoked:** supplies an alternative/check and citations; its actual response is passed into the writer's revision prompt. It uses the existing HotpotQA passages, not an external search tool or hidden evidence.

The Strands verifier/reranker provides another observation. It is not counted as a third language-model agent. Every mixed episode starts with writer → reviewer → verifier. The orchestrator then executes:

| Action | Actual transition |
|---|---|
| STOP | Return the current draft; no additional agent call. |
| REVISE | Writer uses current reviewer feedback → reviewer checks the new draft → verifier scores it. |
| INVOKE | Challenger returns an evidence-linked check → writer uses it → reviewer checks the revision → verifier scores it. |

There is at most one challenger invocation, three drafts, and ten attempted role calls in matched arms. One format correction is allowed and consumes that call budget. A budget cap is recorded separately from policy approval. The final answer is the **last returned draft** for every primary comparison.

## Training and freeze

`data/Yash_Nov2_HotpotQA_Plan.json` freezes 20 training, 10 development, and 20 test questions from pinned HotpotQA distractor validation. It excludes 350 previously used or prepared IDs. Whole questions are disjoint; type/level sampling uses public fields before label hashes are computed. Those previous IDs are used only to prevent overlap, not to train the new policy from previous outcomes.

On training and development questions, the runner starts from a common public state, executes every available STOP/REVISE/INVOKE branch, and repeats one-step branching at successful depth-one children. Shared ancestors are charged once in physical collection accounting. Failed branches remain logged with zero task score and their actual call cost; failed initial states remain in coverage counts.

The label is the highest measured one-transition utility:

`answer F1 − 0.02 × incremental attempted role calls / 10`

Ties prefer STOP, then REVISE, then INVOKE. This is **supervised one-step utility imitation**, not reinforcement learning or proof of an optimal multi-step policy. The new linear head uses 11 public features including the continuous Strands score, reviewer approval, remaining drafts, answer change/repetition, evidence/contradiction cues, challenger usage, and answer/feedback lengths.

Five initializations (0–4) fit on training questions with question-balanced masked cross-entropy and L2 regularization. Default settings are 200 epochs, learning rate 0.03, L2 0.001. Each seed's earliest best development-utility checkpoint is saved. All five checkpoint hashes freeze **before any test inference**. Seed 0 is the primary policy; seed 1 is a predeclared replication, never selected from test scores. These settings are our study design, not hyperparameters from the linked paper.

## Matched evaluation and resources

Both new Colabs run the same pinned Qwen2.5-1.5B writer in FP16, Hobson verifier, Qwen3.5-2B verifier base, and BGE reranker. Writer/reviewer/challenger output caps are 512/256/256 tokens. Model revisions and actual package/device metadata are retained in runtime receipts. A real three-role engineering qualification precedes training; it is not benchmark accuracy.

Each run evaluates seven arms on the **same 20 test questions**:

| Arm | Purpose |
|---|---|
| fixed1 | One writer call reference. |
| legacy_combined6 | Existing six-round combined function under the current shared output caps; secondary, different budget. |
| strands | Strands-only stopping under matched three-draft/ten-call caps. |
| binary_and | Inherited binary head AND Strands, under the same matched caps. |
| always_revise3 | Fixed revision control under the matched caps. |
| always_invoke | One actual challenger invocation then revisions, under the matched caps. |
| learned3 | New Strands-informed three-action linear policy. |

Primary comparisons are learned3 against matched strands, binary_and, and always_revise3. The six-round legacy arm is not an equal-budget control or an exact reproduction of older 768-token runs. An optional best-verifier-prefix selection is diagnostic only and does not replace primary last-draft scores.

Record answer EM/F1, supporting-fact and joint metrics, attempted/completed role calls, attempted writer iterations, route counts, critic calls, response tokens, wall time, failure coverage, and termination reasons. Raw response receipts correlate client and response IDs; missing or invalid receipts yield unknown tokens, not zero. Verifier/reranker compute and training collection costs are reported separately from test role-response tokens. No dollar-cost saving is inferred from token counts alone.

All planned questions remain in denominators, including failed, interrupted, and pending attempts. Two runs on the same panel are **20 distinct questions, not 40 independent questions**. This is a small paired pilot. Report negative, identical, and positive results; do not change the frozen policy from held-out feedback.

## Manual start and outputs

Use the new Yash primary notebook on one provisioned account and the new Yash replication notebook on the other. Select a GPU, run all cells, and complete the accounts' own Drive/runtime prompts. The notebooks contain verified source and the frozen plan, so they do not depend on a mutable GitHub checkout. Each run uses a new Yash-owned output directory and refuses existing study artifacts.

The runner exports `summary.json`, `results.csv`, `Yash-Nov2-Study-Report.txt`, `Yash-Nov2-Answer-F1.png`, all five heads, the freeze/configuration, public state/route traces, training branches, and physical agent/critic receipts. The notebook exports an evidence ZIP, excluding model weights and caches. Keep it for review and the professor/poster writeup.

The automatic pre-deadline stop targets November 2 at 09:20 Chicago, leaving time to export before the requested 09:30 cutoff. Runtime availability and manual startup still determine how much completes. An interrupted export is evidence of partial work, not a completed study. New tau-squared testing is outside this HotpotQA reopening; earlier tau artifacts remain separate.

## Pre-benchmark engineering checkpoint

At the November 2, approximately 01:40 Central checkpoint, both new Yash Drive copies had connected to Tesla T4 GPUs and their approved Drives. Three startup attempts per account had exited during the invented Paris qualification; no training, development, or test questions had been collected and no new study head had been fitted.

The first attempt exposed an HTTP route-binding error, corrected by binding the actual FastAPI `Request` class before route registration and verifying a real streaming HTTP request on CPU. The next two attempts exposed invalid writer messages: descriptive citation keys, then an extra trailing quote and numeric citation titles despite explicit examples. The strict parser rejected these messages. No citation was guessed or repaired into a scored answer. Failed engineering attempts and their raw-response/archive locators are retained separately from benchmark denominators and costs.

The new schema-constrained implementation is being qualified before any benchmark collection. It applies identically across every comparison arm, preserves existing token/context/call limits, and retains strict final validation. Each request gets a fresh parser and token enforcer; only tokenizer preprocessing is cached. Citation objects use the existing accepted `title`/`sentence_index` contract. Required fields and JSON types are constrained during decoding; negative sentence indices, incomplete output, duplicate keys and malformed raw JSON remain rejected after generation, with physical receipts/correlation retained before an HTTP failure. A formatting constraint establishes message structure; it does not establish answer correctness or valid supporting evidence. This checkpoint is a record of startup failures and a pending live qualification, not evidence that the classifier or orchestration improves accuracy.

`requirements-nov2-colab.txt` preserves the original dependencies and adds `lm-format-enforcer==0.11.3`. Its released Transformers helper uses an import removed in Transformers 5.17, so the owned runtime adapts the library's MIT-licensed tokenizer bridge and uses the public `TokenEnforcer` and `generate(prefix_allowed_tokens_fn=...)` APIs without changing the model or installed package. Effective limits are frozen: at most 20 citation items, 12 consecutive whitespace characters, and flexible field order. The library does not enforce numeric `minimum`; strict postvalidation enforces nonnegative integer indices. It can filter some valid merged-token encodings of escaped quotes, although an alternative tokenization of the same JSON is accepted. Tokenizer checks establish these limits, not model accuracy. Primary references: [released integration](https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/integrations/transformers.py), [release](https://pypi.org/project/lm-format-enforcer/0.11.3/). A reporting repair also retains measured token subsets when an episode's full token total is unknown; it does not replace an unknown total with zero.

Local evidence: `testing/nov2-orchestration-2026-11-02/Yash-Live-Colab-Qualification-Failures-20261102.json` and its six saved browser-output views. Cloud archives have observed identities; their bytes had not yet been downloaded and independently hash-checked at this checkpoint.

## Separate paper summary

[Zhang, Sun, Guo, and Gogate, arXiv:2506.22385v2](https://arxiv.org/pdf/2506.22385v2) studies defeasible **video** entailment. Its fixed pipeline generates opposing rationales, integrates video descriptions and Whisper transcripts, and uses another language model to correct/select an explanation. The 986 examples balance strengthening and weakening updates. Table 3 reports Video-ChatGPT accuracy of 61.05% before and 77.79% after the proposed pipeline.

The useful adaptation here is a competing explanation followed by evidence-based correction. The paper does not train a Strands controller, linear routing head, or stopping policy; its video gains are not HaltIQ results. A standalone brief is provided separately from study outputs.

## Source and verification references

- Plan: `data/Yash_Nov2_HotpotQA_Plan.json`; pinned dataset revision `1908d6afbbead072334abe2965f91bd2709910ab`.
- Inherited-head mathematical evidence: `testing/team-alignment-2026-10-30/Yash-Retest-Design/Yash-Inherited-Head-Nonbinding-Proof.json` and `Yash-Head-Nonbinding-Independent-Verification.json` in Yash's local project artifacts.
- New implementation: `haltiq_pair/nov2_{orchestration,policy,protocol,runtime,runner}.py` and corresponding `tests_pair/test_nov2_*.py`.
- Offline plan receipt: `testing/nov2-orchestration-2026-11-02/Yash-Plan-Preparation-20261102.json` in local project artifacts; it records zero new model calls at preparation.
