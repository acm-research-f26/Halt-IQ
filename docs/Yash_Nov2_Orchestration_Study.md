# Yash — November 2 orchestration study

Closed out November 2, 2026 (America/Chicago), at Yash's request to wrap up using the existing evidence. **Execution and analysis are closed; the declared benchmark remains incomplete.** Both pending Colab connections were cancelled, and both study accounts showed **No active sessions** at the 09:10 closeout check. No new model run or recovery was started. Teammate files and the earlier evidence records remain preserved.

## Current evidence and results

The [concise professor results report](Yash_Nov2_Results_20261102/Yash-Nov2-Results-Report-20261102.pdf) presents the observed eleven-question accuracy, F1 and role-call comparison, with a [results package index](Yash_Nov2_Results_20261102/README.md). The earlier [professor closeout report](Yash_Nov2_Closeout_20261102/Yash-Nov2-Study-Closeout-20261102.pdf) preserves the full study-status record. The [Ashley explanation](Yash_Nov2_Closeout_20261102/Yash-Nov2-Ashley-Explanation-20261102.pdf) provides a one-minute script and short answers. The earlier [partial report](Yash_Nov2_Partial_Results_20261102/Yash-Nov2-Professor-Report-20261102.pdf), [twelve-metric CSV](Yash_Nov2_Partial_Results_20261102/Yash-Nov2-Primary11-Prefix-Table-20261102.csv), [figures and evidence](Yash_Nov2_Partial_Results_20261102/README.md), and [separate paper brief](Yash_Nov2_Partial_Results_20261102/Yash-Nov2-Paper-Brief-20261102.txt) remain preserved. The closeout uses the November study's verified results; earlier discarded Yash panels do not establish completion of this study.

The recovered primary snapshot contains all seven arms on the first **11 of 20 planned test questions**: 77 recorded attempts, 69 completed predictions and 8 failures. The other 63 planned attempts remain unknown. The table below uses those same 11 questions for every arm, with failures scored zero; it is a descriptive partial-panel comparison, not a completed 20-question result.

| Arm | Completed / 11 | Answer F1 | Recorded role calls / question |
|---|---:|---:|---:|
| fixed1 | 11 | 65.2% | 1.00 |
| legacy_combined6 | 10 | 40.9% | 8.55 |
| strands | 10 | 40.9% | 4.91 |
| binary_and | 10 | 40.9% | 4.91 |
| always_revise3 | 10 | 40.9% | 5.64 |
| always_invoke | 8 | 40.9% | 6.00 |
| learned3 | 10 | 59.1% | 2.00 |

The learned policy retained more answer quality with fewer calls than the matched verifier-gated loops in this prefix, but the one-writer baseline was better and cheaper. The primary learned policy chose STOP on all ten completed attempts and used no challenger; actual challenger interactions are verified in the invoke control, training/qualification and the interrupted seed-1 replication. This does not establish an advantage from adaptive three-action routing or from the verifier feature alone. Joint exact match was zero throughout this prefix.

The primary raw-answer, supporting-fact and joint metrics were independently rescored against the pinned dataset. Its unchanged export auditor retains **EVIDENCE_MISMATCH**: the final summary is absent, 77/140 attempts are recorded, and one started writer call on question 12 has no recorded outcome. All 363 physical calls inside the 11-question prefix reconcile, with 783,702 known role-response tokens; whole-run totals remain unknown. A separately labeled derived recovery restores 785 raw receipts byte-exactly from the original compatibility journal, verified against all separately downloaded original receipt hashes and 21 individually downloaded originals. It does not invent a final summary or missing response.

The original seed-1 replication stopped after 20/140 test attempts (19 completed, one interrupted) because a parent-notebook status-file read race interrupted its child. Both runs completed train/development collection and fitted/froze all five heads before testing. The monitor was repaired without changing the frozen inference source, scientific settings or question plan. Its queued recovery connection was cancelled during closeout. Both accounts have no active sessions at the final check; the historical unresolved primary invocation still has no recorded outcome.

The final signed-in primary Drive check found the same 13 study-folder items, no final summary, and the same persisted status as the recovered snapshot: 77 attempts, 69 valid predictions, 780 recorded role calls and 343 verifier calls. Its stored `status: running` is stale and does not override the current no-active-sessions observation. No later result export was found in the inspected local evidence locations. Current tau-squared/7B testing was not run; poster ideas and figures are prepared, while poster submission, presentation and meeting coordination are not verified.

Current local evidence includes `Yash-Primary-Derived-Recovery-Offline-Audit-Final-20261102.json`, `Yash-Primary-Derived-Recovery-Verification-20261102.json`, `Yash-Primary-Unmatched-Invocation-Scope-20261102.json`, and `Yash-Interrupted-Replication-Independent-Audit-20261102.json` under `testing/nov2-orchestration-2026-11-02`. The dated preparation/startup sections below are historical method records.

## Research question

Can a Strands-informed linear policy improve the quality/resource tradeoff by deciding **STOP, REVISE, or INVOKE**, rather than applying a second binary stopping gate?

The inherited six-round gate had a concrete limitation: the verified affine feature bounds made all five original heads approve at threshold 0.5 throughout that feature contract. An AND with an always-approving component cannot add a stopping restriction. This motivates changing the learned decision and collecting new action branches. It does not establish that the replacement will improve performance. Newer team heads must be assessed separately.

The latest inspected team integration is `dd3f4a6d2984f300590f0aa760d907388cf226bf`. Its cached answer-selection and headroom analyses are useful references, but do not provide observed outcomes for a newly invoked challenger. This study therefore generates its own training branches instead of relabeling old trajectories as third-agent interactions.

## What is multi-agent about it?

An explicit orchestrator maintains public question/evidence, current draft, reviewer feedback, verifier score, recent answer change, challenger usage, and remaining budget. It executes separate AWS Strands SDK agent calls with independent role prompts and structured responses. All roles use the same pinned language model; this is role-specialized orchestration, not a claim of different model weights or parallel execution.

1. **Writer:** proposes an answer with supporting sentence citations.
2. **Reviewer:** independently checks that answer against the supplied public passages and returns approval plus feedback.
3. **Challenger, when invoked:** supplies an alternative/check and citations; its actual response is passed into the writer's revision prompt. It uses the existing HotpotQA passages, not an external search tool or hidden evidence.

Vedanshi's separate Hobson/Decider verifier and BGE reranker provide another observation. This score is not an AWS Strands SDK correctness score, and the verifier is not counted as a third language-model role. The Strands SDK runs the separate role agents against Qwen served on the Colab GPU; no Bedrock or AWS-hosted inference is claimed. Every mixed episode starts with writer → reviewer → verifier. The orchestrator then executes:

| Action | Actual transition |
|---|---|
| STOP | Return the current draft; no additional agent call. |
| REVISE | Writer uses current reviewer feedback → reviewer checks the new draft → verifier scores it. |
| INVOKE | Challenger returns an evidence-linked check → writer uses it → reviewer checks the revision → verifier scores it. |

There is at most one INVOKE transition, three drafts, and ten attempted role calls in matched arms. A transport-completed response rejected by subsequent parsing may receive one format correction, consuming another call inside that budget; one INVOKE transition can therefore make two physical challenger calls. Transport/server-validation exceptions fail the branch or episode. A budget cap is recorded separately from policy approval. The final answer is the **last returned draft** for every primary comparison.

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

The full-plan auditor retains all 20 planned questions and explicitly reports pending coverage; its zero-filled incomplete totals are not final quality estimates. The separately labeled descriptive prefix above uses only the same first 11 fully attempted planned questions, retains all failures at zero and leaves the remaining nine questions unknown. Two runs on the same panel are **20 distinct questions, not 40 independent questions**. This is a small paired pilot. Report negative, identical, and positive results; do not change the frozen policy from held-out feedback.

Test arms make separate initial writer calls rather than sharing one cached draft. In the recovered first 11 questions, their initial draft objects match across all seven arms. Common-state cloning is used during action-branch training. Generation is greedy; seeds 0/1 select the deployed learned checkpoint and cyclic arm order, not language-model sampling randomness. The two recorded collections have identical training-example bytes and train/development record digests; corresponding fitted weight/bias arrays match, while provenance makes head payload hashes differ.

## Prepared workflow and outputs (historical execution procedure)

This record is closed. The instructions below describe the prepared execution procedure; no recovery or further model testing remains queued by this closeout.

Use the new Yash primary notebook on one provisioned account and the new Yash replication notebook on the other. Select a GPU, run all cells, and complete the accounts' own Drive/runtime prompts. The notebooks contain verified source and the frozen plan, so they do not depend on a mutable GitHub checkout. Each run uses a new Yash-owned output directory and refuses existing study artifacts.

The runner exports `summary.json`, `results.csv`, `Yash-Nov2-Study-Report.txt`, `Yash-Nov2-Answer-F1.png`, all five heads, the freeze/configuration, public state/route traces, training branches, and physical agent/critic receipts. The notebook exports an evidence ZIP, excluding model weights and caches. Keep it for review and the professor/poster writeup.

The execution windows were 08:50 Central for new/repaired templates and 09:20 for the original primary notebook. These are historical time guards, not current completion targets. The full twenty-question comparison was not completed by the requested 09:00 cutoff. At the user-directed closeout, both pending connections were cancelled, interruption controls were disabled and both accounts showed no active sessions. An interrupted export remains evidence of partial work. Earlier tau artifacts are separate and do not validate the current study.

## Pre-benchmark engineering checkpoint

At the November 2, approximately 01:40 Central checkpoint, both new Yash Drive copies had connected to Tesla T4 GPUs and their approved Drives. Three startup attempts per account had exited during the invented Paris qualification; no training, development, or test questions had been collected and no new study head had been fitted.

The first attempt exposed an HTTP route-binding error, corrected by binding the actual FastAPI `Request` class before route registration and verifying a real streaming HTTP request on CPU. The next two attempts exposed invalid writer messages: descriptive citation keys, then an extra trailing quote and numeric citation titles despite explicit examples. The strict parser rejected these messages. No citation was guessed or repaired into a scored answer. Failed engineering attempts and their raw-response/archive locators are retained separately from benchmark denominators and costs.

At that checkpoint, the new schema-constrained implementation was awaiting qualification before benchmark collection. It applies identically across every comparison arm, preserves existing token/context/call limits, and retains strict final validation. Each request gets a fresh parser and token enforcer; only tokenizer preprocessing is cached. Citation objects use the existing accepted `title`/`sentence_index` contract. Required fields and JSON types are constrained during decoding; negative sentence indices, incomplete output, duplicate keys and malformed raw JSON remain rejected after generation, with physical receipts/correlation retained before an HTTP failure. A formatting constraint establishes message structure; it does not establish answer correctness or valid supporting evidence. This historical checkpoint records startup failures; later raw evidence above separately verifies successful qualification and partial benchmark work.

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
