# Yash's Strands + linear-classifier simulations

This is Yash's new setup. Existing files under `strands/` and `data/` are
teammate reference inputs and are unchanged. No earlier Yash model, run or result
is used. New notebooks are named `Yash_HotpotQA_Strands_Linear.ipynb` and
`Yash_Tau2_Strands_Linear.ipynb`; create new Drive copies rather than editing a
teammate's notebook.

The two writer/reviewer agents use the **AWS Strands Agents SDK** and a local
Qwen2.5-1.5B model. Vedanshi's **Strands Decider** (Hobson v19 with BGE top-two
evidence) supplies approval. Shree's pretrained `Linear(5,1)` head supplies a
stop probability. The paired policy requires both stop >= 0.5 and approval >=
0.8. The two meanings of Strands are separate components.

## Colab

Open a new notebook in Colab, save a copy under Yash's name, select a GPU and
run its cells in order. The bootstrap pins the source before installing. Saved
Yash copies are [HotpotQA](https://colab.research.google.com/drive/161BXIOvt_kKGgmOPC0Uv5dXQLaWS9Mpb)
and [Tau2](https://colab.research.google.com/drive/1vjkuOGxrOinXnnXMzCZEeX-RxTzCcEQv?authuser=1).
The provisioned Google account supplies compute, not an LLM API key. No paid API
account or AWS credential is needed.

Drive saves the result journals when its mount succeeds. Set `USE_DRIVE=False`
to use runtime storage; a failed mount also falls back to
`/content/Yash_HaltIQ_Results`. `storage.json` records whether results are
persistent. Download the evidence ZIP before disconnecting a temporary runtime.
The ZIP retains results, logs and serving receipts while excluding model caches.

HotpotQA defaults to 20 fresh validation questions across five independent arms
(Fixed-1, Fixed-6, linear, Strands, combined). It excludes all 170 question IDs in
the current teammate reference inputs. Reference answers and supporting-fact
labels never enter writer, reviewer or gate inputs. All twelve official
answer/support/joint metrics are saved; failures count as zero. Each continuing
adaptive round has a real reviewer call before its gate, included in usage.

Tau defaults to ten pinned official English retail tasks across four arms
(first proposal, linear, Strands, combined). Tau alone executes approved
tool calls. Rejected proposals route to review/revision; a rejection at the cap
fails the episode. Its local user/judges differ from leaderboard conditions, so
these are official-domain local simulations, not leaderboard-comparable scores.
Malformed model output fails the attempt without automatic retries. The frozen
Tau summary's `technical_failures` field is a broad failed/error bucket; inspect
route events to distinguish malformed output, gate rejection at the revision
cap, and simulator or evaluator errors.

The writer processes prefill in 512-token chunks while retaining the complete
16,384-token context and KV cache. This changes computation, not the supplied
conversation; context overflow fails explicitly without truncation. Writer
receipts record the attention backend, chunk size and library versions.

Hotpot citation pairs and exact two-field objects using `sentence_index` with
either `title` or `exact_passage_title` share a canonical pair representation.
Titles and integer indices are preserved. Extra fields, ambiguous objects and
invalid indices fail validation. The manifest freezes this format policy;
reference answers and supporting labels are not used for conversion.

`plan.json`/`manifest.json` freeze source, models, questions/tasks and policies.
Incremental journals retain failures and actual token receipts. Runtime-owned
processes stop on normal completion or an exception. Colab's runtime allocation
and lifetime are controlled by Google; save/export artifacts before releasing it.

## What the existing evidence says

The teammate-only 80-question replay has been recomputed with the actual
pretrained heads. The head requests stopping immediately, so the combined AND
rule equals Strands-only on these trajectories. Pair F1 is 0.37843 versus
Fixed-6 0.39205; saved LLM-prefix tokens fall 43.52%, excluding all Strands and
reranker compute. This is replay evidence, not new live inference or proof of
classifier benefit. All five saved seeds reuse the same 80 questions.

The original writer used Apple MLX 4-bit inference. Colab uses Hugging Face
inference, and a T4 uses FP16 Hobson serving because it lacks native BF16.
Runtime provenance records these numerical changes; historical probability
identity is not claimed. Neither model is trained anew in this setup.

## Local verification

```powershell
python -B -m unittest discover -s tests_pair -v
python -B -m haltiq_pair replay --output outputs/teammate-replay
```

GPU model inference runs through the two new Colab notebooks. The optional Tau
integration tests require the exact editable upstream checkout. Local synthetic
tests verify wiring and isolation, not benchmark accuracy. Each run creates a
plain-language `Research Writeup.txt` from its saved receipts.

Sources: teammate integration `6c72991652400dcfec8f464a331efb95d5d1cf8b`,
Shree `5503cabeb2ec0094413b7d491ccb08ec9c991c61`, and Tau
`cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2`. All reference files remain
byte-identical to the integration branch.
