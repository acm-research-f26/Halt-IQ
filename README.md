# HaltIQ: Jev as a critic

A working Python prototype for Yash and Vedanshi's Monday discussion: **can a typed Jev critic stop a writer's revision loop earlier without accepting more wrong answers?**

Yash's implementation lives on the [`Yash` branch of acm-research-f26/Halt-IQ](https://github.com/acm-research-f26/Halt-IQ/tree/Yash).

The writer runs locally through Ollama. You can swap a conventional language-model critic for official Jev, a compatible local OpenJev server, or an explicitly labeled local typed baseline. A separate evaluator scores short answers against reference labels. The research environment is now **HotpotQA distractor**, with complete supplied Wikipedia evidence and official answer EM/F1 conventions. There is no model training or separate CUDA Python dependency.

**Environment and setup:** [HotpotQA research environment](Docs/ENVIRONMENT.md).

**Monday explanation:** [Walkthrough and measured results](Docs/MONDAY.md).

**Start here:** `python3 -m haltiq demo` works offline immediately. For an actual HotpotQA run, use `python3 -m haltiq run --environment hotpotqa --limit 4`. The 100-question subset is already prepared. Python 3.10+ is the only Python requirement; no pip install is needed when running from this folder. Windows uses `py -3` in place of `python3`.

## Benchmark environment

[HotpotQA distractor](https://hotpotqa.github.io/) provides multi-hop questions and a small set of Wikipedia passages containing relevant evidence and distractors. This is a supplied-context QA environment: the writer revises answers using the same evidence throughout an episode. It does not perform live web search or train an RL policy.

The prepared [100-question subset](data/hotpotqa/README.md) is included in this branch and has **20 development and 80 local held-out questions**, sampled proportionally across bridge and comparison questions. The full 7,405-question source validation release is downloaded and cached only when regenerating the subset. Both subsets come from public official development data; `--split test` is our local held-out validation, not HotpotQA's hidden official test set. All supplied passages are preserved; gold supporting facts never select or label evidence in prompts.

`run` defaults to HotpotQA. `run --environment diagnostic` selects the earlier 24-item synthetic dataset. `demo` stays entirely scripted. The [environment guide](Docs/ENVIRONMENT.md) explains installation, dataset provenance, regeneration, and research limitations.

## What is implemented

```text
Question + evidence → local writer → candidate answer
                                         ↓
                                    chosen critic
                              approve ↙         ↘ reject
                            return answer     repair feedback
                                                 ↓
                                            local writer
```

The default cap is three drafts (the initial answer and at most two revisions). At the cap, return the current answer even if the critic still rejects it. Errors stop that episode and remain visible in the report. Every arm receives the exact same initial draft for each task; later drafts can differ because of feedback.

| Arm | Critic | Revision feedback | Stops |
| --- | --- | --- | --- |
| `fixed` | None | Generic self-check | At round cap |
| `llm` | Local Qwen, ordinary critique | Generated explanation | Approval or cap |
| `jev` | Official TypeSafe Jev | Templates for failed typed checks | All checks pass or cap |
| `local-typed` | Local Qwen, JSON scores | Same templates as Jev | All checks pass or cap |
| `openjev` | User-supplied compatible local server | Same templates as Jev | All checks pass or cap |

Jev evaluates **supported**, **complete**, and **relevant** in one request with three `noul` questions. Approval requires every probability to be at least `0.8`; a high relevance score cannot hide low factual support. These probabilities are not multiplied into a purported joint confidence. The threshold is an initial development choice, not a validated optimum.

Jev does not write critique prose. The adapter translates a failed support check into an instruction such as “Recheck the facts and reasoning against the evidence.” This means the experiment compares full feedback-and-stopping systems. It does not isolate only the stopping rule.

`local-typed` uses a generative LLM's self-reported scores. It is **neither Jev nor OpenJev**, and its scores are uncalibrated. It provides a free ablation for exercising the typed critic design.

## Run on your Mac

Install [Python 3.10+](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads), and [Ollama](https://ollama.com/download), then open Ollama. In Terminal:

```bash
git clone --branch Yash https://github.com/acm-research-f26/Halt-IQ.git
cd Halt-IQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ollama pull qwen3:8b
python -m haltiq doctor
python -m haltiq demo
python -m unittest discover -s tests -v
python -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed --split dev --limit 4
```

Ollama and `qwen3:8b` were already installed on this Mac. The observed model is Q4_K_M, approximately 5.2 GB on disk. The manifest records the full local model digest for each run. Default generation is temperature 0, seed 42, thinking disabled, 32,768-token context, and a 512-token output cap. Use `--context-tokens` to change the window. A conservative request-size guard rejects oversized input instead of shortening the evidence. Seeds do not guarantee identical results across hardware or Ollama versions.

For an existing checkout, skip the clone and change-directory commands. The optional `.[data]` extra is only needed to re-download and convert the full benchmark. Ollama supports Apple GPU acceleration through Metal; there is no separate PyTorch/MPS setup in this project. Your 48 GB Mac has ample capacity for this selected model, although runtime and context memory still need to be measured.

## Run on your Windows PC

Install current Python, Git, and Ollama, use an up-to-date NVIDIA driver, and open Ollama. Create a fresh Windows virtual environment; do not copy the Mac `.venv`. In PowerShell:

```powershell
git clone --branch Yash https://github.com/acm-research-f26/Halt-IQ.git
cd Halt-IQ
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
ollama pull qwen3:8b
.\.venv\Scripts\python.exe -m haltiq doctor
.\.venv\Scripts\python.exe -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed --split dev --limit 4
```

Your RTX 5070 Ti is listed in [Ollama's supported GPU hardware](https://docs.ollama.com/gpu). The 8B quantized default is a conservative starting point for the PC and Mac. Writer and conventional critic share the same model; you do not need two copies loaded. Check `ollama ps` during a run to verify GPU use. This PC configuration is documented but has not been executed here.

To disable Ollama cloud features, set `OLLAMA_NO_CLOUD=1` **on the Ollama server process** and restart it. A terminal variable does not reconfigure an already-running app. The client additionally restricts endpoints to loopback and checks that the selected model has local weights and is not marked as a remote alias. See [Ollama's local-only configuration](https://docs.ollama.com/faq).

## Enable real Jev

Obtain your own key through [TypeSafe](https://typesafe.ai/). The program reads `TYPESAFE_API_KEY`; it does not read `.env` files. Keep keys out of source, screenshots, result files, and messages.

On macOS zsh, enter the key without putting its literal value into shell history:

```zsh
read -s 'TYPESAFE_API_KEY?TypeSafe API key: '
export TYPESAFE_API_KEY
python3 -m haltiq run --critics fixed,llm,jev --split dev --limit 4 --budget-usd 5
```

On PowerShell 7, use `Read-Host -MaskInput`; on older PowerShell, the secure-string version below works:

```powershell
$jevCredential = Get-Credential -UserName "typesafe" -Message "Put the API key in the password field"
$env:TYPESAFE_API_KEY = $jevCredential.GetNetworkCredential().Password
py -3 -m haltiq run --critics fixed,llm,jev --split dev --limit 4 --budget-usd 5
```

The adapter calls `POST https://api.typesafe.ai/v1/systemone` with the pinned model `jev-1.13.0`. It validates all returned probabilities, usage fields, and model identity. There are no automatic retries or silent substitutions. The official endpoint has not been called in the saved local pilot; API access and live response compatibility remain to be checked with your key.

Once prompts and threshold are frozen on development cases:

```bash
python3 -m haltiq run --critics fixed,llm,jev --split test --budget-usd 5
```

This uses 80 locally held-out HotpotQA cases. The 16 held-out synthetic cases are still available with `--environment diagnostic`. A custom dataset can replace either via `--dataset path/to/tasks.jsonl`; its schema is in [data/README.md](data/README.md).

## Budget

The default local workflow makes no hosted inference calls. Official Jev's documented price checked September 27, 2026 is **$0.042 per million input tokens**, with free output tokens and a 64k total input context. See [TypeSafe's model pricing](https://docs.typesafe.ai/models).

The persistent SQLite ledger at `.haltiq/budget.sqlite` defaults to a cumulative **$5** allowance. `--budget-usd` changes that allowance, not a new allowance for each run. Values above $30 are rejected. Before each hosted call it reserves 65,536 tokens × the published rate, about **$0.002753**, then reconciles a validated response to actual usage. Timeouts and malformed responses retain their reservation. Concurrent runs on the same ledger cannot race past its allowance.

At that price, 100 tasks × three rounds means at most **300 Jev requests** (each request contains three typed questions), with a conservative context-based reservation total of **$0.826**. Actual small-prompt usage should be lower; no paid pilot has established the observed cost yet.

The guard covers this harness at the verified model price. It is not an account-wide billing limit and does not include credit purchases, taxes, changed provider prices, other applications, or electricity. Disable automatic credit refill in your account if appropriate. **Do not delete or replace the ledger to restart spending.** When moving between PC and Mac, use the same ledger sequentially or allocate separate allowances whose sum stays below $30; independent ledgers cannot coordinate a global cap.

## OpenJev option

Several unrelated projects use the name OpenJev. They are independent implementations, not official Jev weights. A local compatible server can be used like this:

```bash
python3 -m haltiq run --critics fixed,llm,openjev --openjev-url http://localhost:8080 --openjev-model openjev-latest --split dev --limit 4
```

The API root must be loopback; the client adds `/v1/systemone`. Supply the model identifier actually supported by your server. No TypeSafe environment key is forwarded to the local server. No OpenJev model/server was installed or tested in this implementation.

For example, [razorback16/openjev](https://github.com/razorback16/openjev) documents a DiffusionGemma backend requiring at least 24 GB NVIDIA VRAM, or an MLX path using roughly 16 GB on a Mac. That makes it a poor default for the 5070 Ti. Your Mac could be a candidate for that MLX path, but validate memory and server compatibility separately. The lightweight hosted Jev adapter plus local writer is the practical first experiment.

## Results and collaboration

Every run creates a new folder under `results/runs/`, or the new directory supplied to `--output`. Existing output folders are refused to prevent accidental result replacement.

- `report.md`: human-readable comparison.
- `episodes.csv`: one row per task and arm, suitable for Vedanshi's comparisons.
- `summary.json`: aggregates and actual versus per-arm operational usage.
- `traces.jsonl`: drafts, raw model outputs, decisions, and errors, flushed as work completes.
- `manifest.json`: task IDs, dataset/code hashes, settings, models, platform, budget state, and completion status.
- `dataset_manifest.json`, `dataset_README.md`, and `dataset_CITATION.bib`: benchmark preparation, source license, and attribution when available.
- `source_snapshot/` and `dataset_snapshot.jsonl`: the exact Python source and dataset present when the run started (included in final verification and subsequent runs).

The reference labels are never passed to writers or critics. Success is normalized exact match; token F1 is a secondary diagnostic. Critic approval and actual success are different fields. The report keeps false approvals and operational failures visible. Semantic equivalence beyond the listed aliases can be missed by exact match, so inspect disputed answers before making research claims.

The [HotpotQA development pilot](results/hotpotqa-dev-pilot/report.md) uses actual benchmark questions and local Qwen calls. The earlier [synthetic pilot](results/local-verified/report.md) remains a separate historical run; do not pool its results with HotpotQA. The separate [offline demo](results/demo/report.md) uses scripted drafts and scores. Neither is evidence about official Jev's performance.

To use the critic in Vedanshi's existing harness, review [Docs/INTEGRATION.md](Docs/INTEGRATION.md). The single-state CLI is also available:

```bash
python3 -m haltiq judge --backend llm --input data/example_state.json
python3 -m haltiq judge --backend jev --input data/example_state.json --budget-usd 5
```

See [Docs/EXPERIMENT.md](Docs/EXPERIMENT.md) for the research protocol. The older files in `Docs/` supplied project context; this is an independent prototype, not a reproduction of their RL training proposal or the upstream Semantic Halting Problem implementation.

## Source references

Verified September 27, 2026: [TypeSafe HTTP API](https://docs.typesafe.ai/api), [model/pricing reference](https://docs.typesafe.ai/models), [Ollama chat API](https://docs.ollama.com/api/chat), [Ollama hardware](https://docs.ollama.com/gpu), and the [Semantic Halting Problem repository](https://github.com/SahilShrivastava-Dev/semantic-halting-problem) for the related writer–critic stopping setup. No upstream implementation was copied into this prototype.
