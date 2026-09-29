# The HaltIQ research environment

HaltIQ now uses **HotpotQA in the distractor setting** for real model experiments. The earlier environment was a hand-written diagnostic dataset: useful for checking the revision loop, but too small and artificial for evaluating Jev. `python -m haltiq run` now selects HotpotQA by default; `python -m haltiq demo` remains the scripted diagnostic demonstration.

HotpotQA contains questions requiring information from multiple Wikipedia paragraphs, including bridge questions and comparisons. In the distractor setting, the model receives a small collection of paragraphs containing relevant information and distractors. That makes it a practical first environment for testing whether a critic identifies unsupported answers and stops revisions appropriately. The dataset authors describe the task and its evaluation on the [HotpotQA project page](https://hotpotqa.github.io/) and in the [EMNLP 2018 paper](https://aclanthology.org/D18-1259/).

The implemented environment is **static question answering with supplied evidence**. A writer reads a question and its paragraphs, produces a short answer, and revises it after critic feedback. The evidence remains fixed during that episode. It does not perform Wikipedia search, use a Gym simulator, or learn a reinforcement-learning policy. The research intervention is the choice of critic and the resulting stopping/revision behavior.

| Part of the experiment | Implementation |
| --- | --- |
| Input | A real HotpotQA question and every supplied context paragraph |
| Writer | Local `qwen3:8b` through Ollama |
| Baselines | Fixed rounds, conventional local LLM critic, local typed-score critic |
| Proposed critic | Official Jev through the existing TypeSafe adapter |
| Decision | Approve and stop, or supply revision feedback |
| Episode cap | Three drafts by default |
| Success evaluation | Reference-answer exact match and token F1 |
| Main comparison | Success, rounds, critic calls, false approvals, tokens, latency, and hosted cost |

All arms share a question's initial draft. Their later answers can differ because feedback differs. The experiment therefore compares complete critic-and-revision systems; it does not isolate the stopping threshold alone. `local-typed` is a generative model producing uncalibrated scores, not Jev or OpenJev.

The [OpenJev guide](OPENJEV.md) describes the installed community server, its separate environment, and the eight-question live comparison. The [code walkthrough](CODE_WALKTHROUGH.md) explains the regular writer and critic path.

**The prepared dataset is ready to run.** [tasks.jsonl](../data/hotpotqa/tasks.jsonl) contains 100 genuine benchmark questions: 20 for development and 80 for local held-out evaluation. Both subsets derive from the **official public development split**, not the hidden official test set. The split is proportional by question type: 16 bridge / 4 comparison development cases and 64 bridge / 16 comparison held-out cases.

The source contains 7,405 cases. Seed 42, stable source IDs, and proportional allocation determine the subset; answers and model performance do not influence selection. IDs and normalized questions cannot cross the two local splits. The [data manifest](../data/hotpotqa/manifest.json) records the seed, selected IDs, type counts, duplicate handling, source hashes, and output hash. No duplicate normalized questions were found in this source.

Every provided paragraph and sentence is preserved in original order, with titles and zero-based sentence numbers. Most source cases contain ten paragraphs; 60 contain fewer. The selected development subset contains one two-paragraph case and nineteen ten-paragraph cases; all 80 held-out cases contain ten. These differences come from the upstream data. Nothing is filtered using the gold supporting facts, and evidence is not shortened to fit a prompt.

Only the question, evidence, and current answer reach the critic. Gold answers, supporting-fact annotations, and annotation warnings remain outside model prompts. The full source has 49 blank or whitespace sentence strings, which remain preserved, and one out-of-range supporting-fact annotation at source ID `5ae61bfd5542992663a4f261`. The manifest records that annotation issue; it does not remove the case from the sampling pool. Neither selected subset contains that issue.

**Run it on the Mac.** The local `.venv` has Python 3.14.3 and the optional data-preparation dependency `pyarrow==25.0.1`. From the project folder:

```bash
source .venv/bin/activate
python -m haltiq doctor
python -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed --split dev --limit 4
```

Ollama must be running with `qwen3:8b` available. On a fresh Mac, install Python and [Ollama](https://ollama.com/download), then create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ollama pull qwen3:8b
python -m haltiq doctor
```

The bundled JSONL dataset and model runs do not require PyArrow. The `data` extra is needed only to download and convert the pinned Parquet source. The package supports Python 3.10 or newer.

**Run it on the Windows PC.** Copy the project and bundled `data/hotpotqa` folder, then create a new virtual environment on Windows; do not copy the Mac `.venv`. Install Python, Ollama, and an appropriate NVIDIA driver. In PowerShell, from the copied project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
ollama pull qwen3:8b
.\.venv\Scripts\python.exe -m haltiq doctor
.\.venv\Scripts\python.exe -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed --split dev --limit 4
```

This setup uses Ollama for model execution rather than installing a separate CUDA or PyTorch environment. The Windows commands are documented but have not been executed on your PC. For the remaining examples, replace `python` with `.\.venv\Scripts\python.exe` on Windows if the environment is not activated.

The default Ollama context window is **32,768 tokens**, exposed as `--context-tokens 32768`. A conservative byte-based preflight guard rejects prompts that might exceed the configured space, including room for output, rather than silently dropping evidence. It is a safeguard rather than an exact tokenizer measurement. If a request fails the guard, increase context within the supported limits and available memory; retain the complete evidence. Longer contexts use more memory, so measure runtime and memory on both machines before increasing the model size.

**Rebuild or enlarge the dataset when needed.** The supplied 100-task file is sufficient to start. To reproduce it in a new directory:

```bash
python -m pip install -e ".[data]"
python -m haltiq prepare-hotpot --output data/hotpotqa-new --dev-size 20 --test-size 80 --seed 42
```

Preparation refuses existing output directories. The default downloader uses the [HotpotQA Hugging Face mirror pinned to revision `1908d6afbbead072334abe2965f91bd2709910ab`](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/distractor), verifies the Parquet size and SHA-256, and converts its validation rows back to the original HotpotQA JSON structure. The download is approximately 27.5 MB; the converted JSON cache is approximately 44 MB. Both live under `.haltiq/datasets`. The prepared dataset has its own attribution, citation, and manifest.

An existing original-format JSON file can also be supplied without PyArrow:

```bash
python -m haltiq prepare-hotpot --source .haltiq/datasets/hotpot_dev_distractor_v1.json --output data/hotpotqa-from-json --dev-size 20 --test-size 80 --seed 42
```

To run an alternate prepared subset, supply its file explicitly:

```bash
python -m haltiq run --dataset data/hotpotqa-new/tasks.jsonl --critics fixed,llm,local-typed --split dev --limit 4
```

**Use Jev after the free development pilot.** Set `TYPESAFE_API_KEY` in the process environment. The program does not load `.env` files. On macOS zsh, this prompts without putting the literal key in shell history:

```zsh
read -s 'TYPESAFE_API_KEY?TypeSafe API key: '
export TYPESAFE_API_KEY
python -m haltiq run --environment hotpotqa --critics fixed,llm,jev --split dev --limit 4 --budget-usd 5
```

On PowerShell:

```powershell
$jevCredential = Get-Credential -UserName "typesafe" -Message "Put the API key in the password field"
$env:TYPESAFE_API_KEY = $jevCredential.GetNetworkCredential().Password
.\.venv\Scripts\python.exe -m haltiq run --environment hotpotqa --critics fixed,llm,jev --split dev --limit 4 --budget-usd 5
```

The local arms have no hosted inference charge. Official Jev uses the persistent spending ledger and defaults to a cumulative $5 allowance; the program rejects allowances above $30. Keep the same ledger across repeated runs. Separate machines' ledgers do not coordinate a shared cap, so transfer the ledger sequentially or divide the total allowance between machines. Live Jev compatibility and results require a configured key; a local typed baseline does not establish Jev's performance.

Develop prompts and choose the threshold using `dev`. Freeze the model, prompts, critic checks, threshold, round cap, and seed before evaluating the local held-out split:

```bash
python -m haltiq run --environment hotpotqa --critics fixed,llm,jev --split test --budget-usd 5
```

The experiment scores **answer exact match and token F1 using HotpotQA's answer conventions**, including its special handling of yes/no/noanswer responses. It does not request or score predicted supporting facts, so it does not produce the supporting-fact or joint scores in the official leaderboard. Critic approval is recorded separately from reference-answer correctness; an approved wrong answer counts as a false approval. Consult the [official evaluation implementation](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py) for the benchmark conventions.

The latest local benchmark results cover all 20 development questions: [run report](../results/hotpotqa-dev-20-20260928/report.md) and [checked comparison including one draft](../results/hotpotqa-dev-20-20260928/analysis.md). All 60 episodes completed. The [initial eight-question pilot](../results/hotpotqa-dev-pilot/report.md) overlaps this run; do not pool them as independent observations. The previous [diagnostic pilot](../results/local-verified/report.md) measures a different dataset and should not be pooled with HotpotQA. Reproduce that older environment with `--environment diagnostic`; `demo` uses scripted outputs and reports no actual model result.

A 100-question development-derived subset is a pilot, not an official benchmark ranking. A four-question run verifies the pipeline and gives very weak evidence about accuracy. These public questions may already appear in model training data; local separation of tuning and evaluation does not guarantee freedom from pretraining contamination. Report the source split, subset size, seed, hardware, model digest, and metric scope alongside findings.

**Possible next environments.** [MuSiQue](https://github.com/StonyBrookNLP/musique) is a useful later check because its questions are composed from connected single-hop questions; its answerable and fuller variants can broaden the reasoning and evidence-sufficiency evaluation. It requires a separate importer and its own evaluation rules. An interactive Wikipedia agent following [ReAct](https://react-lm.github.io/) would instead let the model search and inspect evidence before answering; that extends the research question to deciding when to stop information gathering. Neither extension is installed in this implementation. The static HotpotQA setup gives Yash and Vedanshi a shared, reproducible critic comparison first.

HotpotQA and this derived dataset are distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Cite Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, and Manning, *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018. A ready-to-use [BibTeX entry](../data/hotpotqa/CITATION.bib) accompanies the prepared data. The transformations are sampling, local splitting, annotation of source issues, and formatting for HaltIQ; source text remains preserved.
