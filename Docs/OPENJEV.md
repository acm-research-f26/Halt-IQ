# Local OpenJev setup

This setup uses [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev), a community implementation, with Gemma 3 4B. On this Mac, the model runs through MLX using a pinned 4-bit checkpoint. It requires no TypeSafe API key or hosted inference credits. Official Jev is a separate model and service.

The upstream scorer compares the likelihoods of `yes` and `no` for each typed question. It shares the prefix across those options, then normalizes their log probabilities. Our three checks still require three upstream scoring passes within one HTTP request. The resulting scores are uncalibrated; using the same API shape does not make them equivalent to Jev probabilities.

## What is installed

| Component | Location or version |
| --- | --- |
| Upstream code | `daseinlabs/open-jev`, commit `6cb37c3cee9d8477d49d8ac0c99056e48ccc292c` |
| Mac checkpoint | `mlx-community/gemma-3-4b-it-4bit`, revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |
| Local model ID | `openjev-dasein-gemma3-4b-4bit` |
| Separate environment | `.haltiq/openjev-venv` |
| Model files | `.haltiq/models/gemma-3-4b-it-4bit` |
| Runtime manifest | `.haltiq/openjev-runtime.json` |
| HTTP server | `http://127.0.0.1:8080` |

The checkpoint download is approximately 3.4 GB plus tokenizer files. The model and virtual environment are ignored by Git. Their source revisions, installed dependency versions, and checkpoint hashes are recorded so a run can identify what produced its scores. The model files remain under the Gemma license; upstream OpenJev code uses the MIT license. See the [checkpoint card](https://huggingface.co/mlx-community/gemma-3-4b-it-4bit) and [upstream license](https://github.com/daseinlabs/open-jev/blob/6cb37c3cee9d8477d49d8ac0c99056e48ccc292c/LICENSE).

## Start on the Mac

From the project folder, use Python 3.12 or newer. The verified setup used Python 3.14 on this Apple silicon Mac.

```bash
# First setup or reinstall; the model download is cached.
python3 scripts/setup_openjev.py

# Run the server in this terminal. Ctrl+C stops it.
.haltiq/openjev-venv/bin/python scripts/serve_openjev.py
```

Setup installs OpenJev separately from the normal HaltIQ environment. [configs/openjev.json](../configs/openjev.json) pins the code and model revisions; [the Mac constraints](../configs/openjev-mac-constraints.txt) record the tested dependencies. Startup checks model hashes and package versions before loading the scorer.

In another terminal, with Ollama running:

```bash
curl http://127.0.0.1:8080/health
.venv/bin/python -m haltiq judge --backend openjev --input data/example_state.json
.venv/bin/python -m haltiq run --critics fixed,llm,openjev --split dev --limit 8
```

The generic `openjev-latest` request alias resolves to the installed Gemma checkpoint. The response records its actual model ID. Requests for an unrelated model are rejected. The server accepts at most three Noul checks and limits each rendered question to 32,768 tokens including the answer label. It rejects oversized requests without shortening their evidence. It binds only to loopback and makes no hosted inference calls.

For a different port, start with `--port 8081` and add `--openjev-url http://127.0.0.1:8081` to HaltIQ commands.

## Keeping memory use down

Start one OpenJev server only when running an experiment. It uses one worker and processes model requests one at a time. Avoid running the upstream numerical check while that server is active: the check loads a second copy of the model. The 3.4 GB checkpoint size is not a RAM limit; inference also needs memory for activations and context caches.

When finished, stop the OpenJev terminal with Ctrl+C and unload the writer:

```bash
ollama stop qwen3:8b
ollama ps
```

An empty `ollama ps` list means Ollama has no model loaded. The downloaded files stay on disk for the next run. Reading reports and running the unit tests do not load model weights, so those tasks do not need either model server. Neither server is configured to start automatically by this setup.

## Validation

The existing 63 HaltIQ tests and five new server tests passed. The latter check model identity, input limits, supported question types, and serialization of concurrent requests. Run them in their respective environments:

```bash
.venv/bin/python -m unittest discover -s tests -v
.haltiq/openjev-venv/bin/python -m unittest discover -s tests_openjev -v
```

The [live smoke-test records](../results/openjev-smoke-20260928/reviews.json) contain calls to the actual Gemma model. The incorrect answer received low scores and was rejected. The correct answer was also rejected because its completeness score was low. That is a limitation of this model/rubric combination at the 0.8 threshold; it is not a successful calibration result.

The [upstream numerical check](../results/openjev-smoke-20260928/upstream-numerical-check.txt) compared cached scoring with full re-encoding, including a 1,501-token prefix. It passed upstream's default 0.5 absolute log-probability tolerance, with a largest observed difference of about 0.375. This is a numerical check on a quantized model, not an accuracy or calibration benchmark.

## HotpotQA results

The [eight-question comparison](../results/hotpotqa-openjev-dev-8-20260928/report.md) used the same first draft for fixed revision, the regular Qwen critic, and OpenJev. All 24 episodes completed without provider failures. The threshold stayed at 0.8 for each typed check.

| Configuration | Answer exact match | Mean drafts | Critic requests |
| --- | ---: | ---: | ---: |
| Fixed revision | 4/8 | 3.00 | 0 |
| Regular LLM critic | 5/8 | 1.375 | 11 |
| OpenJev | 4/8 | 3.00 | 24 |

OpenJev made zero approvals and rejected 12 drafts that were correct by exact match. It reached the cap on every question, so zero false approvals is not evidence of a good stopping policy here. Its scores need investigation on development examples before using this configuration for early stopping. The model, rubric, and threshold all contribute; this run does not isolate which causes the rejection behavior.

The run made 78 physical model requests and incurred $0 hosted inference cost. OpenJev's reported output tokens count scored candidate-label tokens, not generated prose. Its input usage counts the state separately for each typed question. Different tokenizers and scoring methods mean token totals are not directly comparable units of computation.

The [checked records](../results/hotpotqa-openjev-dev-8-20260928/analysis.md), [OpenJev provenance](../results/hotpotqa-openjev-dev-8-20260928/openjev_provenance.json), and [server snapshot](../results/hotpotqa-openjev-dev-8-20260928/openjev_server_snapshot.py) identify the actual code and model used. All 24 OpenJev responses had matching provenance. These eight cases overlap previous development runs; they are not eight new independent test questions. The 80 held-out cases remain unused for inference.

## Windows PC

The upstream code has a PyTorch backend, but this Mac cannot verify Windows execution. MLX checkpoint files cannot be reused by PyTorch. First download original `google/gemma-3-4b-it` weights through Hugging Face, following its access/license process, into a directory such as `.haltiq/models/gemma-3-4b-it`. Then, with Python 3.12+ and Git installed:

```powershell
py -3 scripts/setup_openjev.py --model-path .haltiq/models/gemma-3-4b-it
.\.haltiq\openjev-venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.haltiq\openjev-venv\Scripts\python.exe scripts/serve_openjev.py --device cuda
```

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) if the installed build does not support your GPU. The setup records dependency versions, so run setup again after changing that environment before starting the server.

The original checkpoint needs roughly 8 GB for 16-bit weights before activations and caches. Running it alongside the 8B writer at the Mac's context setting may exceed the PC's available GPU memory. Measure memory first and use a smaller writer if necessary; record that change as a different experiment. A Windows run and a Mac run should not be pooled as if their hardware and quantization were identical.

The [code walkthrough](CODE_WALKTHROUGH.md) explains the normal writer and critic loop and where this server connects.
