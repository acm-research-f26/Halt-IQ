# Monday walkthrough: Jev as the critic on HotpotQA

## Environment and research question

The research environment is now **HotpotQA distractor**, an established multi-hop question-answering benchmark. A local Qwen model receives a question and all its supplied Wikipedia passages, drafts a short answer, and revises after critic feedback. Evidence stays fixed during an episode. The study asks whether replacing the conventional critic with Jev reduces unnecessary rounds while maintaining answer quality.

The complete 7,405-question validation release was downloaded from a pinned, checksum-verified Hugging Face mirror. A reproducible subset contains 20 development cases and 80 locally held-out cases. Both derive from the public official development split; the held-out subset is not the official hidden test set. No gold supporting-fact labels are used to select or highlight evidence. See [the environment guide](ENVIRONMENT.md) and [data manifest](../data/hotpotqa/manifest.json).

Here, “critic” means an answer checker, not a learned RL value network. There is no training or GRPO. The Jev adapter asks three yes/no questions in one API request: is the answer supported, complete, and relevant? All three probabilities must reach 0.8 to stop. Failed dimensions become deterministic repair instructions. At most three drafts are produced. The main comparison evaluates complete feedback-and-stopping systems; later drafts differ across critics.

## What actually ran

The [HotpotQA development pilot](../results/hotpotqa-dev-pilot/report.md) ran on eight real benchmark questions using local Qwen3 8B Q4_K_M, seed 42, temperature 0, thinking disabled, and a 32,768-token context. All 24 episodes completed without provider failures. Ollama reported 100% GPU execution and approximately 10 GB loaded model/context memory on the Mac.

| System | Answer EM | Answer F1 | Mean drafts | Critic requests | EM-based false approvals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed three drafts, self-revision | 4/8 (50%) | 57.5% | 3.00 | 0 | Not applicable |
| Conventional local LLM critic | 5/8 (62.5%) | 70.0% | 1.375 | 11 | 3/7 approvals |
| Local typed critic | 4/8 (50%) | 57.5% | 1.25 | 10 | 4/7 approvals |

The run made 50 model requests, with 85,207 reported input tokens and 1,351 output tokens. **Hosted API spend was $0.** Local hardware and electricity are not priced. This eight-question development pilot verifies the real benchmark pipeline; it is too small to establish significance or Jev superiority.

The answer scorer matches the official HotpotQA answer EM/F1 conventions. A separate parity check compared it with the official evaluator on 529 answer pairs, including punctuation and yes/no/noanswer edge cases; every result matched. The [parity record](../results/hotpotqa-metric-parity.json) includes the reference file hash. Supporting-fact and joint leaderboard scores are not implemented because the writer currently outputs answers only.

All 63 automated tests pass, including dataset preparation, split isolation, answer metric behavior, spending enforcement, HTTP transport, and conservative input-size checks. The earlier [synthetic diagnostic pilot](../results/local-verified/report.md) remains a separate historical result and must not be pooled with HotpotQA.

**Official Jev remains unrun.** No TypeSafe key is configured. The adapter is implemented and contract-tested, but authentication, availability, and live Jev behavior still need to be checked. The local typed critic is a generative Qwen baseline with uncalibrated scores; it is neither Jev nor OpenJev. No OpenJev server has been installed.

## A failure case worth discussing

One question asks the founding year of a joint venture involving RLJ Companies and a film studio founded in 2005. The reference answer is **2006**. The shared writer draft answers **2005**. The local typed critic approves it, and the fixed-round arm finishes incorrectly as well. The conventional critic prompts a correction and finishes correctly after two drafts. Inspect source ID `5ac240ef55429951e9e684ef` in the saved traces.

Another response uses a longer, factually related phrase about Emilie du Chatelet than the gold answer and earns token F1 of 0.6 but exact match of 0. This illustrates a metric limitation: an “EM-based false approval” is an approval of a non-matching answer, not always proof of a factual mistake. Keep both EM and F1, and manually inspect disagreements without silently changing the gold answers.

## Show it working

From the project folder on the Mac:

```bash
source .venv/bin/activate
python -m haltiq doctor
python -m haltiq run --environment hotpotqa --critics fixed,llm,local-typed --split dev --limit 4
```

The dataset is already prepared. These commands call local models and incur no hosted inference charge. The original scripted demonstration remains `python -m haltiq demo`. Use the [environment guide](ENVIRONMENT.md) for Windows setup; create a new Windows virtual environment rather than copying the Mac `.venv`.

After setting `TYPESAFE_API_KEY` in the environment:

```bash
python -m haltiq run --environment hotpotqa --critics fixed,llm,jev --split dev --limit 4 --budget-usd 5
```

The ledger reserves a conservative charge before every Jev request and retains spending across runs. Its default allowance is $5, and values above $30 are rejected. Preserve the ledger or allocate budgets between machines whose sum is below the overall cap. This guard does not control credit purchases or other applications.

Freeze prompts, threshold, model settings, and round cap using only development cases, then run the 80 locally held-out questions with `--split test`. Those 80 questions have not been used for model calls in the saved pilot. This is still a modest public-development-derived subset; later studies should broaden data and quantify variability.

## Partnership handoff

Yash owns the typed critic interface, checks, error handling, local environment, and saved records. Vedanshi can run the same question IDs through the baseline and Jev paths or insert the adapter into her existing harness using [the integration note](INTEGRATION.md). Compare independent answer scores, rounds, critic calls, wrong approvals, latency, and cost.

The defensible progress statement is: **the critic comparison now works on genuine HotpotQA data, with reproducible subsets and benchmark answer scoring; the next experiment measures official Jev using the prepared adapter.**
