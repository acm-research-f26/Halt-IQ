# How the regular code works

The main program runs a writer, lets a critic review its answer, and decides whether to ask the writer to try again. Everything here happens at inference time. There is no training or reinforcement learning.

## Start with the writer

`Writer.write()` in [haltiq/critics.py](../haltiq/critics.py) sends Qwen the question and the supplied evidence through Ollama. It asks for a short answer in JSON. Reference answers are kept out of this request.

For each question, [haltiq/runner.py](../haltiq/runner.py) generates the first answer once and gives that exact text to every configuration in the comparison. That keeps the starting answer constant.

## What the regular critic does

`LLMCritic.review()` in [haltiq/critics.py](../haltiq/critics.py) uses Qwen3 8B again, with a different prompt. It receives the question, evidence, and current draft and checks support, completeness, and relevance. It returns something like:

```json
{"approved": false, "feedback": "The question asks for the joint venture's founding year, not the film studio's founding year."}
```

This is an illustrative response format, not a quotation from the saved run. The regular critic writes its own feedback and approval decision. It does not produce three probability scores.

If the critic approves, the runner stops. If it rejects the answer, the writer receives the previous answer and feedback along with the original question and evidence. The default maximum is three drafts: the first answer plus up to two revisions. At the cap, the runner returns the current answer even if the critic still rejects it. A provider error is recorded as a failed episode.

## What changes between configurations

| Configuration | Review and feedback | When it stops |
| --- | --- | --- |
| `fixed` | Generic self-check instruction; no critic call | After three drafts |
| `llm` | Qwen generates an approval decision and feedback | Approval or draft cap |
| `local-typed` | Qwen generates three numeric scores; failed checks select repair templates | All scores reach 0.8, or draft cap |
| `openjev` | The local community OpenJev scores yes/no continuations; the same repair templates use its scores | All scores reach 0.8, or draft cap |
| `jev` | Official TypeSafe API supplies the three scores | All scores reach 0.8, or draft cap |

The local typed baseline and OpenJev use different scoring mechanisms. OpenJev reads the language model's likelihoods of the labels `yes` and `no`; it does not ask the model to write a number such as `0.9`. These normalized likelihoods still need calibration. They are not measurements from official Jev.

Feedback changes later drafts, so this experiment compares complete revision systems. It does not isolate stopping decisions on otherwise identical trajectories.

## Who checks whether the final answer is right?

[haltiq/tasks.py](../haltiq/tasks.py) keeps the reference answers for evaluation. [haltiq/metrics.py](../haltiq/metrics.py) calculates HotpotQA answer exact match and token F1. A critic can approve an answer that the evaluator scores as wrong, and those approvals remain visible in the results. A wording difference can fail exact match without necessarily being a factual error.

The runner saves each answer, critic response, stop reason, and token usage. It also writes the summary table, episode CSV, and copies of the source and dataset used for the run. [haltiq/providers.py](../haltiq/providers.py) handles the HTTP calls and response validation; [haltiq/budget.py](../haltiq/budget.py) limits spending for the official hosted Jev path.

## How OpenJev connects

Your normal HaltIQ environment sends a request to `http://127.0.0.1:8080/v1/systemone`. A separate environment loads Gemma and runs the upstream OpenJev scorer. The response comes back through the existing `JevClient` and `JevCritic`, so the writer, dataset, and evaluation code stay the same.

[scripts/serve_openjev.py](../scripts/serve_openjev.py) is a small wrapper around the upstream scorer. It binds to loopback, serializes requests to the shared model, verifies the requested model, and rejects inputs over the token limit before inference. It records the installed model and source information in each response. It does not change OpenJev's scoring prompts or train the model.

Use the [OpenJev setup guide](OPENJEV.md) for server commands and measured results.
