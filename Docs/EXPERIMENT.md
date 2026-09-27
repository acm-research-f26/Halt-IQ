# Monday JEV critic prototype

The implementation tests a narrow question: can a JEV critic decide when an evidence-grounded answer is good enough to stop revising, while preserving answer correctness and reducing unnecessary rounds? This is an inference-time control experiment. Training, reinforcement learning, and a broad benchmark study are outside this prototype.

The current research environment is HotpotQA distractor, prepared as 20 development and 80 locally held-out questions in `data/hotpotqa/tasks.jsonl`. Both are disjoint subsets of public official development data, not the hidden official test split. See [the environment guide](ENVIRONMENT.md). The earlier 24-item synthetic diagnostic fixture in `data/diagnostic.jsonl` remains available; its offline demo confirms data flow, logging, and stopping behavior. Scripted drafts and scores cannot establish whether JEV works. Evidence about a real critic requires real writer and critic calls.

## Comparison

Use the same local Ollama writer model, model revision, generation settings, evidence, and maximum number of answer drafts for all arms. Create the first draft once for each item and reuse that exact text in every arm. The current harness records it in traces and shares it within a run; a new run regenerates it. Reusing a cached initial draft across separate runs is future work, so compare arms together in one invocation. With a cap of three drafts, there are at most two revisions after that shared first draft.

| Arm | Behavior |
| --- | --- |
| Fixed rounds | Follow the configured revision loop through the full draft cap, without allowing a critic approval to halt it. Record any feedback-producing calls separately. |
| Conventional LLM critic | Obtain ordinary critique and an explicit approval decision from a language model; revise on rejection and stop on approval or the cap. |
| JEV typed critic | Score answer support, completeness, and relevance with the JEV adapter; convert the recorded scores into an approval decision using thresholds frozen on development items; revise on rejection and stop on approval or the cap. |

The harness and reported configuration must specify what feedback each revision receives. Feedback differences change later drafts, so the main experiment compares complete systems; equal initial drafts alone do not isolate stopping quality. For a stricter stopping-only follow-up, generate one common sequence of revisions per item, then ask both critics to select a stopping point on those identical candidates. Do not present those two experiments as interchangeable.

If the fixed-round arm uses conventional critique to generate revisions, keep that critique even after an approval but ignore the approval for stopping. Count those calls and describe this arm as fixed rounds with conventional feedback. If it instead uses unguided self-revision, report that explicitly. An observed difference can then reflect both feedback and halting.

Choose a local writer that fits both machines, keep its identity fixed within a comparison, and smoke-test the same configuration on the PC and Mac. Ollama model availability is a setup prerequisite, not an assumption that either machine has a model installed. CPU/GPU differences affect latency; measure and report each machine separately.

JEV and OpenJEV are distinct implementations. Record which endpoint or local adapter actually supplied scores, with version/model information where available. Do not relabel scripted scores or an unrelated LLM judge as JEV. A local typed judge can exercise the interface if JEV access is pending, but its results are a separately named baseline.

## Development and test discipline

1. Check that all fixture answers follow from the supplied evidence, and independently review the ambiguous or insufficient-evidence cases before a real run.
2. Run the offline demo to validate loops, shared first drafts, bounded rounds, scoring, and failure reporting. Its default use should not incur API charges.
3. Use only the 20 HotpotQA development cases (8 for the synthetic diagnostic environment) for prompt changes, score thresholds, parser checks, retry limits, or model selection. Record every threshold and aggregation rule: for example, whether approval requires each score to exceed its threshold, rather than an average that hides one weak dimension.
4. Freeze the configuration and source revision. Save model identifiers, prompt text or prompt hashes, settings, round cap, retry policy, and threshold values with the run.
5. Run the 80 locally held-out HotpotQA cases once (16 for the synthetic diagnostic environment) with the frozen settings. Keep the individual item records, not just averages. Label these results exploratory; this sample is too small to establish significance or broad superiority.
6. Any change motivated by test outcomes invalidates that split as a fresh held-out check. Make the change openly, then use new independently reviewed examples for the next confirmatory run.

The fixture and its gold fields are visible to developers. Its development/test labels enforce a workflow, not secrecy. A later research study should use a larger external evaluation set or independently authored hidden cases, chosen before seeing comparative outcomes.

## Model input boundary

Give the writer only the question and evidence, with the requested short-answer format. Revisions may additionally receive the previous answer and the assigned critic's feedback. Give each critic the same question, evidence, and current candidate answer. A typed output schema may define what support, completeness, and relevance mean, but it must not contain the item's gold answer.

Use an allowlist when constructing these inputs. `answers` is evaluator-only. `demo_drafts` and `demo_scores` are offline-demo-only. Do not paste full JSONL rows into real prompts, expose gold to a critic to make its scores look better, or use demo scores as real results. Log enough request metadata to audit this boundary without recording secrets.

## Measurements

HotpotQA runs dispatch to the official answer EM/F1 conventions, including punctuation deletion and special yes/no/noanswer handling; no supporting-fact or joint leaderboard score is implemented. Synthetic diagnostics retain their original numeric-aware normalizer. The independent evaluator compares the returned short answer against the acceptable gold strings, after a frozen normalization rule. Report normalized exact match (EM) as the primary correctness measure. Also report token F1: compute precision and recall from the multiset overlap of normalized answer tokens, take their harmonic mean, and keep the best score over acceptable aliases. For these short answers, partial token overlap is a diagnostic and does not mean a fact is correct. Report the exact normalizer and empty-answer convention alongside results.

For each arm, retain:

- Final-answer EM and token F1, with item-level differences from the shared first draft and from the other arms.
- Drafts produced and revisions performed, reported separately so the initial answer is not confused with a revision.
- Logical critic decisions, underlying critic requests, writer requests, retries, failures, and any batched evaluations. One logical typed decision can involve multiple billable evaluations; do not conflate the two counts.
- End-to-end wall time, writer and critic time, and whether the shared first-draft time is included or amortized. Keep cached and uncached runs separate, and identify hardware.
- Approval rate and false approvals: an approval of an answer with EM = 0 is false. Show the count and denominator both as a fraction of all approvals and as a fraction of all evaluated items. If there are no approvals, the conditional rate is undefined, not zero.
- Premature stopping on a wrong answer, unnecessary rejection of a correct answer, cap exhaustion, malformed outputs, network/model errors, and successful completed items.
- Provider-reported usage and cost where available, estimated usage/cost with assumptions where unavailable, and any unpriced calls separately. Local inference has no hosted API charge but still consumes hardware time and electricity.

Count approval events across every judged draft separately from the final stopping event. In a fixed-round arm, an intermediate approval does not stop the loop; it is still useful to know whether that approval was false. Keep operational failures visible instead of dropping failed items and reporting an inflated accuracy. Report accuracy among completed cases and completion rate over all intended cases; also show a conservative success fraction with failed cases counted as unsuccessful.

A three-draft cap is reached when the third draft is the returned answer, whether or not it is correct. It is not itself a model failure. If a critic fails, this prototype ends that episode as failed and preserves the last draft; it does not retry or switch critics. Never silently treat a parse error or missing score as approval. Preserve `UNKNOWN` as a valid answer only where the evidence cannot determine the requested fact.

Report paired item-level outcomes because each arm starts from the same draft. Repeated runs can reveal stochastic variability, but do not claim statistical significance or population-level gains from a small development pilot. A promising Monday result is a working adapter, reproducible records, and a transparent comparison showing both successes and failure cases.

## Budget and run order

The user's total hosted-inference ceiling is $30. The first milestone should use the free offline demo and a local writer. Before making paid calls, verify the relevant current pricing or account quota and set a conservative experiment allowance below the remaining ceiling, leaving headroom for retries and uncertain usage. Do not assume a request count is a dollar cost, or that a free-tier label guarantees a zero bill.

Reserve a small development pilot first. Estimate the full test run from observed usage and the worst-case draft cap, including every arm, every scored dimension, and retries. Enforce the declared request/round limits and stop when the next operation would exceed the remaining conservative allowance. If the service cannot supply a reliable spend bound, keep paid execution disabled until a provider-side limit or a defensible upper bound is available. Record spending from earlier runs toward the same $30 ceiling.

Save a run manifest and machine-readable item records before interpreting results. The Monday handoff should include the adapter/interface, a successful offline demonstration, setup instructions for both operating systems, the frozen configuration for any real pilot, and the result files or an explicit statement that live JEV evaluation remains unrun. Vedanshi can then run the baseline comparison using the same item IDs and shared first drafts. Sending messages to collaborators is outside this implementation step.
