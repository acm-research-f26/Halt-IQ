# HaltIQ HotpotQA distractor subset

Derived from [HotpotQA](https://hotpotqa.github.io/) by Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, and Manning (2018). See `CITATION.bib`.

Source: [hotpot_dev_distractor_v1.json](https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet). Source and derived data are distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The changes are deterministic sampling, duplicate-question exclusion, local splitting, and formatting as HaltIQ JSONL.

The 20 `dev` and 80 `test` tasks both come from **official development data**. The `test` split is locally held out, **not the official hidden test set**. Tune the critic on dev; freeze settings before evaluating test.

Sampling is proportional by question type (bridge/comparison), seeded and stable under source row reordering. Duplicate normalized questions keep their smallest source ID. The manifest records parameters, IDs, type and context-count distributions, excluded duplicates, and source/output SHA-256 hashes. Selection never uses gold answers or model outcomes.

Every task preserves all supplied context paragraphs (up to 10), their titles, all sentences, and original order. Sentence numbers are zero-based. No gold supporting-fact filter or context truncation is applied. Only `question` and `evidence` enter writer/critic prompts. Answers and supporting-fact annotations are private evaluation data. Upstream annotation issues are recorded as evaluator-only warnings in metadata and the manifest; affected rows remain eligible. Empty sentence strings are preserved. No scripted mock answers or critic scores are included.

Report answer exact match and token F1 using the HotpotQA answer convention. This answer-only experiment does not compute supporting-fact or joint leaderboard scores. A small development-derived subset supports a critic/halting pilot, not an official leaderboard claim.
