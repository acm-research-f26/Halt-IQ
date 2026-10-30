Yash's October 30 extension implements the teammate's continuous Strands-score
feature, trains new linear heads, and provides separate fresh HotpotQA and
official τ²-bench runners. All source changes are additions owned by this
extension. Teammate source, notebooks and saved data remain unchanged.

Verified runtime source: `696f8063471b616a24288249d79cce0accec53e3`.
The two `Yash_*_Team_Aligned_20261030.ipynb` notebook templates embed the exact
selected head and prospective cohort receipts. They install that pinned source
in a separate checkout and write new timestamped result directories.

Two training objectives are reported separately:

- `python -m haltiq_pair.team_training --output NEW_DIRECTORY` reproduces the
  teammate's five-seed policy-gradient setup with the continuous score feature.
  It fits train50, selects on dev20, freezes all five checkpoints, then replays
  the known cached test80. All five selected heads still stop in round one.
- `python -m haltiq_pair.quality_training --output NEW_DIRECTORY` trains a
  separately declared balanced exact-match correctness classifier using the
  same five public features. It performs 200 Adam steps per seed and selects
  the earliest minimum dev BCE. This is an exploratory objective change, not
  a reproduction of Sri's reward optimization. The class-weighted output is
  a correctness score, not a calibrated probability of correctness.

The primary fresh-deployment head is correctness-classifier seed0, declared
before any fresh calls. File SHA256:
`b0d5256f6538898d09a0a07322124def66b2abe505a08abf92c13f81fd9dcd0b`.
Every seed remains reported; no seed was selected using test performance.

Cached test80, predeclared seed0:

| Policy | Answer F1 | Mean rounds | Mean saved writer/reviewer tokens |
|---|---:|---:|---:|
| Fixed1 | 0.35878 | 1.0000 | 1,514.21 |
| Fixed6 | 0.39205 | 6.0000 | 18,227.36 |
| Repeated answer | 0.39205 | 2.1625 | 5,076.33 |
| Strands only | 0.37843 | 4.0000 | 10,294.74 |
| Correctness classifier | 0.37843 | 2.8375 | 7,034.09 |
| Correctness classifier AND Strands | 0.37843 | 4.0000 | 10,294.74 |

The new classifier changes stopping without improving cached answer F1.
Seeds3/4 never approve and return at the six-round cap, showing seed sensitivity.
The repeated-answer baseline remains competitive. These are 80 unique questions,
not 400 independent questions across five seeds. Test80 was previously viewed;
it is held out from fitting and checkpoint selection, but is not a fresh blind
confirmation set. Saved prefix-token estimates exclude verifier/reranker
compute, and Fixed6's cached total includes its terminal reviewer. No total
compute, measured live speedup or noninferiority claim follows from this table.

The fresh HotpotQA primary study uses 20 seed44 questions and five independently
executed arms: Fixed1, Fixed6, Strands, retrained head, and retrained head with
maximum-Strands selection among its observed prefix. A separately frozen
40-question seed43 collection can follow if enabled before calls. Its ten
policy results are counterfactual replays of full six-round trajectories; the
collector's actual physical cost is reported separately. Both cohorts exclude
all earlier team/pilot/final questions and are mutually disjoint. The runner
rejects a changed predeclared head or reused teammate train/dev/test identity.

The new τ² variant uses official pinned retail tasks/tools/evaluators and the
Qwen2.5-7B-Instruct writer in NF4. It requires an invented-input SDK/parser/critic
check and a full long-context GPU memory qualification before its two-task
development smoke study or ten-task evaluation. The development and evaluation
IDs exclude the original ten-task panel and each other. All local simulator
and judge models are identified explicitly. QA-to-retail thresholds are
uncalibrated transfer assumptions. A cap refusal does not execute a tool call.

Both runners allow at most one public formatting-correction call, retain the
original malformed output and every attempted call, and never repair an answer
using gold labels. Failures receive zero reward in the planned denominator;
partial runs remain incomplete. Dollars are not inferred from missing local
model pricing.

Validation: 57 focused controls pass across the CPU SDK and official τ²
environments, including optimizer/test-label boundaries, checkpoint freezing,
ID normalization, prefix-only selection, correction costs, real official-client
loopback model identity and tool execution isolation. These checks do not
qualify the actual 7B Colab GPU configuration or provide new benchmark outcomes.

At this milestone, the ten new heads and cached evaluations are complete.
Fresh GPU inference has not started: both prior Colab GPU sessions expired and
their new presence checks require current approval or user completion. The
completed October 29 live studies and their failure records remain separate.
The new supplemental writeup and Friday/poster/professor preparation accompany
the evidence bundle; no meeting or presentation eligibility is claimed.
