# Public HotpotQA snapshots and evaluator boundary

`fresh20-evaluator-only.jsonl` is the exact 20-row full-context panel used by the
completed October 18 study, SHA256
`6584e2bf7a66bade630b1f1f88687c6e6962e4dc6192b27384503924d2b304ad`.
It contains 16 bridge and four comparison questions sampled without answer/model
outcome selection from public official validation data.

`train90-evaluator-only.jsonl` is the exact 90-row source for actual new writer
generation, SHA256
`1a789919c3d563999337109a4d5e97d5cb89dcadd042e1c825b66fcba59df1ff`.
Each row retains its public source row, original dataset revision, source-row
index and original partition: 60 fit, 15 tuning and 15 calibration. It does not
relabel the interrupted historical TRAIN600 as a complete training source. The
selection manifest is preserved in `../protocol/train90-selection-data-only.json`.

Both snapshots are byte-exact copies of the study inputs; evidence, order and
references are unchanged. Training and fresh question IDs are disjoint. Both are
public benchmark material, so local holdout discipline cannot exclude prior model
pretraining exposure. Fresh20 is not HotpotQA's official hidden test.

**Evaluator-only:** reference answers, supporting facts and annotation labels must
not be passed to the writer or critic. `pc_strands.data.Task.public()` constructs
only question/evidence and optionally a current draft. Full-context input is kept
in original paragraph order, with no gold supporting-fact selection or silent
truncation. Native Strands uses a 4,096-token prompt limit; overflows would remain
explicit failures. No overflow occurred in this completed fresh20 panel.

`fresh20-writer-trajectories.json` contains the actual saved 200 drafts and their
evaluator scores/usage. The wrapper records the original trajectory SHA256. Local
response bodies were removed during this review export; the untouched original
raw trajectory remains external. This derivative must not be presented to native
hash gates as the original file or called a new writer execution.

Attribution: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen,
Ruslan Salakhutdinov and Christopher D. Manning, *HotpotQA: A Dataset for Diverse,
Explainable Multi-hop Question Answering*, EMNLP 2018.
[Project](https://hotpotqa.github.io/), [citation](CITATION.bib),
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
Derived data is distributed under CC BY-SA 4.0. Retain attribution, the license
link and a description of subsequent changes when redistributing it. This does
not change or assert a separate license for repository implementation code.

Public source mirror/revision:
[`hotpotqa/hotpot_qa`, `1908d6afbbead072334abe2965f91bd2709910ab`, distractor](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/distractor).
