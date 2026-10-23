# Selected trained Strands pointer head

This directory contains the actual selected 3,151,360-parameter head from the
October 18 study, not a synthetic model or a new training run.

- `head.safetensors`: 12,605,896 bytes; SHA256
  `d95b640d54ccc004cc136dfbb38d896e942f299d972bed555bf07f7ab648ccf5`.
- `strands_decider_config.json`: exact 779-byte selected config, SHA256
  `c8524a28dd0b38acbfd9462f19a03d576cea0533ae148a3b3382c00a32c729d1`.
- Selected LR 1e-5, epoch 1, two optimizer steps. Twelve optimizer steps were
  performed across all candidate fits; the selected head does not accumulate all
  candidate trials. Matched calibration set size 15, Noul temperature
  `1.4266376751158028`.
- `header-verification.json`: non-Torch Safetensors header/hash verification.
  Six F32 tensors, 3,151,360 parameters. The native training/reload proofs are in
  `../../proofs/` and were generated during the completed real model run.

The head is not a standalone chat model. Download the required public components
separately, preserving these pins:

- [Strands checkpoint](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/tree/216d1ccb46a9e9b036eb9187b248db98480f77d4)
  revision `216d1ccb46a9e9b036eb9187b248db98480f77d4` for tokenizer/template and
  the **unchanged released adapter**. Adapter weight SHA256
  `ae1d0cd1ac3edec6df0a519a3ad18f1c382099032d493f8f98c96a80cfebb4c7`.
- [Qwen base](https://huggingface.co/Qwen/Qwen3.5-2B-Base/tree/b1485b2fa6dfa1287294f269f5fb618e03d52d7c)
  revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
- [Strands source](https://github.com/strands-labs/strands-decider/tree/930e6d5de16309be32579f5631a7369a3322140c)
  commit `930e6d5de16309be32579f5631a7369a3322140c`; environment and public
  download SHA256 queue are linked from the bundle README.

To restore for a **new** local run, copy the pinned released checkpoint directory
to a new external runtime directory. Keep its tokenizer, template, LoRA config
and frozen adapter unchanged. Replace only that copy's `head.safetensors` and
`strands_decider_config.json` with the two exact files here. Keep this bundle's
wrapped training/calibration proofs as review provenance; do not rename them as
original raw proof files or use them to satisfy historical campaign hash gates. Do not
overwrite the released checkpoint or previous experiment outputs. Verify hashes
before loading; retain the supplied trained temperature rather than the released
temperature. Recorded source/model settings describe the completed experiment;
create a new explicit plan for native execution rather than reusing old deadlines.

The model author explicitly releases the checkpoint under Apache-2.0 in the pinned
`LICENSE-Strands.md`; its full text and notes are retained byte-exact here. The
base model's Apache-2.0 `LICENSE-Qwen` is also retained. See `MODIFICATIONS.md` for
the prominent derivative notice. The upstream source's third-party notices are
included for attribution context; their MPS-kernel notice concerns source code,
not an assertion that an MPS kernel is contained in this pointer-head file.

This selected head did **not** improve fresh20 answer scores or first-decision
AUROC over the calibrated released head. The file is supplied for reproducibility
and further research, with that negative result preserved.
