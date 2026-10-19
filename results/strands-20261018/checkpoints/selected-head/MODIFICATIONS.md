# Modified checkpoint notice — October 18, 2026

The `head.safetensors` and `strands_decider_config.json` in this directory are
modified derivatives of the Apache-2.0 StrandsAgents
`strands-decider-2B-qwen3.5-v1-2610` checkpoint at revision
`216d1ccb46a9e9b036eb9187b248db98480f77d4`.

Yash Baruah's HaltIQ experiment updated only the existing pointer-head parameters
through supervised correctness training on a disjoint HotpotQA 60-fit/15-tuning
cohort. The selected head is learning-rate 1e-5, epoch 1. A separate 15-question
calibration fit changed the Noul temperature to 1.4266376751158028 in the config.
The pretrained Qwen encoder and released Strands LoRA adapter were frozen and are
not redistributed in this directory. No RL or additional LoRA update is claimed.

Model/source attribution: Strands Decider contributors,
https://github.com/strands-labs/strands-decider and
https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610.
The required base is Qwen/Qwen3.5-2B-Base, Copyright 2026 Alibaba Cloud, as stated
in its retained `LICENSE-Qwen`. Preserve the upstream full Apache-2.0 licenses,
the upstream notices and this modification notice when redistributing these files.
This scoped checkpoint attribution does not assign a new license to unrelated
repository implementation files. Bundled HotpotQA data retains its separate
CC BY-SA 4.0 attribution and change notice.
