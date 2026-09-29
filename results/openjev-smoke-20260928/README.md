# OpenJev integration checks

These records exercise the installed `daseinlabs/open-jev` scorer with the pinned Gemma 3 4B MLX checkpoint. They are two synthetic examples and a numerical check, not benchmark accuracy results.

`reviews.json` records the question/evidence/draft and actual server responses for an incorrect answer (Hillford) and a correct answer (Bellport). Both were rejected by the three-check, 0.8-threshold policy: the correct answer's completeness score was approximately 0.0097. This exposes a model/rubric limitation rather than demonstrating successful calibration.

`upstream-numerical-check.txt` is the output of:

```bash
.haltiq/openjev-venv/bin/openjev check --model .haltiq/models/gemma-3-4b-it-4bit --backend mlx --batch-size 2 --context-tokens 1500
```

It passed upstream's default absolute log-probability tolerance of 0.5; the largest observed difference was approximately 0.375. Cached and naive scores are not bitwise identical on this quantized model. This check does not assess answer accuracy or calibration.

`runtime.json` contains checkpoint hashes and installed package versions, with the local model path made relative for portability. `setup_config.json` pins the upstream and model revisions. `server_snapshot.py` is the wrapper used for these calls. Model weights and virtual environments are not committed.

For the real benchmark comparison, see [the eight-question HotpotQA run](../hotpotqa-openjev-dev-8-20260928/report.md). Community OpenJev results do not establish official TypeSafe Jev performance.
