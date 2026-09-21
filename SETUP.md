# Baseline setup notes (Apple Silicon Mac, macOS)

Upstream harness: https://github.com/SahilShrivastava-Dev/semantic-halting-problem
Commit used: 6c3eb99

## Environment
- Python 3.12 via uv. System Python 3.14 is too new for the unpinned dependencies.
- Native arm64 only. Avoid an old Intel Homebrew at /usr/local.
- From backend/: `uv venv ../venv --python 3.12 && source ../venv/bin/activate`
- `uv pip install -r requirements.txt`
- Required fix: `uv pip install "langchain-community<0.4"`.
  Without it, `import ragas` fails with
  `No module named 'langchain_community.chat_models.vertexai'`.
- Full known-good package list: requirements-working.txt
  (a freeze from an Apple Silicon Mac, so it may not install as-is on other platforms).

## Verified
- `python -m shp.theory_checks` passes.
- `python experiments/run_experiment.py --mock --split dev --max-rounds 6 --ablations --run-id mocktest`
  completes (20 scenarios, 18 policies). Output: mock_results/summary.csv.
  This is synthetic mock data, not a real result.

## Not done yet
- Real run against an API provider on HotpotQA.
- HaltIQ-specific code (StopPolicyHead, GRPO training): not yet located.
