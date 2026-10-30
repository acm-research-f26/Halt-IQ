"""Build new, personally named Colab notebooks from a verified source commit."""
import argparse
import json
from pathlib import Path
import re


def cell(kind, text):
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(commit, benchmark):
    tau = benchmark == "Tau2"
    title = f"Yash — {benchmark} — Strands + Linear Classifier"
    slug = "tau2" if tau else "hotpotqa"
    setup = f'''import sys, subprocess, pathlib
SOURCE_COMMIT = "{commit}"
ROOT = pathlib.Path("/content/Yash_HaltIQ_Pair")
def checked(*args, **kwargs):
    subprocess.run(list(args), check=True, **kwargs)
assert (3, 12) <= sys.version_info[:2] < (3, 14), "Use a Python 3.12/3.13 runtime"
if not ROOT.exists():
    checked("git", "clone", "--no-checkout", "https://github.com/acm-research-f26/Halt-IQ.git", str(ROOT))
checked("git", "-C", str(ROOT), "fetch", "--depth", "1", "origin", SOURCE_COMMIT)
checked("git", "-C", str(ROOT), "checkout", "--detach", SOURCE_COMMIT)
checked(sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT/"requirements-colab.txt"))
# These FP16 models do not use TorchAO. Older Colab images preinstall a version
# that PEFT 0.21 rejects while importing its optional quantization dispatcher.
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
try:
    unused_torchao_version = version("torchao")
except PackageNotFoundError:
    unused_torchao_version = None
if unused_torchao_version is not None and Version(unused_torchao_version) < Version("0.16"):
    print("Removing unused preinstalled TorchAO", unused_torchao_version,
          "because PEFT requires TorchAO >= 0.16 when it is installed; this study uses FP16.")
    checked(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")
checked(sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT))
'''
    if tau:
        setup += '''TAU_ROOT = pathlib.Path("/content/tau2-bench")
TAU_COMMIT = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"
if not TAU_ROOT.exists():
    checked("git", "clone", "--no-checkout", "https://github.com/sierra-research/tau2-bench.git", str(TAU_ROOT))
checked("git", "-C", str(TAU_ROOT), "fetch", "--depth", "1", "origin", TAU_COMMIT)
checked("git", "-C", str(TAU_ROOT), "checkout", "--detach", TAU_COMMIT)
checked(sys.executable, "-m", "pip", "install", "-q", "-e", str(TAU_ROOT))
'''
    setup += 'print("Pinned source installed:", SOURCE_COMMIT)\n'
    storage = f'''from google.colab import drive
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid, os, json
drive.mount("/content/drive")
RUN_ID = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
OUT = pathlib.Path("/content/drive/MyDrive/Yash_HaltIQ_Pair") / "{slug}" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
(OUT/"environment.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
print("Yash's result folder:", OUT)
'''
    verification = '''checked("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
test_env = dict(os.environ, LOGURU_LEVEL="ERROR")
'''
    if tau:
        verification += 'test_env["HALTIQ_TAU_UPSTREAM"] = str(TAU_ROOT)\n'
    verification += '''checked(sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests_pair", "-q", cwd=ROOT, env=test_env)
'''
    replay = '''checked(sys.executable, "-u", "-B", "-m", "haltiq_pair", "replay", "--output", str(OUT/"teammate-replay"), cwd=ROOT)
'''
    smoke = '''# An invented example checks real SDK + GPU + critic wiring, not benchmark accuracy.
smoke_code = """
import json, sys
from pathlib import Path
from haltiq_pair.runtime import Runtime
with Runtime(Path(sys.argv[1])) as (agents, gate):
    draft = agents.generate('writer', 'Return exactly one JSON object with the key answer.',
                            'Public passage: Paris is the capital of France. What is its capital?')
    result = gate.decide('combined', 'Paris', None, 1, 6,
                         'What is the capital of France?', ['France: Paris is its capital.'])
    print(json.dumps({'model_output':draft,'gate':result,'usage':agents.usage}, indent=2))
"""
checked(sys.executable, "-u", "-B", "-c", smoke_code, str(OUT/"gpu-smoke"), cwd=ROOT)
print("Real GPU smoke completed. The invented example is separate from benchmark results.")
'''
    command = '"tau"' if tau else '"hotpot"'
    count = 10 if tau else 20
    live = f'''# Each arm makes fresh live calls. Existing teammate drafts are not used here.
import time
LIMIT = {count}
SEED = 42
LIVE = OUT / "live"
log_path = OUT / "live-console.log"
command = [sys.executable, "-u", "-B", "-m", "haltiq_pair", {command},
           "--output", str(LIVE), "--limit", str(LIMIT), "--seed", str(SEED)]
'''
    if tau:
        live += 'command += ["--upstream-root", str(TAU_ROOT)]\n'
    live += '''with log_path.open("w") as log:
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, LOGURU_LEVEL="ERROR", TOKENIZERS_PARALLELISM="false"))
    last = None
    try:
        while process.poll() is None:
            summary = LIVE / "summary.json"
            if summary.exists():
                content = summary.read_text()
                if content != last:
                    value = json.loads(content)
                    print(json.dumps(value.get("arms", value), ensure_ascii=False), flush=True)
                    last = content
            time.sleep(10)
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise
    if process.returncode:
        print(log_path.read_text()[-6000:])
        raise RuntimeError(f"Study failed with exit {process.returncode}; saved logs: {log_path}")
print((LIVE/"Research Writeup.txt").read_text())
print("Saved all results:", OUT)
'''
    export = '''import shutil
from google.colab import files
archive = shutil.make_archive("/content/Yash_HaltIQ_Results_"+RUN_ID, "zip", OUT)
files.download(archive)
print("Drive journals and this ZIP are Yash's new study artifacts.")
'''
    cells = [
        cell("markdown", f"# {title}\n\nThis is a **new notebook for Yash**. Save your own copy in Drive; do not edit a teammate's notebook. It uses only Vedanshi's and Shree's prior evidence. Your earlier experiments are excluded.\n\nSelect **Runtime → Change runtime type → GPU** before running. No AWS or paid model API key is required. A standard Colab Drive mount saves results incrementally."),
        cell("markdown", "## 1. Install the frozen source\nThe source commit, model revisions, tasks, policies and thresholds are recorded. No training or threshold tuning on evaluation questions is performed."),
        cell("code", setup),
        cell("markdown", "## 2. Save Yash's results to Drive\nApprove the ordinary Colab Drive connection in your research account. No credentials are written to the notebook or result files."),
        cell("code", storage),
        cell("markdown", "## 3. Verify the GPU and software wiring\nOffline tests are engineering checks. Their synthetic fixtures are not benchmark results."),
        cell("code", verification),
        cell("markdown", "## 4. Recompute existing teammate replay\nThis uses the actual pretrained linear heads. The inherited head stops immediately; combined AND can equal Strands-only. Replay token prefixes omit critic/reranker compute and are not live savings."),
        cell("code", replay),
        cell("markdown", "## 5. Run a real GPU smoke test\nThis checks the new AWS Strands agents, local writer and pinned Hobson/BGE critic with an invented fact. FP16 on T4 is recorded rather than treated as identical to historical BF16 probabilities."),
        cell("code", smoke),
        cell("markdown", "## 6. Run the new live simulation\n" + ("Ten official English retail tasks × four arms. The local user and judges differ from released leaderboard conditions. Failed episodes stay in denominators; rejected tool proposals never execute. These are official-domain local simulations." if tau else "Twenty fresh HotpotQA validation questions × five arms. Full public distractor evidence is available; gold answers and supporting facts stay in the evaluator. Each arm runs independently. Six-round budget exits are distinguished from gate approvals.") + "\n\nThe console prints incremental results; full traces and logs remain in Drive. Run this cell once per new output directory."),
        cell("code", live),
        cell("markdown", "## 7. Download Yash's evidence and writeup\nThe ZIP includes manifests, runtime provenance, model receipts, traces, outcomes and a plain-language writeup. Keep incomplete/failing outputs labeled as such."),
        cell("code", export),
    ]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": f"Yash_{benchmark}_Strands_Linear.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "gpuClass": "standard", "haltiq_source_commit": commit}, "cells": cells}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        raise ValueError("Use an immutable full source commit")
    root = Path(__file__).resolve().parents[1]/"notebooks"
    root.mkdir(exist_ok=True)
    for name in ("HotpotQA", "Tau2"):
        path = root / f"Yash_{name}_Strands_Linear.ipynb"
        notebook = build(args.commit, name)
        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(path), "exec")
        path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)
