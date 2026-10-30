"""Build two new Yash notebooks with exact frozen head/cohort bytes.

The required full source commit is independently checked for the new modules.
A draft pin can be inspected, but its bootstrap refuses to run missing modules.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO.parent / "team-alignment-2026-10-30"
HEAD_SHA256 = "b0d5256f6538898d09a0a07322124def66b2abe505a08abf92c13f81fd9dcd0b"
TAU_COMMIT = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"
REQUIRED_MODULES = (
    "haltiq_pair/team_training.py", "haltiq_pair/aligned_gate.py",
    "haltiq_pair/aligned_hotpot.py", "haltiq_pair/aligned_tau.py",
    "haltiq_pair/strong_writer.py",
)
FILENAMES = {
    "hotpot": "Yash_HotpotQA_Team_Aligned_20261030.ipynb",
    "tau": "Yash_Tau2_7B_Team_Aligned_20261030.ipynb",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def cell(kind, source):
    result = {"cell_type": kind, "id": "yash-" + digest(source.encode())[:16],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def source_missing(commit):
    return [name for name in REQUIRED_MODULES if subprocess.run(
        ["git", "cat-file", "-e", commit + ":" + name], cwd=REPO,
        capture_output=True, check=False).returncode]


def load_inputs(root):
    paths = {
        "Yash-BCE-Strands-Head-seed0.json": root / "Yash-Quality-Strands-Head/seed0/head.json",
        "Yash-HotpotQA-Cohort40.json": root / "Yash-Fresh-HotpotQA-Cohort-40.json",
        "Yash-HotpotQA-Online20.json": root / "Yash-Fresh-HotpotQA-Online-Cohort-20.json",
        "Yash-Tau-Smoke-IDs.json": root / "Yash-Team-Colab-Audit/Yash-Aligned-Tau-Smoke-Task-Freeze.json",
        "Yash-Tau-Evaluation-IDs.json": root / "Yash-Team-Colab-Audit/Yash-Aligned-Tau-Task-Freeze.json",
    }
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    head, collection, online = (json.loads(payloads[name]) for name in list(paths)[:3])
    if (digest(payloads["Yash-BCE-Strands-Head-seed0.json"]) != HEAD_SHA256
            or head["seed"] != 0 or head["metadata"]["objective"] != "balanced_BCE_exact_match"
            or head["metadata"]["live_seed_predeclared"] != 0):
        raise ValueError("The predeclared balanced-BCE seed-zero head changed")
    if (collection["limit"] != 40 or collection["seed"] != 43
            or online["limit"] != 20 or online["seed"] != 44
            or online["primary_live_head_sha256"] != HEAD_SHA256
            or collection["unique_exclusion_count"] != 290
            or online["unique_exclusion_count"] != 330):
        raise ValueError("The predeclared prospective cohort changed")
    for cohort in (collection, online):
        excluded = set().union(*(set(ids) for ids in cohort["exclusion_groups"].values()))
        if (len(cohort["question_ids"]) != len(set(cohort["question_ids"]))
                or set(cohort["question_ids"]) & excluded):
            raise ValueError("Prospective questions are duplicated or overlap exclusions")
    if set(collection["question_ids"]) & set(online["question_ids"]):
        raise ValueError("Collection and independent online cohorts must be disjoint")
    return {name: {"sha256": digest(raw), "text": raw.decode("utf-8")}
            for name, raw in payloads.items()}


def bootstrap(commit, benchmark):
    source = f'''import sys, subprocess, pathlib, os, json
SOURCE_COMMIT = {commit!r}
ROOT = pathlib.Path("/content/Yash_HaltIQ_Aligned_20261030_{benchmark}")
def checked(*arguments, **kwargs):
    subprocess.run(list(arguments), check=True, **kwargs)
assert (3, 12) <= sys.version_info[:2] < (3, 14), "Use a Python 3.12/3.13 runtime"
if not ROOT.exists():
    checked("git", "clone", "--no-checkout", "https://github.com/acm-research-f26/Halt-IQ.git", str(ROOT))
checked("git", "-C", str(ROOT), "fetch", "--depth", "1", "origin", SOURCE_COMMIT)
checked("git", "-C", str(ROOT), "checkout", "--detach", SOURCE_COMMIT)
required_modules = {list(REQUIRED_MODULES)!r}
missing_modules = [name for name in required_modules if not (ROOT/name).is_file()]
assert not missing_modules, "Draft source pin is missing the new modules; regenerate with the verified integration commit: " + repr(missing_modules)
checked(sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT/"requirements-colab.txt"))
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
try:
    installed_torchao = version("torchao")
except PackageNotFoundError:
    installed_torchao = None
if installed_torchao is not None and Version(installed_torchao) < Version("0.16"):
    print("Removing unused incompatible TorchAO", installed_torchao)
    checked(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")
checked(sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT))
'''
    if benchmark == "Tau2":
        source += f'''checked(sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2")
TAU_ROOT = pathlib.Path("/content/Yash_Tau2_Official_20261030")
TAU_COMMIT = {TAU_COMMIT!r}
if not TAU_ROOT.exists():
    checked("git", "clone", "--no-checkout", "https://github.com/sierra-research/tau2-bench.git", str(TAU_ROOT))
checked("git", "-C", str(TAU_ROOT), "fetch", "--depth", "1", "origin", TAU_COMMIT)
checked("git", "-C", str(TAU_ROOT), "checkout", "--detach", TAU_COMMIT)
checked(sys.executable, "-m", "pip", "install", "-q", "-e", str(TAU_ROOT))
assert version("bitsandbytes") == "0.50.2"
'''
    return source + 'print("Pinned source installed:", SOURCE_COMMIT)\n'


def storage(benchmark):
    return f'''from google.colab import drive
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
USE_DRIVE = True
storage_root = pathlib.Path("/content/Yash_HaltIQ_Aligned_Results_20261030")
backend, mount_error = "runtime", None
if USE_DRIVE:
    try:
        drive.mount("/content/drive")
        storage_root = pathlib.Path("/content/drive/MyDrive/Yash_HaltIQ_Pair")
        backend = "google_drive"
    except Exception as error:
        mount_error = type(error).__name__
        print("Drive connection failed; this new result folder will be temporary.")
RUN_ID = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
OUT = storage_root / "team-aligned-20261030" / "{benchmark.lower()}" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
storage_metadata = dict(backend=backend, requested_drive=USE_DRIVE,
    persistent=backend == "google_drive", drive_mount_error_type=mount_error,
    requires_download_before_disconnect=backend != "google_drive", result_directory=str(OUT))
with (OUT/"storage.json").open("x", encoding="utf-8") as stream:
    json.dump(storage_metadata, stream, indent=2)
with (OUT/"environment.txt").open("x", encoding="utf-8") as stream:
    stream.write(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
print("Yash's new result folder:", OUT)
if not storage_metadata["persistent"]:
    print("WARNING: download the final evidence ZIP before disconnecting this temporary runtime.")
'''


def frozen_inputs(inputs, benchmark):
    names = (["Yash-BCE-Strands-Head-seed0.json", "Yash-HotpotQA-Cohort40.json", "Yash-HotpotQA-Online20.json"]
             if benchmark == "HotpotQA" else
             ["Yash-BCE-Strands-Head-seed0.json", "Yash-Tau-Smoke-IDs.json", "Yash-Tau-Evaluation-IDs.json"])
    selected = {name: inputs[name] for name in names}
    source = "import hashlib\nEMBEDDED_FILES = " + repr(selected) + "\n"
    if benchmark == "HotpotQA":
        source += "RUN_EXTENDED_COLLECTION = False  # Choose before any fresh model calls.\n"
    source += f'''FROZEN = OUT/"frozen-inputs"
FROZEN.mkdir(exist_ok=True)
for name, item in EMBEDDED_FILES.items():
    raw = item["text"].encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == item["sha256"], "Embedded input bytes changed"
    path = FROZEN/name
    if path.exists():
        assert path.read_bytes() == raw, "Existing frozen input changed; choose a new OUT"
    else:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
HEAD = FROZEN/"Yash-BCE-Strands-Head-seed0.json"
assert hashlib.sha256(HEAD.read_bytes()).hexdigest() == {HEAD_SHA256!r}
from haltiq_pair.team_training import StrandsHead
selected_head = StrandsHead.from_json(HEAD)
assert selected_head.seed == 0
assert selected_head.metadata["objective"] == "balanced_BCE_exact_match"
plan = dict(source_commit=SOURCE_COMMIT, benchmark={benchmark!r},
    frozen_input_sha256={{name: item["sha256"] for name, item in EMBEDDED_FILES.items()}},
    primary_head_objective="balanced_BCE_exact_match", deployment_seed_predeclared=0,
    previously_viewed_cached_test80="exploratory, not a fresh blind evaluation",
    PG_negative_result="matched PG retraining stopped at round one; retained separately",
    frozen_before_fresh_model_calls=True)
'''
    if benchmark == "HotpotQA":
        source += '''plan.update(primary="independent online20 first", optional_extension="six-round collection40 after primary",
    run_extended_collection=RUN_EXTENDED_COLLECTION,
    extension_choice="frozen before model calls; no result-based protocol adaptation")
'''
    source += '''
plan_path = OUT/"notebook-plan.json"
if plan_path.exists():
    assert json.loads(plan_path.read_text()) == plan
else:
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
print("Exact frozen inputs verified:", plan["frozen_input_sha256"])
'''
    return source


RUN_HELPER = '''import time
def run_stage(name, arguments, destination):
    destination = pathlib.Path(destination)
    assert not destination.exists(), "Preserve previous stage outputs; choose a new OUT"
    log_path = OUT/(name + "-console.log")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "-B", *arguments], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, LOGURU_LEVEL="ERROR", TOKENIZERS_PARALLELISM="false"))
        previous = None
        try:
            while process.poll() is None:
                summary = destination/"summary.json"
                if summary.exists():
                    current = summary.read_text(encoding="utf-8")
                    if current != previous:
                        value = json.loads(current)
                        print(name, json.dumps(value.get("policies", value.get("arms", value))), flush=True)
                        previous = current
                time.sleep(5)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise
        if process.returncode:
            print(log_path.read_text(encoding="utf-8")[-8000:])
            raise RuntimeError(f"{name} exited {process.returncode}; preserved log: {log_path}")
    summary = json.loads((destination/"summary.json").read_text(encoding="utf-8"))
    assert summary["complete"], "Stage is incomplete; preserve evidence and inspect its logs"
    print(name, "finished; failures remain in all planned denominators.", flush=True)
    return summary
'''


HOTPot_SMOKE = '''import json, sys
from pathlib import Path
from haltiq_pair.aligned_gate import Runtime
from haltiq_pair.hotpot import WRITER_SYSTEM, REVIEWER_SYSTEM, parse_draft, parse_review
from haltiq_pair.team_training import StrandsHead
out, head_path = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), "Preserve earlier engineering qualifications"
with Runtime(out, StrandsHead.from_json(head_path)) as (agents, gate):
    public = 'Question: What is the capital of France?\\nPassage title: France\\n[0] Paris is the capital of France.'
    raw = agents.generate('writer', WRITER_SYSTEM, public)
    draft = parse_draft(raw)
    review_raw = agents.generate('reviewer', REVIEWER_SYSTEM, public + '\\nProposed draft:\\n' + raw)
    review = parse_review(review_raw)
    decision = gate.decide('retrained', draft['answer'], None, 1, 6,
        'What is the capital of France?', ['France: Paris is the capital of France.'])
    assert isinstance(decision['p_approve'], float)
    receipt = dict(status='passed', kind='invented engineering input, not benchmark accuracy',
        parsed_draft=draft, parsed_review=review, gate=decision, usage=agents.usage,
        runtime=json.loads((out/'runtime.json').read_text()))
    with (out/'qualification.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)
'''


TAU_SMOKE = '''import json, sys
from pathlib import Path
from haltiq_pair.strong_writer import StrongRuntime, MODEL, validate_health
from haltiq_pair.tau import parse_proposal
from haltiq_pair.team_training import StrandsHead
out, head_path = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), "Preserve earlier engineering qualifications"
with StrongRuntime(out, StrandsHead.from_json(head_path)) as (agents, gate):
    prompt = 'Public fact: Paris is the capital of France. Propose the next assistant text answering the capital. Return exactly JSON with content (nonempty string) and tool_calls ([]).'
    raw = agents.generate('writer', 'Propose only the next assistant action. No tool has executed.', prompt)
    proposal = parse_proposal(raw, set())
    review = agents.generate('reviewer', 'Use only the supplied public fact. Give concise feedback.', prompt + '\\nProposed action:\\n' + raw)
    decision = gate.decide('retrained_AND', json.dumps(proposal), None, 1, 6,
        'Is the next assistant action supported by the public fact?', ['Paris is the capital of France.'])
    metadata = json.loads((out/'runtime.json').read_text())
    validate_health(metadata['writer'])
    assert metadata['writer']['model'] == MODEL
    assert isinstance(decision['p_approve'], float)
    receipt = dict(status='passed', kind='invented engineering input, not official Tau task success',
        proposal=proposal, review=review, gate=decision, usage=agents.usage, runtime=metadata)
    with (out/'qualification.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)
'''


TAU_LONG_CONTEXT = '''import json, subprocess, sys, threading, time
from pathlib import Path
from transformers import AutoTokenizer
from openai import OpenAI
from haltiq_pair.core import Agents
from haltiq_pair.strong_writer import StrongRuntime, MODEL, REVISION, validate_health
from haltiq_pair.team_training import StrandsHead
out, head_path = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), "Preserve earlier engineering qualifications"
with StrongRuntime(out, StrandsHead.from_json(head_path)) as (agents, gate):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    system = 'This is an invented capacity test. Continue repeating only the word moss separated by spaces indefinitely. Do not finish voluntarily, explain, or add other words.'
    def prompt_for(count):
        return 'moss ' * count + '\\nContinue now with only moss repeated indefinitely, without stopping.'
    def token_count(prompt):
        return len(tokenizer.apply_chat_template([{'role':'system','content':system},
            {'role':'user','content':prompt}], tokenize=True, return_dict=False, add_generation_prompt=True))
    low, high = 1, 32000
    while low < high:
        middle = (low + high + 1)//2
        if token_count(prompt_for(middle)) <= 15300: low = middle
        else: high = middle - 1
    prompt = prompt_for(low)
    expected = token_count(prompt)
    assert expected == 15300, ('Exact stress-input length not reached', expected)
    samples, monitor_errors, done = [], [], threading.Event()
    def monitor():
        while not done.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True, timeout=5)
                samples.append(int(value.strip().splitlines()[0]))
            except Exception as error:
                monitor_errors.append(type(error).__name__)
            done.wait(.5)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        assert gate.critic is not None
        client = OpenAI(api_key='local', base_url='http://127.0.0.1:8001/v1', timeout=600, max_retries=0)
        response = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=768,
            messages=[{'role':'system','content':[{'type':'text','text':system}]},
                      {'role':'user','content':[{'type':'text','text':prompt}]}])
        actual = dict(prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,total_tokens=response.usage.total_tokens)
        assert actual['prompt_tokens'] == expected, ('Prompt was changed or truncated', actual, expected)
        assert actual['completion_tokens'] == 768, ('Full decode stress was not reached', actual)
        short = Agents(model_id=MODEL, max_tokens=64)
        short_output = short.generate('writer','Return only the capital name.',
            'Paris is the capital of France. What is its capital?')
        decision = gate.decide('retrained_AND','Paris',None,1,6,
            'What is the capital of France?',['France: Paris is its capital.'])
        assert isinstance(decision['p_approve'], float)
    finally:
        done.set()
        thread.join(timeout=10)
    metadata = json.loads((out/'runtime.json').read_text())
    validate_health(metadata['writer'])
    assert metadata['writer']['prefill_chunk_size'] == 512
    assert metadata['writer']['max_context'] == 16384
    assert samples and not monitor_errors
    receipt = dict(status='passed', kind='invented engineering qualification, not benchmark performance',
        expected_prompt_tokens=expected, actual_usage=actual, generation_cap=768,
        complete_prompt_preserved=True, full_decode_budget_exercised=True,
        critic_resident_during_writer=True, peak_sampled_total_gpu_mib=max(samples),
        peak_scope='sampled during stress inference, not model-loading peak',
        gpu_memory_samples=len(samples),monitor_errors=monitor_errors,
        seconds=time.monotonic()-started, stress_finish_reason=response.choices[0].finish_reason,
        stress_output=response.choices[0].message.content,post_stress_sdk_output=short_output,
        post_stress_gate=decision, runtime=metadata)
    with (out/'qualification.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2),flush=True)
print('YASH_7B_LONG_CONTEXT_QUALIFIED',flush=True)
'''


def engineering_cell(script, folder):
    return ("engineering_script = " + repr(script) + "\n" + f'''import signal
engineering_log = OUT/({folder!r} + "-console.log")
print("Engineering output is being preserved in", engineering_log, flush=True)
with engineering_log.open("x", encoding="utf-8") as log:
    process = subprocess.Popen([sys.executable, "-u", "-B", "-c", engineering_script,
        str(OUT/{folder!r}), str(HEAD)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, LOGURU_LEVEL="ERROR", TOKENIZERS_PARALLELISM="false"))
    try:
        process.wait()
    except BaseException:
        if process.poll() is None:
            # SIGINT gives the owned runtime its context-manager cleanup path.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log.flush()
        print(engineering_log.read_text(encoding="utf-8")[-10000:], flush=True)
        raise
print(engineering_log.read_text(encoding="utf-8")[-10000:], flush=True)
if process.returncode:
    raise RuntimeError("Engineering qualification failed; complete output preserved at " + str(engineering_log))
''')


EXPORT = '''import zipfile, hashlib
from google.colab import files
def create_evidence_zip(result_directory, archive_path):
    result_directory, archive_path = pathlib.Path(result_directory).resolve(), pathlib.Path(archive_path)
    assert not archive_path.resolve().is_relative_to(result_directory), "ZIP must be outside OUT"
    serving_receipts = {"hobson_config.json", "haltiq-serving-provenance.json"}
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as evidence:
        for path in sorted(result_directory.rglob("*")):
            relative = path.relative_to(result_directory)
            if not path.is_file() or "__pycache__" in relative.parts:
                continue
            if "critic-cache" in relative.parts and path.name not in serving_receipts:
                continue
            evidence.write(path, arcname=relative.as_posix())
    return archive_path
archive = create_evidence_zip(OUT, "/content/Yash_Team_Aligned_" + OUT.parent.name + "_" + RUN_ID + ".zip")
print("Evidence ZIP:", archive, "bytes:", archive.stat().st_size,
      "SHA256:", hashlib.sha256(archive.read_bytes()).hexdigest())
files.download(str(archive))
if not storage_metadata["persistent"]:
    print("Keep the downloaded ZIP: this OUT is temporary runtime storage.")
'''


def build(commit, benchmark, inputs):
    tau = benchmark == "Tau2"
    filename = FILENAMES["tau" if tau else "hotpot"]
    missing = source_missing(commit)
    draft = ("\n\n**DRAFT SOURCE PIN — DO NOT RUN YET.** The chosen commit lacks the new modules. "
             "Regenerate this notebook with the reviewed integration commit before allocation or testing."
             if missing else "")
    title = "Yash — " + ("Tau2 7B" if tau else "HotpotQA") + " — Team-Aligned Strands Classifier"
    introduction = (f"# {title}\n\nThis is a brand-new Yash notebook. Teammate notebooks and the earlier recorded panels remain unchanged."
        + draft + "\n\nThe primary head is **seed zero, balanced BCE on strict answer exact-match**, using the continuous Strands score. "
        "Its exact bytes and all cohort IDs are frozen below before fresh calls. This is a different objective from the matched PG retraining, "
        "which retained the immediate-stop failure and is reported separately. All five heads were frozen before cached test scoring, "
        "but the cached test80 had already been viewed and is exploratory, not a new blind evaluation. Fresh cohorts supply the prospective test."
        "\n\nNo credentials or paid APIs are needed. Select a GPU. Approve any required Colab presence/Drive checks yourself. "
        "Keep failures in the denominators; running the notebook is not a promise of positive results.")
    if tau:
        introduction += ("\n\nThis optional engineering variant uses pinned **Qwen2.5-7B, NF4 weights and FP16 computation**. "
            "Official Tau tools and evaluators remain unchanged; local customers/judges use the same 7B model. "
            "The QA-trained head is uncalibrated for retail actions, so these results are not leaderboard comparable. "
            "Two development tasks (15,62; eight episodes) stay separate from ten fresh evaluation tasks (forty episodes).")
    else:
        introduction += ("\n\nThe primary 1.5B study runs five policies independently online on twenty fresh questions first. "
            "An optional, separately frozen forty-question collection can follow; ten policy comparisons then replay those saved prefixes. "
            "The extension is off by default and must be chosen before fresh calls. Collection cost and replay estimates remain separate.")
    checks = ('test_modules = ["tests_pair.test_hotpot", "tests_pair.test_aligned_hotpot", "tests_pair.test_team_training"]\n'
              if not tau else
              'test_modules = ["tests_pair.test_strong_writer", "tests_pair.test_aligned_tau"]\n')
    checks += 'test_env = dict(os.environ, LOGURU_LEVEL="ERROR")\n'
    if tau:
        checks += 'test_env["HALTIQ_TAU_UPSTREAM"] = str(TAU_ROOT)\n'
    checks += 'checked(sys.executable, "-B", "-m", "unittest", *test_modules, "-q", cwd=ROOT, env=test_env)\n'
    checks += 'checked("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")\n'
    cells = [cell("markdown", introduction),
        cell("markdown", "## 1. Install the exact source\nThis checkout belongs to this new notebook. It does not edit a teammate's working files."),
        cell("code", bootstrap(commit, benchmark)),
        cell("markdown", "## 2. Create Yash's new result folder\nDrive is preferred; a failed mount falls back to temporary storage. Each Run all creates a fresh timestamp/UUID directory. Download the ZIP before disconnecting if persistent=False."),
        cell("code", storage(benchmark)),
        cell("markdown", "## 3. Freeze the exact head and public cohort files\nSHA256 checks preserve the original JSON bytes. Existing frozen files must match exactly. No training or threshold selection happens in this notebook."),
        cell("code", frozen_inputs(inputs, benchmark)),
        cell("markdown", "## 4. Run software controls\nThese tests use scripted fixtures and are not fresh benchmark results."),
        cell("code", checks),
        cell("markdown", "## 5. Qualify the actual GPU wiring\nThe invented public fact exercises AWS Strands writer/reviewer calls, output parsing and the pinned verifier. This is engineering evidence, not a benchmark score."),
        cell("code", engineering_cell(TAU_SMOKE if tau else HOTPot_SMOKE, "gpu-smoke-7b" if tau else "gpu-smoke-1.5b")),
    ]
    if tau:
        qualification = ('assert json.loads((OUT/"gpu-smoke-7b/qualification.json").read_text())["status"] == "passed"\n'
                         + engineering_cell(TAU_LONG_CONTEXT, "gpu-long-context-7b"))
        cells += [cell("markdown", "## 6. Qualify full-context capacity\nThis required invented stress input preserves exactly 15,300 prompt tokens and exercises 768 output tokens with the critic resident. GPU memory is sampled. An early EOS, missing samples or any error fails qualification; preserve that evidence and inspect it before continuing."),
                  cell("code", qualification)]
        smoke = RUN_HELPER + '''qualification = json.loads((OUT/"gpu-long-context-7b/qualification.json").read_text())
assert qualification["status"] == "passed" and qualification["full_decode_budget_exercised"]
TAU_SMOKE = OUT/"tau-development-2"
development = run_stage("tau-development-2", ["-m", "haltiq_pair.aligned_tau",
    "--head", str(HEAD), "--output", str(TAU_SMOKE), "--upstream-root", str(TAU_ROOT),
    "--phase", "smoke", "--limit", "2", "--seed", "43",
    "--taskids-manifest", str(FROZEN/"Yash-Tau-Smoke-IDs.json")], TAU_SMOKE)
'''
        full = '''assert json.loads((OUT/"tau-development-2/summary.json").read_text())["complete"]
assert json.loads((OUT/"gpu-long-context-7b/qualification.json").read_text())["status"] == "passed"
TAU_EVAL = OUT/"tau-evaluation-10"
evaluation = run_stage("tau-evaluation-10", ["-m", "haltiq_pair.aligned_tau",
    "--head", str(HEAD), "--output", str(TAU_EVAL), "--upstream-root", str(TAU_ROOT),
    "--phase", "evaluation", "--limit", "10", "--seed", "43",
    "--taskids-manifest", str(FROZEN/"Yash-Tau-Evaluation-IDs.json")], TAU_EVAL)
'''
        cells += [cell("markdown", "## 7. Run the separate official development smoke\nTwo public-hash-selected tasks × four policies. These eight attempts are development evidence and are excluded from the final ten-task panel. Invalid outputs may receive exactly one logged public-contract correction. Refused actions never execute."),
                  cell("code", smoke),
                  cell("markdown", "## 8. Run the fresh official evaluation\nTen different retail tasks × four policies. Keep all forty attempts, failures and refusals. Do not replace tasks based on their outcomes. Local-user/judge conditions and QA-to-retail transfer limits remain explicit."),
                  cell("code", full)]
    else:
        online = RUN_HELPER + '''assert json.loads((OUT/"gpu-smoke-1.5b/qualification.json").read_text())["status"] == "passed"
ONLINE = OUT/"hotpot-online-20"
online = run_stage("hotpot-online-20", ["-m", "haltiq_pair.aligned_hotpot",
    "--head", str(HEAD), "--cohort", str(FROZEN/"Yash-HotpotQA-Online20.json"),
    "--output", str(ONLINE), "--mode", "online"], ONLINE)
'''
        collection = '''assert json.loads((OUT/"hotpot-online-20/summary.json").read_text())["complete"]
assert json.loads((OUT/"notebook-plan.json").read_text())["run_extended_collection"] == RUN_EXTENDED_COLLECTION
if RUN_EXTENDED_COLLECTION:
    COLLECTION = OUT/"hotpot-collection-40"
    collection = run_stage("hotpot-collection-40", ["-m", "haltiq_pair.aligned_hotpot",
        "--head", str(HEAD), "--cohort", str(FROZEN/"Yash-HotpotQA-Cohort40.json"),
        "--output", str(COLLECTION), "--mode", "collect"], COLLECTION)
else:
    print("Optional collection40 is disabled in the pre-call plan; no extension model calls made.")
'''
        cells += [cell("markdown", "## 6. Run the primary twenty-question online study\nFive predeclared policies make independent fresh calls: Fixed1, Fixed6, Strands, retrained head and retrained head with best-seen answer selection. These one hundred planned episodes run first. Return-at-cap is recorded separately from approval."),
                  cell("code", online),
                  cell("markdown", "## 7. Optional forty-trajectory extension\nThis is disabled by default. Set RUN_EXTENDED_COLLECTION=True in the frozen-input cell before any fresh model calls if runtime is available. The choice is frozen in notebook-plan.json. A complete extension collects six writer drafts, five reviews and six verifier scores per question, then replays ten policies. Keep its full physical cost and counterfactual prefixes separate from the primary online denominator."),
                  cell("code", collection)]
    cells += [cell("markdown", "## Export Yash's complete evidence\nThe ZIP keeps frozen inputs, manifests, exact runtime receipts, primary response JSONs, traces, outcomes, summaries and logs. Critic-cache weights and Python bytecode are excluded without deleting them. Preserve incomplete and negative runs. The ZIP is created outside OUT and never overwrites another archive."),
              cell("code", EXPORT)]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": filename, "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU", "gpuClass": "standard",
        "haltiq_source_commit": commit, "haltiq_source_ready": not missing,
        "haltiq_missing_modules": missing, "haltiq_head_sha256": HEAD_SHA256}, "cells": cells}


def validate(notebook, inputs):
    embedded_seen = False
    ids = []
    for item in notebook["cells"]:
        ids.append(item["id"])
        if item["cell_type"] != "code":
            continue
        source = "".join(item["source"])
        compile(source, "new Yash notebook cell", "exec")
        tree = ast.parse(source)
        for statement in tree.body:
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "EMBEDDED_FILES"):
                embedded = ast.literal_eval(statement.value)
                for name, value in embedded.items():
                    assert value == inputs[name]
                    assert digest(value["text"].encode("utf-8")) == value["sha256"]
                embedded_seen = True
    assert embedded_seen and len(ids) == len(set(ids)), "Exact payload or stable cell IDs invalid"
    for script in (HOTPot_SMOKE, TAU_SMOKE, TAU_LONG_CONTEXT):
        compile(script, "embedded engineering script", "exec")
    try:
        import nbformat
    except ImportError:
        pass
    else:
        nbformat.validate(notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be an immutable full SHA")
    inputs = load_inputs(args.artifact_root)
    output = REPO/"notebooks"
    output.mkdir(exist_ok=True)
    for benchmark in ("HotpotQA", "Tau2"):
        notebook = build(args.commit, benchmark, inputs)
        validate(notebook, inputs)
        path = output/FILENAMES["tau" if benchmark == "Tau2" else "hotpot"]
        path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8")) == notebook
        print(json.dumps({"path": str(path), "sha256": digest(path.read_bytes()),
            "cells": len(notebook["cells"]), "source_ready": notebook["metadata"]["haltiq_source_ready"],
            "head_sha256": HEAD_SHA256}))


if __name__ == "__main__":
    main()
