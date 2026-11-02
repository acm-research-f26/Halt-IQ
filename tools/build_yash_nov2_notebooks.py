"""Build brand-new manual Colabs from an already reviewed full Git commit.

Only committed Python/source requirements and the committed frozen public-ID
plan are embedded. No working-tree source, teammate trajectories, model weights,
credentials or benchmark outcomes enter these prepared/unrun notebooks.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
PLAN_PATH = "data/Yash_Nov2_HotpotQA_Plan.json"
REQUIRED = {"haltiq_pair/nov2_orchestration.py", "haltiq_pair/nov2_policy.py",
            "haltiq_pair/nov2_protocol.py", "haltiq_pair/nov2_runner.py",
            "haltiq_pair/nov2_runtime.py", "pyproject.toml", "requirements-colab.txt",
            "requirements-nov2-colab.txt"}
FILENAMES = {"Primary": "Yash_HotpotQA_Orchestration_Primary_20261102.ipynb",
             "Replication": "Yash_HotpotQA_Orchestration_Replication_20261102.ipynb"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO)


def _safe_name(name):
    path = PurePosixPath(name)
    return bool(name and "\\" not in name and not path.is_absolute()
                and all(part not in {"", ".", ".."} for part in path.parts)
                and (re.fullmatch(r"haltiq_pair/[A-Za-z0-9_]+\.py", name)
                     or name in {"pyproject.toml", "requirements-colab.txt", "requirements-nov2-colab.txt"}))


def verify_payload(source_raw, manifest, plan_raw):
    if not isinstance(manifest, dict) or not REQUIRED <= set(manifest) or not all(_safe_name(name) for name in manifest):
        raise ValueError("Embedded source paths differ from the declared code-only bundle")
    with zipfile.ZipFile(io.BytesIO(source_raw)) as archive:
        members = archive.infolist()
        if len(members) != len(manifest) or {member.filename for member in members} != set(manifest):
            raise ValueError("Source ZIP has missing, extra or duplicate members")
        for member in members:
            if member.is_dir() or member.file_size > 2_000_000 or (member.external_attr >> 16) & 0o170000 not in {0, 0o100000}:
                raise ValueError("Only bounded regular source files are allowed")
            raw = archive.read(member)
            if manifest[member.filename] != {"sha256": digest(raw), "bytes": len(raw)}:
                raise ValueError("Source member hash or size differs")
    plan = json.loads(plan_raw)
    if (plan.get("schema") != "yash-nov2-plan-v1"
            or plan.get("dataset", {}).get("revision") != "1908d6afbbead072334abe2965f91bd2709910ab"):
        raise ValueError("The plan does not identify the pinned HotpotQA protocol")
    partitions = plan.get("partitions", {})
    expected = {"train": 20, "dev": 10, "test": 20}
    if set(partitions) != set(expected):
        raise ValueError("Exactly the predeclared train20/dev10/test20 partitions are required")
    seen = set(plan.get("excluded_ids", []))
    for name, count in expected.items():
        ids = partitions[name].get("question_ids", [])
        if len(ids) != count or len(set(ids)) != count or set(ids) & seen:
            raise ValueError("Question groups duplicate or overlap the frozen exclusions")
        seen.update(ids)
    design = plan.get("study_design", {})
    if (design.get("primary_seed") != 0 or design.get("replication_seed") != 1
            or design.get("same_20_test_questions_both_runs") is not True
            or design.get("independent_questions") != 20):
        raise ValueError("Both notebooks must retain the same twenty-question paired design")
    return plan


def committed_payload(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("An immutable full forty-character commit SHA is required")
    resolved = git("rev-parse", "--verify", commit + "^{commit}").decode().strip()
    if resolved != commit:
        raise ValueError("The source identity is not the exact declared commit")
    names = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    selected = sorted(name for name in names if _safe_name(name))
    if not REQUIRED <= set(selected) or PLAN_PATH not in names:
        raise ValueError("The committed checkpoint lacks the reviewed November source or plan")
    raw_files = {name: git("show", commit + ":" + name) for name in selected}
    manifest = {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in raw_files.items()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in raw_files.items():
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, raw)
    source_raw, plan_raw = stream.getvalue(), git("show", commit + ":" + PLAN_PATH)
    verify_payload(source_raw, manifest, plan_raw)
    return source_raw, manifest, plan_raw


def cell(kind, source):
    value = {"cell_type": kind, "id": "yash-" + digest(source.encode())[:16],
             "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


BOOTSTRAP = r'''
import base64, hashlib, io, json, os, pathlib, re, subprocess, sys, uuid, zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
DEADLINE = datetime(2026, 11, 2, 8, 50, tzinfo=ZoneInfo("America/Chicago"))
def remaining_seconds():
    return (DEADLINE - datetime.now(ZoneInfo("America/Chicago"))).total_seconds()
if remaining_seconds() <= 0:
    raise RuntimeError("NO RUN: the frozen November 2 08:50 Central stop time has passed. No inference was started.")
assert (3, 12) <= sys.version_info[:2] < (3, 14), "Use a Python 3.12/3.13 Colab runtime"
assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT)
SOURCE_RAW = base64.b64decode(EMBEDDED_SOURCE_B64, validate=True)
PLAN_RAW = base64.b64decode(EMBEDDED_PLAN_B64, validate=True)
assert hashlib.sha256(SOURCE_RAW).hexdigest() == EMBEDDED_SOURCE_SHA256, "Embedded source ZIP changed"
assert hashlib.sha256(PLAN_RAW).hexdigest() == PLAN_SHA256, "Embedded plan changed"
verified_source = {}
with zipfile.ZipFile(io.BytesIO(SOURCE_RAW)) as bundle:
    members = bundle.infolist()
    assert len(members) == len(SOURCE_MANIFEST) and {m.filename for m in members} == set(SOURCE_MANIFEST)
    for member in members:
        name = member.filename
        parts = pathlib.PurePosixPath(name)
        assert not parts.is_absolute() and "\\" not in name and all(p not in {"", ".", ".."} for p in parts.parts)
        assert re.fullmatch(r"haltiq_pair/[A-Za-z0-9_]+\.py", name) or name in {"pyproject.toml", "requirements-colab.txt", "requirements-nov2-colab.txt"}
        assert not member.is_dir() and member.file_size <= 2000000
        assert (member.external_attr >> 16) & 0o170000 in {0, 0o100000}, "Nonregular bundle member"
        raw = bundle.read(member)
        assert SOURCE_MANIFEST[name] == dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        verified_source[name] = raw
plan = json.loads(PLAN_RAW)
assert {name:len(value["question_ids"]) for name,value in plan["partitions"].items()} == {"train":20,"dev":10,"test":20}
assert plan["study_design"]["same_20_test_questions_both_runs"] is True
BOOT_ID = datetime.now(ZoneInfo("America/Chicago")).strftime("%H%M%S") + "-" + uuid.uuid4().hex[:8]
ROOT = pathlib.Path("/content/Yash_HaltIQ_Orchestration_" + ARM + "_20261102_" + BOOT_ID)
ROOT.mkdir(parents=False, exist_ok=False)  # Never reuse a previous notebook's files.
for name, raw in verified_source.items():
    destination = ROOT/name
    assert destination.resolve().is_relative_to(ROOT.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
PLAN = ROOT/"data/Yash_Nov2_HotpotQA_Plan.json"
PLAN.parent.mkdir()
with PLAN.open("xb") as stream:
    stream.write(PLAN_RAW)
with (ROOT/"Yash-Source-Manifest.json").open("x", encoding="utf-8") as stream:
    json.dump(dict(source_commit=SOURCE_COMMIT, source_zip_sha256=EMBEDDED_SOURCE_SHA256,
        files=SOURCE_MANIFEST, plan_sha256=PLAN_SHA256, prepared_notebook_has_no_results=True), stream, indent=2)
def checked(*arguments):
    subprocess.run(list(arguments), check=True, cwd=ROOT)
checked(sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT/"requirements-nov2-colab.txt"),
        "Pillow>=11,<13", "matplotlib>=3.9,<4")
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
try:
    torchao_version = version("torchao")
except PackageNotFoundError:
    torchao_version = None
if torchao_version is not None and Version(torchao_version) < Version("0.16"):
    print("Removing unused incompatible TorchAO", torchao_version)
    checked(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")
checked(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT))
PINNED_PACKAGES = {"strands-agents":"1.62.0", "strands-decider":"0.2.0", "transformers":"5.17.0",
                   "peft":"0.21.0", "datasets":"5.1.0", "fastapi":"0.162.0", "uvicorn":"0.54.0",
                   "lm-format-enforcer":"0.11.3"}
for package, expected in PINNED_PACKAGES.items():
    assert version(package) == expected, ("Installed package differs from frozen requirements", package, version(package), expected)
# A new interpreter avoids trusting a notebook kernel's previously imported packages.
checked(sys.executable, "-B", "-c", "import strands, strands_decider, transformers, peft, datasets, fastapi, uvicorn, torch, lmformatenforcer; from haltiq_pair.nov2_runtime import constraint_dependencies; constraint_dependencies(); print('Fresh-child dependency import smoke passed; torch=' + torch.__version__)")
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU under Runtime > Change runtime type. No model inference was started.")
import PIL, matplotlib
print("Verified code-only source:", SOURCE_COMMIT)
print("Plan SHA256:", PLAN_SHA256, "Pillow:", PIL.__version__, "matplotlib:", matplotlib.__version__)
print("GPU:", torch.cuda.get_device_name(), "remaining seconds before 08:50 Central:", int(remaining_seconds()))
'''


STORAGE = r'''
import uuid
USE_DRIVE = True  # Optional manual connection; choose False for runtime storage and download the ZIP.
storage_root = ROOT/"results"
backend, drive_error = "runtime", None
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        storage_root = pathlib.Path("/content/drive/MyDrive/Yash_HaltIQ_Nov2")
        backend = "google_drive"
    except Exception as error:
        drive_error = type(error).__name__
        print("Drive was not connected. Results will be temporary until you download the evidence ZIP.")
RUN_ID = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
OUT = storage_root/(RUN_ID + "_" + ARM)
OUT.mkdir(parents=True, exist_ok=False)
STUDY = OUT/"study"
(OUT/"frozen").mkdir()
for name, raw in {"Yash-Nov2-Source.zip":SOURCE_RAW,"Yash_Nov2_HotpotQA_Plan.json":PLAN_RAW}.items():
    with (OUT/"frozen"/name).open("xb") as stream:
        stream.write(raw)
storage = dict(backend=backend, persistent=backend=="google_drive", drive_mount_error_type=drive_error,
    source_commit=SOURCE_COMMIT, source_zip_sha256=EMBEDDED_SOURCE_SHA256, plan_sha256=PLAN_SHA256,
    deployment_seed=DEPLOYMENT_SEED, role=ARM, same_20_test_questions_both_notebooks=True,
    independent_test_questions=20, prepared_notebook_has_no_results=True,
    deadline_central=DEADLINE.isoformat(), result_directory=str(OUT), no_prior_files_overwritten=True)
with (OUT/"notebook-storage.json").open("x", encoding="utf-8") as stream:
    json.dump(storage, stream, indent=2)
with (OUT/"environment.txt").open("x", encoding="utf-8") as stream:
    stream.write(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
print("New Yash output:", OUT)
print("Seed", DEPLOYMENT_SEED, "is predeclared; all five heads will be fitted and frozen before test.")
'''


RUN_AND_EXPORT = r'''
import signal, time, traceback
from IPython.display import display, Image, Markdown
from google.colab import files

def read_status_text(path, monitor):
    # Drive can remove/replace a status file between filesystem operations.
    # This optional observation must never decide whether the owned job lives.
    monitor["poll_attempts"] += 1
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        name = type(error).__name__
        monitor["read_errors"] += 1
        monitor["errors_by_type"][name] = monitor["errors_by_type"].get(name, 0) + 1
        monitor["last_error_type"] = name
        if monitor["warnings_emitted"] < 5:
            monitor["warnings_emitted"] += 1
            try:
                print("Optional status read unavailable; owned job continues:", name,
                      "read errors:", monitor["read_errors"], flush=True)
            except OSError:
                monitor["warning_output_errors"] += 1
        else:
            monitor["warnings_suppressed"] += 1
        return None

def interrupt_owned(child):
    if child.poll() is not None:
        return "already_exited"
    child.send_signal(signal.SIGINT)  # Runtime context closes its own writer/critic services.
    try:
        child.wait(timeout=60)
        return "sigint_exited"
    except subprocess.TimeoutExpired:
        child.terminate()  # Only this Popen handle is eligible for escalation.
        try:
            child.wait(timeout=10)
            return "owned_child_terminated_after_sigint_timeout"
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
            return "owned_child_killed_after_sigint_timeout"

def export_evidence():
    archive = OUT.parent/("Yash_HotpotQA_Orchestration_" + ARM + "_" + RUN_ID + ".zip")
    assert not archive.resolve().is_relative_to(OUT.resolve())
    excluded_extensions = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".pkl", ".pickle"}
    manifest = []
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as evidence:
        for path in sorted(OUT.rglob("*")):
            relative = path.relative_to(OUT)
            if not path.is_file() or any(part in {"critic-cache","__pycache__",".git"} for part in relative.parts):
                continue
            if path.suffix.lower() in excluded_extensions:
                continue
            raw = path.read_bytes()
            manifest.append(dict(path=relative.as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            evidence.writestr(relative.as_posix(),raw)
        evidence.writestr("Yash-Archive-Manifest.json",json.dumps(dict(files=manifest,
            exclusions="model weights, critic-cache, Python bytecode; no result/failure filtering"),indent=2))
    print("Evidence archive:",archive,"bytes:",archive.stat().st_size,
          "SHA256:",hashlib.sha256(archive.read_bytes()).hexdigest())
    files.download(str(archive))
    return archive

process = None
monitor_counts = dict(poll_attempts=0, read_errors=0, errors_by_type={},
    warnings_emitted=0, warnings_suppressed=0, warning_output_errors=0, last_error_type=None)
run_record = dict(status="not_started",source_commit=SOURCE_COMMIT,seed=DEPLOYMENT_SEED,
                  deadline_central=DEADLINE.isoformat(),no_positive_outcome_promised=True,
                  monitor=monitor_counts)
console = OUT/"study-console.log"
try:
    remaining = remaining_seconds()
    if remaining <= 0:
        raise RuntimeError("NO RUN: frozen 08:50 Central stop time has passed; no study inference started.")
    allowance = min(27000.,remaining)
    hard_stop_at = time.monotonic() + allowance  # Includes runtime startup and qualification.
    arguments = [sys.executable,"-u","-B","-m","haltiq_pair.nov2_runner",
        "--plan",str(PLAN),"--output",str(STUDY),"--seed",str(DEPLOYMENT_SEED),
        "--timeout-seconds",str(allowance)]
    run_record.update(status="started",arguments=arguments,started_central=datetime.now(ZoneInfo("America/Chicago")).isoformat())
    with console.open("x",encoding="utf-8") as log:
        process = subprocess.Popen(arguments,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONPATH=str(ROOT),TOKENIZERS_PARALLELISM="false",LOGURU_LEVEL="ERROR"))
        previous, last_print = None, 0.
        while process.poll() is None:
            if remaining_seconds() <= 0 or time.monotonic() >= hard_stop_at:
                run_record.update(status="deadline_or_27000_second_stop_requested",cleanup=interrupt_owned(process))
                break
            status_path = STUDY/"status.json"
            current = read_status_text(status_path, monitor_counts)
            if current is not None and current != previous:
                print("Progress:",current.strip(),flush=True)
                previous = current
                last_print = time.monotonic()
            if time.monotonic()-last_print >= 30:
                print("Running; seconds before frozen stop:",max(0,int(remaining_seconds())),"log:",console,flush=True)
                last_print = time.monotonic()
            time.sleep(5)
        run_record["returncode"] = process.returncode
        if run_record["status"] == "started":
            run_record["status"] = "process_exited" if process.returncode==0 else "process_failed_or_partial"
except BaseException as error:
    run_record.update(status="interrupted_or_failed",error_type=type(error).__name__,error=str(error))
    if process is not None and process.poll() is None:
        run_record["cleanup"] = interrupt_owned(process)
        run_record["returncode"] = process.returncode
    print(traceback.format_exc())
finally:
    run_record["finished_central"] = datetime.now(ZoneInfo("America/Chicago")).isoformat()
    with (OUT/"notebook-run-receipt.json").open("x",encoding="utf-8") as stream:
        json.dump(run_record,stream,indent=2)
    if console.exists():
        print("Console tail:\n"+console.read_text(encoding="utf-8")[-12000:])
    summary_path = STUDY/"summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("Recorded study status:",summary.get("study_status"),"complete:",summary.get("complete"))
    else:
        print("No final study summary exists. Preserve the setup/failure receipts; benchmark completion is not claimed.")
    report = STUDY/"Yash-Nov2-Study-Report.txt"
    if report.exists():
        print(report.read_text(encoding="utf-8"))
    results = STUDY/"results.csv"
    if results.exists():
        import pandas as pd
        display(pd.read_csv(results))
    figure = STUDY/"Yash-Nov2-Answer-F1.png"
    if figure.exists():
        display(Image(filename=str(figure)))
    ARCHIVE = export_evidence()
    print("Failures, interrupted and pending attempts remain in the recorded denominators.")
    print("The replication repeats the same twenty test questions; it does not create forty independent questions.")
'''


def build(commit, arm, source_raw, manifest, plan_raw):
    if arm not in FILENAMES:
        raise ValueError("Choose Primary or Replication")
    verify_payload(source_raw, manifest, plan_raw)
    seed = 0 if arm == "Primary" else 1
    literals = "\n".join(f"{key} = {value!r}" for key, value in {
        "SOURCE_COMMIT": commit, "ARM": arm.lower(), "DEPLOYMENT_SEED": seed,
        "EMBEDDED_SOURCE_B64": base64.b64encode(source_raw).decode(),
        "EMBEDDED_SOURCE_SHA256": digest(source_raw), "SOURCE_MANIFEST": manifest,
        "EMBEDDED_PLAN_B64": base64.b64encode(plan_raw).decode(), "PLAN_SHA256": digest(plan_raw)}.items()) + "\n"
    introduction = (f"# Yash — HotpotQA learned orchestration — {arm}\n\n"
        "**Prepared and unrun.** This new notebook contains no benchmark results. Select a GPU and use **Runtime → Run all** to start it manually. "
        "No teammate files or earlier notebooks are edited. No paid API key is required; pinned public model/dataset downloads need internet.\n\n"
        f"This run predeclares **deployment seed {seed}**. Both notebooks use the same train20/dev10/test20 whole-question partitions. "
        "They provide twenty independent held-out questions with a second run on those same questions, not forty independent questions. "
        "All five linear action heads are fitted and selected on dev utility before any test call. No favorable outcome is promised.\n\n"
        "The actual AWS Strands SDK writer and reviewer interact with an optional third evidence challenger. Its response reaches the next writer. "
        "The Hobson verifier supplies a separate public score. A new supervised linear action policy chooses STOP, REVISE or INVOKE; "
        "this is a DVidE-inspired transfer proposal, not a reproduction of the video paper or an RL claim.\n\n"
        "Seven arms run on the same frozen test20: Fixed1; legacy combined six-draft control; matched Strands; matched binary AND; "
        "always revise to three drafts; always invoke the challenger once; learned three-action routing. Matched controls share draft/call caps, "
        "model weights, role output caps, formatting policy and last-draft answer selection. Legacy6 is a secondary control with a different budget. "
        "One optional best-prefix diagnostic stays separate from the primary results.\n\n"
        "The notebook stops requesting work by **November 2, 2026, 8:50 AM America/Chicago** and exports partial evidence. "
        "After that time it refuses a new run. Installation or GPU capacity may prevent completion; failed and pending attempts remain visible.")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": FILENAMES[arm], "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU", "gpuClass": "standard",
        "yash_status": "prepared_unrun", "yash_source_commit": commit,
        "yash_source_zip_sha256": digest(source_raw), "yash_plan_sha256": digest(plan_raw),
        "yash_deployment_seed": seed, "yash_unique_test_questions": 20},
        "cells": [cell("markdown", introduction),
            cell("markdown", "## 1. Verify and install embedded source\nEvery source member and the exact plan are checked before writing a fresh Yash folder. There is no GitHub login or working-tree dependency."),
            cell("code", literals + BOOTSTRAP),
            cell("markdown", "## 2. Create a fresh result folder\nDrive is optional and requires your manual connection. Connected runs save checkpoints, raw receipts and results under MyDrive/Yash_HaltIQ_Nov2. Each run has a new timestamp and UUID; previous outputs are never overwritten."),
            cell("code", STORAGE),
            cell("markdown", "## 3. Run the complete study and export evidence\nThe child process first qualifies all three roles on an invented fact, collects actual train/dev action branches, fits and freezes all five heads, then runs the seven matched/secondary test arms. The console and status files track progress. Interruption or the deadline triggers SIGINT to the owned child, a wait for its cleanup, and a ZIP export. No existing service or unrelated process is stopped. Model weights and critic-cache are excluded from the archive. Actual reports, CSVs and the figure appear only after execution."),
            cell("code", RUN_AND_EXPORT)]}
    validate(notebook)
    return notebook


def validate(notebook):
    ids, literal_values = [], {}
    wanted = {"SOURCE_COMMIT", "ARM", "DEPLOYMENT_SEED", "EMBEDDED_SOURCE_B64",
              "EMBEDDED_SOURCE_SHA256", "SOURCE_MANIFEST", "EMBEDDED_PLAN_B64", "PLAN_SHA256"}
    for item in notebook["cells"]:
        ids.append(item["id"])
        if item["cell_type"] != "code":
            continue
        if item["execution_count"] is not None or item["outputs"]:
            raise ValueError("Prepared notebooks must contain no execution or output claims")
        source = "".join(item["source"])
        compile(source, "new Yash notebook cell", "exec")
        for statement in ast.parse(source).body:
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in wanted):
                literal_values[statement.targets[0].id] = ast.literal_eval(statement.value)
    if len(ids) != len(set(ids)) or set(literal_values) != wanted:
        raise ValueError("Notebook cell IDs or embedded literals are incomplete")
    source_raw = base64.b64decode(literal_values["EMBEDDED_SOURCE_B64"], validate=True)
    plan_raw = base64.b64decode(literal_values["EMBEDDED_PLAN_B64"], validate=True)
    if (digest(source_raw) != literal_values["EMBEDDED_SOURCE_SHA256"]
            or digest(plan_raw) != literal_values["PLAN_SHA256"]
            or not re.fullmatch(r"[0-9a-f]{40}", literal_values["SOURCE_COMMIT"])):
        raise ValueError("The notebook's source or plan checksum differs")
    verify_payload(source_raw, literal_values["SOURCE_MANIFEST"], plan_raw)
    try:
        import nbformat
    except ImportError:
        pass
    else:
        nbformat.validate(notebook)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "notebooks")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        source_raw, manifest, plan_raw = committed_payload(args.commit)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    notebooks = {arm: build(args.commit, arm, source_raw, manifest, plan_raw) for arm in FILENAMES}
    if not args.validate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if any((args.output_dir / name).exists() for name in FILENAMES.values()):
            raise FileExistsError("Brand-new notebook filenames already exist; preserve them rather than overwrite")
    for arm, notebook in notebooks.items():
        raw = (json.dumps(notebook, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        path = args.output_dir / FILENAMES[arm]
        if not args.validate_only:
            with path.open("xb") as stream:
                stream.write(raw)
            validate(json.loads(path.read_bytes()))
        print(json.dumps({"path": str(path), "written": not args.validate_only,
            "sha256": digest(raw), "source_commit": args.commit, "source_zip_sha256": digest(source_raw),
            "plan_sha256": digest(plan_raw), "deployment_seed": 0 if arm == "Primary" else 1,
            "status": "prepared_unrun", "unique_test_questions": 20, "cells": len(notebook["cells"])}))


if __name__ == "__main__":
    main()
