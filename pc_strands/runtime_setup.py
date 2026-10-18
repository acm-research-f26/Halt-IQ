"""Serialized, resumable public downloads with immutable integrity receipts.

The download queue is data in an external runtime directory. The optional install
watcher installs verified local wheels only, never downloads or loads a model.
A PAUSE file stops current transfer/new installation admission; remove it only for
an explicit operator resume. Keep the downloader as sole bulk network owner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination(root: Path, relative: str):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise ValueError("Download destination escapes runtime root")
    return candidate


def available_ram_bytes():
    """Read Windows available physical memory without importing a monitor stack."""
    import ctypes
    import struct
    memory = ctypes.create_string_buffer(64)
    struct.pack_into("<I", memory, 0, 64)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(memory):
        raise OSError("Windows available-memory probe failed")
    return struct.unpack_from("<Q", memory, 16)[0]


HEALTH_GPU_LEASE_SOURCE = '''from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json, msvcrt, os, time

@contextmanager
def health_gpu_lease(path, deadline):
    """Same OS byte-zero lease as study workers; metadata is not ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    locked = False
    record = None
    try:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        while True:
            if time.time() >= deadline:
                raise TimeoutError('CUDA health could not acquire shared GPU lease before its deadline')
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
                break
            except OSError:
                time.sleep(min(.5, max(0, deadline - time.time())))
        record = {'path': str(path.resolve()), 'holder_pid': os.getpid(),
                  'protocol': 'Windows nonblocking OS byte-zero lock',
                  'acquired_at': datetime.now(timezone.utc).isoformat(),
                  'held_during_probe': True, 'released': False}
        handle.seek(1)
        handle.truncate()
        handle.write(json.dumps({'pid': os.getpid(), 'purpose': 'runtime-cuda-health',
                                 'acquired_at': record['acquired_at']}).encode())
        handle.flush()
        yield record
    finally:
        try:
            if locked:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()
            if record is not None:
                record.update(released=True, released_at=datetime.now(timezone.utc).isoformat())
'''


def cuda_health_source(root: Path, deadline: float):
    """Self-contained child: shared ownership precedes every heavy import."""
    bindings = f"\nruntime_root=Path({str(root.resolve())!r})\nhealth_deadline={deadline!r}\n"
    return HEALTH_GPU_LEASE_SOURCE + bindings + '''
with health_gpu_lease(runtime_root / 'gpu-owner.lock', health_deadline) as lease:
    # Recheck admission after waiting for the lease; no Torch import came first.
    import ctypes, struct
    memory = ctypes.create_string_buffer(64)
    struct.pack_into('<I', memory, 0, 64)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(memory):
        raise OSError('Windows available-memory probe failed')
    if not (runtime_root / 'ALLOW-CUDA-HEALTH').is_file() or (runtime_root / 'PAUSE').exists():
        raise RuntimeError('CUDA health ownership approval was removed or setup paused')
    if struct.unpack_from('<Q', memory, 16)[0] < 8*1024**3:
        raise RuntimeError('CUDA health requires 8GiB available RAM after acquiring ownership')
    import importlib.metadata as m, platform
    import torch, numpy as np, transformers, peft, strands_decider.modeling
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'packages':{x:m.version(x) for x in ['transformers','peft','accelerate','strands-decider','numpy']}}
    result['gpu_lease'] = lease
    assert torch.__version__=='2.7.1+cu128'
    assert result['cuda_available']
    assert torch.from_numpy(np.array([1,2],dtype=np.float32)).sum().item()==3
    result['gpu']=torch.cuda.get_device_name(0)
    result['capability']=torch.cuda.get_device_capability(0)
    result['cuda_free_bytes'],result['cuda_total_bytes']=torch.cuda.mem_get_info()
    if result['cuda_free_bytes']>=3*1024**3:
        t=torch.arange(8,device='cuda',dtype=torch.float32)
        assert (t*t).sum().item()==140
        b=torch.ones((2,2),device='cuda',dtype=torch.bfloat16)
        assert (b@b).float().sum().item()==8
        torch.cuda.synchronize()
        result['cuda_kernel_checks']='passed: float32 arithmetic and bf16 matmul; no model loaded'
    else:
        result['cuda_kernel_checks']='deferred: less than 3GiB free; coordinate GPU owner'
    result['model_inference']=False
print(json.dumps(result))
'''


def validate_resume(url: str, offset: int, size: int):
    """Require exact 206 for resume; fresh immutable files may return 200."""
    request = Request(url, headers={"Range": f"bytes={offset}-{offset}", "User-Agent": "HaltIQ-runtime-setup/1"})
    with urlopen(request, timeout=30) as response:
        if offset == 0 and response.status == 200:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != size:
                raise ValueError("Fresh response length differs from pinned size")
            return {"offset": 0, "size": size, "http_status": 200, "resume": False, "checked_at": now()}
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or content_range != f"bytes {offset}-{offset}/{size}":
            raise ValueError("Server did not validate exact HTTP range resume")
        if len(response.read(2)) != 1:
            raise ValueError("Range probe did not return exactly one byte")
    return {"offset": offset, "size": size, "http_status": 206, "checked_at": now()}


def run(root: Path, queue: Path, rate: int):
    root.mkdir(parents=True, exist_ok=True)
    # Windows byte-range lock is released by the OS even if this helper crashes.
    import msvcrt
    with (root / "download.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        _download_locked(root, queue, rate)


def _download_locked(root: Path, queue: Path, rate: int):
    status_path = root / "download-status.json"
    receipt_path = root / "download-receipts.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for relative, receipt in receipts.items():
        completed = destination(root, relative)
        if not completed.is_file() or completed.stat().st_size != receipt["size"] or sha256(completed) != receipt["sha256"]:
            raise ValueError("Previously verified runtime file changed or disappeared")
    status = {"helper_pid": os.getpid(), "started_at": now(), "rate_bytes_per_second": rate}
    active = None
    try:
        while True:
            queue_data = json.loads(queue.read_text(encoding="utf-8"))
            tasks = queue_data["files"]
            deadline = datetime.fromisoformat(queue_data["deadline_utc"]).timestamp() if queue_data.get("deadline_utc") else float("inf")
            if time.time() >= deadline:
                raise TimeoutError("Runtime setup deadline reached; partial files preserved")
            pending = [task for task in tasks if task["path"] not in receipts]
            if not pending:
                status.update(state="completed", updated_at=now(), completed_files=len(receipts), curl_pid=None)
                atomic_json(status_path, status)
                return
            task = pending[0]
            parsed = urlsplit(task["url"])
            if parsed.scheme != "https" or parsed.hostname not in {"download.pytorch.org", "download-r2.pytorch.org", "huggingface.co", "files.pythonhosted.org"} or parsed.query or parsed.username:
                raise ValueError("Only allowlisted public HTTPS URLs without credentials/query")
            expected = task["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("A pinned SHA256 is required before download")
            final = destination(root, task["path"])
            partial = final.with_name(final.name + ".part")
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                if final.stat().st_size != task["size"] or sha256(final) != expected:
                    raise ValueError("Existing final file does not match pinned integrity")
            else:
                while True:
                    if (root / "PAUSE").exists():
                        status.update(state="paused", updated_at=now(), file=task["path"], curl_pid=None)
                        atomic_json(status_path, status)
                        time.sleep(2)
                        if time.time() >= deadline:
                            raise TimeoutError("Runtime setup deadline reached while paused")
                        continue
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset > task["size"]:
                        raise ValueError("Partial download exceeds pinned size")
                    if offset == task["size"]:
                        break
                    probe = validate_resume(task["url"], offset, task["size"])
                    command = ["curl.exe", "--silent", "--show-error", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https", "--limit-rate", str(rate), "--continue-at", "-", "--connect-timeout", "30", "--speed-limit", "16384", "--speed-time", "120", "--output", str(partial), task["url"]]
                    active = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    status.update(state="downloading", file=task["path"], expected_bytes=task["size"], curl_pid=active.pid, public_url=task["url"], range_probe=probe)
                    next_report = 0
                    while active.poll() is None:
                        if time.time() >= deadline:
                            raise TimeoutError("Runtime setup deadline reached during transfer")
                        if (root / "PAUSE").exists():
                            active.terminate()
                            active.wait(timeout=10)
                            break
                        if time.monotonic() >= next_report:
                            status.update(updated_at=now(), partial_bytes=partial.stat().st_size if partial.exists() else 0)
                            atomic_json(status_path, status)
                            next_report = time.monotonic() + 30
                        time.sleep(1)
                    result = active.returncode
                    active = None
                    if (root / "PAUSE").exists():
                        continue
                    if result:
                        raise RuntimeError(f"curl transfer failed with exit {result}; partial preserved, no automatic retry")
                    break
                status.update(state="verifying", updated_at=now(), curl_pid=None)
                atomic_json(status_path, status)
                if partial.stat().st_size != task["size"] or sha256(partial) != expected:
                    raise ValueError("Downloaded size/SHA256 mismatch; partial not promoted")
                partial.replace(final)
            receipts[task["path"]] = {"public_url": task["url"], "size": final.stat().st_size, "sha256": expected, "verified_at": now()}
            atomic_json(receipt_path, receipts)
    except BaseException as error:
        if active is not None and active.poll() is None:
            active.terminate()
            active.wait(timeout=10)
        status.update(state="failed", updated_at=now(), curl_pid=None, error_type=type(error).__name__, error=str(error))
        atomic_json(status_path, status)
        raise
def install_when_ready(root: Path, queue: Path, source: Path):
    """Prevent duplicate package installers across supervisor restarts."""
    import msvcrt
    with (root / "install.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        _install_locked(root, queue, source)


def _install_locked(root: Path, queue: Path, source: Path):
    """Wait for all verified wheels, then install offline and check CUDA only."""
    queue_data = json.loads(queue.read_text(encoding="utf-8"))
    wheels = [task for task in queue_data["files"] if task["path"].endswith(".whl")]
    deadline = datetime.fromisoformat(queue_data["deadline_utc"]).timestamp()
    status_path = root / "install-status.json"
    status = {"helper_pid": os.getpid(), "started_at": now(), "state": "waiting_for_verified_wheels"}
    environment = os.environ.copy()
    for key in ("TYPESAFE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1", PYTHONUTF8="1",
                       HF_HOME=str(root / "hf-cache"), PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INDEX="1")
    try:
        while True:
            if time.time() >= deadline:
                raise TimeoutError("Setup deadline reached before offline installation")
            receipts_path = root / "download-receipts.json"
            receipts = json.loads(receipts_path.read_text(encoding="utf-8")) if receipts_path.exists() else {}
            ready = all(task["path"] in receipts for task in wheels)
            available = available_ram_bytes()
            status.update(updated_at=now(), available_ram_bytes=available,
                          verified_wheels=sum(task["path"] in receipts for task in wheels), expected_wheels=len(wheels))
            if ready and available < 8 * 1024**3:
                status["state"] = "waiting_for_8gib_available_ram"
            atomic_json(status_path, status)
            if ready and available >= 8 * 1024**3 and not (root / "PAUSE").exists():
                break
            if (root / "download-status.json").exists():
                download_state = json.loads((root / "download-status.json").read_text(encoding="utf-8"))
                if download_state.get("state") == "failed":
                    raise RuntimeError("Download worker failed; offline installer will not guess/retry")
            time.sleep(30)
        for task in wheels:
            path = destination(root, task["path"])
            if path.stat().st_size != task["size"] or sha256(path) != task["sha256"]:
                raise ValueError("Wheel changed before offline installation")
        actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if actual_commit != "930e6d5de16309be32579f5631a7369a3322140c":
            raise ValueError("Upstream source is not at pinned commit")
        commands = [
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *[str(destination(root, task["path"])) for task in wheels]],
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation", "--editable", str(source)],
            [sys.executable, "-m", "pip", "check"],
        ]
        with (root / "offline-install.txt").open("w", encoding="utf-8") as log:
            for index, command in enumerate(commands):
                while available_ram_bytes() < 8 * 1024**3 or (root / "PAUSE").exists():
                    status.update(state="waiting_for_memory_or_operator_resume", updated_at=now())
                    atomic_json(status_path, status)
                    if time.time() >= deadline:
                        raise TimeoutError("Setup deadline reached while waiting for memory/operator")
                    time.sleep(30)
                if (root / "PAUSE").exists() or time.time() >= deadline:
                    raise RuntimeError("Installation paused/deadline reached between stages")
                status.update(state="installing_offline", stage=index + 1, updated_at=now())
                atomic_json(status_path, status)
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=max(1, deadline - time.time()),
                               creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        packages = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], env=environment, text=True)
        atomic_json(root / "installed-packages.json", json.loads(packages))
        # The supervisor owns this explicit GPU-health handoff; installation alone
        # never grants GPU ownership, even for tiny readiness kernels.
        while not (root / "ALLOW-CUDA-HEALTH").is_file() or available_ram_bytes() < 8 * 1024**3 or (root / "PAUSE").exists():
            status.update(state="installed_waiting_for_cuda_owner_approval", updated_at=now(),
                          source_commit=actual_commit, model_inference=False)
            atomic_json(status_path, status)
            if time.time() >= deadline:
                raise TimeoutError("Setup deadline reached without CUDA-owner approval/resources")
            time.sleep(30)
        # Reserve at most90seconds for ownership contention inside the existing
        #120-second health-process cap. The setup's own deadline always wins.
        health = cuda_health_source(root, min(deadline, time.time() + 90))
        checked = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", health], env=environment,
                                 capture_output=True, text=True, timeout=max(1, min(120, deadline - time.time())),
                                 creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        (root / "health-stderr.txt").write_text(checked.stderr, encoding="utf-8")
        if checked.returncode:
            raise RuntimeError("Torch/import/CUDA readiness failed; see health-stderr.txt")
        atomic_json(root / "health.json", json.loads(checked.stdout))
        status.update(state="completed", updated_at=now(), source_commit=actual_commit, model_inference=False)
        atomic_json(status_path, status)
    except BaseException as error:
        status.update(state="failed", updated_at=now(), error_type=type(error).__name__, error=str(error))
        atomic_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--rate", type=int, default=1048576)
    parser.add_argument("--install-when-ready", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rate <= 1048576:
        parser.error("Rate must be 1..1048576 bytes/second")
    if args.install_when_ready:
        if args.source is None:
            parser.error("--source is required for offline installation")
        install_when_ready(args.root.resolve(), args.queue.resolve(), args.source.resolve())
    else:
        run(args.root.resolve(), args.queue.resolve(), args.rate)


if __name__ == "__main__":
    main()
