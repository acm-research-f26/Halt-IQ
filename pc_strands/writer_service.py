"""Own a private Ollama server and one bounded writer stage.

The Windows job contains only children created here. Closing it releases this
study's server/worker without touching the user's other Ollama or LM Studio apps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from .control import RunControl, StudyStopped, atomic_json, resource_snapshot
from .windows_job import ProcessContainment


def server_environment(args) -> dict[str, str]:
    """Private-process controls; no registry/global environment changes.

    llama-server inherits LLAMA_ARG_CACHE_RAM even when Ollama does not expose
    --cache-ram. The pilot must still confirm the effective value in server.log.
    """
    environment = dict(os.environ)
    environment.pop("TYPESAFE_API_KEY", None)
    environment.update(OLLAMA_HOST=f"127.0.0.1:{args.port}", OLLAMA_MODELS=str(args.models.resolve()),
                       OLLAMA_NO_CLOUD="1", OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1",
                       OLLAMA_MAX_QUEUE="1", OLLAMA_KEEP_ALIVE="5m", OLLAMA_CONTEXT_LENGTH="16384",
                       OLLAMA_GPU_OVERHEAD=str(3 * 1024**3), OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                       LLAMA_ARG_CACHE_RAM="0", OLLAMA_KV_CACHE_TYPE=args.kv_cache_type,
                       OLLAMA_FLASH_ATTENTION="true" if args.flash_attention else "false")
    return environment


def run(args) -> int:
    args.supervision.mkdir(parents=True, exist_ok=True)
    host, port = "127.0.0.1", args.port
    with socket.socket() as probe:
        if probe.connect_ex((host, port)) == 0:
            raise RuntimeError(f"Private port {port} is occupied; no process was changed.")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Supply an explicit writer-stage command after --.")
    control = RunControl(args.supervision, args.deadline)
    control.before_work(require_gpu=True)
    environment = server_environment(args)
    creation = 0x08004000 if os.name == "nt" else 0
    manifest = {"state": "starting", "parent_pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
                "deadline": args.deadline, "command": command, "port": port,
                "server_executable": str(args.ollama), "models": str(args.models),
                "writer_service_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "writer_runtime_profile": {"name": f"background-v2-ram7-{args.kv_cache_type}",
                    "context_tokens": 16384, "host_prompt_cache_mib": 0,
                    "kv_cache_type": args.kv_cache_type, "flash_attention_requested": args.flash_attention,
                    "environment": {key: environment[key] for key in
                        ("LLAMA_ARG_CACHE_RAM", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION")}},
                "resource_profile": {"cpu_threads": 4, "minimum_free_vram_gib": 3,
                                     "minimum_ram_gib": 7, "one_gpu_owner": True}}
    server = worker = None
    job = ProcessContainment()
    try:
        # The stage worker owns the shared GPU lease and unloads its model before
        # releasing it. Merely starting this server does not load a GPU model.
        with (args.supervision / "server.log").open("ab") as server_log, \
             (args.supervision / "worker.log").open("ab") as worker_log:
            server = subprocess.Popen([str(args.ollama), "serve"], env=environment,
                                      stdout=server_log, stderr=subprocess.STDOUT, creationflags=creation)
            job.attach(server.pid)
            manifest["server_pid"] = server.pid
            atomic_json(args.supervision / "supervisor.json", manifest)
            ready_until = time.monotonic() + 45
            while True:
                control._check_stop()
                if server.poll() is not None:
                    raise RuntimeError("Owned Ollama exited before readiness; inspect its local log.")
                try:
                    with urlopen(f"http://{host}:{port}/api/tags", timeout=2) as response:
                        tags = json.load(response)
                    break
                except (URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= ready_until:
                        raise RuntimeError("Owned Ollama did not become ready within 45 seconds.") from None
                    time.sleep(.5)
            manifest["model_inventory"] = tags.get("models", [])
            worker = subprocess.Popen(command, cwd=args.repo, env=environment,
                                      stdout=worker_log, stderr=subprocess.STDOUT, creationflags=creation)
            job.attach(worker.pid)
            manifest.update(state="running", worker_pid=worker.pid)
            atomic_json(args.supervision / "supervisor.json", manifest)
            last_sample = 0
            while worker.poll() is None:
                control._check_stop()
                if server.poll() is not None:
                    raise RuntimeError("Owned Ollama exited while the writer stage was active.")
                if time.monotonic() - last_sample >= 30:
                    manifest.update(updated_at=datetime.now(timezone.utc).isoformat(), resources=resource_snapshot())
                    atomic_json(args.supervision / "supervisor.json", manifest)
                    last_sample = time.monotonic()
                time.sleep(.5)
            manifest.update(state="completed" if worker.returncode == 0 else "failed", worker_exit_code=worker.returncode)
            return worker.returncode
    except BaseException as error:
        manifest.update(state="stopped" if isinstance(error, StudyStopped) else "failed", error=str(error))
        raise
    finally:
        job.close()
        for process in (worker, server):
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),
                        server_exit_code=server.returncode if server else None,
                        worker_exit_code=worker.returncode if worker else None)
        atomic_json(args.supervision / "supervisor.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--gpu-lock", required=True, type=Path)
    parser.add_argument("--port", default=11435, type=int)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--kv-cache-type", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--flash-attention", action="store_true",
                        help="Explicitly request flash attention; mandatory for a separate q8_0 profile")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.ollama.is_file() or not args.models.is_dir() or not args.repo.is_dir():
        parser.error("The installed Ollama, existing model cache and source checkout must exist.")
    if args.kv_cache_type == "q8_0" and not args.flash_attention:
        parser.error("q8_0 requires --flash-attention and a separately frozen output/profile")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
