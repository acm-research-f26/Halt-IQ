"""Cooperative background limits and one-owner GPU lease for this study.

These are admission checks and safe-boundary waits, not GPU percentage quotas.
Only the study's own work is paused. No system settings or other processes change.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Callable


class StudyStopped(RuntimeError):
    """A deadline or explicit stop prevents new work."""


class ResourcePressure(RuntimeError):
    """The selected stage cannot continue under the declared resource profile."""


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_deadline(value: str) -> datetime:
    deadline = datetime.fromisoformat(value)
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("Deadline must include its UTC offset.")
    return deadline


def memory_available_bytes() -> int | None:
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total_physical", "available_physical", "total_pagefile", "available_pagefile",
                 "total_virtual", "available_virtual", "extended_virtual")]
        state = MemoryStatus()
        state.length = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return None
        return int(state.available_physical)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def resource_snapshot() -> dict:
    result = {"at": utc_now(), "available_ram_bytes": memory_available_bytes()}
    try:
        command = ["nvidia-smi", "--query-gpu=memory.total,memory.used,temperature.gpu,utilization.gpu",
                   "--format=csv,noheader,nounits"]
        process = subprocess.run(command, capture_output=True, text=True, timeout=8,
                                 creationflags=0x08000000 if os.name == "nt" else 0)
        if process.returncode == 0:
            values = [float(value.strip()) for value in process.stdout.splitlines()[0].split(",")]
            result.update(gpu_total_mib=values[0], gpu_used_mib=values[1],
                          gpu_free_mib=values[0] - values[1], gpu_temperature_c=values[2],
                          gpu_utilization_percent=values[3])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return result


class GpuLease:
    """OS-released file lock; stale metadata never grants GPU ownership."""
    def __init__(self, path: Path, purpose: str):
        self.path, self.purpose, self.file = Path(path), purpose, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            handle.close()
            raise ResourcePressure("Another owned study stage holds the GPU lease.") from None
        self.file = handle
        try:
            handle.seek(1)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "purpose": self.purpose,
                                     "acquired_at": utc_now()}).encode())
            handle.flush()
        except BaseException:
            # __enter__ failures do not invoke the context manager's __exit__.
            # Release the OS lock even when disk/metadata writing fails.
            try:
                self.__exit__()
            except OSError:
                pass  # Closing the handle also releases its lock; retain the write error.
            raise
        return self

    def __exit__(self, *_):
        if self.file is not None:
            try:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_UN)
            finally:
                self.file.close()
                self.file = None


class RunControl:
    def __init__(self, run_dir: Path, deadline: str, *, duty_cycle: float = .5,
                 min_rest_seconds: float = 5, minimum_ram_gib: float = 7,
                 minimum_free_vram_gib: float = 3, maximum_pause_seconds: float = 120,
                 sampler: Callable[[], dict] = resource_snapshot):
        if not 0 < duty_cycle <= 1 or min_rest_seconds < 0:
            raise ValueError("Invalid duty-cycle profile.")
        if minimum_ram_gib < 0 or minimum_free_vram_gib < 0 or maximum_pause_seconds < 0:
            raise ValueError("Resource thresholds cannot be negative.")
        self.directory = Path(run_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.deadline = parse_deadline(deadline)
        self.duty_cycle, self.min_rest = duty_cycle, min_rest_seconds
        self.ram_bytes = minimum_ram_gib * 1024**3
        self.vram_mib = minimum_free_vram_gib * 1024
        self.maximum_pause = maximum_pause_seconds
        self.sampler = sampler
        self.active_seconds = self.idle_seconds = 0.0
        self.last_resources = None

    def _check_stop(self):
        if (self.directory / "stop.request").exists():
            raise StudyStopped("Explicit study stop requested.")
        if datetime.now(timezone.utc) >= self.deadline:
            raise StudyStopped("Declared study deadline reached.")

    def status(self, state: str, **extra):
        atomic_json(self.directory / "control-status.json", {
            "state": state, "pid": os.getpid(), "updated_at": utc_now(),
            "deadline": self.deadline.isoformat(), "active_seconds": self.active_seconds,
            "intentional_idle_seconds": self.idle_seconds, "resources": self.last_resources,
            **extra})

    def before_work(self, *, require_gpu: bool = False):
        pressure_started = None
        while True:
            self._check_stop()
            if (self.directory / "pause.request").exists():
                self.status("user_paused")
                self._sleep(1)
                continue
            snapshot = self.sampler()
            self.last_resources = snapshot
            reasons = []
            available = snapshot.get("available_ram_bytes")
            if available is None:
                reasons.append("RAM observation unavailable")
            elif available < self.ram_bytes:
                reasons.append("available RAM below declared threshold")
            if require_gpu:
                free_vram = snapshot.get("gpu_free_mib")
                if free_vram is None:
                    reasons.append("GPU observation unavailable")
                elif free_vram < self.vram_mib:
                    reasons.append("free VRAM below declared threshold")
                if snapshot.get("gpu_temperature_c", 0) >= 80:
                    reasons.append("GPU temperature at workflow pause threshold")
            if not reasons:
                self.status("ready")
                return
            pressure_started = pressure_started or time.monotonic()
            self.status("resource_pause", reasons=reasons)
            if time.monotonic() - pressure_started >= self.maximum_pause:
                raise ResourcePressure("; ".join(reasons))
            self._sleep(min(5, self.maximum_pause))

    def _sleep(self, seconds: float):
        started = time.monotonic()
        try:
            while time.monotonic() - started < seconds:
                self._check_stop()
                time.sleep(min(.25, max(0, seconds - (time.monotonic() - started))))
        finally:
            self.idle_seconds += time.monotonic() - started

    def rest_after_block(self, elapsed_seconds: float):
        if elapsed_seconds < 0:
            raise ValueError("Active duration cannot be negative.")
        self.active_seconds += elapsed_seconds
        rest = max(self.min_rest, elapsed_seconds * (1 / self.duty_cycle - 1))
        if rest:
            self.status("intentional_rest", rest_seconds=rest)
            self._sleep(rest)
        self._check_stop()

    @contextmanager
    def work(self, *, require_gpu: bool = False):
        self.before_work(require_gpu=require_gpu)
        started = time.monotonic()
        try:
            yield
        except BaseException:
            self.active_seconds += time.monotonic() - started
            self.status("work_failed")
            raise
        else:
            self.rest_after_block(time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "pause", "resume", "stop"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.action in ("pause", "stop"):
        atomic_json(args.run_dir / (args.action + ".request"),
                    {"requested_at": utc_now(), "source": "explicit_local_command"})
    elif args.action == "resume":
        (args.run_dir / "pause.request").unlink(missing_ok=True)
    status_path = args.run_dir / "control-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "not_started"}
    status.update(pause_requested=(args.run_dir / "pause.request").exists(),
                  stop_requested=(args.run_dir / "stop.request").exists())
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
