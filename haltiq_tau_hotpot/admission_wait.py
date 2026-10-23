"""Optional finite startup waiting; no model imports or inference retries."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

from pc_strands.control import ResourcePressure
from .hobson import file_hash
from .journal import durable_write

CLOCK_TOLERANCE_SECONDS = 1.0


def wait_limit(config):
    value = config.get("question_admission_wait_seconds", 0)
    if type(value) not in (int, float) or not math.isfinite(value) or value != 0 and not 30 <= value <= 1200:
        raise ValueError("Question admission wait must be0 or explicitly bounded30..1200seconds")
    return value


def wait_for_admission(config, control, output, *, sampler, sleep, clock, lease_factory, port_probe, monotonic=None):
    """Wait before dispatch; the cap includes sampling/IO and the30s window.

    A30s budget can conservatively expire when sampling/IO has nonzero overhead.
    Pressure never keeps the admission lease held during its following wait.
    """
    limit = wait_limit(config)
    if not limit:
        raise ValueError("Bounded admission requires an explicit nonzero frozen opt-in")
    monotonic = monotonic or time.monotonic
    started, idle_before = monotonic(), control.idle_seconds
    wall_started = clock()
    history_path = Path(output).with_name(Path(output).stem + ".observations.jsonl")
    # Never append observations to a previous admission attempt.
    with history_path.open("x", encoding="utf-8", newline="\n"):
        pass
    history = []
    result = {"schema": "haltiq-main-question-admission-v1", "status": "BLOCKED_ADMISSION",
        "samples": [], "ram_gib": 11, "vram_gib": 9, "gpu_lease_released": None,
        "private_port_closed": None, "model_calls": 0, "automatic_retry": False}

    def check(*, completing_window=False):
        control._check_stop()
        if clock() >= control.deadline:
            raise ResourcePressure("Question admission reached the model cutoff")
        elapsed = monotonic() - started
        if elapsed > limit or elapsed == limit and not completing_window:
            raise ResourcePressure("Bounded question admission wait expired before dispatch")

    def rest(seconds, *, completing_window=False):
        check(completing_window=completing_window)
        remaining = min(limit - (monotonic() - started), (control.deadline - clock()).total_seconds())
        sleep(min(seconds, max(0, remaining)))
        check(completing_window=completing_window)

    def observe(kind, sample=None):
        row = {"index": len(history), "at": clock().isoformat(), "elapsed_seconds": monotonic() - started,
               "kind": kind, "sample": sample}
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        history.append(row)

    class NotReady(Exception):
        pass

    try:
        while True:
            check()
            if (control.directory / "pause.request").exists():
                result["samples"] = []
                observe("paused")
                rest(1)
                continue
            if port_probe(11435):
                result["private_port_closed"] = False
                result["samples"] = []
                observe("port_busy")
                rest(10)
                continue
            try:
                lease = lease_factory(Path(config["gpu_lease"]), "bounded main question startup admission")
                lease.__enter__()
            except ResourcePressure:
                result["samples"] = []
                observe("gpu_busy")
                rest(10)
                continue
            ready = False
            try:
                result["samples"] = []
                for index in range(4):
                    if (control.directory / "pause.request").exists():
                        observe("paused")
                        raise NotReady
                    if index:
                        rest(10, completing_window=index == 3)
                    check(completing_window=index == 3)
                    if (control.directory / "pause.request").exists():
                        observe("paused")
                        raise NotReady
                    observed = sampler()
                    ram, vram = observed.get("available_ram_bytes"), observed.get("gpu_free_mib")
                    sample = {"at": clock().isoformat(), "available_ram_gib": ram / 1024**3 if type(ram) in (int, float) else None,
                        "free_vram_gib": vram / 1024 if type(vram) in (int, float) else None,
                        "gpu_temperature_c": observed.get("gpu_temperature_c")}
                    sample = {k: v if k == "at" or type(v) in (int, float) and math.isfinite(v) else None for k, v in sample.items()}
                    port_clear = not port_probe(11435)
                    good = (all(sample[k] is not None for k in ("available_ram_gib", "free_vram_gib", "gpu_temperature_c"))
                        and sample["available_ram_gib"] >= 11 and sample["free_vram_gib"] >= 9
                        and 0 <= sample["gpu_temperature_c"] < 80 and port_clear)
                    observe("good_sample" if good else "resource_pressure" if port_clear else "port_busy", sample)
                    if not good:
                        result["samples"] = []
                        raise NotReady
                    result["samples"].append(sample)
                check(completing_window=True)
                if (control.directory / "pause.request").exists():
                    observe("paused")
                    raise NotReady
                ready = True
            except NotReady:
                pass
            finally:
                lease.__exit__()
            if not ready:
                # All pressure waits happen after admission ownership is released.
                rest(1 if (control.directory / "pause.request").exists() else 10)
                continue
            with lease_factory(Path(config["gpu_lease"]), "verify bounded question admission release"):
                pass
            result["gpu_lease_released"] = True
            result["private_port_closed"] = not port_probe(11435)
            if not result["private_port_closed"]:
                raise ResourcePressure("Private port ownership changed after the admission window")
            check(completing_window=True)
            times = [datetime.fromisoformat(s["at"]) for s in result["samples"]]
            if any(t.tzinfo is None for t in times) or any((b-a).total_seconds() < 10 for a,b in zip(times,times[1:])) or (times[-1]-times[0]).total_seconds() < 30:
                raise ResourcePressure("Fresh bounded question admission timing did not pass")
            result["status"] = "PASS_ADMISSION"
            break
    except BaseException as error:
        result.update(error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        result["wait"] = {"schema": "haltiq-question-admission-wait-v1", "timeout_seconds": limit,
            "started_at": wall_started.isoformat(), "finished_at": clock().isoformat(),
            "elapsed_seconds": monotonic() - started, "idle_seconds": control.idle_seconds - idle_before,
            "history_file": history_path.name, "history_sha256": file_hash(history_path), "observations": len(history)}
        durable_write(output, result)
    return result


def verify_wait_history(config, admitted, directory):
    """Validate opt-in proof without changing the legacy four-sample contract."""
    limit = wait_limit(config)
    proof = admitted.get("wait")
    if not limit:
        if proof is not None:
            raise ValueError("Admission waiting was not declared in the frozen config")
        return
    if not isinstance(proof, dict) or proof.get("schema") != "haltiq-question-admission-wait-v1" or proof.get("timeout_seconds") != limit:
        raise ValueError("Bounded admission wait proof is absent or differs from its frozen opt-in")
    path = Path(directory) / "admission.observations.jsonl"
    if proof.get("history_file") != path.name or path.is_symlink() or file_hash(path) != proof.get("history_sha256"):
        raise ValueError("Admission observation history changed")
    history = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    begin, end = (datetime.fromisoformat(proof[k]) for k in ("started_at", "finished_at"))
    elapsed, idle = proof.get("elapsed_seconds"), proof.get("idle_seconds")
    if (begin.tzinfo is None or end.tzinfo is None or begin > end or proof.get("observations") != len(history)
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 30 <= elapsed <= limit
            or type(idle) not in (int, float) or not math.isfinite(idle) or not 30 <= idle <= elapsed + .01
            or abs((end-begin).total_seconds() - elapsed) > CLOCK_TOLERANCE_SECONDS):
        raise ValueError("Admission wait timing/counts do not reconcile")
    previous, previous_at = -1, begin
    for index, row in enumerate(history):
        at = datetime.fromisoformat(row["at"])
        measured = row.get("elapsed_seconds")
        if (row.get("index") != index or at.tzinfo is None or not begin <= at <= end
                or type(measured) not in (int, float) or not math.isfinite(measured) or not previous <= measured <= elapsed
                or at < previous_at or abs((at-begin).total_seconds() - measured) > CLOCK_TOLERANCE_SECONDS
                or row.get("kind") not in {"paused", "port_busy", "gpu_busy", "good_sample", "resource_pressure"}):
            raise ValueError("Admission observation order/clock is invalid")
        if row["kind"] == "good_sample":
            sample = row.get("sample") or {}
            sample_at = datetime.fromisoformat(sample["at"])
            values = [sample.get(k) for k in ("available_ram_gib", "free_vram_gib", "gpu_temperature_c")]
            if (not all(type(v) in (int, float) and math.isfinite(v) for v in values)
                    or values[0] < 11 or values[1] < 9 or not 0 <= values[2] < 80
                    or sample_at.tzinfo is None or not 0 <= (at-sample_at).total_seconds() <= CLOCK_TOLERANCE_SECONDS):
                raise ValueError("Passing admission history lowered or omitted startup reserves")
        previous, previous_at = measured, at
    final = history[-4:]
    if len(final) != 4 or any(row["kind"] != "good_sample" for row in final) or [row["sample"] for row in final] != admitted.get("samples"):
        raise ValueError("Admission must finish with four new consecutive passing observations")
