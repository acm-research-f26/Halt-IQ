# Background study controls

`pc_strands.control` provides cooperative resource checks and an operating-system
file lock for one study GPU owner. It changes only the study's own work.

The current background profile uses four writer threads, a 7 GiB available-RAM
reserve, a 3 GiB free-VRAM reserve, and roughly equal work/rest intervals. These
are admission checks and waits, not GPU percentage quotas. The original 8 GiB
profile stopped safely on this PC; the revised profile is tested separately and
must be frozen across compared policies. A single forward can still be intensive.

The control module uses the standard library. Run from this checkout or install
the package with an existing compatible Python environment.

```powershell
python -m pc_strands.control status --run-dir <worker-output-directory>
python -m pc_strands.control pause --run-dir <worker-output-directory>
python -m pc_strands.control resume --run-dir <worker-output-directory>
python -m pc_strands.control stop --run-dir <worker-output-directory>
```

Use the actual worker's output directory. A wrapper's supervision directory is
not the worker's pause target. A pause takes effect at the next safe boundary
and may retain model memory; stop and normal cleanup release owned resources.
A pending explicit pause remains until resumed. Deadlines include a UTC offset.
Missing resource observations fail closed, and sustained pressure stops the stage.

`GpuLease` uses a native byte/file lock, released when the process exits. Metadata
is diagnostic and never proves ownership. `windows_job.ProcessContainment` owns
only processes explicitly attached by the study. Its Windows kill-on-close job
releases those children without stopping unrelated Ollama, LM Studio or user apps.

Validation covers native cross-process lock exclusion/release, metadata-write
failure recovery, deadline and explicit-stop precedence, missing telemetry,
CPU-only work, and failed-work time accounting. The Windows job was also probed
with an owned disposable child. Native model inference/training has separate
runtime and checkpoint verification gates; passing control tests is not evidence
of model quality or successful fine-tuning.
