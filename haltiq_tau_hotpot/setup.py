"""Verify existing offline assets and emit a private machine configuration."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from haltiq_hotpot.native import PINS
from haltiq_hotpot.protocol import digest
from .journal import durable_write


WRITER_DIGEST = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
TAU_REVISION = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"


def machine_config(testing_root, data_freeze, output):
    root, data_freeze = Path(testing_root).resolve(), Path(data_freeze).resolve()
    directories = {"hobson": root / "partner-pilot-runtime/models/hobson-v19",
        "minilm": root / "partner-pilot-runtime/models/minilm-cross-encoder",
        "base": root / "strands-runtime/models/base",
        "upstream": root / "strands-runtime/source/src/strands_decider"}
    inputs = []
    for name, expected in PINS.items():
        group, relative = name.split("/", 1)
        path = directories[group] / relative
        if digest(path) != expected:
            raise ValueError("Cached approved model/source changed: " + name)
        inputs.append({"path": str(path), "sha256": expected})
    native = root / "strands-runtime/source"
    native_revision = subprocess.run(["git", "-C", str(native), "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=True).stdout.strip()
    native_changed = subprocess.run(["git", "-C", str(native), "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip()
    if native_revision != "930e6d5de16309be32579f5631a7369a3322140c" or native_changed:
        raise ValueError("Use the clean pinned native Strands source")
    already = {item["path"] for item in inputs}
    inputs += [{"path": str(path), "sha256": digest(path)} for path in sorted(directories["upstream"].rglob("*.py"))
               if str(path) not in already]
    tau = root / "tau2-upstream-review"
    revision = subprocess.run(["git", "-C", str(tau), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    changed = subprocess.run(["git", "-C", str(tau), "status", "--porcelain"], capture_output=True,
                             text=True, check=True).stdout.strip()
    if revision != TAU_REVISION or changed:
        raise ValueError("Use the clean pinned official Tau2 source")
    inputs += [{"path": str(path), "sha256": digest(path)} for path in sorted((tau / "src/tau2").rglob("*.py"))]
    executable = root.parent / "runtime/ollama-v0.32.15/ollama.exe"
    if digest(executable) != "0a9d42eabc59fdafde8d2d3e7964f6050b31a17b3e3795bfacb367c12df790f4":
        raise ValueError("Existing portable Ollama executable changed")
    models = root.parent / "runtime/models"
    manifest = models / "manifests/registry.ollama.ai/library/qwen3/8b"
    if digest(manifest) != WRITER_DIGEST:
        raise ValueError("Existing Qwen3:8B model manifest differs")
    inputs += [{"path": str(p), "sha256": digest(p)} for p in (executable, manifest)]
    definition = json.loads(manifest.read_text(encoding="utf-8"))
    for layer in [definition["config"]] + definition["layers"]:
        expected = layer["digest"].removeprefix("sha256:")
        path = models / "blobs" / ("sha256-" + expected)
        if digest(path) != expected:
            raise ValueError("Writer model layer changed")
        inputs.append({"path": str(path), "sha256": expected})
    config = {"purpose": "qualification_pilot", "data_freeze": str(data_freeze),
        "data_freeze_sha256": digest(data_freeze), "checkpoint": str(directories["hobson"]),
        "base": str(directories["base"]), "reranker": str(directories["minilm"]),
        "ollama": str(executable), "ollama_models": str(models), "writer_digest": WRITER_DIGEST,
        "gpu_lease": str(root / "strands-runtime/gpu-owner.lock"),
        "study_lease": str(root / "partner-pilot-runtime/unattended-study.lock"),
        "tau_revision": revision, "model_inputs": inputs, "hosted_services": False,
        "runtime_modules": {"tau2": str(tau / "src/tau2/__init__.py"),
                            "strands_decider": str(directories["upstream"] / "__init__.py")},
        "new_downloads": False, "long_study_started": False}
    durable_write(output, config)
    return config
