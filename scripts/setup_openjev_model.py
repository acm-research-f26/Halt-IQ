"""Called inside OpenJev's separate environment by setup_openjev.py."""

import argparse
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/openjev.json").read_text())
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if apple:
        from huggingface_hub import snapshot_download
        spec = config["mlx_model"]
        model = root / ".haltiq/models" / spec["directory"]
        snapshot_download(spec["repo_id"], revision=spec["revision"], local_dir=model,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "README.md"], token=False, max_workers=4)
        model_id, revision, repository = spec["model_id"], spec["revision"], spec["repo_id"]
    else:
        if args.model_path is None:
            parser.error("--model-path is required for the PyTorch setup")
        model = args.model_path.resolve()
        checkpoint = json.loads((model / "config.json").read_text())
        text_config = checkpoint.get("text_config", {})
        if (checkpoint.get("model_type") != "gemma3" or text_config.get("hidden_size") != 2560
                or text_config.get("num_hidden_layers") != 34
                or "quantization" in checkpoint or "quantization_config" in checkpoint):
            parser.error("Expected original, unquantized Gemma 3 4B weights for PyTorch")
        model_id, revision, repository = "openjev-dasein-gemma3-4b-torch", None, "user-supplied original Hugging Face checkpoint"
    files = sorted(p for p in model.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".model"})
    if not any(p.suffix == ".safetensors" for p in files):
        raise RuntimeError("No safetensors model weights found")
    print("Recording checkpoint hashes...", flush=True)
    provenance = {
        "implementation": config["implementation"], "upstream_revision": config["upstream_revision"],
        "model_id": model_id, "model_path": str(model), "model_repository": repository,
        "model_revision": revision, "backend": "mlx" if apple else "torch", "device": "auto",
        "max_prompt_tokens": config["max_prompt_tokens"],
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files},
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
    }
    target = root / ".haltiq/openjev-runtime.json"
    target.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded local setup: {target}")


if __name__ == "__main__":
    main()
