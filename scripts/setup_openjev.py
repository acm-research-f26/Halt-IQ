"""Install a pinned community OpenJev in its own environment; no hosted inference."""

import argparse
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Existing original Hugging Face model directory for the PyTorch backend")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("OpenJev requires Python 3.12 or newer")
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if apple and args.model_path is not None:
        parser.error("The Mac setup uses the pinned MLX checkpoint; --model-path is for Windows/Linux")
    if not apple and (args.model_path is None or not (args.model_path / "config.json").is_file()):
        parser.error("On Windows/Linux, first download original Gemma 3 4B weights, then pass --model-path DIRECTORY; MLX weights cannot run on PyTorch")
    config = json.loads((ROOT / "configs/openjev.json").read_text())
    local = ROOT / ".haltiq"
    local.mkdir(exist_ok=True)
    source, environment = local / "openjev-src", local / "openjev-venv"
    if not source.exists():
        run("git", "clone", config["upstream_url"], source)
    origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip()
    if origin != config["upstream_url"]:
        parser.error("Existing OpenJev checkout has a different origin; it was left untouched")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        parser.error("Existing OpenJev checkout has local changes; it was left untouched")
    run("git", "-C", source, "fetch", "origin", config["upstream_revision"])
    run("git", "-C", source, "checkout", "--detach", config["upstream_revision"])
    if not environment.exists():
        venv.create(environment, with_pip=True)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    install_args = [python, "-m", "pip", "install", source]
    constraints = ROOT / "configs/openjev-mac-constraints.txt"
    if apple and constraints.exists():
        install_args += ["-c", constraints]
    run(*install_args)
    run(python, ROOT / "scripts/setup_openjev_model.py", *(["--model-path", args.model_path.resolve()] if args.model_path else []))
    command = [str(python), str(ROOT / "scripts/serve_openjev.py")]
    shown = subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)
    print(f"Setup complete. Start the local server with:\n{shown}")


if __name__ == "__main__":
    main()
