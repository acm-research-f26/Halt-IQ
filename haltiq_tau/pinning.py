"""Pin the source checkout and refuse silent benchmark-version changes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

from . import UPSTREAM_REVISION


def verify_upstream(path: str | Path, *, require_import=False) -> dict:
    root = Path(path).resolve(strict=True)
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, check=False, timeout=20)
        if result.returncode:
            raise ValueError("could not verify the pinned Tau source checkout")
        return result.stdout.strip()
    if git("rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError("Tau checkout revision differs from the reviewed benchmark pin")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tau tracked source/data has local modifications")
    if require_import:
        spec = importlib.util.find_spec("tau2")
        if spec is None or spec.origin is None:
            raise ValueError("Tau is not installed in this interpreter")
        expected = root / "src" / "tau2" / "__init__.py"
        if Path(spec.origin).resolve() != expected:
            raise ValueError("imported Tau package is not the pinned checkout")
    return {"upstream_revision": UPSTREAM_REVISION, "tracked_checkout_clean": True,
            "import_bound_to_checkout": require_import}
