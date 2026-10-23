"""Containment bootstrap: no study import or child creation before parent release."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def main():
    # The supervisor assigns this waiting process to its OS job before GO.
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit(125)
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if "input" in spec:
        if type(spec["input"]) is not str:
            raise ValueError("bootstrap input must be an explicit string")
        options = {"input": spec["input"], "text": True, "encoding": "utf-8"}
    else:
        options = {"stdin": subprocess.DEVNULL}
    result = subprocess.run(spec["argv"], cwd=spec["cwd"], shell=False, check=False, **options)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
