"""Owned Colab child processes; no configuration or credentials persisted."""
from __future__ import annotations
import json
import subprocess
import socket
import sys
import time
import urllib.request
from pathlib import Path


class Runtime:
    def __init__(self, output, timeout=900):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.critic_server = None
        self.log = None
        self.timeout = timeout

    def start(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Select a Colab GPU runtime before running the live study")
        from .critic import start_critic, Critic
        from .core import Agents, Gate
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8001)) == 0:
                raise RuntimeError("Writer port 8001 is occupied; existing service was not changed")
        self.critic_server = start_critic(self.output / "critic-cache",
                         log_path=self.output / "critic-server.log", timeout=self.timeout)
        self.log = (self.output / "writer-server.log").open("w", encoding="utf-8")
        self.writer = subprocess.Popen([sys.executable, "-m", "haltiq_pair", "serve-writer",
                    "--output", str(self.output / "writer")], stdout=self.log,
                    stderr=subprocess.STDOUT)
        end = time.monotonic() + self.timeout
        while time.monotonic() < end:
            if self.writer.poll() is not None:
                raise RuntimeError("Writer startup failed; read writer-server.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as r:
                    self.writer_metadata = json.load(r)
                from .server import MODEL, REVISION
                if (self.writer_metadata.get("model") != MODEL
                        or self.writer_metadata.get("revision") != REVISION):
                    raise RuntimeError("Writer health does not match the owned pinned model")
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("Writer model did not load before the startup deadline")
        agents = Agents()
        critic = Critic(endpoint=self.critic_server.endpoint, ranker_device="cpu")
        gate = Gate(critic=critic)
        (self.output / "runtime.json").write_text(json.dumps({"writer": self.writer_metadata,
            "critic_server": self.critic_server.provenance, "controller": gate.provenance}, indent=2))
        return agents, gate

    def close(self):
        if self.writer and self.writer.poll() is None:
            self.writer.terminate()
            try:
                self.writer.wait(20)
            except subprocess.TimeoutExpired:
                self.writer.kill()
                self.writer.wait(10)
        if self.log:
            self.log.close()
        if self.critic_server:
            self.critic_server.close()

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()
