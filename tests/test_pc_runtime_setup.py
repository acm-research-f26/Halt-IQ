"""Offline bootstrap checks; temporary leases only, no CUDA/network/models."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from pc_strands import runtime_setup as setup
from pc_strands.control import GpuLease


class Response:
    def __init__(self, status=206, content_range="bytes 2-2/5", body=b"x", length=None):
        self.status, self.body = status, body
        self.headers = {"Content-Range": content_range}
        if length is not None:
            self.headers["Content-Length"] = str(length)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, maximum):
        return self.body[:maximum]


class RuntimeSetupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows available-memory API")
    def test_available_ram_is_read_without_allocation(self):
        value = setup.available_ram_bytes()
        self.assertIsInstance(value, int)
        self.assertGreater(value, 0)

    def test_past_deadline_prevents_network_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.json"
            queue.write_text(json.dumps({"deadline_utc": "2000-01-01T00:00:00+00:00", "files": []}))
            with patch.object(setup.subprocess, "Popen") as launch:
                with self.assertRaises(TimeoutError):
                    setup._download_locked(root, queue, 1048576)
                launch.assert_not_called()
            self.assertEqual(json.loads((root / "download-status.json").read_text())["state"], "failed")

    def test_exact_resume_206(self):
        with patch.object(setup, "urlopen", return_value=Response()):
            self.assertEqual(setup.validate_resume("https://example.invalid/x", 2, 5)["http_status"], 206)

    def test_reject_ignored_resume(self):
        with patch.object(setup, "urlopen", return_value=Response(status=200)):
            with self.assertRaises(ValueError):
                setup.validate_resume("https://example.invalid/x", 2, 5)

    def test_reject_wrong_range_or_total(self):
        for value in ("bytes 1-1/5", "bytes 2-2/6"):
            with self.subTest(value=value), patch.object(setup, "urlopen", return_value=Response(content_range=value)):
                with self.assertRaises(ValueError):
                    setup.validate_resume("https://example.invalid/x", 2, 5)

    def test_reject_long_probe_body(self):
        with patch.object(setup, "urlopen", return_value=Response(body=b"xy")):
            with self.assertRaises(ValueError):
                setup.validate_resume("https://example.invalid/x", 2, 5)

    def test_fresh_200_permitted_but_length_pinned(self):
        with patch.object(setup, "urlopen", return_value=Response(status=200, length=5)):
            self.assertFalse(setup.validate_resume("https://example.invalid/x", 0, 5)["resume"])
            with self.assertRaises(ValueError):
                setup.validate_resume("https://example.invalid/x", 0, 6)

    def test_destination_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(setup.destination(root, "models/base.bin"), root / "models" / "base.bin")
            for relative in ("..", "../escape.bin", str(root.parent / "outside.bin")):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    setup.destination(root, relative)

    def test_sha_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            path.write_bytes(b"hello")
            self.assertEqual(setup.sha256(path), hashlib.sha256(b"hello").hexdigest())
            setup.atomic_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})

    def transfer(self, content, expected):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        queue = root / "queue.json"
        queue.write_text(json.dumps({"files": [{"url": "https://huggingface.co/public/file",
            "path": "models/file.bin", "size": len(content), "sha256": hashlib.sha256(expected).hexdigest()}]}))

        class Curl:
            pid = 123
            returncode = 0

            def __init__(self, command, **_):
                Path(command[command.index("--output") + 1]).write_bytes(content)

            def poll(self):
                return 0

        # No real subprocess is created. Constants are supplied on non-Windows
        # so the integrity state machine can be checked there as well.
        with patch.object(setup, "validate_resume", return_value={"test": True}), \
             patch.object(setup.subprocess, "Popen", Curl), \
             patch.object(setup.subprocess, "CREATE_NO_WINDOW", 0, create=True), \
             patch.object(setup.subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0, create=True):
            if content == expected:
                setup._download_locked(root, queue, 1048576)
            else:
                with self.assertRaises(ValueError):
                    setup._download_locked(root, queue, 1048576)
        return root, queue

    def test_promotion_receipt_and_restart_reverification(self):
        root, queue = self.transfer(b"hello", b"hello")
        self.assertEqual((root / "models/file.bin").read_bytes(), b"hello")
        self.assertIn("models/file.bin", json.loads((root / "download-receipts.json").read_text()))
        (root / "models/file.bin").write_bytes(b"other")
        with self.assertRaises(ValueError):
            setup._download_locked(root, queue, 1048576)

    def test_wrong_hash_never_promoted(self):
        root, _ = self.transfer(b"other", b"hello")
        self.assertFalse((root / "models/file.bin").exists())
        self.assertTrue((root / "models/file.bin.part").exists())
        self.assertEqual(json.loads((root / "download-status.json").read_text())["state"], "failed")


@unittest.skipUnless(os.name == "nt", "Actual Windows OS byte-range lease protocol")
class HealthLeaseTests(unittest.TestCase):
    def test_child_cannot_enter_while_study_worker_holds_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu-owner.lock"
            code = (setup.HEALTH_GPU_LEASE_SOURCE + "\nwith health_gpu_lease(Path(" + repr(str(path))
                    + "),time.time()+.3):\n    print('UNEXPECTED')\n")
            with GpuLease(path, "offline-contention-test"):
                child = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code],
                    capture_output=True, text=True, timeout=4,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn("TimeoutError:", child.stderr)
            self.assertEqual(child.stdout, "")

    def test_released_lease_records_owner_and_releases_for_worker(self):
        namespace = {}
        exec(setup.HEALTH_GPU_LEASE_SOURCE, namespace)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu-owner.lock"
            with namespace["health_gpu_lease"](path, time.time() + 2) as lease:
                self.assertEqual(lease["holder_pid"], os.getpid())
                self.assertTrue(lease["held_during_probe"])
                self.assertFalse(lease["released"])
            self.assertTrue(lease["released"])
            with GpuLease(path, "offline-after-release"):
                pass

    def test_metadata_error_releases_lock_in_same_process(self):
        namespace = {}
        exec(setup.HEALTH_GPU_LEASE_SOURCE, namespace)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu-owner.lock"
            with patch.object(namespace["json"], "dumps", side_effect=OSError("simulated metadata write failure")):
                with self.assertRaises(OSError):
                    with namespace["health_gpu_lease"](path, time.time() + 2):
                        self.fail("must not yield")
            with GpuLease(path, "offline-after-metadata-failure"):
                pass


class HealthScriptStructureTests(unittest.TestCase):
    def test_torch_import_is_inside_lease_and_after_admission_checks(self):
        source = setup.cuda_health_source(Path("runtime"), time.time() + 90)
        tree = ast.parse(source)
        owner = next(node for node in tree.body if isinstance(node, ast.With))
        self.assertEqual(owner.items[0].context_expr.func.id, "health_gpu_lease")
        torch_import = next(node for node in owner.body if isinstance(node, ast.Import)
                            and any(alias.name == "torch" for alias in node.names))
        marker_check = next(node for node in owner.body if isinstance(node, ast.If)
                            and "ALLOW-CUDA-HEALTH" in ast.get_source_segment(source, node))
        self.assertGreater(torch_import.lineno, marker_check.lineno)
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
                             for node in tree.body))


if __name__ == "__main__":
    unittest.main()
