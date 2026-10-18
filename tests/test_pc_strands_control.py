import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pc_strands.control import GpuLease, ResourcePressure, RunControl, StudyStopped, atomic_json, parse_deadline


class ControlTests(unittest.TestCase):
    def test_deadline_requires_offset_and_stops_new_work(self):
        with self.assertRaises(ValueError):
            parse_deadline("2026-10-18T20:30:00")
        with tempfile.TemporaryDirectory() as directory:
            control = RunControl(Path(directory), "2000-01-01T00:00:00+00:00")
            with self.assertRaises(StudyStopped):
                control.before_work()

    def test_resource_admission_fails_closed_without_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RunControl(Path(directory), "2099-01-01T00:00:00+00:00",
                                 maximum_pause_seconds=0, sampler=lambda: {})
            with self.assertRaises(ResourcePressure):
                control.before_work(require_gpu=True)
            self.assertEqual(json.loads((Path(directory)/"control-status.json").read_text())["state"], "resource_pause")

    def test_stop_precedes_healthy_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/"stop.request", {"explicit": True})
            control = RunControl(root, "2099-01-01T00:00:00+00:00",
                                 sampler=lambda: {"available_ram_bytes": 32*1024**3})
            with self.assertRaises(StudyStopped):
                control.before_work()

    def test_cpu_work_does_not_require_gpu_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RunControl(Path(directory), "2099-01-01T00:00:00+00:00", min_rest_seconds=0,
                                 duty_cycle=1, sampler=lambda: {"available_ram_bytes": 32*1024**3})
            with control.work():
                pass
            self.assertGreaterEqual(control.active_seconds, 0)

    def test_os_lock_prevents_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"gpu.lock"
            code = "from pathlib import Path; from pc_strands.control import GpuLease; import sys\nwith GpuLease(Path(sys.argv[1]), 'child'): print('owned')"
            with GpuLease(path, "parent"):
                child = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
                self.assertNotEqual(child.returncode, 0)
            child = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIn("owned", child.stdout)

    def test_metadata_write_failure_closes_handle_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            path.write_bytes(b"0")
            handle = path.open("a+b")
            class FailingMetadataWriter:
                def __getattr__(self, name):
                    return getattr(handle, name)
                def write(self, _):
                    raise OSError("injected metadata failure")
            lease = GpuLease(path, "test")
            with patch.object(Path, "open", return_value=FailingMetadataWriter()):
                with self.assertRaisesRegex(OSError, "injected metadata failure"):
                    lease.__enter__()
            self.assertTrue(handle.closed)
            self.assertIsNone(lease.file)
            with GpuLease(path, "reacquired"):
                pass

    def test_failed_work_keeps_its_active_time(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RunControl(Path(directory), "2099-01-01T00:00:00+00:00", min_rest_seconds=0,
                                 sampler=lambda: {"available_ram_bytes": 32*1024**3})
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                with patch("pc_strands.control.time.monotonic", side_effect=[1, 1.5]):
                    with control.work():
                        raise RuntimeError("model failed")
            state = json.loads((Path(directory)/"control-status.json").read_text())
            self.assertEqual(state["state"], "work_failed")
            self.assertEqual(state["active_seconds"], .5)


if __name__ == "__main__":
    unittest.main()
