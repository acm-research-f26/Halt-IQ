"""Allocator lifecycle controls without importing Torch or executing a model."""
from types import SimpleNamespace
import unittest

from haltiq_hotpot.native import OfficialDecider, release_cuda_allocations
from haltiq_hotpot.runner import release_decider


class Cuda:
    def __init__(self, *, live=0, workspace=8 * 1024**2):
        self.live, self.workspace, self.calls = live, workspace, []
    def synchronize(self):
        self.calls.append("sync")
    def memory_allocated(self):
        return self.live + self.workspace
    def memory_reserved(self):
        return self.memory_allocated()
    def empty_cache(self):
        self.calls.append("empty_cache")
    def clear_workspaces(self):
        self.calls.append("clear_workspaces")
        self.workspace = 0


def backend(cuda, with_clear=True):
    decider = OfficialDecider.__new__(OfficialDecider)
    decider.model = object()
    decider.engine = object()
    decider.torch = SimpleNamespace(cuda=cuda, _C=SimpleNamespace())
    if with_clear:
        decider.torch._C._cuda_clearCublasWorkspaces = cuda.clear_workspaces
    return decider


class CleanupTests(unittest.TestCase):
    def test_process_workspace_is_explicitly_cleared_before_empty_cache(self):
        cuda = Cuda()
        decider = backend(cuda)
        result = decider.close()
        self.assertEqual(result["allocated_before_workspace_clear_bytes"], 8 * 1024**2)
        self.assertEqual(result["allocated_after_cleanup_bytes"], 0)
        self.assertLess(cuda.calls.index("clear_workspaces"), cuda.calls.index("empty_cache"))
        self.assertIsNone(decider.model)
        self.assertIsNone(decider.engine)

    def test_real_remaining_allocation_still_fails_and_retains_byte_diagnostic(self):
        decider = backend(Cuda(live=37))
        result = {}
        with self.assertRaisesRegex(RuntimeError, "37 bytes"):
            release_decider(decider, "hobson", result)
        attempt = result["cleanup_attempts"][0]
        self.assertEqual(attempt["diagnostics"]["allocated_after_cleanup_bytes"], 37)
        self.assertGreaterEqual(attempt["seconds"], 0)

    def test_missing_workspace_api_never_bypasses_residual_allocation_check(self):
        decider = backend(Cuda(), with_clear=False)
        with self.assertRaises(RuntimeError):
            decider.close()
        self.assertFalse(decider.last_cleanup["cublas_workspace_clear_available"])
        zero = backend(Cuda(workspace=0), with_clear=False)
        self.assertEqual(zero.close()["allocated_after_cleanup_bytes"], 0)

    def test_cleanup_can_be_called_again_after_no_live_allocations(self):
        decider = backend(Cuda())
        decider.close()
        self.assertEqual(decider.close()["allocated_after_cleanup_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
