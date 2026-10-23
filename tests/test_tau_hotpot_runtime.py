"""Mock-only runtime reliability checks; no models, GPU probes, HTTP or processes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from haltiq_tau_hotpot.journal import UncertainCall, fingerprint
from haltiq_tau_hotpot.runtime import LocalRuntime
from haltiq_tau_hotpot.generation import MAIN_PROFILE


class Control:
    def __init__(self, directory):
        self.directory = directory
        self.deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
        self.active_seconds = self.idle_seconds = 0
        self.calls = []
    def _check_stop(self): pass
    def before_work(self, **kwargs): self.calls.append(("guard", kwargs))
    def rest_after_block(self, seconds): self.calls.append(("rest", seconds))


class TensorFixture:
    """Only the cache serialization interface, never a numeric/model substitute."""
    def contiguous(self): return self


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = {"gpu_lease": str(self.root / "gpu.lock"), "writer_digest": "pinned-writer",
            "ollama_models": str(self.root / "models"), "ollama": "NEVER-RUN.exe",
            "reranker": "NEVER-LOAD", "checkpoint": "NEVER-LOAD", "base": "NEVER-LOAD"}
        self.control = Control(self.root / "control")
        self.runtime = LocalRuntime(self.config, self.root / "runtime", self.control)
        self.public = {"question": "Synthetic public question?", "context": [
            ["low", ["zero"]], ["middle", ["one"]], ["high", ["two"]]]}
        self.draft = {"answer": "synthetic", "supporting_facts": [["high", 0]]}

    def fake_modules(self):
        package, module = ModuleType("safetensors"), ModuleType("safetensors.torch")
        def write(tensors, path):
            self.assertEqual(set(tensors), {"decide", "options"})
            Path(path).write_bytes(b"SYNTHETIC-CACHE-NOT-MODEL-TENSORS")
        module.save_file = write
        return {"safetensors": package, "safetensors.torch": module}

    def test_physical_result_is_durable_before_postcall_stop_and_never_retried(self):
        self.control.rest_after_block = Mock(side_effect=RuntimeError("post-call stop"))
        operation = Mock(return_value={"physical_forwards": 1, "value": "actual-fixture-return"})
        with self.assertRaisesRegex(RuntimeError, "post-call stop"):
            self.runtime._physical("writer", {"input": "fixture"}, operation)
        self.assertEqual(operation.call_count, 1)
        self.assertEqual(self.runtime.journal.verify()["completed"], 1)
        self.assertFalse(self.runtime.journal.summary()["pending_or_uncertain"])
        response = next((self.runtime.directory / "physical/responses").glob("*.json"))
        self.assertEqual(json.loads(response.read_text())["value"], "actual-fixture-return")

    def test_lease_is_held_before_ranker_construction_and_only_owned_server_is_started(self):
        events, held = [], {"value": False}
        class Lease:
            def __init__(self, *args): pass
            def __enter__(self):
                if held["value"]: raise RuntimeError("duplicate owner")
                held["value"] = True
                events.append("lease_acquired")
                return self
            def __exit__(self, *args):
                held["value"] = False
                events.append("lease_released")
        class Ranker:
            def __init__(self, path):
                self_test.assertTrue(held["value"])
                events.append("ranker_constructed")
            def close(self): events.append("ranker_closed")
        self_test = self
        server = SimpleNamespace(code=None)
        server.poll = lambda: server.code
        server.terminate = lambda: setattr(server, "code", 0)
        server.wait = lambda **kwargs: server.code
        server.kill = lambda: setattr(server, "code", -9)
        self.runtime._http = lambda path, **kwargs: {"models": []} if path == "/api/ps" else {"version": "fixture"}
        with patch("haltiq_tau_hotpot.runtime.GpuLease", Lease), \
             patch("haltiq_tau_hotpot.runtime.port_open", return_value=False), \
             patch("haltiq_hotpot.native.MiniLMRanker", Ranker), \
             patch("haltiq_tau_hotpot.runtime.subprocess.Popen", return_value=server) as spawn:
            with self.runtime:
                self.assertTrue(held["value"])
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(spawn.call_args.args[0], ["NEVER-RUN.exe", "serve"])
            self.assertEqual(spawn.call_args.kwargs["stdin"], subprocess.DEVNULL)
            environment = spawn.call_args.kwargs["env"]
            self.assertEqual(environment["OLLAMA_MODELS"], str(Path(self.config["ollama_models"]).resolve()))
            self.assertEqual(environment["OLLAMA_HOST"], "127.0.0.1:11435")
            self.assertEqual(environment["OLLAMA_KV_CACHE_TYPE"], "q8_0")
        self.assertEqual(events[:2], ["lease_acquired", "ranker_constructed"])
        self.assertFalse(held["value"])
        self.assertTrue(self.runtime.cleanup["verified"])

    def test_transport_error_remains_uncertain_without_postcall_rest_or_retry(self):
        operation = Mock(side_effect=OSError("fixture transport failure"))
        with self.assertRaises(UncertainCall):
            self.runtime._physical("writer", {"input": "fixture"}, operation)
        self.assertEqual(operation.call_count, 1)
        self.assertTrue(self.runtime.journal.summary()["pending_or_uncertain"])
        self.assertFalse(any(kind == "rest" for kind, _ in self.control.calls))

    def test_retrieval_cache_reuses_exact_public_query_but_draft_change_is_new_work(self):
        self.runtime.ranker = SimpleNamespace(rank=Mock(return_value={"scores": [.1, .5, .9], "physical_forwards": 1}))
        first = self.runtime.rank(self.public, "first draft")
        repeated = self.runtime.rank(self.public, "first draft")
        changed = self.runtime.rank(self.public, "second draft")
        self.assertEqual(first["indices"], [2, 1, 0])
        self.assertFalse(first["cache_reused"])
        self.assertTrue(repeated["cache_reused"])
        self.assertFalse(changed["cache_reused"])
        self.assertEqual(self.runtime.ranker.rank.call_count, 2)
        self.assertEqual(self.runtime.journal.summary()["by_role"], {"retrieval": 2})

    def test_observe_preserves_rank_order_and_preflights_before_native_pending(self):
        seen = []
        def preflight(state, **kwargs):
            seen.append(state)
            self.assertEqual(self.runtime.journal.summary()["dispatched"], 0)
            self.assertLess(state.index("high: two"), state.index("middle: one"))
            self.assertLess(state.index("middle: one"), state.index("low: zero"))
            return {"prepared": True}
        def forward(state, **kwargs):
            self.assertEqual(self.runtime.journal.summary()["dispatched"], 1)
            self.assertEqual(kwargs["prepared"], {"prepared": True})
            return {"physical_forwards": 1, "head_scores": 2,
                "feature": {"decide": TensorFixture(), "options": TensorFixture(), "provenance_sha256": "fixture"}}
        self.runtime.rank = Mock(return_value={"indices": [2, 1, 0]})
        self.runtime._native_ready = Mock()
        self.runtime.backend = SimpleNamespace(prepare_state=Mock(side_effect=preflight), observe=Mock(side_effect=forward))
        with patch.dict(sys.modules, self.fake_modules()):
            one = self.runtime.observe(self.public, self.draft)
            # A repeated native observation is actual new work and gets a new cache file.
            self.runtime.backend.prepare_state.side_effect = lambda *a, **kw: {"prepared": True}
            self.runtime.backend.observe.side_effect = lambda *a, **kw: {"physical_forwards": 1, "head_scores": 2,
                "feature": {"decide": TensorFixture(), "options": TensorFixture(), "provenance_sha256": "fixture"}}
            two = self.runtime.observe(self.public, self.draft)
        self.assertEqual(one["retrieval_indices"], [2, 1, 0])
        self.assertNotEqual(one["feature"]["path"], two["feature"]["path"])
        self.assertEqual(self.runtime.journal.summary()["by_role"], {"native_answer": 2})
        self.assertTrue(all(Path(r["feature"]["path"]).is_file() for r in (one, two)))
        self.assertNotIn("Citations:", seen[0])

    def test_window_failure_is_predispatch_without_native_calls_or_uncertain_journal(self):
        self.runtime.rank = Mock(return_value={"indices": [2, 1, 0]})
        self.runtime._native_ready = Mock()
        self.runtime.backend = SimpleNamespace(prepare_state=Mock(side_effect=ValueError("context window")), observe=Mock())
        with self.assertRaisesRegex(ValueError, "context window"):
            self.runtime.observe(self.public, self.draft)
        self.runtime.backend.observe.assert_not_called()
        self.assertEqual(self.runtime.journal.summary()["dispatched"], 0)
        self.assertFalse(self.runtime.journal.summary()["pending_or_uncertain"])

    def test_main_native_cache_reuses_only_hash_bound_real_observation_with_one_cpu_score(self):
        self.runtime.config["native_feature_cache"] = True
        checkpoint = self.root / "released"
        checkpoint.mkdir()
        (checkpoint / "head.safetensors").write_bytes(b"SYNTHETIC HEAD; NEVER LOADED")
        self.runtime.config["checkpoint"] = str(checkpoint)
        self.runtime.rank = Mock(return_value={"indices": [2, 1, 0]})
        self.runtime._native_ready = Mock()
        def forward(state, **kwargs):
            return {"physical_forwards": 1, "head_scores": 2, "feature": {"decide": TensorFixture(),
                "options": TensorFixture(), "provenance_sha256": "synthetic",
                "provenance": {"state_sha256": fingerprint(state)}}}
        self.runtime.backend = SimpleNamespace(prepare_state=Mock(return_value={}), observe=Mock(side_effect=forward))
        modules = self.fake_modules()
        modules["safetensors.torch"].load_file = lambda *args, **kwargs: {
            "decide": TensorFixture(), "options": TensorFixture()}
        score = {"physical_forwards": 0, "head_scores": 1, "p_true": .9}
        with patch.dict(sys.modules, modules), \
             patch("haltiq_tau_hotpot.hobson.load_head", return_value="SYNTHETIC-HEAD-NOT-A-MODEL"), \
             patch("haltiq_tau_hotpot.hobson.score_head", return_value=score) as cpu_score:
            first = self.runtime.observe(self.public, self.draft)
            self.runtime.writer_loaded = True
            second = self.runtime.observe(self.public, self.draft)
            self.assertEqual(self.runtime._native_ready.call_count, 1)
            self.assertTrue(self.runtime.writer_loaded)
            self.assertEqual(second["physical_forwards"], 0)
            self.assertEqual(second["head_scores"], 1)
            self.assertTrue(second["cache_reused"])
            self.assertEqual(cpu_score.call_count, 1)
            self.assertEqual(self.runtime.journal.summary()["by_role"], {"native_answer": 1, "cached_head_answer": 1})
            entry = next(iter(self.runtime.native_cache.values()))
            original_id = entry["physical_event"]["id"]
            entry["physical_event"]["id"] = "forged"
            with self.assertRaisesRegex(ValueError, "cache evidence changed"):
                self.runtime.observe(self.public, self.draft)
            entry["physical_event"]["id"] = original_id
            original_state = entry["request"]["state"]
            entry["request"]["state"] = "another public state"
            with self.assertRaisesRegex(ValueError, "cache evidence changed"):
                self.runtime.observe(self.public, self.draft)
            entry["request"]["state"] = original_state
            trained = self.root / "trained.safetensors"
            trained.write_bytes(b"DIFFERENT SYNTHETIC HEAD")
            self.runtime.observe(self.public, self.draft, head_path=trained, temperature=2.)
            self.assertEqual(cpu_score.call_count, 2)
            self.assertEqual(cpu_score.call_args.kwargs["temperature"], 2.)
            Path(first["feature"]["path"]).write_bytes(b"CHANGED")
            with self.assertRaisesRegex(ValueError, "cache evidence changed"):
                self.runtime.observe(self.public, self.draft)
            self.assertEqual(cpu_score.call_count, 2)
            self.assertEqual(self.runtime._native_ready.call_count, 1)

    def test_native_startup_orders_writer_unload_admission_activation_then_postload_guard(self):
        events = []
        self.runtime._unload_writer = lambda: events.append("writer_unloaded")
        self.runtime._startup_guard = lambda: events.append("admitted")
        self.control.before_work = lambda **kw: events.append("postload_guard")
        class Backend:
            metadata = {"fixture": True}
            def __init__(self, *args): events.append("construct_after_admission")
            def activate(self): events.append("activate")
        with patch("haltiq_tau_hotpot.hobson.HobsonBackend", Backend):
            self.runtime._native_ready()
        self.assertEqual(events, ["writer_unloaded", "admitted", "construct_after_admission", "activate", "postload_guard"])
        self.assertEqual(self.runtime.journal.summary()["by_role"], {"native_startup": 1})

    def test_writer_cannot_load_until_native_cleanup_and_admission_succeed(self):
        events = []
        self.runtime._close_native = lambda: events.append("native_closed")
        self.runtime._startup_guard = lambda: events.append("admitted")
        def request(path, data=None, **kwargs):
            events.append("load" if path == "/api/generate" else "resident_identity")
            return {"models": [{"digest": "pinned-writer"}]} if path == "/api/ps" else {}
        self.runtime._http = request
        self.control.before_work = lambda **kwargs: events.append("postload_guard")
        self.runtime._writer_ready()
        self.assertEqual(events, ["native_closed", "admitted", "load", "resident_identity", "postload_guard"])
        self.assertTrue(self.runtime.writer_loaded)
        self.runtime.writer_loaded = False
        self.runtime._close_native = Mock(side_effect=RuntimeError("cleanup failure"))
        self.runtime._http = Mock()
        with self.assertRaises(RuntimeError): self.runtime._writer_ready()
        self.runtime._http.assert_not_called()

    def test_wrong_resident_writer_identity_blocks_generation(self):
        self.runtime._close_native = Mock()
        self.runtime._startup_guard = Mock()
        self.runtime._http = Mock(side_effect=[{}, {"models": [{"digest": "foreign"}]}])
        with self.assertRaisesRegex(RuntimeError, "resident writer identity"):
            self.runtime._writer_ready()
        self.assertFalse(self.runtime.writer_loaded)
        self.assertEqual(self.runtime._http.call_count, 2)

    def test_evicted_writer_after_pause_requires_startup_admission_before_reload(self):
        events = []
        self.runtime.writer_loaded = True
        self.runtime._close_native = lambda: events.append("native_closed")
        self.runtime._startup_guard = lambda: events.append("startup_admitted")
        responses = iter([{"models": []}, {}, {"models": [{"digest": "pinned-writer"}]}])
        def request(path, *args, **kwargs):
            events.append(path)
            return next(responses)
        self.runtime._http = request
        self.runtime._writer_ready()
        self.assertEqual(events, ["/api/ps", "native_closed", "startup_admitted", "/api/generate", "/api/ps"])
        self.assertTrue(self.runtime.writer_loaded)
        self.assertEqual(self.runtime.journal.verify()["by_role"], {"writer_load": 1})
        self.runtime._http = Mock(return_value={"models": [{"digest": "foreign"}]})
        with self.assertRaisesRegex(RuntimeError, "resident writer identity"):
            self.runtime._writer_ready()
        self.runtime._http.assert_called_once()

    def test_unload_requires_empty_resident_models_and_records_physical_request_once(self):
        self.runtime.writer_loaded = True
        self.runtime._http = Mock(side_effect=[{}, {"models": [{"digest": "still-owned"}]}])
        with self.assertRaisesRegex(RuntimeError, "unload is not verified"):
            self.runtime._unload_writer()
        self.assertEqual(self.runtime.journal.summary()["by_role"], {"writer_unload": 1})

    def test_http_uses_private_local_url_and_rejects_oversized_response(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return b"x" * size
        with patch("haltiq_tau_hotpot.runtime.urlopen", return_value=Response()) as request:
            with self.assertRaisesRegex(ValueError, "bounded payload"):
                self.runtime._http("/api/chat", {"fixture": True})
        self.assertEqual(request.call_args.args[0].full_url, "http://127.0.0.1:11435/api/chat")

    def test_generation_schema_and_output_cap_checks_do_not_repeat_request(self):
        self.runtime._writer_ready = Mock()
        self.runtime._http = Mock(return_value={"done": True, "done_reason": "length"})
        with self.assertRaisesRegex(ValueError, "output budget"):
            self.runtime.generate({"evidence": "fixture"})
        self.assertEqual(self.runtime._http.call_count, 1)
        self.assertEqual(self.runtime.journal.summary()["by_role"], {"writer": 1})
        self.runtime._http.reset_mock()
        with self.assertRaisesRegex(ValueError, "context bound"):
            self.runtime.generate({"evidence": "x" * 17000})
        self.runtime._http.assert_not_called()

    def test_exact_response_cache_reuses_only_identical_role_prompt_and_feedback(self):
        self.runtime.config["response_cache"] = True
        self.runtime._writer_ready = Mock()
        self.runtime._http = Mock(return_value={"done": True, "done_reason": "stop",
            "message": {"content": json.dumps(self.draft)}})
        state = {"question": "fixture", "evidence": "public passage"}
        first = self.runtime.generate(state)
        self.assertEqual(self.runtime.generate(state), first)
        self.assertEqual(self.runtime._writer_ready.call_count, 1)
        self.assertEqual(self.runtime._http.call_count, 1)
        self.assertEqual(self.runtime.journal.summary()["exact_response_reuses"], 1)
        self.runtime.generate(state, "different feedback")
        self.assertEqual(self.runtime._http.call_count, 2)
        self.runtime._http.return_value = {"done": True, "message": {"content": "Retain the answer."}}
        self.assertEqual(self.runtime.generate(state, review=True), "Retain the answer.")
        self.assertEqual(self.runtime.generate(state, review=True), "Retain the answer.")
        self.assertEqual(self.runtime._http.call_count, 3)
        self.assertEqual(self.runtime.journal.verify()["by_role"], {"writer": 2, "reviewer": 1})
        self.assertEqual(self.runtime.journal.summary()["exact_response_reuses"], 2)
        self.runtime.config["writer_digest"] = "changed-model"
        self.runtime.generate(state, review=True)
        self.assertEqual(self.runtime._http.call_count, 4)

    def test_main_role_budget_and_instruction_are_physical_payload_and_cache_identity(self):
        from copy import deepcopy
        self.runtime.config.update(purpose="runtime_probe", generation_profile=deepcopy(MAIN_PROFILE), response_cache=True)
        self.runtime._writer_ready = Mock()
        self.runtime._http = Mock(return_value={"done": True, "done_reason": "stop", "message": {"content": "Retain."}})
        self.runtime.generate({"question": "Synthetic?"}, review=True, profile="legacy")
        first = self.runtime._http.call_args.args[1]
        self.runtime.generate({"question": "Synthetic?"}, review=True, profile="main")
        second = self.runtime._http.call_args.args[1]
        self.assertEqual(first["options"]["num_predict"], 384)
        self.assertEqual(second["options"]["num_predict"], 768)
        self.assertTrue(second["messages"][0]["content"].endswith(MAIN_PROFILE["reviewer_instruction"]))
        self.runtime.generate({"question": "Synthetic?"}, review=True, profile="main")
        self.assertEqual(self.runtime._http.call_count, 2)
        self.assertEqual(self.runtime.journal.verify()["exact_response_reuses"], 1)
        self.runtime.config["purpose"] = "qualification_recovery"
        with self.assertRaises(ValueError): self.runtime.generate({"question": "Synthetic?"}, review=True)
        self.assertEqual(self.runtime._http.call_count, 2)

    def test_known_invalid_terminal_output_is_replayed_as_failure_without_resampling(self):
        self.runtime.config["response_cache"] = True
        self.runtime._writer_ready = Mock()
        self.runtime._http = Mock(return_value={"done": True, "done_reason": "length"})
        with self.assertRaises(ValueError):
            self.runtime.generate({"evidence": "fixture"})
        self.assertEqual(len(self.runtime.response_cache), 1)
        with self.assertRaises(ValueError):
            self.runtime.generate({"evidence": "fixture"})
        self.assertEqual(self.runtime._http.call_count, 1)
        self.assertEqual(self.runtime.journal.summary()["exact_response_reuses"], 1)
        self.runtime._http.return_value = {"done": True, "message": {"content": "[]"}}
        with self.assertRaisesRegex(ValueError, "schema"):
            self.runtime.generate({"evidence": "different public input"})
        with self.assertRaisesRegex(ValueError, "schema"):
            self.runtime.generate({"evidence": "different public input"})
        self.assertEqual(len(self.runtime.response_cache), 2)
        self.assertEqual(self.runtime._http.call_count, 2)
        self.assertEqual(self.runtime.journal.summary()["exact_response_reuses"], 2)

    def test_empty_review_reuses_known_failure_but_unknown_transport_never_enters_cache(self):
        self.runtime.config["response_cache"] = True
        self.runtime._writer_ready = Mock()
        self.runtime._http = Mock(return_value={"done": True, "message": {"content": ""}})
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "empty"):
                self.runtime.generate({"evidence": "fixture"}, review=True)
        self.assertEqual(self.runtime._http.call_count, 1)
        self.assertEqual(self.runtime.journal.summary()["exact_response_reuses"], 1)
        before = len(self.runtime.response_cache)
        self.runtime._http.side_effect = TimeoutError("actual completion unknown")
        with self.assertRaises(UncertainCall):
            self.runtime.generate({"evidence": "changed input"}, review=True)
        self.assertEqual(len(self.runtime.response_cache), before)
        self.assertTrue(self.runtime.journal.summary()["pending_or_uncertain"])

    def test_failed_startup_wait_still_has_honest_idle_accounting(self):
        guard = SimpleNamespace(idle_seconds=120., before_work=Mock(side_effect=RuntimeError("RAM hold expired")))
        with patch("haltiq_tau_hotpot.runtime.RunControl", return_value=guard):
            with self.assertRaisesRegex(RuntimeError, "RAM hold"):
                self.runtime._startup_guard()
        self.assertEqual(self.control.idle_seconds, 120.)

    def test_mutated_cached_response_stops_before_new_load_or_dispatch(self):
        self.runtime.config["response_cache"] = True
        self.runtime._writer_ready = Mock()
        self.runtime._http = Mock(return_value={"done": True,
            "message": {"content": json.dumps(self.draft)}})
        self.runtime.generate({"evidence": "fixture"})
        event = self.runtime.journal.records[-1]
        (self.runtime.journal.directory / event["response"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.runtime.generate({"evidence": "fixture"})
        self.assertEqual(self.runtime._http.call_count, 1)
        self.assertEqual(self.runtime._writer_ready.call_count, 1)

    def cleanup_fixture(self, *, failed_actions=()):
        events = []
        def action(name):
            def perform(*args, **kwargs):
                events.append(name)
                if name in failed_actions: raise OSError(name + " fixture failure")
            return perform
        held = {"active": True}
        class Lease:
            def __enter__(self):
                if held["active"]: raise RuntimeError("still locked")
                events.append("probe_acquired")
                held["active"] = True
                return self
            def __exit__(self, *args):
                events.append("probe_released")
                held["active"] = False
        class Owned:
            def __exit__(self, *args):
                events.append("owned_lease_released")
                held["active"] = False
        self.runtime.lease = Owned()
        self.runtime._close_native = action("native_close")
        self.runtime._unload_writer = action("writer_unload")
        self.runtime.server = SimpleNamespace(poll=lambda: None, terminate=action("terminate"),
            wait=action("wait"), kill=action("kill"))
        self.runtime.log = SimpleNamespace(close=action("log_close"))
        self.runtime.ranker = SimpleNamespace(close=action("ranker_close"))
        return events, Lease

    def test_cleanup_errors_still_release_and_probe_lease_then_preserve_failed_cleanup(self):
        events, lease = self.cleanup_fixture(failed_actions={"native_close", "terminate", "wait", "log_close", "ranker_close"})
        with patch("haltiq_tau_hotpot.runtime.GpuLease", side_effect=lambda *args: lease()), \
             patch("haltiq_tau_hotpot.runtime.port_open", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                self.runtime.__exit__(None, None, None)
        self.assertLess(events.index("ranker_close"), events.index("owned_lease_released"))
        self.assertLess(events.index("owned_lease_released"), events.index("probe_acquired"))
        self.assertTrue(self.runtime.cleanup["gpu_lease_released"])
        self.assertFalse(self.runtime.cleanup["verified"])
        self.assertGreaterEqual(len(self.runtime.cleanup["errors"]), 5)

    def test_cleanup_does_not_mask_existing_exception_and_open_port_never_verifies(self):
        events, lease = self.cleanup_fixture(failed_actions={"ranker_close"})
        with patch("haltiq_tau_hotpot.runtime.GpuLease", side_effect=lambda *args: lease()), \
             patch("haltiq_tau_hotpot.runtime.port_open", return_value=True):
            self.runtime.__exit__(RuntimeError, RuntimeError("original"), None)
        self.assertIn("owned_lease_released", events)
        self.assertFalse(self.runtime.cleanup["verified"])
        self.assertFalse(self.runtime.cleanup["private_port_closed"])

    def test_residual_native_allocation_blocks_next_writer_even_after_close_returned(self):
        cuda = SimpleNamespace(memory_allocated=lambda: 128, memory_reserved=lambda: 512)
        backend = SimpleNamespace(torch=SimpleNamespace(cuda=cuda), close=Mock())
        self.runtime.backend = backend
        self.runtime._startup_guard = Mock()
        self.runtime._http = Mock()
        with self.assertRaisesRegex(RuntimeError, "Native GPU tensors remain"):
            self.runtime._writer_ready()
        backend.close.assert_called_once()
        self.assertEqual(self.runtime.native_unloads[-1]["allocated_bytes"], 128)
        self.runtime._startup_guard.assert_not_called()
        self.runtime._http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
