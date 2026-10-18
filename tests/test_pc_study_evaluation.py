import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import contextlib
import io

from pc_strands.data import Task, fingerprint, load_tasks, public_state, task_from_row
from pc_strands.metrics import evaluate, paired_delta
from pc_strands.evaluate import (GENERIC_FEEDBACK, Journal, OllamaWriter, UncertainRequest,
                                  generate_trajectories, replay_trajectories, export_common_schema,
                                  CachedHeadDecider)


class NoWait:
    idle_seconds = 0
    def before(self):
        pass
    def after(self, elapsed):
        pass


class FakeWriter:
    def __init__(self):
        self.calls = 0
        self.unloaded = False
        self.payloads = []
    def payload(self, state, previous=None, feedback=None):
        value = public_state(state, draft_required=False)
        value.update(previous=previous, feedback=feedback)
        self.payloads.append(value)
        return value
    def generate(self, payload):
        self.calls += 1
        return {"response": {"done": True, "done_reason": "stop", "message": {
                    "content": json.dumps({"answer": "right" if payload["previous"] is None else "wrong"})}},
                "input_tokens": 10, "output_tokens": 2, "elapsed_seconds": .1, "cost_usd": 0}
    answer = staticmethod(OllamaWriter.answer)
    def unload(self):
        self.unloaded = True


class FakeDecider:
    def __init__(self, p=.9):
        self.calls, self.states, self.p = 0, [], p
    def score(self, state, temperature=None):
        self.calls += 1
        self.states.append(state)
        return {"p_true": self.p, "input_tokens": 20, "elapsed_seconds": .01,
                "logits": [0, 2], "physical_input_tokens": 20}
    def offload(self):
        return 0
    def metadata(self):
        return {"checkpoint_head_sha256": "head"}


def config(maximum=3):
    return {"max_drafts": maximum, "feedback": GENERIC_FEEDBACK,
            "model_fingerprint": "fixed-test-model", "threshold": .8}


class StudyEvaluationTests(unittest.TestCase):
    def test_serving_profile_changes_the_frozen_writer_identity(self):
        from unittest.mock import patch
        writer = OllamaWriter()
        inventory = {"models": [{"name": "qwen3:8b", "digest": "pinned"}]}
        with patch.object(writer, "_request", return_value=inventory):
            with patch.dict("os.environ", {"OLLAMA_KV_CACHE_TYPE": "f16", "UNRELATED_PRIVATE_VALUE": "not exported"}):
                old = writer.identity()
            with patch.dict("os.environ", {"OLLAMA_KV_CACHE_TYPE": "q8_0", "UNRELATED_PRIVATE_VALUE": "not exported"}):
                new = writer.identity()
        self.assertNotEqual(fingerprint(old), fingerprint(new))
        self.assertNotIn("UNRELATED_PRIVATE_VALUE", new["serving_profile"])
        self.assertEqual(new["digest"], "pinned")

    def test_answer_semantics(self):
        self.assertTrue(evaluate("The Cat!", ["cat"])["exact_match"])
        self.assertEqual(evaluate("yes", ["yes indeed"])["token_f1"], 0)
        self.assertAlmostEqual(evaluate("Wilmette Illinois", ["Wilmette"])["token_f1"], 2 / 3)
        self.assertEqual(evaluate("a b b", ["b b c"])["token_f1"], .8)

    def test_public_allowlist(self):
        state = {"question": "Q", "evidence": ["E"], "draft": "D", "answer": "secret gold", "supporting_facts": []}
        self.assertEqual(set(public_state(state)), {"question", "evidence", "draft"})

    def test_native_contexts_and_partition_preserved(self):
        row = {"local_partition": "fit", "source_row": {"id": "x", "question": "Q", "answer": "A", "type": "bridge",
               "context": {"title": ["T", "U"], "sentences": [["s1", "s2"], ["s3"]]}}}
        task = task_from_row(row)
        self.assertEqual(task.partition, "fit")
        self.assertEqual(task.evidence, ["T\n[0] s1\n[1] s2", "U\n[0] s3"])
        self.assertNotIn("answers", task.public("draft"))

    def test_duplicate_question_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tasks.jsonl"
            p.write_text("\n".join(json.dumps({"id": i, "question": "The QUESTION?", "answer": "a", "context": [["T", ["E"]]]}) for i in ["1", "2"]))
            with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                load_tasks(p)

    def test_writer_payload_is_frozen_and_no_gold(self):
        writer = OllamaWriter()
        payload = writer.payload({"question": "q", "evidence": ["e"], "answers": ["gold"]}, "old", GENERIC_FEEDBACK)
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertEqual(payload["options"]["num_thread"], 4)
        self.assertNotIn("gold", json.dumps(payload))
        self.assertEqual(json.loads(payload["messages"][1]["content"])["revision_feedback"], GENERIC_FEEDBACK)

    def test_writer_guard_never_truncates(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            OllamaWriter().payload({"question": "q", "evidence": ["x" * 20000]})

    def test_length_stop_is_failure(self):
        with self.assertRaises(ValueError):
            OllamaWriter.answer({"response": {"done": True, "done_reason": "length"}})

    def test_request_resume_reuses_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "journal"
            j = Journal(p)
            self.assertEqual(j.call("1", {"a": 1}, lambda: {"result": 2}, NoWait()), {"result": 2})
            self.assertEqual(Journal(p).call("1", {"a": 1}, lambda: self.fail("repeated dispatch"), NoWait()), {"result": 2})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                Journal(p).call("1", {"a": 2}, lambda: {}, NoWait())

    def test_uncertain_dispatch_never_retries(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "journal"
            with self.assertRaises(UncertainRequest):
                Journal(p).call("1", {}, lambda: (_ for _ in ()).throw(TimeoutError()), NoWait())
            with self.assertRaises(UncertainRequest):
                Journal(p)

    def test_truncated_journal_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "journal"
            p.write_text('{"event":')
            with self.assertRaises(ValueError):
                Journal(p)

    def test_generation_resume_and_no_outcome_feedback(self):
        task = Task("hotpotqa:1", "q", ["e"], ["right"])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            writer = FakeWriter()
            rows = generate_trajectories([task], writer, out, config(), NoWait())
            self.assertEqual(writer.calls, 3)
            self.assertEqual([r["draft"] for r in rows[0]["rounds"]], ["right", "wrong", "wrong"])
            self.assertEqual(writer.payloads[1]["feedback"], GENERIC_FEEDBACK)
            self.assertTrue(writer.unloaded)
            second = FakeWriter()
            self.assertEqual(generate_trajectories([task], second, out, config(), NoWait(), resume=True), rows)
            self.assertEqual(second.calls, 0)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                generate_trajectories([task], second, out, config(2), NoWait(), resume=True)

    def test_replay_prefixes_are_not_gold_selected(self):
        task = Task("hotpotqa:1", "q", ["e"], ["right"])
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            trajectories = generate_trajectories([task], FakeWriter(), base / "gen", config(), NoWait())
            decider = FakeDecider(.1)
            eps = replay_trajectories([task], trajectories, decider, base / "eval", config(), NoWait())
            by = {e["policy"]: e for e in eps}
            self.assertTrue(by["fixed-1-prefix"]["exact_match"])
            self.assertFalse(by["fixed-2-prefix"]["exact_match"])
            self.assertFalse(by["strands-untuned"]["exact_match"])
            self.assertEqual(by["strands-untuned"]["rounds"], 3)
            self.assertEqual(decider.calls, 3)
            self.assertTrue(all(set(s) == {"question", "evidence", "draft"} for s in decider.states))
            self.assertIsNone(by["strands-untuned"]["live_elapsed_seconds"])

    def test_strict_overflow_is_failure_not_retry(self):
        class ContextWindowError(ValueError):
            token_count, max_length = 5000, 4096
        class Overflow(FakeDecider):
            def score(self, *args, **kwargs):
                raise ContextWindowError()
        task = Task("hotpotqa:1", "q", ["e"], ["right"])
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            tr = generate_trajectories([task], FakeWriter(), base / "g", config(), NoWait())
            eps = replay_trajectories([task], tr, Overflow(), base / "e", config(), NoWait())
            ep = next(e for e in eps if e["policy"] == "strands-untuned")
            self.assertEqual(ep["status"], "failed")
            self.assertEqual(ep["stop_reason"], "ContextWindowError")
            self.assertEqual(ep["rounds"], 1)  # future cached writer calls were not executed by this policy
            Journal(base / "e/requests.jsonl")

    def test_ten_drafts_and_three_draft_parity_are_separate(self):
        task = Task("hotpotqa:1", "q", ["e"], ["right"])
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            tr = generate_trajectories([task], FakeWriter(), base / "g", config(10), NoWait())
            eps = replay_trajectories([task], tr, FakeDecider(.1), base / "e", config(10), NoWait())
            by = {r["policy"]: r for r in eps}
            self.assertEqual(by["strands-untuned"]["rounds"], 10)
            self.assertEqual(by["strands-untuned-cap3"]["rounds"], 3)
            self.assertEqual(by["fixed-5-prefix"]["rounds"], 5)
            self.assertEqual(by["fixed-10-prefix"]["rounds"], 10)

    def test_baseline_receipt_hashes_final_files(self):
        task = Task("hotpotqa:1", "q", ["e"], ["right"])
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            tr = generate_trajectories([task], FakeWriter(), base / "g", config(), NoWait())
            cfg = {**config(), "phase": "untuned_pretraining"}
            replay_trajectories([task], tr, FakeDecider(), base / "e", cfg, NoWait())
            receipt = json.loads((base / "e/untuned-pretraining-receipt.json").read_text())
            for name in ("manifest", "episodes"):
                self.assertEqual(receipt[name + "_sha256"], hashlib.sha256((base / "e" / (name + ".json")).read_bytes()).hexdigest())

    def test_zero_draft_failure_kept_outside_v0_not_coerced(self):
        task = Task("hotpotqa:1", "q", ["e"], ["right"])
        with tempfile.TemporaryDirectory() as d:
            eps = [{"task_id": task.id, "policy": "strands", "rounds": 0, "status": "failed"}]
            self.assertEqual(export_common_schema(Path(d), eps, [task], {}), [])
            proof = json.loads((Path(d) / "common-schema-provenance.json").read_text())
            self.assertEqual(proof["planned_n"], 1)
            self.assertEqual(len(proof["omitted_unrepresentable_rows"]), 1)

    def test_failure_inclusive_twenty_question_baseline_gate(self):
        from pc_strands.train_head import validate_baseline_receipt
        class ContextWindowError(ValueError):
            token_count, max_length = 5000, 4096
        class LaterOverflow(FakeDecider):
            def score(self, state, temperature=None):
                if state["draft"] == "wrong":
                    raise ContextWindowError()
                return super().score(state, temperature)
        tasks = [Task("hotpotqa:" + str(i), "question " + str(i), ["e"], ["right"]) for i in range(20)]
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d)
            tr = generate_trajectories(tasks, FakeWriter(), base / "g", config(), NoWait())
            replay_trajectories(tasks, tr, LaterOverflow(.1), base / "e", {**config(), "phase": "untuned_pretraining"}, NoWait())
            receipt = json.loads((base / "e/untuned-pretraining-receipt.json").read_text())
            self.assertEqual(receipt["attempted"], 20)
            self.assertEqual(receipt["completed"], 0)
            self.assertEqual(receipt["failed"], 20)
            self.assertEqual(receipt["valid_decision_forwards"], 20)
            self.assertEqual(len(validate_baseline_receipt(receipt, "head")), 20)

    def test_cached_encoder_time_is_counterfactual_not_physical_tokens(self):
        class Cached(FakeDecider):
            def score(self, state, temperature=None):
                return {**super().score(state, temperature), "physical_input_tokens": 0,
                        "reused_encoding": True, "encoding_seconds": 3}
        task = Task("hotpotqa:1", "q", ["e"], ["right"])
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            tr = generate_trajectories([task], FakeWriter(), base / "g", config(), NoWait())
            eps = replay_trajectories([task], tr, Cached(), base / "e", config(), NoWait())
            ep = next(r for r in eps if r["policy"] == "strands-untuned")
            self.assertAlmostEqual(ep["observed_prefix_model_seconds"], 3.11)
            manifest = json.loads((base / "e/manifest.json").read_text())
            self.assertEqual(manifest["actual_evaluation_usage"]["reported_physical_input_tokens"], 0)

    def test_head_replay_rejects_changed_cache_before_loading_torch(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "cache.jsonl"
            cache.write_text("")
            Path(str(cache) + ".manifest.json").write_text(json.dumps({"status": "completed", "unreconciled_inferences": 0,
                "cached_rows_sha256": "changed"}))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                CachedHeadDecider(Path(d) / "checkpoint", cache)

    def test_paired_interval_uses_planned_ids(self):
        a = {"1": {"status": "completed", "exact_match": True}}
        b = {"1": {"status": "completed", "exact_match": False}, "2": {"status": "completed", "exact_match": True}}
        result = paired_delta(a, b, ["1", "2"], "exact_match", draws=100)
        self.assertEqual(result["candidate_minus_baseline"], 0)
        self.assertEqual(result["planned_n"], 2)
        self.assertEqual(result["complete_pair_n"], 1)

    def test_cli_backend_construction_and_metadata_follow_owned_admission(self):
        from unittest.mock import patch
        from pc_strands.control import GpuLease as ActualLease, ResourcePressure
        from pc_strands.evaluate import main, WRITER_SYSTEM
        for mode in ("replay", "head-replay"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                base = Path(d)
                dataset = base / "tasks.jsonl"
                dataset.write_text(json.dumps({"id": "hotpotqa:1", "question": "q", "evidence": ["e"],
                                               "answers": ["right"], "split": "dev"}) + "\n")
                tasks = load_tasks(dataset)
                generate_trajectories(tasks, FakeWriter(), base / "source", {**config(1), "writer_system": WRITER_SYSTEM}, NoWait())
                checkpoint = base / "checkpoint"
                checkpoint.mkdir()
                state, events = {"held": False, "admitted": False}, []
                test = self
                lock = base / "shared.lock"
                class ObservedLease:
                    def __init__(self, path, purpose):
                        test.assertEqual(path, lock)
                        self.actual = ActualLease(path, purpose)
                    def __enter__(self):
                        self.actual.__enter__()
                        state["held"] = True
                        events.append("owned")
                        return self
                    def __exit__(self, *args):
                        self.actual.__exit__(*args)
                        state["held"] = False
                        events.append("released")
                class Admission(NoWait):
                    def __init__(self, *args, **kwargs):
                        self.deadline = float("inf")
                    def before(self):
                        test.assertTrue(state["held"])
                        state["admitted"] = True
                        events.append("admitted")
                class Backend(FakeDecider):
                    def __init__(self, *args, **kwargs):
                        test.assertTrue(state["held"] and state["admitted"])
                        # Prove the actual OS lock, rather than only a sentinel.
                        with test.assertRaises(ResourcePressure):
                            with ActualLease(lock, "conflicting startup"):
                                pass
                        events.append("constructed")
                        super().__init__()
                        self.frozen = {"trajectories_sha256": hashlib.sha256((base / "source/trajectories.json").read_bytes()).hexdigest()}
                        self.cache_manifest_sha256 = "observed-cache-manifest"
                    def metadata(self):
                        test.assertTrue(state["held"] and state["admitted"])
                        return {**super().metadata(), "feature_cache_sha256": "observed-cache"}
                    def offload(self):
                        test.assertTrue(state["held"])
                        events.append("offloaded")
                        return 0
                with patch("pc_strands.evaluate.GpuLease", ObservedLease), patch("pc_strands.evaluate.Governor", Admission), \
                     patch("pc_strands.decider.StrandsDecider", Backend), patch("pc_strands.evaluate.CachedHeadDecider", Backend):
                    main([mode, "--dataset", str(dataset), "--output", str(base / "result"),
                          "--trajectory-dir", str(base / "source"), "--checkpoint-dir", str(checkpoint),
                          "--base-dir", str(base / "base"), "--feature-cache", str(base / "cache.jsonl"),
                          "--max-drafts", "1", "--gpu-lock", str(lock), "--deadline", "2099-01-01T20:00:00-05:00"])
                self.assertEqual(events[:3], ["owned", "admitted", "constructed"])
                self.assertEqual(events[-2:], ["offloaded", "released"])

    def test_cli_denied_startup_never_constructs_backend(self):
        from unittest.mock import patch
        from pc_strands.evaluate import main, WRITER_SYSTEM
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d)
            dataset = base / "tasks.jsonl"
            dataset.write_text(json.dumps({"id": "hotpotqa:1", "question": "q", "evidence": ["e"], "answers": ["right"]}) + "\n")
            generate_trajectories(load_tasks(dataset), FakeWriter(), base / "source", {**config(1), "writer_system": WRITER_SYSTEM}, NoWait())
            class Denied(NoWait):
                def __init__(self, *args, **kwargs):
                    pass
                def before(self):
                    raise RuntimeError("startup admission denied")
            with patch("pc_strands.evaluate.Governor", Denied), patch("pc_strands.decider.StrandsDecider") as constructor:
                with self.assertRaisesRegex(RuntimeError, "startup admission denied"):
                    main(["replay", "--dataset", str(dataset), "--output", str(base / "result"),
                          "--trajectory-dir", str(base / "source"), "--checkpoint-dir", str(base / "checkpoint"),
                          "--base-dir", str(base / "base"), "--max-drafts", "1", "--gpu-lock", str(base / "shared.lock")])
                constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
