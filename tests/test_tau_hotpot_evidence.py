"""Counterfeit/stale evidence mutations must fail before real head updates."""
from argparse import Namespace
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from haltiq_hotpot.protocol import digest
from haltiq_tau_hotpot.journal import CallJournal, durable_write, fingerprint
from haltiq_tau_hotpot.hobson import stable_hash
from haltiq_tau_hotpot.worker import verify_dispatch, verify_native_rows


class EvidenceBoundaryTests(unittest.TestCase):
    def test_native_rows_join_actual_response_public_input_gold_and_cache_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            draft = {"answer": "A", "supporting_facts": [["Public", 0]]}
            case = {"partition": "engineering_heldout", "public": {"id": "q", "question": "Who?",
                "type": "bridge", "level": "hard", "context": [["Public", ["A is correct."]]]},
                "evaluator": {"answer": "A", "supporting_facts": [["Public", 0]]}}
            cache = run / "synthetic-fixture.safetensors"
            cache.write_bytes(b"SYNTHETIC TEST BYTES; NEVER MODEL EVIDENCE")
            observed = {"physical_forwards": 1, "retrieval_indices": [0], "feature": {
                "path": str(cache), "sha256": digest(cache), "provenance": {
                    "state_sha256": stable_hash("Question: Who?\nEvidence: Public: A is correct.\nDraft answer: A")}}}
            physical = CallJournal(run / "features/physical")
            physical.call("native_answer", {"synthetic": True}, lambda: observed)
            durable_write(run / "collect/episodes.json", [{"question_id": "q", "episode": {"submitted": draft}}])
            rows = [{"question_id": "q", "partition": "engineering_heldout", "criterion": "answer",
                     "status": "completed", "label": 1, "observed": observed}]
            baseline = {"rows_sha256": fingerprint(rows)}
            args = Namespace(previous=run)
            verify_native_rows(args, {}, {"heldout": [case]}, rows, baseline)
            for change in ("probability", "label", "state", "baseline"):
                mutated, proof = deepcopy(rows), dict(baseline)
                if change == "probability": mutated[0]["observed"]["p_true"] = .9
                if change == "label": mutated[0]["label"] = 0
                if change == "state": mutated[0]["observed"]["feature"]["provenance"]["state_sha256"] = "x" * 64
                if change == "baseline": proof["rows_sha256"] = "x" * 64
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify_native_rows(args, {}, {"heldout": [case]}, mutated, proof)
            cache.write_bytes(b"CHANGED")
            with self.assertRaises(ValueError):
                verify_native_rows(args, {}, {"heldout": [case]}, rows, baseline)

    def test_actual_worker_arguments_cannot_escape_frozen_current_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp).resolve()
            args = Namespace(previous=run, phase="collect", config=run / "config.json",
                output=run / "collect", control=run / "workers/collect", deadline="2030-01-01T00:00:00+00:00")
            plan = {"deadline": args.deadline, "qualification_protocol": {"path": str(args.config)},
                    "stages": [{"id": "collect"}]}
            state = {"run_id": "run", "plan_sha256": "sha", "stages": {"collect": {"status": "started"}}}
            env = {"HALTIQ_RUN_ID": "run", "HALTIQ_PLAN_SHA256": "sha", "HALTIQ_PHASE": "collect"}
            with patch("haltiq_tau_hotpot.worker.__file__", str(run / "source/haltiq_tau_hotpot/worker.py")), patch.dict(os.environ, env):
                verify_dispatch(args, plan, state)
                for name, value in (("config", run / "other.json"), ("output", run / "other"),
                                    ("control", run / "other"), ("deadline", "2030-02-01T00:00:00+00:00")):
                    changed = Namespace(**vars(args))
                    setattr(changed, name, value)
                    with self.subTest(argument=name), self.assertRaises(ValueError):
                        verify_dispatch(changed, plan, state)


if __name__ == "__main__":
    unittest.main()
