"""Synthetic report contracts only; no API, model or benchmark evidence."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import shutil
import tempfile
import unittest

from tools import build_strands_report as report
from pc_strands.compare import compare_runs
from pc_strands.data import fingerprint, load_tasks
from test_compare import fixture as comparison_fixture

AS_OF = "2026-10-18T20:00:00-05:00"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, terminal=False):
    train = root / "study-runs/train"
    ids = ["train0", "train1", "train2"]
    write(train / "manifest.json", {"status": "interrupted", "python": "fixture",
        "frozen": {"task_ids": ids, "config": {}}, "writer_unload_verified": True,
        "actual_evaluation_usage": {"completed_requests_including_invalid_outputs": 2,
            "reported_physical_input_tokens": 100, "hosted_cost_usd": 0}})
    write(train / "trajectories.json", [{"task_id": ids[0], "status": "completed"},
                                      {"task_id": ids[1], "status": "failed"}])
    write(train / "tasks-evaluator-only.json", [{"id": x, "partition": "fit"} for x in ids])
    events = [{"event": kind, "request_id": str(i)} for i in range(2)
              for kind in ("request_started", "request_completed")]
    (train / "requests.jsonl").write_text("".join(json.dumps(x)+"\n" for x in events))
    if not terminal: return
    baseline = root / "study-runs/untuned-dev20-decisions"
    write(baseline / "manifest.json", {"status": "completed"})
    write(baseline / "episodes.json", [])
    released = root / "models/decider/head.safetensors"
    released.parent.mkdir(parents=True)
    released.write_bytes(b"SYNTHETIC RELEASED HEAD; NOT MODEL WEIGHTS")
    receipt = baseline / "untuned-pretraining-receipt.json"
    write(receipt, {"phase": "untuned_pretraining", "attempted": 20, "completed": 20,
        "failed": 0, "valid_decision_forwards": 20, "pending_requests": 0,
        "uncertain_requests": 0, "checkpoint_head_sha256": report.digest(released),
        "planned_task_ids": [f"dev{i}" for i in range(20)],
        "manifest_sha256": report.digest(baseline / "manifest.json"),
        "episodes_sha256": report.digest(baseline / "episodes.json")})
    trained = root / "checkpoints/head-trained"
    trained.mkdir(parents=True)
    (trained / "head.safetensors").write_bytes(b"SYNTHETIC CHANGED HEAD; NOT MODEL WEIGHTS")
    write(trained / "training-proof.json", {"status": "trained", "fit_question_ids": ["fit1"],
        "tune_question_ids": ["tune1"], "head_delta_l2": .1, "reload_verified": True,
        "reload_max_logit_error": 0, "initial_head_tensor_sha256": "initial-fixture",
        "trained_head_tensor_sha256": "changed-fixture", "source_head_unchanged": True,
        "source_adapter_unchanged": True, "frozen_torso_adapter": True,
        "calibration_used": False, "fresh100_used": False,
        "original_head_file_sha256": report.digest(released),
        "baseline_receipt_sha256": report.digest(receipt),
        "checkpoint_head_sha256": report.digest(trained / "head.safetensors")})
    calibrated = root / "checkpoints/released-calibrated"
    calibrated.mkdir(parents=True)
    shutil.copyfile(released, calibrated / "head.safetensors")
    adapter = root / "models/decider/lora/adapter_model.safetensors"
    adapter.parent.mkdir(parents=True); adapter.write_bytes(b"SYNTHETIC ADAPTER")
    proof_path = trained / "training-proof.json"; proof = json.loads(proof_path.read_text())
    proof.update(frozen_adapter_sha256=report.digest(adapter), feature_manifest_sha256={})
    for partition in ("fit", "tune"):
        cache_path = root / f"feature-caches/{partition}.jsonl"
        write(cache_path, {"synthetic_fixture": True})
        write(Path(str(cache_path)+".manifest.json"), {"synthetic_fixture": True})
        proof[f"{partition}_cache_sha256"] = report.digest(cache_path)
        proof["feature_manifest_sha256"][partition] = report.digest(Path(str(cache_path)+".manifest.json"))
    write(proof_path, proof)
    for kind, checkpoint in (("untuned", calibrated), ("trained", trained)):
        fit = {"status": "fitted_on_calibration_only_not_test", "question_ids": ["calibration1"],
            "temperature": 1., "checkpoint_head_sha256": report.digest(checkpoint / "head.safetensors")}
        write(root / f"study-runs/calibration-{kind}/temperature-fit.json", fit)
        write(checkpoint / "calibration-proof.json", fit)
        write(checkpoint / "strands_decider_config.json", {"temperature": 1., "temperature_by_kind": {"noul": 1.}})
    # Expand the maintained independent comparison fixture to the explicit N20 panel.
    scratch = root / "fixture-source"; scratch.mkdir()
    old_untuned, old_tuned, old_generation, old_cache = comparison_fixture(scratch)
    originals = json.loads((old_generation / "tasks-evaluator-only.json").read_text())
    rows = [{**copy.deepcopy(originals[i % 3]), "id": f"hotpotqa:fresh{i}",
             "question": f"Unique synthetic question{i}?", "local_partition": "confirmation"} for i in range(20)]
    dataset = root / "candidate-panels/fresh20/tasks.jsonl"; dataset.parent.mkdir(parents=True)
    dataset.write_text("".join(json.dumps(r)+"\n" for r in rows))
    tasks = [asdict(t) for t in load_tasks(dataset)]; ids = [t["id"] for t in tasks]
    original_tr = json.loads((old_generation / "trajectories.json").read_text())
    trajectories = [{**copy.deepcopy(original_tr[i % 3]), "task_id": ids[i]} for i in range(20)]
    generation = root / "study-runs/fresh20-fixed10"
    write(generation / "tasks-evaluator-only.json", tasks)
    write(generation / "trajectories.json", trajectories)
    gen = json.loads((old_generation / "manifest.json").read_text())
    gen["started_at_utc"] = "2026-10-18T19:01:00-05:00"
    write(generation / "manifest.json", gen)
    cache = root / "feature-caches/fresh20-allrounds.jsonl"; cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(old_cache, cache)
    cache_manifest = json.loads(Path(str(old_cache)+".manifest.json").read_text())
    cache_manifest["frozen"].update(trajectories_sha256=report.digest(generation / "trajectories.json"),
        tasks_sha256=report.digest(generation / "tasks-evaluator-only.json"))
    write(Path(str(cache)+".manifest.json"), cache_manifest)
    runs = []
    for old, name, checkpoint in ((old_untuned, "strands-untuned", calibrated),
                                  (old_tuned, "strands-trained", trained)):
        run = root / ("study-runs/fresh20-untuned" if name == "strands-untuned" else "study-runs/fresh20-trained")
        run.mkdir(parents=True); shutil.copytree(old / "source_snapshot", run / "source_snapshot")
        old_episodes = json.loads((old / "episodes.json").read_text()); episodes = []
        for i, qid in enumerate(ids):
            for episode in old_episodes:
                if episode["task_id"] == f"q{i % 3+1}":
                    row = {**copy.deepcopy(episode), "task_id": qid}
                    if row["policy"] == "strands-tuned": row["policy"] = name
                    episodes.append(row)
        events = [{"request_id": qid, "event": event} for qid in ids
                  for event in ("request_started", "request_completed")]
        (run / "requests.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
        write(run / "episodes.json", episodes); write(run / "tasks-evaluator-only.json", tasks)
        manifest = json.loads((old / "manifest.json").read_text()); frozen = manifest["frozen"]
        frozen.update(task_ids=ids, dataset_state_sha256=fingerprint(tasks))
        frozen["config"].update(dataset_sha256=report.digest(dataset), policy_name=name,
            trajectory_sha256=report.digest(generation / "trajectories.json"),
            feature_cache_sha256=report.digest(cache),
            feature_manifest_sha256=report.digest(Path(str(cache)+".manifest.json")),
            model_fingerprint=fingerprint({str(p.relative_to(checkpoint)): report.digest(p)
                for p in sorted(checkpoint.rglob("*")) if p.is_file()}))
        manifest["actual_evaluation_usage"]["completed_requests_including_invalid_outputs"] = 20
        write(run / "manifest.json", manifest); runs.append(run)
    write(root / "study-runs/fresh20-paired-analysis/paired-analysis.json", compare_runs(*runs, generation, cache))
    write(root / "campaign/plan.json", {"fresh_options": [{"n": 20, "dataset": str(dataset),
        "dataset_sha256": report.digest(dataset), "steps": []}]})
    freeze = {"n": 20, "answer_blind": True, "basis": "measured_throughput_and_resources",
        "fresh_scoring_started": False, "frozen_at": "2026-10-18T19:00:00-05:00",
        "dataset_sha256": report.digest(dataset), "plan_sha256": report.digest(root / "campaign/plan.json")}
    write(root / "fresh-N-freeze.json", freeze)
    shutil.copyfile(root / "fresh-N-freeze.json", root / "campaign/fresh-freeze.json")
    write(root / "campaign/campaign.json", {"status": "completed", "fresh_n": 20,
        "plan_sha256": freeze["plan_sha256"], "fresh_freeze_sha256": report.digest(root / "fresh-N-freeze.json")})


def collect(root, n=20):
    return report.collect_report(root, "study-runs/train", n, AS_OF, campaign_run="campaign")


class ReportTests(unittest.TestCase):
    def test_partial_preserves_denominator_and_omits_quality_and_private_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root); result = collect(root, 0)
            self.assertEqual(result["status"], "DRAFT / PARTIAL CHECKPOINT")
            self.assertEqual(result["sections"][0]["rows"][0][1:3], ["3", "2"])
            self.assertNotIn("EM %", report.plain_text(result)); self.assertNotIn(str(root), report.plain_text(result))

    def test_positive_n_requires_explicit_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            with self.assertRaisesRegex(ValueError, "explicit frozen campaign"):
                report.collect_report(root, "study-runs/train", 20, AS_OF)

    def test_terminal_recomputes_canonical_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True); result = collect(root)
            self.assertTrue(result["validation"]["complete_checkpoint"], result["warnings"])
            self.assertIn("EM %", report.plain_text(result))

    def test_missing_baseline_or_training_proof_cannot_render_quality(self):
        for name in ("study-runs/untuned-dev20-decisions/untuned-pretraining-receipt.json", "checkpoints/head-trained/training-proof.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture(root, True); (root / name).unlink()
                self.assertNotIn("EM %", report.plain_text(collect(root)))

    def test_equal_policy_f1_cannot_have_positive_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True)
            path = root / "study-runs/fresh20-paired-analysis/paired-analysis.json"; raw = json.loads(path.read_text())
            for name in ("strands-untuned", "strands-trained"): raw["policies"][name]["token_f1"] = .7
            raw["paired_comparisons"]["strands-trained_minus_strands-untuned"]["token_f1"]["candidate_minus_baseline"] = .05
            with self.assertRaisesRegex(ValueError, "differs from policy means"): report.validate_comparison(raw, 20)
            write(path, raw); self.assertNotIn("EM %", report.plain_text(collect(root)))

    def test_changed_metrics_source_freeze_or_checkpoint_omit_quality(self):
        for change in ("mean", "source", "time", "freeze", "checkpoint", "training-cache", "calibration"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture(root, True)
                if change == "mean":
                    path = root / "study-runs/fresh20-paired-analysis/paired-analysis.json"; raw = json.loads(path.read_text())
                    raw["policies"]["strands-trained"]["mean_actual_drafts"] += .1; write(path, raw)
                elif change == "source": (root / "study-runs/fresh20-untuned/source_snapshot/evaluate.py").write_text("changed")
                elif change == "time":
                    path = root / "study-runs/fresh20-fixed10/manifest.json"; raw = json.loads(path.read_text())
                    raw["started_at_utc"] = "2026-10-18T18:59:00-05:00"; write(path, raw)
                elif change == "freeze": write(root / "campaign/fresh-freeze.json", {"wrong": True})
                elif change == "training-cache": (root / "feature-caches/fit.jsonl").write_text("changed")
                elif change == "calibration": write(root / "checkpoints/head-trained/calibration-proof.json", {"wrong": True})
                else: (root / "checkpoints/released-calibrated/head.safetensors").write_bytes(b"changed")
                self.assertNotIn("EM %", report.plain_text(collect(root)))

    def test_negative_integer_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True)
            raw = json.loads((root / "study-runs/fresh20-paired-analysis/paired-analysis.json").read_text())
            raw["policies"]["strands-trained"].update(completed=-1, failed_or_missing=21)
            with self.assertRaisesRegex(ValueError, "integer policy coverage"): report.validate_comparison(raw, 20)

    def test_duplicate_train_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            write(root / "study-runs/train/trajectories.json", [{"task_id": "train0"}]*2)
            with self.assertRaisesRegex(ValueError, "duplicate or foreign"): collect(root, 0)

    def test_control_copy_is_not_trained(self):
        self.assertFalse(report.training_valid({"status": "untuned_control_copy"}))

    def test_changed_input_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write(root / "input.json", {"x": 1}); evidence = report.Evidence(root)
            evidence.read("input.json", "test"); write(root / "input.json", {"x": 2})
            with self.assertRaisesRegex(ValueError, "inputs changed"): evidence.stable()

    def test_existing_report_preserved_before_renderer_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); old = root / "professor-review.pdf"; old.write_bytes(b"KEEP")
            with self.assertRaises(FileExistsError): report.write_report({}, root)
            self.assertEqual(old.read_bytes(), b"KEEP")

    def test_journal_and_time_validation(self):
        self.assertEqual(report.journal_counts([{"event": "request_started", "request_id": "x"}])["pending"], 1)
        with self.assertRaises(ValueError): report.journal_counts([{"event": "request_completed", "request_id": "x"}])
        with self.assertRaises(ValueError): report.timestamp("2026-10-18T20:00:00")


if __name__ == "__main__": unittest.main()
