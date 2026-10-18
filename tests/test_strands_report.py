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


def fixture(root, terminal=False, alternate=False, reserialized_control=False):
    def where(relative):
        if alternate:
            replacements = {
                "checkpoints/head-trained": "checkpoints/train120-original-v4/head-trained",
                "checkpoints/released-calibrated": "checkpoints/train120-original-v4/released-calibrated",
                "study-runs/untuned-dev20-decisions": "gates/original20-cap1",
                "study-runs/calibration-": "study-runs/train120-original-v4-calibration-",
                "study-runs/fresh20-": "study-runs/fresh20-v4-",
                "feature-caches/fit.jsonl": "feature-caches/train120-original-v4/fit.jsonl",
                "feature-caches/tune.jsonl": "feature-caches/train120-original-v4/tune.jsonl",
                "feature-caches/calibration.jsonl": "feature-caches/train120-original-v4/calibration.jsonl",
                "feature-caches/fresh20-allrounds.jsonl": "feature-caches/fresh20-v4-allrounds.jsonl",
                "candidate-panels/fresh20": "prepared/confirmation20",
            }
            for original, replacement in replacements.items():
                if relative.startswith(original):
                    relative = replacement + relative[len(original):]
                    break
        return root / relative
    train = root / "study-runs/train"
    ids = ["train0", "train1", "train2"]
    write(train / "manifest.json", {"status": "interrupted", "python": "fixture",
        "frozen": {"task_ids": ids, "config": {}}, "writer_unload_verified": True,
        "actual_evaluation_usage": {"completed_requests_including_invalid_outputs": 2,
            "reported_physical_input_tokens": 100, "hosted_cost_usd": 0}})
    write(train / "trajectories.json", [{"task_id": ids[0], "status": "completed", "rounds": [{"round": 1}]},
                                      {"task_id": ids[1], "status": "failed", "rounds": [{"round": 1}]}])
    write(train / "tasks-evaluator-only.json", [{"id": x, "partition": "fit"} for x in ids])
    events = [{"event": kind, "request_id": f"{ids[i]}|fixed|writer|1"} for i in range(2)
              for kind in ("request_started", "request_completed")]
    (train / "requests.jsonl").write_text("".join(json.dumps(x)+"\n" for x in events))
    if not terminal: return
    baseline = where("study-runs/untuned-dev20-decisions")
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
    trained = where("checkpoints/head-trained")
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
    calibrated = where("checkpoints/released-calibrated")
    calibrated.mkdir(parents=True)
    shutil.copyfile(released, calibrated / "head.safetensors")
    if reserialized_control:
        (calibrated / "head.safetensors").write_bytes(b"SYNTHETIC SAME TENSORS DIFFERENT SERIALIZATION AT COPY")
    adapter = root / "models/decider/lora/adapter_model.safetensors"
    adapter.parent.mkdir(parents=True); adapter.write_bytes(b"SYNTHETIC ADAPTER")
    for checkpoint in (trained, calibrated):
        (checkpoint / "lora").mkdir()
        shutil.copyfile(adapter, checkpoint / "lora/adapter_model.safetensors")
    write(calibrated / "training-proof.json", {"status": "untuned_control_copy", "weights_updated": False,
        "head_tensor_sha256": "initial-fixture"})
    proof_path = trained / "training-proof.json"; proof = json.loads(proof_path.read_text())
    proof.update(frozen_adapter_sha256=report.digest(adapter), feature_manifest_sha256={})
    for partition in ("fit", "tune"):
        cache_path = where(f"feature-caches/{partition}.jsonl")
        write(cache_path, {"synthetic_fixture": True})
        write(Path(str(cache_path)+".manifest.json"), {"synthetic_fixture": True})
        proof[f"{partition}_cache_sha256"] = report.digest(cache_path)
        proof["feature_manifest_sha256"][partition] = report.digest(Path(str(cache_path)+".manifest.json"))
    write(proof_path, proof)
    for kind, checkpoint in (("untuned", calibrated), ("trained", trained)):
        fit = {"status": "fitted_on_calibration_only_not_test", "question_ids": ["calibration1"],
            "temperature": 1., "checkpoint_head_sha256": report.digest(checkpoint / "head.safetensors"),
            "bounds": [.25, 6.], "objective": "calibration-question NLL",
            "before": {"threshold": .8}, "after": {"threshold": .8}}
        write(where(f"study-runs/calibration-{kind}/temperature-fit.json"), fit)
        write(checkpoint / "calibration-proof.json", fit)
        write(checkpoint / "strands_decider_config.json", {"temperature": 1., "temperature_by_kind": {"noul": 1.}})
    # Expand the maintained independent comparison fixture to the explicit N20 panel.
    scratch = root / "fixture-source"; scratch.mkdir()
    old_untuned, old_tuned, old_generation, old_cache = comparison_fixture(scratch)
    originals = json.loads((old_generation / "tasks-evaluator-only.json").read_text())
    rows = [{**copy.deepcopy(originals[i % 3]), "id": f"hotpotqa:fresh{i}",
             "question": f"Unique synthetic question{i}?", "local_partition": "confirmation"} for i in range(20)]
    dataset = where("candidate-panels/fresh20/tasks.jsonl"); dataset.parent.mkdir(parents=True)
    dataset.write_text("".join(json.dumps(r)+"\n" for r in rows))
    tasks = [asdict(t) for t in load_tasks(dataset)]; ids = [t["id"] for t in tasks]
    original_tr = json.loads((old_generation / "trajectories.json").read_text())
    trajectories = [{**copy.deepcopy(original_tr[i % 3]), "task_id": ids[i]} for i in range(20)]
    generation = where("study-runs/fresh20-fixed10")
    write(generation / "tasks-evaluator-only.json", tasks)
    write(generation / "trajectories.json", trajectories)
    gen = json.loads((old_generation / "manifest.json").read_text())
    gen["started_at_utc"] = "2026-10-18T19:01:00-05:00"
    write(generation / "manifest.json", gen)
    cache = where("feature-caches/fresh20-allrounds.jsonl"); cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(old_cache, cache)
    cache_manifest = json.loads(Path(str(old_cache)+".manifest.json").read_text())
    cache_manifest["frozen"].update(trajectories_sha256=report.digest(generation / "trajectories.json"),
        tasks_sha256=report.digest(generation / "tasks-evaluator-only.json"))
    write(Path(str(cache)+".manifest.json"), cache_manifest)
    runs = []
    for old, name, checkpoint in ((old_untuned, "strands-untuned", calibrated),
                                  (old_tuned, "strands-trained", trained)):
        run = where("study-runs/fresh20-untuned" if name == "strands-untuned" else "study-runs/fresh20-trained")
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
    analysis = where("study-runs/fresh20-paired-analysis")
    write(analysis / "paired-analysis.json", compare_runs(*runs, generation, cache))
    def command(module, *args):
        return ["python", "-m", module, *map(str, args)]
    core = [{"argv": command("pc_strands.train_head", "--output", trained, "--checkpoint", released.parent,
        "--baseline-receipt", receipt, "--fit-cache", where("feature-caches/fit.jsonl"),
        "--tune-cache", where("feature-caches/tune.jsonl"))}]
    for partition in ("fit", "tune", "calibration"):
        core.append({"argv": command("pc_strands.decider", "cache", "--partition", partition,
            "--output", where(f"feature-caches/{partition}.jsonl"), "--trajectory-dir", train)})
    for kind, checkpoint in (("untuned", calibrated), ("trained", trained)):
        core.append({"argv": command("pc_strands.calibrate", "--checkpoint", checkpoint,
            "--output", where(f"study-runs/calibration-{kind}/temperature-fit.json"))})
    core.append({"id": "copy_released_for_calibration", "argv": ["python", "-c", "synthetic copy fixture; never executed"],
        "outputs": [{"path": str(calibrated / name)} for name in ("head.safetensors", "training-proof.json")]})
    fresh_steps = [{"argv": command("pc_strands.evaluate", "generate", "--dataset", dataset, "--output", generation)},
        {"argv": command("pc_strands.decider", "cache", "--output", cache, "--trajectory-dir", generation)},
        *[{"argv": command("pc_strands.evaluate", "head-replay", "--dataset", dataset, "--output", run,
            "--checkpoint-dir", checkpoint, "--feature-cache", cache, "--trajectory-dir", generation,
            "--policy-name", name)} for run, checkpoint, name in zip(runs, (calibrated, trained), ("strands-untuned", "strands-trained"))],
        {"argv": command("pc_strands.compare", "--untuned", runs[0], "--tuned", runs[1],
            "--trajectory-dir", generation, "--feature-cache", cache, "--output", analysis)}]
    write(root / "campaign/plan.json", {"steps": core, "operator_fresh_freeze_path": str(root / "fresh-N-freeze.json"),
        "fresh_options": [{"n": 20, "dataset": str(dataset),
        "dataset_sha256": report.digest(dataset), "steps": fresh_steps}]})
    freeze = {"n": 20, "answer_blind": True, "basis": "measured_throughput_and_resources",
        "fresh_scoring_started": False, "frozen_at": "2026-10-18T19:00:00-05:00",
        "dataset_sha256": report.digest(dataset), "plan_sha256": report.digest(root / "campaign/plan.json")}
    write(root / "fresh-N-freeze.json", freeze)
    shutil.copyfile(root / "fresh-N-freeze.json", root / "campaign/fresh-freeze.json")
    write(root / "campaign/campaign.json", {"status": "completed", "fresh_n": 20,
        "plan_sha256": freeze["plan_sha256"], "fresh_freeze_sha256": report.digest(root / "fresh-N-freeze.json"),
        "stages": {"copy_released_for_calibration": {"status": "completed", "output_sha256":
            {str(calibrated / name): report.digest(calibrated / name) for name in ("head.safetensors", "training-proof.json")}}}})


def collect(root, n=20):
    return report.collect_report(root, "study-runs/train", n, AS_OF,
        campaign_run="campaign" if (root / "campaign/plan.json").exists() else None)


def rebound_comparison(root):
    """Rebind synthetic fingerprints so identity-link tests reach the control gate."""
    evidence = report.Evidence(root)
    plan = json.loads((root / "campaign/plan.json").read_text())
    paths = report.artifact_paths(evidence, plan, 20, "campaign")
    for kind in ("untuned", "trained"):
        path = paths["fresh_" + kind] / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["frozen"]["config"]["model_fingerprint"] = report.checkpoint_fingerprint(evidence, paths[kind])
        write(path, manifest)
    write(paths["fresh_analysis"] / "paired-analysis.json", compare_runs(paths["fresh_untuned"],
        paths["fresh_trained"], paths["fresh_source"], paths["fresh_cache"]))


def segment_fixture(root):
    fixture(root)
    train, archive = root / "study-runs/train", root / "archive/segment1"
    current = json.loads((train / "manifest.json").read_text())
    current.update(finished_at_utc="2026-10-18T17:20:00-05:00",
        this_invocation_wall_seconds=20., this_invocation_intentional_idle_seconds=8.)
    write(train / "manifest.json", current)
    previous = copy.deepcopy(current)
    previous.update(finished_at_utc="2026-10-18T17:00:00-05:00",
        this_invocation_wall_seconds=10., this_invocation_intentional_idle_seconds=5.)
    previous["actual_evaluation_usage"]["completed_requests_including_invalid_outputs"] = 1
    write(archive / "manifest.json", previous)
    first_call = (train / "requests.jsonl").read_text().splitlines(keepends=True)[:2]
    (archive / "requests.jsonl").write_text("".join(first_call))
    def seal():
        write(archive / "snapshot-receipt.json", {"source": str(train), "files_sha256":
            {name: report.digest(archive / name) for name in ("manifest.json", "requests.jsonl")}})
    seal()
    return train, archive, seal


def historical_expense_fixture(root):
    """Distinct synthetic writer executions, deliberately sharing question IDs."""
    current, historical = root / "study-runs/current120", root / "study-runs/historical600"
    for directory, planned, saved, physical in ((current, 120, 120, 120), (historical, 600, 590, 591)):
        ids = [f"train{i}" for i in range(planned)]
        write(directory / "tasks-evaluator-only.json", [{"id": qid, "partition": "fit"}
            for qid in ids])
        write(directory / "trajectories.json", [{"task_id": qid, "status": "completed",
            "rounds": [{"round": 1}]} for qid in ids[:saved]])
        events = [{"event": event, "request_id": f"{qid}|fixed|writer|1",
            "at_utc": "2026-10-18T23:00:00+00:00" if planned == 120 else "2026-10-18T21:00:00+00:00"}
            for qid in ids[:physical] for event in ("request_started", "request_completed")]
        (directory / "requests.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
        write(directory / "manifest.json", {"status": "completed" if planned == saved else "interrupted",
            "frozen": {"task_ids": ids, "config": {"dataset_sha256": f"synthetic-{planned}"}},
            "finished_at_utc": "2026-10-18T19:00:00-05:00" if planned == 120 else "2026-10-18T17:20:00-05:00",
            "this_invocation_wall_seconds": 20., "this_invocation_intentional_idle_seconds": 8.,
            "actual_evaluation_usage": {"completed_requests_including_invalid_outputs": physical,
                "unreconciled_requests": 0, "reported_physical_input_tokens": physical*10,
                "reported_output_tokens": physical, "hosted_cost_usd": 0}})
    archive = root / "archive/historical-first328"
    previous = json.loads((historical / "manifest.json").read_text())
    previous.update(finished_at_utc="2026-10-18T17:00:00-05:00",
        this_invocation_wall_seconds=10., this_invocation_intentional_idle_seconds=5.)
    previous["actual_evaluation_usage"]["completed_requests_including_invalid_outputs"] = 328
    write(archive / "manifest.json", previous)
    (archive / "requests.jsonl").write_text("".join((historical / "requests.jsonl").read_text().splitlines(keepends=True)[:656]))
    write(archive / "snapshot-receipt.json", {"source": str(historical), "files_sha256":
        {name: report.digest(archive / name) for name in ("manifest.json", "requests.jsonl")}})
    return current, historical, archive


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

    def test_nested_plan_paths_work_without_canonical_alias_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True, alternate=True); result = collect(root)
            self.assertTrue(result["validation"]["complete_checkpoint"], result["validation_errors"])
            self.assertTrue(result["validation"]["unchanged_control_proof"])
            self.assertFalse((root / "checkpoints/head-trained").exists())
            self.assertFalse((root / "study-runs/fresh20-untuned").exists())
            self.assertEqual(result["artifact_roles"]["trained"], str(root / "checkpoints/train120-original-v4/head-trained"))
            self.assertIn("EM %", report.plain_text(result))
            self.assertNotIn(str(root), report.plain_text(result))

    def test_control_tensor_adapter_and_source_links_are_independently_required(self):
        for change in ("control-tensor", "initial-tensor", "missing-control-proof", "control-adapter", "trained-adapter", "source-head"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture(root, True, alternate=True)
                plan = json.loads((root / "campaign/plan.json").read_text())
                paths = report.artifact_paths(report.Evidence(root), plan, 20, "campaign")
                if change in ("control-tensor", "missing-control-proof"):
                    path = paths["untuned"] / "training-proof.json"
                    if change == "missing-control-proof": path.unlink()
                    else:
                        proof = json.loads(path.read_text()); proof["head_tensor_sha256"] = "wrong-control-tensor"; write(path, proof)
                elif change.endswith("adapter"):
                    kind = "untuned" if change == "control-adapter" else "trained"
                    (paths[kind] / "lora/adapter_model.safetensors").write_bytes(b"SYNTHETIC TAMPERED ADAPTER")
                else:
                    path = paths["trained"] / "training-proof.json"; proof = json.loads(path.read_text())
                    proof["initial_head_tensor_sha256" if change == "initial-tensor" else "original_head_file_sha256"] = "wrong-source"
                    write(path, proof)
                # Current-checkpoint fingerprints now bind the altered files.
                # The independent original/control linkage must still fail.
                rebound_comparison(root)
                result = collect(root)
                self.assertFalse(result["validation"]["unchanged_control_proof"])
                self.assertNotIn("EM %", report.plain_text(result))

    def test_control_reserialized_bytes_can_differ_with_matching_tensor_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True, alternate=True, reserialized_control=True)
            plan = json.loads((root / "campaign/plan.json").read_text())
            paths = report.artifact_paths(report.Evidence(root), plan, 20, "campaign")
            head = paths["untuned"] / "head.safetensors"
            result = collect(root)
            self.assertTrue(result["validation"]["complete_checkpoint"], result["validation_errors"])
            self.assertNotEqual(report.digest(head), report.digest(paths["released"] / "head.safetensors"))

    def test_post_copy_head_change_cannot_be_hidden_by_rebinding_calibration_and_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True, alternate=True)
            plan = json.loads((root / "campaign/plan.json").read_text())
            paths = report.artifact_paths(report.Evidence(root), plan, 20, "campaign")
            head = paths["untuned"] / "head.safetensors"
            head.write_bytes(b"SYNTHETIC POST-COPY WEIGHT MUTATION")
            fit_path = paths["calibration_untuned"]; fit = json.loads(fit_path.read_text())
            fit["checkpoint_head_sha256"] = report.digest(head)
            write(fit_path, fit); write(paths["untuned"] / "calibration-proof.json", fit)
            rebound_comparison(root)
            result = collect(root)
            self.assertFalse(result["validation"]["unchanged_control_proof"])
            self.assertNotIn("EM %", report.plain_text(result))

    def test_explicit_missing_plan_cannot_read_valid_canonical_decoys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True); (root / "campaign/plan.json").unlink()
            with self.assertRaisesRegex(ValueError, "explicit campaign plan is missing"):
                report.collect_report(root, "study-runs/train", 0, AS_OF, campaign_run="campaign")

    def test_same_calibration_questions_and_predeclared_protocol_are_required(self):
        for change in ("ids", "bounds", "threshold"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture(root, True)
                path = root / "study-runs/calibration-trained/temperature-fit.json"; fit = json.loads(path.read_text())
                if change == "ids": fit["question_ids"] = ["different-calibration-question"]
                elif change == "bounds": fit["bounds"] = [.1, 10.]
                else: fit["after"]["threshold"] = .7
                write(path, fit); write(root / "checkpoints/head-trained/calibration-proof.json", fit)
                rebound_comparison(root)
                self.assertNotIn("EM %", report.plain_text(collect(root)))

    def test_explicit_freeze_path_is_used_and_bound_without_canonical_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True, alternate=True)
            path = root / "campaign/plan.json"; plan = json.loads(path.read_text())
            selected = root / "explicit-freezes/selected.json"; selected.parent.mkdir()
            plan["operator_fresh_freeze_path"] = str(selected); write(path, plan)
            freeze = json.loads((root / "fresh-N-freeze.json").read_text())
            freeze["plan_sha256"] = report.digest(path)
            write(selected, freeze); write(root / "campaign/fresh-freeze.json", freeze)
            write(root / "fresh-N-freeze.json", {"invalid_canonical_decoy": True})
            campaign = json.loads((root / "campaign/campaign.json").read_text())
            campaign.update(plan_sha256=freeze["plan_sha256"], fresh_freeze_sha256=report.digest(selected))
            write(root / "campaign/campaign.json", campaign)
            self.assertTrue(collect(root)["validation"]["complete_checkpoint"])

    def test_missing_or_conflicting_plan_roles_cannot_use_old_canonical_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True)
            path = root / "campaign/plan.json"; plan = json.loads(path.read_text())
            plan["fresh_options"][0]["steps"].pop()  # Old canonical comparison remains present.
            write(path, plan)
            result = collect(root)
            self.assertNotIn("EM %", report.plain_text(result))
            self.assertIn("canonical comparison stage", result["validation_errors"][0]["detail"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True)
            plan = json.loads((root / "campaign/plan.json").read_text())
            replay = plan["fresh_options"][0]["steps"][2]["argv"]
            replay[replay.index("--feature-cache") + 1] = str(root / "foreign-features.jsonl")
            with self.assertRaisesRegex(ValueError, "fresh replay paths differ"):
                report.artifact_paths(report.Evidence(root), plan, 20, "campaign")

    def test_explicit_training_source_must_match_reviewed_cache_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True)
            with self.assertRaisesRegex(ValueError, "TRAIN source differs"):
                report.collect_report(root, "study-runs/unrelated", 20, AS_OF, campaign_run="campaign")

    def test_baseline_only_plan_reports_partial_historical_train_without_optional_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, True)
            # Same two role-bearing CLI stages as the adopted real V3, with no
            # TRAIN, calibration, checkpoint-copy or fresh branches. Canonical
            # old trained artifacts deliberately remain as unrelated decoys.
            write(root / "campaign/plan.json", {"steps": [
                {"id": "dev20_writer1", "argv": ["python", "-m", "pc_strands.evaluate", "generate",
                    "--output", str(root / "study-runs/dev20-fixed1-v3"), "--max-drafts", "1"]},
                {"id": "untuned_before_training", "argv": ["python", "-m", "pc_strands.evaluate", "replay",
                    "--phase", "untuned_pretraining", "--max-drafts", "1",
                    "--output", str(root / "study-runs/untuned-dev20-decisions"),
                    "--checkpoint-dir", str(root / "models/decider")]}]})
            result = collect(root, 0)
            self.assertEqual(result["status"], "DRAFT / PARTIAL CHECKPOINT")
            self.assertTrue(result["validation"]["untouched_gate"])
            self.assertFalse(result["validation"]["trained_head_proof"])
            self.assertIsNone(result["artifact_roles"]["trained"])
            self.assertIsNone(result["artifact_roles"]["cache_fit"])
            self.assertNotIn("EM %", report.plain_text(result))
            self.assertEqual(result["train_reconciliation"]["saved_trajectories"], 2)
            self.assertNotIn(str(root / "checkpoints/head-trained/training-proof.json"),
                {r["path"] for r in result["input_files"]})

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

    def test_completed_call_without_trajectory_remains_charged_and_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root); train = root / "study-runs/train"
            ids = [f"train{i}" for i in range(600)]
            write(train / "tasks-evaluator-only.json", [{"id": qid, "partition":
                "fit" if i < 400 else "tune" if i < 500 else "calibration"} for i, qid in enumerate(ids)])
            write(train / "trajectories.json", [{"task_id": qid, "status": "completed",
                "rounds": [{"round": 1}]} for qid in ids[:590]])
            events = [{"event": event, "request_id": f"{qid}|fixed|writer|1"}
                for qid in ids[:591] for event in ("request_started", "request_completed")]
            (train / "requests.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
            write(train / "manifest.json", {"status": "interrupted", "frozen": {"task_ids": ids},
                "actual_evaluation_usage": {"completed_requests_including_invalid_outputs": 591,
                    "reported_physical_input_tokens": 1052407, "reported_output_tokens": 5827}})
            result = collect(root, 0)
            self.assertEqual(result["train_reconciliation"], {"planned_questions": 600,
                "saved_trajectories": 590, "missing_trajectories": 10, "undispatched_questions": 9,
                "completed_calls_without_saved_round": 1, "completed_questions_without_saved_trajectory": 1,
                "saved_trajectory_failures": 0})
            usage = next(s for s in result["sections"] if s["title"].startswith("6."))["rows"]
            self.assertEqual(usage[0][1:4], ["591", "1,052,407", "5,827"])
            self.assertIn("not invented scored drafts", report.plain_text(result))
            self.assertNotIn("EM %", report.plain_text(result))

    def test_archived_and_current_timing_sum_without_double_counting_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); train, archive, _ = segment_fixture(root)
            result = report.collect_report(root, train, 0, AS_OF, train_segments=[archive])
            timings = result["train_invocations"]
            self.assertEqual([t["additional_physical_calls"] for t in timings], [None, 1])
            self.assertEqual([t["cumulative_physical_calls"] for t in timings], [1, 2])
            self.assertEqual(sum(t["wall_seconds"] for t in timings), 30.)
            self.assertEqual(sum(t["intentional_idle_seconds"] for t in timings), 13.)
            usage = next(s for s in result["sections"] if s["title"].startswith("6."))["rows"]
            self.assertEqual(usage[0][1], "2")  # Current journal is cumulative, not an increment.
            self.assertEqual(len({r["id"] for r in result["input_files"]}), len(result["input_files"]))

    def test_changed_unrelated_or_overlapping_archives_rejected(self):
        for change in ("hash", "prefix", "source", "overlap", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); train, archive, seal = segment_fixture(root)
                archives = [archive]
                if change in ("hash", "prefix"):
                    with (archive / "requests.jsonl").open("a") as stream: stream.write("\n")
                    if change == "prefix": seal()
                elif change == "source":
                    raw = json.loads((archive / "manifest.json").read_text())
                    raw["frozen"]["config"]["seed"] = 99; write(archive / "manifest.json", raw); seal()
                elif change == "overlap":
                    raw = json.loads((train / "manifest.json").read_text())
                    raw["finished_at_utc"] = "2026-10-18T17:00:15-05:00"; write(train / "manifest.json", raw)
                else: archives.append(archive)
                with self.assertRaises(ValueError):
                    report.collect_report(root, train, 0, AS_OF, train_segments=archives)

    def test_latest_timing_only_does_not_invent_additional_call_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); train, _, _ = segment_fixture(root)
            result = report.collect_report(root, train, 0, AS_OF)
            self.assertIsNone(result["train_invocations"][0]["additional_physical_calls"])
            self.assertNotIn("Sum of selected invocations", report.plain_text(result))

    def test_recorded_evidence_mutation_before_parse_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write(root / "input.json", {"x": 1}); evidence = report.Evidence(root)
            evidence.record_file(root / "input.json", "before parse")
            write(root / "input.json", {"x": 2})
            with self.assertRaisesRegex(ValueError, "changed before parsing"): evidence.read("input.json", "parse")

    def test_historical591_and_current120_are_separate_physical_expenses_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); current, historical, archive = historical_expense_fixture(root)
            result = report.collect_report(root, current, 0, AS_OF,
                historical_train_run=historical, historical_train_segments=[archive])
            self.assertEqual(result["train_reconciliation"]["planned_questions"], 120)
            self.assertEqual(result["train_reconciliation"]["saved_trajectories"], 120)
            old = result["historical_train_expense"]
            self.assertEqual(old["reconciliation"]["planned_questions"], 600)
            self.assertEqual(old["reconciliation"]["saved_trajectories"], 590)
            self.assertEqual(old["reconciliation"]["completed_calls_without_saved_round"], 1)
            self.assertEqual(old["actual_evaluation_usage"]["completed_requests_including_invalid_outputs"], 591)
            usage = next(s for s in result["sections"] if s["title"].startswith("6."))["rows"]
            self.assertEqual([(r[0], r[1]) for r in usage], [("TRAIN writer", "120"), ("Historical TRAIN preparation", "591")])
            self.assertEqual([v["additional_physical_calls"] for v in old["invocations"]], [None, 263])
            self.assertEqual(sum(v["wall_seconds"] for v in old["invocations"]), 30.)
            self.assertTrue(any(s["title"] == "Historical TRAIN invocation timing" for s in result["sections"]))
            self.assertNotIn("EM %", report.plain_text(result))
            self.assertNotIn(str(root), report.plain_text(result))

    def test_historical_archive_cannot_be_cross_bound_to_current_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); current, historical, archive = historical_expense_fixture(root)
            with self.assertRaisesRegex(ValueError, "receipt does not bind this source"):
                report.collect_report(root, current, 0, AS_OF, train_segments=[archive], historical_train_run=historical)

    def test_historical_source_cannot_duplicate_current_or_omit_its_source_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); current, _, archive = historical_expense_fixture(root)
            with self.assertRaisesRegex(ValueError, "duplicates the current physical source"):
                report.collect_report(root, current, 0, AS_OF, historical_train_run=current)
            with self.assertRaisesRegex(ValueError, "segments require an explicit historical source"):
                report.collect_report(root, current, 0, AS_OF, historical_train_segments=[archive])

    def test_copied_or_partly_overlapping_physical_sources_cannot_be_charged_twice(self):
        for overlap in ("whole", "one-dispatch"):
            with self.subTest(overlap=overlap), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); current, historical, _ = historical_expense_fixture(root)
                if overlap == "whole":
                    historical = root / "copied-current"
                    shutil.copytree(current, historical)
                else:
                    path = historical / "requests.jsonl"
                    rows = [json.loads(v) for v in path.read_text().splitlines()]
                    rows[0]["at_utc"] = "2026-10-18T18:00:00-05:00"  # Same physical UTC instant.
                    path.write_text("".join(json.dumps(v)+"\n" for v in rows))
                with self.assertRaisesRegex(ValueError, "repeats current physical dispatches"):
                    report.collect_report(root, current, 0, AS_OF, historical_train_run=historical)

    def test_partial_historical_source_cannot_be_relabelled_terminal_or_hide_bad_journal(self):
        for change in ("terminal", "journal"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); current, historical, _ = historical_expense_fixture(root)
                if change == "terminal":
                    path = historical / "manifest.json"; value = json.loads(path.read_text())
                    value["status"] = "completed"; write(path, value)
                else:
                    with (historical / "requests.jsonl").open("a") as stream:
                        stream.write(json.dumps({"event": "request_completed", "request_id": "foreign|fixed|writer|1"})+"\n")
                with self.assertRaises(ValueError):
                    report.collect_report(root, current, 0, AS_OF, historical_train_run=historical)


if __name__ == "__main__": unittest.main()
