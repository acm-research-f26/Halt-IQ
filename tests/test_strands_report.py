"""Synthetic report contracts only; no API, model or benchmark evidence."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import math
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import build_strands_report as report
from pc_strands.compare import compare_runs
from pc_strands.data import fingerprint, load_tasks
from test_compare import fixture as comparison_fixture

AS_OF = "2026-10-18T20:00:00-05:00"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def native_baseline_fixture(root, directory=None, released=None):
    """Synthetic native-shaped records; never an actual model experiment."""
    directory = directory or root / "study-runs/untuned-dev20-decisions"
    released = released or root / "models/decider/head.safetensors"
    if not released.exists():
        released.parent.mkdir(parents=True, exist_ok=True)
        released.write_bytes(b"SYNTHETIC RELEASED HEAD; NOT MODEL WEIGHTS")
    tasks, episodes, events = [], [], []
    for i in range(20):
        qid, answer, correct = f"dev{i}", f"gold{i}" if i < 9 else f"wrong{i}", i < 9
        task = {"id": qid, "question": f"Synthetic baseline question {i}?", "evidence": ["Synthetic evidence"],
            "answers": [f"gold{i}"], "partition": "dev"}
        tasks.append(task)
        probability = .9 if i % 2 == 0 else .4
        raw = {"p_true": probability, "logits": [0., math.log(probability / (1-probability))],
            "temperature": 1., "input_tokens": 100+i, "physical_input_tokens": 100+i,
            "elapsed_seconds": .01, "encoding_seconds": .005, "reused_encoding": False,
            "answer": "true" if probability > .5 else "false", "model_id": "SYNTHETIC"}
        public = {"state": {"question": task["question"], "evidence": task["evidence"], "draft": answer},
            "model": "synthetic-baseline", "temperature": None}
        request = f"{qid}|strands-untuned|decision|1"
        events.extend([{"event": "request_started", "request_id": request, "input": public,
            "input_sha256": fingerprint(public), "at_utc": f"2026-10-18T22:00:{i:02d}+00:00"},
            {"event": "request_completed", "request_id": request, "input_sha256": fingerprint(public),
            "result": raw, "request_elapsed_seconds": .5+i*.001}])
        fixed = {"task_id": qid, "policy": "fixed-1-prefix", "status": "completed", "final_answer": answer,
            "rounds": 1, "exact_match": correct, "token_f1": float(correct), "first_draft_exact_match": correct,
            "decision_calls": 0, "decision_input_tokens": 0}
        episodes.extend([fixed, {**fixed, "policy": "strands-untuned", "decision_calls": 1,
            "decision_input_tokens": raw["input_tokens"], "stop_reason": "approved" if probability >= .8 else "round_cap",
            "decisions": [{"round": 1, "approved": probability >= .8, "draft_exact_match": correct, "score": raw}]}])
    write(directory / "tasks-evaluator-only.json", tasks)
    write(directory / "episodes.json", episodes)
    (directory / "requests.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    write(directory / "manifest.json", {"status": "completed", "frozen": {"task_ids": [t["id"] for t in tasks],
        "dataset_state_sha256": fingerprint(tasks), "source_sha256": "SYNTHETIC SOURCE",
        "config": {"phase": "untuned_pretraining", "mode": "replay", "max_drafts": 1, "threshold": .8,
            "decision_device": "cuda", "model_fingerprint": "synthetic-baseline", "policy_name": "strands-untuned"}},
        "actual_decision_requests": 20, "actual_evaluation_usage": {"completed_requests_including_invalid_outputs": 20,
            "unreconciled_requests": 0, "reported_physical_input_tokens": 2190, "reported_output_tokens": 0,
            "requests_with_unknown_usage": 0, "reported_request_seconds": sum(.5+i*.001 for i in range(20)), "hosted_cost_usd": 0},
        "this_invocation_wall_seconds": 120., "this_invocation_intentional_idle_seconds": 100.,
        "backend_startup_this_invocation": {"wall_seconds": 3., "intentional_idle_seconds": 1.}})
    receipt = directory / "untuned-pretraining-receipt.json"
    write(receipt, {"phase": "untuned_pretraining", "attempted": 20, "completed": 20, "failed": 0,
        "valid_decision_forwards": 20, "pending_requests": 0, "uncertain_requests": 0,
        "checkpoint_head_sha256": report.digest(released), "planned_task_ids": [t["id"] for t in tasks],
        "source_sha256": "SYNTHETIC SOURCE", "model": {"max_length": 4096},
        "manifest_sha256": report.digest(directory / "manifest.json"),
        "episodes_sha256": report.digest(directory / "episodes.json"),
        "requests_sha256": report.digest(directory / "requests.jsonl")})
    return directory


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
    released = root / "models/decider/head.safetensors"
    released.parent.mkdir(parents=True)
    released.write_bytes(b"SYNTHETIC RELEASED HEAD; NOT MODEL WEIGHTS")
    native_baseline_fixture(root, baseline, released)
    receipt = baseline / "untuned-pretraining-receipt.json"
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


def development_receipt_fixture(root):
    """Synthetic consumer boundary: the separately tested auditor is not inference."""
    from pc_strands.compare import first_draft_diagnostics
    fixture(root, True)
    owner = root / "campaign"
    paths = report.artifact_paths(report.Evidence(root), json.loads((owner / "plan.json").read_text()), 0, owner)
    qualification = root / "synthetic-matched-qualification.json"
    write(qualification, {"actual_training_campaign_plan_sha256": report.digest(owner / "plan.json"),
        "actual_training_campaign_sha256": report.digest(owner / "campaign.json")})
    baseline = json.loads((paths["baseline"].parent / "episodes.json").read_text())
    diagnostics = first_draft_diagnostics({r["task_id"]:r for r in baseline if r["policy"] == "strands-untuned"}, [f"dev{i}" for i in range(20)])
    delta = {"candidate_minus_baseline":0., "bootstrap95":[0.,0.], "planned_n":20,
        "complete_pair_n":20, "draws":5000, "seed":42}
    pairs = {"planned_n":20,"paired_scored_n":20,"missing_probability_pairs":0,"draws":5000,"seed":42,
        **{k+"_trained_minus_untuned":copy.deepcopy(delta) for k in ("nll","brier","accuracy","approval_coverage_of_planned")},
        **{k+"_paired_bootstrap":{"valid_draws":5000,"undefined_draws":0,"bootstrap95":[0.,0.]}
            for k in ("auroc","approval_precision")}}
    roles = {"plan":str(owner / "plan.json"),"campaign":str(owner / "campaign.json"),
        "original_baseline_directory":str(paths["baseline"].parent),
        "untuned_checkpoint_directory":str(paths["untuned"]),"trained_checkpoint_directory":str(paths["trained"]),
        "matched_training_control_CAL_qualification":str(qualification)}
    receipt = {"status":"verified_secondary_development20_posttraining_diagnostics","phase":"development",
        "planned_n":20,"max_drafts":1,"task_ids":[f"dev{i}" for i in range(20)],
        "analysis_helper_sha256":report.DEVELOPMENT_AUDITOR_SHA,"audited_at_utc":"2026-10-18T19:30:00-05:00",
        "all20_original_development_questions_verified":True,"no_fresh_confirmation":True,
        "no_further_weight_or_temperature_selection":True,"held_out_improvement_claim":False,"artifact_roles":roles,
        "input_files_sha256":{str(p):report.digest(p) for p in (owner/"plan.json",owner/"campaign.json",qualification)},
        "source_writer_answer_quality":{"planned_n":20,"em":.45,"token_f1":.45,"same_saved_answers_in_both_arms":True},
        "encoding_expense":{"physical_encoder_forwards":20,"physical_encoder_input_tokens":2190,"encoding_seconds":.1},
        "paired_deltas":pairs,"per_policy":{k:{"first_draft_diagnostics":copy.deepcopy(diagnostics),
            "head_score_requests_including_known_failures":20,"request_seconds":.02,"failed_or_missing_episodes":0}
            for k in ("untuned","trained")}}
    path = root / "synthetic-development-diagnostic.json"; write(path, receipt)
    return path, receipt, paths, owner


def split_campaign_fixture(root):
    """Separate synthetic owners; no training rows imported into fresh state."""
    fixture(root, True, alternate=True)
    fresh = root / "campaign"
    original = json.loads((fresh / "plan.json").read_text())
    owner = root / "training-campaign-v8"
    core = copy.deepcopy(original["steps"])
    for i, step in enumerate(core):
        step.setdefault("id", f"training-stage-{i}")
        argv = step["argv"]
        if "pc_strands.train_head" in argv:
            directory = Path(argv[argv.index("--output")+1])
            outputs = [directory / "head.safetensors", directory / "training-proof.json"]
        elif "pc_strands.decider" in argv:
            path = Path(argv[argv.index("--output")+1])
            if not path.exists(): write(path, {"synthetic_calibration_features": True})
            outputs = [path]
        elif "pc_strands.calibrate" in argv:
            directory = Path(argv[argv.index("--checkpoint")+1])
            outputs = [Path(argv[argv.index("--output")+1]), directory / "calibration-proof.json",
                directory / "strands_decider_config.json", directory / "head.safetensors"]
        else:
            outputs = [Path(o["path"]) for o in step["outputs"]]
        step["outputs"] = [{"path": str(path)} for path in outputs]
    source = {}
    for i, name in enumerate(sorted(p.name for p in (Path(report.__file__).parents[1] / "pc_strands").glob("*.py"))):
        for directory in (owner, fresh):
            path = directory / "source/pc_strands" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# Synthetic source identity only {i}; never executed\n")
        source[name] = report.digest(path)
    pins = {"base_revision": "synthetic-base", "decider_revision": "synthetic-decider"}
    training_plan = {"steps": core, "fresh_options": [], "reviewed_source_sha256": source, "model_pins": pins}
    write(owner / "plan.json", training_plan)
    write(owner / "campaign.json", {"status": "completed", "plan_sha256": report.digest(owner / "plan.json"),
        "source_sha256": source, "stages": {step["id"]: {"status": "completed", "exit_code": 0,
            "argv": step["argv"], "output_sha256": {o["path"]: report.digest(o["path"]) for o in step["outputs"]}}
            for step in core}})
    original.update(steps=[], reviewed_source_sha256=source, model_pins=pins,
        training_campaign_reference={"path": str(owner), "final_plan_sha256": report.digest(owner / "plan.json"),
            "terminal_campaign_sha256": report.digest(owner / "campaign.json")})
    for i, step in enumerate(original["fresh_options"][0]["steps"]):
        step["id"] = f"fresh20-stage-{i}"
    write(fresh / "plan.json", original)
    freeze = json.loads((root / "fresh-N-freeze.json").read_text())
    freeze["plan_sha256"] = report.digest(fresh / "plan.json")
    write(root / "fresh-N-freeze.json", freeze); write(fresh / "fresh-freeze.json", freeze)
    state = json.loads((fresh / "campaign.json").read_text())
    state.update(plan_sha256=freeze["plan_sha256"], fresh_freeze_sha256=report.digest(root / "fresh-N-freeze.json"),
        source_sha256=source, stages={})
    write(fresh / "campaign.json", state)
    return owner, fresh


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
    def test_selected_actual90_description_preserves_blind_groups_not_ancestor600(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); owner=root/"owner"
            tasks=[{"id":f"s{i}","partition":"fit" if i<60 else "tune" if i<75 else "calibration"} for i in range(90)]
            selection={"status":"PROPOSED_NOT_ADOPTED","count":90,"ordered_task_ids":[t["id"] for t in tasks],
                "partition_counts":{"fit":60,"tune":15,"calibration":15},"selection":{"uses_only":["canonical_question_id","local_partition","source_row.type"],"no_repartitioning":True}}
            chosen=root/"selection.json";write(chosen,selection)
            plan={"training_source":{"manifest_sha256":report.digest(chosen),"dataset_sha256":"selected-dataset","planned_n":90},
                "prerequisites":{"selected":[{"path":str(chosen),"sha256":report.digest(chosen)}]}}
            write(owner/"plan.json",plan);state={"status":"completed","plan_sha256":report.digest(owner/"plan.json")}
            manifest={"frozen":{"config":{"dataset_sha256":"selected-dataset"}}}
            value=report.selected_train_description(report.Evidence(root),plan,state,owner,manifest,tasks)
            self.assertIn("90 planned questions, FIT 60 / TUNE 15 / CALIBRATION 15",value)
            self.assertIn("answer-blind reduction",value);self.assertIn("historical preparation expense",value)
            self.assertIn("manifest still says PROPOSED",value)
            with self.assertRaisesRegex(ValueError,"owner/source state"):
                report.selected_train_description(report.Evidence(root),plan,None,owner,manifest,tasks)
            write(chosen,["malformed"])
            with self.assertRaisesRegex(ValueError,"sampling manifest"):
                report.selected_train_description(report.Evidence(root),plan,state,owner,manifest,tasks)
            selection["selection"]["uses_only"].append("outcome");write(chosen,selection)
            with self.assertRaises(ValueError):report.selected_train_description(report.Evidence(root),plan,state,owner,manifest,tasks)

    def test_training_work_total_is_distinct_from_selected_checkpoint_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);fixture(root,True);path=root/"checkpoints/head-trained/training-proof.json"
            proof=json.loads(path.read_text());proof.update(history=[{"lr":lr,"epoch":epoch,"optimizer_steps":2} for lr in (1e-5,5e-5) for epoch in (1,2,3)],selected={"lr":1e-5,"epoch":1});write(path,proof)
            result=collect(root,0);rows=next(s for s in result["sections"] if s["title"].startswith("4."))["rows"]
            self.assertIn(["Total candidate-fit epochs / optimizer steps","6 / 12"],rows)
            self.assertIn(["Selected saved checkpoint LR / epoch","1e-05 / 1"],rows)
            self.assertIn("need not accumulate every candidate's updates",report.plain_text(result))

    def test_captured_physical_usage_reconciles_raw_writer_and_zero_encoder_cpu_head(self):
        from types import SimpleNamespace
        # The independently reviewed capture auditor is mocked only at its boundary.
        for head,n in ((False,20),(True,15)):
            with self.subTest(head=head),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);owner=root/"owner";directory=root/"phase";ids=[f"q{i}" for i in range(n)]
                step={"id":"phase","argv":["python","--output",str(directory)]};plan={"steps":[step]};write(owner/"plan.json",plan)
                state={"status":"completed","plan_sha256":report.digest(owner/"plan.json")};events=[]
                for qid in ids:
                    key=f"{qid}|{'critic|decision' if head else 'fixed|writer'}|1"
                    events.extend([{"event":"request_started","request_id":key},{"event":"request_completed","request_id":key,"request_elapsed_seconds":.5,
                        "result":{"input_tokens":100,"output_tokens":0 if head else 3,"physical_input_tokens":0 if head else 100,"reused_encoding":head,"cost_usd":0}}])
                directory.mkdir();journal=directory/"requests.jsonl";journal.write_text(''.join(json.dumps(e)+'\n' for e in events))
                usage={"completed_requests_including_invalid_outputs":n,"unreconciled_requests":0,"reported_physical_input_tokens":0 if head else 100*n,
                    "reported_output_tokens":0 if head else 3*n,"requests_with_unknown_usage":0,"reported_request_seconds":.5*n,"hosted_cost_usd":0}
                m={"status":"completed","frozen":{"task_ids":ids,"config":{"mode":"head-replay" if head else "generate","phase":"development","max_drafts":1,
                    "policy_name":"critic","decision_device":"cpu","head_replay_no_torso_loaded":True}},"actual_evaluation_usage":usage}
                write(directory/"manifest.json",m)
                write(directory/"tasks-evaluator-only.json",[{"id":qid,"partition":"calibration" if head else "dev"} for qid in ids])
                capture={"output_sha256":{str(directory/name):report.digest(directory/name) for name in ("manifest.json","requests.jsonl","tasks-evaluator-only.json")}}
                q=SimpleNamespace(Evidence=lambda:SimpleNamespace(files={}),stage=Mock(return_value=capture),argument=lambda s,flag:s["argv"][s["argv"].index(flag)+1])
                with patch.object(report,"development_auditor",return_value=SimpleNamespace(q=q)):
                    _,actual=report.captured_phase_usage(report.Evidence(root),plan,state,owner,step,ids,head=head)
                    self.assertEqual(actual["completed_requests_including_invalid_outputs"],n)
                    self.assertEqual(actual["reported_physical_input_tokens"],0 if head else 100*n)
                    q.stage.assert_called_once()
                    with self.assertRaisesRegex(ValueError,"owning plan/state"):
                        report.captured_phase_usage(report.Evidence(root),plan,None,owner,step,ids,head=head)
                    for change in ("aggregate","pending","negative","phase","capture","manifest","tasks"):
                        write(directory/"manifest.json",m);journal.write_text(''.join(json.dumps(e)+'\n' for e in events));q.stage.side_effect=None
                        if change=="aggregate":bad=copy.deepcopy(m);bad["actual_evaluation_usage"]["reported_request_seconds"]+=1;write(directory/"manifest.json",bad)
                        elif change=="pending":journal.write_text(''.join(json.dumps(e)+'\n' for e in events[:-1]))
                        elif change=="negative":bad=copy.deepcopy(events);bad[1]["result"]["cost_usd"]=-1;journal.write_text(''.join(json.dumps(e)+'\n' for e in bad))
                        elif change=="phase":bad=copy.deepcopy(m);bad["frozen"]["config"]["phase"]="confirmation";write(directory/"manifest.json",bad)
                        elif change=="manifest":write(directory/"manifest.json",["malformed"])
                        elif change=="tasks":write(directory/"tasks-evaluator-only.json",["malformed"])
                        else:q.stage.side_effect=ValueError("captured output changed")
                        with self.subTest(change=change),self.assertRaises(ValueError):report.captured_phase_usage(report.Evidence(root),plan,state,owner,step,ids,head=head)
                        write(directory/"tasks-evaluator-only.json",[{"id":qid,"partition":"calibration" if head else "dev"} for qid in ids])

    def test_secondary_development_validated_audit_is_separate_and_expense_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path, receipt, _, _ = development_receipt_fixture(root)
            auditor = Mock(); auditor.analyze.return_value = copy.deepcopy(receipt)
            with patch.object(report, "development_auditor", return_value=auditor):
                result = report.collect_report(root,"study-runs/train",0,AS_OF,campaign_run="campaign",development_diagnostic=path)
            self.assertTrue(result["validation"]["secondary_development_diagnostics"])
            self.assertFalse(result["validation"]["paired_analysis"])
            self.assertFalse(result["validation"]["complete_checkpoint"])
            self.assertEqual(result["status"],"DRAFT / PARTIAL CHECKPOINT")
            text = report.plain_text(result)
            self.assertIn("Secondary post-training DEVELOPMENT probability diagnostics", text)
            self.assertIn("same 20 development questions were inspected before training", text)
            self.assertNotIn(str(root),text)
            usage = next(s for s in result["sections"] if s["title"].startswith("6."))["rows"]
            extra = [r for r in usage if r[0].startswith("Secondary DEVELOPMENT")]
            self.assertEqual([(r[0],r[1],r[2]) for r in extra],[("Secondary DEVELOPMENT encoder once","20","2,190"),
                ("Secondary DEVELOPMENT untuned CPU head","20","0"),("Secondary DEVELOPMENT trained CPU head","20","0")])
            auditor.analyze.assert_called_once_with(root/"campaign/plan.json",root/"campaign")

    def test_secondary_missing_receipt_suppresses_only_secondary_with_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); fixture(root,True); before=collect(root,0)
            result=report.collect_report(root,"study-runs/train",0,AS_OF,campaign_run="campaign",development_diagnostic="missing.json")
            self.assertIsNone(result["secondary_development"])
            for key in ("untouched_gate","trained_head_proof","unchanged_control_proof","paired_analysis","complete_checkpoint"):
                self.assertEqual(result["validation"][key],before["validation"][key])
            self.assertEqual(result["validation_errors"][-1]["stage"],"secondary development diagnostics")
            self.assertIn("explicit DEVELOPMENT diagnostic receipt is missing",result["validation_errors"][-1]["detail"])
            self.assertNotIn("Secondary post-training DEVELOPMENT probability diagnostics",report.plain_text(result))

    def test_secondary_phase_denominator_time_and_current_roles_cannot_be_relabelled(self):
        for change in ("phase","n","bool-n","cap","duplicate","time","checkpoint","owner","qualification-list","receipt-list"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); path,receipt,paths,owner=development_receipt_fixture(root)
                if change=="phase": receipt["phase"]="confirmation"
                elif change=="n": receipt["planned_n"]=19
                elif change=="bool-n": receipt["planned_n"]=True
                elif change=="cap": receipt["max_drafts"]=10
                elif change=="duplicate": receipt["task_ids"][-1]=receipt["task_ids"][0]
                elif change=="time": receipt["audited_at_utc"]="2026-10-18T20:01:00-05:00"
                elif change=="checkpoint": receipt["artifact_roles"]["trained_checkpoint_directory"]=str(paths["untuned"])
                elif change in ("owner","qualification-list"):
                    q=Path(receipt["artifact_roles"]["matched_training_control_CAL_qualification"])
                    write(q,["malformed"] if change=="qualification-list" else {"actual_training_campaign_plan_sha256":"wrong-owner","actual_training_campaign_sha256":report.digest(owner/"campaign.json")})
                else: receipt=["malformed"]
                write(path,receipt)
                with patch.object(report,"development_auditor") as audit:
                    with self.assertRaises(ValueError): report.development_diagnostics(report.Evidence(root),path,paths,owner,AS_OF)
                    audit.assert_not_called()

    def test_secondary_changed_bound_file_or_metrics_fail_closed(self):
        for change in ("input","metrics","raw-audit-reject"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); path,receipt,paths,owner=development_receipt_fixture(root)
                actual=copy.deepcopy(receipt)
                if change=="input": (owner/"plan.json").write_text('{}')
                elif change=="metrics": receipt["per_policy"]["trained"]["first_draft_diagnostics"]["auroc"]=1.
                write(path,receipt); auditor=Mock(); auditor.analyze.return_value=actual
                if change=="raw-audit-reject": auditor.analyze.side_effect=ValueError("raw stage/feature/label join refused")
                with patch.object(report,"development_auditor",return_value=auditor):
                    with self.assertRaises(ValueError): report.development_diagnostics(report.Evidence(root),path,paths,owner,AS_OF)

    def test_secondary_nulls_and_missing_pairs_are_displayed_without_imputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            _,receipt,_,_=development_receipt_fixture(Path(temporary))
            receipt["paired_deltas"].update(paired_scored_n=0,missing_probability_pairs=20)
            for arm in receipt["per_policy"].values():
                arm["first_draft_diagnostics"].update(scored_first_drafts=0,auroc=None,brier=None,nll=None,
                    first_draft_approval_precision=None,approval_coverage_of_planned=0)
            for key in ("nll","brier","accuracy"): receipt["paired_deltas"][key+"_trained_minus_untuned"]=None
            for key in ("auroc","approval_precision"):
                receipt["paired_deltas"][key+"_paired_bootstrap"]={"valid_draws":0,"undefined_draws":5000,"bootstrap95":None}
            sections=report.development_sections(receipt); text=report.plain_text({"title":"test","status":"synthetic","as_of":AS_OF,"sections":sections,"input_files":[]})
            self.assertIn("undefined",text); self.assertIn("0/20 observed pairs; 20 absent pairs are not imputed",text)
            self.assertIn("0 / 5000",text)
            self.assertIn("not drawn",text)

    def test_secondary_auditor_requires_both_exact_reviewed_programs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            helper=root/"operator_dev20_diagnostics.py"
            helper.write_text("raise RuntimeError('must never execute')")
            with self.assertRaisesRegex(ValueError,"missing or changed"): report.development_auditor(report.Evidence(root))
            with patch.object(report,"DEVELOPMENT_AUDITOR_SHA",report.digest(helper)):
                with self.assertRaisesRegex(ValueError,"missing or changed"): report.development_auditor(report.Evidence(root))
            sibling=root/"operatorqualification.py"; sibling.write_text("# changed sibling")
            with patch.object(report,"DEVELOPMENT_AUDITOR_SHA",report.digest(helper)):
                with self.assertRaisesRegex(ValueError,"missing or changed"): report.development_auditor(report.Evidence(root))

    def test_secondary_import_is_read_only_and_restores_bytecode_setting(self):
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); helper=root/"operator_dev20_diagnostics.py"; sibling=root/"operatorqualification.py"
            helper.write_text("import sys\nassert sys.dont_write_bytecode\nvalue = 'synthetic stdlib boundary'\n")
            sibling.write_text("# synthetic pinned evidence boundary\n")
            previous=sys.dont_write_bytecode
            with patch.object(report,"DEVELOPMENT_AUDITOR_SHA",report.digest(helper)),patch.object(report,"DEVELOPMENT_EVIDENCE_SHA",report.digest(sibling)):
                module=report.development_auditor(report.Evidence(root))
            self.assertEqual(module.value,"synthetic stdlib boundary")
            self.assertEqual(sys.dont_write_bytecode,previous)
            self.assertFalse((root/"__pycache__").exists())

    def test_separate_training_owner_supplies_real_roles_without_imported_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); owner, fresh = split_campaign_fixture(root)
            result = report.collect_report(root, "study-runs/train", 20, AS_OF,
                campaign_run=fresh, training_campaign_run=owner)
            self.assertTrue(result["validation"]["complete_checkpoint"], result["validation_errors"])
            self.assertTrue(result["validation"]["unchanged_control_proof"])
            self.assertEqual(result["training_campaign_run"], str(owner))
            self.assertEqual(json.loads((fresh / "campaign.json").read_text())["stages"], {})
            self.assertEqual(Path(result["artifact_roles"]["trained"]),
                root / "checkpoints/train120-original-v4/head-trained")
            self.assertNotIn(str(root), report.plain_text(result))
            partial = report.collect_report(root, "study-runs/train", 0, AS_OF,
                campaign_run=fresh, training_campaign_run=owner)
            self.assertEqual(partial["status"], "DRAFT / PARTIAL CHECKPOINT")
            self.assertTrue(partial["validation"]["baseline_diagnostics"])
            self.assertNotIn("EM %", report.plain_text(partial))

    def test_separate_training_plan_stage_source_and_import_tampering_fail_closed(self):
        for change in ("plan", "stage-capture", "stage-argv", "source", "import", "state-import", "reference"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); owner, fresh = split_campaign_fixture(root)
                if change == "plan":
                    path = owner / "plan.json"; value = json.loads(path.read_text()); value["tampered"] = True
                elif change in ("stage-capture", "stage-argv"):
                    path = owner / "campaign.json"; value = json.loads(path.read_text())
                    stage = next(iter(value["stages"].values()))
                    if change == "stage-capture": stage["output_sha256"][next(iter(stage["output_sha256"]))] = "wrong"
                    else: stage["argv"].append("unrecorded-change")
                    write(path, value)
                    fp = fresh / "plan.json"; f = json.loads(fp.read_text())
                    f["training_campaign_reference"]["terminal_campaign_sha256"] = report.digest(path); write(fp, f)
                elif change == "source":
                    path = fresh / "source/pc_strands/decider.py"; path.write_text("# Synthetic changed frozen source\n")
                elif change == "state-import":
                    path = fresh / "campaign.json"; value = json.loads(path.read_text())
                    value["stages"]["copy_released_for_calibration"] = {"status": "completed"}; write(path, value)
                else:
                    path = fresh / "plan.json"; value = json.loads(path.read_text())
                    if change == "import": value["steps"] = json.loads((owner / "plan.json").read_text())["steps"]
                    else: value["training_campaign_reference"]["path"] = str(root / "unrelated-training")
                if change in ("plan", "import", "reference"): write(path, value)
                with self.assertRaises(ValueError):
                    report.collect_report(root, "study-runs/train", 20, AS_OF,
                        campaign_run=fresh, training_campaign_run=owner)

    def test_separate_training_owner_is_explicit_and_cannot_fallback_to_old_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); owner, fresh = split_campaign_fixture(root)
            without_owner = report.collect_report(root, "study-runs/train", 20, AS_OF, campaign_run=fresh)
            self.assertFalse(without_owner["validation"]["complete_checkpoint"])
            self.assertIsNone(without_owner["artifact_roles"]["trained"])
            with self.assertRaisesRegex(ValueError, "explicit fresh campaign"):
                report.collect_report(root, "study-runs/train", 0, AS_OF, training_campaign_run=owner)

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
            directory = root / "study-runs/untuned-dev20-decisions"
            receipt = directory / "untuned-pretraining-receipt.json"
            value = json.loads(receipt.read_text()); value.pop("requests_sha256"); write(receipt, value)
            write(root / "campaign/campaign.json", {"status": "completed",
                "plan_sha256": report.digest(root / "campaign/plan.json"), "stages": {
                    "untuned_before_training": {"status": "completed", "output_sha256": {
                        str(directory / name): report.digest(directory / name)
                        for name in ("untuned-pretraining-receipt.json", "requests.jsonl")}}}})
            result = collect(root, 0)
            self.assertEqual(result["status"], "DRAFT / PARTIAL CHECKPOINT")
            self.assertTrue(result["validation"]["untouched_gate"])
            self.assertFalse(result["validation"]["trained_head_proof"])
            self.assertIsNone(result["artifact_roles"]["trained"])
            self.assertIsNone(result["artifact_roles"]["cache_fit"])
            self.assertNotIn("EM %", report.plain_text(result))
            self.assertEqual(result["train_reconciliation"]["saved_trajectories"], 2)
            self.assertTrue(result["validation"]["baseline_diagnostics"])
            native = result["baseline_development"]
            self.assertAlmostEqual(native["policies"]["strands-untuned"]["em"], .45)
            self.assertAlmostEqual(native["policies"]["fixed-1-prefix"]["token_f1"], .45)
            self.assertEqual((native["approvals"], native["correct_approvals"], native["wrong_approvals"],
                              native["rejections_of_correct_drafts"]), (10, 5, 5, 4))
            self.assertEqual(native["usage"]["completed_requests_including_invalid_outputs"], 20)
            self.assertEqual(native["usage"]["reported_physical_input_tokens"], 2190)
            self.assertEqual(native["timing"]["startup_plus_phase_wall_seconds"], 123.)
            self.assertIn("DEVELOPMENT / cap1", report.plain_text(result))
            usage = next(s for s in result["sections"] if s["title"].startswith("6."))["rows"]
            self.assertEqual([(r[0], r[1]) for r in usage],
                [("TRAIN writer", "2"), ("Untouched DEVELOPMENT native decisions", "20")])
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
        for change in ("missing-journal", "unbound-journal", "quality", "raw-score", "usage", "tasks", "public-state", "idle", "summary", "missing-startup"):
            with self.subTest(baseline_change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture(root); directory = native_baseline_fixture(root)
                receipt_path = directory / "untuned-pretraining-receipt.json"
                receipt = json.loads(receipt_path.read_text())
                if change == "missing-journal": (directory / "requests.jsonl").unlink()
                elif change == "unbound-journal": receipt.pop("requests_sha256")
                elif change == "quality":
                    path = directory / "episodes.json"; rows = json.loads(path.read_text())
                    rows[1]["exact_match"] = False; write(path, rows); receipt["episodes_sha256"] = report.digest(path)
                elif change in ("usage", "idle", "missing-startup"):
                    path = directory / "manifest.json"; value = json.loads(path.read_text())
                    if change == "usage": value["actual_evaluation_usage"]["reported_physical_input_tokens"] += 1
                    elif change == "idle": value["this_invocation_intentional_idle_seconds"] = 121.
                    else: value.pop("backend_startup_this_invocation")
                    write(path, value); receipt["manifest_sha256"] = report.digest(path)
                elif change == "tasks":
                    path = directory / "tasks-evaluator-only.json"; rows = json.loads(path.read_text())
                    rows[0]["answers"] = ["different reference"]; write(path, rows)
                elif change in ("raw-score", "public-state"):
                    path = directory / "requests.jsonl"; rows = [json.loads(s) for s in path.read_text().splitlines()]
                    if change == "raw-score": rows[1]["result"]["p_true"] = .1
                    else:
                        rows[0]["input"]["state"]["question"] = "Foreign public state"
                        rows[0]["input_sha256"] = rows[1]["input_sha256"] = fingerprint(rows[0]["input"])
                    path.write_text("".join(json.dumps(r)+"\n" for r in rows)); receipt["requests_sha256"] = report.digest(path)
                else: write(directory / "summary.json", {"em": 1., "fabricated_summary": True})
                write(receipt_path, receipt)
                result = collect(root, 0)
                if change in ("summary", "missing-startup"):
                    self.assertTrue(result["validation"]["baseline_diagnostics"], result["validation_errors"])
                    self.assertAlmostEqual(result["baseline_development"]["policies"]["strands-untuned"]["em"], .45)
                    if change == "missing-startup":
                        self.assertIsNone(result["baseline_development"]["timing"]["startup_plus_phase_wall_seconds"])
                    self.assertNotIn(str(directory / "summary.json"), {r["path"] for r in result["input_files"]})
                else:
                    self.assertFalse(result["validation"]["untouched_gate"])
                    self.assertIsNone(result["baseline_development"])
                    self.assertFalse(any(r[0] == "Untouched DEVELOPMENT native decisions" for s in result["sections"] for r in s["rows"]))

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
