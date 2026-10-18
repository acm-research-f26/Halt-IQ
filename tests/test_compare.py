"""Synthetic protocol fixtures only; these are not benchmark/model results."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from pc_strands.compare import auroc, compare_runs, first_draft_diagnostics, policy_statistics
from pc_strands.data import fingerprint
from pc_strands.decider import sha256_file
from pc_strands.metrics import evaluate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(root):
    """Three toy questions; one explicit overflow in both decision policies."""
    tasks = [{"id": f"q{i}", "question": f"Question{i}?", "evidence": ["public"],
              "answers": [answer], "category": "toy", "partition": "confirmation"}
             for i, answer in enumerate(("Alpha", "Beta", "Gamma"), 1)]
    drafts = (("wrong", "Alpha"), ("Beta", "wrong"), ("Gamma", "Gamma"))
    trajectories = [{"task_id": t["id"], "status": "completed", "rounds": [
        {"round": n, "status": "completed", "draft": answer, **evaluate(answer, t["answers"])}
        for n, answer in enumerate(ds, 1)]} for t, ds in zip(tasks, drafts)]
    trajectory = root / "generation"
    write(trajectory / "tasks-evaluator-only.json", tasks)
    write(trajectory / "trajectories.json", trajectories)
    usage = {"completed_requests_including_invalid_outputs": 6, "unreconciled_requests": 0,
             "reported_physical_input_tokens": 600, "reported_output_tokens": 6}
    write(trajectory / "manifest.json", {"status": "completed", "writer_unload_verified": True,
                                        "actual_evaluation_usage": usage})
    cache = root / "features.jsonl"
    cache.write_text('{"synthetic_fixture":true}\n')
    manifest = {"status": "completed", "unreconciled_inferences": 0,
                "cached_rows_sha256": sha256_file(cache), "events_sha256": None, "failures_sha256": None,
                "frozen": {"trajectories_sha256": sha256_file(trajectory / "trajectories.json"),
                           "backend_sha256": "frozen-backend",
                           "tasks_sha256": sha256_file(trajectory / "tasks-evaluator-only.json")},
                "physical_encoding_usage": {"completed_encoder_forwards": 6, "input_tokens": 600},
                "encoder": {"temperature": 1.0704394404346953}}
    write(Path(str(cache) + ".manifest.json"), manifest)
    def episode(index, number, policy, *, failed=False, probabilities=()):
        tr, t = trajectories[index], tasks[index]
        answer = tr["rounds"][number - 1]["draft"]
        return {"task_id": t["id"], "policy": policy, "status": "failed" if failed else "completed",
                "rounds": number, "final_answer": answer,
                **(evaluate(answer, t["answers"]) if not failed else {"exact_match": False, "token_f1": 0.0}),
                "first_draft_exact_match": tr["rounds"][0]["exact_match"],
                "writer_input_tokens": 100 * number, "writer_output_tokens": number,
                "decision_input_tokens": 100 * len(probabilities), "decision_calls": len(probabilities),
                "observed_prefix_model_seconds": .1 * number, "cost_usd": 0.0,
                "stop_reason": "ContextWindowError" if failed else "approved" if probabilities else "fixed_prefix",
                "decisions": [{"round": n, "draft_exact_match": tr["rounds"][n - 1]["exact_match"],
                    "approved": p >= .8, "score": {"p_true": p, "temperature": 1.,
                    "logits": [0., math.log(p / (1 - p))]}} for n, p in enumerate(probabilities, 1)]}
    runs = []
    for name, selections, probs in (("strands-untuned", (1, 2, 1), ((.9,), (.2, .9), ())),
                                     ("strands-tuned", (2, 1, 1), ((.2, .9), (.9,), ()))):
        directory = root / name
        write(directory / "tasks-evaluator-only.json", tasks)
        snapshot = directory / "source_snapshot"
        snapshot.mkdir()
        (snapshot / "evaluate.py").write_text('# Synthetic source fingerprint fixture.\n')
        source_hash = fingerprint({"evaluate.py": sha256_file(snapshot / "evaluate.py")})
        config = {"mode": "head-replay", "dataset_sha256": "fixture", "max_drafts": 2,
                  "seed": 42, "feedback": "fixed generic", "phase": "confirmation",
                  "resource_profile": {"ram_gib": 7},
                  "trajectory_sha256": sha256_file(trajectory / "trajectories.json"),
                  "feature_cache_sha256": sha256_file(cache),
                  "feature_manifest_sha256": sha256_file(Path(str(cache) + ".manifest.json")),
                  "backend_source_sha256": "frozen-backend", "threshold": .8, "policy_name": name}
        episodes = [episode(i, n, f"fixed-{n}-prefix") for n in (1, 2) for i in range(3)]
        episodes += [episode(i, n, name, failed=i == 2, probabilities=probs[i]) for i, n in enumerate(selections)]
        events = []
        for i, probabilities in enumerate(probs):
            for n in range(max(1, len(probabilities))):
                key = f"q{i+1}|{name}|{n+1}"
                events.extend([{"event": "request_started", "request_id": key},
                               {"event": "request_completed", "request_id": key}])
        (directory / "requests.jsonl").write_text(''.join(json.dumps(e) + '\n' for e in events))
        write(directory / "episodes.json", episodes)
        write(directory / "manifest.json", {"status": "completed_with_errors", "frozen": {
            "config": config, "source_sha256": source_hash, "task_ids": [t["id"] for t in tasks],
            "dataset_state_sha256": fingerprint(tasks)}, "actual_evaluation_usage": {
            "unreconciled_requests": 0, "completed_requests_including_invalid_outputs": len(events) // 2,
            "reported_physical_input_tokens": 0}})
        runs.append(directory)
    return *runs, trajectory, cache


class ProbabilityTests(unittest.TestCase):
    def test_auc_ranks_and_ties(self):
        self.assertEqual(auroc([0, 1], [.1, .9]), 1.)
        self.assertEqual(auroc([0, 1], [.9, .1]), 0.)
        self.assertEqual(auroc([0, 1], [.5, .5]), .5)
        self.assertIsNone(auroc([1, 1], [.2, .8]))
        with self.assertRaises(ValueError):
            auroc([0, 2], [.1, .8])

    def test_diagnostic_missing_probability_is_coverage_not_zero_probability(self):
        rows = {"q1": {"decisions": [{"round": 1, "draft_exact_match": True,
                  "approved": True, "score": {"p_true": .9}}]}}
        result = first_draft_diagnostics(rows, ["q1", "q2"])
        self.assertEqual(result["scored_first_drafts"], 1)
        self.assertEqual(result["probability_coverage_of_planned"], .5)
        self.assertAlmostEqual(result["brier"], .01)

    def test_probability_cannot_disagree_with_temperature_logits(self):
        rows = {"q": {"decisions": [{"round": 1, "draft_exact_match": True, "approved": True,
                "score": {"p_true": .9, "logits": [0., 0.], "temperature": 1.}}]}}
        with self.assertRaises(ValueError):
            first_draft_diagnostics(rows, ["q"])


class ComparisonTests(unittest.TestCase):
    def test_planned_failures_zero_paired_ids_and_dedup_shared_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = fixture(Path(directory))
            result = compare_runs(*paths, draws=100)
            self.assertEqual(result["planned_n"], 3)
            self.assertEqual(result["policies"]["strands-tuned"]["em"], 2 / 3)
            self.assertEqual(result["policies"]["strands-tuned"]["completed"], 2)
            self.assertEqual(result["deduplicated_identical_fixed_controls"], ["fixed-1-prefix", "fixed-2-prefix"])
            pair = result["paired_comparisons"]["strands-tuned_minus_strands-untuned"]["exact_match"]
            self.assertEqual(pair["planned_n"], 3)
            self.assertEqual(pair["complete_pair_n"], 2)
            self.assertEqual(pair["candidate_minus_baseline"], 2 / 3)
            self.assertEqual(result["physical_evaluation_usage"]["writer_generation_once"]["reported_physical_input_tokens"], 600)
            self.assertEqual(result["physical_evaluation_usage"]["tuned_head_scoring"]["reported_physical_input_tokens"], 0)
            again = compare_runs(*paths, draws=100)
            self.assertEqual(result["paired_comparisons"], again["paired_comparisons"])

    def test_changed_protocol_or_order_is_rejected(self):
        for change in ("threshold", "source_sha256", "order"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                paths = fixture(Path(directory))
                p = paths[1] / "manifest.json"
                manifest = json.loads(p.read_text())
                if change == "threshold": manifest["frozen"]["config"]["threshold"] = .9
                elif change == "source_sha256": manifest["frozen"][change] = "different"
                else: manifest["frozen"]["task_ids"].reverse()
                write(p, manifest)
                with self.assertRaises(ValueError): compare_runs(*paths, draws=10)

    def test_duplicate_or_nonidentical_fixed_controls_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                paths = fixture(Path(directory))
                p = paths[1] / "episodes.json"
                rows = json.loads(p.read_text())
                if duplicate: rows.append(copy.deepcopy(rows[0]))
                else: rows[0]["writer_input_tokens"] += 1
                write(p, rows)
                with self.assertRaises(ValueError): compare_runs(*paths, draws=10)

    def test_interrupted_or_pending_run_is_rejected(self):
        for pending in (False, True):
            with self.subTest(pending=pending), tempfile.TemporaryDirectory() as directory:
                paths = fixture(Path(directory))
                if pending:
                    with (paths[1] / "requests.jsonl").open('a') as handle:
                        handle.write(json.dumps({"event": "request_started", "request_id": "unsettled"}) + '\n')
                else:
                    p = paths[1] / "manifest.json"; manifest = json.loads(p.read_text())
                    manifest["status"] = "interrupted"; write(p, manifest)
                with self.assertRaises(ValueError): compare_runs(*paths, draws=10)

    def test_changed_actual_cache_or_trajectory_is_rejected(self):
        for index in (2, 3):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                paths = fixture(Path(directory))
                p = paths[index] / "trajectories.json" if index == 2 else paths[index]
                p.write_text(p.read_text() + ' ')
                with self.assertRaises(ValueError): compare_runs(*paths, draws=10)

    def test_optional_retention_is_secondary_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = fixture(Path(directory))
            primary = compare_runs(*paths, draws=10)
            retained = compare_runs(*paths, draws=10, retain_best=True)
            self.assertEqual(primary["policies"]["strands-tuned"], retained["policies"]["strands-tuned"])
            self.assertEqual(retained["policies"]["strands-tuned-retainbest"]["mean_raw_reported_tokens"],
                             retained["policies"]["strands-tuned"]["mean_raw_reported_tokens"])
            self.assertIn("strands-tuned-retainbest", retained["secondary_retain_best_policies"])
            self.assertEqual(retained["policies"]["strands-tuned-retainbest"]["failed_or_missing"], 1)

    def test_missing_policy_row_is_zero_quality_but_unknown_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = fixture(Path(directory))
            p = paths[0] / "episodes.json"
            rows = [r for r in json.loads(p.read_text()) if not (r["policy"] == "strands-untuned" and r["task_id"] == "q3")]
            write(p, rows)
            result = compare_runs(*paths, draws=10)
            row = result["policies"]["strands-untuned"]
            self.assertEqual(row["planned_n"], 3)
            self.assertEqual(row["failed_or_missing"], 1)
            self.assertEqual(row["missing_episode_rows"], 1)
            self.assertIsNone(row["mean_raw_reported_tokens"])
            self.assertFalse(row["operational_usage_complete"])


if __name__ == "__main__":
    unittest.main()
