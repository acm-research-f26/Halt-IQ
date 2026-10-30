"""Fresh Hotpot collection, declared policy replay and independent online tests.

Evaluator labels are confined to scoring after each gold-blind episode. A full
trajectory collection pays its complete physical cost; policy prefix costs are
separately labelled counterfactual estimates, never measured live savings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

from . import hotpot
from .aligned_gate import Runtime
from .linear import lexical_distance
from .replay import load_trajectories
from .team_training import StrandsHead


SPECS = {
    "fixed1": ("fixed1", "last"), "fixed6": ("fixed6", "last"),
    "answer_unchanged": ("repeat", "last"), "strands": ("strands", "last"),
    "orchestrator": ("orchestrator", "last"),
    "retrained": ("retrained", "last"),
    "retrained_and": ("retrained_and", "last"),
    "retrained_best": ("retrained", "best_seen"),
    "retrained_and_best": ("retrained_and", "best_seen"),
    "fixed6_best": ("fixed6", "best_seen"),
}
ONLINE_POLICIES = ("fixed1", "fixed6", "strands", "retrained", "retrained_best")


def _digest(value):
    return hashlib.sha256(hotpot._json(value).encode("utf-8")).hexdigest()


def token_totals(entries):
    aliases = {"prompt_tokens": "inputTokens", "completion_tokens": "outputTokens",
               "total_tokens": "totalTokens"}
    result = {}
    for key, alias in aliases.items():
        values = [entry.get("usage", {}).get(key, entry.get("usage", {}).get(alias))
                  for entry in entries]
        result[key] = (sum(values) if values and all(type(v) is int and v >= 0
                       for v in values) else None)
    return result


def bounded_call(agents, role, system, prompt, parser, trace, *, corrections=1):
    """One public formatting correction, with every original call preserved."""
    if corrections not in (0, 1):
        raise ValueError("The declared formatting budget is zero or one correction")
    original = prompt
    for attempt in range(corrections + 1):
        raw = agents.generate(role, system, prompt)
        trace({"event": role + "_output", "format_attempt": attempt + 1, "raw": raw})
        try:
            return parser(raw)
        except ValueError as error:
            trace({"event": "format_failure", "role": role,
                   "format_attempt": attempt + 1, "message": str(error)})
            if attempt == corrections:
                raise
            prompt = (original + "\nYour previous response did not satisfy the declared output format."
                " Return a new response using only the supplied public passages.\n"
                + hotpot._json({"previous_output": raw, "format_error": str(error)}))


def select_seen(steps, selector):
    """Choose among the supplied observed prefix only; ties prefer earlier."""
    if not steps:
        raise ValueError("A returned answer requires an observed valid draft")
    if selector == "last":
        return steps[-1]
    if selector != "best_seen":
        raise ValueError("Unknown predeclared answer selector")
    if any(step["decision"].get("p_approve") is None for step in steps):
        raise ValueError("Verifier selection requires a score for every seen draft")
    return max(steps, key=lambda step: (step["decision"]["p_approve"], -step["round"]))


def execute_episode(agents, gate, public, policy, *, collect=False,
                    max_rounds=6, corrections=1, trace=None):
    """The entire generation boundary: no gold object is accepted here."""
    if policy not in SPECS or type(max_rounds) is not int or not 1 <= max_rounds <= 6:
        raise ValueError("Unknown policy or invalid declared round cap")
    trace = trace or (lambda event: None)
    mode, selector = SPECS[policy]
    prompt = hotpot._question_prompt(public)
    passages = [title + "\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
                for title, sentences in public.passages]
    previous, previous_distance, feedback = None, None, ""
    steps, status, error, forced_cap = [], "completed", None, False
    started, usage_start = time.perf_counter(), len(agents.usage)
    selected, critic_attempts = None, 0
    try:
        for round_index in range(1, max_rounds + 1):
            current = prompt
            if previous is not None:
                current += "\nRevise this previous answer using independent reviewer feedback:\n" + hotpot._json({
                    "previous_draft": steps[-1]["draft"], "feedback": feedback})
            call_start = len(agents.usage)
            draft = bounded_call(agents, "writer", hotpot.WRITER_SYSTEM, current,
                                 hotpot.parse_draft, trace, corrections=corrections)
            writer_usage = list(agents.usage[call_start:])
            step = {"round": round_index, "draft": draft,
                    "writer_usage": writer_usage, "reviewer_usage": [],
                    "decision": {"p_approve": None,
                        "distance": 1. if previous is None else lexical_distance(previous, draft["answer"]),
                        "physical_critic_calls": 0}}
            # Preserve a valid draft even if its verifier subsequently fails.
            # A verifier-free counterfactual can still return that draft.
            steps.append(step)
            # Collection deliberately continues to six; scores/head decisions
            # are observations, not routes. Best-of-six also requires all scores.
            scoring_mode = "retrained" if collect or selector == "best_seen" else mode
            critic_attempts += int(scoring_mode not in {"fixed1", "fixed6", "repeat"})
            decision = gate.decide(scoring_mode, draft["answer"], previous,
                round_index, max_rounds, public.question, passages,
                previous_distance=previous_distance)
            if selector == "best_seen" and not collect:
                # The trained best-prefix arm shares the head's stopping rule;
                # a full-six selector keeps its fixed budget independently.
                if mode == "fixed6":
                    decision = dict(decision, stop=round_index == max_rounds,
                                    reason="fixed_six_collection_for_selection")
                elif mode == "retrained_and":
                    decision = dict(decision, stop=decision["stop"]
                                    and decision["p_approve"] >= gate.approve_threshold)
            step["decision"] = decision
            trace({"event": "draft_scored", **step})
            terminal = round_index == max_rounds or (decision["stop"] and not collect)
            if terminal:
                forced_cap = round_index == max_rounds and not decision["stop"] and mode != "fixed6"
                selected = select_seen(steps, selector if not collect else "last")
                break
            call_start = len(agents.usage)
            review = bounded_call(agents, "reviewer", hotpot.REVIEWER_SYSTEM,
                prompt + "\nProposed draft:\n" + hotpot._json(draft),
                hotpot.parse_review, trace, corrections=corrections)
            step["review"] = review
            step["reviewer_usage"] = list(agents.usage[call_start:])
            previous, previous_distance, feedback = draft["answer"], decision["distance"], review["feedback"]
    except Exception as exc:
        status, error = "failed", {"type": type(exc).__name__, "message": str(exc)}
        trace({"event": "episode_error", "error": error})
    usage = list(agents.usage[usage_start:])
    return {"question_id": public.id, "policy": policy, "status": status,
            "prediction": selected["draft"] if selected is not None else None,
            "generated_rounds": len(steps),
            "selected_round": selected["round"] if selected is not None else None,
            "forced_cap_return": forced_cap, "steps": steps, "error": error,
            "max_rounds": max_rounds,
            "usage": usage, "physical_model_calls": len(usage),
            "model_call_count_scope": "writer/reviewer request attempts, including failed calls",
            "critic_call_attempts": critic_attempts,
            "physical_critic_calls": sum(s["decision"]["physical_critic_calls"] for s in steps),
            "seconds": time.perf_counter() - started,
            "reported_agent_tokens": token_totals(usage)}


def replay_prefix(collection, policy, *, head, approve_threshold=.8):
    """Counterfactual stopping/selection over an already collected trajectory."""
    from .team_training import make_strands_features
    if policy not in SPECS:
        raise ValueError("Unknown replay policy")
    mode, selector = SPECS[policy]
    cap = collection.get("max_rounds", 6)
    if type(cap) is not int or not 1 <= cap <= 6:
        raise ValueError("Invalid collected round cap")
    seen, used, selected, previous_distance = [], [], None, None
    stop, forced, head_calls, critic_calls = False, False, 0, 0
    for step in collection["steps"]:
        i = step["round"]
        score = step["decision"]["p_approve"]
        previous = seen[-1]["draft"]["answer"] if seen else None
        repeated = previous is not None and step["draft"]["answer"].strip().lower() == previous.strip().lower()
        seen.append(step)
        used.extend(step["writer_usage"])
        needs_score = mode in {"strands", "orchestrator", "retrained", "retrained_and"} or selector == "best_seen"
        if needs_score and score is None:
            break
        p_stop = None
        if mode in {"retrained", "retrained_and"} and i < cap:
            vector = make_strands_features(step["draft"]["answer"], previous, i - 1, cap,
                                           score, previous_distance)
            p_stop = head.probability(vector)
            head_calls += 1
        if needs_score:
            critic_calls += 1
        approved = score is not None and score >= approve_threshold
        stop = {"fixed1": True, "fixed6": i == cap, "repeat": repeated,
                "strands": approved,
                "orchestrator": approved or repeated,
                "retrained": p_stop is not None and p_stop >= .5,
                "retrained_and": p_stop is not None and p_stop >= .5 and approved}[mode]
        if stop or i == cap:
            forced = i == cap and not stop and mode != "fixed6"
            selected = select_seen(seen, selector)
            break
        used.extend(step["reviewer_usage"])
        previous_distance = step["decision"]["distance"]
    if selected is None:
        # A policy requiring the missing future cannot silently return an older
        # answer. Retain every attempted physical call on this failed path.
        used = collection["usage"]
    return {"question_id": collection["question_id"], "policy": policy,
            "execution": "counterfactual_prefix_replay",
            "status": "completed" if selected is not None else "failed",
            "prediction": selected["draft"] if selected is not None else None,
            "metrics": selected.get("metrics", {}) if selected is not None else {},
            "generated_rounds": seen[-1]["round"] if seen else 0,
            "selected_round": selected["round"] if selected is not None else None,
            "forced_cap_return": forced, "strands_calls_estimated": critic_calls,
            "head_calls_estimated": head_calls,
            "writer_reviewer_tokens_estimated": token_totals(used),
            "cost_scope": "saved measured-call prefix estimate; excludes verifier/reranker compute and is not live latency"}


def summarize(records, policies, planned):
    result = {}
    baseline = {r["question_id"]: r for r in records if r["policy"] == "fixed6"}
    for policy in policies:
        rows = [r for r in records if r["policy"] == policy]
        value = {"planned": planned, "recorded": len(rows),
                 "valid": sum(r["status"] == "completed" for r in rows),
                 "failed": sum(r["status"] != "completed" for r in rows),
                 "metrics": {m: sum(r.get("metrics", {}).get(m, 0.) for r in rows) / planned
                             for m in hotpot.METRICS},
                 "mean_generated_rounds": sum(r["generated_rounds"] for r in rows) / planned,
                 "forced_cap_returns": sum(r["forced_cap_return"] for r in rows)}
        counts = [r.get("reported_agent_tokens", r.get("writer_reviewer_tokens_estimated", {})).get("total_tokens") for r in rows]
        value["mean_writer_reviewer_tokens"] = (sum(counts) / planned
            if rows and all(type(x) is int for x in counts) else None)
        value["token_scope"] = ("counterfactual measured-call prefix estimates"
            if rows and rows[0].get("execution") == "counterfactual_prefix_replay"
            else "reported independent online writer/reviewer calls")
        value["mean_verifier_calls"] = sum(r.get("strands_calls_estimated", r.get("critic_call_attempts", 0)) for r in rows) / planned
        if len(rows) == planned and len(baseline) == planned:
            differences = [r.get("metrics", {}).get("f1", 0.)
                - baseline[r["question_id"]].get("metrics", {}).get("f1", 0.) for r in rows]
            rng = random.Random(42)
            samples = sorted(sum(rng.choices(differences, k=planned)) / planned for _ in range(5000))
            value["paired_f1_delta_ci95_vs_fixed6"] = [samples[124], samples[4874]]
        result[policy] = value
    return {"policies": result, "planned_per_policy": planned,
            "complete": all(len([r for r in records if r["policy"] == p]) == planned for p in policies),
            "denominator": "every planned question, failed outputs zero",
            "ci_scope": "descriptive paired pilot intervals; no multiplicity adjustment"}


def validate_fresh_inputs(head, cohort, head_sha256):
    """Reject reused questions or a different predeclared deployment head."""
    trajectories = load_trajectories(Path(__file__).resolve().parents[1] / "data/sri_rounds_with_strands.csv")
    expected = {split: {t["id"] for t in trajectories if t["split"] == split}
                for split in ("train", "dev", "test")}
    ids = cohort["question_ids"]
    if (cohort["dataset"] != hotpot.DATASET_ID or cohort["revision"] != hotpot.DATASET_REVISION
            or type(cohort["limit"]) is not int or cohort["limit"] < 1
            or len(ids) != cohort["limit"] or len(set(ids)) != len(ids)):
        raise ValueError("A unique prospective cohort on the pinned dataset is required")
    if (head.seed != 0 or head.metadata.get("trained") is not True
            or len(head.metadata.get("train_ids", [])) != 50
            or len(head.metadata.get("dev_ids", [])) != 20
            or set(head.metadata["train_ids"]) != expected["train"]
            or set(head.metadata["dev_ids"]) != expected["dev"]):
        raise ValueError("The head must be seed zero fitted on the pinned train50/dev20 only")
    if ({hotpot._bare_id(qid) for qid in ids}.intersection(
            hotpot._bare_id(qid) for qid in set().union(*expected.values()))):
        raise ValueError("Fresh confirmation cannot reuse teammate training, dev or cached test questions")
    primary = cohort.get("primary_live_head_sha256")
    if primary is not None and primary != head_sha256:
        raise ValueError("The head differs from the prospectively declared live checkpoint")


def run(output, head_path, cohort_path, *, mode="collect", policies=ONLINE_POLICIES):
    if (mode not in {"collect", "online"} or not policies or len(set(policies)) != len(policies)
            or set(policies) - SPECS.keys()):
        raise ValueError("Declare collection or online mode and known policies")
    output, head_path, cohort_path = map(Path, (output, head_path, cohort_path))
    head = StrandsHead.from_json(head_path)
    head_sha256 = hashlib.sha256(head_path.read_bytes()).hexdigest()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    validate_fresh_inputs(head, cohort, head_sha256)
    excluded = set().union(*(set(ids) for ids in cohort["exclusion_groups"].values()))
    chosen = hotpot.select_questions(hotpot.load_validation(), cohort["limit"], cohort["seed"], excluded)
    dataset = [hotpot.split_public_gold(row) for row in chosen]
    if ([public.id for public, _ in dataset] != cohort["question_ids"]
            or _digest([asdict(public) for public, _ in dataset]) != cohort["public_inputs_sha256"]):
        raise ValueError("Prospective public cohort identity changed")
    output.mkdir(parents=True, exist_ok=False)
    selected_policies = tuple(SPECS) if mode == "collect" else tuple(policies)
    manifest = {"schema": "yash-aligned-hotpot-v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode, "dataset": hotpot.DATASET_ID, "dataset_revision": hotpot.DATASET_REVISION,
        "cohort": cohort, "head": head.provenance,
        "head_file_sha256": head_sha256,
        "source_sha256": hotpot._source_hashes(), "policies": selected_policies,
        "policy_specs": {p: SPECS[p] for p in selected_policies}, "max_rounds": 6,
        "max_format_corrections_per_call": 1,
        "format_correction": "new public-only model call; original invalid output and all costs retained",
        "answer_selection": "maximum Strands among observed valid prefix drafts; earliest tie",
        "collector": "six writer drafts/five LLM reviews and six verifier scores; head does not stop collection",
        "cost_scope": "physical collection/online calls separate from counterfactual prefix estimates",
        "labels": "evaluator only, never writer/reviewer/head/selection input"}
    hotpot._atomic_json(output / "manifest.json", manifest)
    hotpot._atomic_json(output / "public_inputs.json", [asdict(public) for public, _ in dataset])
    records, collections = [], []
    with Runtime(output / "runtime", head) as (agents, gate):
        for public, gold in dataset:
            routes = ("fixed6",) if mode == "collect" else selected_policies
            offset = int(_digest([cohort["seed"], public.id, "arm-order"])[:8], 16) % len(routes)
            for policy in routes[offset:] + routes[:offset]:
                def trace(event):
                    hotpot._append_jsonl(output / "traces.jsonl", {"question_id": public.id,
                        "policy": "collection" if mode == "collect" else policy, **event})
                trace({"event": "episode_started"})
                episode = execute_episode(agents, gate, public, policy, collect=mode == "collect", trace=trace)
                # This is the only scope with gold labels. Selectors and the
                # generation boundary above cannot receive them.
                for step in episode["steps"]:
                    step["metrics"] = hotpot.score_prediction(step["draft"], gold)
                if mode == "collect":
                    collections.append(episode)
                    hotpot._append_jsonl(output / "collections.jsonl", episode)
                    for candidate in selected_policies:
                        record = replay_prefix(episode, candidate, head=head)
                        records.append(record)
                        hotpot._append_jsonl(output / "policy-results.jsonl", record)
                else:
                    episode["execution"] = "independent_online_policy"
                    episode["metrics"] = (hotpot.score_prediction(episode["prediction"], gold)
                        if episode["status"] == "completed" else {m: 0. for m in hotpot.METRICS})
                    records.append(episode)
                    hotpot._append_jsonl(output / "policy-results.jsonl", episode)
                trace({"event": "episode_finished", "status": episode["status"]})
                summary = summarize(records, selected_policies, cohort["limit"])
                summary.update(execution_mode=mode, physical_collected_trajectories=len(collections))
                if mode == "collect":
                    physical_usage = [item for c in collections for item in c["usage"]]
                    summary["physical_collection"] = {
                        "trajectories": len(collections),
                        "complete_six_round_trajectories": sum(c["status"] == "completed" for c in collections),
                        "writer_reviewer_call_attempts": len(physical_usage),
                        "verifier_call_attempts": sum(c["critic_call_attempts"] for c in collections),
                        "reported_agent_tokens": token_totals(physical_usage),
                        "cost_scope": "complete collection; never substitute policy-prefix estimates for physical cost"}
                hotpot._atomic_json(output / "summary.json", summary)
                print(json.dumps({"progress": {p: summary["policies"][p]["recorded"] for p in selected_policies}}), flush=True)
    hotpot._atomic_json(output / "evaluator_reference.json", {
        public.id: asdict(gold) for public, gold in dataset})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("collect", "online"), default="collect")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.head, args.cohort, mode=args.mode), indent=2))


if __name__ == "__main__":
    main()
