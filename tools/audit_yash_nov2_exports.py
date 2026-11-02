"""Offline, read-only audit of explicitly supplied November evidence exports.

No network, model, exported-code execution or runtime imports are used. Default
scores independently aggregate recorded metrics. Raw-answer scoring requires an
explicit post-freeze opt-in and a supplied dataset whose frozen digests match.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import string
import zipfile

ARMS = ("fixed1", "legacy_combined6", "strands", "binary_and", "always_revise3", "always_invoke", "learned3")
METRICS = ("em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec", "sp_recall", "joint_em", "joint_f1", "joint_prec", "joint_recall")
SOURCE_PIN = "1387ebd08ff907f0f07699e1d338e50b5552d1a9"
SOURCE_ZIP_PIN = "58ce7bab2786c45ced19389f4e3398e5fe446049c026f82b0c68f1e5c55b34d5"
PLAN_PIN = "100586312f72919898729f288c11934e9c9cc2c6dde1174d5778fc75796028d2"
DATASET = {"id": "hotpotqa/hotpot_qa", "revision": "1908d6afbbead072334abe2965f91bd2709910ab", "configuration": "distractor", "split": "validation"}
PACKAGES = {"strands-agents": "1.62.0", "strands-decider": "0.2.0", "transformers": "5.17.0", "peft": "0.21.0", "datasets": "5.1.0", "lm-format-enforcer": "0.11.3"}
WRITER = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
CRITIC_PINS = {"critic_model": "StrandsAgents/strands-decider-2B-hobson-v19", "critic_revision": "bb282d786bc251fd4e3068de3ada9ddbb38127cd", "base_revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c", "reranker_revision": "2cfc18c9415c912f9d8155881c133215df768a70"}
FEATURES = ("verifier_score", "reviewer_approved", "draft_fraction", "has_previous", "lexical_distance", "exact_repeat", "feedback_length_fraction", "evidence_issue_cue", "contradiction_cue", "challenger_used", "answer_length_fraction")
OBJECTIVE = "supervised_measured_one_step_utility_action_classification"
MAX_MEMBER, MAX_TOTAL, MAX_MEMBERS = 128 * 1024**2, 1024**3, 20000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value, *, compact=False):
    options = {"separators": (",", ":")} if compact else {}
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, **options).encode()


def strict_json(raw):
    def pairs(items):
        if len({key for key, _ in items}) != len(items):
            raise ValueError("Duplicate JSON keys")
        return dict(items)
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Nonfinite JSON number, including exponent overflow")
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))


def safe_name(name):
    if not isinstance(name, str):
        return False
    path = PurePosixPath(name)
    return (isinstance(name, str) and name and "\\" not in name and ":" not in name
            and not path.is_absolute() and name == path.as_posix()
            and all(part not in {"", ".", ".."} for part in path.parts))


class Evidence:
    """A bounded virtual file tree. ZIP members are never extracted."""
    def __init__(self, path):
        self.path, self.archive, self.files = Path(path), None, {}
        if self.path.is_symlink():
            raise ValueError("Evidence root must not be a symbolic link")
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path)
            if len(self.archive.infolist()) > MAX_MEMBERS:
                raise ValueError("Archive member-count cap exceeded, including directories")
            total, seen = 0, set()
            for member in self.archive.infolist():
                name = member.filename.rstrip("/")
                if not safe_name(name) or stat.S_ISLNK(member.external_attr >> 16) or name in seen:
                    raise ValueError("Unsafe archive path or symbolic link")
                seen.add(name)
                if member.is_dir():
                    continue
                if name in self.files or member.file_size > MAX_MEMBER:
                    raise ValueError("Duplicate archive member or member-size cap exceeded")
                self.files[name] = member
                total += member.file_size
                if total > MAX_TOTAL or len(self.files) > MAX_MEMBERS:
                    raise ValueError("Archive uncompressed size/member-count cap exceeded")
        elif self.path.is_dir():
            root = self.path.resolve()
            total = 0
            for item in sorted(self.path.rglob("*")):
                if item.is_symlink() or not item.resolve().is_relative_to(root):
                    raise ValueError("Evidence directory contains a symbolic link or escaping path")
                if not item.is_file():
                    continue
                size = item.stat().st_size
                if size > MAX_MEMBER:
                    raise ValueError("Evidence member-size cap exceeded")
                self.files[item.relative_to(self.path).as_posix()] = item
                total += size
                if total > MAX_TOTAL or len(self.files) > MAX_MEMBERS:
                    raise ValueError("Evidence size/member-count cap exceeded")
        else:
            raise ValueError("Supply an existing local ZIP or directory")

    def close(self):
        if self.archive:
            self.archive.close()

    def read(self, name):
        if name not in self.files:
            raise FileNotFoundError(name)
        if self.archive:
            with self.archive.open(self.files[name]) as stream:
                return bounded_read(stream, MAX_MEMBER)
        with self.files[name].open("rb") as stream:
            return bounded_read(stream, MAX_MEMBER)

    def json(self, name):
        return strict_json(self.read(name))

    def rows(self, name):
        rows = []
        for number, raw in enumerate(self.read(name).split(b"\n"), 1):
            if raw.strip():
                value = strict_json(raw)
                if not isinstance(value, dict):
                    raise ValueError(f"Non-object journal entry: {name}:{number}")
                rows.append(value)
        return rows

    def locate(self, basename):
        names = [name for name in self.files if PurePosixPath(name).name == basename]
        if len(names) > 1:
            raise ValueError("Ambiguous evidence file " + basename)
        return names[0] if names else None


class Audit:
    def __init__(self, evidence):
        self.evidence, self.findings = evidence, []

    def issue(self, code, detail, level="mismatch"):
        self.findings.append({"level": level, "code": code, "detail": str(detail)})

    def check(self, condition, code, detail, level="mismatch"):
        if not condition:
            self.issue(code, detail, level)
        return bool(condition)

    def read(self, name, *, rows=False, required=True):
        try:
            return self.evidence.rows(name) if rows else self.evidence.json(name)
        except FileNotFoundError:
            if required:
                self.issue("missing_file", name, "incomplete")
        except (ValueError, TypeError, UnicodeError, zipfile.BadZipFile) as error:
            self.issue("invalid_json_or_journal", f"{name}: {error}")
        return [] if rows else None

    def equal(self, left, right, code, label):
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            okay = math.isfinite(left) and math.isfinite(right) and abs(left-right) <= 1e-9
        else:
            okay = left == right
        return self.check(okay, code, f"{label}: recorded={left!r}, independently computed={right!r}")


def bounded_read(stream, limit):
    chunks, count = [], 0
    while True:
        chunk = stream.read(min(65536, limit-count+1))
        if not chunk:
            return b"".join(chunks)
        count += len(chunk)
        if count > limit:
            raise ValueError("Actual streamed evidence bytes exceeded member cap")
        chunks.append(chunk)


def verify_manifest(audit):
    evidence = audit.evidence
    name = evidence.locate("Yash-Archive-Manifest.json")
    if name is None:
        audit.issue("archive_manifest_missing", "No archive-wide hash manifest; directory/journal checks remain available", "incomplete")
        return
    manifest = audit.read(name)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        audit.issue("archive_manifest_invalid", name)
        return
    prefix = name[:-len("Yash-Archive-Manifest.json")]
    seen = set()
    for entry in manifest["files"]:
        member = entry.get("path")
        if not safe_name(member) or member in seen:
            audit.issue("archive_manifest_duplicate_or_unsafe", member)
            continue
        seen.add(member)
        try:
            raw = evidence.read(prefix + member)
            audit.equal(entry.get("bytes"), len(raw), "archive_member_size", member)
            audit.equal(entry.get("sha256"), sha(raw), "archive_member_hash", member)
        except FileNotFoundError:
            audit.issue("manifest_member_missing", member)
    expected = {prefix + member for member in seen} | {name}
    audit.equal(set(evidence.files), expected, "archive_membership", "Manifest must cover every retained file exactly")


def verify_source(audit, config, expected_source_zip_sha256):
    name = audit.evidence.locate("Yash-Nov2-Source.zip")
    if not name:
        audit.issue("embedded_source_missing", "No preserved source ZIP", "incomplete")
        return {}
    raw = audit.evidence.read(name)
    audit.equal(sha(raw), expected_source_zip_sha256, "trusted_source_zip_pin", "prepared source ZIP")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        members = archive.infolist()
        bounded = (len(members) <= 100 and sum(item.file_size for item in members) <= 5*1024**2
            and all(item.file_size <= 1024**2 and not stat.S_ISLNK(item.external_attr >> 16) for item in members))
        if not audit.check(bounded and len(names) == len(set(names)) and all(safe_name(path) for path in names), "source_zip_paths", "bounded regular source members"):
            return {}
        def source_read(path):
            with archive.open(path) as stream:
                return bounded_read(stream, 1024**2)
        source = config.get("source_sha256", {})
        audit.equal(set(source), {PurePosixPath(path).name for path in names if path.startswith("haltiq_pair/") and path.endswith(".py")}, "executed_source_membership", "all preserved modules")
        for basename, digest in source.items():
            path = "haltiq_pair/" + basename
            audit.check(path in names, "source_member_missing", path)
            if path in names:
                audit.equal(sha(source_read(path)), digest, "executed_source_hash", path)
        values = {}
        def literal(node):
            if isinstance(node, ast.Name):
                return values[node.id]
            if isinstance(node, ast.Dict):
                return {literal(key): literal(value) for key, value in zip(node.keys, node.values)}
            if isinstance(node, (ast.List, ast.Tuple)):
                return [literal(value) for value in node.elts]
            return ast.literal_eval(node)
        for node in ast.parse(source_read("haltiq_pair/nov2_orchestration.py")).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"ROLE_FORMAT_VERSION", "_CITATION_SCHEMA", "ROLE_JSON_SCHEMAS"}:
                values[node.targets[0].id] = literal(node.value)
        return values


def finite(value, low=0., high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def matches_schema(value, schema):
    kind = schema.get("type")
    if kind == "object":
        return (isinstance(value, dict) and set(value) == set(schema["required"])
                and all(matches_schema(value[key], nested) for key, nested in schema["properties"].items()))
    if kind == "array":
        return (isinstance(value, list) and len(value) <= schema.get("maxItems", 20)
                and all(matches_schema(item, schema["items"]) for item in value))
    if kind == "string":
        return isinstance(value, str) and len(value) >= schema.get("minLength", 0)
    if kind == "integer":
        return type(value) is int and value >= schema.get("minimum", 0)
    if kind == "boolean":
        return type(value) is bool
    return False


def canonical_facts(facts):
    if not isinstance(facts, list):
        raise ValueError("Citation list required")
    result = []
    for fact in facts:
        if isinstance(fact, dict):
            if set(fact) == {"title", "sentence_index"}:
                fact = [fact["title"], fact["sentence_index"]]
            elif set(fact) == {"exact_passage_title", "sentence_index"}:
                fact = [fact["exact_passage_title"], fact["sentence_index"]]
        if not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str) or type(fact[1]) is not int or fact[1] < 0:
            raise ValueError("Unsupported citation contract; no alias repair or index coercion")
        result.append(list(fact))
    return result


def canonical_writer(value):
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise ValueError("Nonempty raw writer answer required")
    return {"answer": value["answer"].strip(), "supporting_facts": canonical_facts(value["supporting_facts"])}


def canonical_challenger(value):
    if not isinstance(value, dict) or set(value) != {"alternative", "evidence", "feedback"} or any(not isinstance(value[key], str) or not value[key].strip() for key in ("alternative", "feedback")):
        raise ValueError("Valid raw challenger response required")
    return {"alternative": value["alternative"].strip(), "evidence": canonical_facts(value["evidence"]), "feedback": value["feedback"].strip()}


def raw_call(audit, study, call):
    identity = call.get("call_id")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity):
        audit.issue("raw_call_identity_invalid", identity)
        return None
    client = audit.read(study + "runtime/writer/by-client/" + identity + ".json")
    if not client:
        return None
    relative = client.get("primary_receipt")
    if not safe_name(relative) or not re.fullmatch(r"requests/chatcmpl-[0-9a-f]{32}\.json", relative):
        audit.issue("raw_call_path_invalid", identity)
        return None
    return audit.read(study + "runtime/writer/" + relative)


def message_text(messages):
    if isinstance(messages, str):
        return messages
    parts = []
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str))
    return "\n".join(parts)


def completed_path(audit, study, episode, calls, critics):
    key, arm = (episode["question_id"], episode["arm"]), episode["arm"]
    roles = Counter(call.get("role") for call in calls)
    audit.check(all(call.get("status") == "completed" for call in calls), "completed_episode_failed_role_call", key)
    writers = [call for call in calls if call.get("role") == "writer" and call.get("status") == "completed"]
    audit.check(bool(writers), "completed_episode_missing_writer", key)
    if writers:
        physical = raw_call(audit, study, writers[-1])
        if physical:
            audit.equal(episode.get("prediction"), canonical_writer(strict_json(physical["response_text"])), "primary_prediction_vs_raw_last_writer", key)
    rounds = episode.get("returned_draft_number")
    if arm == "fixed1":
        audit.check(roles == {"writer": 1} and not critics and rounds == 1, "fixed1_role_path", key)
    elif arm == "legacy_combined6":
        audit.check(type(rounds) is int and 1 <= rounds <= 6 and roles["writer"] == rounds
            and roles["reviewer"] == (rounds if rounds < 6 else 5) and not roles["challenger"] and len(critics) == rounds,
            "legacy_role_path", key)
        history = episode.get("legacy_trajectory")
        audit.check(isinstance(history, list) and len(history) == rounds, "legacy_draft_history", key)
    else:
        state = episode.get("state")
        if not isinstance(state, dict):
            audit.issue("completed_orchestrator_state_missing", key)
            return
        history = [item for item in state.get("history", []) if "draft" in item]
        audit.check(type(rounds) is int and 1 <= rounds <= 3 and len(history) == rounds
            and rounds <= roles["writer"] <= 2*rounds and rounds <= roles["reviewer"] <= 2*rounds
            and len(critics) == rounds and len(calls) <= 10, "matched_role_path", key)
        audit.equal(state.get("draft"), episode.get("prediction"), "state_primary_prediction", key)
        audit.equal(state.get("draft_number"), rounds, "state_returned_round", key)
        audit.equal([item.get("call_id") for item in state.get("usage", [])], [item.get("call_id") for item in calls], "state_physical_path_ids", key)
        invokes = episode.get("actions", []).count("INVOKE")
        audit.check(invokes <= 1 and (1 <= roles["challenger"] <= 2 if invokes else roles["challenger"] == 0), "challenger_action_call_binding", key)
        if arm == "always_invoke":
            audit.check(invokes == 1 and rounds >= 2, "always_invoke_path", key)
        if arm == "always_revise3":
            audit.check(rounds == 3 or (episode.get("termination") == "agent_budget_cap" and len(calls) >= 9), "always_revise_horizon", key)


def answer_scores(prediction, answer, facts):
    """Independent implementation of the twelve official HotpotQA quantities."""
    def normalize(text):
        text = "".join(char for char in str(text).lower() if char not in string.punctuation)
        return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", text).split())
    predicted, gold = normalize(prediction["answer"]), normalize(answer)
    pt, gt = predicted.split(), gold.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    special = {"yes", "no", "noanswer"}
    if predicted != gold and (predicted in special or gold in special):
        common = 0
    precision = common / len(pt) if pt else 0.
    recall = common / len(gt) if gt else 0.
    harmonic = lambda p, r: 2*p*r/(p+r) if p+r else 0.
    pf, gf = {tuple(item) for item in prediction["supporting_facts"]}, {tuple(item) for item in facts}
    overlap = len(pf & gf)
    spp, spr = overlap / len(pf) if pf else 0., overlap / len(gf) if gf else 0.
    em, spem = float(predicted == gold), float(pf == gf)
    return dict(zip(METRICS, (em, harmonic(precision, recall), precision, recall, spem,
        harmonic(spp, spr), spp, spr, em*spem, harmonic(precision*spp, recall*spr), precision*spp, recall*spr)))


def load_gold(path, plan, allow_heldout_scoring):
    # This guard deliberately precedes Path.read_bytes/open/stat of the gold file.
    if not allow_heldout_scoring:
        raise ValueError("Raw-answer rescoring requires explicit --allow-heldout-scoring after freeze confirmation")
    envelope = strict_json(Path(path).read_bytes())
    if not isinstance(envelope, dict) or envelope.get("dataset") != DATASET or not isinstance(envelope.get("rows"), list):
        raise ValueError("Dataset JSON must contain the exact pinned dataset metadata and rows")
    indexed = {}
    for row in envelope["rows"]:
        identity = row.get("id", row.get("_id"))
        if identity in indexed:
            raise ValueError("Dataset contains duplicate question IDs")
        indexed[identity] = row
    gold = {}
    for partition in ("train", "dev", "test"):
        publics, labels = [], []
        for identity in plan["partitions"][partition]["question_ids"]:
            row = indexed[identity]
            context = row["context"]
            passages = list(zip(context["title"], context["sentences"])) if isinstance(context, dict) else context
            facts = row["supporting_facts"]
            facts = list(zip(facts["title"], facts["sent_id"])) if isinstance(facts, dict) else facts
            publics.append({"id": identity, "question": str(row["question"]), "passages": passages})
            labels.append({"answer": str(row["answer"]), "supporting_facts": facts})
            gold[identity] = (str(row["answer"]), facts)
        if sha(canonical(publics)) != plan["partitions"][partition]["public_inputs_sha256"] or sha(canonical(labels)) != plan["partitions"][partition]["label_digest"]:
            raise ValueError("Provided dataset public-input or gold-label digest differs from the frozen plan")
    return gold


def aggregate(episodes, identities):
    """Pending resources remain unknown; their scores stay in the denominator."""
    answer = {}
    for arm in ARMS:
        rows = [row for row in episodes if row["arm"] == arm]
        status = Counter(row["status"] for row in rows)
        known = sum(row.get("known_recorded_tokens", row.get("total_tokens") or 0) for row in rows)
        complete = len(rows) == len(identities) and all(row.get("total_tokens") is not None and row.get("journal_complete") is True for row in rows)
        tokens = sum(row["total_tokens"] for row in rows) if complete else None
        calls = Counter()
        for row in rows:
            calls.update(row["agent_calls"])
        answer[arm] = {"planned": len(identities), "attempted": len(rows), "completed": status["completed"],
            "failed": status["failed"], "interrupted": status["interrupted"], "pending": len(identities)-len(rows),
            "metrics": {metric: sum(row["metrics"][metric] for row in rows if row["status"] == "completed") / len(identities) for metric in METRICS},
            "iterations": sum(row["iterations"] for row in rows),
            "mean_writer_iterations": sum(row["iterations"] for row in rows) / len(identities),
            "agent_calls": dict(sorted(calls.items())), "attempted_agent_calls": sum(row["attempted_agent_calls"] for row in rows),
            "total_tokens": tokens, "mean_total_tokens": tokens / len(identities) if tokens is not None else None,
            "known_recorded_tokens": known, "journal_complete": complete,
            "critic_calls": sum(row["critic_calls"] for row in rows)}
    return answer


def reconcile_calls(audit, study, calls, qualification, schemas):
    """Resolve portable relative paths, never the original machine's absolute paths."""
    evidence, seen_calls, seen_responses = audit.evidence, set(), set()
    reconciled, known, unknown = 0, 0, 0
    all_entries = [("study", item) for item in calls] + [("qualification", item) for item in (qualification or {}).get("role_usage", [])]
    for scope, call in all_entries:
        identity = call.get("call_id")
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity) or identity in seen_calls:
            audit.issue("call_identity_duplicate_or_invalid", identity)
            continue
        seen_calls.add(identity)
        path = study + "runtime/writer/by-client/" + identity + ".json"
        client = audit.read(path, required=False)
        if client is None:
            if scope == "study" and call.get("total_tokens") is not None:
                audit.issue("known_tokens_without_client_receipt", identity)
            else:
                audit.issue("unknown_response_tokens", identity, "incomplete")
            unknown += 1
            continue
        response_id = client.get("response_id")
        if not isinstance(response_id, str) or not re.fullmatch(r"chatcmpl-[0-9a-f]{32}", response_id) or response_id in seen_responses:
            audit.issue("response_identity_duplicate_or_invalid", response_id)
            continue
        seen_responses.add(response_id)
        primary_path = study + "runtime/writer/requests/" + response_id + ".json"
        physical = audit.read(primary_path)
        if physical is None:
            continue
        raw = evidence.read(primary_path)
        metadata = {"yash_call_id": identity, "yash_role": call.get("role")}
        usage = client.get("usage", {})
        if not all(type(usage.get(key)) is int and usage[key] >= 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            audit.issue("invalid_token_usage", identity)
            continue
        audit.equal(usage["total_tokens"], usage["prompt_tokens"]+usage["completion_tokens"], "token_sum", identity)
        for left, right, code in ((client.get("metadata"), metadata, "client_metadata"), (client.get("role"), call.get("role"), "client_role"),
                (client.get("call_id"), identity, "client_identity"), (physical.get("request", {}).get("metadata"), metadata, "primary_metadata"),
                (physical.get("id"), response_id, "primary_identity"), (physical.get("usage"), usage, "physical_usage"),
                (client.get("primary_sha256"), sha(raw), "physical_hash"),
                (client.get("primary_receipt"), "requests/"+response_id+".json", "physical_relative_path"),
                (client.get("finish_reason"), physical.get("finish_reason"), "finish_reason")):
            audit.equal(left, right, code, identity)
        if scope == "study":
            audit.equal(call.get("total_tokens"), usage["total_tokens"], "ledger_token_receipt", identity)
            receipt = call.get("response_receipt", {})
            audit.equal(receipt.get("primary_receipt_sha256"), sha(raw), "ledger_primary_hash", identity)
            audit.equal(receipt.get("client_receipt_sha256"), sha(evidence.read(path)), "ledger_client_hash", identity)
            known += usage["total_tokens"]
        check = audit.read(study + "runtime/writer/format-checks/" + identity + ".json")
        if check is not None:
            audit.equal(check.get("raw_response_sha256"), sha(physical.get("response_text", "").encode()), "format_raw_hash", identity)
            if call.get("status") == "completed":
                audit.equal(check.get("status"), "validated", "completed_format_status", identity)
                try:
                    value = strict_json(physical["response_text"])
                    audit.check(matches_schema(value, schemas.get(call.get("role"), {})), "completed_raw_schema", identity)
                except (ValueError, KeyError, TypeError):
                    audit.issue("completed_raw_json_invalid", identity)
        reconciled += 1
    primary_names = {name for name in evidence.files if name.startswith(study + "runtime/writer/requests/") and name.endswith(".json")}
    expected = {study + "runtime/writer/requests/" + identity + ".json" for identity in seen_responses}
    for path in primary_names - expected:
        physical = audit.read(path)
        identity = (physical or {}).get("request", {}).get("metadata", {}).get("yash_call_id")
        audit.issue("raw_response_without_client_mapping" if identity in seen_calls else "unmapped_physical_response",
                    path, "incomplete" if identity in seen_calls else "mismatch")
    return {"study_calls": len(calls), "qualification_calls": len(all_entries)-len(calls), "reconciled_responses": reconciled,
            "known_study_response_tokens": known, "unknown_response_calls": unknown,
            "scope": "writer/reviewer/challenger responses; no verifier token-pricing or hardware-total claim"}


def audit_heads(audit, study, config, plan, calls):
    freeze = audit.read(study + "freeze.json")
    training = audit.read(study + "heads/training-freeze.json")
    protocol = audit.read(study + "heads/protocol.json")
    if not all(isinstance(value, dict) for value in (freeze, training, protocol)):
        return {"status": "not_fully_available"}
    audit.equal(freeze.get("config_sha256"), sha(audit.evidence.read(study + "study-config.json")), "freeze_config_hash", "config")
    audit.equal(training.get("protocol_sha256"), sha(audit.evidence.read(study + "heads/protocol.json")), "freeze_training_protocol_hash", "training protocol")
    audit.equal(freeze.get("test_calls_at_freeze"), 0, "test_calls_at_freeze", "freeze marker")
    audit.equal(freeze.get("deployment_seed"), config.get("deployment_seed"), "frozen_deployment_seed", "declared deployment")
    audit.equal(training.get("all_selected_checkpoints_frozen_before_test_scoring"), True, "all_five_frozen_marker", "training freeze")
    audit.equal(training.get("test_used_to_fit_or_select"), False, "test_fit_marker", "training freeze")
    training_entries = {entry.get("seed"): entry for entry in training.get("checkpoints", [])}
    audit.equal(sorted(training_entries), list(range(5)), "training_freeze_seeds", "all five seeds")
    examples = audit.read(study + "training-examples.jsonl", rows=True)
    actual_ids = {split: {row["question_id"] for row in examples if row.get("split") == split} for split in ("train", "dev")}
    for split in ("train", "dev"):
        audit.equal(set(protocol.get(split + "_ids", [])), actual_ids[split], "training_protocol_ids", split)
    audit.equal(protocol.get("configuration"), config.get("policy_config"), "training_protocol_configuration", "fit configuration")
    collection = [call for call in calls if call.get("stage") in {"train", "dev"}]
    audit.equal(freeze.get("physical_collection_invocations"), len(collection), "freeze_collection_calls", "collection count")
    stages = [call.get("stage") for call in calls]
    if "test" in stages:
        audit.check(all(stage == "test" for stage in stages[stages.index("test") :]), "post_test_collection_call", "No collection calls may follow test start")
    entries = freeze.get("checkpoints", [])
    audit.check(all(type(entry.get("seed")) is int for entry in entries), "freeze_seed_types", "Seeds must be integers, not booleans")
    audit.equal([entry.get("seed") for entry in entries], list(range(5)), "freeze_seeds", "all five seeds")
    test_ids = set(plan["partitions"]["test"]["question_ids"])
    selected = []
    for entry in entries:
        path = entry.get("path")
        if not safe_name(path) or not path.startswith("heads/seed"):
            audit.issue("unsafe_checkpoint_path", path)
            continue
        head = audit.read(study + path)
        if head is None:
            continue
        audit.equal(entry.get("sha256"), sha(audit.evidence.read(study + path)), "frozen_head_hash", path)
        audit.equal(training_entries.get(entry["seed"], {}).get("sha256"), entry.get("sha256"), "both_freeze_head_hashes", path)
        payload = dict(head)
        digest = payload.pop("payload_sha256", None)
        audit.equal(digest, sha(canonical(payload, compact=True)), "head_payload_hash", path)
        audit.equal(head.get("seed"), entry["seed"], "head_seed", path)
        audit.equal(head.get("features"), list(FEATURES), "head_features", path)
        audit.equal(head.get("actions"), ["STOP", "REVISE", "INVOKE"], "head_actions", path)
        audit.equal(head.get("objective"), OBJECTIVE, "head_objective", path)
        audit.check(len(head.get("weights", [])) == 3 and all(len(row) == 11 and all(finite(x, -math.inf) for x in row) for row in head.get("weights", []))
                    and len(head.get("bias", [])) == 3 and all(finite(x, -math.inf) for x in head.get("bias", [])), "head_parameters", path)
        metadata = head.get("metadata", {})
        train, dev = set(metadata.get("train_ids", [])), set(metadata.get("dev_ids", []))
        audit.equal(train, actual_ids["train"], "head_actual_train_ids", path)
        audit.equal(dev, actual_ids["dev"], "head_actual_dev_ids", path)
        audit.check(bool(train) and bool(dev) and not (train & dev or (train | dev) & test_ids)
            and train <= set(plan["partitions"]["train"]["question_ids"]) and dev <= set(plan["partitions"]["dev"]["question_ids"]), "head_partition_leakage", path)
        history = audit.read(study + path.rsplit("/", 1)[0] + "/history.json")
        epochs = history.get("epochs") if isinstance(history, dict) else None
        correct = (isinstance(epochs, list) and len(epochs) == config.get("policy_config", {}).get("epochs")
            and all(isinstance(row, dict) and type(row.get("epoch")) is int and finite(row.get("dev_greedy_utility_after_step"), -math.inf) for row in epochs))
        if audit.check(correct, "head_history_structure", path):
            audit.equal([row["epoch"] for row in epochs], list(range(1, len(epochs)+1)), "head_epoch_order", path)
            first_best = max(epochs, key=lambda row: row["dev_greedy_utility_after_step"])
            audit.equal(metadata.get("selected_epoch"), first_best["epoch"], "dev_checkpoint_selection", path)
            audit.equal(head.get("weights"), first_best["weights"], "selected_weights", path)
            audit.equal(head.get("bias"), first_best["bias"], "selected_bias", path)
            audit.equal(history.get("optimizer_steps_completed"), config.get("policy_config", {}).get("epochs"), "optimizer_steps", path)
        selected.append({"seed": entry["seed"], "sha256": entry.get("sha256"), "selected_epoch": metadata.get("selected_epoch")})
    frozen_time = datetime.fromisoformat(freeze["frozen_utc"]).timestamp()
    training_time = datetime.fromisoformat(training["frozen_utc"]).timestamp()
    audit.check(training_time <= frozen_time, "freeze_time_order", "Training freeze precedes study freeze")
    for call in calls:
        if call.get("stage") == "test" and call.get("response_receipt", {}).get("response_id"):
            path = study + "runtime/writer/requests/" + call["response_receipt"]["response_id"] + ".json"
            primary = audit.read(path, required=False)
            if primary:
                audit.check(finite(primary.get("timestamp")) and primary["timestamp"] >= frozen_time,
                            "test_response_before_freeze", call.get("call_id"))
    return {"heads": selected, "freeze_markers_checked": True,
            "limitation": "Journal order, declared zero-test markers and response timestamps support the freeze; response completion timestamps do not independently prove invocation-start time."}


def reconcile_critics(audit, study, critics, qualification):
    entries = [("study", item) for item in critics] + [("qualification", item) for item in (qualification or {}).get("critic_usage", [])]
    seen = set()
    for scope, call in entries:
        identity = call.get("call_id")
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity) or identity in seen:
            audit.issue("critic_identity", identity)
            continue
        seen.add(identity)
        physical = audit.read(study + "runtime/critic-requests/" + identity + ".json")
        if physical is None:
            continue
        audit.equal(physical.get("status"), call.get("status"), "critic_status", identity)
        audit.equal(physical.get("score"), call.get("score"), "critic_score", identity)
        if scope == "study":
            for key in ("stage", "question_id", "arm", "state_id", "branch_action"):
                if key in call:
                    audit.equal(physical.get("context", {}).get(key), call[key], "critic_context", f"{identity}:{key}")
        if call.get("status") == "completed":
            audit.check(finite(physical.get("score"), 0, 1), "critic_invalid_probability", identity)
    actual = {name for name in audit.evidence.files if name.startswith(study + "runtime/critic-requests/") and name.endswith(".json")}
    expected = {study + "runtime/critic-requests/" + identity + ".json" for identity in seen}
    audit.equal(actual, expected, "unmapped_critic_receipts", "Every critic receipt maps to study or qualification")
    return {"recorded_study_calls": len(critics), "qualification_calls": len(entries)-len(critics),
            "scope": "separate verifier/reranker attempts and time; no language-model response token-pricing"}


def audit_input(path, *, expected_source_commit=SOURCE_PIN, expected_plan_sha256=PLAN_PIN,
                expected_source_zip_sha256=SOURCE_ZIP_PIN, dataset_json=None, allow_heldout_scoring=False):
    if dataset_json is not None and not allow_heldout_scoring:
        raise ValueError("--allow-heldout-scoring is required before opening a dataset")
    evidence = Evidence(path)
    audit = Audit(evidence)
    report = {"input": str(Path(path).resolve()), "schema": "yash-independent-export-audit-v1",
              "scoring_scope": "independent aggregation of recorded metrics; raw-answer correctness not revalidated"}
    try:
        verify_manifest(audit)
        config_name = evidence.locate("study-config.json")
        storage_name = evidence.locate("notebook-storage.json")
        storage = audit.read(storage_name) if storage_name else None
        if storage:
            audit.equal(storage.get("source_commit"), expected_source_commit, "source_commit_pin", "notebook source")
            audit.equal(storage.get("plan_sha256"), expected_plan_sha256, "plan_pin", "notebook plan")
            expected_seed = {"primary": 0, "replication": 1}.get(storage.get("role"))
            audit.check(expected_seed is not None, "notebook_role_missing", "Predeclared primary/replication identity", "incomplete")
            audit.equal(storage.get("deployment_seed"), expected_seed, "notebook_seed_role", "predeclared seed")
        else:
            audit.issue("storage_provenance_missing", "No notebook source/plan identity", "incomplete")
        if config_name is None:
            qualification_name = evidence.locate("qualification.json")
            qualification = audit.read(qualification_name) if qualification_name else None
            audit.issue("study_not_started", "No study-config.json; no completed benchmark is claimed", "incomplete")
            report.update(qualification_status=(qualification or {}).get("status"), attempted_test_episodes=0, planned_test_episodes=140)
        else:
            study = config_name[:-len("study-config.json")]
            config = audit.read(config_name)
            if storage:
                audit.equal(config.get("deployment_seed"), storage.get("deployment_seed"), "notebook_config_seed", "deployment seed")
            source_values = verify_source(audit, config, expected_source_zip_sha256)
            plan = config["plan"]
            report["test_question_ids"] = plan["partitions"]["test"]["question_ids"]
            audit.equal(config.get("arms"), list(ARMS), "declared_arms", "seven frozen arms")
            for key, value in {"max_drafts": 3, "max_agent_calls": 10, "format_corrections": 1,
                    "primary_answer_selection": "last returned draft", "policy_config": {"epochs": 200, "learning_rate": .03, "l2": .001}}.items():
                audit.equal(config.get(key), value, "frozen_method_configuration", key)
            audit.check(config.get("deployment_seed") in (0, 1) and type(config.get("deployment_seed")) is int, "deployment_seed", "predeclared primary0/replica1")
            audit.equal({name: len(plan["partitions"][name]["question_ids"]) for name in ("train", "dev", "test")}, {"train": 20, "dev": 10, "test": 20}, "partition_counts", "frozen 20/10/20")
            all_ids = [identity for value in plan["partitions"].values() for identity in value["question_ids"]]
            audit.check(len(all_ids) == len(set(all_ids)) and not set(all_ids) & set(plan.get("excluded_ids", [])), "plan_overlap", "Question groups/exclusions")
            audit.equal(plan.get("dataset"), DATASET, "dataset_pin", "frozen dataset")
            plan_name = evidence.locate("Yash_Nov2_HotpotQA_Plan.json")
            if plan_name:
                audit.equal(sha(evidence.read(plan_name)), expected_plan_sha256, "plan_raw_hash", "frozen plan bytes")
                audit.equal(evidence.json(plan_name), plan, "plan_config_equality", "frozen plan vs config")
            else:
                audit.issue("frozen_plan_file_missing", "Config plan remains available", "incomplete")
            runtime = audit.read(study + "runtime/runtime.json")
            audit.check(isinstance(runtime, dict) and bool(runtime), "required_runtime_structure", "Nonempty runtime provenance is required")
            if isinstance(runtime, dict) and runtime:
                writer = runtime.get("writer", {})
                audit.equal((writer.get("model"), writer.get("revision")), WRITER, "writer_pin", "writer")
                audit.equal(writer.get("dtype"), "torch.float16", "writer_precision", "writer")
                for package, pin in PACKAGES.items():
                    audit.equal(runtime.get("packages", {}).get(package), pin, "package_pin", package)
                constraints = writer.get("format_constraints", {})
                audit.equal(constraints.get("schemas"), source_values.get("ROLE_JSON_SCHEMAS"), "constraint_schemas_vs_source", "actual shared schema")
                if constraints.get("schemas") is not None:
                    audit.equal(constraints.get("schemas_sha256"), sha(canonical(constraints["schemas"])), "constraint_schema_hash", "schema")
                audit.equal(constraints.get("version"), "0.11.3", "constraint_version", "LMFE")
                audit.equal(constraints.get("schema_version"), "constrained-json-object-citations-v2", "constraint_schema", "shared schema")
                audit.equal(constraints.get("unconstrained_fallback"), False, "constraint_fallback", "format filtering")
                audit.equal(constraints.get("effective_parser_config"), {"max_json_array_length": 20, "max_consecutive_whitespaces": 12, "force_json_field_order": False}, "constraint_config", "effective parser configuration")
            for key, pin in CRITIC_PINS.items():
                audit.equal(config.get("critic", {}).get(key), pin, "critic_pin", key)
            qualification = audit.read(study + "runtime/qualification.json")
            report["qualification_status"] = (qualification or {}).get("status")
            audit.equal(report["qualification_status"], "PASS", "benchmark_qualification_required_PASS", "Study configuration requires prior successful qualification")
            if qualification and qualification.get("status") == "PASS":
                roles = [item["role"] for item in qualification.get("role_usage", []) if item.get("status") == "completed"]
                audit.check(all(role in roles for role in ("writer", "reviewer", "challenger")), "qualification_roles", roles)
                history = qualification.get("after_invoke", {}).get("history", [])
                invoke = [step for step in history if step.get("action") == "INVOKE"]
                audit.check(bool(invoke and invoke[-1].get("challenger_response")), "qualification_challenger_response", "Actual third-agent response is required")
                if invoke:
                    challenge = invoke[-1].get("challenger_response")
                    completed = [item for item in qualification.get("role_usage", []) if item.get("status") == "completed"]
                    challenger_calls = [item for item in completed if item.get("role") == "challenger"]
                    audit.check(bool(challenger_calls), "qualification_raw_challenger_missing", "Actual challenger response required")
                    if challenger_calls:
                        physical_challenge = raw_call(audit, study, challenger_calls[-1])
                        if physical_challenge:
                            audit.equal(challenge, canonical_challenger(strict_json(physical_challenge["response_text"])), "qualification_challenger_raw_binding", "Recorded challenger must equal actual raw completed response")
                    writers = [item for item in qualification.get("role_usage", []) if item.get("role") == "writer"]
                    if len(writers) >= 2:
                        client = audit.read(study + "runtime/writer/by-client/" + writers[-1]["call_id"] + ".json")
                        physical = audit.read(study + "runtime/writer/" + client["primary_receipt"]) if client else None
                        text = message_text((physical or {}).get("request", {}).get("messages", []))
                        needle = canonical(challenge).decode()
                        audit.check(bool(challenge) and needle in text, "qualification_feedback_flow", "Actual canonical challenger response must reach subsequent writer request")
            episodes = audit.read(study + "episodes.jsonl", rows=True)
            calls = audit.read(study + "physical-calls.jsonl", rows=True)
            critics = audit.read(study + "critic-calls.jsonl", rows=True)
            traces = audit.read(study + "traces.jsonl", rows=True)
            summary = audit.read(study + "summary.json")
            audit.check(isinstance(summary, dict) and isinstance(summary.get("arms"), dict) and set(summary["arms"]) == set(ARMS), "required_summary_structure", "All seven arm summaries are required")
            ids = set(report["test_question_ids"])
            seen = set()
            valid = []
            gold = load_gold(dataset_json, plan, allow_heldout_scoring) if dataset_json is not None else None
            if gold is not None:
                report["scoring_scope"] = "raw-answer and supporting/joint metrics independently rescored against provided pinned digest-verified dataset"
            for episode in episodes:
                key = (episode.get("question_id"), episode.get("arm"))
                if key in seen or key[0] not in ids or key[1] not in ARMS:
                    audit.issue("episode_grid_duplicate_or_unplanned", key)
                    continue
                seen.add(key)
                if episode.get("status") not in {"completed", "failed", "interrupted"} or set(episode.get("metrics", {})) != set(METRICS) or not all(finite(value, 0, 1) for value in episode.get("metrics", {}).values()):
                    audit.issue("episode_invalid_fields", key)
                    continue
                count_keys = ("iterations", "attempted_agent_calls", "critic_calls", "known_recorded_tokens")
                if (any(type(episode.get(key)) is not int or episode[key] < 0 for key in count_keys)
                        or (episode.get("total_tokens") is not None and (type(episode["total_tokens"]) is not int or episode["total_tokens"] < 0))
                        or not isinstance(episode.get("agent_calls"), dict)
                        or any(role not in {"writer", "reviewer", "challenger"} or type(count) is not int or count < 0 for role, count in episode.get("agent_calls", {}).items())):
                    audit.issue("episode_invalid_count_types", key)
                    continue
                physical = [call for call in calls if call.get("stage") == "test" and (call.get("question_id"), call.get("arm")) == key]
                recorded_critics = [call for call in critics if call.get("stage") == "test" and (call.get("question_id"), call.get("arm")) == key]
                audit.equal(episode.get("attempted_agent_calls"), len(physical), "episode_physical_calls", key)
                audit.equal(episode.get("iterations"), sum(call.get("role") == "writer" for call in physical), "episode_writer_iterations", key)
                audit.equal(episode.get("critic_calls"), len(recorded_critics), "episode_critic_calls", key)
                audit.equal(episode.get("agent_calls"), dict(Counter(call["role"] for call in physical if call.get("status") == "completed")), "episode_completed_role_calls", key)
                known = sum(call["total_tokens"] for call in physical if call.get("total_tokens") is not None)
                tokens = known if all(call.get("total_tokens") is not None for call in physical) else None
                audit.equal(episode.get("known_recorded_tokens"), known, "episode_known_token_lower_bound", key)
                audit.equal(episode.get("total_tokens"), tokens, "episode_total_tokens", key)
                if episode["status"] != "completed":
                    audit.check(all(value == 0 for value in episode["metrics"].values()), "failed_episode_nonzero_score", key)
                else:
                    completed_path(audit, study, episode, physical, recorded_critics)
                    if gold is not None:
                        independently_scored = answer_scores(episode["prediction"], *gold[key[0]])
                        for metric in METRICS:
                            audit.equal(episode["metrics"][metric], independently_scored[metric], "raw_prediction_score", f"{key}:{metric}")
                if episode.get("termination") in {"budget_cap", "agent_budget_cap"}:
                    audit.equal(episode.get("gate_approved"), False, "cap_misreported_as_approval", key)
                valid.append(episode)
            report.update(planned_test_episodes=140, attempted_test_episodes=len(seen), metrics=aggregate(valid, ids), deployment_seed=config.get("deployment_seed"))
            unassigned = [call for call in calls if call.get("stage") == "test" and (call.get("question_id"), call.get("arm")) not in seen]
            if unassigned:
                audit.issue("physical_calls_without_episode_record", f"{len(unassigned)} recorded test calls belong to missing/pending episodes", "incomplete")
            report["physical_test_by_arm"] = {arm: {
                "attempted_calls": sum(call.get("stage") == "test" and call.get("arm") == arm for call in calls),
                "known_recorded_tokens": sum(call["total_tokens"] for call in calls if call.get("stage") == "test" and call.get("arm") == arm and call.get("total_tokens") is not None)} for arm in ARMS}
            report["failed_or_interrupted_test_episodes"] = sum(row["status"] != "completed" for row in valid)
            if len(seen) != 140:
                audit.issue("incomplete_test_grid", f"{len(seen)}/140 attempted", "incomplete")
            if isinstance(summary, dict) and summary:
                for arm, values in report["metrics"].items():
                    for key, value in values.items():
                        if key == "metrics":
                            for metric in METRICS:
                                audit.equal(summary.get("arms", {}).get(arm, {}).get("metrics", {}).get(metric), value[metric], "summary_metric", f"{arm}:{metric}")
                        else:
                            audit.equal(summary.get("arms", {}).get(arm, {}).get(key), value, "summary_resource_or_count", f"{arm}:{key}")
                report["reported_study_status"] = summary.get("study_status")
            audit.equal(sum(item.get("event") == "agent_invocation_started" for item in traces), len(calls), "started_agent_call_coverage", "traces vs physical ledger")
            audit.equal(sum(item.get("event") == "critic_invocation_started" for item in traces), len(critics), "started_critic_call_coverage", "traces vs critic ledger")
            audit.equal([row.get("invocation_index") for row in calls], list(range(len(calls))), "physical_order", "contiguous invocation indices")
            report["physical_receipts"] = reconcile_calls(audit, study, calls, qualification, source_values.get("ROLE_JSON_SCHEMAS", {}))
            report["critic_work"] = reconcile_critics(audit, study, critics, qualification)
            report["heads"] = audit_heads(audit, study, config, plan, calls)
        levels = Counter(item["level"] for item in audit.findings)
        failed = report.get("qualification_status") == "FAILED" or report.get("reported_study_status") == "failed"
        report["status"] = "EVIDENCE_MISMATCH" if levels["mismatch"] else "FAILED" if failed else "INCOMPLETE" if levels["incomplete"] else "VERIFIED_COMPLETE_WITH_FAILED_EPISODES" if report.get("failed_or_interrupted_test_episodes") else "VERIFIED_COMPLETE"
        report["findings"] = audit.findings
        report["limitations"] = ["A matching archive manifest provides internal integrity, not independent proof of execution or protection against coordinated rewriting of all evidence.",
            "Default answer EM/F1 aggregation trusts each recorded metric; independent raw-answer scoring needs the explicit post-freeze dataset opt-in.",
            "Each run has the same twenty test questions; seeds/arms are correlated repeats and are never pooled into forty independent questions.",
            "Known response-token sums exclude unknown/missing responses and verifier/reranker hardware work; no end-to-end dollar or total-compute claim."]
        return report
    except (ValueError, KeyError, TypeError, ZeroDivisionError, SyntaxError, RecursionError, zipfile.BadZipFile) as error:
        audit.issue("malformed_evidence_structure", f"{type(error).__name__}: {error}")
        return dict(report, status="EVIDENCE_MISMATCH", findings=audit.findings)
    finally:
        evidence.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", default=SOURCE_PIN)
    parser.add_argument("--expected-plan-sha256", default=PLAN_PIN)
    parser.add_argument("--expected-source-zip-sha256", default=SOURCE_ZIP_PIN)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--allow-heldout-scoring", action="store_true")
    args = parser.parse_args(argv)
    if args.dataset_json is not None and not args.allow_heldout_scoring:
        parser.error("Dataset reading is forbidden without explicit post-freeze --allow-heldout-scoring")
    reports = []
    for path in args.inputs:
        try:
            reports.append(audit_input(path, expected_source_commit=args.expected_source_commit,
                expected_plan_sha256=args.expected_plan_sha256, dataset_json=args.dataset_json,
                expected_source_zip_sha256=args.expected_source_zip_sha256,
                allow_heldout_scoring=args.allow_heldout_scoring))
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            reports.append({"input": str(path), "status": "EVIDENCE_MISMATCH", "findings": [{"level": "mismatch", "code": "unsafe_or_unreadable_input", "detail": str(error)}]})
    groups = [report.get("test_question_ids") for report in reports]
    result = {"schema": "yash-independent-export-audit-batch-v1", "audited_utc": datetime.now(timezone.utc).isoformat(),
        "runs": reports, "same_test_questions_across_runs": bool(groups and groups[0] and all(group == groups[0] for group in groups)),
        "independent_test_questions": len(set().union(*(set(group or []) for group in groups))),
        "pooled_forty_question_claim": False, "no_network_models_or_inference": True}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "statuses": [row["status"] for row in reports]}))
    return 0 if all(row["status"] == "VERIFIED_COMPLETE" for row in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
