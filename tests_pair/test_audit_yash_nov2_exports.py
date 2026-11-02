"""Independent export-auditor corruption tests; every fixture is synthetic."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from tools import audit_yash_nov2_exports as audit


def encoded(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode()


class Fixture:
    def __init__(self):
        self.files, self.counter = {}, 0
        ids = [f"{i:024x}" for i in range(1, 51)]
        self.plan = {"schema": "yash-nov2-plan-v1", "dataset": audit.DATASET, "seed": 20261102,
            "excluded_ids": [], "partitions": {"train": {"question_ids": ids[:20]}, "dev": {"question_ids": ids[20:30]}, "test": {"question_ids": ids[30:]}}}
        self.dataset = {"dataset": audit.DATASET, "rows": []}
        for identity in ids:
            self.dataset["rows"].append({"id": identity, "question": "Synthetic question", "context": {"title": ["T"], "sentences": [["Synthetic sentence"]]},
                "answer": "Alpha", "supporting_facts": {"title": ["T"], "sent_id": [0]}})
        for partition in self.plan["partitions"].values():
            subset = partition["question_ids"]
            partition["public_inputs_sha256"] = audit.sha(audit.canonical([{"id": identity, "question": "Synthetic question", "passages": [["T", ["Synthetic sentence"]]]} for identity in subset]))
            partition["label_digest"] = audit.sha(audit.canonical([{"answer": "Alpha", "supporting_facts": [["T", 0]]} for identity in subset]))
        self.plan_raw = encoded(self.plan)
        citation = {"type": "object", "properties": {"title": {"type": "string"}, "sentence_index": {"type": "integer", "minimum": 0}}, "required": ["title", "sentence_index"]}
        self.schemas = {
            "writer": {"type": "object", "properties": {"answer": {"type": "string"}, "supporting_facts": {"type": "array", "items": citation}}, "required": ["answer", "supporting_facts"]},
            "reviewer": {"type": "object", "properties": {"approved": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["approved", "feedback"]},
            "challenger": {"type": "object", "properties": {"alternative": {"type": "string"}, "evidence": {"type": "array", "items": citation}, "feedback": {"type": "string"}}, "required": ["alternative", "evidence", "feedback"]}}
        source = ("ROLE_FORMAT_VERSION='constrained-json-object-citations-v2'\n_CITATION_SCHEMA={}\nROLE_JSON_SCHEMAS=" + repr(self.schemas) + "\n").encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("haltiq_pair/nov2_orchestration.py", source)
        self.source_raw = buffer.getvalue()
        self.put("frozen/Yash_Nov2_HotpotQA_Plan.json", self.plan_raw)
        self.put("frozen/Yash-Nov2-Source.zip", self.source_raw)
        self.config = {"plan": self.plan, "arms": list(audit.ARMS), "deployment_seed": 0,
            "max_drafts": 3, "max_agent_calls": 10, "format_corrections": 1,
            "primary_answer_selection": "last returned draft", "policy_config": {"epochs": 200, "learning_rate": .03, "l2": .001},
            "source_sha256": {"nov2_orchestration.py": audit.sha(source)}, "critic": audit.CRITIC_PINS}
        self.json("notebook-storage.json", {"source_commit": audit.SOURCE_PIN, "plan_sha256": audit.sha(self.plan_raw), "role": "primary", "deployment_seed": 0})
        self.json("study/study-config.json", self.config)
        self.json("study/runtime/runtime.json", {"writer": {"model": audit.WRITER[0], "revision": audit.WRITER[1], "dtype": "torch.float16",
            "format_constraints": {"schemas": self.schemas, "schemas_sha256": audit.sha(audit.canonical(self.schemas)), "version": "0.11.3", "schema_version": "constrained-json-object-citations-v2",
                "unconstrained_fallback": False, "effective_parser_config": {"max_json_array_length": 20, "max_consecutive_whitespaces": 12, "force_json_field_order": False}}}, "packages": audit.PACKAGES})
        challenge = {"alternative": "A", "evidence": [["T", 0]], "feedback": "ACTUAL_SYNTHETIC_FEEDBACK"}
        qualification = [self.call(role, messages=encoded({"challenger_analysis": challenge}).decode()) for role in ("writer", "reviewer", "challenger", "writer", "reviewer")]
        qual_critics = [self.critic({"phase": "engineering_qualification"}) for _ in range(2)]
        self.json("study/runtime/qualification.json", {"status": "PASS", "role_usage": qualification, "critic_usage": qual_critics,
            "after_invoke": {"history": [{"action": "INVOKE", "challenger_response": challenge}]}})
        self.calls, self.critics, self.episodes, self.traces = [], [], [], []
        for identity in ids[30:]:
            for arm in audit.ARMS:
                rounds = 3 if arm in {"always_revise3", "always_invoke"} else 1
                path_calls, path_critics, history = [], [], []
                prediction = {"answer": "Alpha", "supporting_facts": [["T", 0]]}
                for draft in range(rounds):
                    roles = (["challenger"] if arm == "always_invoke" and draft == 1 else []) + ["writer"] + ([] if arm == "fixed1" else ["reviewer"])
                    for role in roles:
                        call = self.call(role)
                        call.update(stage="test", question_id=identity, arm=arm, invocation_index=len(self.calls))
                        self.calls.append(call);path_calls.append(call)
                        self.traces.append({"event": "agent_invocation_started", "stage": "test", "question_id": identity, "arm": arm})
                    if arm != "fixed1":
                        critic=self.critic({"stage":"test","question_id":identity,"arm":arm})
                        critic.update(stage="test",question_id=identity,arm=arm,critic_index=len(self.critics))
                        self.critics.append(critic);path_critics.append(critic)
                        self.traces.append({"event":"critic_invocation_started"})
                    history.append({"draft":prediction,"action":"INITIAL" if draft==0 else "INVOKE" if arm=="always_invoke" and draft==1 else "REVISE"})
                metrics = audit.answer_scores({"answer": "Alpha", "supporting_facts": [["T", 0]]}, "Alpha", [["T", 0]])
                counts={role:sum(call["role"]==role for call in path_calls) for role in set(call["role"] for call in path_calls)}
                self.episodes.append({"question_id": identity, "arm": arm, "status": "completed", "prediction": prediction,
                    "metrics": metrics, "iterations": rounds, "returned_draft_number":rounds,"agent_calls":counts,"attempted_agent_calls":len(path_calls),
                    "total_tokens":13*len(path_calls),"known_recorded_tokens":13*len(path_calls),"journal_complete":True,"critic_calls":len(path_critics),
                    "actions":["INVOKE","REVISE"] if arm=="always_invoke" else ["REVISE","REVISE"] if rounds==3 else ["STOP"],
                    "state":None if arm in {"fixed1","legacy_combined6"} else {"draft":prediction,"draft_number":rounds,"history":history,"usage":path_calls},
                    "legacy_trajectory":[{"draft":prediction}] if arm=="legacy_combined6" else None,
                    "termination":"budget_cap" if rounds==3 else "policy_stop","gate_approved":rounds!=3})
        train_ids, dev_ids = ids[:20], ids[20:30]
        self.rows("study/training-examples.jsonl", [{"split": split, "question_id": identity} for split, names in (("train", train_ids), ("dev", dev_ids)) for identity in names])
        self.json("study/heads/protocol.json", {"train_ids": train_ids, "dev_ids": dev_ids,"configuration":self.config["policy_config"]})
        entries = []
        for seed in range(5):
            weights, bias = [[0.] * 11 for _ in range(3)], [0.] * 3
            payload = {"schema": "yash-nov2-three-action-linear-v1", "objective": audit.OBJECTIVE, "features": list(audit.FEATURES), "actions": ["STOP", "REVISE", "INVOKE"],
                "seed": seed, "weights": weights, "bias": bias, "metadata": {"train_ids": train_ids, "dev_ids": dev_ids, "selected_epoch": 1}}
            payload["payload_sha256"] = audit.sha(audit.canonical(payload, compact=True))
            path = f"heads/seed{seed}/head.json"
            self.json("study/" + path, payload)
            entries.append({"seed": seed, "path": path, "sha256": audit.sha(self.files["study/" + path])})
            self.json(f"study/heads/seed{seed}/history.json", {"optimizer_steps_completed": 200,
                "epochs": [{"epoch": index+1, "dev_greedy_utility_after_step": 1., "weights": weights, "bias": bias} for index in range(200)]})
        self.json("study/heads/training-freeze.json", {"protocol_sha256": audit.sha(self.files["study/heads/protocol.json"]), "frozen_utc": "2026-11-02T07:00:00+00:00",
            "all_selected_checkpoints_frozen_before_test_scoring": True, "test_used_to_fit_or_select": False,
            "checkpoints": [{**entry, "path": entry["path"].removeprefix("heads/")} for entry in entries]})
        self.json("study/freeze.json", {"frozen_utc": "2026-11-02T07:00:01+00:00", "config_sha256": audit.sha(self.files["study/study-config.json"]),
            "checkpoints": entries, "test_calls_at_freeze": 0, "physical_collection_invocations": 0, "deployment_seed": 0})
        self.refresh()

    def put(self, path, raw):
        self.files[path] = raw

    def json(self, path, value):
        self.put(path, encoded(value))

    def rows(self, path, rows):
        self.put(path, b"".join(encoded(row) for row in rows))

    def call(self, role, messages="public"):
        self.counter += 1
        identity, response = f"{self.counter:032x}", "chatcmpl-" + f"{self.counter:032x}"
        usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        metadata = {"yash_call_id": identity, "yash_role": role}
        outputs = {"writer": {"answer": "Alpha", "supporting_facts": [{"title":"T","sentence_index":0}]}, "reviewer": {"approved": True, "feedback": "public"}, "challenger": {"alternative": "A", "evidence": [{"title":"T","sentence_index":0}], "feedback": "ACTUAL_SYNTHETIC_FEEDBACK"}}
        raw_output = json.dumps(outputs[role])
        physical = {"id": response, "usage": usage, "timestamp": 1793606400., "finish_reason": "stop", "response_text": raw_output, "request": {"metadata": metadata, "messages": messages}}
        primary = "runtime/writer/requests/" + response + ".json"
        self.json("study/" + primary, physical)
        client = {"call_id": identity, "role": role, "response_id": response, "metadata": metadata, "usage": usage, "finish_reason": "stop",
            "primary_sha256": audit.sha(self.files["study/" + primary]), "primary_receipt": "requests/"+response+".json"}
        path = "runtime/writer/by-client/" + identity + ".json"
        self.json("study/" + path, client)
        self.json("study/runtime/writer/format-checks/"+identity+".json", {"status": "validated", "raw_response_sha256": audit.sha(raw_output.encode())})
        return {"call_id": identity, "role": role, "status": "completed", "total_tokens": 13,
                "response_receipt": {"response_id": response, "primary_receipt_sha256": audit.sha(self.files["study/"+primary]), "client_receipt_sha256": audit.sha(self.files["study/"+path])}}

    def critic(self, context):
        self.counter+=1
        identity=f"{self.counter:032x}"
        result={"call_id":identity,"context":context,"status":"completed","score":.9}
        self.json("study/runtime/critic-requests/"+identity+".json",result)
        return result

    def refresh(self):
        self.rows("study/episodes.jsonl", self.episodes)
        self.rows("study/physical-calls.jsonl", self.calls)
        self.rows("study/critic-calls.jsonl", self.critics)
        self.rows("study/traces.jsonl", self.traces)
        self.json("study/summary.json", {"study_status": "completed", "arms": audit.aggregate(self.episodes, set(self.plan["partitions"]["test"]["question_ids"]))})

    def write(self, path, *, zipped=False):
        files = dict(self.files)
        files["Yash-Archive-Manifest.json"] = encoded({"files": [{"path": name, "bytes": len(raw), "sha256": audit.sha(raw)} for name, raw in files.items()]})
        if zipped:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, raw in files.items():
                    archive.writestr(name, raw)
        else:
            path.mkdir()
            for name, raw in files.items():
                target = path/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)

    def run(self, path, **kwargs):
        return audit.audit_input(path, expected_plan_sha256=audit.sha(self.plan_raw), expected_source_zip_sha256=audit.sha(self.source_raw), **kwargs)


class AuditorTests(unittest.TestCase):
    def test_complete_zip_and_directory_recompute_without_network_or_models(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            for zipped in (False, True):
                path = Path(tmp)/("evidence.zip" if zipped else "directory")
                fixture.write(path, zipped=zipped)
                result = fixture.run(path)
                self.assertEqual(result["status"], "VERIFIED_COMPLETE", result["findings"][:5])
                self.assertEqual(result["attempted_test_episodes"], 140)
                self.assertEqual(result["metrics"]["learned3"]["metrics"]["f1"], 1.)

    def test_member_tampering_and_duplicate_grid_are_detected(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"directory"
            fixture.write(path)
            (path/"study/summary.json").write_bytes(b"{}")
            result = fixture.run(path)
            self.assertIn("archive_member_hash", {row["code"] for row in result["findings"]})
        fixture.episodes.append(deepcopy(fixture.episodes[0]))
        fixture.refresh()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"duplicate"
            fixture.write(path)
            result=fixture.run(path)
            self.assertIn("episode_grid_duplicate_or_unplanned", {row["code"] for row in result["findings"]})

    def test_pending_grid_keeps_denominator_and_unknown_tokens(self):
        fixture=Fixture()
        removed=fixture.episodes.pop()
        fixture.refresh()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"partial"
            fixture.write(path)
            result=fixture.run(path)
            arm=result["metrics"][removed["arm"]]
            self.assertEqual(arm["metrics"]["f1"], 19/20)
            self.assertEqual(arm["pending"], 1)
            self.assertIsNone(arm["total_tokens"])
            self.assertEqual(arm["known_recorded_tokens"],19*26)

    def test_head_tamper_is_detected_even_after_refreshing_archive_manifest(self):
        fixture=Fixture()
        fixture.files["study/heads/seed0/head.json"] += b" "
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"tampered"
            fixture.write(path)
            result=fixture.run(path)
            self.assertIn("frozen_head_hash", {row["code"] for row in result["findings"]})

    def test_partial_failed_episode_keeps_known_subset_and_missing_receipt_unknown(self):
        fixture=Fixture()
        episode=next(row for row in fixture.episodes if row["arm"]=="learned3")
        call=fixture.call("reviewer")
        call.update(stage="test",question_id=episode["question_id"],arm=episode["arm"],invocation_index=len(fixture.calls),status="failed",total_tokens=None,response_receipt={"status":"missing"})
        fixture.calls.append(call)
        fixture.traces.append({"event":"agent_invocation_started"})
        del fixture.files["study/runtime/writer/by-client/"+call["call_id"]+".json"]
        episode.update(status="failed",metrics={name:0. for name in audit.METRICS},attempted_agent_calls=3,total_tokens=None,journal_complete=False,termination="error",gate_approved=False)
        fixture.refresh()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"unknown"
            fixture.write(path)
            result=fixture.run(path)
            self.assertEqual(result["status"],"INCOMPLETE",result["findings"])
            self.assertEqual(result["metrics"]["learned3"]["known_recorded_tokens"],20*26)
            self.assertIsNone(result["metrics"]["learned3"]["total_tokens"])
            self.assertEqual(result["metrics"]["learned3"]["metrics"]["f1"],19/20)

    def test_completed_grid_with_failed_episodes_is_distinguished_from_success(self):
        fixture=Fixture()
        fixture.episodes[0].update(status="failed",metrics={name:0. for name in audit.METRICS},termination="error",gate_approved=False)
        fixture.refresh()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"failure"
            fixture.write(path)
            result=fixture.run(path)
            self.assertEqual(result["status"],"VERIFIED_COMPLETE_WITH_FAILED_EPISODES",result["findings"])
            self.assertEqual(result["failed_or_interrupted_test_episodes"],1)

    def test_challenger_state_and_primary_prediction_cannot_replace_actual_raw_outputs(self):
        for mutation,expected_code in (("challenge","qualification_challenger_raw_binding"),("prediction","primary_prediction_vs_raw_last_writer")):
            with self.subTest(mutation=mutation):
                fixture=Fixture()
                if mutation=="challenge":
                    qualification=json.loads(fixture.files["study/runtime/qualification.json"])
                    qualification["after_invoke"]["history"][0]["challenger_response"]["feedback"]="FORGED_STATE_FEEDBACK"
                    fixture.json("study/runtime/qualification.json",qualification)
                else:
                    fixture.episodes[0]["prediction"]={"answer":"Alpha","supporting_facts":[]}
                    fixture.refresh()
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/"forged"
                    fixture.write(path)
                    result=fixture.run(path)
                    self.assertIn(expected_code,{row["code"] for row in result["findings"]})

    def test_empty_required_documents_and_nonfinite_exponent_do_not_pass(self):
        for name in ("study/runtime/runtime.json","study/summary.json","study/heads/seed0/history.json"):
            with self.subTest(name=name):
                fixture=Fixture();fixture.json(name,{})
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/"empty";fixture.write(path)
                    self.assertEqual(fixture.run(path)["status"],"EVIDENCE_MISMATCH")
        fixture=Fixture()
        path="study/heads/seed0/history.json"
        fixture.files[path]=fixture.files[path].replace(b'"dev_greedy_utility_after_step": 1.0',b'"dev_greedy_utility_after_step": 1e309',1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"overflow";fixture.write(path)
            self.assertEqual(fixture.run(path)["status"],"EVIDENCE_MISMATCH")
        with self.assertRaises(ValueError):audit.strict_json(b'{"value":1e309}')

    def test_completed_nonbaseline_requires_actual_review_and_verifier_path(self):
        fixture=Fixture()
        episode=next(row for row in fixture.episodes if row["arm"]=="legacy_combined6")
        matches=lambda row:(row.get("question_id"),row.get("arm"))==(episode["question_id"],episode["arm"])
        removed=[row for row in fixture.calls if matches(row) and row["role"]=="reviewer"]
        fixture.calls=[row for row in fixture.calls if row not in removed]
        removed_critics=[row for row in fixture.critics if matches(row)]
        fixture.critics=[row for row in fixture.critics if not matches(row)]
        for row in removed:
            client=json.loads(fixture.files.pop("study/runtime/writer/by-client/"+row["call_id"]+".json"))
            fixture.files.pop("study/runtime/writer/"+client["primary_receipt"])
            fixture.files.pop("study/runtime/writer/format-checks/"+row["call_id"]+".json")
        for row in removed_critics:fixture.files.pop("study/runtime/critic-requests/"+row["call_id"]+".json")
        for index,row in enumerate(fixture.calls):row["invocation_index"]=index
        for index,row in enumerate(fixture.critics):row["critic_index"]=index
        for event in ("agent_invocation_started","critic_invocation_started"):
            fixture.traces.pop(next(index for index,row in enumerate(fixture.traces) if row["event"]==event))
        episode.update(agent_calls={"writer":1},attempted_agent_calls=1,total_tokens=13,known_recorded_tokens=13,critic_calls=0)
        fixture.refresh()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"roleless";fixture.write(path)
            result=fixture.run(path)
            self.assertIn("legacy_role_path",{row["code"] for row in result["findings"]})

    def test_zip_directory_count_and_streamed_actual_byte_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"many-directories.zip"
            with zipfile.ZipFile(path,"w") as archive:
                for name in ("one/","two/","three/"):archive.writestr(name,b"")
            with patch.object(audit,"MAX_MEMBERS",2):
                with self.assertRaises(ValueError):audit.Evidence(path)
        with self.assertRaises(ValueError):audit.bounded_read(io.BytesIO(b"x"*2048),1024)

    def test_unsafe_and_duplicate_zip_names_are_refused_without_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            for names in (("../escape",), ("same", "same")):
                path=Path(tmp)/"unsafe.zip"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with zipfile.ZipFile(path,"w") as archive:
                        for name in names:archive.writestr(name,b"content")
                with self.assertRaises(ValueError):audit.Evidence(path)
            self.assertFalse((Path(tmp).parent/"escape").exists())

    def test_binary_lf_journal_parser_preserves_unicode_line_separator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"rows.jsonl"
            path.write_bytes(encoded({"feedback":"before\u2028after"}))
            evidence=audit.Evidence(tmp)
            self.assertEqual(evidence.rows("rows.jsonl"),[{"feedback":"before\u2028after"}])
            evidence.close()

    def test_gold_guard_precedes_any_path_read_and_synthetic_opt_in_checks_digests(self):
        with patch.object(Path,"read_bytes",side_effect=AssertionError("Must not read heldout")):
            with self.assertRaises(ValueError):audit.load_gold("protected-dataset.json",{},False)
            with self.assertRaises(ValueError):audit.audit_input("unused",dataset_json="protected-dataset.json")
        fixture=Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp)/"synthetic-dataset.json"
            dataset.write_bytes(encoded(fixture.dataset))
            gold=audit.load_gold(dataset,fixture.plan,True)
            self.assertEqual(len(gold),50)
            changed=deepcopy(fixture.dataset);changed["rows"][0]["answer"]="changed"
            dataset.write_bytes(encoded(changed))
            with self.assertRaises(ValueError):audit.load_gold(dataset,fixture.plan,True)

    def test_independent_official_scorer_special_answers_and_joint_formula(self):
        value=audit.answer_scores({"answer":"The ALPHA beta!","supporting_facts":[["T",0],["T",1]]},"alpha gamma delta",[["T",0],["T",2],["T",3]])
        self.assertAlmostEqual(value["f1"],.4)
        self.assertAlmostEqual(value["joint_f1"],2/13)
        self.assertEqual(audit.answer_scores({"answer":"yes other","supporting_facts":[]},"yes",[])["f1"],0.)


if __name__=="__main__":unittest.main()
