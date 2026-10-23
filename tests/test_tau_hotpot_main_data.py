"""Synthetic corpus preparation only; no real Hotpot rows or model outcomes."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from haltiq_tau_hotpot import data
from pc_strands.data import question_key


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def row(identifier, kind="bridge"):
    return {"_id": identifier, "question": "Synthetic question " + identifier + "?", "type": kind,
        "level": "hard", "context": [["Title " + identifier, ["Sentence zero.", "Sentence one."]],
                                     ["Other " + identifier, ["Preserve this exact second passage."]]],
        "answer": "SYNTHETIC REFERENCE " + identifier,
        "supporting_facts": [["Title " + identifier, 1], ["Other " + identifier, 0]]}


class MainDataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.freeze_path = self.root / "qualification/data-freeze.json"
        self.corpus = self.root / "preflight-2026-10-18/hotpot_dev_distractor_v1.hf-rows-verified.json"
        self.partner = self.root / "partner-pilot-runtime/sources/haltiq_strands_experiments2.csv"
        self.earlier = self.root / "partner-pilot-runtime/new20-writer-prepared-v1/tasks.jsonl"
        self.history = self.root / "history.json"
        self.all_rows, files = [], {}
        for name, count in (("pilot_fit", 40), ("pilot_tune", 10), ("pilot_calibration", 10),
                            ("pilot_heldout", 4), ("main_fit", 400), ("main_tune", 100),
                            ("main_calibration", 100), ("main_fresh60", 60)):
            raw = [row(name + f"-{i:03}", "bridge" if i % 2 == 0 else "comparison") for i in range(count)]
            self.all_rows.extend(raw)
            partition = "engineering_heldout" if name == "pilot_heldout" else name.split('_', 1)[1]
            cases = [data.case_from_row(r, partition) for r in raw]
            path = self.root / "qualification" / (name + ".json")
            write(path, cases)
            files[name] = {"path": str(path), "sha256": data.digest(path), "count": count,
                "type_counts": {"bridge": count // 2, "comparison": count // 2}}
        self.original20 = [row("earlier-" + str(i), "bridge" if i % 2 == 0 else "comparison") for i in range(20)]
        self.all_rows.extend(self.original20)
        write(self.earlier, [{"source_row": r} for r in self.original20])
        historical = row("historical")
        partner = row("partner", "comparison")
        write(self.history, [{"source_row": historical}])
        self.partner.parent.mkdir(parents=True, exist_ok=True)
        with self.partner.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'question'])
            writer.writeheader()
            writer.writerow({'id': partner['_id'], 'question': partner['question']})
        # Different IDs with equal normalized questions must also be excluded.
        shadow = row('historical-shadow')
        shadow['question'] = historical['question'].upper() + '   '
        shadow2 = row('partner-shadow', 'comparison')
        shadow2['question'] = partner['question']
        self.all_rows.extend([historical, partner, shadow, shadow2])
        self.candidates = [row('new-' + str(i), 'bridge' if i % 2 == 0 else 'comparison') for i in range(40)]
        self.all_rows.extend(self.candidates)
        write(self.corpus, self.all_rows)
        self.freeze = {"schema": "tau2-hotpot-qualification-v1", "dataset_sha256": data.digest(self.corpus),
            "original20_sha256": data.digest(self.earlier), "files": files,
            "exposure_sources": [{"path": str(self.history), "sha256": data.digest(self.history)}]}
        write(self.freeze_path, self.freeze)
        self.known_dataset, self.known_partner = data.digest(self.corpus), data.digest(self.partner)

    def prepare(self, output="main"):
        with patch.object(data, 'DATASET_SHA256', self.known_dataset), patch.object(data, 'PARTNER_CSV_SHA256', self.known_partner):
            return data.prepare_main(self.root, self.freeze_path, self.root / output)

    def test_balanced_nested_disjoint_cohorts_preserve_all_source_fields_and_sentence_ids(self):
        result = self.prepare()
        self.assertEqual(result['status'], 'prepared_not_started')
        self.assertEqual(result['main']['hours'], 14)
        self.assertIsNone(result['main']['selected_fresh_n'])
        self.assertIsNone(result['main']['start_time'])
        groups = {name: data.records(spec['path']) for name, spec in result['files'].items()}
        expected_counts = {'main_fit': 400, 'main_tune': 100, 'main_calibration': 100,
            'main_development': 20, 'main_fresh20': 20, 'main_fresh40': 40, 'main_fresh60': 60}
        originals = {r['_id']: r for r in self.all_rows}
        for name, cases in groups.items():
            self.assertEqual(len(cases), expected_counts[name])
            self.assertEqual(result['files'][name]['sha256'], data.digest(result['files'][name]['path']))
            self.assertEqual(result['files'][name]['type_counts'], {'bridge': len(cases) // 2, 'comparison': len(cases) // 2})
            for case in cases:
                self.assertEqual(case, data.case_from_row(originals[case['public']['id']], case['partition']))
                self.assertEqual(len(case['public']['context']), 2)
                self.assertEqual(case['evaluator']['supporting_facts'][0][1], 1)
        ids = lambda name: {c['public']['id'] for c in groups[name]}
        self.assertLess(ids('main_fresh20'), ids('main_fresh40'))
        self.assertLess(ids('main_fresh40'), ids('main_fresh60'))
        union_ids, union_keys = set(), set()
        for name in ('main_fit', 'main_tune', 'main_calibration', 'main_development', 'main_fresh60'):
            keys = {question_key(c['public']['question']) for c in groups[name]}
            self.assertFalse(union_ids & ids(name))
            self.assertFalse(union_keys & keys)
            union_ids.update(ids(name)); union_keys.update(keys)
        self.assertTrue(ids('main_development') <= {r['_id'] for r in self.candidates})

    def test_selection_is_stable_under_corpus_order_and_reference_changes(self):
        first = self.prepare('first')
        initial_ids = [c['public']['id'] for c in data.records(first['files']['main_development']['path'])]
        changed = deepcopy(list(reversed(self.all_rows)))
        for item in changed:
            item['answer'] = 'DIFFERENT SYNTHETIC REFERENCE'
        write(self.corpus, changed)
        self.known_dataset = data.digest(self.corpus)
        self.freeze['dataset_sha256'] = self.known_dataset
        write(self.freeze_path, self.freeze)
        second = self.prepare('second')
        self.assertEqual(initial_ids, [c['public']['id'] for c in data.records(second['files']['main_development']['path'])])

    def test_changed_original20_is_rejected_before_output_creation(self):
        write(self.earlier, [{'source_row': self.original20[0]}])
        with self.assertRaisesRegex(ValueError, 'earlier20'):
            self.prepare()
        self.assertFalse((self.root / 'main').exists())

    def test_forged_freeze_corpus_digest_and_changed_actual_corpus_are_rejected(self):
        original = self.corpus.read_bytes()
        self.freeze['dataset_sha256'] = '0' * 64
        write(self.freeze_path, self.freeze)
        with self.assertRaisesRegex(ValueError, 'corpus'):
            self.prepare('forged')
        self.assertFalse((self.root / 'forged').exists())
        self.freeze['dataset_sha256'] = self.known_dataset
        write(self.freeze_path, self.freeze)
        self.corpus.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'corpus'):
            self.prepare('changed')
        self.assertFalse((self.root / 'changed').exists())

    def test_changed_history_or_prepared_case_cannot_make_a_new_cohort(self):
        self.history.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'exposure'):
            self.prepare('history')
        self.assertFalse((self.root / 'history').exists())
        write(self.history, [{'source_row': row('historical')}])
        path = Path(self.freeze['files']['main_fit']['path'])
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Prepared'):
            self.prepare('case')
        self.assertFalse((self.root / 'case').exists())


if __name__ == '__main__':
    unittest.main()
