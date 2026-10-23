"""Owned question-process contracts with synthetic responses; no model work."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pc_strands.control import ResourcePressure
from haltiq_tau_hotpot.journal import CallJournal, durable_write, fingerprint
from haltiq_tau_hotpot.main_question import QuestionRuntime, admission
from haltiq_tau_hotpot.runtime import LocalRuntime
from haltiq_tau_hotpot.hobson import file_hash


class Control:
    def __init__(self, directory):
        self.directory = directory
        self.deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        self.active_seconds = self.idle_seconds = 0.
    def _check_stop(self): pass
    def before_work(self, **_): pass
    def rest_after_block(self, _): pass


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.control = Control(self.root / 'control')
        self.config = {'gpu_lease': str(self.root / 'gpu.lock'), 'writer_digest': 'synthetic-writer', 'response_cache': True}

    def test_import_of_parent_and_child_has_no_model_or_tau_initialization(self):
        code = 'import sys;import haltiq_tau_hotpot.main_worker,haltiq_tau_hotpot.main_question;assert not any(x in sys.modules for x in ("torch","transformers","tau2"))'
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_venv_redirector_identity_and_creation_are_verified_without_models(self):
        from haltiq_tau_hotpot.main_question import validate_process_binding, exited_process_binding
        at = datetime.now(timezone.utc)
        dispatched = {'at': at.isoformat(), 'compare_parent_pid': 100, 'run_id': 'synthetic',
            'plan_sha256': 'a' * 64, 'question_index': 0}
        started = {**dispatched, 'at': (at + timedelta(seconds=2)).isoformat(), 'pid': 300, 'parent_pid': 200}
        launcher = {'pid': 200, 'parent_pid': 100, 'created_at': (at + timedelta(seconds=.1)).isoformat(),
            'executable': sys.executable, 'active': True}
        worker = {**launcher, 'pid': 300, 'parent_pid': 200, 'created_at': (at + timedelta(seconds=.2)).isoformat()}
        parent = {**launcher, 'pid': 100, 'created_at': (at - timedelta(seconds=20)).isoformat()}
        args = dict(launcher_pid=200, parent_pid=100, executables=[sys.executable], compare_parent=parent)
        observed = (at + timedelta(seconds=3)).isoformat()
        validate_process_binding(launcher, worker, started, dispatched, observed, **args)
        for alteration in ({'parent_pid': 999}, {'created_at': (at - timedelta(seconds=1)).isoformat()},
                           {'executable': str(self.root / 'foreign.exe')}, {'active': False}):
            with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, 'ancestry'):
                validate_process_binding(launcher, {**worker, **alteration}, started, dispatched, observed, **args)
        reused = {**worker, 'created_at': (at + timedelta(seconds=20)).isoformat()}
        with patch('haltiq_tau_hotpot.main_question.process_identity', return_value=reused):
            proof = exited_process_binding({'launcher': launcher, 'worker': worker})
        self.assertTrue(proof['worker']['original_identity_absent'])
        with patch('haltiq_tau_hotpot.main_question.process_identity', side_effect=[launcher, worker]), \
                self.assertRaisesRegex(RuntimeError, 'remains'):
            exited_process_binding({'launcher': launcher, 'worker': worker})

    def test_handshake_signal_is_atomic_after_fsync_and_never_overwrites(self):
        from haltiq_tau_hotpot import main_question
        path = self.root / 'identity.json'
        actual_link = main_question.os.link
        def link(source, target):
            self.assertFalse(target.exists())
            self.assertEqual(json.loads(source.read_text()), {'fixed': 'synthetic identity only'})
            actual_link(source, target)
        with patch('haltiq_tau_hotpot.main_question.os.link', side_effect=link):
            main_question.publish_signal(path, {'fixed': 'synthetic identity only'})
        self.assertFalse(path.with_name(path.name + '.publishing').exists())
        with self.assertRaises(FileExistsError):
            main_question.publish_signal(path, {'must': 'never replace the approval'})
        self.assertEqual(json.loads(path.read_text()), {'fixed': 'synthetic identity only'})

    def test_placement_envelope_uses_only_this_question_and_blocks_eviction_before_reload(self):
        directory = self.root / 'compare'
        old = CallJournal(directory / 'physical')
        old.call('writer_load', {}, lambda: {})
        old.call('native_startup', {}, lambda: {})
        runtime = QuestionRuntime(self.config, directory, self.control)
        def place_writer(instance):
            instance.journal.call('writer_load', {}, lambda: {})
            instance.writer_loaded = True
        with patch.object(LocalRuntime, '_writer_ready', place_writer):
            runtime._writer_ready()  # Prior question's native placement is irrelevant.
            self.assertEqual(runtime.roles(), ['writer_load'])
            runtime._http = Mock(return_value={'models': [{'digest': 'synthetic-writer'}]})
            runtime._writer_ready()
            self.assertEqual(runtime.roles(), ['writer_load'])
            runtime._http.return_value = {'models': []}
            with self.assertRaisesRegex(ResourcePressure, 'evicted'):
                runtime._writer_ready()
            self.assertEqual(runtime.roles(), ['writer_load'])
        runtime.journal.call('native_startup', {}, lambda: {})
        with self.assertRaisesRegex(ResourcePressure, 'order is exhausted'):
            runtime._writer_ready()
        runtime.backend = None
        with patch.object(LocalRuntime, '_native_ready') as native, self.assertRaisesRegex(ResourcePressure, 'never reload'):
            runtime._native_ready()
        native.assert_not_called()

    def test_known_failed_response_reuse_after_native_placement_needs_no_writer_reload(self):
        runtime = QuestionRuntime(self.config, self.root / 'compare', self.control)
        runtime._http = Mock(return_value={'done': True, 'message': {'content': '[]'}})
        with patch.object(runtime, '_writer_ready') as ready:
            with self.assertRaisesRegex(ValueError, 'schema'):
                runtime.generate({'question': 'Synthetic fixture?'})
            runtime.journal.call('native_startup', {}, lambda: {})
            with self.assertRaisesRegex(ValueError, 'schema'):
                runtime.generate({'question': 'Synthetic fixture?'})
        ready.assert_called_once()
        self.assertEqual(runtime.journal.verify()['by_role'], {'writer': 1, 'native_startup': 1})
        self.assertEqual(runtime.journal.summary()['exact_response_reuses'], 1)

    def sample_admission(self, ram_values):
        clock, holds, observations = [datetime.now(timezone.utc)], set(), []
        values = iter(ram_values)
        def sleep(seconds):
            clock[0] += timedelta(seconds=seconds)
            self.control.idle_seconds += seconds
        class Lease:
            def __init__(_, *args): pass
            def __enter__(_):
                self.assertFalse(holds)
                holds.add('GPU')
            def __exit__(_, *args): holds.clear()
        def sample():
            self.assertEqual(holds, {'GPU'})
            value = next(values)
            observations.append(value)
            return {'available_ram_bytes': value * 1024**3 if value is not None else None,
                    'gpu_free_mib': 9.1 * 1024, 'gpu_temperature_c': 40}
        output = self.root / 'admission.json'
        with patch('haltiq_tau_hotpot.main_question.port_open', return_value=False), \
                patch('subprocess.Popen') as process:
            try:
                result = admission(self.config, self.control, output, sampler=sample,
                    sleep=sleep, clock=lambda: clock[0], lease_factory=Lease)
            finally:
                process.assert_not_called()
                self.assertFalse(holds)
        return result, json.loads(output.read_text()), observations

    def test_fresh_admission_is_four_locked_samples_spanning30seconds_before_models(self):
        result, saved, observations = self.sample_admission([11.1] * 4)
        self.assertEqual(result, saved)
        self.assertEqual(result['status'], 'PASS_ADMISSION')
        self.assertEqual(len(observations), 4)
        self.assertEqual(self.control.idle_seconds, 30.)
        self.assertTrue(result['gpu_lease_released'])
        self.assertTrue(result['private_port_closed'])
        self.assertEqual(result['model_calls'], 0)

    def test_admission_pressure_preserves_partial_samples_without_retry_or_model_work(self):
        with self.assertRaises(ResourcePressure):
            self.sample_admission([11.1, 10.9])
        saved = json.loads((self.root / 'admission.json').read_text())
        self.assertEqual(saved['status'], 'BLOCKED_ADMISSION')
        self.assertEqual(len(saved['samples']), 2)
        self.assertEqual(self.control.idle_seconds, 10.)
        self.assertFalse(saved['automatic_retry'])

    def test_unavailable_telemetry_is_null_and_blocks_instead_of_importing_a_model(self):
        with self.assertRaises(ResourcePressure):
            self.sample_admission([None])
        saved = json.loads((self.root / 'admission.json').read_text())
        self.assertIsNone(saved['samples'][0]['available_ram_gib'])
        self.assertEqual(saved['error_type'], 'ResourcePressure')

    def test_question_worker_executes_real_native_tau_tools_with_scripted_eight_arm_callbacks(self):
        from haltiq_tau_hotpot.main_question import run_question
        from haltiq_tau_hotpot.protocol import ARMS
        from loguru import logger
        logger.disable('tau2')
        public = {'id': 'scripted', 'question': 'What is the correct name?', 'type': 'bridge', 'level': 'synthetic',
                  'context': [['Public', ['The correct name is right.']]]}
        case = {'partition': 'fresh', 'public': public,
                'evaluator': {'answer': 'right', 'supporting_facts': [['Public', 0]]}}
        trained = self.root / 'train/learned/answer-head.safetensors'
        trained.parent.mkdir(parents=True)
        trained.write_bytes(b'SYNTHETIC IDENTITY, NEVER LOADED')
        durable_write(self.root / 'train/learning.json', {'status': 'completed',
            'frozen_torso_adapter_files_unchanged': True, 'criteria': {'answer': {
                'trained_head_sha256': file_hash(trained), 'released_calibration': {'temperature': 1.},
                'trained_calibration': {'temperature': 2.}}}, 'released_evidence_calibration': {'temperature': 1.}})
        owner = self
        class ScriptedRuntime:
            def __init__(_, config, directory, control):
                _.journal, _.record_start = CallJournal(directory / 'physical'), 0
                _.timings, _.cleanup = {'startup_seconds': 0., 'switch_seconds': 0.}, {'verified': True}
            def __enter__(_): return _
            def __exit__(_, *args): pass
            def rank(_, received, *args):
                owner.assertEqual(received, public)
                return {'indices': [0]}
            def generate(_, state, feedback='', review=False):
                owner.assertNotIn('evaluator', state)
                return 'Use right.' if review else {'answer': 'right' if state['previous_draft'] else 'wrong',
                                                    'supporting_facts': [['Public', 0]]}
            def observe(_, received, draft, criterion, head_path, temperature):
                owner.assertEqual(received, public)
                return {'p_true': .9 if head_path is not None or draft['answer'] == 'right' else .1,
                        'scope': 'scripted callback, not a measured decision model'}
            def assert_envelope(_): pass
        args = SimpleNamespace(previous=self.root, output=self.root / 'compare', question_index=0)
        context = {'config': {}, 'cohorts': {'main_fresh': [case]}}
        with patch('haltiq_tau_hotpot.main_question.QuestionRuntime', ScriptedRuntime):
            result = run_question(args, context, self.control)
        self.assertEqual([row['arm'] for row in result['episodes']], ARMS)
        by_arm = {row['arm']: row for row in result['episodes']}
        self.assertEqual(by_arm['fixed1']['metrics']['em'], 0.)
        self.assertEqual(by_arm['fixed3']['metrics']['em'], 1.)
        self.assertEqual(by_arm['released']['metrics']['em'], 1.)
        self.assertEqual(by_arm['trained']['metrics']['em'], 0.)
        self.assertTrue(all(row['native_simulation'] for row in result['episodes']))


class ParentProcessTests(unittest.TestCase):
    """Synthetic child artifacts test ownership, never benchmark model quality."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.now = datetime.now(timezone.utc)
        self.clock = [self.now]
        self.state = {'run_id': 'SYNTHETIC-PROCESS-CONTROL', 'plan_sha256': 'a' * 64,
                      'source_sha256': {'synthetic.py': 'b' * 64}}
        self.config_path = self.run / 'config.json'
        durable_write(self.config_path, {'scope': 'synthetic process fixture, no model readiness'})
        self.config = {'compare_process_layout': 'per_question', 'fresh_n_freeze_sha256': 'c' * 64}
        self.plan = {'python': sys.executable, 'deadline': (self.now + timedelta(hours=14)).isoformat(),
                     'main_protocol': {'path': str(self.config_path)}}
        durable_write(self.run / 'plan.json', self.plan)
        durable_write(self.run / 'supervisor.json', self.state)
        self.args = SimpleNamespace(previous=self.run, output=self.run / 'compare', config=self.config_path,
            control=self.run / 'workers/compare', deadline=self.plan['deadline'], phase='compare')
        self.args.output.mkdir()
        self.control = Control(self.args.control)
        self.cases = [{'public': {'id': 'fixture-' + str(i)}} for i in range(2)]
        self.context = {'config': self.config, 'fresh_n': 2, 'cohorts': {'main_fresh': self.cases}}
        self.process_count = 0
        import os
        self.parent_pid = os.getpid()
        self.identities = {self.parent_pid: {'pid': self.parent_pid, 'parent_pid': 1,
            'created_at': (self.now - timedelta(seconds=100)).isoformat(), 'executable': sys.executable, 'active': True}}
        self.ended = set()

    def fake_child(self, command, **kwargs):
        from haltiq_tau_hotpot.protocol import ARMS
        index = int(command[command.index('--question-index') + 1])
        qid, launcher_pid, pid = self.cases[index]['public']['id'], 2000 + index, 3000 + index
        directory = self.args.output / 'questions' / f'{index:03d}'
        self.assertTrue((directory / 'dispatch-pending.json').is_file())
        self.assertEqual(command[:4], [sys.executable, '-B', '-m', 'haltiq_tau_hotpot.main_question'])
        self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(kwargs['cwd'], self.run / 'source')
        self.process_count += 1
        begin = self.clock[0] + timedelta(seconds=1)
        self.identities[launcher_pid] = {'pid': launcher_pid, 'parent_pid': self.parent_pid,
            'created_at': (begin - timedelta(seconds=.9)).isoformat(), 'executable': sys.executable, 'active': True}
        self.identities[pid] = {'pid': pid, 'parent_pid': launcher_pid,
            'created_at': (begin - timedelta(seconds=.8)).isoformat(), 'executable': sys._base_executable, 'active': True}
        durable_write(directory / 'started.json', {**self.state, 'pid': pid, 'parent_pid': launcher_pid,
            'question_index': index, 'at': begin.isoformat()})
        samples = [{'at': (begin + timedelta(seconds=1 + 10 * i)).isoformat(),
                    'available_ram_gib': 11.1, 'free_vram_gib': 9.1, 'gpu_temperature_c': 40} for i in range(4)]
        durable_write(directory / 'admission.json', {'schema': 'haltiq-main-question-admission-v1',
            'status': 'PASS_ADMISSION', 'samples': samples, 'ram_gib': 11, 'vram_gib': 9,
            'gpu_lease_released': True, 'private_port_closed': True, 'model_calls': 0, 'automatic_retry': False})
        journal = CallJournal(self.args.output / 'physical')
        start = len(journal.records)
        self.clock[0] = begin + timedelta(seconds=32)
        class JournalClock:
            @staticmethod
            def now(_): return self.clock[0]
        with patch('haltiq_tau_hotpot.journal.datetime', JournalClock):
            journal.call('retrieval', {'fixture': qid}, lambda: {'scope': 'synthetic response, not model inference'})
        initial = {'answer': 'synthetic', 'supporting_facts': []}
        bindings = {}
        initial_relative = 'initial/' + qid + '.json'
        durable_write(self.args.output / initial_relative, {'submitted': initial})
        bindings[initial_relative] = file_hash(self.args.output / initial_relative)
        for arm in ARMS:
            relative = 'episodes/' + qid + '-' + arm + '.json'
            durable_write(self.args.output / relative, {'question_id': qid, 'arm': arm, 'status': 'completed',
                'scope': 'synthetic ownership fixture; native outcome not asserted'})
            bindings[relative] = file_hash(self.args.output / relative)
        cleanup = {k: True for k in ('verified', 'private_port_closed', 'owned_server_exited', 'gpu_lease_released')}
        cleanup.update(native_unloads=[], errors=[])
        durable_write(directory / 'runtime-terminal.json', {'cleanup': cleanup, 'calls': journal.verify(),
            'timings': {'startup_seconds': 1., 'switch_seconds': 2.}, 'active_seconds': 1., 'intentional_idle_seconds': 30.})
        receipt = {'schema': 'haltiq-main-question-v1', 'terminal': True, 'cleanup_verified': True,
            'pending_or_uncertain': False, 'run_id': self.state['run_id'], 'plan_sha256': self.state['plan_sha256'],
            'source_sha256': fingerprint(self.state['source_sha256']), 'config_sha256': file_hash(self.config_path),
            'question_index': index, 'question_id': qid, 'pid': pid, 'parent_pid': launcher_pid,
            'launcher_pid': launcher_pid, 'started_at': begin.isoformat(),
            'finished_at': (begin + timedelta(seconds=34)).isoformat(), 'wall_seconds': 34.,
            'active_seconds': 1., 'intentional_idle_seconds': 30., 'physical_record_start': start,
            'physical_record_end': len(journal.records), 'physical_record_prefix_sha256': fingerprint(journal.records[:start]),
            'physical_record_slice_sha256': fingerprint(journal.records[start:]), 'outputs_sha256': bindings,
            'admission_sha256': file_hash(directory / 'admission.json'),
            'runtime_terminal_sha256': file_hash(directory / 'runtime-terminal.json')}
        durable_write(directory / 'receipt.json', receipt)
        self.clock[0] = begin + timedelta(seconds=.5)
        def wait(**_):
            receipt['identity_sha256'] = file_hash(directory / 'identity-verified.json')
            (directory / 'receipt.json').write_text(json.dumps(receipt))
            self.ended.update((launcher_pid, pid))
            self.clock[0] = begin + timedelta(seconds=35)
            return self.exit_code
        return SimpleNamespace(pid=launcher_pid, wait=wait, poll=lambda: None, terminate=Mock())

    def build(self, exit_code=0):
        from haltiq_tau_hotpot.main_worker import compare
        self.exit_code = exit_code
        def identity(pid):
            if pid in self.ended: raise ProcessLookupError('Synthetic owned process is absent')
            return deepcopy(self.identities[pid])
        with patch('subprocess.Popen', side_effect=self.fake_child), \
                patch('haltiq_tau_hotpot.main_question.stamp', side_effect=lambda: self.clock[0].isoformat()), \
                patch('haltiq_tau_hotpot.main_question.process_identity', side_effect=identity), \
                patch('haltiq_supervisor.engine.check_handoff'), \
                patch('haltiq_tau_hotpot.main_worker.analyse', return_value={'scope': 'synthetic process control'}), \
                patch.dict('os.environ', {'HALTIQ_RUN_ID': self.state['run_id'],
                    'HALTIQ_PLAN_SHA256': self.state['plan_sha256'], 'HALTIQ_PHASE': 'compare'}):
            return compare(self.args, self.context, self.control)

    def audit_children(self):
        from haltiq_tau_hotpot.main_audit import verify_question_owners
        records = CallJournal(self.args.output / 'physical').records
        return verify_question_owners(self.run, self.state, self.cases, records)

    def test_parent_serializes_two_owned_children_and_counts_cumulative_journal_once(self):
        self.build()
        terminal = json.loads((self.args.output / 'runtime-terminal.json').read_text())
        self.assertEqual(self.process_count, 2)
        self.assertEqual(terminal['calls']['completed'], 2)  # Not child1 + cumulative child2 ==3.
        self.assertEqual(terminal['timings'], {'startup_seconds': 2., 'switch_seconds': 4.})
        self.assertEqual(self.control.active_seconds, 2.)
        self.assertEqual(self.control.idle_seconds, 60.)
        self.assertTrue(self.audit_children()['verified'])
        receipt = json.loads((self.args.output / 'questions/000/receipt.json').read_text())
        self.assertNotEqual(receipt['pid'], receipt['launcher_pid'])

    def test_nonzero_child_is_preserved_and_never_automatically_redispatched(self):
        with self.assertRaisesRegex(RuntimeError, 'never retry'):
            self.build(exit_code=1)
        self.assertEqual(self.process_count, 1)
        self.assertTrue((self.args.output / 'questions/000/process-exit.json').is_file())
        self.assertFalse((self.args.output / 'questions/001').exists())
        self.assertFalse((self.args.output / 'completion.json').exists())

    def test_rebound_overlapping_slice_cannot_pass_final_audit(self):
        from haltiq_tau_hotpot.qualification import QualificationError
        self.build()
        directory = self.args.output / 'questions/001'
        receipt = json.loads((directory / 'receipt.json').read_text())
        receipt['physical_record_start'] = 0
        (directory / 'receipt.json').write_text(json.dumps(receipt))
        terminal_path = self.args.output / 'runtime-terminal.json'
        terminal = json.loads(terminal_path.read_text())
        terminal['question_receipts'][1]['sha256'] = file_hash(directory / 'receipt.json')
        terminal_path.write_text(json.dumps(terminal))
        with self.assertRaisesRegex(QualificationError, 'physical slice'):
            self.audit_children()

    def test_rebound_low_resource_receipt_cannot_pass_final_audit(self):
        from haltiq_tau_hotpot.qualification import QualificationError
        self.build()
        directory = self.args.output / 'questions/000'
        path = directory / 'admission.json'
        admitted = json.loads(path.read_text())
        admitted['samples'][0]['available_ram_gib'] = 10.99
        path.write_text(json.dumps(admitted))
        receipt_path = directory / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['admission_sha256'] = file_hash(path)
        receipt_path.write_text(json.dumps(receipt))
        terminal_path = self.args.output / 'runtime-terminal.json'
        terminal = json.loads(terminal_path.read_text())
        terminal['question_receipts'][0]['sha256'] = file_hash(receipt_path)
        terminal_path.write_text(json.dumps(terminal))
        with self.assertRaisesRegex(QualificationError, 'startup reserves'):
            self.audit_children()

    def test_rebound_unrelated_live_worker_ancestry_cannot_pass_final_audit(self):
        self.build()
        directory = self.args.output / 'questions/000'
        path = directory / 'identity-verified.json'
        identity = json.loads(path.read_text())
        identity['worker']['parent_pid'] = 999999
        path.write_text(json.dumps(identity))
        # Rebind hashes too: semantic ancestry must independently fail.
        for name in ('receipt.json', 'process-exit.json'):
            target = directory / name
            value = json.loads(target.read_text()); value['identity_sha256'] = file_hash(path)
            target.write_text(json.dumps(value))
        terminal_path = self.args.output / 'runtime-terminal.json'
        terminal = json.loads(terminal_path.read_text()); entry = terminal['question_receipts'][0]
        entry.update(sha256=file_hash(directory / 'receipt.json'), identity_sha256=file_hash(path),
            process_exit_sha256=file_hash(directory / 'process-exit.json'))
        terminal_path.write_text(json.dumps(terminal))
        with self.assertRaisesRegex(ValueError, 'ancestry'):
            self.audit_children()


if __name__ == '__main__':
    unittest.main()
