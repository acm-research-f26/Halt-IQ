"""Native tools plus synthetic provider/cache controls; zero real model calls."""
from __future__ import annotations

from array import array
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from haltiq_tau_hotpot.main_audit import AuditIntegrityError, verify_compare
from haltiq_tau_hotpot.journal import fingerprint
from haltiq_tau_hotpot.qualification import QualificationError, sha
from haltiq_tau_hotpot.runtime import LocalRuntime


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class Control:
    def __init__(self, directory):
        self.directory = directory
        self.deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        self.active_seconds = self.idle_seconds = 0.
    def before_work(self, **_): pass
    def rest_after_block(self, _): pass


class Tensor:
    def __init__(self, shape): self.shape = shape
    def contiguous(self): return self


class MainAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from loguru import logger
        logger.disable("tau2")

    def setUp(self):
        from haltiq_tau_hotpot.agent import run_episode
        from haltiq_tau_hotpot.hobson import INSTRUCTIONS_BY_CRITERION
        self.run_episode, self.instructions = run_episode, INSTRUCTIONS_BY_CRITERION
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        checkpoint = self.root / "released"
        checkpoint.mkdir()
        (checkpoint / "head.safetensors").write_bytes(b"SYNTHETIC IDENTIFIER; NOT A MODEL")
        trained = self.root / "train/learned/answer-head.safetensors"
        trained.parent.mkdir(parents=True)
        trained.write_bytes(b"DIFFERENT SYNTHETIC IDENTIFIER; NOT A MODEL")
        self.config = {"checkpoint": str(checkpoint), "writer_digest": "synthetic-writer-identity",
                       "response_cache": True, "native_feature_cache": True}
        self.learning = {"released_head_sha256": sha(checkpoint / "head.safetensors"),
            "encoder_sha256": {"synthetic-not-a-torso": "1" * 64}, "criteria": {"answer": {
                "trained_head_sha256": sha(trained), "released_calibration": {"temperature": 1.},
                "trained_calibration": {"temperature": 2.}}}, "released_evidence_calibration": {"temperature": 1.}}
        dump(self.root / "train/learning.json", self.learning)
        self.case = {"partition": "fresh", "public": {"id": "synthetic", "question": "Who is correct?",
            "type": "bridge", "level": "synthetic", "context": [["Public", ["The correct name is right."]]]},
            "evaluator": {"answer": "right", "supporting_facts": [["Public", 0]]}}
        self.states = {}

    def modules(self):
        package, module = ModuleType("safetensors"), ModuleType("safetensors.torch")
        def save(tensors, path):
            header, raw = {}, bytearray()
            for name, tensor in tensors.items():
                start = len(raw)
                raw.extend(array('f', [0.] * math.prod(tensor.shape)).tobytes())
                header[name] = {"dtype": "F32", "shape": tensor.shape, "data_offsets": [start, len(raw)]}
            encoded = json.dumps(header).encode()
            Path(path).write_bytes(struct.pack('<Q', len(encoded)) + encoded + raw)
        module.save_file = save
        module.load_file = lambda *_, **__: {"decide": Tensor([2048]), "options": Tensor([2, 2048])}
        return {"safetensors": package, "safetensors.torch": module}

    def score(self, state, *, trained, temperature):
        positive = 'Draft answer: right' in state or trained
        logits = ([-2., 2.] if trained else [-1., 1.]) if positive else [1., -1.]
        p = 1 / (1 + math.exp(-(logits[1] - logits[0]) / temperature))
        return {"raw_logits_false_true": logits, "temperature": temperature, "p_true_unrounded": p,
                "p_true": round(p, 4), "physical_forwards": 0, "head_scores": 1}

    def build(self, failure=None):
        from haltiq_tau_hotpot.protocol import ARMS
        from haltiq_tau_hotpot import hobson
        if failure == 'overflow': self.case['public']['context'][0][1] = ['x' * 20000]
        public = self.case['public']
        runtime = LocalRuntime(self.config, self.root / 'compare', Control(self.root / 'control'))
        runtime._writer_ready = Mock()  # No processes, resource probes, or HTTP.
        runtime._native_ready = Mock()
        runtime.ranker = SimpleNamespace(rank=lambda _: {'scores': [.7], 'pair_count': 1, 'physical_forwards': 1})
        def http(path, request, **_):
            state = json.loads(request['messages'][1]['content'])['state']
            if 'Review the proposed' in request['messages'][0]['content']:
                content = '' if failure == 'review_empty' else 'Use right.'
            elif failure == 'parse' or (failure == 'revision_parse' and state.get('previous_draft')): content = '[]'
            else:
                answer = 'right' if state.get('previous_draft') else 'wrong'
                content = json.dumps({'answer': answer, 'supporting_facts': [['Foreign' if failure == 'citation' else 'Public', 0]]})
            return {'done': True, 'done_reason': 'stop', 'message': {'content': content}}
        runtime._http = http
        def prepare(state, **_):
            if failure == 'gate_window': raise ValueError('Synthetic tokenization-only overflow')
            return {}
        def observe(state, criterion='answer', head_path=None, temperature=None, **_):
            self.states[fingerprint(state)] = state
            provenance = {'criterion': criterion, 'state_sha256': fingerprint(state),
                'instructions_sha256': fingerprint(self.instructions[criterion]),
                'rendered_question_sha256': '2' * 64, 'token_ids_sha256': '3' * 64,
                'option_order': ['false', 'true'], 'source_sha256': sha(hobson.__file__)}
            feature = {'schema': 'haltiq-hobson-pointer-feature-v1', 'criterion': criterion,
                'option_order': ['false', 'true'], 'encoder_sha256': self.learning['encoder_sha256'],
                'provenance': provenance, 'provenance_sha256': fingerprint(provenance),
                'decide': Tensor([2048]), 'options': Tensor([2, 2048])}
            value = self.score(state, trained=head_path is not None, temperature=temperature)
            return {**value, 'physical_forwards': 1, 'head_scores': 2, 'feature': feature,
                    'native_logits_false_true': value['raw_logits_false_true'], 'cached_replay_max_logit_error': 0.}
        runtime.backend = SimpleNamespace(prepare_state=prepare, observe=observe)
        def cpu(head, feature, *, temperature):
            return self.score(self.states[feature['provenance']['state_sha256']], trained=head == 'trained', temperature=temperature)
        def load(_, path): return 'trained' if 'answer-head' in str(path) else 'released'
        bare = {'public': public, 'evaluator': self.case['evaluator']}
        fallback = {'cap_fallback': self.config['cap_fallback']} if 'cap_fallback' in self.config else {}
        with patch.dict(sys.modules, self.modules()), patch('haltiq_tau_hotpot.hobson.load_head', side_effect=load), \
             patch('haltiq_tau_hotpot.hobson.score_head', side_effect=cpu):
            indices = runtime.rank(public)['indices']
            first = self.run_episode(bare, runtime.generate, arm='fixed1', max_drafts=1, passage_indices=indices, **fallback)
            initial = {'synthetic': first['submitted']}
            dump(self.root / 'compare/initial/synthetic.json', first)
            episodes = []
            for arm in ARMS:
                def writer(state, feedback):
                    if state['previous_draft'] is None:
                        if first['submitted'] is None: raise ValueError('Shared initial failure; no redispatch')
                        return deepcopy(first['submitted'])
                    return runtime.generate(state, feedback)
                def reviewer(state, draft):
                    ranked = runtime.rank(public, draft['answer'])['indices']
                    evidence = [{'title': public['context'][i][0], 'sentences': [
                        {'index': n, 'text': text} for n, text in enumerate(public['context'][i][1])]} for i in ranked]
                    return runtime.generate({'question': public['question'], 'evidence': evidence, 'draft': draft}, review=True)
                def gate(state, draft, criterion):
                    return runtime.observe(public, draft, criterion=criterion,
                        head_path=self.root / 'train/learned/answer-head.safetensors' if arm == 'trained' else None,
                        temperature=2. if arm == 'trained' else 1.)
                episode = self.run_episode(bare, writer, reviewer, gate, arm=arm, max_drafts=10, passage_indices=indices, **fallback)
                episode.update(shared_initial_draft_sha256=fingerprint(first['submitted']), shared_initial_physical_calls=0)
                dump(self.root / 'compare/episodes' / ('synthetic-' + arm + '.json'), episode)
                episodes.append(episode)
        dump(self.root / 'compare/episodes.json', episodes)
        dump(self.root / 'compare/initial-drafts.json', initial)
        completed = sum(e['status'] == 'completed' for e in episodes)
        dump(self.root / 'compare/completion.json', {'run_id': 'synthetic-run', 'plan_sha256': '0' * 64,
            'phase': 'compare', 'terminal': True, 'cleanup_verified': True, 'pending_or_uncertain': False,
            'planned_units_accounted': True, 'planned_units': 8, 'completed_units': completed,
            'failed_units': 8 - completed, 'not_run_units': 0})
        dump(self.root / 'plan.json', {'stages': [{'id': 'compare', 'planned_units': 8}]})
        dump(self.root / 'supervisor.json', {'run_id': 'synthetic-run', 'plan_sha256': '0' * 64,
            'stages': {'compare': {'status': 'completed', 'exit_code': 0, 'owned_tree_clear': True,
                'os_handoff_verified': True, 'output_sha256': {}}}})
        context = {'config': self.config, 'fresh_n': 1, 'cohorts': {'main_fresh': [self.case]}}
        return runtime, context, episodes, initial

    def verify(self, context, episodes, initial):
        return verify_compare(SimpleNamespace(previous=self.root), context, episodes, initial)

    def test_real_native_tools_all_eight_arms_join_genuine_synthetic_cache_and_reuse_receipts(self):
        runtime, context, episodes, initial = self.build()
        result = self.verify(context, episodes, initial)
        self.assertEqual(result['native_initial_and_arm_replays'], 9)
        self.assertEqual(result['planned_episodes'], 8)
        self.assertEqual(result['logical_callbacks']['shared_initial_writer_zero_new_calls'], 8)
        self.assertGreater(result['physical']['exact_response_reuses'], 0)
        self.assertGreater(result['physical']['by_role']['cached_head_answer'], 0)
        self.assertEqual(result['new_encoder_forwards'], 0)
        self.assertEqual(episodes[-1]['submitted']['answer'], 'wrong')
        self.assertGreater(len(runtime.native_cache), 1)

    def test_main_reviewer_profile_is_reconstructed_exactly_and_legacy_relabel_fails(self):
        from haltiq_tau_hotpot.generation import MAIN_PROFILE
        self.config.update(purpose='main_study', generation_profile=deepcopy(MAIN_PROFILE), cap_fallback='last_valid')
        runtime, context, episodes, initial = self.build()
        self.assertEqual(self.verify(context, episodes, initial)['planned_episodes'], 8)
        reviewer = next(r for r in runtime.journal.records if r['event'] == 'pending' and r['role'] == 'reviewer')
        from haltiq_tau_hotpot.main_audit import payload
        state = {'question': self.case['public']['question'], 'evidence': [{'title': 'Public', 'sentences': [
            {'index': 0, 'text': 'The correct name is right.'}]}], 'draft': {'answer': 'wrong', 'supporting_facts': [['Public', 0]]}}
        request = payload(state, '', review=True, config=self.config)
        self.assertEqual(request['options']['num_predict'], 768)
        self.assertIn('120 words', request['messages'][0]['content'])
        self.assertEqual(reviewer['request_sha256'], fingerprint(request))
        self.config.pop('purpose'); self.config.pop('generation_profile')
        with self.assertRaises(AuditIntegrityError):
            self.verify(context, episodes, initial)

    def test_pre_dispatch_window_failure_and_post_response_parse_or_citation_failures_keep_all_arms(self):
        for failure in ('overflow', 'parse', 'citation', 'gate_window', 'revision_parse', 'review_empty'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                # Isolate each synthetic attempt without touching real run directories.
                original_root = self.root
                self.root = Path(temp)
                checkpoint = self.root / 'released'; checkpoint.mkdir()
                (checkpoint / 'head.safetensors').write_bytes(b'SYNTHETIC IDENTIFIER; NOT A MODEL')
                trained = self.root / 'train/learned/answer-head.safetensors'; trained.parent.mkdir(parents=True)
                trained.write_bytes(b'DIFFERENT SYNTHETIC IDENTIFIER; NOT A MODEL')
                self.config['checkpoint'] = str(checkpoint)
                dump(self.root / 'train/learning.json', self.learning)
                runtime, context, episodes, initial = self.build(failure)
                result = self.verify(context, episodes, initial)
                self.assertEqual(result['planned_episodes'], 8)
                self.assertGreater(result['failed_policy_episodes'], 0)
                if failure == 'overflow': self.assertNotIn('writer', runtime.journal.summary()['by_role'])
                self.case['public']['context'][0][1] = ['The correct name is right.']
                self.root = original_root

    def test_native_gate_trace_tool_reward_and_completion_mutations_fail(self):
        _, context, episodes, initial = self.build()
        for kind in ('gate', 'tool', 'reward', 'completion'):
            altered = deepcopy(episodes)
            if kind == 'gate':
                event = next(e for e in altered[-1]['trace'] if e['event'] == 'gate')
                event['score']['p_true'] = .1234
            elif kind == 'tool': altered[0]['native_simulation']['messages'][-1]['content'] = '{"submitted":false}'
            elif kind == 'reward': altered[0]['native_reward'] = 1
            else:
                path = self.root / 'compare/completion.json'
                value = json.loads(path.read_text()); value['completed_units'] -= 1; value['failed_units'] += 1; dump(path, value)
            dump(self.root / 'compare/episodes.json', altered)
            for row in altered: dump(self.root / 'compare/episodes' / ('synthetic-' + row['arm'] + '.json'), row)
            with self.subTest(kind=kind), self.assertRaises((QualificationError, ValueError, AuditIntegrityError)):
                self.verify(context, altered, initial)
            # Each case starts with the original validated source records.
            dump(self.root / 'compare/episodes.json', episodes)
            for row in episodes: dump(self.root / 'compare/episodes' / ('synthetic-' + row['arm'] + '.json'), row)

    def test_reuse_claiming_new_physical_work_and_changed_raw_request_are_rejected(self):
        runtime, context, episodes, initial = self.build()
        path = runtime.journal.path
        original = path.read_bytes()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        next(r for r in records if r['event'] == 'exact_response_reuse')['new_physical_calls'] = 1
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with self.assertRaises((ValueError, AuditIntegrityError)): self.verify(context, episodes, initial)
        path.write_bytes(original)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        next(r for r in records if r['event'] == 'pending' and r['role'] == 'writer')['request_sha256'] = '0' * 64
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with self.assertRaises((ValueError, AuditIntegrityError)): self.verify(context, episodes, initial)

    def test_integrity_error_inside_known_failed_callback_is_not_swallowed_as_an_expected_policy_failure(self):
        runtime, context, episodes, initial = self.build('parse')
        records = [json.loads(line) for line in runtime.journal.path.read_text().splitlines()]
        next(r for r in records if r['event'] == 'pending' and r['role'] == 'writer')['request_sha256'] = '0' * 64
        runtime.journal.path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with self.assertRaises(AuditIntegrityError):
            self.verify(context, episodes, initial)

    def test_changed_native_cache_tensor_or_trained_calibration_is_rejected(self):
        runtime, context, episodes, initial = self.build()
        learning_path = self.root / 'train/learning.json'
        changed = deepcopy(self.learning)
        changed['criteria']['answer']['trained_calibration']['temperature'] = 7.
        dump(learning_path, changed)
        with self.assertRaises(AuditIntegrityError): self.verify(context, episodes, initial)
        dump(learning_path, self.learning)
        feature = next(iter(runtime.native_cache.values()))['observed']['feature']
        Path(feature['path']).write_bytes(b'CHANGED; NOT A FEATURE')
        with self.assertRaises(AuditIntegrityError): self.verify(context, episodes, initial)


if __name__ == '__main__':
    unittest.main()
