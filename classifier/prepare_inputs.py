"""Audit supplied scores against immutable original data; write only classifier/."""
import csv, hashlib, json, math, subprocess
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
SRI='5503cabeb2ec0094413b7d491ccb08ec9c991c61'
BASE='implementation-3/study150'
def git_read(path):
    return subprocess.check_output(['git','-C',str(REPO),'show',f'{SRI}:{path}'])
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    qbytes=git_read(BASE+'/questions.json'); questions=json.loads(qbytes)
    rounds_path=REPO/'data/sri_rounds_with_strands.csv'
    scores_path=REPO/'data/haltiq_sri_trajectory_strands.csv'
    rows=list(csv.DictReader(rounds_path.open(newline='')))
    scores=list(csv.DictReader(scores_path.open(newline='')))
    assert len(rows)==900 and len(scores)==166
    by_round={(r['id'],int(r['round'])):r for r in rows}
    by_score={(r['id'],r['answer']):float(r['strands']) for r in scores}
    assert len(by_round)==900 and len(by_score)==166
    assert all(math.isfinite(s) and 0<=s<=1 for s in by_score.values())
    items=[]; hashes={}; used=set()
    for q in sorted(questions,key=lambda q:q['id']):
        raw=git_read(BASE+'/cache/'+q['id']+'.json'); source=json.loads(raw)
        assert source['split']==q['split'] and len(source['rounds'])==6
        hashes[q['id']]=sha(raw)
        clean=[]
        for k,r in enumerate(source['rounds'],1):
            row=by_round[(q['id'],k)];key=(q['id'],r['answer']);used.add(key)
            assert row['split']==q['split'] and row['answer']==r['answer']
            assert float(row['strands'])==by_score[key]
            assert float(row['f1'])==r['quality'] and float(row['em'])==r['answer_em']
            assert int(row['tokens'])==r['tokens']
            for role in ['writer','critic']:
                u=r[role]['usage']
                for f in ['prompt_tokens','completion_tokens','total_tokens']:
                    assert int(row[f'{role}_{f}'])==u[f]
                assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
            assert r['tokens']==r['writer']['usage']['total_tokens']+r['critic']['usage']['total_tokens']
            clean.append({'round':k,'answer':r['answer'],'distance':r['distance'],
                          'critic_approved':r['critic_approved'],'strands':by_score[key],
                          'quality':r['quality'],'answer_em':r['answer_em'],'tokens':r['tokens'],
                          'writer_tokens':r['writer']['usage']['total_tokens'],
                          'critic_tokens':r['critic']['usage']['total_tokens']})
        items.append({'id':q['id'],'question':q['question'],'split':q['split'],'rounds':clean})
    assert used==set(by_score)
    counts=dict(Counter(x['split'] for x in items));assert counts=={'train':50,'dev':20,'test':80}
    (ROOT/'inputs').mkdir(exist_ok=True);(ROOT/'reference').mkdir(exist_ok=True)
    for seed in range(5):
        (ROOT/f'reference/original_seed{seed}.pt').write_bytes(git_read(BASE+f'/results/seed{seed}/stop_head.pt'))
    # Keep original algorithm byte-for-byte for parity tests and attribution.
    (ROOT/'reference/original_hotpot.py').write_bytes(git_read('implementation-3/hotpot.py'))
    (ROOT/'inputs/trajectories.json').write_text(json.dumps(items,indent=2)+'\n')
    manifest={'source_commit':SRI,'integration_base_commit':'6c72991652400dcfec8f464a331efb95d5d1cf8b',
              'splits':counts,'rounds':900,'unique_scored_drafts':166,
              'question_manifest_sha256':sha(qbytes),
              'input_hashes':{str(p.relative_to(REPO)):sha(p.read_bytes()) for p in [rounds_path,scores_path]},
              'original_trajectory_sha256':hashes,
              'joined_sha256':sha((ROOT/'inputs/trajectories.json').read_bytes()),
              'audit':'All IDs, splits, answers, labels, scores and token usage match pinned original cache.',
              'cost_scope':'Original Qwen writer+critic tokens; no measured Strands/reranker overhead.',
              'trajectory_scope':'Original Qwen trajectories post-hoc scored by Strands, not a live Strands feedback loop.'}
    (ROOT/'inputs/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['splits','rounds','unique_scored_drafts','audit']},indent=2))
if __name__=='__main__':main()
