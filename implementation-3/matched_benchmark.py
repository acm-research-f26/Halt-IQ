"""Run unchanged local critic on the team's exact 220 saved drafts, resumably."""
import sys,json,csv,time,urllib.request,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
TEAM=ROOT.parent/'.team-reference/vedanshi'; YASH=ROOT.parent/'.team-reference/Yash'
sys.path.insert(0,str(YASH));from haltiq.metrics import hotpot_evaluate
sys.path.insert(0,str(TEAM/'critic-bench'));import drafts
# Resolve original harness's machine-specific path without changing source files.
drafts.YASH_REPO=YASH
from report import auroc,approves
OUT=ROOT/'study150/results/matched220';OUT.mkdir(exist_ok=True)
def jsonl(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows=list(csv.DictReader((TEAM/'strands-reranker/results/haltiq_strands_experiments2.csv').open()))
tasks={t['id'].split(':')[-1]:t for t in jsonl(YASH/'data/hotpotqa/tasks.jsonl')+jsonl(TEAM/'critic-bench/extra/tasks.jsonl')}
assert len(rows)==220 and len({r['id'] for r in rows})==220
for r in rows:
 t=tasks[r['id']];assert t['question']==r['question'];assert int(hotpot_evaluate(r['draft'],t['answers'])['exact_match'])==int(r['em'])
assert sum(int(r['em']) for r in rows)==126
manifest={'n':220,'writer':'qwen3:8b (saved team drafts)','critic':'mlx-community/Qwen2.5-1.5B-Instruct-4bit','temperature':0,'max_tokens':256,'threshold':0.8,'score':'binary exact APPROVED, not probability','evidence_lengths':{str(k):sum(len(tasks[r['id']]['evidence'])==k for r in rows) for k in sorted({len(tasks[r['id']]['evidence']) for r in rows})},'team_commit':'d0688aed897c18d45fea8aacf040896ab06c86ee','yash_commit':'43b0a61c47845292eee747a01af4a5c87dbd60ab'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
path=OUT/'scores.jsonl';done={r['id']:r for r in jsonl(path)} if path.exists() else {}
for r in rows:
 if r['id'] in done:continue
 t=tasks[r['id']]
 payload={'model':manifest['critic'],'temperature':0,'max_tokens':256,'messages':[{'role':'system','content':'Check whether the answer is supported by the supplied passages and fully answers the question. Return APPROVED if so. Otherwise give one brief correction.'},{'role':'user','content':json.dumps({'question':t['question'],'contexts':t['evidence'],'answer':r['draft']})}]}
 start=time.perf_counter()
 req=urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer local'})
 with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
 elapsed=time.perf_counter()-start;feedback=result['choices'][0]['message']['content'];score=float(feedback.strip()=='APPROVED')
 out={'id':r['id'],'draft_id':drafts.draft_id(t['id'],r['draft']),'task_id':t['id'],'draft':r['draft'],'em':int(r['em']),'score':score,'decision':'approve' if score else 'reject','feedback':feedback,'seconds':elapsed,'usage':result['usage'],'input_sha256':hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
 with path.open('a') as f:f.write(json.dumps(out)+'\n')
 done[r['id']]=out
 if len(done)%10==0:print(f'{len(done)}/220',flush=True)
y=np.array([int(r['em']) for r in rows]);boot=np.random.default_rng(0).integers(0,len(y),size=(1000,len(y)))
def metric(name,s):
 s=np.array(s);point=auroc(list(s[y==1]),list(s[y==0]));bs=[float(((s[i][y[i]==1,None] > s[i][y[i]==0][None,:]).sum()+0.5*(s[i][y[i]==1,None] == s[i][y[i]==0][None,:]).sum()) / ((y[i]==1).sum()*(y[i]==0).sum())) for i in boot];ok=np.array([approves({'score':v},.8) for v in s]);return {'critic':name,'n':220,'auroc':point,'ci95':np.percentile(bs,[2.5,97.5]).tolist(),'approved':int(ok.sum()),'precision':float(y[ok].mean()) if ok.any() else None,'wrong_approvals':int((ok&(y==0)).sum()),'n_wrong':94}
metrics=[metric('local_qwen1.5b',[done[r['id']]['score'] for r in rows])]+[metric(c,[float(r[c]) for r in rows]) for c in ['strands','strands_top3','bge_top2','jev']]
(OUT/'metrics.json').write_text(json.dumps({'metrics':metrics,'local_mean_critic_seconds':np.mean([r['seconds'] for r in done.values()]),'local_total_tokens':sum(r['usage']['total_tokens'] for r in done.values())},indent=2))
print(json.dumps(metrics,indent=2),flush=True)
