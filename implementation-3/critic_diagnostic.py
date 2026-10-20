"""Reference-style critic metrics on LOCAL first drafts, never the shared 220 benchmark."""
import json,random,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=ROOT/'study150/results/reference_alignment';out.mkdir(exist_ok=True)
rows=[]
for p in sorted((ROOT/'study150/cache').glob('hotpot_*.json')):
 x=json.loads(p.read_text())
 if x['split']!='test':continue
 r=x['rounds'][0]
 rows.append({'id':x['id'],'question':x['question'],'draft':r['answer'],'em':int(r['answer_em']), 'score':float(r['critic_approved']),'critic_seconds':r['critic']['seconds']})
def auc(rs):
 pos=[r['score'] for r in rs if r['em']];neg=[r['score'] for r in rs if not r['em']]
 if not pos or not neg:return None
 return sum(float(a>b)+.5*float(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def metrics(rs):
 approved=[r for r in rs if r['score']>=.8];tp=sum(r['em'] for r in approved);negative=sum(not r['em'] for r in rs)
 rng=random.Random(0);boot=[auc(rng.choices(rs,k=len(rs))) for _ in range(1000)];valid=sorted(v for v in boot if v is not None)
 return {'n':len(rs),'positives':sum(r['em'] for r in rs),'negatives':negative,'auroc':auc(rs),'auroc_ci95':[valid[int(.025*(len(valid)-1))],valid[int(.975*(len(valid)-1))]],'bootstrap_valid':len(valid),'approve_threshold':.8,'approved':len(approved),'approve_rate':len(approved)/len(rs),'precision':tp/len(approved) if approved else None,'wrong_approvals':len(approved)-tp,'wrong_approval_rate_among_negatives':(len(approved)-tp)/negative,'correct_rejections':sum(r['em'] for r in rs)-tp,'mean_critic_seconds':sum(r['critic_seconds'] for r in rs)/len(rs)}
result={'scope':'Local 80 Qwen2.5-1.5B test first drafts; NOT shared 220 Qwen3:8B drafts','label':'strict normalized answer EM','score':'binary APPROVED=1, otherwise0; not calibrated confidence','bootstrap':'1000 draft resamples, seed0; single-class resamples excluded','metrics':metrics(rows)}
(out/'critic_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'first_draft_scores.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
assert result['metrics']['approved']==78 and len(rows)==80
assert auc([{'score':1,'em':1},{'score':0,'em':0}])==1
assert auc([{'score':1,'em':1},{'score':1,'em':0}])==.5
print(json.dumps(result,indent=2))
